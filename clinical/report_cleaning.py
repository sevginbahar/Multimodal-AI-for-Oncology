"""
Pathology report cleaning for melanocytic skin lesion classification
===================================================================
Preprocessing for the clinical-text modality (BioClinicalBERT).

Design
------
Each source field ("diagnosis" = diagnostic_summary_english,
"macroscopic" = macroscopic_description_english) is processed with one policy:

    keep    normalise + de-identify + drop processing artefacts, keep all
            clinical content.
    strip   as `keep`, then DELETE diagnostic / staging / invasion spans
            with regex.
    fact    an LLM keeps only literal morphological observation, dropping the
            diagnosis and everything implying it (Watson et al.'s FFilt).
    drop    return "" (exclude the field entirely).

Every policy emits ORDINARY PROSE -- safe for a frozen BioClinicalBERT encoder.
There is no semantic-tagging policy on purpose: a frozen (non-fine-tuned) encoder
has no token for `<DX>` / `@DEPTH@` and shreds them into junk subwords. Watson
et al.'s SemanticTagging only worked because they fine-tuned end-to-end. If you
ever need their per-tag ablation, do it on a fine-tuned encoder with the tags
registered via tokenizer.add_special_tokens + model.resize_token_embeddings.

Watson et al. (Commun Med 2026, s43856-026-01456-2) finding: regex removes the
diagnosis *word* but not the phrasing/structure that implies it, and fails on
typos -- so `strip` is a floor, not a solution; `fact` (LLM) is the real
debiased input. Report the downstream classifier at SEVERAL levels
(see LEVEL_PRESETS), like their Table 6, rather than picking one.

Caveat specific to this project: unlike Watson's consultation notes (written
against a separate SNOMED/histopath ground truth), our pathology report is
plausibly the SOURCE of `source_diagnosis` -- i.e. input and label are one
document. If so, only the `fact` / morphology-only level gives a defensible
text/fusion result. Confirm label provenance.

Usage
-----
    from report_cleaning import build_full_report, CleanConfig
    cfg = CleanConfig.from_level("diagdrop")   # or CleanConfig(diagnosis=..., macroscopic=...)
    df["full_report"] = df.apply(lambda r: build_full_report(r, cfg), axis=1)

CLI:
    python clinical/report_cleaning.py --level termfilt --show 8 --audit
    python clinical/report_cleaning.py --audit-ladder          # probe every regex level
    python clinical/report_cleaning.py --level diagdrop --out reports_audit.csv   # raw+clean for manual review
    OPENAI_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1:8b \
        python clinical/report_cleaning.py --level fact --out reports_fact.csv
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Field names (override via CleanConfig.text_cols if your columns differ)
# --------------------------------------------------------------------------
DEFAULT_TEXT_COLS = {
    "diagnosis":   "diagnostic_summary_english",
    "macroscopic": "macroscopic_description_english",
}
DEFAULT_LABEL_COL = "source_diagnosis"

_FLAGS = re.IGNORECASE

# --------------------------------------------------------------------------
# 1. De-identification  (always applied; span -> placeholder)
# --------------------------------------------------------------------------
_PHI_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:MATIAS[-\s]?GUIU|EGIDO)\b", _FLAGS),                 "<NAME>"),
    (re.compile(r"\b(?:Dr|Dra|Prof|Prof\.?[- ]?Dr)\.?\s+"
                r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:[-\s][A-ZÁÉÍÓÚÑ][\wáéíóúñ]+){0,2}"), "<NAME>"),
    (re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b"),                  "<DATE>"),
    (re.compile(r"\b[A-Z]{1,3}[-\s]?\d{2,}[-/]\d{2,}\b"),                   "<ID>"),   # accession S23-12345
    (re.compile(r"\b\d{6,}\b"),                                            "<ID>"),   # MRN / accession
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),                          "<EMAIL>"),
]

# --------------------------------------------------------------------------
# 2. Processing artefacts  (always deleted, in every mode)
# --------------------------------------------------------------------------
_ARTIFACT_RULES: list[re.Pattern] = [re.compile(p, _FLAGS) for p in [
    r"\[text appears truncated\]",
    # I.T. / I.P. / "I.T 2B" grossing codes -- CASE-SENSITIVE so it can't eat
    # ordinary lowercase words ("it", "its", "is").
    r"(?-i:\bI\.?\s?[TP]\.?(?:\s*\d{1,2}[A-Z])?)\b(?:\s*\(intra[-\s]?lesional\))?(?:\s*(?:MDC|SMC|JBC))?",
    r"(?m)(?<![\w.])\d{1,2}[A-Z]\b\s*\.?\s*$",          # trailing cassette code "2B."
    r"\bin\s+a\s+cross\s+pattern\b",
    r"\bcross[-\s]?section\w*\b",
    r"\bin cross[-\s]?section\b",
    r"\(intra[-\s]?lesional\)",
    r"\b(?:total|partial|previous)\s+inclusion\b",
    r"\bprevia\s+bisecc?i[oó]n\w*\b",
    r"\bbisect(?:ed|ion|s)?\b",
    r"\b(?:china|india|chinese)\s+ink\b",
    r"\bmark(?:ing|ed)\b[^.\n;]*\bink\b[^.\n;]*",
    r"\b\d+[A-Z]\s+cassettes?\b",
    r"\bcassettes?\b",
    r"\bH\s?&\s?E\b",
    r"\b(?:ha?ematoxylin)\s+(?:and|&)\s+eosin\b",
    r"(?<![\w])nan(?![\w])",                             # stray 'nan' from CSV NaNs
]]

# --------------------------------------------------------------------------
# 3. Diagnostic / staging / invasion rules  ('strip' deletes every match).
#    The leading label on each tuple is documentation only -- it groups the
#    patterns and names what category the span belongs to.
#    Longest / most specific patterns first within each group.
#    `[^.\n;]*` tails mean "rest of the synoptic clause"; they are rewritten to
#    _CLAUSE below so a decimal point (e.g. "3.8 mm") does not end the clause.
# --------------------------------------------------------------------------
_CLAUSE = r"(?:[^.\n;]|\.(?=\d))*"   # to end of clause, tolerating decimals

_RULES: list[tuple[str, re.Pattern]] = [
    (name, re.compile(p.replace(r"[^.\n;]*", _CLAUSE), _FLAGS)) for name, p in [
    # ---- diagnosis / entity mentions ------------------------------------
    ("DX", r"\bmelanoma\s+in\s+situ\b"),
    ("DX", r"\b(?:malignant|invasive|in[-\s]?situ|superficial\s+spreading|nodular|"
           r"acral(?:\s+lentiginous)?|desmoplastic)\s+melanoma\b"),
    ("DX", r"\blentigo\s+maligna(?:\s+melanoma)?\b"),
    ("DX", r"\bmelanoma(?:\s+stage\s+[ivx]+[abc]?)?\b"),
    ("DX", r"\b(?:severely?|moderately?|mildly?|low[-\s]grade|high[-\s]grade|"
           r"architectural(?:ly)?|cytologic(?:ally)?)?\s*"
           r"dysplastic\s+n(?:a)?ev(?:us|i)\b"),
    ("DX", r"\b(?:melanocytic|compound|junctional|intradermal|dermal|blue|spitz|"
           r"congenital|recurrent)\s+n(?:a)?ev(?:us|i)\b"),
    ("DX", r"\bn(?:a)?ev(?:us|i)\b"),
    ("DX", r"\bmelanocytic\s+dysplasia\b"),
    ("DX", r"\b(?:severe|moderate|mild|low[-\s]grade|high[-\s]grade)\s+"
           r"(?:melanocytic\s+|architectural\s+|cytologic\s+)?dysplasia\b"),
    ("DX", r"\bdysplas(?:ia|tic)\b"),
    ("DX", r"\bdispl[aáeé]s\w*\b"),                      # Spanish variants (displasia/displásico/displastic)
    ("DX", r"\bmalign(?:ant|o|a)\b"),

    # ---- staging ------------------------------------------------------
    ("STAGE", r"\b[pyr]{0,3}T\s?(?:[0-4][a-d]?|is)\b"),   # pT1a, T4b, pTis, Tis
    ("STAGE", r"\bN[0-3][a-c]?\b"),
    ("STAGE", r"\bM[01][a-c]?\b"),
    ("STAGE", r"\b(?:AJCC\s+|pathologic(?:al)?\s+)?stage\s*[:=]?\s*[0IV]+[abc]?\b"),
    ("STAGE", r"\bclark(?:'?s)?\s*(?:level)?\s*[:=]?\s*[ivx]+\b"),
    ("STAGE", r"\bnivel\s+[ivx]+\s+de\s+clark\b"),
    ("STAGE", r"\bpathologic(?:al)?\s+stage\b[^.\n;]*"),

    # ---- Breslow / tumour thickness / depth --------------------------
    ("BRESLOW", r"\bbreslow(?:'?s)?(?:\s+(?:thickness|depth|index))?\s*[:=]?\s*"
                r"(?:of\s+)?\d+(?:[.,]\d+)?\s*mm\b"),
    ("BRESLOW", r"\bbreslow\b"),
    ("THICKNESS", r"\b(?:tumou?r\s+|maximum\s+|max\.?\s+)?(?:thickness|depth\s+of\s+"
                  r"invasion|invasion\s+depth|infiltration\s+depth)\s*[:=]?\s*"
                  r"\d+(?:[.,]\d+)?\s*(?:mm|[µu]m|micr[oa]ns?|micras)\b"),
    ("THICKNESS", r"\b\d+(?:[.,]\d+)?\s*mm\s+(?:in\s+)?(?:thick(?:ness)?|deep|"
                  r"depth|of\s+(?:thickness|invasion))\b"),
    # NB: no bare-"N mm" rule -- in this data mm values are mostly margin
    # clearances / gross sizes, not depth; deleting them mangled sentences.
    # Depth is caught by the keyworded BRESLOW/THICKNESS rules above, and the
    # whole synoptic line is removed at the `diagdrop` / `fact` levels anyway.

    # ---- mitoses -----------------------------------------------------
    ("MITOSES", r"\bmitos(?:is|es|ic)\b(?:\s*(?:rate|index|count|figures?))?\s*[:=]?\s*"
                r"(?:\d+\s*(?:/|per)\s*mm[²2]?|\d+\s*mm[²2]?|\d+\s*hpf|absent|present|"
                r"not\s+identified|none)?"),
    ("MITOSES", r"\b\d+\s*mitos(?:is|es)\s*(?:/|per)\s*mm[²2]?\b"),
    ("MITOSES", r"\bmitotic\s+(?:figures?|activity|index|rate)\b"),

    # ---- margins ---------------------------------------------------
    ("MARGIN", r"\b(?:the\s+)?(?:peripheral|deep|lateral|surgical|closest|nearest|resection)?"
               r"\s*margins?\s+(?:of\s+resection\s+)?(?:lateral\s+and\s+deep\s+)?"
               r"(?:are\s+|is\s+|appear\s+)?(?:free|clear|involved|positive|negative|"
               r"not\s+involved|uninvolved|reach(?:ed|es)?)\b[^.\n;]*"),
    ("MARGIN", r"\bmargins?\s+of\s+resection\b[^.\n;]*"),
    ("MARGIN", r"\bfree\s+(?:resection\s+)?margins?\b"),
    ("MARGIN", r"\b(?:complete|incomplete)(?:ly)?\s+(?:excision|removal|resection|"
               r"lesion\s+excision|excision\s+of\s+the\s+lesion)\b[^.\n;]*"),
    ("MARGIN", r"\bexcision\s+(?:is\s+)?(?:complete|incomplete)\b"),
    ("MARGIN", r"\b(?:distance\s+to|clearance\s+from)\s+(?:the\s+)?"
               r"(?:deep|peripheral|nearest)\s+margin\b[^.\n;]*"),

    # ---- growth phase / pattern --------------------------------------
    ("GROWTH", r"\b(?:radial|vertical|horizontal)\s+growth\s+phase\b"),
    ("GROWTH", r"\bgrowth\s+phase\s*[:=]?\s*\w+"),
    ("GROWTH", r"\bpagetoid\b(?:\s+(?:spread|scatter\w*|infiltration|melanocytes|"
               r"upward\s+migration))?"),
    ("GROWTH", r"\bepidermal\s+migration\s+of\s+melanin\b"),
    ("GROWTH", r"\b(?:signs?|areas?|foci)\s+of\s+regress(?:ion|ive)\b"),
    ("GROWTH", r"\bregress(?:ion|ive)(?:\s+changes?)?\b"
               r"(?:\s*[:=]?\s*(?:present|absent|partial|early|extensive|<?\s*\d+\s*%))?"),

    # ---- invasion / dermis level -----------------------------------
    ("INVASION", r"\b(?:vascular|lymphatic|lymphovascular|angio(?:lymphatic|linfatic)?|"
                 r"perineural)\s+invasion\b[^.\n;]*"),
    ("INVASION", r"\bangioinvasion\b"),
    ("INVASION", r"\bmicrosatellitosis\b"),
    ("INVASION", r"\b(?:micro)?satellite(?:\s+(?:nodules?|metastas[ei]s))?\b"),
    ("INVASION", r"\bulcerat(?:ed|ion)\b(?:\s*[:=]?\s*(?:present|absent|yes|no))?"),
    ("INVASION", r"\b(?:no|absence\s+of|without)\s+ulceration\b"),
    ("INVASION", r"\bausencia\s+de\s+ulceraci[oó]n\b|\bsin\s+ulceraci[oó]n\b"),
    ("INVASION", r"\b(?:invasion|infiltration)\s+(?:of|into|to|through)\s+the\s+"
                 r"(?:papillary|reticular|periadventitial|follicular)\s+"
                 r"(?:dermis|epithelium)\b[^.\n;]*"),
    ("INVASION", r"\b(?:papillary|reticular|periadventitial)\s+dermis\b"),
    ("INVASION", r"\bdermal\s+(?:tissue\s+)?(?:invasion|infiltration)\b"),
    ("INVASION", r"\blevel\s+[ivx]+\s+invasion\b"),
    ("INVASION", r"\bin[-\s]?situ\b"),
    ("INVASION", r"\bmicro[-\s]?invasi(?:on|ve)\b"),
    ("INVASION", r"\binvasive\b"),
    ("INVASION", r"\bacral\b"),
    ("INVASION", r"\blentiginoso\b|\blentiginous\b"),

    # ---- immunohistochemistry -------------------------------------
    ("IHC", r"\b(?:HMB[-\s]?45|Melan[-\s]?A|MART[-\s]?1|SOX[-\s]?10|S[-\s]?100|"
            r"Ki[-\s]?67|MIB[-\s]?1|MITF|PRAME|p16|BRAF(?:\s+V600E)?)\b"
            r"(?:\s*[:(]?\s*(?:positive|negative|pos|neg|\+|-|\d+\s*%|loss|retained)\)?)?"),
]]


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Put the text into one canonical character set.

    Runs first, before any regex, so every downstream pattern (and later the
    BioClinicalBERT tokenizer) sees consistent characters.
    """
    # NFKC: fold compatibility chars -- ligatures, full-width digits, and
    # non-breaking / narrow spaces all collapse to their plain ASCII form.
    text = unicodedata.normalize("NFKC", text)
    # curly quotes and en/em dashes -> straight ASCII (dictation software emits these)
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-")
                .replace(" ", " "))
    # Windows / old-Mac line endings -> "\n" so "\n" in the regexes behaves
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


# A connective word ("and", "with", "the", ...) that is now stranded right
# before punctuation or end-of-line because the noun after it was deleted.
_DANGLING = re.compile(
    r"[ \t]*\b(?:and|or|with|without|of|in|on|to|by|over|the|a|an|is|are|"
    r"showing|shows|present|presence|marked)\b(?=[ \t]*(?:[.,;:)\]]|\n|$))",
    re.IGNORECASE)


def _tidy(text: str) -> str:
    """Repair the punctuation / whitespace debris that span deletion leaves.

    Deleting "...melanoma..." from "shows a melanoma." leaves "shows a ." --
    this turns that back into readable prose. Runs last in clean_field().
    """
    text = re.sub(r"\(\s*\)", " ", text)              # empty "()" left by a deletion
    text = re.sub(r"\[\s*\]", " ", text)              # empty "[]"
    for _ in range(3):        # repeat: each pass can expose a new dangler ("with a." -> "with." -> ".")
        text = _DANGLING.sub("", text)                            # drop stranded "and"/"with"/...
        text = re.sub(r"[ \t]*[-–—•][ \t]*(?=[.,;:)\]\n]|$)", "", text)  # bullet "- ." -> ""
        text = re.sub(r"[ \t]*[,;:]+[ \t]*(?=[.)\]\n]|$)", "", text)     # ", ." -> "."
        text = re.sub(r"\s+([,.;:%)\]])", r"\1", text)                   # "word ," -> "word,"
        text = re.sub(r"(?:\s*[,;:]\s*){2,}", ", ", text)               # ", ,"  -> ", "
        text = re.sub(r"[.,;:]*\.(?:\s*\.)*", ".", text)                # ",." / ".." -> "."
    text = re.sub(r"([(\[])\s+", r"\1", text)         # "( word" -> "(word"
    text = re.sub(r"[ \t]{2,}", " ", text)            # collapse runs of spaces
    # Flatten to ONE line: split on newlines, trim leading bullets/punctuation
    # from each line, drop lines with no letters/digits, rejoin with a space.
    # (BioClinicalBERT mean-pools tokens -- newlines carry no signal.)
    lines = (ln.strip(" \t-–—•*·:;,") for ln in text.split("\n"))
    text = " ".join(ln for ln in lines if re.search(r"[A-Za-z0-9]", ln))
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _apply(text: str) -> str:
    """Run every rule in _RULES and DELETE each match (this is the `strip` policy).

    _RULES holds ~60 compiled regexes for diagnosis terms, staging, Breslow /
    depth, mitoses, margins, growth phase, invasion and immunostains.
    """
    for _name, pat in _RULES:        # _name is just the category label, unused here
        text = pat.sub("", text)     # replace every match with "" (delete)
    return text


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
# keep   normalise + de-id + drop artefacts only               -> plain prose
# strip  + regex-delete diagnostic/staging/invasion spans      -> plain prose
# fact   + LLM keeps only literal morphological observation     -> plain prose
# drop   field excluded entirely
#
# All four emit ordinary text -- safe for a FROZEN BioClinicalBERT encoder.
# There is deliberately no semantic-tagging policy: BioClinicalBERT's WordPiece
# vocab has no tag token, so tags injected into a frozen (non-fine-tuned) encoder
# are shredded into junk subwords. Watson et al.'s SemanticTagging worked only
# because they fine-tuned the encoder end-to-end. If you later need their
# per-category ablation, add it on a fine-tuned encoder with tags registered via
# tokenizer.add_special_tokens + model.resize_token_embeddings.
POLICIES = ("keep", "strip", "fact", "drop")
_LLM_POLICIES = ("fact",)

# Watson et al. (Commun Med 2026) style ladder of increasingly strict removal,
# adapted for pathology reports. Report the downstream classifier at SEVERAL
# levels, like their Table 6 -- don't pick one.
LEVEL_PRESETS: dict[str, dict] = {
    "orig":     dict(diagnosis="keep",  macroscopic="keep"),   # L0 unfiltered (upper bound / leak demo)
    "termfilt": dict(diagnosis="strip", macroscopic="strip"),  # L1 regex: entity names + values
    "diagdrop": dict(diagnosis="drop",  macroscopic="strip"),  # L2 drop synoptic dx, strip gross text
    "fact":     dict(diagnosis="fact",  macroscopic="fact"),   # L3 LLM: literal observation only (Watson FFilt)
    "notext":   dict(diagnosis="drop",  macroscopic="drop"),   # L4 image-only lower bound
}


@dataclass
class CleanConfig:
    diagnosis: str = "strip"
    macroscopic: str = "strip"
    lowercase: bool = False          # Bio_ClinicalBERT is cased -> keep False
    text_cols: dict | None = None
    label_col: str = DEFAULT_LABEL_COL
    llm: object = None               # callable(text, policy) -> str; see llm_filter

    def col(self, key: str) -> str:
        """Map "diagnosis"/"macroscopic" to the actual dataframe column name."""
        return (self.text_cols or DEFAULT_TEXT_COLS)[key]

    def __post_init__(self):
        # fail fast on a typo'd policy string
        for f in ("diagnosis", "macroscopic"):
            if getattr(self, f) not in POLICIES:
                raise ValueError(f"{f} policy must be one of {POLICIES}")

    @classmethod
    def from_level(cls, level: str, **kw) -> "CleanConfig":
        """Build a config from a named ladder rung (see LEVEL_PRESETS).

        e.g. from_level("diagdrop") -> diagnosis="drop", macroscopic="strip".
        Extra kwargs (label_col=, llm=, ...) are passed straight through.
        """
        if level not in LEVEL_PRESETS:
            raise ValueError(f"level must be one of {list(LEVEL_PRESETS)}")
        return cls(**LEVEL_PRESETS[level], **kw)


def clean_field(text: str, policy: str, llm=None) -> str:
    """Clean ONE field (the diagnosis text OR the macroscopic text).

    Pipeline, in order:
        0. bail out if the field is dropped / empty / "nan"
        1. _normalise   -- canonical characters
        2. _PHI_RULES   -- de-identify   (always)
        3. _ARTIFACT_RULES -- delete lab/grossing boilerplate  (always)
        4. policy-specific step:
             strip -> _apply   (regex-delete diagnostic spans)
             fact  -> llm       (LLM rewrites to observation-only)
             keep  -> nothing
        5. _tidy        -- repair debris, flatten to one line
    """
    # 0. "drop" policy, or a blank / literal-"nan" cell -> contribute nothing
    if policy == "drop" or not text or not str(text).strip() or str(text).lower() == "nan":
        return ""

    # 1. canonical character set
    text = _normalise(str(text))

    # 2. de-identify: name/date/id/email spans -> <NAME> <DATE> <ID> <EMAIL>
    for pat, repl in _PHI_RULES:
        text = pat.sub(repl, text)
    # 3. drop processing artefacts (I.T./I.P., cassette codes, "H&E", stray "nan", ...)
    for pat in _ARTIFACT_RULES:
        text = pat.sub(" ", text)

    # 4. the part that differs per ladder rung
    if policy == "strip":                       # termfilt (both fields), diagdrop (macro)
        text = _apply(text)                     # regex-delete diagnosis/staging/invasion spans
    elif policy in _LLM_POLICIES:               # fact
        if llm is None:
            raise RuntimeError(f"policy '{policy}' needs an llm callable (see llm_filter)")
        # regex-strip FIRST (kills "melanoma"/"nevus"/staging deterministically),
        # then the LLM only has to remove residual framing + criteria + management
        text = llm(_tidy(_apply(text)), policy)
    # policy == "keep" (orig level) -> leave the clinical content as-is

    # 5. punctuation repair + single-line flatten
    return _tidy(text)


def build_full_report(row: pd.Series, cfg: CleanConfig | None = None) -> str:
    """Clean both fields of one row and join them into the string BioBERT sees.

    This is the function you `df.apply(...)` over.
    """
    cfg = cfg or CleanConfig()
    parts = [
        # each field is cleaned with its own policy (they can differ -- see LEVEL_PRESETS)
        clean_field(row.get(cfg.col("diagnosis"), ""),   cfg.diagnosis,   cfg.llm),
        clean_field(row.get(cfg.col("macroscopic"), ""), cfg.macroscopic, cfg.llm),
    ]
    report = " ".join(p for p in parts if p)          # join with a space, skip empty parts
    report = re.sub(r"\s{2,}", " ", report).strip()   # collapse the seam
    return report.lower() if cfg.lowercase else report   # lowercase only if explicitly asked


# --------------------------------------------------------------------------
# LLM filtering  (Watson et al. used Llama 3.1 8B, not fine-tuned, few-shot,
# one report at a time, freetext only -- no metadata/label given to the LLM,
# with manual inspection of samples afterwards. Replicate that.)
#
# Backend: any OpenAI-compatible /chat/completions endpoint
#   Ollama : OPENAI_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.1:8b
#   vLLM   : OPENAI_BASE_URL=http://localhost:8000/v1   LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
# --------------------------------------------------------------------------
_FACT_SYSTEM = """You clean ONE excerpt from a skin pathology report for a label-leakage study.
The excerpt is either a DIAGNOSIS section or a MACROSCOPIC (gross) description; the diagnosis
terms have already been blanked out, leaving gaps.

KEEP ONLY: the anatomical site; specimen dimensions and shape; the number, size, colour, pigmentation,
symmetry and border of the lesion(s); the type of biopsy (punch, excision, shave).

DELETE EVERYTHING ELSE, in particular:
- any statement of what the lesion IS, or any leftover fragment of one (naevus, nevus, melanoma,
  dysplasia, "the diagnosis is", "compatible with", "consistent with", "shows a", "with mild atypia")
- microscopic architecture / cytology: fibroplasia, lamellar/concentric changes, pigmentary
  incontinence, lymphoid or lymphocytic infiltration, junctional/dermal component, atypia, mitoses
- grading, Breslow/Clark/depth, stage, margin or excision-completeness status, invasion, ulceration,
  regression, growth phase, immunostains, comments, referral or management wording
- lab/grossing notes (fixation, orientation, "received", inclusion codes)

RULES:
- If nothing survives, output an empty string.
- Do not paraphrase or invent. Reuse the report's own words. Keep the original language.
- Output the cleaned text only -- no preamble, no explanation, no bullet points."""

_FACT_FEWSHOT = [
    # diagnosis section that is ENTIRELY diagnosis -> nothing survives
    ("The diagnosis is a junctional  with  and involutive changes compatible with a .",
     ""),
    # diagnosis section -> keep only the site
    ("Cutaneous lesion on the right shoulder: compound  with mild-moderate changes in the "
     "junctional component, in the superficial dermis.",
     "Lesion on the right shoulder."),
    # diagnosis section with a benign entity -> keep only the site
    ("Skin biopsy from the lateral region of the neck: junctional  with , compatible with a . "
     "Presence of marked inflammation.",
     "Skin biopsy from the lateral region of the neck."),
    # macroscopic description -> keep dimensions + lesion description, drop grossing note
    ("A cutaneous sample measuring 1.8 x 0.8 cm with a pigmented macule of up to 0.8 cm in diameter "
     "on its surface, showing mild-moderate changes in the junctional component. I.T.",
     "A skin sample measuring 1.8 x 0.8 cm with a pigmented macule up to 0.8 cm in diameter on its surface."),
    # macroscopic description -> keep the lesion's physical description
    ("A cutaneous spindle 20 x 8 mm is received, fixed and unoriented, bearing an asymmetric pigmented "
     "lesion 9 mm in diameter with irregular borders and variegated pigmentation.",
     "A skin spindle 20 x 8 mm bearing an asymmetric pigmented lesion 9 mm in diameter with irregular "
     "borders and variegated pigmentation."),
]


def llm_filter(base_url: str | None = None, model: str | None = None,
               temperature: float = 0.0, cache_path: str | None = None):
    """Return the callable that the `fact` policy uses: filter(text) -> text.

    Talks to any OpenAI-compatible /chat/completions server (Ollama, vLLM).
    Results are cached to a JSON file keyed by a hash of (policy, model, text),
    so a second run -- or the audit -- costs no API calls.
    """
    import json, os, hashlib, urllib.request

    # connection settings: explicit arg > environment variable > local Ollama default
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
    model = model or os.environ.get("LLM_MODEL", "llama3.1:8b")
    key = os.environ.get("OPENAI_API_KEY", "not-needed")   # Ollama ignores it; vLLM may want it

    # fingerprint the prompt+examples: if you edit _FACT_SYSTEM / _FACT_FEWSHOT
    # the cache key changes, so stale answers are NOT reused
    _prompt_fp = hashlib.sha1(
        (_FACT_SYSTEM + repr(_FACT_FEWSHOT)).encode()).hexdigest()[:8]

    # load an existing cache file if one was given (create its folder if needed)
    cache: dict[str, str] = {}
    cpath = Path(cache_path) if cache_path else None
    if cpath:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        if cpath.exists():
            cache = json.loads(cpath.read_text(encoding="utf-8"))

    def _call(messages: list[dict]) -> str:
        """One HTTP POST to the chat-completions endpoint; return the reply text."""
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": temperature}).encode()
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def _filter(text: str, policy: str = "fact") -> str:
        """Rewrite ONE field to observation-only text (cached)."""
        if not text.strip():
            return ""
        h = hashlib.sha1(f"{policy}|{model}|{_prompt_fp}|{text}".encode()).hexdigest()
        if h in cache:                       # already done on a previous run (same prompt + text)
            return cache[h]
        # system prompt + 3 few-shot examples + the field to rewrite
        # (the LLM sees ONLY the text -- no label, no patient metadata)
        msgs = [{"role": "system", "content": _FACT_SYSTEM}]
        for src, tgt in _FACT_FEWSHOT:
            msgs += [{"role": "user", "content": src},
                     {"role": "assistant", "content": tgt}]
        msgs.append({"role": "user", "content": text})
        out = _call(msgs)
        cache[h] = out
        if cpath:                            # persist after every call (crash-safe)
            cpath.write_text(json.dumps(cache), encoding="utf-8")
        return out

    return _filter


# Backwards-compatible shim: clinical_pipeline.py used to define its own
# strip_diagnosis_terms(); keep the name working so older code/imports don't break.
def strip_diagnosis_terms(text: str) -> str:
    return clean_field(text, "strip")


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
def leakage_probe(reports: list[str], labels: list, n_splits: int = 5, top_k: int = 15):
    """Ask: can a dumb word-count model still guess the label from the CLEANED text?

    A TF-IDF (1-2 gram) bag-of-words + logistic regression, scored by 5-fold CV
    balanced accuracy. If that stays well above chance, the diagnosis is still
    leaking through vocabulary / phrasing / report structure that cleaning missed.
    Returns the balanced accuracy.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score, classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder

    # string labels ("Nevus", ...) -> integers 0..k-1
    y = LabelEncoder().fit(labels)
    y_enc = y.transform(labels)
    classes = list(y.classes_)
    n_classes = len(classes)

    # the probe model: TF-IDF features -> class-balanced logistic regression
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    # out-of-fold predictions: every report is predicted by a model that did NOT see it
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(pipe, reports, y_enc, cv=skf)

    # headline number + a plain-language verdict
    bal = balanced_accuracy_score(y_enc, preds)
    chance = 1.0 / n_classes
    print("\n== Leakage probe (TF-IDF + logreg on cleaned text) ==")
    print(f"  balanced accuracy : {bal:.3f}   (chance = {chance:.3f})")
    verdict = ("STRONG residual leakage" if bal > chance + 0.30 else
               "some residual leakage"  if bal > chance + 0.12 else
               "near chance - little obvious leakage")
    print(f"  verdict           : {verdict}")
    print(classification_report(y_enc, preds, target_names=classes, zero_division=0))

    # refit on all data just to read off which n-grams drive each class --
    # this list tells you WHAT is still leaking (e.g. "clark", "fibroplasia")
    pipe.fit(reports, y_enc)
    vec = pipe.named_steps["tfidfvectorizer"]
    clf = pipe.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    print(f"  top {top_k} features per class (what the probe keys on):")
    # binary logreg has one coef vector; make it look like two for uniform handling
    coefs = clf.coef_ if n_classes > 2 else np.vstack([-clf.coef_[0], clf.coef_[0]])
    for ci, cls in enumerate(classes):
        top = vocab[np.argsort(coefs[ci])[::-1][:top_k]]   # highest-weight n-grams for this class
        print(f"    {cls:<12}: {', '.join(top)}")
    return bal


def _audit_ladder(df: pd.DataFrame, label_col: str) -> None:
    """Run the leakage probe at every regex level -> a Table-6-style summary.

    A defensible morphology-only result is one where the probe balanced
    accuracy falls toward chance as the level gets stricter.
    """
    if label_col not in df.columns:
        print(f"[audit-ladder skipped: no '{label_col}' column]")
        return
    print(f"\n{'level':<10} {'diag':<6} {'macro':<7} {'n':>4}  {'probe bal.acc':>13}")
    print("-" * 48)
    for lvl, pol in LEVEL_PRESETS.items():
        # skip rungs that need an LLM (fact) or have no text at all (notext)
        if set(pol.values()) & set(_LLM_POLICIES) or lvl == "notext":
            print(f"{lvl:<10} {pol['diagnosis']:<6} {pol['macroscopic']:<7}   (needs LLM / n/a)")
            continue
        # clean the whole dataframe at this rung
        cfg = CleanConfig.from_level(lvl, label_col=label_col)
        cleaned = df.apply(lambda r: build_full_report(r, cfg), axis=1)
        mask = cleaned.str.len() > 0             # exclude rows that cleaned to ""
        try:
            # run the probe but swallow its verbose per-class printout -- we only
            # want the single balanced-accuracy number for the summary table
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bal = leakage_probe(cleaned[mask].tolist(), df.loc[mask, label_col].tolist())
            print(f"{lvl:<10} {pol['diagnosis']:<6} {pol['macroscopic']:<7} {mask.sum():>4}  {bal:>13.3f}")
        except Exception as e:
            print(f"{lvl:<10} {pol['diagnosis']:<6} {pol['macroscopic']:<7}   ERROR: {e}")
    print("\nRun a single level with --audit for the per-class breakdown and top features.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load(path: Path) -> pd.DataFrame:
    """Read the reports file -- Excel, or CSV with a utf-8 -> latin-1 fallback."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def main() -> None:
    """CLI entry point: clean a reports CSV, optionally preview / save / audit."""
    # default --input comes from config.CLINICAL_INPUT if the repo config imports
    default_input = None
    try:  # optional repo config
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import CLINICAL_INPUT
        default_input = str(CLINICAL_INPUT)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=default_input, required=default_input is None)
    ap.add_argument("--level", choices=list(LEVEL_PRESETS),
                    help="Watson-style preset; sets --diagnosis/--macroscopic together")
    ap.add_argument("--diagnosis",   choices=POLICIES, default="strip")
    ap.add_argument("--macroscopic", choices=POLICIES, default="strip")
    ap.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint for --level fact")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-cache", default=None, help="JSON cache path for LLM outputs")
    ap.add_argument("--out", default=None, help="write cleaned reports to this CSV")
    ap.add_argument("--show", type=int, default=6, help="print N before/after diffs")
    ap.add_argument("--audit", action="store_true", help="run the leakage probe")
    ap.add_argument("--audit-ladder", action="store_true",
                    help="run the probe across every regex level and print a Table-6-style summary")
    args = ap.parse_args()

    df = _load(Path(args.input))

    # resolve the two per-field policies: --level preset overrides --diagnosis/--macroscopic
    preset = LEVEL_PRESETS[args.level] if args.level else {}
    _pol = {**{"diagnosis": args.diagnosis, "macroscopic": args.macroscopic}, **preset}
    # only spin up the LLM client if a field actually needs it (fact)
    llm = None
    if set(_pol.values()) & set(_LLM_POLICIES):
        llm = llm_filter(args.llm_base_url, args.llm_model, cache_path=args.llm_cache)
    cfg = CleanConfig(**_pol, lowercase=args.lowercase, label_col=args.label_col, llm=llm)
    print(f"Loaded {len(df)} rows  |  diagnosis={cfg.diagnosis}  macroscopic={cfg.macroscopic}")

    # --audit-ladder: probe every rung and print the summary table, then stop
    if args.audit_ladder:
        return _audit_ladder(df, cfg.label_col)

    # raw = the two fields joined, for the side-by-side preview / CSV
    raw = df.apply(lambda r: "\n".join(
        str(r.get(cfg.col(k), "") or "") for k in ("diagnosis", "macroscopic")).strip(),
        axis=1)
    # cleaned = the actual model input at this config
    cleaned = df.apply(lambda r: build_full_report(r, cfg), axis=1)

    print(f"\navg length  raw {raw.str.len().mean():6.0f}  ->  clean {cleaned.str.len().mean():6.0f} chars")
    print(f"empty after cleaning : {(cleaned.str.len() == 0).sum()}")

    # --show N: print the first N rows raw-vs-clean so you can eyeball the rules
    for i in list(df.index)[: args.show]:
        print("\n" + "-" * 78)
        lbl = df.at[i, cfg.label_col] if cfg.label_col in df.columns else "?"
        print(f"row {i}   label={lbl}")
        print(f"  RAW  : {raw[i][:400]!r}")
        print(f"  CLEAN: {cleaned[i][:400]!r}")

    # --out FILE: write raw + cleaned columns side by side for manual review of every row
    if args.out:
        cols = {}
        if cfg.label_col in df.columns:
            cols[cfg.label_col] = df[cfg.label_col].values
        cols["raw_diagnosis"]   = df.get(cfg.col("diagnosis"), "")
        cols["raw_macroscopic"] = df.get(cfg.col("macroscopic"), "")
        cols["full_report"]     = cleaned.values          # what BioBERT sees
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cols).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  (raw + cleaned side by side -- eyeball all {len(df)} rows)")

    # --audit: run the leakage probe on this single config (full printout)
    if args.audit:
        if cfg.label_col not in df.columns:
            print(f"\n[audit skipped: no '{cfg.label_col}' column]")
        else:
            mask = cleaned.str.len() > 0     # probe only rows with text left
            leakage_probe(cleaned[mask].tolist(), df.loc[mask, cfg.label_col].tolist())


if __name__ == "__main__":
    main()
