"""
Transfer evaluation: public dermoscopy cohort -> in-house cohort
=================================================================
Uses the patient-level features both pipelines already saved (frozen PanDerm-L, so the
1,024 numbers mean the same thing in both cohorts) and the same classifier as evaluate.py
(LogisticRegression, class_weight="balanced", lbfgs, C from {0.01, 0.03, 0.1, 0.3, 1.0}
chosen by inner 3-fold CV on balanced accuracy). Nothing is re-extracted.

Four experiments per feature set (e.g. frozen/none = raw-image CLS, maskguided/sam_v2):

    A0 public       the usual public 5-fold CV (public folds from prepare_data.py): train AND test
                    on public patients. Reproduces the public evaluate.py results; the baseline for A.
    A  external     fit on ALL public patients (C chosen on public only), predict ALL 163
                    in-house patients once. No in-house patient is used for training or for
                    choosing C. 95% CIs by bootstrapping the in-house patients.
    B0 inhouse      the usual in-house 5-fold CV (the in-house folds from prepare_data.py).
                    Reproduces evaluate.py up to the inner-CV grouping; the baseline for B.
    B  combined     the same in-house folds, but every training fold also contains ALL public
                    patients. Test patients are in-house only. Does public data help in-house?

Inner CV for choosing C groups rows by patient (StratifiedGroupKFold), so an MSKCC patient
with lesions in two classes never sits on both sides of an inner split.

Inputs:
    PhD_Data/public_pipeline/outputs/features/<mode>/<backend>/
    PhD_Data/inhouse_pipeline/outputs/features/<mode>/<backend>/
        patient_features_fold0.npy, patient_labels.npy, patient_group_ids.npy, group_fold_mapping.csv

Outputs (PhD_Data/transfer_eval/):
    transfer_summary.csv                        one row per feature set x experiment: overall + per-class metrics, C
    predictions/<experiment>_<mode>_<backend>.csv   one row per test patient (public for A0, in-house
                                                    otherwise), with its source (D7P / MILK / MSKCC / inhouse)
    confusion/<experiment>_<mode>_<backend>.csv     3 x 3 confusion matrix (rows = true)

Usage (oncology env, from the repo root):
    python panderm/transfer_eval.py                                   # every feature set present in both cohorts
    python panderm/transfer_eval.py --features frozen/none maskguided/sam_v2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, precision_recall_fscore_support, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASS_LABELS, LOGREG_MAX_ITER, RANDOM_SEED, N_FOLDS

PHD = Path(r"C:\Users\qp263329\OneDrive - Queen Mary, University of London\Desktop\PhD_Data")
PUBLIC = PHD / "public_pipeline" / "outputs" / "features"
INHOUSE = PHD / "inhouse_pipeline" / "outputs" / "features"
OUT = PHD / "transfer_eval"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
CLASSES = sorted(CLASS_LABELS, key=CLASS_LABELS.get)       # ["DN", "MIA", "Minsitu"] in label order
N_BOOT = 2000


# ── Data ─────────────────────────────────────────────────────────────────

def load(root: Path, feature_set: str) -> dict:
    """Patient-level features of one cohort. Frozen features are identical for every fold,
    so fold 0's file holds them all. patient = group id without the '__<diagnosis>' suffix."""
    d = root / feature_set
    groups = np.load(d / "patient_group_ids.npy", allow_pickle=True).astype(str)
    fold_map = pd.read_csv(d / "group_fold_mapping.csv").set_index("group_id")["fold"]
    return {"X": np.load(d / "patient_features_fold0.npy"),
            "y": np.load(d / "patient_labels.npy"),
            "groups": groups,
            "patients": np.array([g.split("__")[0] for g in groups]),
            "folds": fold_map.reindex(groups).to_numpy()}


def shared_feature_sets() -> list:
    def sets(root):
        return {str(p.parent.relative_to(root)).replace("\\", "/")
                for p in root.glob("*/*/patient_features_fold0.npy")}
    return sorted(sets(PUBLIC) & sets(INHOUSE))


# ── Classifier (same as evaluate.py, inner CV grouped by patient) ───────

def fit(X, y, patients):
    gs = GridSearchCV(
        LogisticRegression(max_iter=LOGREG_MAX_ITER, class_weight="balanced",
                           solver="lbfgs", random_state=RANDOM_SEED),
        param_grid={"C": C_GRID},
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        scoring="balanced_accuracy", n_jobs=-1)
    gs.fit(X, y, groups=patients)
    return gs.best_estimator_, gs.best_params_["C"]


# ── Metrics ──────────────────────────────────────────────────────────────

def metrics(y, pred, proba) -> dict:
    y_bin = label_binarize(y, classes=range(len(CLASSES)))
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    out = {"balanced_accuracy": balanced_accuracy_score(y, pred),
           "accuracy": accuracy_score(y, pred),
           "macro_auc": roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"),
           "macro_f1": f1_score(y, pred, average="macro"),
           "cohen_kappa": cohen_kappa_score(y, pred)}
    for k, c in enumerate(CLASSES):
        out[f"{c}_recall"], out[f"{c}_precision"], out[f"{c}_f1"] = rec[k], prec[k], f1[k]
        out[f"{c}_auc"] = roc_auc_score(y_bin[:, k], proba[:, k])
    return out


def bootstrap_ci(y, pred, proba, rng) -> dict:
    """95% CI of balanced accuracy and macro AUC by resampling the test patients."""
    bal, auc = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < len(CLASSES):
            continue
        bal.append(balanced_accuracy_score(y[i], pred[i]))
        auc.append(roc_auc_score(label_binarize(y[i], classes=range(len(CLASSES))), proba[i],
                                 average="macro", multi_class="ovr"))
    return {"balanced_accuracy_ci95": f"{np.percentile(bal, 2.5):.3f}-{np.percentile(bal, 97.5):.3f}",
            "macro_auc_ci95": f"{np.percentile(auc, 2.5):.3f}-{np.percentile(auc, 97.5):.3f}"}


# ── Experiments ──────────────────────────────────────────────────────────

def external(pub, inh, rng):
    """A: fit on all public patients, predict all in-house patients once."""
    clf, c = fit(pub["X"], pub["y"], pub["patients"])
    proba = clf.predict_proba(inh["X"])
    pred = proba.argmax(axis=1)
    row = {**metrics(inh["y"], pred, proba), **bootstrap_ci(inh["y"], pred, proba, rng),
           "C": str(c), "n_train": len(pub["y"]), "n_test": len(inh["y"])}
    return row, pred, proba, np.full(len(inh["y"]), -1), inh


def cross_validate(test, extra=None):
    """5-fold CV over the `test` cohort's own folds (from its prepare_data.py). With `extra`,
    every training fold also gets ALL of the extra cohort's patients; test patients are always
    from `test` only. A0 = cv(public), B0 = cv(in-house), B = cv(in-house, extra=public)."""
    pred = np.zeros(len(test["y"]), int)
    proba = np.zeros((len(test["y"]), len(CLASSES)))
    fold_rows, cs = [], []
    for f in range(N_FOLDS):
        tr, te = test["folds"] != f, test["folds"] == f
        X, y, p = test["X"][tr], test["y"][tr], test["patients"][tr]
        if extra is not None:
            X = np.vstack([extra["X"], X])
            y = np.concatenate([extra["y"], y])
            p = np.concatenate([np.char.add("extra:", extra["patients"]), np.char.add("test:", p)])
        clf, c = fit(X, y, p)
        proba[te] = clf.predict_proba(test["X"][te])
        pred[te] = proba[te].argmax(axis=1)
        fold_rows.append(metrics(test["y"][te], pred[te], proba[te]))
        cs.append(c)
    folds = pd.DataFrame(fold_rows)
    row = metrics(test["y"], pred, proba)
    for k in ["balanced_accuracy", "macro_auc", "macro_f1", "cohen_kappa"]:
        row[f"{k}_fold_mean"], row[f"{k}_fold_sd"] = folds[k].mean(), folds[k].std()
    row.update({"C": ",".join(map(str, cs)), "n_train": int(len(y)), "n_test": len(test["y"])})
    return row, pred, proba, test["folds"], test


EXPERIMENTS = {"A0_public": lambda pub, inh, rng: cross_validate(pub),
               "A_external": lambda pub, inh, rng: external(pub, inh, rng),
               "B0_inhouse": lambda pub, inh, rng: cross_validate(inh),
               "B_combined": lambda pub, inh, rng: cross_validate(inh, extra=pub)}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=None,
                    help="feature sets as <mode>/<backend> (default: all present in both cohorts)")
    args = ap.parse_args()
    feature_sets = args.features or shared_feature_sets()
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "confusion").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"Feature sets: {feature_sets}\nOutputs: {OUT}\n")

    rows = []
    for fs in feature_sets:
        pub, inh = load(PUBLIC, fs), load(INHOUSE, fs)
        if pub["X"].shape[1] != inh["X"].shape[1]:
            print(f"  {fs}: feature sizes differ ({pub['X'].shape[1]} vs {inh['X'].shape[1]}), skipped")
            continue
        for name, run in EXPERIMENTS.items():
            row, pred, proba, folds, test = run(pub, inh, rng)
            rows.append({"feature_set": fs, "experiment": name, **row})
            tag = f"{name}_{fs.replace('/', '_')}"
            pd.DataFrame({"group_id": test["groups"], "source": [g.split("-")[0] if "-" in g else "inhouse"
                                                                  for g in test["groups"]],
                          "fold": folds, "true_label": test["y"], "pred_label": pred,
                          **{f"prob_{c}": proba[:, k] for k, c in enumerate(CLASSES)}}
                         ).to_csv(OUT / "predictions" / f"{tag}.csv", index=False)
            pd.DataFrame(confusion_matrix(test["y"], pred, labels=range(len(CLASSES))),
                         index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]
                         ).to_csv(OUT / "confusion" / f"{tag}.csv")
            sd = f" ± {row['balanced_accuracy_fold_sd']:.3f}" if "balanced_accuracy_fold_sd" in row else \
                 f" [{row['balanced_accuracy_ci95']}]"
            print(f"  {fs:24s} {name:11s} BalAcc={row['balanced_accuracy']:.3f}{sd}  AUC={row['macro_auc']:.3f}  "
                  f"recall DN/MIA/in situ = {row['DN_recall']:.2f}/{row['MIA_recall']:.2f}/{row['Minsitu_recall']:.2f}"
                  f"  C={row['C']}")
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "transfer_summary.csv", index=False)
    print(f"Saved {OUT / 'transfer_summary.csv'}")
    print("Balanced accuracy, pooled over the 163 in-house patients:")
    print(summary.pivot(index="feature_set", columns="experiment", values="balanced_accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
