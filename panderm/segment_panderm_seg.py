"""
PanDerm segmentation — inference on the in-house cohort (no ground truth needed)
================================================================================
Applies a PanDerm segmentation model trained with the authors' code
(C:\\Users\\qp263329\\PanDerm\\segmentation, run.py / run.ps1) to every image in
dataset_manifest.csv, the same way their test stage does, but without scoring:

    image -> resize to 224x224 (bicubic, aspect ratio NOT kept, as in their loader)
          -> normalise mean/std 0.5 -> PanDerm-L + UPerNet -> argmax (lesion vs background)
          -> their largestConnectComponent() (largest component + fill holes)
          -> resize back to the original image size (nearest) -> binary mask

Note on their largestConnectComponent(): if the model predicts NO lesion pixel,
it returns the whole image as the mask. Kept for fidelity; such images are
flagged as empty_prediction in the QC csv.

Images are read with PIL (no EXIF rotation), like extract_features_maskguided.py,
so the masks line up with the images used for mask-guided pooling.

Must run in the `dermseg` conda environment (needs mmseg). Every run rewrites
all masks, so re-running with the final checkpoint replaces earlier ones.

Then compares the masks with Otsu v2 (color_v2) and SAM v2 masks already cached
in features/maskguided/<backend>/masks (no ground truth: agreement, not accuracy)
and draws montages: raw | Otsu v2 | SAM v2 | PanDerm-seg, each mask as an outline
on the image, labelled with its size (% of the image).

Outputs:
    features/maskguided/panderm_seg/masks/<class>/<patient>/<file>.png   0/255 masks
    segmented_cache/panderm_seg/segmentation_qc_panderm_seg.csv          (same place as the other backends' QC)
    segmented_cache/panderm_seg/agreement_panderm_seg.csv                per-image IoU with Otsu v2 / SAM v2
    segmented_cache/panderm_seg/montages/
        montage_random_<class>.png     8 random images per class (seed 0)
        montage_known_cases.png        14149059 (heterogeneous), 16070324 and 14059385 (multi-lesion)
        montage_disagreement.png       12 images where PanDerm-seg agrees least with BOTH Otsu v2 and SAM v2
    segmented_cache/panderm_seg/<class>/<patient>/<file>                 lesion crops for Frozen-CLS, made
        like segment_lesions.process_dataset() (bounding box + 10% margin, background black, 224 px)
    segmentation_qc_montage_panderm_seg.png   (in OUTPUT_DIR) the same QC montage as the other backends:
        original | mask overlay | cropped lesion, the same 20 images (seed 42)

Usage (from the repo root, dermseg env):
    python panderm/segment_panderm_seg.py --ckpt C:/Users/qp263329/phd_runs/panderm_seg/check/0/model_best_0.ckpt
    python panderm/segment_panderm_seg.py          # default: the full 100-epoch model (out/0/model_best_0.ckpt)
    python panderm/segment_panderm_seg.py --montage-only   # redo agreement + montages from the saved masks
    python panderm/segment_panderm_seg.py --crops-only     # redo the Frozen-CLS crops from the saved masks
    python panderm/segment_panderm_seg.py --isic-test --ckpt <ckpt>
        # instead: the 1,000 ISIC 2018 test images -> isic2018_benchmark/test/masks/panderm_seg/
        # (scored by segmentation_benchmark.py --split test --methods color_v2 sam_v2 panderm_seg)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

PANDERM_SEG = Path("C:/Users/qp263329/PanDerm/segmentation")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent))
from config import OUTPUT_DIR, DATA_ROOT, SEGMENTED_DIR, IMAGE_SIZE, CROP_MARGIN
from segment_lesions import apply_mask_and_crop

sys.path.insert(0, str(PANDERM_SEG))
import models.cae_backbone  # noqa: F401  registers the CAE backbone with mmseg (run.py does the same)
from models.cae_seg import CAEv2_seg
from utils.train_utils import largestConnectComponent

BACKEND = "panderm_seg"
MASKS_ROOT = OUTPUT_DIR / "features" / "maskguided"                     # where every backend's masks live
MASK_DIR = MASKS_ROOT / BACKEND / "masks"
QC_DIR = SEGMENTED_DIR / BACKEND                                       # where every backend's QC csv lives
SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Montage: columns after the raw image, as (label, backend folder)
COMPARE = [("Otsu v2", "color_v2"), ("SAM v2", "sam_v2"), ("PanDerm-seg", BACKEND)]
KNOWN_CASES = {"14149059": "heterogeneous", "16070324": "multi-lesion", "14059385": "multi-lesion"}
N_RANDOM, N_DISAGREE, PANEL_W = 8, 12, 360
OUTLINE = (255, 255, 0)   # yellow (RGB)

# --isic-test: masks for the official ISIC 2018 Task 1 test images, written into the cache that
# segmentation_benchmark.py --split test reads (same mask format as its other methods)
ISIC_TEST_IMAGES = DATA_ROOT.parent / "Data" / "ISIC2018_Task1-2_Test_Input" / "ISIC2018_Task1-2_Test_Input"
ISIC_TEST_MASKS = DATA_ROOT.parent / "isic2018_benchmark" / "test" / "masks" / BACKEND


def load_model(ckpt: Path) -> CAEv2_seg:
    """Build their CAEv2_seg and load the trained Lightning checkpoint into it.
    CAEv2_seg reads models/cae_config.py and model_weights/ with relative paths,
    so it is built from inside their segmentation folder."""
    cwd = os.getcwd()
    os.chdir(PANDERM_SEG)
    try:
        model = CAEv2_seg()
    finally:
        os.chdir(cwd)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    # Lightning keys are "model.segmentor...." (Segmentation_Module.model = CAEv2_seg)
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)   # strict: every trained weight must match
    return model.to(DEVICE).eval()


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """Their test transform: cv2 bicubic resize to 224x224, ToTensor, Normalize(0.5, 0.5)."""
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def segment(model: CAEv2_seg, rgb: np.ndarray):
    logits = model(preprocess(rgb).to(DEVICE))[0]                       # 2 x 224 x 224
    prob = torch.softmax(logits, dim=0)[1].cpu().numpy()                # lesion probability
    pred = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    n_comp = cv2.connectedComponents(pred, connectivity=8)[0] - 1       # before their LCC
    lcc = largestConnectComponent(pred).astype(np.uint8)
    h, w = rgb.shape[:2]
    mask = cv2.resize(lcc, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    mean_prob = float(prob[lcc > 0].mean()) if lcc.any() else float("nan")
    return mask, n_comp, mean_prob


def mask_path(backend: str, image_path: str) -> Path:
    return (MASKS_ROOT / backend / "masks" / Path(image_path).relative_to(DATA_ROOT)).with_suffix(".png")


def load_mask(backend: str, image_path: str) -> np.ndarray:
    return np.array(Image.open(mask_path(backend, image_path))) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def agreement(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per image: mask sizes of the three methods and PanDerm-seg's IoU with Otsu v2 and SAM v2."""
    rows = []
    for _, row in manifest.iterrows():
        m = {b: load_mask(b, row["image_path"]) for _, b in COMPARE}
        rows.append({"image_path": row["image_path"], "patient_id": row["patient_id"],
                     "diagnosis": row["diagnosis"],
                     **{f"frac_{b}": float(m[b].mean()) for _, b in COMPARE},
                     "iou_vs_color_v2": iou(m[BACKEND], m["color_v2"]),
                     "iou_vs_sam_v2": iou(m[BACKEND], m["sam_v2"]),
                     "iou_color_v2_vs_sam_v2": iou(m["color_v2"], m["sam_v2"])})
    ag = pd.DataFrame(rows)
    ag["iou_vs_best"] = ag[["iou_vs_color_v2", "iou_vs_sam_v2"]].max(axis=1)
    return ag


def panel(rgb: np.ndarray, mask, label: str) -> np.ndarray:
    """Image shrunk to PANEL_W wide, mask outline drawn on it, label in a black bar on top."""
    h = int(round(rgb.shape[0] * PANEL_W / rgb.shape[1]))
    img = cv2.resize(rgb, (PANEL_W, h), interpolation=cv2.INTER_AREA)
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (PANEL_W, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, OUTLINE, 2)
    bar = np.zeros((22, PANEL_W, 3), np.uint8)
    cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def montage(rows: pd.DataFrame, path: Path, title: str):
    """One row per image: raw | Otsu v2 | SAM v2 | PanDerm-seg. Rows padded to a common height."""
    if rows.empty:   # e.g. the in-house known cases on another cohort
        return
    lines = []
    for _, r in rows.iterrows():
        rgb = np.array(Image.open(r["image_path"]).convert("RGB"))
        name = f"{r['diagnosis']} {r['patient_id']} {Path(r['image_path']).stem[-4:]}"
        name = name.encode("ascii", "replace").decode()          # cv2 text is ASCII only
        cells = [panel(rgb, None, name)]
        for label, b in COMPARE:
            extra = f"  IoU vs Otsu {r['iou_vs_color_v2']:.2f} SAM {r['iou_vs_sam_v2']:.2f}" if b == BACKEND else ""
            cells.append(panel(rgb, load_mask(b, r["image_path"]),
                               f"{label} {100 * r[f'frac_{b}']:.0f}%{extra}"))
        lines.append(np.hstack(cells))
    width = max(l.shape[1] for l in lines)
    lines = [np.pad(l, ((0, 4), (0, width - l.shape[1]), (0, 0)), constant_values=255) for l in lines]
    head = np.full((30, width, 3), 255, np.uint8)
    cv2.putText(head, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack([head] + lines)).save(path)
    print(f"  {path.name}: {len(rows)} images")


def compare_and_draw(manifest: pd.DataFrame):
    missing = [b for _, b in COMPARE if not (MASKS_ROOT / b / "masks").exists()]
    if missing:
        print(f"\nSkipping agreement + montages: no cached masks for {missing} in {MASKS_ROOT} yet "
              f"(made by extract_features_maskguided.py). Later: --montage-only")
        return
    ag = agreement(manifest)
    ag.to_csv(QC_DIR / f"agreement_{BACKEND}.csv", index=False)
    print("\nAgreement (median IoU; no ground truth, so agreement != accuracy):")
    print(ag.groupby("diagnosis")[["iou_vs_color_v2", "iou_vs_sam_v2", "iou_color_v2_vs_sam_v2"]]
          .median().round(3).to_string())
    print("Median mask size (% of image) by class:")
    print((100 * ag.groupby("diagnosis")[[f"frac_{b}" for _, b in COMPARE]].median()).round(1).to_string())

    out = QC_DIR / "montages"
    print(f"Montages -> {out}")
    for dx, g in ag.groupby("diagnosis"):
        montage(g.sample(min(N_RANDOM, len(g)), random_state=0), out / f"montage_random_{dx}.png",
                f"{dx}: random {N_RANDOM} (seed 0)   yellow = mask outline")
    known = ag[ag.patient_id.astype(str).isin(KNOWN_CASES)]
    montage(known, out / "montage_known_cases.png",
            "Known cases: " + ", ".join(f"{k} {v}" for k, v in KNOWN_CASES.items()))
    montage(ag.nsmallest(N_DISAGREE, "iou_vs_best"), out / "montage_disagreement.png",
            f"{N_DISAGREE} images where PanDerm-seg agrees least with both Otsu v2 and SAM v2")


def write_crops(manifest: pd.DataFrame):
    """Lesion crops for the Frozen-CLS crop pipeline, made exactly like segment_lesions.process_dataset():
    bounding box + CROP_MARGIN, background blacked out, resized to IMAGE_SIZE, saved at
    segmented_cache/panderm_seg/<class>/<patient>/<file> (= config.input_image() for that backend), where
    extract_features.py reads them when SEGMENTATION_BACKEND = "panderm_seg"."""
    for i, row in manifest.iterrows():
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        crop = apply_mask_and_crop(image, load_mask(BACKEND, row["image_path"]).astype(np.uint8),
                                   IMAGE_SIZE, CROP_MARGIN)
        # always panderm_seg's folder (NOT config.input_image(): that follows SEGMENTATION_BACKEND
        # and for "none" is the raw image itself); extract_features.py reads it with backend "panderm_seg"
        out = SEGMENTED_DIR / BACKEND / Path(row["image_path"]).relative_to(DATA_ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(out, quality=95)
        if (i + 1) % 100 == 0:
            print(f"  crops [{i + 1}/{len(manifest)}]")
    print(f"Crops -> {QC_DIR}")
    qc_montage(manifest)
    print("Next (oncology env): python panderm/extract_features.py, "
          "then python panderm/evaluate.py --mode frozen")


def qc_montage(manifest: pd.DataFrame):
    """The same QC montage segment_lesions.generate_qc_montage() makes for every other backend
    (original | mask overlay | cropped lesion, 20 random images, seed 42 -> the same images),
    drawn from the saved PanDerm-seg masks -> OUTPUT_DIR/segmentation_qc_montage_panderm_seg.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.sort(np.random.RandomState(42).choice(len(manifest), size=min(20, len(manifest)), replace=False))
    fig, axes = plt.subplots(len(idx), 3, figsize=(12, 3 * len(idx)))
    for i, k in enumerate(idx):
        row = manifest.iloc[k]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = load_mask(BACKEND, row["image_path"])
        overlay = image.copy()
        overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        crop = apply_mask_and_crop(image, mask.astype(np.uint8), IMAGE_SIZE, CROP_MARGIN)
        for ax, im, title in zip(axes[i], (image, overlay, crop),
                                 (f"{row['diagnosis']}/{row['patient_id']}", "Mask overlay", "Cropped lesion")):
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    plt.suptitle("Segmentation QC Montage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / f"segmentation_qc_montage_{BACKEND}.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"QC montage saved to: {path}")


def segment_isic_test(model: CAEv2_seg):
    """The 1,000 ISIC 2018 test images -> isic2018_benchmark/test/masks/panderm_seg/ISIC_*.png
    (full original resolution, 0/255), for scoring by segmentation_benchmark.py --split test."""
    paths = sorted(ISIC_TEST_IMAGES.glob("ISIC_*.jpg"))
    if not paths:
        raise SystemExit(f"No ISIC_*.jpg in {ISIC_TEST_IMAGES}")
    ISIC_TEST_MASKS.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    for i, p in enumerate(paths, 1):
        mask, n_comp, _ = segment(model, np.array(Image.open(p).convert("RGB")))
        n_empty += n_comp == 0
        Image.fromarray(mask.astype(np.uint8) * 255).save(ISIC_TEST_MASKS / f"{p.stem}.png")
        if i % 100 == 0:
            print(f"  [{i}/{len(paths)}]")
    print(f"{len(paths)} ISIC test masks -> {ISIC_TEST_MASKS}  (empty predictions: {n_empty})")
    print("Next (oncology env): python panderm/segmentation_benchmark.py --split test "
          "--methods color_v2 sam_v2 panderm_seg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("C:/Users/qp263329/phd_runs/panderm_seg/out/0/model_best_0.ckpt"),
                    help="trained checkpoint from their run.py (model_best_0.ckpt)")
    ap.add_argument("--montage-only", action="store_true",
                    help="skip segmentation; redo agreement + montages from the saved masks")
    ap.add_argument("--isic-test", action="store_true",
                    help="segment the 1,000 ISIC 2018 test images (for segmentation_benchmark.py --split test) "
                         "instead of the in-house cohort")
    ap.add_argument("--crops-only", action="store_true",
                    help="skip segmentation; write Frozen-CLS lesion crops from the saved masks")
    args = ap.parse_args()

    if args.crops_only:
        write_crops(pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv"))
        return
    if args.isic_test:
        if not args.ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {args.ckpt}")
        print(f"ISIC 2018 test images | checkpoint: {args.ckpt} | device: {DEVICE}")
        segment_isic_test(load_model(args.ckpt))
        return

    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    if args.montage_only:
        compare_and_draw(manifest)
        return
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    print(f"Manifest: {len(manifest)} images | checkpoint: {args.ckpt} | device: {DEVICE}")
    model = load_model(args.ckpt)
    print("Trained weights loaded (strict).")

    rows = []
    for i, row in manifest.iterrows():
        rgb = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask, n_comp, mean_prob = segment(model, rgb)

        p = (MASK_DIR / Path(row["image_path"]).relative_to(DATA_ROOT)).with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(p)

        rows.append({"image_path": row["image_path"], "diagnosis": row["diagnosis"],
                     "lesion_frac": float(mask.mean()), "n_components": n_comp,
                     "empty_prediction": n_comp == 0, "mean_lesion_prob": mean_prob,
                     "checkpoint": str(args.ckpt)})
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(manifest)}]")

    qc = pd.DataFrame(rows)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    qc.to_csv(QC_DIR / f"segmentation_qc_{BACKEND}.csv", index=False)
    print(f"\nMasks -> {MASK_DIR}")
    print(f"Empty predictions (whole image used as mask): {int(qc.empty_prediction.sum())}")
    print(f"Images with >1 predicted component (only the largest kept): {int((qc.n_components > 1).sum())}")
    print("Median lesion fraction by class:", qc.groupby("diagnosis").lesion_frac.median().round(3).to_dict())
    print("Re-running with another checkpoint overwrites these masks; re-extract any "
          "mask-guided features made from them.")

    write_crops(manifest)
    compare_and_draw(manifest)


if __name__ == "__main__":
    main()
