"""
PanDerm Fine-Tuning — 5-Fold Cross-Validation
==============================================
Fine-tunes PanDerm Large (ViT-L/16) on melanocytic skin lesion dermoscopy images
using patient-level stratified 5-fold cross-validation.

Usage:
    python panderm_finetuning.py

Outputs (per fold, saved to CONFIG["output_dir"]/results_fold{i}/):
    checkpoint-best.pth   — best model checkpoint
    test.csv              — per-image predictions

Requirements:
    - PanDerm repo cloned: git clone https://github.com/SiyuanYan1/PanDerm.git
    - timm==0.9.16
    - Checkpoint: panderm_ll_data6_checkpoint-499.pth
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "panderm_repo":    "/path/to/PanDerm/classification",
    "checkpoint":      "/path/to/panderm_ll_data6_checkpoint-499.pth",
    "csv_dir":         "/path/to/cross-fold-csv",
    "image_root":      "/path/to/dermoscopy",
    "segmented_cache": "/path/to/segmented_cache",
    "manifest":        "/path/to/dataset_manifest.csv",
    "output_dir":      "./results",
    "model":           "PanDerm_Large_FT",
    "nb_classes":      3,
    "batch_size":      32,
    "epochs":          50,
    "warmup_epochs":   5,
    "layer_decay":     0.65,
    "drop_path":       0.2,
    "weight_decay":    0.05,
    "mixup":           0.8,
    "cutmix":          1.0,
    "num_workers":     4,
    "n_folds":         5,
}

CSV_PATHS = {i: str(Path(CONFIG["csv_dir"]) / f"panderm_finetuning_fold{i}.csv")
             for i in range(CONFIG["n_folds"])}


# ── Patch PanDerm for PyTorch 2.x ────────────────────────────────────────
def patch_weights_only():
    script = Path(CONFIG["panderm_repo"]) / "run_class_finetuning.py"
    result = subprocess.run(
        ["sed", "-i",
         "s/torch.load(checkpoint_path)/torch.load(checkpoint_path, weights_only=False)/g",
         str(script)],
        capture_output=True, text=True,
    )
    r = subprocess.run(["grep", "-c", "weights_only=False", str(script)],
                       capture_output=True, text=True)
    ok = r.stdout.strip() not in ("", "0")
    print(f"weights_only patch: {'applied' if ok else 'NOT APPLIED'}")
    return ok


# ── Fine-tune all folds ───────────────────────────────────────────────────
def run_finetuning(folds_to_run=None):
    if folds_to_run is None:
        folds_to_run = list(range(CONFIG["n_folds"]))

    for fold in folds_to_run:
        out_dir = Path(CONFIG["output_dir"]) / f"results_fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "test.csv").exists():
            print(f"Fold {fold}: already done — skipping")
            continue

        cmd = [
            "python", "run_class_finetuning.py",
            "--model",                CONFIG["model"],
            "--pretrained_checkpoint", CONFIG["checkpoint"],
            "--nb_classes",           str(CONFIG["nb_classes"]),
            "--batch_size",           str(CONFIG["batch_size"]),
            "--epochs",               str(CONFIG["epochs"]),
            "--layer_decay",          str(CONFIG["layer_decay"]),
            "--drop_path",            str(CONFIG["drop_path"]),
            "--weight_decay",         str(CONFIG["weight_decay"]),
            "--mixup",                str(CONFIG["mixup"]),
            "--cutmix",               str(CONFIG["cutmix"]),
            "--warmup_epochs",        str(CONFIG["warmup_epochs"]),
            "--num_workers",          str(CONFIG["num_workers"]),
            "--no_auto_resume",
            "--weights", "--sin_pos_emb",
            "--wandb_name",           f"PanDerm_FT_fold{fold}",
            "--csv_path",             CSV_PATHS[fold],
            "--root_path",            "",
            "--output_dir",           str(out_dir),
        ]

        print(f"\n{'='*60}\nStarting fold {fold} ...\n{'='*60}")
        result = subprocess.run(cmd, cwd=CONFIG["panderm_repo"])

        if result.returncode == 0:
            print(f"\nFold {fold} complete")
        else:
            print(f"\nFold {fold} FAILED (return code {result.returncode})")
            break


if __name__ == "__main__":
    patch_weights_only()
    run_finetuning()
