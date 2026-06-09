"""
PanDerm Fine-Tuning — 5-Fold Cross-Validation
==============================================
Fine-tunes PanDerm Large (ViT-L/16) on melanocytic skin lesion dermoscopy images
using patient-level stratified 5-fold cross-validation.

Usage:
    python panderm_finetuning.py

Outputs (per fold, saved to OUTPUT_DIR/results_fold{i}/):
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PANDERM_CLASS, CHECKPOINT_LARGE,
    CSV_DIR, DATA_ROOT, SEGMENTED_DIR, OUTPUT_DIR,
    NB_CLASSES, BATCH_SIZE, NUM_WORKERS, N_FOLDS,
)

# ── Fine-tuning hyperparameters ───────────────────────────────────────────
MODEL         = "PanDerm_Large_FT"
EPOCHS        = 50
WARMUP_EPOCHS = 5
LAYER_DECAY   = 0.65
DROP_PATH     = 0.2
WEIGHT_DECAY  = 0.05
MIXUP         = 0.8
CUTMIX        = 1.0

CSV_PATHS = {i: str(CSV_DIR / f"panderm_finetuning_fold{i}.csv") for i in range(N_FOLDS)}


# ── Patch PanDerm for PyTorch 2.x ────────────────────────────────────────
def patch_weights_only():
    script = PANDERM_CLASS / "run_class_finetuning.py"
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
        folds_to_run = list(range(N_FOLDS))

    for fold in folds_to_run:
        out_dir = OUTPUT_DIR / f"results_fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "test.csv").exists():
            print(f"Fold {fold}: already done — skipping")
            continue

        cmd = [
            "python", "run_class_finetuning.py",
            "--model",                 MODEL,
            "--pretrained_checkpoint", str(CHECKPOINT_LARGE),
            "--nb_classes",            str(NB_CLASSES),
            "--batch_size",            str(BATCH_SIZE),
            "--epochs",                str(EPOCHS),
            "--layer_decay",           str(LAYER_DECAY),
            "--drop_path",             str(DROP_PATH),
            "--weight_decay",          str(WEIGHT_DECAY),
            "--mixup",                 str(MIXUP),
            "--cutmix",                str(CUTMIX),
            "--warmup_epochs",         str(WARMUP_EPOCHS),
            "--num_workers",           str(NUM_WORKERS),
            "--no_auto_resume",
            "--weights", "--sin_pos_emb",
            "--wandb_name",            f"PanDerm_FT_fold{fold}",
            "--csv_path",              CSV_PATHS[fold],
            "--root_path",             "",
            "--output_dir",            str(out_dir),
        ]

        print(f"\n{'='*60}\nStarting fold {fold} ...\n{'='*60}")
        result = subprocess.run(cmd, cwd=str(PANDERM_CLASS))

        if result.returncode == 0:
            print(f"\nFold {fold} complete")
        else:
            print(f"\nFold {fold} FAILED (return code {result.returncode})")
            break


if __name__ == "__main__":
    patch_weights_only()
    run_finetuning()
