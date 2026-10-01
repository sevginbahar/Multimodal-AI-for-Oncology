"""
PanDerm Partial Fine-Tuning — 5-Fold Cross-Validation
=======================================================
Freezes the patch embedding, CLS token and the first FREEZE_UNTIL transformer
blocks, and fine-tunes the last few blocks + head, in plain PyTorch/AdamW.

Training recipe (v2 — restores the scaffolding of the old full fine-tune):
  - layer decay over the trainable blocks only: head/final norm get PEAK_LR,
    block 23 PEAK_LR*0.65, ... block 18 PEAK_LR*0.65^6
  - linear warmup for WARMUP_EPOCHS, then cosine decay to MIN_LR, stepped
    every optimizer update (not once per epoch)
  - label smoothing on top of the class-weighted loss
  - drop-path on the trainable blocks only (frozen blocks get none)
  - RandomResizedCrop / flips / 45° rotation / hue jitter; no mixup/cutmix,
    which can cut out or blend the subtle features separating MIA / in situ

Evaluation: after each fold, the best checkpoint's OWN head predicts that
fold's test split, softmax outputs are averaged per patient, and the result is
written to test_predictions_fold{i}.csv. evaluate.py --mode finetune reads
those files — no feature extraction or second logistic regression, which would
be trained on features of images this backbone was fine-tuned on.

Outputs per fold:
    checkpoints/finetune/<backend>/results_fold{i}/
        checkpoint-best.pth          full model incl. head
        test_predictions_fold{i}.csv one row per test patient
        training_log_fold{i}.csv     per-epoch train_loss / val_loss / val_bal_acc / lr

A fold whose checkpoint-best.pth already exists is NOT retrained — delete the
old results_fold{i}/ folders before training with a changed recipe.

Usage:
    python panderm/panderm_finetuning.py                 # train + test all folds
    python panderm/panderm_finetuning.py --eval-only     # re-score existing checkpoints only
    python panderm/panderm_finetuning.py --folds 0       # a single fold

Requirements:
    - Checkpoint: panderm_ll_data6_checkpoint-499.pth
    - Per-fold CSVs already built: run_pipeline.py --stage make_csv
"""

import argparse
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PANDERM_CLASS, CHECKPOINT_LARGE, CSV_DIR, OUTPUT_DIR, SEGMENTATION_BACKEND,
    CLASS_NAMES, NB_CLASSES, BATCH_SIZE, NUM_WORKERS, N_FOLDS,
    IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, input_image,
)

# Matches make_panderm_csv.py's own backend-nesting of CSV_DIR.
CSV_DIR = CSV_DIR / SEGMENTATION_BACKEND

# ── Hyperparameters ──────────────────────────────────────────────────────
FREEZE_UNTIL    = 18      # freeze blocks 0..FREEZE_UNTIL-1 of 24; train the rest + head
EPOCHS          = 30
WARMUP_EPOCHS   = 5
PEAK_LR         = 2.5e-4  # head's LR; old full FT used 5e-4 at batch 128, scaled for batch 32
MIN_LR          = 1e-6
LAYER_DECAY     = 0.65    # block 23 = PEAK_LR*0.65 ... block 18 = PEAK_LR*0.65^6
WEIGHT_DECAY    = 0.05
LABEL_SMOOTHING = 0.1
DROP_PATH       = 0.1     # model spreads 0 -> DROP_PATH over depth; frozen blocks set to 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Model ─────────────────────────────────────────────────────────────────

def build_model() -> torch.nn.Module:
    """PanDerm-L with a fresh 3-class head. With the constructor defaults it
    reads out the CLS token after the final `norm` (no fc_norm / mean pooling),
    so the pretrained `norm.*` weights load as-is — no key renaming needed."""
    for p in [str(PANDERM_CLASS.parent), str(PANDERM_CLASS)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.modeling_finetune import panderm_large_patch16_224
    return panderm_large_patch16_224(num_classes=NB_CLASSES, drop_path_rate=DROP_PATH)


def load_state(model: torch.nn.Module, ckpt_path: Path, strict: bool) -> None:
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean = OrderedDict((k.replace("encoder.", "").replace("module.", ""), v)
                        for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=strict)
    if not strict:
        print(f"  Pretrained weights loaded: missing={missing} unexpected={len(unexpected)} "
              f"(missing should be only head.weight/head.bias; unexpected = SSL decoder)")


def freeze_early_blocks(model: torch.nn.Module, freeze_until: int) -> None:
    """patch_embed, cls_token and blocks[:freeze_until] get requires_grad=False
    and no drop-path. pos_embed is already non-trainable (fixed sin-cos)."""
    for p in model.patch_embed.parameters():
        p.requires_grad_(False)
    model.cls_token.requires_grad_(False)   # feeds block 0 — training it shifts the frozen blocks' input
    for block in model.blocks[:freeze_until]:
        block.drop_path = nn.Identity()
        for p in block.parameters():
            p.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen blocks 0-{freeze_until - 1} of {len(model.blocks)}  "
          f"({trainable:,}/{total:,} params trainable, {100 * trainable / total:.1f}%)")


def layer_decay_param_groups(model: torch.nn.Module) -> list:
    """One AdamW group per (LR scale, weight decay) pair. Depth is counted from
    the top: head and final norm = 0 (scale 1), block 23 = 1, ..., block 18 = 6.
    Biases, norms and layer-scale gammas (1-D params) get no weight decay."""
    n_blocks = len(model.blocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = n_blocks - int(name.split(".")[1]) if name.startswith("blocks.") else 0
        scale = LAYER_DECAY ** depth
        wd    = 0.0 if p.ndim <= 1 else WEIGHT_DECAY
        g = groups.setdefault((depth, wd), {"params": [], "lr": PEAK_LR * scale,
                                            "weight_decay": wd, "depth": depth})
        g["params"].append(p)
    for depth in sorted({d for d, _ in groups}):
        print(f"    depth {depth} ({'head/norm' if depth == 0 else f'block {n_blocks - depth}'}): "
              f"peak LR {PEAK_LR * LAYER_DECAY ** depth:.2e}")
    return list(groups.values())


def warmup_cosine(total_steps: int, warmup_steps: int):
    """LR multiplier per optimizer step: linear 0 -> 1, then cosine 1 -> MIN_LR/PEAK_LR."""
    floor = MIN_LR / PEAK_LR
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


# ── Data ──────────────────────────────────────────────────────────────────

class DermDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),   # dermoscopy has no canonical orientation
    transforms.RandomRotation(45),
    transforms.ColorJitter(hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def make_loader(df: pd.DataFrame, transform, shuffle: bool) -> DataLoader:
    ds = DermDataset(df["image"].tolist(), df["label"].tolist(), transform)
    # persistent_workers: on Windows each worker is a fresh process, so
    # re-spawning them every epoch costs several seconds per loader.
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      persistent_workers=NUM_WORKERS > 0)


def group_ids_by_image() -> dict:
    """Image path -> patient group id (patient_id__diagnosis), the same key
    extract_features.py / evaluate.py use for patient-level aggregation."""
    m = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    return dict(zip(m["image_path"].map(input_image),
                    m["patient_id"].astype(str) + "__" + m["diagnosis"]))


def load_fold_csv(fold: int) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / f"panderm_finetuning_fold{fold}.csv")
    missing = ~df["image"].map(lambda p: Path(p).exists())
    if missing.any():
        raise FileNotFoundError(
            f"{missing.sum()} of {len(df)} images in fold {fold} not found, e.g. "
            f"{df.loc[missing, 'image'].iloc[0]}\nRe-run prepare_data.py and make_panderm_csv.py "
            f"if the data has moved.")
    df["group_id"] = df["image"].map(group_ids_by_image())
    if df["group_id"].isna().any():
        raise ValueError(f"Fold {fold}: {df['group_id'].isna().sum()} images not in dataset_manifest.csv")
    return df


# ── Train / test one fold ────────────────────────────────────────────────

@torch.no_grad()
def predict_proba(model: torch.nn.Module, df: pd.DataFrame) -> np.ndarray:
    model.eval()
    probs = []
    for imgs, _ in make_loader(df, eval_tf, shuffle=False):
        probs.append(model(imgs.to(DEVICE)).softmax(dim=1).cpu().numpy())
    return np.concatenate(probs)


def train_fold(fold: int, out_dir: Path) -> None:
    ckpt_path = out_dir / "checkpoint-best.pth"
    if ckpt_path.exists():
        print(f"Fold {fold}: checkpoint already exists — skipping training")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(RANDOM_SEED + fold)

    df = load_fold_csv(fold)
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]

    # class_weight="balanced"-equivalent: n_samples / (n_classes * count_c) —
    # same philosophy as LogisticRegression(class_weight="balanced") elsewhere.
    counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        (counts.sum() / (len(counts) * counts)).values, dtype=torch.float32
    ).to(DEVICE)

    train_loader = make_loader(train_df, train_tf, shuffle=True)
    val_loader   = make_loader(val_df,   eval_tf,  shuffle=False)

    model = build_model()
    load_state(model, CHECKPOINT_LARGE, strict=False)
    freeze_early_blocks(model, FREEZE_UNTIL)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(layer_decay_param_groups(model))
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(EPOCHS * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    head_group = next(g for g in optimizer.param_groups if g["depth"] == 0)

    log, best_val_bal_acc = [], -1.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        y_true, y_pred, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(DEVICE))
                val_loss += criterion(logits, labels.to(DEVICE)).item() * imgs.size(0)
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())
        val_loss /= len(val_loader.dataset)
        val_bal_acc = balanced_accuracy_score(y_true, y_pred)

        log.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                    "val_bal_acc": val_bal_acc, "head_lr": head_group["lr"]})
        pd.DataFrame(log).to_csv(out_dir / f"training_log_fold{fold}.csv", index=False)
        print(f"  Fold {fold} epoch {epoch + 1}/{EPOCHS}: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_bal_acc={val_bal_acc:.3f}  head_lr={head_group['lr']:.2e}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_bal_acc": val_bal_acc}, ckpt_path)

    print(f"  Fold {fold} done — best val balanced accuracy: {best_val_bal_acc:.3f}")
    # optimizer must go too — AdamW's per-parameter momentum/variance buffers
    # (~2x the trainable params' size) stay resident on GPU until this object
    # is actually dereferenced; empty_cache() alone only releases memory
    # Python has already garbage-collected, not memory a live object still holds.
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()


def test_fold(fold: int, out_dir: Path) -> None:
    """Best checkpoint + its own head -> test split only -> patient-averaged softmax."""
    ckpt_path = out_dir / "checkpoint-best.pth"
    if not ckpt_path.exists():
        print(f"Fold {fold}: no checkpoint at {ckpt_path} — skipping test")
        return
    test_df = load_fold_csv(fold).query("split == 'test'").reset_index(drop=True)

    model = build_model()
    load_state(model, ckpt_path, strict=True)
    model.to(DEVICE)
    probs = predict_proba(model, test_df)
    del model
    torch.cuda.empty_cache()

    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    img = pd.concat([test_df[["group_id", "label"]], pd.DataFrame(probs, columns=prob_cols)], axis=1)
    patients = img.groupby("group_id").agg(
        true_label=("label", "first"), n_images=("label", "size"),
        **{c: (c, "mean") for c in prob_cols}).reset_index()
    patients["pred_label"] = patients[prob_cols].values.argmax(axis=1)
    out_path = out_dir / f"test_predictions_fold{fold}.csv"
    patients.to_csv(out_path, index=False)

    bal_acc = balanced_accuracy_score(patients["true_label"], patients["pred_label"])
    print(f"  Fold {fold} test: {len(test_df)} images -> {len(patients)} patients, "
          f"patient-level balanced accuracy = {bal_acc:.3f}  -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="don't train; only write test predictions from existing checkpoints")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    args = parser.parse_args()

    run_dir = OUTPUT_DIR / "checkpoints" / "finetune" / SEGMENTATION_BACKEND
    print(f"Output: {run_dir}")
    if not args.eval_only:
        print(f"Partial fine-tuning: freeze blocks 0-{FREEZE_UNTIL - 1}, train blocks "
              f"{FREEZE_UNTIL}-23 + head | {EPOCHS} epochs ({WARMUP_EPOCHS} warmup, cosine) | "
              f"peak LR {PEAK_LR}, layer decay {LAYER_DECAY} | label smoothing {LABEL_SMOOTHING} | "
              f"drop-path {DROP_PATH} | batch {BATCH_SIZE}")

    for fold in args.folds:
        print(f"\n{'=' * 60}\nFold {fold}\n{'=' * 60}")
        out_dir = run_dir / f"results_fold{fold}"
        if not args.eval_only:
            train_fold(fold, out_dir)
        test_fold(fold, out_dir)

    print(f"\nDone. Next:\n  python panderm/evaluate.py --mode finetune")


if __name__ == "__main__":
    main()
