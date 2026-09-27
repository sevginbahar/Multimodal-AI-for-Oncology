"""
PanDerm Partial Fine-Tuning — 5-Fold Cross-Validation
=======================================================
Freezes the first FREEZE_UNTIL transformer blocks (+ patch embedding) and
fine-tunes only the last few blocks + head, plain PyTorch/AdamW — no
layer_decay, no mixup/cutmix/drop-path. Replaces the old full-fine-tune
approach (all blocks trainable via an external layer_decay-scaled LR),
which trained every parameter and saved no GPU memory.

Real memory savings here: frozen blocks get requires_grad=False, so they
get no optimizer state (no Adam m/v buffers), no .grad, and — since their
input also doesn't require grad — no retained activations for backprop.

Checkpoints are saved to the SAME path extract_features.py / evaluate.py's
--mode finetune already expects (results_fold{i}/checkpoint-best.pth), so
neither of those scripts needs to change.

Usage:
    python panderm/panderm_finetuning.py

Requirements:
    - Checkpoint: panderm_ll_data6_checkpoint-499.pth
    - Per-fold CSVs already built: run_pipeline.py --stage make_csv
"""

import sys
from collections import OrderedDict
from pathlib import Path

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
    NB_CLASSES, BATCH_SIZE, NUM_WORKERS, N_FOLDS,
    IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED,
)

# Mode first, then backend — matches extract_features.py/evaluate.py's layout:
# checkpoints/finetune/<backend>/, so a none/color/sam ablation run here can't
# silently reuse another backend's checkpoint via the skip-if-exists guard
# in train_fold(). This script only ever does "finetune" — no --mode flag.
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "finetune" / SEGMENTATION_BACKEND
# Matches make_panderm_csv.py's own backend-nesting of CSV_DIR.
CSV_DIR = CSV_DIR / SEGMENTATION_BACKEND

# ── Hyperparameters ──────────────────────────────────────────────────────
FREEZE_UNTIL = 18     # freeze blocks 0..FREEZE_UNTIL-1 of 24; train the rest + head
EPOCHS       = 25
LR           = 1e-4   # one flat LR for every unfrozen param — no layer_decay
WEIGHT_DECAY = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


# ── Model ─────────────────────────────────────────────────────────────────

def load_trainable_model() -> torch.nn.Module:
    for p in [str(PANDERM_CLASS.parent), str(PANDERM_CLASS)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.modeling_finetune import panderm_large_patch16_224
    model = panderm_large_patch16_224(num_classes=NB_CLASSES)
    ckpt = torch.load(str(CHECKPOINT_LARGE), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean = OrderedDict((k.replace("encoder.", "").replace("module.", ""), v)
                        for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"  Pretrained weights loaded: missing={len(missing)} unexpected={len(unexpected)} "
         f"(missing=head, unexpected=SSL decoder — both expected)")
    return model.to(DEVICE)


def freeze_early_blocks(model: torch.nn.Module, freeze_until: int) -> None:
    """patch_embed + blocks[:freeze_until] get requires_grad=False. pos_embed
    is already non-trainable by construction (fixed sin-cos embedding)."""
    for p in model.patch_embed.parameters():
        p.requires_grad_(False)
    for block in model.blocks[:freeze_until]:
        for p in block.parameters():
            p.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen blocks 0-{freeze_until - 1} of {len(model.blocks)}  "
         f"({trainable:,}/{total:,} params trainable, {100 * trainable / total:.1f}%)")


# ── Data ──────────────────────────────────────────────────────────────────

class DermDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:    img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB", (224, 224))
        return self.transform(img), self.labels[idx]


train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),   # dermoscopy has no canonical orientation
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
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)


# ── Train one fold ───────────────────────────────────────────────────────

def train_fold(fold: int) -> None:
    out_dir   = CHECKPOINT_DIR / f"results_fold{fold}"
    ckpt_path = out_dir / "checkpoint-best.pth"
    if ckpt_path.exists():
        print(f"Fold {fold}: checkpoint already exists — skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_DIR / f"panderm_finetuning_fold{fold}.csv")
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

    model = load_trainable_model()
    freeze_early_blocks(model, FREEZE_UNTIL)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_bal_acc = -1.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(DEVICE))
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())
        val_bal_acc = balanced_accuracy_score(y_true, y_pred)

        print(f"  Fold {fold} epoch {epoch + 1}/{EPOCHS}: "
             f"train_loss={train_loss:.4f}  val_bal_acc={val_bal_acc:.3f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save({"model": model.state_dict()}, ckpt_path)

    print(f"  Fold {fold} done — best val balanced accuracy: {best_val_bal_acc:.3f}")
    # optimizer must go too — AdamW's per-parameter momentum/variance buffers
    # (~2x the trainable params' size) stay resident on GPU until this object
    # is actually dereferenced; empty_cache() alone only releases memory
    # Python has already garbage-collected, not memory a live object still holds.
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    print(f"Partial fine-tuning: freeze blocks 0-{FREEZE_UNTIL - 1}, "
         f"train blocks {FREEZE_UNTIL}-23 + head, {EPOCHS} epochs, LR={LR}")
    for fold in range(N_FOLDS):
        print(f"\n{'=' * 60}\nFold {fold}\n{'=' * 60}")
        train_fold(fold)
    print(f"\nAll folds done. Next:\n"
         f"  python panderm/extract_features.py --mode finetune\n"
         f"  python panderm/evaluate.py --mode finetune")


if __name__ == "__main__":
    main()
