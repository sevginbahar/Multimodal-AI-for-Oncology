"""
Calibration of the patient-level predictions saved by evaluate.py
=================================================================
A classifier is calibrated if its probabilities mean what they say: among patients
given P(MIA) = 0.7, about 70% should truly be MIA. Accuracy and AUC do not measure this.

Reads results/<mode>/<backend>/patient_predictions_<mode>.csv (out-of-fold test predictions:
true class + the three probabilities) for the cohort and backend set in config.py, and reports:

    ECE (top-label)   gap between confidence (highest probability) and accuracy, averaged over
                      10 equal-width confidence bins weighted by their size; 0 = perfect
    ECE (class-wise)  the same per class (one-vs-rest: P(class) vs "is that class"), averaged
    Brier score       mean squared error between the 3 probabilities and the one-hot truth (lower = better)
    log loss          mean -log P(true class) (lower = better)
    mean confidence vs accuracy   > 0: over-confident, < 0: under-confident

Then temperature scaling, the standard post-hoc fix: p_k = softmax(log p_k / T). T > 1 softens
over-confident probabilities, T < 1 sharpens them; the predicted class never changes, so accuracy
and AUC are untouched. T is fitted CROSS-FOLD: each fold's T is fitted (minimum log loss) on the
other folds' predictions only, so no patient's own label calibrates their probabilities.

Note: class_weight="balanced" deliberately shifts probabilities toward the rarer classes, so the raw
probabilities are not expected to match the cohort's class frequencies.

Outputs (results/<mode>/<backend>/):
    calibration_<mode>.csv    metrics before and after temperature scaling (+ the fitted T per fold)
    calibration_<mode>.png    reliability diagrams (top-label and per class, before / after) + confidence histogram

Usage:
    python panderm/calibration.py --mode frozen        # same --mode as evaluate.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, SEGMENTATION_BACKEND, CLASS_NAMES, DISPLAY_NAMES

N_BINS = 10
EPS = 1e-12


def temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, EPS, 1)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def fit_temperature(p, y) -> float:
    return float(minimize_scalar(lambda T: log_loss(temperature(p, T), y), bounds=(0.05, 20), method="bounded").x)


def bins(conf, correct):
    """Per equal-width bin: mean confidence, accuracy, count."""
    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, N_BINS - 1)
    rows = [(conf[idx == b].mean(), correct[idx == b].mean(), int((idx == b).sum()))
            for b in range(N_BINS) if (idx == b).any()]
    return np.array(rows) if rows else np.zeros((0, 3))


def ece(conf, correct) -> float:
    b = bins(conf, correct)
    return float(np.sum(b[:, 2] / len(conf) * np.abs(b[:, 1] - b[:, 0]))) if len(b) else np.nan


def metrics(p, y) -> dict:
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    onehot = np.eye(p.shape[1])[y]
    return {"ece_top_label": ece(conf, (pred == y).astype(float)),
            "ece_classwise": float(np.mean([ece(p[:, k], (y == k).astype(float)) for k in range(p.shape[1])])),
            "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
            "log_loss": log_loss(p, y),
            "mean_confidence": float(conf.mean()), "accuracy": float((pred == y).mean()),
            "overconfidence": float(conf.mean() - (pred == y).mean())}


def reliability(ax, p, y, title):
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    b = bins(p.max(axis=1), (p.argmax(axis=1) == y).astype(float))
    ax.plot(b[:, 0], b[:, 1], "o-", color="black", label="top label")
    for k, c in enumerate(["#4C72B0", "#DD5555", "#8E44AD"]):
        bk = bins(p[:, k], (y == k).astype(float))
        ax.plot(bk[:, 0], bk[:, 1], "s-", ms=4, color=c, alpha=0.8, label=DISPLAY_NAMES[CLASS_NAMES[k]])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted probability (bin mean)", ylabel="observed frequency", title=title)
    ax.legend(fontsize=7, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finetune", "frozen", "maskguided", "meanall"], default="frozen")
    mode = ap.parse_args().mode
    res = OUTPUT_DIR / "results" / mode / SEGMENTATION_BACKEND
    f = res / f"patient_predictions_{mode}.csv"
    if not f.exists():
        raise SystemExit(f"Not found: {f}  (run evaluate.py --mode {mode} first)")
    d = pd.read_csv(f)
    p = d[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy(float)
    p /= p.sum(axis=1, keepdims=True)
    y, folds = d.true_label.to_numpy(int), d.fold.to_numpy()

    # cross-fold temperature scaling
    p_cal, temps = np.zeros_like(p), {}
    for fo in np.unique(folds):
        te = folds == fo
        temps[int(fo)] = fit_temperature(p[~te], y[~te])
        p_cal[te] = temperature(p[te], temps[int(fo)])

    before, after = metrics(p, y), metrics(p_cal, y)
    out = pd.DataFrame([{"version": "raw", **before}, {"version": "temperature-scaled", **after}])
    out["temperature_per_fold"] = ["", ", ".join(f"{k}:{v:.2f}" for k, v in temps.items())]
    out.to_csv(res / f"calibration_{mode}.csv", index=False)

    print(f"Calibration — {mode} / {SEGMENTATION_BACKEND}  ({len(y)} test patients, out-of-fold)\n")
    print(out.drop(columns="temperature_per_fold").set_index("version").T.round(3).to_string())
    print(f"\nFitted temperature per fold: {temps}  (>1 = raw probabilities were over-confident)")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    reliability(ax[0], p, y, f"Raw  (ECE {before['ece_top_label']:.3f}, Brier {before['brier']:.3f})")
    reliability(ax[1], p_cal, y, f"Temperature-scaled  (ECE {after['ece_top_label']:.3f}, Brier {after['brier']:.3f})")
    ax[2].hist(p.max(axis=1), bins=N_BINS, range=(0, 1), alpha=0.6, label="raw")
    ax[2].hist(p_cal.max(axis=1), bins=N_BINS, range=(0, 1), alpha=0.6, label="temperature-scaled")
    ax[2].set(xlabel="confidence (highest probability)", ylabel="patients", title="How confident the predictions are")
    ax[2].legend()
    fig.suptitle(f"Calibration — {mode} / {SEGMENTATION_BACKEND}  ({OUTPUT_DIR.parent.name})")
    fig.tight_layout()
    fig.savefig(res / f"calibration_{mode}.png", dpi=130)
    print(f"\nSaved: {res / f'calibration_{mode}.csv'}\n       {res / f'calibration_{mode}.png'}")


if __name__ == "__main__":
    main()
