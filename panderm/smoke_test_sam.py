"""
One-image sanity check for SAMSegmenter, before trusting it on real data.
Same discipline as the hair-removal calibration: look at an overlay, don't
just trust a metric number.

Usage:
    python panderm/smoke_test_sam.py path\\to\\image.jpg
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from segment_lesions import build_segmenter

img_path = sys.argv[1]
out_path = Path(img_path).with_name(Path(img_path).stem + "_sam_overlay.jpg")

img = np.array(Image.open(img_path).convert("RGB"))
print("Loading SAM2 (downloads the checkpoint on first run)...")
segmenter = build_segmenter("sam")

print("Segmenting...")
mask = segmenter.segment(img)

overlay = img.copy()
overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
Image.fromarray(overlay).save(out_path)

print(f"mask covers {100 * mask.mean():.1f}% of the frame")
print(f"saved overlay: {out_path}")
print("Open it and check: does the red region actually match the lesion boundary?")
