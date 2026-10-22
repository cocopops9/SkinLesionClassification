"""
Compute reference values with OpenCV and the project's ExplainabilityEngine.

Writes raw RGB dumps and reference.json into an output directory, which
compare.js then checks against docs/lab/heuristics.js.

    python docs/lab/verify/make_reference.py /tmp/slh_verify
    node docs/lab/verify/compare.js /tmp/slh_verify
"""

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from explainability import ExplainabilityEngine  # noqa: E402


def load_images():
    images = {}
    for path in sorted(glob.glob(str(ROOT / 'media' / '*.jpg'))):
        images[os.path.basename(path)] = np.array(Image.open(path).convert('RGB'))

    rng = np.random.default_rng(0)
    for k in range(6):
        h, w = int(rng.integers(101, 400)), int(rng.integers(101, 400))
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.arctan2(yy - h / 2, xx - w / 2)
        radius = np.hypot(yy - h / 2, xx - w / 2)
        boundary = min(h, w) * 0.3 * (1 + 0.25 * np.sin(3 * theta + k) + 0.1 * np.sin(7 * theta))
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (220, 175, 150)
        img[radius < boundary] = (90, 55, 40)
        img = np.clip(img.astype(int) + rng.integers(-25, 25, img.shape), 0, 255).astype(np.uint8)
        images[f'synthetic_{k}'] = img
    return images


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = {}
    for name, img in load_images().items():
        img = np.ascontiguousarray(img)
        img.tofile(out / f'{name}.bin')

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        metrics = ExplainabilityEngine.analyze_color_distribution(img)
        abcde = ExplainabilityEngine.compute_abcde_criteria(img, metrics)
        small = cv2.resize(img, (100, 100)) if img.shape[0] > 100 else img

        reference[name] = {
            'w': img.shape[1], 'h': img.shape[0],
            'gray_sum': int(gray.astype(np.int64).sum()),
            'otsu': float(threshold),
            'area': float(cv2.contourArea(contour)),
            'perimeter': float(cv2.arcLength(contour, True)),
            'rect': list(cv2.boundingRect(contour)),
            'A': float(abcde['asymmetry']['score']),
            'B': float(abcde['border']['score']),
            'C': float(abcde['color']['variance']),
            'D': float(abcde['diameter']['value_mm']),
            'dark': float(metrics['darkness_ratio']),
            'unique': int(len(np.unique(small.reshape(-1, 3), axis=0)))
        }
    (out / 'reference.json').write_text(json.dumps(reference, indent=1))
    print(f'{len(reference)} reference images written to {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'slh_verify')
