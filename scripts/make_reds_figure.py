"""Write a REDS4 visual comparison for the report.

Frame 50 of clip 000. The top row is the full frame after 4× restoration.
The bottom row is the 160×160 patch with the strongest bicubic detail.
Ground truth is not required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.io import read_frame
from part1.resample import upsample

CLIP = "000"
INDEX = 50
CROP = 160


def main() -> None:
    low = read_frame(_frame(ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / CLIP))
    panels = {
        "Bicubic": upsample(low, 4, "bicubic"),
        "EDVR-L": read_frame(_frame(ROOT / "outputs" / "part2" / "edvr" / "frames" / CLIP)),
        "BasicVSR++": read_frame(_frame(ROOT / "outputs" / "part3" / "basicvsrpp" / "frames" / CLIP)),
        "STAR": read_frame(_frame(ROOT / "outputs" / "part3" / "star" / "frames" / CLIP)),
    }
    y0, x0 = _detail_origin(panels["Bicubic"])
    full = [_labeled(_fit(image, 426), name) for name, image in panels.items()]
    crops = [_labeled(image[y0 : y0 + CROP, x0 : x0 + CROP], name) for name, image in panels.items()]
    gap = np.full((full[0].shape[0], 8, 3), 255, np.uint8)
    top = _join(full, gap)
    crop_gap = np.full((crops[0].shape[0], 8, 3), 255, np.uint8)
    bottom = _join(crops, crop_gap)
    pad = (top.shape[1] - bottom.shape[1]) // 2
    bottom = cv2.copyMakeBorder(bottom, 0, 0, pad, top.shape[1] - bottom.shape[1] - pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    figure = np.concatenate([top, np.full((12, top.shape[1], 3), 255, np.uint8), bottom], axis=0)
    destination = ROOT / "paper" / "figures" / "reds4_000_frame050.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR)):
        raise OSError(destination)
    print(destination, figure.shape, "crop", y0, x0)


def _frame(directory: Path) -> Path:
    frames = sorted(directory.glob("*.png"))
    if len(frames) <= INDEX:
        raise SystemExit(f"{directory} does not have frame {INDEX}")
    return frames[INDEX]


def _detail_origin(image: np.ndarray) -> tuple[int, int]:
    gray = cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    response = cv2.Laplacian(gray, cv2.CV_32F)
    best = (-1.0, 0, 0)
    for y in range(0, gray.shape[0] - CROP + 1, 40):
        for x in range(0, gray.shape[1] - CROP + 1, 40):
            score = float(response[y : y + CROP, x : x + CROP].var())
            if score > best[0]:
                best = (score, y, x)
    return best[1], best[2]


def _fit(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(np.clip(np.rint(image), 0, 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)


def _labeled(image: np.ndarray, name: str) -> np.ndarray:
    canvas = np.clip(np.rint(image), 0, 255).astype(np.uint8).copy()
    cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _join(panels: list[np.ndarray], gap: np.ndarray) -> np.ndarray:
    rows = [panels[0]]
    for panel in panels[1:]:
        rows.extend([gap, panel])
    return np.concatenate(rows, axis=1)


if __name__ == "__main__":
    main()
