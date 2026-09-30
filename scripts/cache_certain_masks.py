"""Cache face and text masks for the REDS4 hybrid.

The masks are computed on the bicubic-upscaled official low-resolution
frames. STAR output is not required. OpenCV is kept on the CPU so this does
not share a GPU with STAR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.io import read_sequence
from part1.resample import upsample
from part3.detectors import certain_mask

CLIPS = ("000", "011", "015", "020")


def main() -> None:
    output = ROOT / "outputs" / "part3" / "hybrid" / "certain"
    output.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        destination = output / f"{clip}.npy"
        low = read_sequence(sorted((ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip).glob("*.png")))
        guide = [upsample(frame, 4, "bicubic") for frame in low]
        if destination.is_file():
            cached = np.load(destination)
            if cached.shape == (len(guide), guide[0].shape[0], guide[0].shape[1]):
                print(clip, "reused", flush=True)
                continue
        masks = np.stack([certain_mask(frame) for frame in guide])
        np.save(destination, masks)
        print(clip, int(masks.sum()), flush=True)


if __name__ == "__main__":
    main()
