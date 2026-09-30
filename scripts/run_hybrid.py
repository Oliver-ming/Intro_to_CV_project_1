"""Blend BasicVSR++ and official STAR on REDS4.

Certain pixels are face and text boxes on the bicubic-upscaled low-resolution
frame. Those pixels come from BasicVSR++. Every other pixel uses the
per-frame disagreement map. A missing detector leaves that prior empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.fid import InceptionPool
from part1.io import read_sequence, write_sequence, write_video
from part1.perceptual import AlexLPIPS
from part1.resample import upsample
from part1.score import score_reds
from part3.detectors import EAST, YUNET, certain_mask
from part3.hybrid import blend

CLIPS = ("000", "011", "015", "020")


def _certain(output_root: Path, clip: str, guide: np.ndarray) -> np.ndarray:
    path = output_root / "certain" / f"{clip}.npy"
    if path.is_file():
        cached = np.load(path)
        if cached.shape == guide.shape[:3] and cached.dtype == np.bool_:
            return cached
    masks = np.stack([certain_mask(frame) for frame in guide])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, masks)
    return masks


def main() -> None:
    fidelity_root = ROOT / "outputs" / "part3" / "basicvsrpp" / "frames"
    generative_root = ROOT / "outputs" / "part3" / "star" / "frames"
    if not fidelity_root.is_dir() or not generative_root.is_dir():
        raise SystemExit("hybrid needs BasicVSR++ frames and official STAR frames")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = ROOT / "outputs" / "part3" / "hybrid"
    summary = {
        "fidelity": "BasicVSR++ REDS4 bicubic",
        "generative": "official STAR light_deg.pt",
        "yunet": YUNET.is_file(),
        "east": EAST.is_file(),
        "clips": {},
    }
    scorer = None
    inception = None
    for clip in CLIPS:
        fidelity = read_sequence(sorted((fidelity_root / clip).glob("*.png")))
        generative = read_sequence(sorted((generative_root / clip).glob("*.png")))
        low = read_sequence(sorted((ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip).glob("*.png")))
        guide = np.stack([upsample(frame, 4, "bicubic") for frame in low])
        masks = _certain(output_root, clip, guide)
        count = min(len(fidelity), len(generative), len(guide), len(masks))
        if fidelity.shape[1:] != generative.shape[1:] or fidelity.shape[1:] != guide.shape[1:]:
            summary["clips"][clip] = {
                "status": "shape mismatch",
                "fidelity": list(fidelity.shape),
                "generative": list(generative.shape),
                "bicubic": list(guide.shape),
            }
            print(clip, "shape mismatch", flush=True)
            continue
        mixed = np.stack([blend(fidelity[index], generative[index], masks[index]) for index in range(count)])
        write_sequence(output_root / "frames" / clip, mixed)
        write_video(output_root / f"{clip}.mp4", mixed)
        truth_dir = ROOT / "data" / "reds" / "train" / "train_sharp" / clip
        if truth_dir.is_dir() and any(truth_dir.glob("*.png")):
            if scorer is None or inception is None:
                scorer = AlexLPIPS(device)
                inception = InceptionPool(device)
            truth = read_sequence(sorted(truth_dir.glob("*.png")))
            summary["clips"][clip] = score_reds(mixed, truth, scorer, inception)
        else:
            summary["clips"][clip] = {"frames": int(count), "metrics": "waiting for ground truth"}
        print(clip, "done", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.path.insert(0, str(ROOT / "scripts"))
    from reds_means import main as write_means

    write_means()


if __name__ == "__main__":
    main()
