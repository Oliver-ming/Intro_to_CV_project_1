"""One REDS4 Fréchet distance on all 400 frames.

Per-clip distances are not averaged. A method is written only when clips
000, 011, 015, and 020 each have the same number of restored frames as the
ground truth. Part 1 is recomputed from the official low-resolution frames
when --part1 is passed, because those restorations are stored as compressed
video rather than the quantized frames that were scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from part1.fid import InceptionPool, pool_distance
from part1.io import quantize, read_sequence

CLIPS = ("000", "011", "015", "020")
PNG_METHODS = {
    "EDVR-L": ROOT / "outputs" / "part2" / "edvr" / "frames",
    "BasicVSR++": ROOT / "outputs" / "part3" / "basicvsrpp" / "frames",
    "TecoGAN PyTorch, not released": ROOT / "outputs" / "part2" / "tecogan_not_official" / "frames",
    "STAR light degradation": ROOT / "outputs" / "part3" / "star" / "frames",
    "Hybrid": ROOT / "outputs" / "part3" / "hybrid" / "frames",
    "Texture hybrid": ROOT / "outputs" / "part3" / "hybrid_texture" / "frames",
}
RESULT = ROOT / "docs" / "results" / "reds4_pooled_fid.json"
TRUTH = ROOT / "data" / "reds" / "train" / "train_sharp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    if not any(TRUTH.glob("*/*.png")):
        raise SystemExit("REDS4 ground truth is not extracted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = InceptionPool(device)
    truth_frames = quantize(_truth())
    truth_features = inception(truth_frames)
    pooled = json.loads(RESULT.read_text()) if RESULT.is_file() else {}
    for label, directory in PNG_METHODS.items():
        if args.only and label != args.only:
            continue
        frames = _complete(directory)
        if frames is None:
            print(label, "incomplete", flush=True)
            continue
        pooled[label] = pool_distance(inception(quantize(frames)), truth_features)
        _write(pooled)
        print(label, pooled[label], len(frames), flush=True)
    if args.part1:
        _part1(inception, truth_frames, truth_features, pooled)
    from reds_means import main as write_means

    write_means()


def _truth() -> np.ndarray:
    sequences = [_clip(TRUTH / clip) for clip in CLIPS]
    if any(item is None for item in sequences):
        raise SystemExit("REDS4 ground truth is missing a clip")
    return np.concatenate(sequences)


def _complete(directory: Path) -> np.ndarray | None:
    sequences = [_clip(directory / clip) for clip in CLIPS]
    if any(item is None for item in sequences):
        return None
    return np.concatenate(sequences)


def _clip(directory: Path) -> np.ndarray | None:
    if not directory.is_dir():
        return None
    paths = sorted(directory.glob("*.png"))
    if len(paths) != 100:
        return None
    return read_sequence(paths)


def _part1(inception: InceptionPool, truth: np.ndarray, truth_features: np.ndarray, pooled: dict) -> None:
    from part1.edsr import load_edsr
    from part1.restore import restore_part1

    device = inception.device
    checkpoint = ROOT / "weights" / "edsr_baseline_x4-6b446fab.pt"
    edsr = load_edsr(checkpoint, device) if checkpoint.is_file() and checkpoint.stat().st_size == 6081831 else None
    restored = {name: [] for name in ("bicubic", "lanczos", "edsr", "temporal_average", "temporal_unsharp")}
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        low = read_sequence(sorted(low_dir.glob("*.png")))
        outputs = restore_part1(low, edsr, device)
        for name, frames in outputs.items():
            restored[name].append(quantize(frames))
    labels = {
        "bicubic": "Bicubic",
        "lanczos": "Lanczos",
        "edsr": "EDSR",
        "temporal_average": "Temporal average",
        "temporal_unsharp": "Unsharp",
    }
    for name, pieces in restored.items():
        if name not in labels or not pieces:
            continue
        frames = np.concatenate(pieces)
        if len(frames) != len(truth):
            continue
        pooled[labels[name]] = pool_distance(inception(frames), truth_features)
        _write(pooled)
        print(labels[name], pooled[labels[name]], len(frames), flush=True)


def _write(pooled: dict) -> None:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(pooled, indent=2) + "\n")


if __name__ == "__main__":
    main()
