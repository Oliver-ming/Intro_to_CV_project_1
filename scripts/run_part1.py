"""Run Part 1 on extracted REDS4 clips.

Looks under data/reds for clip folders 000, 011, 015, and 020. A folder whose
path contains "bicubic" is the official low-resolution input. The other is
ground truth. Writes restored videos and a metric json under outputs/part1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.edsr import load_edsr
from part1.io import read_sequence, write_video
from part1.metrics import psnr, ssim
from part1.perceptual import AlexLPIPS
from part1.restore import restore_part1

CLIPS = ("000", "011", "015", "020")


def main() -> None:
    data_root = ROOT / "data" / "reds"
    if not data_root.exists():
        raise SystemExit(f"REDS4 frames are not extracted yet: {data_root}")
    pairs = {clip: _pair(data_root, clip) for clip in CLIPS}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / "weights" / "edsr_baseline_x4-6b446fab.pt"
    edsr = load_edsr(checkpoint, device) if checkpoint.is_file() and checkpoint.stat().st_size == 6081831 else None
    scorer = AlexLPIPS(device)
    summary = {}
    output_root = ROOT / "outputs" / "part1"
    for clip, (low_paths, truth_paths) in pairs.items():
        low = read_sequence(low_paths)
        truth = read_sequence(truth_paths)
        restored = restore_part1(low, edsr, device)
        clip_scores = {}
        for name, frames in restored.items():
            write_video(output_root / clip / f"{name}.mp4", frames)
            clip_scores[name] = _score(frames, truth, scorer)
        summary[clip] = clip_scores
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _pair(root: Path, clip: str) -> tuple[list[Path], list[Path]]:
    folders = [path for path in root.rglob(clip) if path.is_dir()]
    low = [path for path in folders if "bicubic" in path.as_posix()]
    truth = [path for path in folders if "bicubic" not in path.as_posix()]
    if len(low) != 1 or len(truth) != 1:
        raise SystemExit(f"expected one low-resolution and one ground-truth folder for {clip}, found {folders}")
    return _frames(low[0]), _frames(truth[0])


def _frames(folder: Path) -> list[Path]:
    frames = sorted(folder.glob("*.png"))
    if not frames:
        raise SystemExit(f"no png frames in {folder}")
    return frames


def _score(restored: np.ndarray, truth: np.ndarray, scorer: AlexLPIPS) -> dict[str, float]:
    count = min(len(restored), len(truth))
    rgb = [psnr(restored[index], truth[index], y_channel=False) for index in range(count)]
    structure = [ssim(restored[index], truth[index], y_channel=False) for index in range(count)]
    return {
        "psnr_rgb": float(np.mean(rgb)),
        "ssim_rgb": float(np.mean(structure)),
        "lpips": float(np.mean([scorer(restored[index], truth[index]) for index in range(count)])),
        "tlpips": scorer.temporal(restored[:count], truth[:count]),
        "frames": count,
    }


if __name__ == "__main__":
    main()
