"""Run Part 1 on the frozen 30-clip Vimeo subset.

PSNR and SSIM are on Y with no border crop. The script stops if the official
low-resolution frames are not on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from part1.edsr import load_edsr
from part1.fid import InceptionPool
from part1.io import read_sequence, write_video
from part1.perceptual import AlexLPIPS
from part1.restore import restore_part1
from part1.score import score_vimeo
from part1.vimeo import pair_clip

DATA = ROOT / "data" / "vimeo"
SUBSET = ROOT / "docs" / "manifests" / "vimeo-subset.txt"


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit("official Vimeo frames are not on disk; the MIT archive returned HTTP 403")
    keys = [line.strip() for line in SUBSET.read_text().splitlines() if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / "weights" / "edsr_baseline_x4-6b446fab.pt"
    edsr = load_edsr(checkpoint, device) if checkpoint.is_file() and checkpoint.stat().st_size == 6081831 else None
    scorer = None
    inception = None
    summary = {}
    output_root = ROOT / "outputs" / "part1_vimeo"
    for key in keys:
        low_dir, high_dir = pair_clip(DATA, key)
        restored = restore_part1(read_sequence(sorted(low_dir.glob("*.png"))), edsr, device)
        name = key.replace("/", "_")
        clip_scores = {}
        for method, frames in restored.items():
            write_video(output_root / name / f"{method}.mp4", frames)
            if high_dir is not None:
                if scorer is None or inception is None:
                    scorer = AlexLPIPS(device)
                    inception = InceptionPool(device)
                clip_scores[method] = score_vimeo(frames, read_sequence(sorted(high_dir.glob("*.png"))), scorer, inception)
            else:
                clip_scores[method] = {"frames": int(len(frames)), "metrics": "waiting for ground truth"}
        summary[key] = clip_scores
        print(key, "videos written", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    from vimeo_means import main as write_means

    write_means()


if __name__ == "__main__":
    main()
