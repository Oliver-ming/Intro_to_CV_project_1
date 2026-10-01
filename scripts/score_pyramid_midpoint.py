"""Score the course Pyramid Flow midpoint sample on REDS4.

The frames come from the released VAE and scheduler, starting at the middle
index. This is not the released generate() path. FID is one pooled distance
on all 400 frames.
"""

from __future__ import annotations

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
from part1.perceptual import AlexLPIPS
from part1.score import score_reds

CLIPS = ("000", "011", "015", "020")
FRAMES = ROOT / "outputs" / "part3" / "pyramid_midpoint"
TRUTH = ROOT / "data" / "reds" / "train" / "train_sharp"
POOLED = ROOT / "docs" / "results" / "reds4_pooled_fid.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = AlexLPIPS(device)
    inception = InceptionPool(device)
    summary_path = FRAMES / "metrics.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    clips = {}
    restored_frames = []
    truth_frames = []
    for clip in CLIPS:
        restored = read_sequence(sorted((FRAMES / "frames" / clip).glob("*.png")))
        truth = read_sequence(sorted((TRUTH / clip).glob("*.png")))
        if len(restored) != 100 or restored.shape[1:] != truth.shape[1:]:
            raise SystemExit(f"{clip} frames do not match REDS4 ground truth: {restored.shape} vs {truth.shape}")
        clips[clip] = score_reds(restored, truth, scorer, inception)
        restored_frames.append(quantize(restored))
        truth_frames.append(quantize(truth))
        print(clip, clips[clip]["psnr_rgb"], flush=True)
    distance = pool_distance(inception(np.concatenate(restored_frames)), inception(np.concatenate(truth_frames)))
    summary["clips"] = clips
    summary["start_index"] = 10
    summary["num_steps"] = 20
    summary["pooled_fid"] = distance
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    pooled = json.loads(POOLED.read_text()) if POOLED.is_file() else {}
    pooled["Pyramid Flow"] = distance
    POOLED.write_text(json.dumps(pooled, indent=2) + "\n")
    (ROOT / "docs" / "results" / "pyramid_midpoint.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("pooled_fid", distance, flush=True)
    from reds_means import main as write_means

    write_means()


if __name__ == "__main__":
    main()
