"""Score restored REDS4 frames once ground truth has been extracted.

Run this after train_sharp extraction. Methods that have not written frames
yet are skipped. Each method that already scored itself is scored again from
the saved frames, so a late ground-truth download still fills the tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.fid import InceptionPool
from part1.io import read_sequence
from part1.perceptual import AlexLPIPS
from part1.score import score_reds

CLIPS = ("000", "011", "015", "020")
METHODS = {
    "edvr": ROOT / "outputs" / "part2" / "edvr",
    "tecogan_not_official": ROOT / "outputs" / "part2" / "tecogan_not_official",
    "basicvsrpp": ROOT / "outputs" / "part3" / "basicvsrpp",
    "star": ROOT / "outputs" / "part3" / "star",
    "hybrid": ROOT / "outputs" / "part3" / "hybrid",
}


def main() -> None:
    truth_root = ROOT / "data" / "reds" / "train" / "train_sharp"
    if not any(truth_root.glob("*/*.png")):
        raise SystemExit("REDS4 ground truth is not extracted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = AlexLPIPS(device)
    inception = InceptionPool(device)
    collected = {}
    for name, directory in METHODS.items():
        frames_root = directory / "frames"
        if not frames_root.is_dir():
            continue
        previous = {}
        metrics_path = directory / "metrics.json"
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text())
        clip_scores = previous.get("clips", previous)
        for clip in CLIPS:
            clip_dir = frames_root / clip
            paths = sorted(clip_dir.glob("*.png"))
            truth_paths = sorted((truth_root / clip).glob("*.png"))
            if not paths or not truth_paths:
                continue
            clip_scores[clip] = score_reds(read_sequence(paths), read_sequence(truth_paths), scorer, inception)
        if "clips" in previous:
            previous["clips"] = clip_scores
            payload = previous
        else:
            payload = clip_scores
        directory.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        collected[name] = payload
        print(name, "scored", flush=True)
    report = ROOT / "docs" / "results" / "reds4_metrics.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(collected, indent=2) + "\n")
    from reds_means import main as write_means

    write_means()


if __name__ == "__main__":
    main()
