"""One Fréchet distance on all 210 Vimeo-subset frames.

A method is written only when every frozen clip has the same number of
restored frames as its ground truth. Clip distances are not averaged.
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
from part1.vimeo import pair_clip

DATA = ROOT / "data" / "vimeo"
SUBSET = ROOT / "docs" / "manifests" / "vimeo-subset.txt"
RESULT = ROOT / "docs" / "results" / "vimeo_pooled_fid.json"
METHODS = {
    "Vimeo bicubic": ("part1_vimeo", "bicubic"),
    "Vimeo Lanczos": ("part1_vimeo", "lanczos"),
    "Vimeo EDSR": ("part1_vimeo", "edsr"),
    "Vimeo temporal average": ("part1_vimeo", "temporal_average"),
    "Vimeo unsharp": ("part1_vimeo", "temporal_unsharp"),
    "Vimeo EDVR-L": ("edvr_vimeo", None),
}


def frame_dir(label_root: str, key: str, method: str | None) -> Path:
    name = key.replace("/", "_")
    if label_root == "edvr_vimeo":
        return ROOT / "outputs" / "part2" / "edvr_vimeo" / "frames" / name
    directory = ROOT / "outputs" / "part1_vimeo" / "frames" / name
    return directory / method if method else directory


def complete_frames(directory: Path, count: int) -> np.ndarray | None:
    if not directory.is_dir():
        return None
    paths = sorted(directory.glob("*.png"))
    if len(paths) != count:
        return None
    return read_sequence(paths)


def main() -> None:
    keys = [line.strip() for line in SUBSET.read_text().splitlines() if line.strip()]
    if not DATA.is_dir():
        _finish()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = None
    truth_features = None
    pooled = json.loads(RESULT.read_text()) if RESULT.is_file() else {}
    for label, (label_root, method) in METHODS.items():
        pieces = []
        truths = []
        for key in keys:
            try:
                _low, high = pair_clip(DATA, key)
            except FileNotFoundError:
                pieces = []
                break
            if high is None:
                pieces = []
                break
            truth = read_sequence(sorted(high.glob("*.png")))
            restored = complete_frames(frame_dir(label_root, key, method), len(truth))
            if restored is None:
                pieces = []
                break
            pieces.append(quantize(restored))
            truths.append(quantize(truth))
        if not pieces:
            print(label, "incomplete", flush=True)
            continue
        if inception is None:
            inception = InceptionPool(device)
            truth_features = inception(np.concatenate(truths))
        frames = np.concatenate(pieces)
        pooled[label] = pool_distance(inception(frames), truth_features)
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(pooled, indent=2) + "\n")
        print(label, pooled[label], len(frames), flush=True)
    _finish()


def _finish() -> None:
    from vimeo_means import main as write_means

    write_means()


if __name__ == "__main__":
    main()
