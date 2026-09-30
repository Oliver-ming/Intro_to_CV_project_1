"""Run official EDVR-L on REDS4.

Uses BasicSR at the pinned commit. The REDS checkpoint is the bicubic
EDVR-L model: five frames, reflection_circle padding, 4× output. Inputs are
RGB in [0, 1], matching BasicSR. Ground-truth metrics are written only when
train_sharp has been extracted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "BasicSR"))
sys.path.insert(0, str(ROOT / "src"))

from basicsr.archs.edvr_arch import EDVR
from basicsr.data.data_util import generate_frame_indices

from part1.fid import InceptionPool
from part1.io import read_sequence, write_sequence, write_video
from part1.perceptual import AlexLPIPS
from part1.score import score_reds

CLIPS = ("000", "011", "015", "020")
WEIGHT = ROOT / "weights" / "EDVR_L_x4_SR_REDS_official-9f5f5039.pth"


def main() -> None:
    if not WEIGHT.is_file() or WEIGHT.stat().st_size < 80_000_000:
        raise SystemExit(f"EDVR REDS weight is not ready: {WEIGHT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(device)
    output_root = ROOT / "outputs" / "part2" / "edvr"
    summary = {}
    scorer = None
    inception = None
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        if not low_dir.is_dir():
            raise SystemExit(f"missing low-resolution clip {low_dir}")
        frames = read_sequence(sorted(low_dir.glob("*.png")))
        restored = _restore(model, frames, device)
        write_sequence(output_root / "frames" / clip, restored)
        write_video(output_root / f"{clip}.mp4", restored)
        truth_dir = ROOT / "data" / "reds" / "train" / "train_sharp" / clip
        if truth_dir.is_dir() and any(truth_dir.glob("*.png")):
            if scorer is None or inception is None:
                scorer = AlexLPIPS(device)
                inception = InceptionPool(device)
            truth = read_sequence(sorted(truth_dir.glob("*.png")))
            summary[clip] = score_reds(restored, truth, scorer, inception)
        else:
            summary[clip] = {"frames": int(len(restored)), "metrics": "waiting for ground truth"}
        print(clip, "done", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _load(device: torch.device) -> EDVR:
    model = EDVR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=128,
        num_frame=5,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=40,
        with_predeblur=False,
        with_tsa=True,
        hr_in=False,
    )
    state = torch.load(WEIGHT, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("params_ema") or state.get("params") or state
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _restore(model: EDVR, frames: np.ndarray, device: torch.device) -> np.ndarray:
    count = len(frames)
    outputs = []
    for index in range(count):
        chosen = generate_frame_indices(index, count, 5, padding="reflection_circle")
        window = torch.from_numpy(np.stack([np.transpose(frames[i] / 255.0, (2, 0, 1)) for i in chosen]))
        window = window.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(window)
        outputs.append(prediction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
    return np.stack(outputs)


if __name__ == "__main__":
    main()
