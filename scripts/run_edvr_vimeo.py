"""Run official EDVR-L on the frozen 30-clip Vimeo subset.

The Vimeo-90K checkpoint is the 7-frame bicubic model, not the 5-frame REDS
model. PSNR and SSIM are on Y with no border crop. The script stops if the
official low-resolution frames are not on disk.
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
from part1.score import score_vimeo
from part1.vimeo import pair_clip

WEIGHT = ROOT / "weights" / "EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth"
DATA = ROOT / "data" / "vimeo"
SUBSET = ROOT / "docs" / "manifests" / "vimeo-subset.txt"


def main() -> None:
    if not WEIGHT.is_file() or WEIGHT.stat().st_size < 80_000_000:
        raise SystemExit(f"EDVR Vimeo weight is not ready: {WEIGHT}")
    if not DATA.is_dir():
        raise SystemExit("official Vimeo frames are not on disk; the MIT archive returned HTTP 403")
    keys = [line.strip() for line in SUBSET.read_text().splitlines() if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(device)
    output_root = ROOT / "outputs" / "part2" / "edvr_vimeo"
    summary = {"checkpoint": WEIGHT.name, "training_degradation": "Vimeo-90K bicubic 4x", "clips": {}}
    scorer = None
    inception = None
    for key in keys:
        low_dir, high_dir = pair_clip(DATA, key)
        frames = read_sequence(sorted(low_dir.glob("*.png")))
        restored = _restore(model, frames, device)
        name = key.replace("/", "_")
        write_sequence(output_root / "frames" / name, restored)
        write_video(output_root / f"{name}.mp4", restored)
        if high_dir is not None:
            if scorer is None or inception is None:
                scorer = AlexLPIPS(device)
                inception = InceptionPool(device)
            truth = read_sequence(sorted(high_dir.glob("*.png")))
            summary["clips"][key] = score_vimeo(restored, truth, scorer, inception)
        else:
            summary["clips"][key] = {"frames": int(len(restored)), "metrics": "waiting for ground truth"}
        print(key, "done", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


def _load(device: torch.device) -> EDVR:
    model = EDVR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=128,
        num_frame=7,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=40,
        with_predeblur=False,
        with_tsa=True,
        hr_in=False,
    )
    state = torch.load(WEIGHT, map_location="cpu", weights_only=False)
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
        chosen = generate_frame_indices(index, count, 7, padding="reflection_circle")
        window = torch.from_numpy(np.stack([np.transpose(frames[i] / 255.0, (2, 0, 1)) for i in chosen]))
        window = window.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(window)
        outputs.append(prediction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
    return np.stack(outputs)


if __name__ == "__main__":
    main()
