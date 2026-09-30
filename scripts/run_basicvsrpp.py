"""Run BasicVSR++ on REDS4 when its bicubic checkpoint and SpyNet weights exist.

The architecture is BasicSR's BasicVSRPlusPlus: 64 mid channels, 7 residual
blocks, low-resolution input. A whole clip is one recurrent sequence. This
does not start until both weight files are present.
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

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

from part1.io import read_sequence, write_video

CLIPS = ("000", "011", "015", "020")
WEIGHT = ROOT / "weights" / "BasicVSRPP_REDS4.pth"
SPYNET = ROOT / "weights" / "spynet_sintel_final-3d2a1287.pth"


def main() -> None:
    if not WEIGHT.is_file() or not SPYNET.is_file():
        raise SystemExit(f"BasicVSR++ weights are not ready: {WEIGHT} and {SPYNET}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(device)
    output_root = ROOT / "outputs" / "part3" / "basicvsrpp"
    summary = {}
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        frames = read_sequence(sorted(low_dir.glob("*.png")))
        restored = _restore(model, frames, device)
        write_video(output_root / f"{clip}.mp4", restored)
        summary[clip] = {"frames": int(len(restored))}
        print(clip, "done", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2))


def _load(device: torch.device) -> BasicVSRPlusPlus:
    model = BasicVSRPlusPlus(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_path=str(SPYNET),
        cpu_cache_length=50,
    )
    state = torch.load(WEIGHT, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("params_ema") or state.get("params") or state.get("state_dict") or state
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _restore(model: BasicVSRPlusPlus, frames: np.ndarray, device: torch.device) -> np.ndarray:
    sequence = torch.from_numpy(np.transpose(frames / 255.0, (0, 3, 1, 2))).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(sequence)
    return prediction.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0


if __name__ == "__main__":
    main()
