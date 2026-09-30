"""Run BasicVSR++ on REDS4 when a bicubic checkpoint is on disk.

The architecture is BasicSR's BasicVSRPlusPlus: 64 mid channels, 7 residual
blocks, low-resolution input. A whole clip is one recurrent sequence. SpyNet
is loaded from its own file when that file exists; otherwise the checkpoint
must already contain those weights.
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

from part1.fid import InceptionPool
from part1.io import read_sequence, write_sequence, write_video
from part1.perceptual import AlexLPIPS
from part1.score import score_reds
from part3.checkpoints import generator_weights

CLIPS = ("000", "011", "015", "020")
WEIGHTS = (
    ROOT / "weights" / "BasicVSRPP_REDS4.pth",
    ROOT / "weights" / "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth",
)
SPYNET = ROOT / "weights" / "spynet_sintel_final-3d2a1287.pth"


def main() -> None:
    weight = _weight()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(weight, device)
    output_root = ROOT / "outputs" / "part3" / "basicvsrpp"
    summary = {"checkpoint": weight.name, "training_degradation": "REDS bicubic 4x"}
    scorer = None
    inception = None
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
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


def _weight() -> Path:
    for path in WEIGHTS:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    raise SystemExit("BasicVSR++ REDS4 bicubic checkpoint is not ready")


def _load(weight: Path, device: torch.device) -> BasicVSRPlusPlus:
    model = BasicVSRPlusPlus(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_path=str(SPYNET) if SPYNET.is_file() else None,
        cpu_cache_length=50,
    )
    state = generator_weights(torch.load(weight, map_location="cpu", weights_only=False))
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {name for name in missing if name.endswith(".mean") or name.endswith(".std")}
    if unexpected or set(missing) - allowed:
        raise SystemExit(f"BasicVSR++ checkpoint does not match the architecture: missing={missing} unexpected={unexpected}")
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
