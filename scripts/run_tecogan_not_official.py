"""Run a non-official PyTorch TecoGAN on REDS4.

The checkpoint is the skycrapers reimplementation trained on Vimeo bicubic
4× data. It is not the released TensorFlow TecoGAN checkpoint. Videos go to
outputs/part2/tecogan_not_official and the label is also written to
docs/results/tecogan_pytorch_not_official.json.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import skimage.measure as skimage_measure
import torch
from skimage.metrics import structural_similarity

if not hasattr(skimage_measure, "compare_ssim"):
    skimage_measure.compare_ssim = structural_similarity
if "IPython" not in sys.modules:
    ipython = types.ModuleType("IPython")
    ipython.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = ipython

ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "third_party" / "TecoGAN-PyTorch" / "codes"
WEIGHT = ROOT / "weights" / "TecoGAN_BI4x_Vimeo_pytorch_NOT_OFFICIAL.pth"
sys.path.insert(0, str(CODES))
sys.path.insert(0, str(ROOT / "src"))

from part1.io import read_sequence, write_video

CLIPS = ("000", "011", "015", "020")


def main() -> None:
    if not (CODES / "models" / "networks" / "tecogan_nets.py").is_file():
        raise SystemExit(f"TecoGAN-PyTorch source is not checked out: {CODES}")
    if not WEIGHT.is_file() or WEIGHT.stat().st_size < 1_000_000:
        raise SystemExit(f"non-official TecoGAN weight is not ready: {WEIGHT}")
    from models.networks.tecogan_nets import FRNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRNet(in_nc=3, out_nc=3, nf=64, nb=10, degradation="BI", scale=4)
    state = torch.load(WEIGHT, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("generator") or state.get("state_dict") or state.get("params") or state
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    output_root = ROOT / "outputs" / "part2" / "tecogan_not_official"
    summary = {
        "released_weights": False,
        "checkpoint": WEIGHT.name,
        "training_degradation": "Vimeo-90K bicubic 4x, skycrapers reimplementation",
        "clips": {},
    }
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        frames = read_sequence(sorted(low_dir.glob("*.png")))
        tensor = torch.from_numpy(np.transpose(frames / 255.0, (0, 3, 1, 2)))
        restored = model.infer_sequence(tensor, device)
        restored = np.asarray(restored, dtype=np.float32)
        if restored.max() <= 1.5:
            restored = restored * 255.0
        write_video(output_root / f"{clip}.mp4", restored)
        summary["clips"][clip] = {"frames": int(len(restored))}
        print(clip, "done", flush=True)
    result = ROOT / "docs" / "results" / "tecogan_pytorch_not_official.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(summary, indent=2) + "\n")
    output_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
