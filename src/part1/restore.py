"""Part 1 restorations from an official low-resolution sequence."""

from __future__ import annotations

import numpy as np
import torch

from part1.edsr import EDSRBaseline
from part1.resample import upsample
from part1.temporal import temporal_average, unsharp


def restore_part1(low_resolution: np.ndarray, edsr: EDSRBaseline | None = None, device: torch.device | None = None) -> dict[str, np.ndarray]:
    bicubic = _upsample_sequence(low_resolution, "bicubic")
    restored = {
        "bicubic": bicubic,
        "lanczos": _upsample_sequence(low_resolution, "lanczos"),
        "temporal_average": temporal_average(bicubic),
    }
    restored["temporal_unsharp"] = unsharp(restored["temporal_average"])
    if edsr is not None:
        restored["edsr"] = _edsr_sequence(low_resolution, edsr, device or torch.device("cpu"))
    return restored


def _upsample_sequence(frames: np.ndarray, method: str) -> np.ndarray:
    return np.stack([upsample(frame, 4, method) for frame in frames])


def _edsr_sequence(frames: np.ndarray, model: EDSRBaseline, device: torch.device) -> np.ndarray:
    outputs = []
    model.eval()
    for frame in frames:
        tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(tensor)
        outputs.append(prediction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
    return np.stack(outputs)
