"""AlexNet LPIPS and the TecoGAN temporal score on frame sequences."""

from __future__ import annotations

import lpips
import numpy as np
import torch

from part1.metrics import tlpips


class AlexLPIPS:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.model = lpips.LPIPS(net="alex").to(self.device)
        self.model.eval()

    def __call__(self, restored: np.ndarray, ground_truth: np.ndarray) -> float:
        with torch.no_grad():
            distance = self.model(self._tensor(restored), self._tensor(ground_truth))
        return float(distance)

    def consecutive(self, frames: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self(frames[index], frames[index + 1]) for index in range(len(frames) - 1)],
            dtype=np.float64,
        )

    def temporal(self, restored: np.ndarray, ground_truth: np.ndarray) -> float:
        return tlpips(self.consecutive(restored), self.consecutive(ground_truth))

    def _tensor(self, frame: np.ndarray) -> torch.Tensor:
        array = np.transpose(frame.astype(np.float32), (2, 0, 1))
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        return tensor / 127.5 - 1.0
