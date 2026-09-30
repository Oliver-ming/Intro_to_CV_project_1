"""Course temporal baseline.

The five-frame weights [1, 4, 6, 4, 1] / 16 are a binomial smoother, not a
kernel from EDSR, EDVR, BasicVSR, or TecoGAN. Missing neighbors at a clip
edge are dropped and the remaining weights are renormalized. Unsharp masking
is an extra variant of that average: y = x + (x - GaussianBlur(x, sigma=1)).
"""

from __future__ import annotations

import cv2
import numpy as np

_OFFSETS = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
_WEIGHTS = np.array([1, 4, 6, 4, 1], dtype=np.float32)


def temporal_average(frames: np.ndarray) -> np.ndarray:
    if frames.ndim not in (3, 4):
        raise ValueError("frames must be T,H,W or T,H,W,C")
    count = frames.shape[0]
    output = np.empty_like(frames, dtype=np.float32)
    values = frames.astype(np.float32, copy=False)
    for index in range(count):
        neighbors = index + _OFFSETS
        valid = (neighbors >= 0) & (neighbors < count)
        weights = _WEIGHTS[valid]
        weights = weights / weights.sum()
        output[index] = np.tensordot(values[neighbors[valid]], weights, axes=(0, 0))
    return output


def unsharp(frames: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    sharpened = np.empty_like(frames, dtype=np.float32)
    for index, frame in enumerate(frames.astype(np.float32, copy=False)):
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharpened[index] = np.clip(frame + amount * (frame - blurred), 0, 255)
    return sharpened
