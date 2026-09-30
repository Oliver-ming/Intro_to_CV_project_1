"""Separable upsamplers checked against OpenCV.

Bicubic uses the OpenCV cubic (a = -0.75), four taps, and replicated borders.
Lanczos uses a radius of 4, eight taps, replicated borders, and renormalized
weights. Both map an output pixel i to source coordinate (i + 0.5) * scale - 0.5.
"""

from __future__ import annotations

import numpy as np

_BICUBIC_A = -0.75


def upsample(image: np.ndarray, scale: int, method: str) -> np.ndarray:
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    height, width = image.shape[:2]
    return resize(image, (height * scale, width * scale), method)


def resize(image: np.ndarray, dst_hw: tuple[int, int], method: str) -> np.ndarray:
    if method == "bicubic":
        kernel, offset, taps, renorm = _cubic, -1, 4, False
    elif method == "lanczos":
        kernel, offset, taps, renorm = _lanczos4, -3, 8, True
    else:
        raise ValueError(f"unknown method {method!r}")
    output = _resize_axis(image, dst_hw[1], 1, kernel, offset, taps, renorm)
    return _resize_axis(output, dst_hw[0], 0, kernel, offset, taps, renorm)


def _cubic(distance: np.ndarray) -> np.ndarray:
    x = np.abs(distance)
    x2 = x * x
    x3 = x2 * x
    weights = np.zeros_like(x)
    near = x <= 1
    far = (x > 1) & (x < 2)
    a = _BICUBIC_A
    weights[near] = ((a + 2) * x3 - (a + 3) * x2 + 1)[near]
    weights[far] = (a * x3 - 5 * a * x2 + 8 * a * x - 4 * a)[far]
    return weights


def _lanczos4(distance: np.ndarray) -> np.ndarray:
    x = np.abs(distance)
    weights = np.zeros_like(x)
    nonzero = (x > 0) & (x < 4)
    narrowed = x[nonzero]
    weights[nonzero] = np.sinc(narrowed) * np.sinc(narrowed / 4)
    weights[x == 0] = 1
    return weights


def _resize_axis(image, dst_len, axis, kernel, offset, taps, renorm):
    source = np.asarray(image, dtype=np.float32)
    length = source.shape[axis]
    inverse = length / dst_len
    moved = np.moveaxis(source, axis, -1)
    positions = (np.arange(dst_len, dtype=np.float32) + 0.5) * inverse - 0.5
    left = np.floor(positions).astype(np.int64) + offset
    indices = left[:, None] + np.arange(taps, dtype=np.int64)[None, :]
    weights = kernel(positions[:, None] - indices).astype(np.float32)
    if renorm:
        weights = weights / weights.sum(axis=1, keepdims=True)
    clipped = np.clip(indices, 0, length - 1)
    samples = moved[..., clipped]
    resized = np.einsum("...dt,dt->...d", samples, weights, optimize=True)
    return np.moveaxis(resized, -1, axis)
