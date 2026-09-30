"""Course hybrid of a fidelity frame and a generative frame.

Certain pixels, supplied by the caller, always come from the fidelity frame.
Every other pixel uses the per-frame-normalized mean absolute RGB difference
as a disagreement map. This is not an uncertainty estimator from EDVR,
BasicVSR++, STAR, or UPSR.
"""

from __future__ import annotations

import numpy as np


def disagreement_map(fidelity: np.ndarray, generative: np.ndarray) -> np.ndarray:
    if fidelity.shape != generative.shape:
        raise ValueError("fidelity and generative frames must have the same shape")
    difference = np.mean(np.abs(fidelity.astype(np.float32) - generative.astype(np.float32)), axis=-1)
    low = float(difference.min())
    high = float(difference.max())
    if high == low:
        return np.zeros_like(difference, dtype=np.float32)
    return ((difference - low) / (high - low)).astype(np.float32)


def blend(fidelity: np.ndarray, generative: np.ndarray, certain: np.ndarray) -> np.ndarray:
    weight = disagreement_map(fidelity, generative)
    certain_mask = certain.astype(bool)
    weight = np.where(certain_mask, 0.0, weight).astype(np.float32)
    mixed = (1.0 - weight[..., None]) * fidelity + weight[..., None] * generative
    return mixed.astype(np.float32)
