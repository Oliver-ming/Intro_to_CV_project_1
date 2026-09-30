"""Score one REDS4 restoration against ground truth.

PSNR and SSIM are RGB with no border crop. LPIPS is AlexNet. tLPIPS is not
multiplied by 100. FID is the small-sample Fréchet distance, and the frame
count is included in the result.
"""

from __future__ import annotations

import numpy as np

from part1.fid import InceptionPool, pool_distance
from part1.io import quantize
from part1.metrics import psnr, ssim
from part1.perceptual import AlexLPIPS


def score_reds(restored: np.ndarray, truth: np.ndarray, scorer: AlexLPIPS, inception: InceptionPool) -> dict[str, float]:
    count = min(len(restored), len(truth))
    restored = quantize(restored[:count])
    truth = quantize(truth[:count])
    scores = {
        "psnr_rgb": float(np.mean([psnr(restored[index], truth[index], False) for index in range(count)])),
        "ssim_rgb": float(np.mean([ssim(restored[index], truth[index], False) for index in range(count)])),
        "lpips": float(np.mean([scorer(restored[index], truth[index]) for index in range(count)])),
        "tlpips": scorer.temporal(restored, truth),
        "frames": count,
    }
    if count >= 2:
        scores["fid"] = pool_distance(inception(restored), inception(truth))
    return scores
