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


def score_vimeo(restored: np.ndarray, truth: np.ndarray, scorer: AlexLPIPS, inception: InceptionPool) -> dict[str, float]:
    count = min(len(restored), len(truth))
    restored = quantize(restored[:count])
    truth = quantize(truth[:count])
    scores = {
        "psnr_y": float(np.mean([psnr(restored[index], truth[index], True) for index in range(count)])),
        "ssim_y": float(np.mean([ssim(restored[index], truth[index], True) for index in range(count)])),
        "lpips": float(np.mean([scorer(restored[index], truth[index]) for index in range(count)])),
        "tlpips": scorer.temporal(restored, truth),
        "frames": count,
    }
    if count >= 2:
        scores["fid"] = pool_distance(inception(restored), inception(truth))
    return scores


def mean_clip_scores(clips: dict[str, dict]) -> dict[str, float] | None:
    """Average one method across clips.

    Returns None unless every clip has a numeric PSNR. Frame counts are summed.
    """

    clip_rows = {key: value for key, value in clips.items() if isinstance(value, dict)}
    scored = [
        value
        for value in clip_rows.values()
        if "psnr_rgb" in value or "psnr_y" in value
    ]
    if len(scored) != len(clip_rows) or not scored:
        return None
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    frames = 0
    for scores in scored:
        frames += int(scores["frames"])
        for key, value in scores.items():
            if key == "frames" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    averaged = {key: totals[key] / counts[key] for key in totals}
    averaged["frames"] = frames
    return averaged
