"""Full-reference metrics for benchmark clips.

REDS4 PSNR and SSIM are computed on RGB with no border crop. The Vimeo
subset uses the Y channel, also with no border crop. tLPIPS is the mean
absolute difference of consecutive-frame LPIPS values and is not scaled
by 100. FID is the Fréchet distance between two Gaussian feature pools;
callers must report how many frames went into the pools.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import linalg


def psnr(restored: np.ndarray, ground_truth: np.ndarray, y_channel: bool) -> float:
    restored_values = _prepare(restored, y_channel)
    truth_values = _prepare(ground_truth, y_channel)
    mse = np.mean((restored_values - truth_values) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim(restored: np.ndarray, ground_truth: np.ndarray, y_channel: bool) -> float:
    restored_values = _prepare(restored, y_channel)
    truth_values = _prepare(ground_truth, y_channel)
    if restored_values.ndim == 2:
        return _ssim_channel(restored_values, truth_values)
    channels = [
        _ssim_channel(restored_values[..., channel], truth_values[..., channel])
        for channel in range(restored_values.shape[2])
    ]
    return float(np.mean(channels))


def tlpips(restored_distances: np.ndarray, ground_truth_distances: np.ndarray) -> float:
    restored_values = np.asarray(restored_distances, dtype=np.float64)
    truth_values = np.asarray(ground_truth_distances, dtype=np.float64)
    if restored_values.shape != truth_values.shape:
        raise ValueError("consecutive-frame LPIPS sequences must have the same length")
    return float(np.mean(np.abs(restored_values - truth_values)))


def frechet_distance(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    difference = mean_a - mean_b
    cov_mean = linalg.sqrtm(cov_a.dot(cov_b))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(difference.dot(difference) + np.trace(cov_a + cov_b - 2 * cov_mean))


def _prepare(image: np.ndarray, y_channel: bool) -> np.ndarray:
    values = image.astype(np.float64)
    if y_channel:
        values = _rgb_to_y(values)
    return values


def _rgb_to_y(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Y-channel metrics require an RGB image")
    scaled = image / 255.0
    y = 65.481 * scaled[..., 0] + 128.553 * scaled[..., 1] + 24.966 * scaled[..., 2] + 16.0
    return y


def _ssim_channel(first: np.ndarray, second: np.ndarray) -> float:
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(first, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(second, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = cv2.filter2D(first * first, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(second * second, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(first * second, -1, window)[5:-5, 5:-5] - mu12
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(numerator / denominator))
