"""Numerical checks for the Part 1 baselines."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part1.resample import upsample
from part1.temporal import temporal_average, unsharp


def test_bicubic_matches_opencv():
    image = np.random.default_rng(0).integers(0, 256, size=(18, 22, 3)).astype(np.float32)
    reference = cv2.resize(image, (88, 72), interpolation=cv2.INTER_CUBIC)
    ours = upsample(image, 4, "bicubic")
    np.testing.assert_allclose(ours, reference, atol=1e-3, rtol=0)


def test_lanczos_matches_opencv():
    image = np.random.default_rng(1).integers(0, 256, size=(15, 19)).astype(np.float32)
    reference = cv2.resize(image, (76, 60), interpolation=cv2.INTER_LANCZOS4)
    ours = upsample(image, 4, "lanczos")
    np.testing.assert_allclose(ours, reference, atol=1e-3, rtol=0)


def test_temporal_average_uses_binomial_weights_and_renormalizes_edges():
    frames = np.zeros((5, 2, 2), dtype=np.float32)
    frames[:, 0, 0] = np.arange(5)
    averaged = temporal_average(frames)
    edge_weights = np.array([6, 4, 1], dtype=np.float32)
    edge_weights /= edge_weights.sum()
    expected_edge = np.dot(np.array([0, 1, 2], dtype=np.float32), edge_weights)
    np.testing.assert_allclose(averaged[0, 0, 0], expected_edge)
    center_weights = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    center_weights /= center_weights.sum()
    expected_center = np.dot(np.arange(5, dtype=np.float32), center_weights)
    np.testing.assert_allclose(averaged[2, 0, 0], expected_center)


def test_unsharp_increases_a_step_edge():
    frame = np.zeros((16, 16), dtype=np.float32)
    frame[:, 8:] = 200
    sharpened = unsharp(frame[None])[0]
    assert sharpened[8, 8] > frame[8, 8]
    assert sharpened.min() >= 0
    assert sharpened.max() <= 255
