"""Numerical checks for the Part 1 baselines."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part1.resample import upsample
from part1.temporal import temporal_average, unsharp
from part1.metrics import frechet_distance, psnr, ssim, tlpips


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


def test_psnr_is_infinite_for_identical_frames_and_finite_for_rgb():
    image = np.full((32, 32, 3), 128, dtype=np.float32)
    assert psnr(image, image, y_channel=False) == float("inf")
    shifted = image.copy()
    shifted[..., 0] = 108
    value = psnr(image, shifted, y_channel=False)
    assert 20 < value < 30


def test_y_psnr_weights_green_more_than_red():
    base = np.zeros((32, 32, 3), dtype=np.float32)
    red = base.copy()
    red[..., 0] = 10
    green = base.copy()
    green[..., 1] = 10
    assert psnr(base, green, y_channel=True) < psnr(base, red, y_channel=True)


def test_ssim_of_identical_frames_is_one():
    image = np.random.default_rng(2).integers(0, 256, size=(40, 40, 3)).astype(np.float32)
    np.testing.assert_allclose(ssim(image, image, y_channel=False), 1.0, atol=1e-6)


def test_tlpips_is_the_absolute_difference_of_consecutive_lpips():
    restored = np.array([0.2, 0.5])
    truth = np.array([0.1, 0.1])
    np.testing.assert_allclose(tlpips(restored, truth), 0.25)


def test_pool_distance_is_zero_for_identical_features():
    from part1.fid import pool_distance

    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(pool_distance(features, features), 0.0, atol=1e-6)


def test_official_edsr_upsamples_by_four():
    weight = Path(__file__).resolve().parents[1] / "weights" / "edsr_baseline_x4-6b446fab.pt"
    if not weight.is_file() or weight.stat().st_size != 6081831:
        return
    import torch
    from part1.edsr import load_edsr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_edsr(weight, device)
    frame = torch.rand(1, 3, 16, 16, device=device) * 255
    with torch.no_grad():
        restored = model(frame)
    assert tuple(restored.shape) == (1, 3, 64, 64)


def test_frechet_distance_is_zero_for_the_same_gaussian():
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    np.testing.assert_allclose(frechet_distance(mean, cov, mean, cov), 0.0, atol=1e-8)


def test_edvr_reflection_circle_matches_basicsr_example():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "third_party" / "BasicSR"
    sys.path.insert(0, str(root))
    from basicsr.data.data_util import generate_frame_indices

    assert generate_frame_indices(0, 100, 5, "reflection_circle") == [4, 3, 0, 1, 2]


def test_disagreement_map_sends_certain_pixels_to_fidelity():
    from part3.hybrid import blend

    fidelity = np.zeros((2, 2, 3), dtype=np.float32)
    generative = np.zeros((2, 2, 3), dtype=np.float32)
    generative[0, 0] = 100
    generative[0, 1] = 100
    certain = np.zeros((2, 2), dtype=bool)
    certain[0, 0] = True
    mixed = blend(fidelity, generative, certain)
    np.testing.assert_allclose(mixed[0, 0], 0)
    np.testing.assert_allclose(mixed[0, 1], 100)


def test_basicvsrpp_upsamples_a_tiny_sequence():
    import sys
    import torch
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "third_party" / "BasicSR"
    sys.path.insert(0, str(root))
    if not torch.cuda.is_available():
        return
    from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

    model = BasicVSRPlusPlus(mid_channels=16, num_blocks=1, is_low_res_input=True, cpu_cache_length=10)
    model.cuda().eval()
    frames = torch.rand(1, 2, 3, 64, 64, device="cuda")
    with torch.no_grad():
        restored = model(frames)
    assert tuple(restored.shape) == (1, 2, 3, 256, 256)


def test_part1_writes_a_video_and_scores_bicubic():
    from part1.io import write_video
    from part1.perceptual import AlexLPIPS
    from part1.restore import restore_part1

    low_resolution = np.random.default_rng(3).integers(0, 256, size=(3, 8, 8, 3)).astype(np.float32)
    restored = restore_part1(low_resolution)
    assert restored["bicubic"].shape == (3, 32, 32, 3)
    assert restored["temporal_unsharp"].shape == restored["bicubic"].shape
    video = Path("/tmp/part1_bicubic.mp4")
    write_video(video, restored["bicubic"], fps=5)
    assert video.stat().st_size > 0
    score = AlexLPIPS()
    assert score(restored["bicubic"][0], restored["bicubic"][0]) < 1e-5
    assert score.temporal(restored["bicubic"], restored["bicubic"]) == 0.0
