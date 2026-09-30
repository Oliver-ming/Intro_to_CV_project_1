"""Read frames and write restored sequences."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frame(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def read_sequence(paths: list[Path]) -> np.ndarray:
    return np.stack([read_frame(path) for path in paths])


def write_sequence(directory: Path, frames: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        write_frame(directory / f"{index:08d}.png", frame)


def write_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(_uint8(frame), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write {path}")


def write_video(path: Path, frames: np.ndarray, fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(_uint8(frame), cv2.COLOR_RGB2BGR))
    writer.release()


def quantize(frames: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frames), 0, 255).astype(np.float32)


def _uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)
