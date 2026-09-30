"""Locate one frozen Vimeo subset clip on disk.

The official super-resolution test archive uses a low-resolution tree and a
high-resolution tree. Clip keys are the `clip/sequence` lines in
docs/manifests/vimeo-subset.txt.
"""

from __future__ import annotations

from pathlib import Path

_LOW = {"low_resolution", "low", "lr", "bicubic"}
_HIGH = {"high_resolution", "target", "gt", "sharp"}


def pair_clip(root: Path, key: str) -> tuple[Path, Path | None]:
    matches = [
        path
        for path in root.glob(f"**/{key}")
        if path.is_dir() and any(path.glob("*.png"))
    ]
    low = [path for path in matches if _role(path) == "low"]
    high = [path for path in matches if _role(path) == "high"]
    if len(low) != 1:
        raise FileNotFoundError(f"expected one low-resolution folder for {key}, found {len(low)}")
    if len(high) > 1:
        raise FileNotFoundError(f"expected at most one high-resolution folder for {key}")
    return low[0], high[0] if high else None


def _role(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if parts & _LOW:
        return "low"
    if parts & _HIGH:
        return "high"
    return "unknown"
