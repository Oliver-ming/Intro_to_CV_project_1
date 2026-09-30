"""Freeze the Vimeo subset and the adaptation clips.

The test list is BasicSR's meta_info_Vimeo90K_test_GT.txt (7824 clips, the
official Vimeo-90K-T set, in that file's order). The subset is 30 clips drawn
once with numpy Generator(0).choice, without replacement. The adaptation list
is 20 clips drawn the same way from the training list. Neither file is redrawn
after it exists; this script only writes a file that is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASICSR = ROOT / "third_party" / "BasicSR" / "basicsr" / "data" / "meta_info"
TEST_LIST = BASICSR / "meta_info_Vimeo90K_test_GT.txt"
TRAIN_LIST = BASICSR / "meta_info_Vimeo90K_train_GT.txt"
SUBSET = ROOT / "docs" / "manifests" / "vimeo-subset.txt"
ADAPTATION = ROOT / "docs" / "manifests" / "vimeo-adaptation.txt"


def main() -> None:
    _write_if_missing(SUBSET, _keys(TEST_LIST), 30)
    _write_if_missing(ADAPTATION, _keys(TRAIN_LIST), 20)


def draw(keys: list[str], count: int) -> list[str]:
    chosen = np.random.default_rng(0).choice(len(keys), size=count, replace=False)
    return [keys[int(index)] for index in chosen]


def _keys(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def _write_if_missing(path: Path, keys: list[str], count: int) -> None:
    if path.exists():
        print(f"kept {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(draw(keys, count)) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
