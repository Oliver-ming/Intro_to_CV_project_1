"""Record whether the official TecoGAN archive can run in env_cv.

The released archive is the TensorFlow checkpoint from Chu et al. env_cv is
Python 3.10, which cannot install TensorFlow 1. This script writes a failure
record when the archive contains TensorFlow checkpoints and no PyTorch state
dict. It does not treat any other checkpoint as the released weights.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "weights" / "tecogan_official_model.zip"
OUTPUT = ROOT / "outputs" / "part2" / "tecogan" / "official_attempt.json"


def main() -> None:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size < 1000:
        raise SystemExit(f"official archive is not ready: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as zipped:
        names = zipped.namelist()
    tensorflow = [name for name in names if name.endswith((".meta", ".index", ".data-00000-of-00001"))]
    pytorch = [name for name in names if name.endswith((".pth", ".pt"))]
    record = {
        "archive": str(ARCHIVE),
        "bytes": ARCHIVE.stat().st_size,
        "tensorflow_checkpoints": tensorflow,
        "pytorch_checkpoints": pytorch,
        "loaded_released_weights": False,
        "reason": (
            "The released TecoGAN files are TensorFlow 1 checkpoints. "
            "env_cv is Python 3.10 and cannot install TensorFlow 1, so those weights were not loaded."
            if tensorflow and not pytorch
            else "The archive does not contain a usable PyTorch state dict."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
