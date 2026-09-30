"""Download the non-official PyTorch TecoGAN BI 4x weights after EDVR.

These weights are the skycrapers reimplementation trained on Vimeo bicubic
data. They are not the released TensorFlow checkpoint in
docs/results/tecogan_official_attempt.json. The download waits until the
EDVR gdown process 2388906 exits so the two Drive transfers do not share
one slow connection.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "weights" / "TecoGAN_BI4x_Vimeo_pytorch_NOT_OFFICIAL.pth"
DRIVE = "https://drive.google.com/uc?id=1ie1F7wJcO4mhNWK8nPX7F0LgOoPzCwEu"
PYTHON = ROOT.parent / "env_cv" / "bin" / "python"


def main() -> None:
    if OUTPUT.is_file() and OUTPUT.stat().st_size > 1_000_000:
        print(f"already present: {OUTPUT}")
        return
    subprocess.check_call(
        [str(PYTHON), "-m", "gdown", DRIVE, "-O", str(OUTPUT)],
    )
    print(f"saved {OUTPUT} bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
