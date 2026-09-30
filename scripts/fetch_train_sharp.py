"""Resume the official REDS train_sharp archive and extract REDS4.

Hugging Face redirects to a signed CDN URL. A dropped connection then retries
that expired URL and gets HTTP 403. This script always restarts from the
stable dataset URL and continues the existing zip. It extracts clips 000,
011, 015, and 020 only after the archive is the published length.
"""

from __future__ import annotations

import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "zips" / "train_sharp.zip"
URL = "https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip"
EXPECTED = 34261573976
CLIPS = {"000", "011", "015", "020"}


def main() -> None:
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    while not ZIP.is_file() or ZIP.stat().st_size != EXPECTED:
        before = ZIP.stat().st_size if ZIP.is_file() else 0
        subprocess.call(["wget", "-c", "--timeout=60", "--read-timeout=60", "-O", str(ZIP), URL])
        after = ZIP.stat().st_size if ZIP.is_file() else 0
        if after == EXPECTED:
            break
        if after <= before:
            time.sleep(20)
    destination = ROOT / "data" / "reds"
    with zipfile.ZipFile(ZIP) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".png") and CLIPS.intersection(name.split("/"))
        ]
        print(f"members {len(names)}", flush=True)
        archive.extractall(destination, members=names)
    (ROOT / "data" / "logs" / "reds4.done").touch()
    print("done", flush=True)


if __name__ == "__main__":
    main()
