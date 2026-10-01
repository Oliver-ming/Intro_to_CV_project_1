"""Download the official Vimeo super-resolution test archive and extract 30 clips.

The MIT host currently returns HTTP 403. This script does not invent
low-resolution frames. It extracts only the frozen subset after the zip
downloads and aria2 has released it.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip"
ZIP = ROOT / "data" / "zips" / "vimeo_super_resolution_test.zip"
SUBSET = ROOT / "docs" / "manifests" / "vimeo-subset.txt"
DESTINATION = ROOT / "data" / "vimeo"


def subset_member(name: str, keys: set[str]) -> bool:
    if not name.lower().endswith(".png"):
        return False
    parts = name.split("/")
    return any(f"{parts[index]}/{parts[index + 1]}" in keys for index in range(len(parts) - 1))


def http_status(url: str) -> int:
    completed = subprocess.run(
        ["curl", "-sI", "-L", "--max-time", "20", "-o", "/dev/null", "-w", "%{http_code}", url],
        check=False,
        capture_output=True,
        text=True,
    )
    text = completed.stdout.strip()
    return int(text) if text.isdigit() else 0


def archive_ready(path: Path) -> bool:
    control = Path(str(path) + ".aria2")
    if not path.is_file() or control.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def main() -> None:
    keys = {line.strip() for line in SUBSET.read_text().splitlines() if line.strip()}
    if len(keys) != 30:
        raise SystemExit(f"expected 30 Vimeo keys, found {len(keys)}")
    if _extracted(keys):
        print("already extracted", flush=True)
        return
    status = http_status(URL)
    if status != 200:
        raise SystemExit(f"official Vimeo archive returned HTTP {status}")
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "aria2c",
            "-c",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "--file-allocation=none",
            "--timeout=60",
            "--max-tries=5",
            "-d",
            str(ZIP.parent),
            "-o",
            ZIP.name,
            URL,
        ]
    )
    if not archive_ready(ZIP):
        raise SystemExit("downloaded Vimeo archive is not a complete zip")
    with zipfile.ZipFile(ZIP) as archive:
        names = [name for name in archive.namelist() if subset_member(name, keys)]
        print(f"members {len(names)}", flush=True)
        if not names:
            raise SystemExit("archive contains none of the frozen subset clips")
        archive.extractall(DESTINATION, members=names)
    (ROOT / "data" / "logs" / "vimeo_sr_test.done").touch()
    print("done", flush=True)


def _extracted(keys: set[str]) -> bool:
    if not DESTINATION.is_dir():
        return False
    return all(any(DESTINATION.glob(f"**/{key}/*.png")) for key in keys)


if __name__ == "__main__":
    main()
