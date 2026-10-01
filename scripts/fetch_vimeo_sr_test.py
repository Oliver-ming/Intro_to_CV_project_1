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
URLS = (
    (
        "http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip",
        ROOT / "data" / "zips" / "vimeo_super_resolution_test.zip",
    ),
    (
        "http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip",
        ROOT / "data" / "zips" / "vimeo_test_clean.zip",
    ),
)
SUBSET = ROOT / "docs" / "manifests" / "vimeo-subset.txt"
DESTINATION = ROOT / "data" / "vimeo"
_LOW = {"low_resolution", "low", "lr", "bicubic"}
_HIGH = {"high_resolution", "target", "gt", "sharp"}


def subset_member(name: str, keys: set[str]) -> bool:
    if not name.lower().endswith(".png"):
        return False
    parts = name.replace("\\", "/").split("/")
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


def clips_ready(root: Path, keys: set[str]) -> bool:
    """True when every key has one low-resolution folder and one ground-truth folder."""

    if not root.is_dir():
        return False
    for key in keys:
        roles = []
        for path in root.glob(f"**/{key}"):
            if not path.is_dir() or not any(path.glob("*.png")):
                continue
            parts = {part.lower() for part in path.parts}
            if parts & _LOW:
                roles.append("low")
            elif parts & _HIGH:
                roles.append("high")
        if roles.count("low") != 1 or roles.count("high") != 1:
            return False
    return True


def main() -> None:
    keys = {line.strip() for line in SUBSET.read_text().splitlines() if line.strip()}
    if len(keys) != 30:
        raise SystemExit(f"expected 30 Vimeo keys, found {len(keys)}")
    if clips_ready(DESTINATION, keys):
        (ROOT / "data" / "logs" / "vimeo_sr_test.done").touch()
        print("already extracted", flush=True)
        return
    statuses = []
    for url, path in URLS:
        if clips_ready(DESTINATION, keys):
            break
        status = http_status(url)
        statuses.append(status)
        if status != 200:
            print(f"{path.name} HTTP {status}", flush=True)
            continue
        _download(url, path)
        if not archive_ready(path):
            raise SystemExit(f"{path.name} is not a complete zip")
        _extract(path, keys)
    if not clips_ready(DESTINATION, keys):
        raise SystemExit(f"official Vimeo frames are incomplete: {statuses}")
    (ROOT / "data" / "logs" / "vimeo_sr_test.done").touch()
    print("done", flush=True)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            str(path.parent),
            "-o",
            path.name,
            url,
        ]
    )


def _extract(path: Path, keys: set[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if subset_member(name, keys)]
        print(f"{path.name} members {len(names)}", flush=True)
        if not names:
            return
        archive.extractall(DESTINATION, members=names)


if __name__ == "__main__":
    main()
