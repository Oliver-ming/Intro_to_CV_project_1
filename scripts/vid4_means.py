"""Average finished Vid4 clip scores and fill that report table.

A method stays blank unless every clip has a numeric RGB PSNR.
FID is filled only from a pooled distance, not from an average of clip distances.
"""

from __future__ import annotations

import fcntl
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.score import mean_clip_scores

PAPER = ROOT / "paper" / "main.tex"
MEANS = ROOT / "docs" / "results" / "vid4_means.json"
POOLED = ROOT / "docs" / "results" / "vid4_pooled_fid.json"
ROWS = (
    ("Vid4 bicubic", ROOT / "outputs" / "vid4" / "part1" / "metrics.json", "bicubic"),
    ("Vid4 Lanczos", ROOT / "outputs" / "vid4" / "part1" / "metrics.json", "lanczos"),
    ("Vid4 EDSR", ROOT / "outputs" / "vid4" / "part1" / "metrics.json", "edsr"),
    ("Vid4 temporal average", ROOT / "outputs" / "vid4" / "part1" / "metrics.json", "temporal_average"),
    ("Vid4 unsharp", ROOT / "outputs" / "vid4" / "part1" / "metrics.json", "temporal_unsharp"),
    ("Vid4 EDVR-L", ROOT / "outputs" / "vid4" / "edvr" / "metrics.json", None),
    ("Vid4 BasicVSR++", ROOT / "outputs" / "vid4" / "basicvsrpp" / "metrics.json", None),
    ("Vid4 TecoGAN, not released", ROOT / "outputs" / "vid4" / "tecogan_not_official" / "metrics.json", None),
    ("Vid4 STAR", ROOT / "outputs" / "vid4" / "star" / "metrics.json", None),
    ("Vid4 hybrid", ROOT / "outputs" / "vid4" / "hybrid" / "metrics.json", None),
)


def clip_map(payload: dict, method: str | None) -> dict[str, dict]:
    if method is None:
        clips = payload.get("clips", payload)
        return {key: value for key, value in clips.items() if isinstance(value, dict) and "psnr_rgb" in value}
    return {
        clip: scores[method]
        for clip, scores in payload.items()
        if isinstance(scores, dict) and isinstance(scores.get(method), dict)
    }


def collect() -> dict[str, dict | None]:
    pooled = json.loads(POOLED.read_text()) if POOLED.is_file() else {}
    means = {}
    cache: dict[Path, dict | None] = {}
    for label, path, method in ROWS:
        if path not in cache:
            cache[path] = json.loads(path.read_text()) if path.is_file() else None
        payload = cache[path]
        averaged = mean_clip_scores(clip_map(payload, method)) if payload else None
        if averaged is not None and label in pooled and isinstance(pooled[label], (int, float)):
            averaged["pooled_fid"] = float(pooled[label])
        means[label] = averaged
    return means


def fill_table(text: str, means: dict[str, dict | None]) -> str:
    for label, _path, _method in ROWS:
        scores = means.get(label)
        if not scores or any(key not in scores for key in ("psnr_rgb", "ssim_rgb", "lpips", "tlpips")):
            continue
        fid = scores.get("pooled_fid")
        fid_cell = f"{fid:.2f}" if isinstance(fid, (int, float)) else "---"
        row = (
            f"{label} & {scores['psnr_rgb']:.2f} & {scores['ssim_rgb']:.4f} "
            f"& {scores['lpips']:.4f} & {scores['tlpips']:.4f} & {fid_cell} \\\\"
        )
        pattern = re.compile(
            rf"^{re.escape(label)} & (?:---|\d+(?:\.\d+)?) & (?:---|\d+(?:\.\d+)?) "
            rf"& (?:---|\d+(?:\.\d+)?) & (?:---|\d+(?:\.\d+)?) & (?:---|\d+(?:\.\d+)?) \\+$",
            re.MULTILINE,
        )
        if pattern.search(text) is None:
            raise SystemExit(f"Vid4 table has no row for {label}")
        text = pattern.sub(lambda _match: row, text, count=1)
    return text


def main() -> None:
    lock_path = ROOT / "data" / "logs" / "reds_means.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        means = collect()
        MEANS.parent.mkdir(parents=True, exist_ok=True)
        MEANS.write_text(json.dumps(means, indent=2) + "\n")
        if PAPER.is_file():
            PAPER.write_text(fill_table(PAPER.read_text(), means))
            _recompile()
        filled = [label for label, scores in means.items() if scores]
        print("filled", ", ".join(filled) if filled else "none", flush=True)


def _recompile() -> None:
    binary = shutil.which("tectonic") or "/tmp/tectonic-musl/tectonic"
    if not Path(binary).is_file():
        print("tectonic is not available; the table tex was updated", flush=True)
        return
    subprocess.call([binary, "main.tex"], cwd=ROOT / "paper")


if __name__ == "__main__":
    main()
