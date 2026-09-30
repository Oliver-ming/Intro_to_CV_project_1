"""Average finished REDS4 clip scores and fill the report table.

A method stays blank unless every one of its clips has a numeric PSNR.
Failure rows in the table are left as written.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1.score import mean_clip_scores

SOURCES = {
    "part1": ROOT / "outputs" / "part1" / "metrics.json",
    "edvr": ROOT / "outputs" / "part2" / "edvr" / "metrics.json",
    "tecogan_not_official": ROOT / "outputs" / "part2" / "tecogan_not_official" / "metrics.json",
    "basicvsrpp": ROOT / "outputs" / "part3" / "basicvsrpp" / "metrics.json",
    "star": ROOT / "outputs" / "part3" / "star" / "metrics.json",
    "hybrid": ROOT / "outputs" / "part3" / "hybrid" / "metrics.json",
}
ROWS = (
    ("Bicubic", "part1", "bicubic"),
    ("Lanczos", "part1", "lanczos"),
    ("EDSR", "part1", "edsr"),
    ("Temporal average", "part1", "temporal_average"),
    ("Unsharp", "part1", "temporal_unsharp"),
    ("EDVR-L", "edvr", None),
    ("BasicVSR++", "basicvsrpp", None),
    ("TecoGAN PyTorch, not released", "tecogan_not_official", None),
    ("STAR light degradation", "star", None),
    ("Hybrid", "hybrid", None),
)
PAPER = ROOT / "paper" / "main.tex"
MEANS = ROOT / "docs" / "results" / "reds4_means.json"


def clip_map(payload: dict) -> dict:
    clips = payload.get("clips")
    if isinstance(clips, dict):
        return clips
    return payload


def part1_method(payload: dict, method: str) -> dict[str, dict]:
    return {
        clip: scores[method]
        for clip, scores in payload.items()
        if isinstance(scores, dict) and isinstance(scores.get(method), dict)
    }


def collect() -> dict[str, dict | None]:
    loaded = {}
    for name, path in SOURCES.items():
        loaded[name] = json.loads(path.read_text()) if path.is_file() else None
    means = {}
    part1 = loaded["part1"] or {}
    for label, source, method in ROWS:
        if method is not None:
            clips = part1_method(part1, method) if part1 else {}
        else:
            clips = clip_map(loaded[source]) if loaded[source] else {}
        means[label] = mean_clip_scores(clips) if clips else None
    return means


def fill_table(text: str, means: dict[str, dict | None]) -> str:
    for label, _source, _method in ROWS:
        scores = means.get(label)
        if not scores or "psnr_rgb" not in scores or "fid" not in scores:
            continue
        row = (
            f"{label} & {scores['psnr_rgb']:.2f} & {scores['ssim_rgb']:.4f} "
            f"& {scores['lpips']:.4f} & {scores['tlpips']:.4f} & {scores['fid']:.2f} \\\\"
        )
        pattern = re.compile(rf"^{re.escape(label)} & .*\\\\$", re.MULTILINE)
        if pattern.search(text) is None:
            raise SystemExit(f"report table has no row for {label}")
        text = pattern.sub(row, text, count=1)
    return text


def main() -> None:
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
