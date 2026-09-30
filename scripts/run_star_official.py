"""Run official STAR light-degradation weights on REDS4.

The checkpoint is SherryX/STAR I2VGen-XL light_deg.pt. Its training
degradation is real-world, not bicubic. The course LoRA row is already a
failure; this script is inference only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAR_ROOT = ROOT / "third_party" / "STAR"
WEIGHT = ROOT / "weights" / "star_light_deg.pt"
EXPECTED = 8165233536
sys.path.insert(0, str(STAR_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from part1.io import read_sequence, write_video

CLIPS = ("000", "011", "015", "020")


def main() -> None:
    control = Path(str(WEIGHT) + ".aria2")
    if control.exists() or not WEIGHT.is_file() or WEIGHT.stat().st_size != EXPECTED:
        raise SystemExit(f"STAR light_deg.pt is not complete: {WEIGHT}")
    from video_super_resolution.scripts.inference_sr import STAR

    output_root = ROOT / "outputs" / "part3" / "star"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": "I2VGen-XL-based/light_deg.pt",
        "training_degradation": "real-world light degradation, not bicubic",
        "lora": "not run; no LoRA interface in train_sr.py",
        "clips": {},
    }
    model = None
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        frames = read_sequence(sorted(low_dir.glob("*.png")))
        source = output_root / f"{clip}_lr.mp4"
        write_video(source, frames, fps=8)
        if model is None:
            model = STAR(
                result_dir=str(output_root),
                file_name=f"{clip}.mp4",
                model_path=str(WEIGHT),
                solver_mode="fast",
                steps=15,
                upscale=4,
                max_chunk_len=32,
            )
        model.file_name = f"{clip}.mp4"
        model.enhance_a_video(str(source), "a realistic video frame")
        summary["clips"][clip] = {"frames": int(len(frames))}
        print(clip, "done", flush=True)
    result = ROOT / "docs" / "results" / "star_inference.json"
    result.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
