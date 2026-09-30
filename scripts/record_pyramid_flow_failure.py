"""Record that released Pyramid Flow cannot restore a whole clip.

The pinned source exposes text-to-video generation from noise and
image-to-video from one clean frame. It does not expose an entry point that
noises a bicubic-upscaled clip and integrates from the middle of the sampler.
Direction A is therefore a failure case. Unconditioned samples are not written.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "Pyramid-Flow" / "pyramid_dit" / "pyramid_dit_for_video_gen_pipeline.py"
OUTPUT = ROOT / "docs" / "results" / "pyramid_flow_attempt.json"
COMMIT = "a012faa1dc4d71301a7a153c7f9554c081947ea2"


def main() -> None:
    text = SOURCE.read_text()
    has_i2v = "def generate_i2v(" in text and "input_image: PIL.Image = None" in text
    keeps_clean_frame = "generated_latents_list = [input_image_latent]" in text
    has_whole_clip_start = "def generate_from_video" in text or "bicubic" in text
    if not (has_i2v and keeps_clean_frame) or has_whole_clip_start:
        raise SystemExit("pinned Pyramid Flow source no longer matches the inspected interface")
    record = {
        "commit": COMMIT,
        "source": "pyramid_dit/pyramid_dit_for_video_gen_pipeline.py",
        "entered_metric_table": False,
        "reason": (
            "generate_i2v takes one clean PIL image and keeps that latent as history. "
            "generate samples from noise. The pinned source has no entry point that noises "
            "a whole bicubic-upscaled clip and integrates from the middle of the sampler."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
