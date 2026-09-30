"""Record that STAR's training script does not optimize LoRA.

The course LoRA row needs an interface that trains injected low-rank weights.
Pinned STAR train_sr.py optimizes VideoControlNet and parameters whose names
contain 'local'. It does not put LoRA parameters into that set.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "STAR" / "video_super_resolution" / "scripts" / "train_sr.py"
OUTPUT = ROOT / "docs" / "results" / "star_lora_attempt.json"
COMMIT = "69b8bc53e4bb04267265fa9a288ee30882c5b2ed"


def main() -> None:
    text = SOURCE.read_text()
    trains_controlnet = "params_to_optimize.update(model.VideoControlNet.parameters())" in text
    trains_local = "if 'local' in name:" in text
    trains_lora = "lora" in text.lower() and "params_to_optimize" in text and "LoRA" in text
    if not trains_controlnet or not trains_local or trains_lora:
        raise SystemExit("pinned STAR training script no longer matches the inspected optimizer")
    record = {
        "commit": COMMIT,
        "source": "video_super_resolution/scripts/train_sr.py",
        "lora_row": "failure",
        "official_inference": "not run; released checkpoint is not on disk",
        "reason": (
            "train_sr.py optimizes VideoControlNet and parameters whose names contain 'local'. "
            "It does not add LoRA weights to params_to_optimize, so the course LoRA row has no interface to attach."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
