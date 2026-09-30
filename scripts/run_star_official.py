"""Run official STAR light-degradation weights on REDS4.

The checkpoint is SherryX/STAR I2VGen-XL light_deg.pt. Its training
degradation is real-world, not bicubic. The course LoRA row is already a
failure; this script is inference only.
"""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STAR_ROOT = ROOT / "third_party" / "STAR"
WEIGHT = ROOT / "weights" / "star_light_deg.pt"
EXPECTED = 8165233536
sys.path.insert(0, str(STAR_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from part1.io import read_sequence, write_frame, write_video

CLIPS = ("000", "011", "015", "020")


def _install_attention_shim() -> None:
    """STAR's U-Net imports xformers, which is pinned to an old torch.

    The inference path calls memory_efficient_attention with one head per
    batch row and no bias. PyTorch scaled-dot-product attention matches that
    call, so the run does not install xformers 0.0.21.
    """

    if "xformers" in sys.modules:
        return
    ops = types.ModuleType("xformers.ops")

    def memory_efficient_attention(query, key, value, attn_bias=None, op=None, p=0.0):
        if attn_bias is not None:
            raise NotImplementedError("this STAR path does not pass an attention bias")
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=p)

    ops.memory_efficient_attention = memory_efficient_attention
    module = types.ModuleType("xformers")
    module.ops = ops
    module.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None)
    ops.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)
    sys.modules["xformers"] = module
    sys.modules["xformers.ops"] = ops


def _enhance(model, source: Path) -> int:
    # 32, 16, and 8 frames at 720×1280 each asked for about 31 GB.
    for chunk in (2, 1):
        model.max_chunk_len = chunk
        try:
            model.enhance_a_video(str(source), "a realistic video frame")
            return chunk
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or chunk == 1:
                raise
            torch.cuda.empty_cache()
            print(f"out of memory at chunk {chunk}, retrying shorter", flush=True)
    raise RuntimeError("STAR did not finish a chunk length")


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    control = Path(str(WEIGHT) + ".aria2")
    if control.exists() or not WEIGHT.is_file() or WEIGHT.stat().st_size != EXPECTED:
        raise SystemExit(f"STAR light_deg.pt is not complete: {WEIGHT}")
    _install_attention_shim()
    import video_super_resolution.scripts.inference_sr as inference_sr

    def _save_video(video, save_dir, file_name, fps=16.0):
        frame_dir = Path(save_dir) / "frames" / Path(file_name).stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(video):
            write_frame(frame_dir / f"{index:08d}.png", image.detach().cpu().numpy())
        return inference_sr_save(video, save_dir, file_name, fps)

    inference_sr_save = inference_sr.save_video
    inference_sr.save_video = _save_video
    STAR = inference_sr.STAR

    output_root = ROOT / "outputs" / "part3" / "star"
    output_root.mkdir(parents=True, exist_ok=True)
    result = ROOT / "docs" / "results" / "star_inference.json"
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
        frame_dir = output_root / "frames" / clip
        existing = sorted(frame_dir.glob("*.png")) if frame_dir.is_dir() else []
        if len(existing) == len(frames):
            summary["clips"][clip] = {"frames": int(len(frames)), "max_chunk_len": "reused"}
            print(clip, "reused", flush=True)
            _write_summary(result, summary)
            continue
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
        chunk = _enhance(model, source)
        summary["clips"][clip] = {"frames": int(len(frames)), "max_chunk_len": chunk}
        print(clip, "done", chunk, flush=True)
        _write_summary(result, summary)
    print(json.dumps(summary, indent=2))
    if any((ROOT / "data" / "reds" / "train" / "train_sharp").glob("*/*.png")):
        sys.path.insert(0, str(ROOT / "scripts"))
        from score_ready import main as score_all

        score_all()


if __name__ == "__main__":
    main()
