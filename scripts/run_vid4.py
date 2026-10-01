"""Run the REDS methods on official Vid4 BIx4 frames.

Vid4 replaces the 30-clip Vimeo subset. The MIT Vimeo test archive returns
HTTP 403. Outputs stay under outputs/vid4, so the REDS4 videos are left as
they are. PSNR and SSIM stay RGB with no border crop, matching REDS4.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from part1.edsr import load_edsr
from part1.fid import InceptionPool, pool_distance
from part1.io import quantize, read_sequence, write_sequence, write_video
from part1.metrics import psnr, ssim
from part1.perceptual import AlexLPIPS
from part1.restore import restore_part1

DATA = ROOT / "data" / "vid4" / "Vid4"
OUTPUT = ROOT / "outputs" / "vid4"
PART1_METHODS = ("bicubic", "lanczos", "edsr", "temporal_average", "temporal_unsharp")


def clips() -> list[str]:
    low_root = DATA / "BIx4"
    if not low_root.is_dir():
        raise SystemExit(f"Vid4 BIx4 frames are not extracted: {low_root}")
    names = sorted(path.name for path in low_root.iterdir() if path.is_dir())
    if names != ["calendar", "city", "foliage", "walk"]:
        raise SystemExit(f"unexpected Vid4 clips: {names}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--star", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    names = clips()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.star:
        _star(names)
        _hybrid(names, device, texture=False)
    elif args.hybrid:
        _hybrid(names, device, texture=False)
    else:
        _part1(names, device)
        _edvr(names, device)
        _basicvsr(names, device)
        _tecogan(names)
    lock_path = ROOT / "data" / "logs" / "vid4.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _score(names, device)
        from vid4_means import main as write_means

        write_means()


def _part1(names: list[str], device: torch.device) -> None:
    checkpoint = ROOT / "weights" / "edsr_baseline_x4-6b446fab.pt"
    edsr = load_edsr(checkpoint, device) if checkpoint.is_file() and checkpoint.stat().st_size == 6081831 else None
    for name in names:
        low = read_sequence(_frames(DATA / "BIx4" / name))
        pending = [method for method in PART1_METHODS if not _complete(OUTPUT / "part1" / "frames" / method / name, len(low))]
        if edsr is None:
            pending = [method for method in pending if method != "edsr"]
        if not pending:
            print(name, "part1 reused", flush=True)
            continue
        restored = restore_part1(low, edsr, device)
        for method, frames in restored.items():
            if method not in pending:
                continue
            directory = OUTPUT / "part1" / "frames" / method / name
            write_sequence(directory, frames)
            write_video(OUTPUT / "part1" / f"{method}_{name}.mp4", frames)
        print(name, "part1", ",".join(pending), flush=True)


def _edvr(names: list[str], device: torch.device) -> None:
    import run_edvr

    model = None
    for name in names:
        low = read_sequence(_frames(DATA / "BIx4" / name))
        directory = OUTPUT / "edvr" / "frames" / name
        if _complete(directory, len(low)):
            print(name, "edvr reused", flush=True)
            continue
        if model is None:
            model = run_edvr._load(device)
        restored = run_edvr._restore(model, low, device)
        write_sequence(directory, restored)
        write_video(OUTPUT / "edvr" / f"{name}.mp4", restored)
        print(name, "edvr", flush=True)
    if model is not None:
        del model
        torch.cuda.empty_cache()


def _basicvsr(names: list[str], device: torch.device) -> None:
    import run_basicvsrpp

    model = None
    for name in names:
        low = read_sequence(_frames(DATA / "BIx4" / name))
        directory = OUTPUT / "basicvsrpp" / "frames" / name
        if _complete(directory, len(low)):
            print(name, "basicvsrpp reused", flush=True)
            continue
        if model is None:
            model = run_basicvsrpp._load(run_basicvsrpp._weight(), device)
        restored = run_basicvsrpp._restore(model, low, device)
        write_sequence(directory, restored)
        write_video(OUTPUT / "basicvsrpp" / f"{name}.mp4", restored)
        print(name, "basicvsrpp", flush=True)
    if model is not None:
        del model
        torch.cuda.empty_cache()


def _tecogan(names: list[str]) -> None:
    import run_tecogan_not_official as tecogan
    from models.networks.tecogan_nets import FRNet

    weight = tecogan.WEIGHT
    if not weight.is_file():
        raise SystemExit(f"non-official TecoGAN weight is not ready: {weight}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    for name in names:
        low = read_sequence(_frames(DATA / "BIx4" / name))
        directory = OUTPUT / "tecogan_not_official" / "frames" / name
        if _complete(directory, len(low)):
            print(name, "tecogan reused", flush=True)
            continue
        if model is None:
            model = FRNet(in_nc=3, out_nc=3, nf=64, nb=10, degradation="BI", scale=4)
            state = torch.load(weight, map_location=device, weights_only=False)
            if isinstance(state, dict):
                state = state.get("generator") or state.get("state_dict") or state.get("params") or state
            model.load_state_dict(state, strict=True)
            model.to(device)
            model.eval()
        tensor = torch.from_numpy(np.transpose(low / 255.0, (0, 3, 1, 2)))
        restored = np.asarray(model.infer_sequence(tensor, device), dtype=np.float32)
        if restored.max() <= 1.5:
            restored = restored * 255.0
        write_sequence(directory, restored)
        write_video(OUTPUT / "tecogan_not_official" / f"{name}.mp4", restored)
        print(name, "tecogan", flush=True)


def _star(names: list[str]) -> None:
    import run_star_official as star

    star._install_attention_shim()
    import video_super_resolution.scripts.inference_sr as inference_sr

    output_root = OUTPUT / "star"
    output_root.mkdir(parents=True, exist_ok=True)

    def _save_video(video, save_dir, file_name, fps=16.0):
        frame_dir = Path(save_dir) / "frames" / Path(file_name).stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(video):
            from part1.io import write_frame

            write_frame(frame_dir / f"{index:08d}.png", image.detach().cpu().numpy())
        return original_save(video, save_dir, file_name, fps)

    original_save = inference_sr.save_video
    inference_sr.save_video = _save_video
    model = None
    for name in names:
        low = _frames(DATA / "BIx4" / name)
        frame_dir = output_root / "frames" / name
        if _complete(frame_dir, len(low)):
            print(name, "star reused", flush=True)
            continue
        source = output_root / f"{name}_lr.mp4"
        write_video(source, read_sequence(low), fps=8)
        if model is None:
            model = inference_sr.STAR(
                result_dir=str(output_root),
                file_name=f"{name}.mp4",
                model_path=str(star.WEIGHT),
                solver_mode="fast",
                steps=15,
                upscale=4,
                max_chunk_len=1,
            )
        model.file_name = f"{name}.mp4"
        model.max_chunk_len = 1
        model.enhance_a_video(str(source), "a realistic video frame")
        print(name, "star", flush=True)


def _hybrid(names: list[str], device: torch.device, texture: bool) -> None:
    from part1.resample import upsample
    from part3.detectors import certain_mask, texture_mask
    from part3.hybrid import blend

    fidelity_root = OUTPUT / "basicvsrpp" / "frames"
    generative_root = OUTPUT / "star" / "frames"
    if not fidelity_root.is_dir() or not generative_root.is_dir():
        print("hybrid waiting for BasicVSR++ and STAR", flush=True)
        return
    output_root = OUTPUT / ("hybrid_texture" if texture else "hybrid")
    for name in names:
        fidelity_paths = sorted((fidelity_root / name).glob("*.png"))
        generative_paths = sorted((generative_root / name).glob("*.png"))
        low_paths = _frames(DATA / "BIx4" / name)
        if len(fidelity_paths) != len(low_paths) or len(generative_paths) != len(low_paths):
            print(name, "hybrid waiting", flush=True)
            return
        directory = output_root / "frames" / name
        if _complete(directory, len(low_paths)):
            print(name, "hybrid reused", flush=True)
            continue
        fidelity = read_sequence(fidelity_paths)
        generative = read_sequence(generative_paths)
        low = read_sequence(low_paths)
        guide = np.stack([upsample(frame, 4, "bicubic") for frame in low])
        certain = np.stack([certain_mask(frame) for frame in guide])
        grass = np.stack([texture_mask(frame) for frame in guide]) if texture else None
        mixed = np.stack(
            [
                blend(fidelity[index], generative[index], certain[index], None if grass is None else grass[index])
                for index in range(len(low_paths))
            ]
        )
        write_sequence(directory, mixed)
        write_video(output_root / f"{name}.mp4", mixed)
        print(name, "hybrid", flush=True)
    del device


def _score(names: list[str], device: torch.device) -> None:
    truth = {name: read_sequence(_frames(DATA / "GT" / name)) for name in names}
    methods = _ready_methods(names)
    if not methods:
        print("vid4 score waiting for frames", flush=True)
        return
    scorer = AlexLPIPS(device)
    inception = InceptionPool(device)
    truth_features = _features(inception, [truth[name] for name in names])
    pooled_path = ROOT / "docs" / "results" / "vid4_pooled_fid.json"
    pooled = json.loads(pooled_path.read_text()) if pooled_path.is_file() else {}
    grouped: dict[Path, dict] = {}
    for label, directory, method in methods:
        if label in pooled:
            print(label, "pooled reused", pooled[label], flush=True)
            continue
        payload = grouped.setdefault(directory.parent / "metrics.json", {})
        clip_scores = {}
        restored_frames = []
        for name in names:
            folder = directory / method / name if method else directory / name
            restored = quantize(read_sequence(sorted(folder.glob("*.png"))))
            target = quantize(truth[name][: len(restored)])
            clip_scores[name] = {
                "psnr_rgb": float(np.mean([psnr(restored[index], target[index], False) for index in range(len(restored))])),
                "ssim_rgb": float(np.mean([ssim(restored[index], target[index], False) for index in range(len(restored))])),
                "lpips": float(np.mean([scorer(restored[index], target[index]) for index in range(len(restored))])),
                "tlpips": scorer.temporal(restored, target),
                "frames": int(len(restored)),
            }
            restored_frames.append(restored)
        pooled[label] = pool_distance(_features(inception, restored_frames), truth_features)
        if method:
            for name, scores in clip_scores.items():
                payload.setdefault(name, {})[method] = scores
        else:
            payload.update(clip_scores)
        print(label, pooled[label], flush=True)
    pooled_path.parent.mkdir(parents=True, exist_ok=True)
    pooled_path.write_text(json.dumps(pooled, indent=2) + "\n")
    for path, payload in grouped.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")


def _features(inception: InceptionPool, sequences: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([inception(quantize(frames)) for frames in sequences])


def _ready_methods(names: list[str]) -> list[tuple[str, Path, str | None]]:
    catalog = (
        ("Vid4 bicubic", OUTPUT / "part1" / "frames", "bicubic"),
        ("Vid4 Lanczos", OUTPUT / "part1" / "frames", "lanczos"),
        ("Vid4 EDSR", OUTPUT / "part1" / "frames", "edsr"),
        ("Vid4 temporal average", OUTPUT / "part1" / "frames", "temporal_average"),
        ("Vid4 unsharp", OUTPUT / "part1" / "frames", "temporal_unsharp"),
        ("Vid4 EDVR-L", OUTPUT / "edvr" / "frames", None),
        ("Vid4 BasicVSR++", OUTPUT / "basicvsrpp" / "frames", None),
        ("Vid4 TecoGAN, not released", OUTPUT / "tecogan_not_official" / "frames", None),
        ("Vid4 STAR", OUTPUT / "star" / "frames", None),
        ("Vid4 hybrid", OUTPUT / "hybrid" / "frames", None),
    )
    ready = []
    for label, directory, method in catalog:
        if all(_matches(directory, method, name) for name in names):
            ready.append((label, directory, method))
    return ready


def _matches(directory: Path, method: str | None, name: str) -> bool:
    folder = directory / method / name if method else directory / name
    low = _frames(DATA / "BIx4" / name)
    return _complete(folder, len(low))


def _frames(folder: Path) -> list[Path]:
    frames = sorted(folder.glob("*.png"))
    if not frames:
        raise SystemExit(f"no png frames in {folder}")
    return frames


def _complete(folder: Path, count: int) -> bool:
    return folder.is_dir() and len(list(folder.glob("*.png"))) == count


if __name__ == "__main__":
    main()
