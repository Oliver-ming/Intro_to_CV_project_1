"""Restore four filmed clips. There is no ground truth.

Each clip is one scene from CV资料/Project1. The filmed frames are
1920×1080. They are area-resized to 1280×720, then downsampled with
OpenCV's 4× bicubic kernel. Official STAR does not fit a 1920×1080
frame in 24 GB. That input is not an official benchmark release, and
these clips are not scored.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from part1.edsr import load_edsr
from part1.io import read_sequence, write_frame, write_sequence, write_video
from part1.restore import restore_part1

SOURCE_ROOT = ROOT.parent / "CV资料" / "Project1"
SOURCES = {
    "face_text": SOURCE_ROOT / "人物与广告牌" / "人与广告牌1.MP4",
    "grass": SOURCE_ROOT / "灌木" / "DJI_20261004135926_0022_D.MP4",
    "water": SOURCE_ROOT / "水面" / "DJI_20261004135504_0019_D.MP4",
    "lowlight": SOURCE_ROOT / "昏暗室内" / "DJI_20261004140823_0024_D.MP4",
}
DATA = ROOT / "data" / "wild"
OUTPUT = ROOT / "outputs" / "wild"
PART1_METHODS = ("bicubic", "lanczos", "edsr", "temporal_average", "temporal_unsharp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--star", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    names = list(SOURCES)
    _prepare(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.star:
        _star(names)
        _hybrid(names, texture=False)
        _hybrid(names, texture=True)
    elif args.hybrid:
        _hybrid(names, texture=False)
        _hybrid(names, texture=True)
    else:
        _part1(names, device)
        _edvr(names, device)
        _basicvsr(names, device)
        _tecogan(names)
    _figure()


def _prepare(names: list[str]) -> None:
    manifest = {}
    for name in names:
        source = SOURCES[name]
        if not source.is_file():
            raise SystemExit(f"missing filmed clip {source}")
        filmed = DATA / name / "filmed"
        view = DATA / name / "view"
        low = DATA / name / "bicubic_x4"
        if not filmed.is_dir() or not any(filmed.glob("*.png")):
            filmed.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source),
                    "-vsync",
                    "0",
                    "-start_number",
                    "0",
                    str(filmed / "%08d.png"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        frames = sorted(filmed.glob("*.png"))
        if not _complete(view, len(frames)):
            view.mkdir(parents=True, exist_ok=True)
            for path in frames:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                resized = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(view / path.name), resized):
                    raise OSError(view / path.name)
        if not _complete(low, len(frames)):
            low.mkdir(parents=True, exist_ok=True)
            for path in frames:
                bgr = cv2.imread(str(view / path.name), cv2.IMREAD_COLOR)
                small = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_CUBIC)
                if not cv2.imwrite(str(low / path.name), small):
                    raise OSError(low / path.name)
        manifest[name] = {
            "source": str(source.relative_to(ROOT.parent)),
            "frames": len(frames),
            "filmed_wh": [1920, 1080],
            "view_wh": [1280, 720],
            "low_wh": [320, 180],
            "degradation": "area resize to 1280x720, then OpenCV INTER_CUBIC 4x",
            "scored": False,
        }
        print(name, "prepared", len(frames), flush=True)
    destination = ROOT / "docs" / "results" / "wild_sources.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _part1(names: list[str], device: torch.device) -> None:
    checkpoint = ROOT / "weights" / "edsr_baseline_x4-6b446fab.pt"
    edsr = load_edsr(checkpoint, device) if checkpoint.is_file() and checkpoint.stat().st_size == 6081831 else None
    for name in names:
        low = read_sequence(_frames(DATA / name / "bicubic_x4"))
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
        print(name, "part1", flush=True)


def _edvr(names: list[str], device: torch.device) -> None:
    import run_edvr

    model = None
    for name in names:
        low = read_sequence(_frames(DATA / name / "bicubic_x4"))
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
        low = read_sequence(_frames(DATA / name / "bicubic_x4"))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    for name in names:
        low = read_sequence(_frames(DATA / name / "bicubic_x4"))
        directory = OUTPUT / "tecogan_not_official" / "frames" / name
        if _complete(directory, len(low)):
            print(name, "tecogan reused", flush=True)
            continue
        if model is None:
            model = FRNet(in_nc=3, out_nc=3, nf=64, nb=10, degradation="BI", scale=4)
            state = torch.load(tecogan.WEIGHT, map_location=device, weights_only=False)
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
            write_frame(frame_dir / f"{index:08d}.png", image.detach().cpu().numpy())
        return original_save(video, save_dir, file_name, fps)

    original_save = inference_sr.save_video
    inference_sr.save_video = _save_video
    model = None
    for name in names:
        low = _frames(DATA / name / "bicubic_x4")
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


def _hybrid(names: list[str], texture: bool) -> None:
    from part1.resample import upsample
    from part3.detectors import certain_mask, texture_mask
    from part3.hybrid import blend

    fidelity_root = OUTPUT / "basicvsrpp" / "frames"
    generative_root = OUTPUT / "star" / "frames"
    output_root = OUTPUT / ("hybrid_texture" if texture else "hybrid")
    for name in names:
        low_paths = _frames(DATA / name / "bicubic_x4")
        fidelity_paths = sorted((fidelity_root / name).glob("*.png"))
        generative_paths = sorted((generative_root / name).glob("*.png"))
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
        print(name, "hybrid", "texture" if texture else "disagreement", flush=True)


def _figure() -> None:
    from part1.io import read_frame

    clip = "face_text"
    index = 80
    panels = [
        ("Filmed", DATA / clip / "view"),
        ("Bicubic", OUTPUT / "part1" / "frames" / "bicubic" / clip),
        ("EDVR-L", OUTPUT / "edvr" / "frames" / clip),
        ("BasicVSR++", OUTPUT / "basicvsrpp" / "frames" / clip),
    ]
    star = OUTPUT / "star" / "frames" / clip
    if _complete(star, len(_frames(DATA / clip / "bicubic_x4"))):
        panels.append(("STAR", star))
    images = []
    for name, directory in panels:
        frames = sorted(directory.glob("*.png"))
        if len(frames) <= index:
            print("figure waiting", name, flush=True)
            return
        images.append((name, read_frame(frames[index])))
    crop = (200, 480)
    full = [_label(_fit(image, 360), name) for name, image in images]
    crops = [_label(image[crop[0] : crop[0] + 180, crop[1] : crop[1] + 180], name) for name, image in images]
    gap = np.full((full[0].shape[0], 8, 3), 255, np.uint8)
    top = _join(full, gap)
    bottom = _join(crops, np.full((crops[0].shape[0], 8, 3), 255, np.uint8))
    pad = max(0, (top.shape[1] - bottom.shape[1]) // 2)
    bottom = cv2.copyMakeBorder(bottom, 0, 0, pad, max(0, top.shape[1] - bottom.shape[1] - pad), cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if bottom.shape[1] != top.shape[1]:
        bottom = bottom[:, : top.shape[1]]
    figure = np.concatenate([top, np.full((12, top.shape[1], 3), 255, np.uint8), bottom], axis=0)
    destination = ROOT / "paper" / "figures" / "wild_face_text.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(np.rint(figure), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(destination), bgr):
        raise OSError(destination)
    print(destination, figure.shape, "crop", crop, flush=True)


def _crop(image: np.ndarray) -> tuple[int, int]:
    gray = cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    response = cv2.Laplacian(gray, cv2.CV_32F)
    best = (-1.0, 0, 0)
    for y in range(0, gray.shape[0] - 180, 40):
        for x in range(0, gray.shape[1] - 180, 40):
            score = float(response[y : y + 180, x : x + 180].var())
            if score > best[0]:
                best = (score, y, x)
    return best[1], best[2]


def _fit(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(np.clip(np.rint(image), 0, 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, name: str) -> np.ndarray:
    canvas = np.clip(np.rint(image), 0, 255).astype(np.uint8).copy()
    cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _join(images: list[np.ndarray], gap: np.ndarray) -> np.ndarray:
    rows = []
    for image in images:
        if rows:
            rows.append(gap)
        rows.append(image)
    return np.concatenate(rows, axis=1)


def _frames(folder: Path) -> list[Path]:
    frames = sorted(folder.glob("*.png"))
    if not frames:
        raise SystemExit(f"no png frames in {folder}")
    return frames


def _complete(folder: Path, count: int) -> bool:
    return folder.is_dir() and len(list(folder.glob("*.png"))) >= count and count > 0


if __name__ == "__main__":
    main()
