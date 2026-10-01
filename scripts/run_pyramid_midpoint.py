"""Condition Pyramid Flow on a bicubic-upscaled REDS4 clip.

The released generate() starts from pure noise. This runner encodes the
bicubic-upscaled frames with the released VAE, mixes in noise at the middle
index of the sampler's own last-stage schedule, and integrates from that
index with scheduler.step. The report must call this a course conditional
sample, not an official video super-resolution interface.
"""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path

if "IPython" not in sys.modules:
    _ipython = types.ModuleType("IPython")
    _ipython.embed = lambda *args, **kwargs: None
    _ipython.__spec__ = importlib.machinery.ModuleSpec("IPython", None)
    sys.modules["IPython"] = _ipython

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PYRAMID_ROOT = ROOT / "third_party" / "Pyramid-Flow"
sys.path.insert(0, str(PYRAMID_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from part1.io import write_sequence, write_video
from part1.resample import upsample

CLIPS = ("000", "011", "015", "020")
MODEL = ROOT / "weights" / "pyramid-flow-sd3"
HEIGHT = 384
WIDTH = 640
PROMPT = "a realistic driving video frame"


def midpoint_index(num_steps: int) -> int:
    if num_steps < 2:
        raise ValueError("the sampler needs at least two steps to have a middle")
    return num_steps // 2


def flow_mix(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    while sigma.ndim < clean.ndim:
        sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
    return sigma * noise + (1.0 - sigma) * clean


def main() -> None:
    dit = MODEL / "diffusion_transformer_384p" / "diffusion_pytorch_model.safetensors"
    vae = MODEL / "causal_video_vae" / "diffusion_pytorch_model.safetensors"
    if not dit.is_file() or not vae.is_file():
        raise SystemExit(f"Pyramid Flow 384p weights are not complete under {MODEL}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from pyramid_dit import PyramidDiTForVideoGeneration

    model = PyramidDiTForVideoGeneration(
        str(MODEL),
        "bf16",
        model_name="pyramid_mmdit",
        model_variant="diffusion_transformer_384p",
        load_text_encoder=True,
        load_vae=True,
    )
    model.vae.enable_tiling()
    model.text_encoder.to("cpu")
    model.dit.to("cpu")
    model.vae.to("cpu")
    prompts = _prompts(model, device)
    output_root = ROOT / "outputs" / "part3" / "pyramid_midpoint"
    summary = {
        "method": "course conditional sample from the sampler midpoint",
        "official_video_sr_api": False,
        "text_encoder": "cpu",
        "classifier_free_guidance": "two sequential forwards",
        "clips": {},
    }
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        low = [upsample(_read(path), 4, "bicubic") for path in sorted(low_dir.glob("*.png"))]
        frame_dir = output_root / "frames" / clip
        restored, start = _restore_clip(model, low, device, prompts, frame_dir)
        write_video(output_root / f"{clip}.mp4", restored)
        summary["clips"][clip] = {"frames": int(len(restored)), "start_index": start}
        print(clip, "done", start, flush=True)
        (ROOT / "docs" / "results" / "pyramid_midpoint.json").write_text(json.dumps(summary, indent=2) + "\n")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "docs" / "results" / "pyramid_midpoint.json").write_text(json.dumps(summary, indent=2) + "\n")


def _read(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def _prompts(model, device: torch.device) -> dict[str, torch.Tensor]:
    positive = model.text_encoder(PROMPT, torch.device("cpu"))
    negative = model.text_encoder("", torch.device("cpu"))
    model.text_encoder.to("cpu")
    keys = ("embeds", "mask", "pooled")
    packed = {}
    for name, value in zip(keys, positive):
        packed[name] = value.to(device=device)
    for name, value in zip(keys, negative):
        packed["negative_" + name] = value.to(device=device)
    return packed


def _restore_clip(
    model,
    frames: list[np.ndarray],
    device: torch.device,
    prompts: dict[str, torch.Tensor],
    frame_dir: Path,
) -> tuple[np.ndarray, int]:
    from part1.io import write_frame

    frame_dir.mkdir(parents=True, exist_ok=True)
    pending = [index for index in range(len(frames)) if not (frame_dir / f"{index:08d}.png").is_file()]
    start = midpoint_index(20)
    if pending:
        model.dit.to("cpu")
        model.vae.to(device)
        torch.cuda.empty_cache()
        encoded = {index: _encode(model, frames[index], device).cpu() for index in pending}
        model.vae.to("cpu")
        torch.cuda.empty_cache()
        model.dit.to(device)
        cache = frame_dir / "latents"
        cache.mkdir(parents=True, exist_ok=True)
        for index in pending:
            cache_path = cache / f"{index:08d}.pt"
            if cache_path.is_file():
                continue
            latent, start = _denoise_latent(model, encoded[index].to(device), device, prompts)
            torch.save(latent.cpu(), cache_path)
            print("denoised", index + 1, "/", len(frames), flush=True)
        model.dit.to("cpu")
        torch.cuda.empty_cache()
        model.vae.to(device)
        for index in pending:
            latent = torch.load(cache / f"{index:08d}.pt", map_location=device, weights_only=True)
            image = _decode(model, latent, frames[index].shape)
            write_frame(frame_dir / f"{index:08d}.png", image)
            print("frame", index + 1, "/", len(frames), flush=True)
        model.vae.to("cpu")
        torch.cuda.empty_cache()
    return np.stack([_read(frame_dir / f"{index:08d}.png") for index in range(len(frames))]), start


def _encode(model, frame: np.ndarray, device: torch.device) -> torch.Tensor:
    import torch.nn.functional as functional

    resized = functional.interpolate(
        torch.from_numpy(np.transpose(frame, (2, 0, 1))).unsqueeze(0).unsqueeze(2),
        size=(1, HEIGHT, WIDTH),
        mode="trilinear",
        align_corners=False,
    )
    video = resized.to(device=device, dtype=model.vae.dtype) / 127.5 - 1.0
    with torch.no_grad():
        latent = model.vae.encode(video).latent_dist.sample()
    return (latent - model.vae_shift_factor) * model.vae_scale_factor


def _denoise_latent(
    model,
    latent: torch.Tensor,
    device: torch.device,
    prompts: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    stage = len(model.stages) - 1
    model.scheduler.set_timesteps(20, stage, device=device)
    start = midpoint_index(int(model.scheduler.timesteps.shape[0]))
    sigma = model.scheduler.sigmas[start]
    noise = torch.randn_like(latent)
    latents = flow_mix(latent, noise, sigma.to(latent.dtype))
    model.scheduler._step_index = start
    model._guidance_scale = 7.0
    model._video_guidance_scale = 7.0
    with torch.no_grad():
        for timestep in model.scheduler.timesteps[start:]:
            uncond = _denoise(model, latents, timestep, prompts, negative=True)
            cond = _denoise(model, latents, timestep, prompts, negative=False)
            guided = uncond + model.guidance_scale * (cond - uncond)
            step = timestep.expand(1).to(latents.dtype)
            latents = model.scheduler.step(guided, step, latents).prev_sample
    return latents, start


def _decode(model, latents: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    with torch.no_grad():
        decoded = model.decode_latent(latents, save_memory=True)
    return np.asarray(decoded[0].resize((shape[1], shape[0]))).astype(np.float32)


def _denoise(model, latents: torch.Tensor, timestep: torch.Tensor, prompts: dict[str, torch.Tensor], negative: bool) -> torch.Tensor:
    prefix = "negative_" if negative else ""
    embeds = prompts[prefix + "embeds"].to(dtype=latents.dtype)
    mask = prompts[prefix + "mask"]
    pooled = prompts[prefix + "pooled"].to(dtype=latents.dtype)
    step = timestep.expand(latents.shape[0]).to(latents.dtype)
    prediction = model.dit(
        sample=[[latents]],
        timestep_ratio=step,
        encoder_hidden_states=embeds,
        encoder_attention_mask=mask,
        pooled_projections=pooled,
    )[0]
    return prediction


if __name__ == "__main__":
    main()
