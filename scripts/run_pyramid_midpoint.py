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
    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)
    model.vae.enable_tiling()
    output_root = ROOT / "outputs" / "part3" / "pyramid_midpoint"
    summary = {
        "method": "course conditional sample from the sampler midpoint",
        "official_video_sr_api": False,
        "clips": {},
    }
    for clip in CLIPS:
        low_dir = ROOT / "data" / "reds" / "train" / "train_sharp_bicubic" / "X4" / clip
        low = [upsample(_read(path), 4, "bicubic") for path in sorted(low_dir.glob("*.png"))]
        restored, start = _restore_clip(model, low, device)
        write_sequence(output_root / "frames" / clip, restored)
        write_video(output_root / f"{clip}.mp4", restored)
        summary["clips"][clip] = {"frames": int(len(restored)), "start_index": start}
        print(clip, "done", start, flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "docs" / "results" / "pyramid_midpoint.json").write_text(json.dumps(summary, indent=2) + "\n")


def _read(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def _restore_clip(model, frames: list[np.ndarray], device: torch.device) -> tuple[np.ndarray, int]:
    outputs = []
    start_index = None
    for frame in frames:
        image, start_index = _restore_frame(model, frame, device)
        outputs.append(image)
    return np.stack(outputs), int(start_index)


def _restore_frame(model, frame: np.ndarray, device: torch.device) -> tuple[np.ndarray, int]:
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
    latent = (latent - model.vae_shift_factor) * model.vae_scale_factor
    stage = len(model.stages) - 1
    model.scheduler.set_timesteps(20, stage, device=device)
    start = midpoint_index(int(model.scheduler.timesteps.shape[0]))
    sigma = model.scheduler.sigmas[start]
    noise = torch.randn_like(latent)
    noisy = flow_mix(latent, noise, sigma.to(latent.dtype))
    model.scheduler._step_index = start
    model._guidance_scale = 7.0
    model._video_guidance_scale = 7.0
    prompt_embeds, prompt_attention_mask, pooled = model.text_encoder(PROMPT, device)
    negative_embeds, negative_mask, negative_pooled = model.text_encoder("", device)
    prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
    pooled = torch.cat([negative_pooled, pooled], dim=0)
    prompt_attention_mask = torch.cat([negative_mask, prompt_attention_mask], dim=0)
    latents = noisy
    for timestep in model.scheduler.timesteps[start:]:
        model_input = torch.cat([latents, latents], dim=0)
        step = timestep.expand(model_input.shape[0]).to(model_input.dtype)
        prediction = model.dit(
            sample=[[model_input]],
            timestep_ratio=step,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled,
        )[0]
        uncond, cond = prediction.chunk(2)
        guided = uncond + model.guidance_scale * (cond - uncond)
        latents = model.scheduler.step(guided, step[:1], latents).prev_sample
    decoded = model.decode_latent(latents, save_memory=True)
    array = np.asarray(decoded[0].resize((frame.shape[1], frame.shape[0]))).astype(np.float32)
    return array, start


if __name__ == "__main__":
    main()
