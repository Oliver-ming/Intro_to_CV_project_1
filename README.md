# Video super-resolution

Part 1 restores each official low-resolution frame with bicubic, Lanczos, the EDSR baseline, a five-frame temporal average, and an unsharp variant of that average.

## Environment

Use `../env_cv`. From this directory:

```bash
../env_cv/bin/python -m pip install -r requirements.txt
```

## Data

Official REDS archives go in `data/zips/` and are not committed. See `docs/manifests/downloads.md`. After `train_sharp.zip` and `train_sharp_bicubic.zip` finish, REDS4 clips `000`, `011`, `015`, and `020` are extracted under `data/reds/`. Resume a dropped ground-truth download with `../env_cv/bin/python scripts/fetch_train_sharp.py`. It keeps using the stable Hugging Face URL.

REDS4 ground truth is extracted. The MIT Vimeo test packages still return HTTP 403, so the 30-clip subset is not scored. `scripts/fetch_vimeo_sr_test.py` downloads `vimeo_super_resolution_test.zip` and extracts the frozen 30 clips only after the zip is complete.

## Run Part 1

```bash
../env_cv/bin/python scripts/run_part1.py
```

Restored videos and `metrics.json` are written to `outputs/part1/`. REDS4 PSNR and SSIM are on RGB with no border crop. Frames are quantized to 8-bit before scoring. tLPIPS is not multiplied by 100. Each clip json stores a Fréchet distance and the frame count. The report table uses one Fréchet distance on all 400 frames, written by `scripts/pool_reds_fid.py`.

## EDVR

```bash
PYTHONPATH=third_party/BasicSR ../env_cv/bin/python scripts/run_edvr.py
```

This needs `weights/EDVR_L_x4_SR_REDS_official-9f5f5039.pth` and the extracted REDS4 low-resolution frames. It uses the pinned BasicSR EDVR-L: five frames and `reflection_circle` padding.

The Vimeo subset uses a different checkpoint, the 7-frame `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth`:

```bash
PYTHONPATH=third_party/BasicSR ../env_cv/bin/python scripts/run_edvr_vimeo.py
```

It reads `docs/manifests/vimeo-subset.txt` and scores PSNR and SSIM on Y. It does not run until `data/vimeo` contains the official low-resolution frames.

## BasicVSR++ and the disagreement-map hybrid

```bash
CUDA_VISIBLE_DEVICES=4 ../env_cv/bin/python scripts/run_basicvsrpp.py
CUDA_VISIBLE_DEVICES=5 ../env_cv/bin/python scripts/run_hybrid.py
```

BasicVSR++ is the fidelity model. It uses the REDS4 bicubic checkpoint, 64 mid channels and 7 blocks. The hybrid reads its frames and the official STAR frames. Face and text boxes on the bicubic-upscaled low-resolution frame come from BasicVSR++. Other pixels use the per-frame disagreement map. After `train_sharp` is extracted, `scripts/score_ready.py` scores every method that has saved frames: RGB PSNR and SSIM with no border crop, AlexNet LPIPS, tLPIPS without the ×100 scale, and small-sample FID with the frame count.

## TecoGAN

The released archive is TensorFlow 1. `scripts/inspect_tecogan_official.py` records that it does not load here. A separate PyTorch checkpoint, trained on Vimeo bicubic 4× and labeled not official, is run with:

```bash
CUDA_VISIBLE_DEVICES=2 ../env_cv/bin/python scripts/run_tecogan_not_official.py
```

## STAR

```bash
CUDA_VISIBLE_DEVICES=3 ../env_cv/bin/python scripts/run_star_official.py
```

This needs the complete `weights/star_light_deg.pt` (8165233536 bytes) and `ffmpeg`. The checkpoint is the official I2VGen-XL light-degradation model. Its training degradation is real-world, not bicubic. On a 24 GB GPU, windows of two or more frames at 720×1280 do not fit, so the script uses one frame at a time. The published STAR requirements pin an older PyTorch and xformers 0.0.21. This environment keeps torch 2.11, installs `requirements-star.txt`, and uses scaled dot-product attention for the released U-Net's bias-free call. The course LoRA row is not run: `train_sr.py` has no LoRA parameters.

## Report

The CVPR report is `paper/main.tex` and `paper/main.pdf`. The REDS4 table is filled from `docs/results/reds4_means.json`. `cvpr.sty` and `ieeenat_fullname.bst` are linked into `paper/` from the author kit. From `paper/`:

```bash
tectonic main.tex
```

If `pdflatex` is installed instead, `TEXINPUTS=author-kit-main:` is enough and the links are unnecessary.
