# Video super-resolution

Part 1 restores each official low-resolution frame with bicubic, Lanczos, the EDSR baseline, a five-frame temporal average, and an unsharp variant of that average.

## Environment

Use `../env_cv`. From this directory:

```bash
../env_cv/bin/python -m pip install -r requirements.txt
```

## Data

Official REDS archives go in `data/zips/` and are not committed. See `docs/manifests/downloads.md`. After `train_sharp.zip` and `train_sharp_bicubic.zip` finish, REDS4 clips `000`, `011`, `015`, and `020` are extracted under `data/reds/`.

The MIT Vimeo test packages return HTTP 403 from this machine, so the 30-clip subset is not available yet.

## Run Part 1

```bash
../env_cv/bin/python scripts/run_part1.py
```

Restored videos and `metrics.json` are written to `outputs/part1/`. REDS4 PSNR and SSIM are on RGB with no border crop. Frames are quantized to 8-bit before scoring. tLPIPS is not multiplied by 100. FID is the small-sample Fréchet distance on that clip, and the json records the frame count.
