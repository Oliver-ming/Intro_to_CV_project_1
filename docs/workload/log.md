# Workload log

Append one entry per change. Each entry states the date, what changed, why, and which files were involved.

## 2026-10-01

- Recorded the agreed project language and the decisions from the planning session: deliverables, Part 3 depth, deferred wild video, REDS4 plus a frozen 30-clip Vimeo subset, shared official bicubic input, paper-aligned metrics, Part 1 temporal average, Part 2 PyTorch checkpoints, conditioned Pyramid Flow, STAR LoRA on adaptation clips, the uncertainty router, and this repository layout.
- Why: later edits need a single place that says which decision they follow.
- Files: `CONTEXT.md`, `docs/adr/0001` through `docs/adr/0012`, `docs/workload/log.md`, `third_party/PINS.md`, `.gitignore`.

## 2026-10-01

- Relabeled the STAR LoRA row as a course experiment. Official inference stays. Adaptation clips are 20 Vimeo-90K training septuplets, not REDS.
- Why: STAR trains ControlNet and LIEM on OpenVid, not LoRA at rank 8.
- Files: `docs/adr/0008-star-lora-on-adaptation-clips.md`, `docs/adr/0013-star-lora-is-a-course-experiment.md`, `CONTEXT.md`, `docs/workload/log.md`.

## 2026-10-01

- Renamed the hybrid weight from uncertainty to a disagreement map. The blend itself is unchanged: face and text pixels come from BasicVSR++, and other pixels use the per-frame-normalized absolute difference with STAR.
- Why: that map is not an uncertainty estimator in EDVR, BasicVSR++, STAR, or UPSR.
- Files: `docs/adr/0009-hybrid-uncertainty-router.md`, `docs/adr/0014-disagreement-map-is-a-course-hybrid.md`, `CONTEXT.md`, `docs/workload/log.md`.

## 2026-10-01

- Reworded the shared input as the official 4× bicubic low-resolution release, not a downsample recomputed from the ground-truth video.
- Why: ADR-0002 still said we downsample the ground truth, which conflicts with ADR-0011 and with the REDS and Vimeo releases used by EDVR and BasicVSR.
- Files: `docs/adr/0002-shared-bicubic-input.md`, `docs/adr/0015-shared-input-is-the-official-release.md`, `docs/workload/log.md`.

## 2026-10-01

- Labeled the Part 1 weights [1, 4, 6, 4, 1] / 16 as a course binomial baseline. The window and the EDSR official-weight row are unchanged.
- Why: EDSR, EDVR, BasicVSR, and TecoGAN do not define that kernel.
- Files: `docs/adr/0005-part1-temporal-average.md`, `docs/adr/0016-temporal-average-is-a-course-baseline.md`, `docs/workload/log.md`.

## 2026-10-01

- Fixed the Vimeo subset draw: 30 septuplets, without replacement, from the official test list in published order, seed 0. The list will live in `docs/manifests/vimeo-subset.txt`. Tables call it "Vimeo subset (30)".
- Why: EDVR and BasicVSR evaluate the full Vimeo-90K-T, so an unfrozen 30-clip draw would not be reproducible and must not be reported as that full set.
- Files: `docs/adr/0017-vimeo-subset-seed.md`, `CONTEXT.md`, `docs/workload/log.md`.

## 2026-10-01

- Set unsharp masking to y = x + 1.0 * (x - GaussianBlur(x, sigma=1)), applied only to the temporal average and clamped to the valid range.
- Why: the cited video super-resolution papers do not define this kernel. The constants are course defaults.
- Files: `docs/adr/0018-unsharp-is-a-course-default.md`, `docs/workload/log.md`.

## 2026-10-01

- Set Direction A’s start to the midpoint of whatever schedule the released Pyramid Flow sampler actually exposes, and excluded official image-to-video and text-to-video from the metric table.
- Why: Pyramid Flow does not define a super-resolution start time. Its image-to-video path animates a clean first frame instead of noising the whole clip. The midpoint is a course SDEdit-style operating point.
- Files: `docs/adr/0007-pyramid-flow-is-conditioned.md`, `docs/adr/0019-pyramid-flow-midpoint-is-a-course-start.md`, `CONTEXT.md`, `docs/workload/log.md`.

## 2026-10-01

- Chose course detectors for certain pixels: OpenCV `FaceDetectorYN` (YuNet) and `TextDetectionModel_EAST`, run on the bicubic-upscaled frame, with no box dilation. A missing detector leaves the disagreement-map blend in place.
- Why: EDVR, BasicVSR++, STAR, and TecoGAN do not specify a face or text detector for this hybrid.
- Files: `docs/adr/0020-course-face-and-text-detectors.md`, `docs/workload/log.md`.

## 2026-10-01

- Set the course LoRA budget to 200 steps, batch size 1, learning rate 1e-4, on one GPU. Only injected LoRA weights are trained.
- Why: STAR’s published budget is ControlNet and LIEM for 15K steps at batch size 8 and learning rate 5e-5. That budget is not this experiment.
- Files: `docs/adr/0021-star-lora-course-budget.md`, `docs/workload/log.md`.

## 2026-10-01

- Recorded the planning baseline in git. The author on each commit is supplied at commit time and is not written into git config.
- Why: the machine has no `user.name` or `user.email`, and the repository still needs a traceable history.
- Files: `CONTEXT.md`, `docs/adr/`, `docs/workload/log.md`, `third_party/PINS.md`, `.gitignore`, `Project 1- video super-resolution.pdf`.

## 2026-10-01

- Started the REDS download from the official Hugging Face release and added Part 1 upsamplers, the temporal average, and unsharp masking. Bicubic and Lanczos match OpenCV on float images within 1e-3.
- Why: the benchmark low-resolution frames have to be the official release, and Part 1 can be checked before those archives finish.
- Files: `src/part1/resample.py`, `src/part1/temporal.py`, `src/part1/__init__.py`, `tests/test_part1.py`, `requirements.txt`, `docs/manifests/downloads.md`, `docs/workload/log.md`.

## 2026-10-01

- Added PSNR, SSIM, tLPIPS, and Fréchet distance, plus the EDSR baseline architecture that matches the official 16-block DIV2K 4× checkpoint. REDS extraction waits until each zip reaches its published size.
- Why: metric numbers have to follow the paper split before any restored frames exist, and EDSR inference should load the released weights rather than a re-trained network.
- Files: `src/part1/metrics.py`, `src/part1/edsr.py`, `tests/test_part1.py`, `requirements.txt`, `docs/manifests/downloads.md`, `.gitignore`, `docs/workload/log.md`.

## 2026-10-01

- Added the Part 1 restore entry, video writer, and AlexNet LPIPS. A three-frame synthetic clip writes an mp4, and identical frames score about zero LPIPS.
- Why: REDS4 can be scored and packed into videos as soon as the official frames are extracted.
- Files: `src/part1/io.py`, `src/part1/perceptual.py`, `src/part1/restore.py`, `scripts/run_part1.py`, `tests/test_part1.py`, `requirements.txt`, `docs/workload/log.md`.

## 2026-10-01

- Loaded the official EDSR baseline checkpoint and confirmed a 16×16 input becomes 64×64. Added small-sample FID on Inception features, and a README for the Part 1 command. Scores use 8-bit quantized frames.
- Why: the released weights have to run before REDS4 is scored, and FID has to record the frame count instead of being compared with a paper's full-set number.
- Files: `src/part1/fid.py`, `scripts/run_part1.py`, `tests/test_part1.py`, `README.md`, `docs/workload/log.md`.

## 2026-10-01

- Cloned BasicSR at `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` for EDVR-L. The REDS4 test config uses five frames and RGB PSNR with no border crop. The official weights are the Google Drive files named in the download manifest.
- Why: Part 2 has to run the released bicubic EDVR-L checkpoint, not the blur-downsampling one.
- Files: `third_party/PINS.md`, `docs/manifests/downloads.md`, `docs/workload/log.md`.

## 2026-10-01

- Extracted the 400 official low-resolution REDS4 frames from `train_sharp_bicubic.zip`. Part 1 can write restored videos before the ground-truth archive finishes.
- Why: the bicubic archive matched its published size, and metrics still wait for `train_sharp`.
- Files: `scripts/run_part1.py`, `docs/workload/log.md`.

## 2026-10-01

- Added the EDVR-L REDS4 runner on pinned BasicSR: five-frame windows, reflection_circle padding, RGB input in [0, 1]. The padding check matches BasicSR's documented example `[4, 3, 0, 1, 2]`.
- Why: Part 2 has to call the released deformable-alignment model rather than a reimplementation.
- Files: `scripts/run_edvr.py`, `tests/test_part1.py`, `README.md`, `docs/workload/log.md`.

## 2026-10-01

- Added an inspector for the official TecoGAN archive from `https://ge.in.tum.de/download/data/TecoGAN/model.zip`. It records a failure when the archive is TensorFlow 1 and contains no PyTorch state dict.
- Why: the released model cannot be loaded in Python 3.10, and a substitute checkpoint must not be labeled as those weights.
- Files: `scripts/inspect_tecogan_official.py`, `docs/manifests/downloads.md`, `docs/workload/log.md`.

## 2026-10-01

- Added the disagreement-map blend and a BasicVSR++ runner. Certain pixels stay on the fidelity frame; a 64×64 random sequence upsamples to 256×256 on a GPU.
- Why: Direction C needs the blend before STAR exists, and BasicVSR++ is the fidelity model once its checkpoint is downloaded.
- Files: `src/part3/hybrid.py`, `src/part3/__init__.py`, `scripts/run_basicvsrpp.py`, `tests/test_part1.py`, `docs/workload/log.md`.

## 2026-10-01

- Froze the Vimeo subset (30 test clips) and the adaptation list (20 training clips) with numpy Generator seed 0, from BasicSR's official 7824-clip test list and training list.
- Why: the subset has to be fixed before the Vimeo frames themselves are reachable, and the tables must not be described as the full Vimeo-90K-T.
- Files: `scripts/freeze_vimeo_lists.py`, `docs/manifests/vimeo-subset.txt`, `docs/manifests/vimeo-adaptation.txt`, `tests/test_part1.py`, `docs/workload/log.md`.

## 2026-10-01

- Drafted the CVPR report in `paper/main.tex` with the agreed method and metric protocol. Experiment numbers are left blank because the REDS ground truth and the Vimeo frames are not on disk.
- Why: the report has to describe official bicubic evaluation and the course choices separately, and it must not invent scores.
- Files: `paper/main.tex`, `paper/refs.bib`, `README.md`, `docs/workload/log.md`.

## 2026-10-01

- Added the course face prior. A blank frame produces an empty mask with YuNet. EAST stays empty until `frozen_east_text_detection.pb` is present.
- Why: Direction C needs face and text boxes on the bicubic-upscaled frame, and a missing text model must not be treated as a failed hybrid.
- Files: `src/part3/detectors.py`, `tests/test_part1.py`, `docs/manifests/downloads.md`, `docs/workload/log.md`.

## 2026-10-01

- Inspected pinned Pyramid Flow `a012faa1`. `generate_i2v` takes one clean image; `generate` samples from noise. There is no whole-clip midpoint entry point, so Direction A is a failure case.
- Why: the course rule only counts a restoration that starts from a noisy bicubic-upscaled clip, and the released interface does not provide that.
- Files: `scripts/record_pyramid_flow_failure.py`, `docs/results/pyramid_flow_attempt.json`, `third_party/PINS.md`, `paper/main.tex`, `docs/workload/log.md`.

## 2026-10-01

- Opened the official TecoGAN archive (11288778 bytes). It contains only `TecoGAN.meta`, `TecoGAN.index`, and `TecoGAN.data-00000-of-00001`. Those weights were not loaded.
- Why: the released checkpoint is TensorFlow 1, and Python 3.10 cannot install that runtime.
- Files: `scripts/inspect_tecogan_official.py`, `docs/results/tecogan_official_attempt.json`, `docs/workload/log.md`.

## 2026-10-01

- Inspected pinned STAR `69b8bc53`. `train_sr.py` optimizes VideoControlNet and parameters named `local`. It does not train LoRA, so the course LoRA row is a failure. Official inference is not run because the released checkpoint is not on disk.
- Why: the course rule only runs LoRA when the released code exposes that interface.
- Files: `scripts/record_star_lora_failure.py`, `docs/results/star_lora_attempt.json`, `third_party/PINS.md`, `paper/main.tex`, `docs/workload/log.md`.

## 2026-10-01

- Queued the non-official PyTorch TecoGAN BI 4× weights until the EDVR Drive download exits. The filename says `NOT_OFFICIAL`.
- Why: the released checkpoint is TensorFlow 1 and cannot be loaded, and a substitute must stay visibly separate from it.
- Files: `scripts/fetch_tecogan_pytorch_not_official.py`, `docs/manifests/downloads.md`, `docs/workload/log.md`.

## 2026-10-01

- Added a REDS4 runner for the skycrapers PyTorch TecoGAN. It records `released_weights: false` and writes videos under `outputs/part2/tecogan_not_official`.
- Why: the released TensorFlow checkpoint cannot be loaded, and the substitute must stay labeled as a different model.
- Files: `scripts/run_tecogan_not_official.py`, `docs/workload/log.md`.
