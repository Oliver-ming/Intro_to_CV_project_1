# Download sources

These archives are stored under `data/zips/` and are not committed.

| Archive | URL | Bytes | Role |
| --- | --- | --- | --- |
| `train_sharp.zip` | `https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip` | 34261573976 | REDS ground truth, including REDS4 |
| `train_sharp_bicubic.zip` | `https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp_bicubic.zip` | 2652551733 | Official REDS 4× bicubic low-resolution frames |

The MIT host `http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip` and `vimeo_test_clean.zip` return HTTP 403 from this machine. The Vimeo subset is not downloaded until a reachable official copy is found.

| `edsr_baseline_x4-6b446fab.pt` | `https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt` | 6081831 | Official EDSR baseline, 16 residual blocks, 64 features, DIV2K 4× |

BasicSR is pinned at `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`. Its REDS4 EDVR-L test config is `options/test/EDVR/test_EDVR_L_x4_SR_REDS.yml`: 5-frame window, RGB PSNR with `crop_border: 0`. The matching weight is `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` (Google Drive id `127KXEjlCwfoPC1aXyDkluNwr9elwyHNb`). The Vimeo weight is `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth` (id `1aVR3lkX6ItCphNLcT7F5bbbC484h4Qqy`).
