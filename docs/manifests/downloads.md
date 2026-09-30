# Download sources

These archives are stored under `data/zips/` and are not committed.

| Archive | URL | Bytes | Role |
| --- | --- | --- | --- |
| `train_sharp.zip` | `https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip` | 34261573976 | REDS ground truth, including REDS4 |
| `train_sharp_bicubic.zip` | `https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp_bicubic.zip` | 2652551733 | Official REDS 4× bicubic low-resolution frames |

The MIT host `http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip` and `vimeo_test_clean.zip` return HTTP 403 from this machine. The Vimeo subset is not downloaded until a reachable official copy is found.

| `edsr_baseline_x4-6b446fab.pt` | `https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt` | 6081831 | Official EDSR baseline, 16 residual blocks, 64 features, DIV2K 4× |

BasicSR is pinned at `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`. Its REDS4 EDVR-L test config is `options/test/EDVR/test_EDVR_L_x4_SR_REDS.yml`: 5-frame window, RGB PSNR with `crop_border: 0`. The matching weight is `EDVR_L_x4_SR_REDS_official-9f5f5039.pth` (Google Drive id `127KXEjlCwfoPC1aXyDkluNwr9elwyHNb`). The Vimeo weight is `EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth` (id `1aVR3lkX6ItCphNLcT7F5bbbC484h4Qqy`, 82850094 bytes, 264 tensors). It is the 7-frame model and is not loaded into the 5-frame REDS architecture. The official Vimeo low-resolution frames are still missing because the MIT archive returns HTTP 403.

| `tecogan_official_model.zip` | `https://ge.in.tum.de/download/data/TecoGAN/model.zip` | 11288778 | Released TecoGAN generator. TensorFlow 1 checkpoint, not a PyTorch state dict. |
| `face_detection_yunet_2023mar.onnx` | `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` | 232589 | YuNet weights for OpenCV `FaceDetectorYN`. Course face prior, not part of a VSR paper. |
| `TecoGAN_BI4x_Vimeo_pytorch_NOT_OFFICIAL.pth` | Google Drive id `1ie1F7wJcO4mhNWK8nPX7F0LgOoPzCwEu` | — | PyTorch reimplementation trained on Vimeo bicubic 4×. Not the released TecoGAN checkpoint. |
| `star_light_deg.pt` | `https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/light_deg.pt` | 8165233536 | Official STAR I2VGen-XL weights for light degradation. Not bicubic training. |
| `basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth` | `https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth` | 29349005 | Official BasicVSR++ REDS4 bicubic checkpoint, 64 mid channels, 7 blocks. SpyNet and upsampling names are rewritten to the pinned BasicSR modules. |
| `frozen_east_text_detection.pb` | `https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb` | — | EAST text detector used only as a course prior. A missing file leaves the text prior empty. |
