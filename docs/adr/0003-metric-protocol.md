---
status: superseded by ADR-0010
---

# Full-reference metrics follow the video super-resolution protocol

PSNR and SSIM are computed on the Y channel after a 4-pixel border crop, averaged over frames and then over clips. LPIPS uses AlexNet on RGB. tLPIPS is LPIPS between consecutive-frame differences of the restored video and the ground-truth video. FID pools every restored frame against every ground-truth frame in that benchmark set, and the report states the frame count so the number is not compared with a full-dataset paper FID. These scores apply only to benchmark clips. A wild video, when one exists, is judged visually.
