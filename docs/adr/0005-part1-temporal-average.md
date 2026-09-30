---
status: superseded by ADR-0016
---

# Part 1 temporal fusion averages neighboring frames

The assignment text says "spatial neighbors" under a temporal-fusion heading. We read that as time, not as blurring inside one frame. After bicubic upscaling, each frame is averaged with two frames on either side using weights [1, 4, 6, 4, 1] / 16, renormalized where the window runs off the clip. Bicubic and Lanczos are our own resamplers, checked against OpenCV. EDSR is frame-wise inference with the official DIV2K 4× bicubic weights and is not trained on the mandatory path. Unsharp masking is an extra version of the temporal average, not its own row.
