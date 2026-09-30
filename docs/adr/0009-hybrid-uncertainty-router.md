---
status: superseded by ADR-0014
---

# The hybrid routes by a face/text mask and pixel disagreement

Direction C does not train a router. BasicVSR++ supplies the fidelity frame, using a 4× bicubic checkpoint matched to the benchmark when one exists. The generative frame is the official STAR inference result, so this direction does not depend on Pyramid Flow or on LoRA. Face and text pixels, detected on the bicubic-upscaled frame, always come from BasicVSR++. Every other pixel is a soft blend: uncertainty is the per-frame-normalized mean absolute RGB difference between the two restorations, and a higher value gives STAR more weight. If STAR does not run, Direction C is a failure case; BasicVSR++ alone is not the hybrid.
