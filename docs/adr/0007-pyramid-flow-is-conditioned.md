---
status: superseded by ADR-0019
---

# Pyramid Flow counts only as conditioned restoration

Pyramid Flow’s released model is a text-to-video generator. Direction A does not train an upscaler. A result enters the metric table only when the official weights integrate the ODE from an intermediate noisy state of the bicubic-upscaled clip, and the output is scaled back to the ground-truth resolution. Sampling steps are reported. If the released code cannot start from that intermediate state, Direction A is a failure case. Unconditioned text-to-video samples are evidence in that write-up, not restored videos.
