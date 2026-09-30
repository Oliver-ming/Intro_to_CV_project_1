---
status: superseded by ADR-0015
---

# Every method sees the same bicubic input

Benchmark comparisons use one shared low-resolution input: 4× bicubic downsampling of the ground-truth video. Part 1 and Part 2 prefer official weights trained to invert that same degradation. A checkpoint trained on another degradation is still run on this input, and the table names its training degradation, so a PSNR gap is not read as a pure architecture comparison. STAR and Pyramid Flow use the same input; a lower PSNR there is reported as the accuracy–perception trade-off.
