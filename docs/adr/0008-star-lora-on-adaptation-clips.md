---
status: superseded by ADR-0013
---

# STAR has an official row and a disjoint LoRA row

Direction B always runs the released STAR weights on the shared 4× bicubic input, with its real-world training degradation named in the table. A second row trains LoRA at rank 8 on adaptation clips only — about 20 Vimeo-90K training septuplets, or REDS training clips that are not REDS4. That row is judged by whether tLPIPS drops, not by PSNR. If the released code has no trainable LoRA interface, the official row remains and LoRA is a failure case.
