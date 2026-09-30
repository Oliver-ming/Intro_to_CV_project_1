# Unsharp masking is a course variant of the temporal average

Unsharp masking is applied only to the temporal-average output and is reported as an extra variant of that row, not as its own method. The formula is y = x + 1.0 * (x - GaussianBlur(x, sigma=1 pixel)), then clamped to the valid image range. Sigma 1 and amount 1.0 are course defaults. EDSR, EDVR, BasicVSR, and TecoGAN do not specify them.
