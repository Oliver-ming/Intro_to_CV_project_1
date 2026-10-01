# Video Super-Resolution

This project reconstructs a high-resolution video from a low-resolution one, recovering spatial detail while keeping the result stable across frames.

## Language

**Low-resolution video**:
A video whose frames do not contain the spatial detail required of the output.
_Avoid_: source clip, input footage

**Restored video**:
The high-resolution video reconstructed from a low-resolution video.
_Avoid_: trajectory, navigation result, upscaled frames

**Temporal consistency**:
Agreement of structure and appearance between neighboring frames of a restored video.
_Avoid_: trajectory smoothness, navigation stability

**Wild video**:
A low-resolution or low-light video filmed in the real world, with no ground-truth high-resolution counterpart.
_Avoid_: real sample, synthetic clip

**Benchmark clip**:
A video from an academic dataset that has a ground-truth high-resolution counterpart.
_Avoid_: wild video, sample footage

**REDS4**:
The REDS benchmark in this project: clips 000, 011, 015, and 020, held out from the REDS training split.
_Avoid_: REDS validation set, REDS test set

**Vid4**:
The extra benchmark in this project: calendar, city, foliage, and walk, with BasicSR's official BIx4 low-resolution frames and ground truth. It is the scored substitute for the Vimeo subset.
_Avoid_: Vimeo-90K, REDS4, wild video

**Vimeo subset**:
Thirty seven-frame clips drawn once, without replacement, from the official Vimeo-90K test list and then frozen. The MIT archive returns HTTP 403, so this subset is not scored.
_Avoid_: full Vimeo-90K, Vimeo-90K-T, Vid4

**Ground-truth video**:
The original high-resolution benchmark clip from which the low-resolution input is made.
_Avoid_: label, target

**Bicubic degradation**:
The synthetic degradation on benchmark clips. The low-resolution frames are the official 4× bicubic release, produced with MATLAB `imresize`. They are not recomputed with another kernel.
_Avoid_: OpenCV bicubic, blur-downsampling, real-world degradation

**Training degradation**:
The degradation a method was trained to invert. It may differ from the bicubic degradation of the shared input.
_Avoid_: test degradation, input degradation

**Full-reference metric**:
A score that compares a restored video with its ground-truth video. It applies only to benchmark clips.
_Avoid_: visual comparison, no-reference score

**tLPIPS**:
The absolute difference between the LPIPS of two consecutive restored frames and the LPIPS of the two corresponding ground-truth frames.
_Avoid_: LPIPS on frame-difference images, tLP100

**Temporal average**:
A restored frame formed by a weighted average of the current bicubic-upscaled frame and nearby frames in time.
_Avoid_: spatial smoothing, neighborhood filter

**Conditioned generation**:
A generative sample integrated from a midpoint noisy state of the whole bicubic-upscaled video.
_Avoid_: text-to-video sample, image-to-video animation, unconditioned sample

**Adaptation clip**:
A seven-frame clip from the Vimeo-90K training split, used only for the course LoRA experiment. It is not a REDS clip and not a member of the Vimeo subset.
_Avoid_: REDS training clip, benchmark clip, test clip

**Certain pixel**:
A pixel marked as a face or as text on the bicubic-upscaled frame. It is taken from the fidelity restoration.
_Avoid_: low-disagreement pixel, salient pixel

**Disagreement map**:
For pixels that are not certain pixels, the per-frame-normalized mean absolute difference between the fidelity restoration and the generative restoration.
_Avoid_: uncertainty, predictive variance, semantic class
