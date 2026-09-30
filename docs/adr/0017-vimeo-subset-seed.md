# The Vimeo benchmark is a frozen 30-clip draw

This narrows ADR-0004. REDS4 is unchanged. The Vimeo subset is 30 seven-frame clips drawn once, without replacement, from the official test list in its published order, using seed 0. The paths are stored in `docs/manifests/vimeo-subset.txt` and are not redrawn. Tables name this set "Vimeo subset (30)". They do not call it the full Vimeo-90K-T. EDVR and BasicVSR report the full test set; this draw is a course subset.
