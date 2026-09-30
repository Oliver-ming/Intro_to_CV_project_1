"""Install prefetched CLIP and SVD-VAE files into the Hugging Face cache.

Official STAR loads these two weights before the released checkpoint.
A direct Hugging Face download of the CLIP bin stalled at zero bytes, so
aria2c writes them under data/cache_stage and this script publishes the
completed files where open_clip and diffusers look.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "cache_stage"
HUB = Path.home() / ".cache" / "huggingface" / "hub"

FILES = (
    {
        "stage": "open_clip_pytorch_model.bin",
        "repo": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
        "commit": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        "name": "open_clip_pytorch_model.bin",
        "oid": "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4",
        "size": 3944692325,
    },
    {
        "stage": "diffusion_pytorch_model.fp16.safetensors",
        "repo": "models--stabilityai--stable-video-diffusion-img2vid",
        "commit": "9cf024d5bfa8f56622af86c884f26a52f6676f2e",
        "name": "vae/diffusion_pytorch_model.fp16.safetensors",
        "oid": "af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3",
        "size": 195531910,
    },
)


def main() -> None:
    for item in FILES:
        source = STAGE / item["stage"]
        control = Path(str(source) + ".aria2")
        if control.exists() or not source.is_file() or source.stat().st_size != item["size"]:
            raise SystemExit(f"not complete: {source}")
        repo = HUB / item["repo"]
        blob = repo / "blobs" / item["oid"]
        snapshot = repo / "snapshots" / item["commit"] / item["name"]
        blob.parent.mkdir(parents=True, exist_ok=True)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists() or blob.is_symlink():
            blob.unlink()
        source.replace(blob)
        if snapshot.exists() or snapshot.is_symlink():
            snapshot.unlink()
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(item["commit"] + "\n")
        print(f"cached {item['name']}", flush=True)


if __name__ == "__main__":
    main()
