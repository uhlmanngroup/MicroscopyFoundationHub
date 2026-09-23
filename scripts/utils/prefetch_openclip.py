#!/usr/bin/env python3
"""Fetch an OpenCLIP checkpoint into the local cache, before a sweep fans out.

open_clip downloads weights lazily on first use, so submitting twenty array jobs sends
twenty processes at the same cache entry at once — repeated downloads of the same ~1.7 GB,
or a torn partial file. Doing it once first avoids that.

This downloads the file and does NOT build the model: constructing a ViT-L costs enough
CPU to trip the ulimit on a login node, and nothing here needs the weights in memory.

Already cached is the normal case (a pilot run warms it), and then this returns at once.

Example:
    python scripts/utils/prefetch_openclip.py --model ViT-L-14 --pretrained laion2b_s32b_b82k
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--cache-dir", default=None,
                    help="Override the download cache (default: open_clip's own).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import open_clip
        from open_clip.pretrained import download_pretrained, get_pretrained_cfg
    except ImportError:
        print("[error] open_clip is not installed in this environment. "
              "It is a declared dependency, so `pip install -e .` (or "
              "`pip install open-clip-torch`) inside the dino-peft env fixes this.",
              file=sys.stderr)
        return 1

    tags = [tag for model, tag in open_clip.list_pretrained() if model == args.model]
    if args.pretrained not in tags:
        print(f"[error] '{args.pretrained}' is not a tag for {args.model}. "
              f"Known tags: {tags}", file=sys.stderr)
        return 1

    cfg = get_pretrained_cfg(args.model, args.pretrained)
    if not cfg:
        print(f"[error] no pretrained cfg for {args.model} / {args.pretrained}", file=sys.stderr)
        return 1

    print(f"[prefetch] {args.model} @ {args.pretrained} ...")
    path = download_pretrained(cfg, cache_dir=args.cache_dir)
    print(f"[prefetch] cached at {path}")
    print("[prefetch] the sweep will not re-download it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
