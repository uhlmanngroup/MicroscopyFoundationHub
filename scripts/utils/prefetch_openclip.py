#!/usr/bin/env python3
"""Download an OpenCLIP checkpoint once, before a sweep fans out.

open_clip fetches weights lazily on first use. Submitting 30 array jobs therefore
sends 30 processes at the same cache directory at the same time, which either
re-downloads the same ~1.7 GB repeatedly or corrupts a partial file. Running this on
the submit host first turns that into one download, and fails immediately (with a
readable error) if the node has no network rather than 30 jobs failing an hour in.

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
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import open_clip
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

    print(f"[prefetch] {args.model} @ {args.pretrained} ...")
    open_clip.create_model(args.model, pretrained=args.pretrained)
    print("[prefetch] weights are in the cache; the sweep will not re-download them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
