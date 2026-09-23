#!/usr/bin/env python3
"""Show which files the per-source metrics will group together, and under what name.

For a joint (paired/triplet) run, eval_em_seg splits the test set by filename stem —
`dino_peft.utils.sample_groups` tokenises each stem and picks the depth that yields the
expected number of groups. That works silently whether or not the resulting groups are
the ones you meant, so a run can report a clean three-way breakdown of the wrong three
things. This prints the grouping before you spend GPU hours on it.

Reads filenames only; loads no model and no image data.

Example:
    python scripts/utils/check_sample_groups.py \
      --cfg configs/cluster/deepbacs/deepbacs_triplet_openclip_vitl14_cluster.yaml
    python scripts/utils/check_sample_groups.py --cfg <cfg> --split train --expect coli aureus subtilis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dino_peft.config import load_config  # noqa: E402
from dino_peft.utils.sample_groups import (  # noqa: E402
    infer_expected_group_count,
    infer_sample_grouping,
)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="The run config to inspect.")
    ap.add_argument("--split", default="test", choices=("test", "train"),
                    help="Which split's filenames to group (default: test — what eval uses).")
    ap.add_argument("--expect", nargs="*", default=None,
                    help="Group names you expect, e.g. --expect coli aureus subtilis. "
                         "Exits non-zero if they do not match.")
    ap.add_argument("--examples", type=int, default=4,
                    help="Example filenames to print per group (default: 4).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.cfg)

    img_dir = Path(cfg[f"{args.split}_img_dir"]).expanduser()
    if not img_dir.is_dir():
        raise SystemExit(f"{args.split}_img_dir does not exist: {img_dir}")
    names = sorted(
        p.stem for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith(".")
    )
    if not names:
        raise SystemExit(f"no images found in {img_dir}")

    expected = infer_expected_group_count(cfg)
    print(f"config    : {args.cfg}")
    print(f"{args.split + ' images':<10}: {len(names)} files in {img_dir}")
    print(f"expected groups: {expected}   "
          f"(from the substring 'triplet'/'paired' in the paths, task_type or experiment_id)")

    if expected not in (2, 3):
        print("\nThis is not a joint run, so eval reports one pooled foreground IoU and no")
        print("per-source breakdown. Nothing to check.")
        return 0

    grouping = infer_sample_grouping(names, expected_groups=expected)
    buckets: dict[str, list[str]] = {}
    for name, label in zip(names, grouping.labels):
        buckets.setdefault(label, []).append(name)

    print(f"token depth used: {grouping.inferred_depth}")
    print(f"groups found    : {len(buckets)}\n")
    for label in sorted(buckets):
        members = buckets[label]
        print(f"  {label!r}  —  {len(members)} files")
        for example in members[: args.examples]:
            print(f"      {example}")
        if len(members) > args.examples:
            print(f"      ... and {len(members) - args.examples} more")
        print()

    if len(buckets) != expected:
        print(f"MISMATCH: expected {expected} groups, got {len(buckets)}. The per-source")
        print("metrics will not line up with the datasets that were composed.")
        return 1

    if args.expect:
        want, got = sorted(str(e).lower() for e in args.expect), sorted(buckets)
        if want != got:
            print(f"MISMATCH: expected groups {want}, got {got}.")
            print("The split may still be correct — check the example filenames above — but the")
            print("labels in metrics.json will not be the dataset names.")
            return 1
        print(f"OK: groups match {want}.")
    else:
        print("Check the example filenames above: each group must be exactly one source")
        print("dataset. Matching counts alone are not enough — three groups of the right")
        print("size can still be the wrong three-way split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
