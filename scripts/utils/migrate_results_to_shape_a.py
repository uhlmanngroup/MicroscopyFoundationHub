#!/usr/bin/env python3
"""Move existing run trees onto the shape-A layout the configs now write.

Shape A is ``<root>/<backbone>/<modality>/<family>[/<group>]/<experiment_id>``,
matching ``dino_peft.utils.paths.resolve_run_dir``.

Three legacy shapes exist on disk:

* ``<backbone>/<family>/<modality>/<group>``  — the newest scratch runs (shape B)
* ``<backbone>/<family>/<group>``             — runs predating the modality level
* ``<backbone>/deepbacs/seg/paired/deepbacs`` — a modality inserted twice

DeepBacs keeps its ``paired``/``single``/``triplet`` level between the family
and the group, mirroring the dataset tree; ``--flatten-combo`` drops it.

Every move is a rename inside one filesystem (``/home`` or ``/scratch``), so
nothing is copied and nothing crosses a storage tier. Run without ``--apply``
first: it prints the plan and refuses to run at all if any destination exists.

    python scripts/utils/migrate_results_to_shape_a.py            # dry run
    python scripts/utils/migrate_results_to_shape_a.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATA = Path("/home/cfuste/data/DINO-LoRA")
SCRATCH = Path("/home/cfuste/scratch/DINO-LoRA")
CKPTS = Path("/home/cfuste/scratch/DINO-LoRA-ckpts")

# Group-name prefixes that identify a DeepBacs combination, longest first.
_TRIPLET = ("coli-aureus-subtilis",)
_PAIRED = ("aureus-subtilis", "coli-aureus", "coli-subtilis")


def _combo(group: str) -> str:
    if group.startswith(_TRIPLET):
        return "triplet"
    if group.startswith(_PAIRED):
        return "paired"
    return "single"


def plan(keep_combo: bool) -> list[tuple[Path, Path]]:
    """Every (source, destination) rename, parents before children."""
    moves: list[tuple[Path, Path]] = []

    def deepbacs_dest(base: Path, group: str, combo: str | None = None) -> Path:
        """Destination for one DeepBacs group.

        ``combo`` is the combination the group already sits in, when the old
        layout records it; otherwise it is read off the group name. A
        ``summary`` directory aggregates its combination rather than being one
        more group, so it keeps that combination in its name either way.
        """
        seg = base / "deepbacs" / "seg"
        if group == "summary":
            if combo is None:
                return seg / "summary"
            return seg / combo / "summary" if keep_combo else seg / f"summary-{combo}"
        if keep_combo:
            return seg / (combo or _combo(group)) / group
        return seg / group

    # 1. shape B -> A: <bb>/seg/<modality>/<group> becomes <bb>/<modality>/seg/<group>
    for backbone_dir in sorted(p for p in SCRATCH.glob("*/seg") if p.is_dir()):
        backbone = backbone_dir.parent
        for modality_dir in sorted(p for p in backbone_dir.iterdir() if p.is_dir()):
            for group in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
                if modality_dir.name == "deepbacs":
                    moves.append((group, deepbacs_dest(backbone, group.name)))
                else:
                    moves.append((group, backbone / modality_dir.name / "seg" / group.name))

    # 2. no modality level -> insert "em" (these trees hold EM runs only)
    for root in (DATA, CKPTS):
        for backbone_dir in sorted(p for p in root.glob("*/seg") if p.is_dir()):
            backbone = backbone_dir.parent
            for entry in sorted(p for p in backbone_dir.iterdir() if p.is_dir()):
                if entry.name == "deepbacs":          # a modality, not a group
                    for group in sorted(p for p in entry.iterdir() if p.is_dir()):
                        moves.append((group, deepbacs_dest(backbone, group.name)))
                else:
                    moves.append((entry, backbone / "em" / "seg" / entry.name))

    # 3. the doubled modality: <bb>/deepbacs/seg/paired/deepbacs/<group>
    for root in (DATA, CKPTS):
        for backbone_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            seg = backbone_dir / "deepbacs" / "seg"
            for combo in ("paired", "single", "triplet"):
                nested = seg / combo / "deepbacs"
                if nested.is_dir():
                    for group in sorted(p for p in nested.iterdir() if p.is_dir()):
                        moves.append((group, deepbacs_dest(backbone_dir, group.name, combo)))
                flat = seg / combo
                if flat.is_dir() and not keep_combo:
                    for group in sorted(p for p in flat.iterdir() if p.is_dir()):
                        if group.name == "deepbacs":
                            continue          # the doubled level, handled above
                        moves.append((group, deepbacs_dest(backbone_dir, group.name, combo)))

    # 4. "domain-analysis" is a family, not a modality: park 3-way runs under "multi"
    three_way = DATA / "dinov2" / "domain-analysis" / "3way"
    if three_way.is_dir():
        moves.append((three_way, DATA / "dinov2" / "multi" / "domain-analysis" / "3way"))

    return [(s, d) for s, d in moves if s != d]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="perform the moves")
    ap.add_argument("--flatten-combo", action="store_true",
                    help="drop the DeepBacs paired/single/triplet level under seg/")
    args = ap.parse_args()

    moves = plan(keep_combo=not args.flatten_combo)
    if not moves:
        print("Nothing to move — the trees are already shape A.")
        return 0

    collisions = [(s, d) for s, d in moves if d.exists()]
    seen: dict[Path, Path] = {}
    for src, dest in moves:
        if dest in seen:
            collisions.append((src, dest))
        seen[dest] = src

    for src, dest in moves:
        print(f"{src}\n  -> {dest}")
    print(f"\n{len(moves)} directories to rename.")

    if collisions:
        print(f"\nREFUSING TO RUN — {len(collisions)} destination(s) already taken:")
        for src, dest in collisions:
            print(f"  {dest}   (from {src})")
        return 1

    if not args.apply:
        print("\nDry run. Re-run with --apply to perform the moves.")
        return 0

    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    print(f"\nMoved {len(moves)} directories.")

    # Drop the now-empty scaffolding the old shapes left behind.
    for root in (DATA, SCRATCH, CKPTS):
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                print(f"removed empty {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
