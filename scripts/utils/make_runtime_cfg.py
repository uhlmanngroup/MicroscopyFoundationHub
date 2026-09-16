#!/usr/bin/env python3
"""Write a one-off runtime config from a base config plus overrides.

Every sweep script used to carry its own copy of this logic as an inline
heredoc, which is how a config reorganisation broke 26 of them at once. The
sweep script now passes what differs and this does the rest.

Prints two lines, which the caller reads in order:

    <path to the runtime config>
    <run directory the config resolves to>

Example:
    python scripts/utils/make_runtime_cfg.py \
      --base-cfg configs/cluster/deepbacs/deepbacs_single_dinov3_cluster.yaml \
      --tuning-mode lora --seed 3 --split-seed 103 \
      --experiment-id 2026-09-08_deepbacs_subtilis_dinov3_lora_seg_rep3 \
      --task-type subtilis-lora \
      --set backbone.variant=vitl16 \
      --set train_img_dir=/data/deepbacs/single/subtilis/train/images
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dino_peft.config import load_config  # noqa: E402
from dino_peft.utils.paths import resolve_run_dir  # noqa: E402

#: How each regime is expressed in the config. Keeping it here means a sweep
#: script cannot get the LoRA/full-finetune flags subtly wrong.
TUNING_MODES = {
    "head": {"use_lora": False, "full_finetune": False},
    "lora": {"use_lora": True, "full_finetune": False},
    "fullft": {"use_lora": False, "full_finetune": True},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-cfg", required=True, help="Config to start from.")
    ap.add_argument("--tuning-mode", choices=sorted(TUNING_MODES),
                    help="Adaptation regime; sets use_lora / full_finetune / lora.enabled.")
    ap.add_argument("--seed", type=int, help="Training seed.")
    ap.add_argument("--split-seed", type=int, help="Train/val split seed.")
    ap.add_argument("--experiment-id", help="Run name.")
    ap.add_argument("--task-type", help="Results subdirectory.")
    ap.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Any other override. Dotted keys nest (backbone.variant=vitl16). "
             "Values are parsed as YAML, so 2, true and null keep their types. "
             "Repeatable.",
    )
    ap.add_argument("--out", help="Where to write; default is a temp file.")
    ap.add_argument("--prefix", default="run_", help="Temp file prefix.")
    return ap.parse_args()


def apply_override(cfg: dict, dotted_key: str, value) -> None:
    """Set ``a.b.c`` on a nested dict, creating intermediate dicts as needed."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        existing = node.get(part)
        if not isinstance(existing, dict):
            existing = {}
            node[part] = existing
        node = existing
    node[parts[-1]] = value


def apply_tuning_mode(cfg: dict, mode: str) -> None:
    cfg.update(TUNING_MODES[mode])
    # The nested `lora:` block, when present, gates injection independently.
    if isinstance(cfg.get("lora"), dict):
        cfg["lora"]["enabled"] = cfg["use_lora"]


def main() -> int:
    args = parse_args()

    # Resolve through the normal loader so `extends` and ${path} placeholders
    # are already applied; the runtime file is then fully self-contained.
    cfg = load_config(args.base_cfg, warn_seed=False)
    cfg.pop("seed_explicit", None)

    if args.tuning_mode:
        apply_tuning_mode(cfg, args.tuning_mode)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.split_seed is not None:
        cfg["split_seed"] = args.split_seed
    if args.experiment_id:
        cfg["experiment_id"] = args.experiment_id
    if args.task_type:
        cfg["task_type"] = args.task_type

    for override in args.overrides:
        if "=" not in override:
            raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
        key, raw = override.split("=", 1)
        apply_override(cfg, key.strip(), yaml.safe_load(raw))

    if "seed" not in cfg:
        print(
            "[make_runtime_cfg] WARNING: no seed set; this run will use seed=0. "
            "Pass --seed so repeats actually differ.",
            file=sys.stderr,
        )

    run_dir = resolve_run_dir(cfg, cfg.get("task_type", "seg"))
    cfg["out_dir"] = str(run_dir)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        handle = tempfile.NamedTemporaryFile(
            prefix=args.prefix, suffix=".yaml", delete=False
        )
        handle.close()
        out_path = Path(handle.name)

    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(out_path)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
