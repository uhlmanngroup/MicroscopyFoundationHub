"""Single entry point for loading experiment configs.

Every script loads its YAML through :func:`load_config` so that three things
happen in exactly one place:

1. **Shared defaults.** A config that opts in with ``extends: defaults`` picks
   up ``configs/defaults.yaml``, which holds the values that are the same
   across training experiments (learning rate, loss, validation split, ...), so
   the config only states what makes it different. Anything the config does set
   always wins. Configs that do not train — feature extraction, PCA, OOD,
   domain analysis — simply omit ``extends`` and get no defaults.

2. **Machine paths in one file.** ``configs/paths.yaml`` holds the handful of
   absolute paths that change per machine. Configs refer to them as
   ``${data_root}/EM/Lucchi++/Train_In``. Any entry can be overridden from the
   environment with ``DINO_PEFT_<KEY>`` (e.g. ``DINO_PEFT_DATA_ROOT``), so the
   same config runs on the cluster and on a laptop without being edited.

3. **Seed visibility.** A config with no ``seed`` silently trains at seed 0.
   That is how a set of "repeated" runs once ended up identical, so loading a
   config without one now prints a loud warning.

Configs that predate this module keep working unchanged: absolute paths are
left alone, and defaults only ever fill in keys that are absent.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "load_config",
    "load_paths",
    "deep_merge",
    "expand_placeholders",
    "repo_root",
    "DEFAULTS_FILE",
    "PATHS_FILE",
]

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_PREFIX = "DINO_PEFT_"

DEFAULTS_FILE = "defaults.yaml"
PATHS_FILE = "paths.yaml"

#: Older configs spell the crop key with a ``deepbacs_`` prefix even though the
#: MoNuSAC pipeline uses it too. The new name is ``center_crop_size``.
_RENAMED_KEYS = {"deepbacs_center_crop_size": "center_crop_size"}


def repo_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    return repo_root() / "configs"


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` onto ``base``; ``override`` always wins.

    Nested mappings are merged key by key. Lists and scalars are replaced
    wholesale rather than concatenated.
    """
    merged = dict(deepcopy(dict(base)))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_paths() -> dict[str, str]:
    """Machine paths from ``configs/paths.yaml``, with environment overrides.

    ``DINO_PEFT_DATA_ROOT=/scratch/me`` overrides the ``data_root`` entry.
    Environment overrides also introduce keys absent from the file, so a new
    placeholder can be supplied without editing it.
    """
    paths = {
        str(key): str(value)
        for key, value in _read_yaml(_config_dir() / PATHS_FILE).items()
        if value is not None
    }
    for env_key, env_value in os.environ.items():
        if env_key.startswith(_ENV_PREFIX) and env_value:
            paths[env_key[len(_ENV_PREFIX) :].lower()] = env_value
    return paths


def expand_placeholders(obj: Any, mapping: Mapping[str, str]) -> Any:
    """Replace ``${name}`` in every string of a nested structure.

    Expansion is iterative, so a path entry may itself refer to another one.
    An unknown placeholder raises rather than silently leaving ``${...}`` in a
    filesystem path, where it would surface much later as a confusing
    "no such file" error.
    """
    if isinstance(obj, Mapping):
        return {key: expand_placeholders(value, mapping) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_placeholders(value, mapping) for value in obj]
    if not isinstance(obj, str) or "${" not in obj:
        return obj

    seen: set[str] = set()
    text = obj
    for _ in range(10):
        missing = [name for name in _PLACEHOLDER.findall(text) if name not in mapping]
        if missing:
            raise KeyError(
                f"Config refers to unknown path placeholder(s) {sorted(set(missing))} "
                f"in {obj!r}. Define them in configs/{PATHS_FILE} or export "
                f"{_ENV_PREFIX}{missing[0].upper()}."
            )
        expanded = _PLACEHOLDER.sub(lambda m: mapping[m.group(1)], text)
        if expanded == text:
            return text
        if expanded in seen:
            raise ValueError(f"Circular path placeholder while expanding {obj!r}")
        seen.add(expanded)
        text = expanded
    raise ValueError(f"Path placeholder nesting too deep while expanding {obj!r}")


def _apply_renames(cfg: dict) -> dict:
    for old, new in _RENAMED_KEYS.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg[old]
    return cfg


def _resolve_extends(name: str) -> Path:
    """Map an ``extends:`` value to a file under ``configs/``."""
    candidate = Path(str(name))
    if candidate.suffix not in (".yaml", ".yml"):
        candidate = candidate.with_suffix(".yaml")
    path = _config_dir() / candidate
    if not path.is_file():
        raise FileNotFoundError(
            f"Config extends '{name}', but configs/{candidate} does not exist."
        )
    return path


def load_config(
    cfg_path: str | Path,
    *,
    apply_defaults: bool = True,
    expand: bool = True,
) -> dict:
    """Load a YAML config, resolving ``extends``, defaults and path placeholders.

    A config opts into shared values with ``extends: defaults``, naming a file
    under ``configs/``. The parent is merged underneath, so the config's own
    keys always win. ``extends`` chains, and is removed from the result.

    Args:
        cfg_path: the config to load.
        apply_defaults: honour ``extends``. Set False to see a config raw.
        expand: resolve ``${name}`` placeholders from ``configs/paths.yaml``.

    Returns:
        The fully resolved config as a plain dict.
    """
    path = Path(cfg_path).expanduser()
    cfg = _apply_renames(_read_yaml(path))
    has_seed = "seed" in cfg

    if apply_defaults:
        seen = {path.resolve()}
        while cfg.get("extends"):
            parent_path = _resolve_extends(cfg.pop("extends"))
            if parent_path.resolve() in seen:
                raise ValueError(f"Circular 'extends' chain reaching {parent_path}")
            seen.add(parent_path.resolve())
            cfg = deep_merge(_apply_renames(_read_yaml(parent_path)), cfg)
    cfg.pop("extends", None)

    if expand:
        cfg = expand_placeholders(cfg, load_paths())

    cfg["seed_explicit"] = has_seed

    return cfg
