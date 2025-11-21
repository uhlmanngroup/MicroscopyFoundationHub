from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class ResizeSpec:
    """
    Normalized representation of the `img_size` config knob used across the repo.

    Modes:
        - fixed:        Always resize to (H, W).
        - longest_edge: Scale so max(H, W) == `target_long_edge` and snap each side
                        to a multiple of `patch_multiple`.
    """

    mode: str
    size: Tuple[int, int] | None = None
    target_long_edge: int | None = None
    patch_multiple: int = 14
    rounding: str = "floor"  # floor, ceil, round

    def describe(self) -> str:
        if self.mode == "fixed":
            return f"fixed {self.size}"
        return (
            f"longest_edge={self.target_long_edge}, "
            f"patch_multiple={self.patch_multiple}, rounding={self.rounding}"
        )


def _round_to_multiple(value: float, multiple: int, how: str) -> int:
    if multiple is None or multiple <= 1:
        return max(1, int(round(value)))
    ratio = value / multiple
    if how == "floor":
        steps = floor(ratio)
    elif how == "ceil":
        steps = ceil(ratio)
    elif how == "round":
        steps = round(ratio)
    else:
        raise ValueError(f"Unknown rounding mode '{how}'")
    steps = max(1, steps)
    return int(steps * multiple)


def parse_img_size_config(cfg: Any, default_patch_multiple: int = 14) -> ResizeSpec:
    """
    Accept int / tuple / dict configs and normalize them into a ResizeSpec.
    """
    if cfg is None:
        raise ValueError("img_size config cannot be None")

    if isinstance(cfg, (list, tuple)):
        if len(cfg) != 2:
            raise ValueError(f"Expected img_size tuple/list of len 2, got {cfg}")
        h, w = int(cfg[0]), int(cfg[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid img_size values {cfg}")
        return ResizeSpec(mode="fixed", size=(h, w), patch_multiple=1, rounding="round")

    if isinstance(cfg, int):
        if cfg <= 0:
            raise ValueError(f"img_size int must be positive, got {cfg}")
        return ResizeSpec(
            mode="longest_edge",
            target_long_edge=cfg,
            patch_multiple=default_patch_multiple,
            rounding="floor",
        )

    if isinstance(cfg, Mapping):
        mode = cfg.get("mode", "fixed")
        rounding = cfg.get("rounding", "floor")
        patch_multiple = int(cfg.get("patch_multiple", default_patch_multiple))
        if mode == "fixed":
            size = cfg.get("size", None)
            if size is None:
                size = (cfg.get("height"), cfg.get("width"))
            if size is None or None in size:
                raise ValueError(f"Fixed img_size dict requires 'size' or (height,width): {cfg}")
            h, w = int(size[0]), int(size[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid fixed img_size values: {size}")
            return ResizeSpec(mode="fixed", size=(h, w), patch_multiple=1, rounding="round")
        elif mode == "longest_edge":
            target = cfg.get("target") or cfg.get("target_long_edge")
            if target is None:
                raise ValueError(f"Longest-edge img_size dict requires 'target': {cfg}")
            target = int(target)
            if target <= 0:
                raise ValueError(f"'target' must be positive, got {target}")
            return ResizeSpec(
                mode="longest_edge",
                target_long_edge=target,
                patch_multiple=patch_multiple,
                rounding=rounding,
            )
        else:
            raise ValueError(f"Unknown img_size mode '{mode}' in {cfg}")

    raise TypeError(f"Unsupported img_size config type: {type(cfg)}")


def compute_resized_hw(orig_hw: Tuple[int, int], spec: ResizeSpec) -> Tuple[int, int]:
    """
    Given original (H, W) integers and a ResizeSpec, return the resized (H, W).
    """
    h, w = int(orig_hw[0]), int(orig_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid original hw={orig_hw}")

    if spec.mode == "fixed":
        if spec.size is None:
            raise ValueError("ResizeSpec(mode='fixed') is missing 'size'")
        return spec.size

    if spec.mode != "longest_edge":
        raise ValueError(f"Unknown resize mode '{spec.mode}'")

    if spec.target_long_edge is None:
        raise ValueError("ResizeSpec(mode='longest_edge') missing target_long_edge")

    long_edge = max(h, w)
    short_edge = min(h, w)
    if long_edge == 0:
        raise ValueError(f"Invalid input hw={orig_hw}")

    scale = spec.target_long_edge / long_edge
    new_long = spec.target_long_edge
    new_short = short_edge * scale

    rounded_long = _round_to_multiple(new_long, spec.patch_multiple, spec.rounding)
    rounded_short = _round_to_multiple(new_short, spec.patch_multiple, spec.rounding)

    if h >= w:
        return rounded_long, rounded_short
    else:
        return rounded_short, rounded_long
