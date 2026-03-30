#!/usr/bin/env python3
"""Merge multiple feature NPZ files into one bundle for joint PCA/UMAP plots.

Example:
  python scripts/analysis/merge_feature_npzs.py \
    --output /tmp/domain_shift_3way/features_merged.npz \
    --input em=/path/to/em/features.npz \
    --input deepbacs=/path/to/deepbacs/features.npz \
    --input openimages=/path/to/openimages/features.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dino_peft.analysis.dimred import load_feature_npz


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input in the form label=/path/to/features.npz. Repeat for each domain.",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output merged NPZ path.")
    return ap.parse_args()


def _parse_input_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected --input label=/path/to/features.npz, got: {text}")
    label, raw_path = text.split("=", 1)
    label = label.strip().lower()
    path = Path(raw_path).expanduser()
    if not label:
        raise ValueError(f"Empty label in input: {text}")
    if not path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {path}")
    return label, path


def main() -> None:
    args = parse_args()
    items = [_parse_input_item(text) for text in args.inputs]

    features_all = []
    dataset_names_all = []
    dataset_ids_all = []
    image_paths_all = []
    dataset_name_to_id = {}

    backbone_name = None
    backbone_variant = None

    for next_id, (label, npz_path) in enumerate(items):
        bundle = load_feature_npz(npz_path)
        feats = np.asarray(bundle.features)
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D features in {npz_path}, got {feats.shape}")

        if backbone_name is None:
            backbone_name = bundle.meta.get("backbone_name") if bundle.meta else None
        if backbone_variant is None:
            backbone_variant = bundle.meta.get("backbone_variant") if bundle.meta else None

        dataset_name_to_id[label] = next_id
        n = feats.shape[0]
        features_all.append(feats.astype(np.float32, copy=False))
        dataset_names_all.append(np.array([label] * n, dtype=object))
        dataset_ids_all.append(np.full(n, next_id, dtype=np.int32))

        if bundle.image_paths is not None:
            image_paths_all.append(np.array(bundle.image_paths, dtype=object))
        else:
            image_paths_all.append(np.array([""] * n, dtype=object))

        print(f"[merge_feature_npzs] {label}: {npz_path} (N={n}, D={feats.shape[1]})")

    merged_features = np.concatenate(features_all, axis=0)
    merged_dataset_names = np.concatenate(dataset_names_all, axis=0)
    merged_dataset_ids = np.concatenate(dataset_ids_all, axis=0)
    merged_image_paths = np.concatenate(image_paths_all, axis=0)
    mapping = np.array([f"{name}:{idx}" for name, idx in dataset_name_to_id.items()], dtype=object)

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=merged_features,
        dataset_names=merged_dataset_names,
        dataset_ids=merged_dataset_ids,
        dataset_name_to_id=mapping,
        image_paths=merged_image_paths,
        backbone_name=np.array([backbone_name], dtype=object),
        backbone_variant=np.array([backbone_variant], dtype=object),
    )
    print(f"[merge_feature_npzs] wrote merged NPZ -> {output}")
    print(f"[merge_feature_npzs] total samples={merged_features.shape[0]} dim={merged_features.shape[1]}")


if __name__ == "__main__":
    main()
