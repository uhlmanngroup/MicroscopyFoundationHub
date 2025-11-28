"""Utilities for quantifying shifts between feature domains."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dino_peft.analysis.dimred import load_feature_npz


Array = np.ndarray


def _normalize_list(values: Sequence | str | int | None) -> list | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


def load_features(
    npz_path: str | Path,
    dataset_ids: Sequence[int] | int | None = None,
    dataset_names: Sequence[str] | str | None = None,
) -> Array:
    """Load a feature matrix from .npz, optionally filtering by dataset IDs or names."""
    path = Path(npz_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")
    bundle = load_feature_npz(path)
    feats = np.asarray(bundle.features)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D array for features, got shape {feats.shape}")

    ds_ids = (
        [int(v) for v in _normalize_list(dataset_ids) or []]
    )
    ds_names = (
        [str(v) for v in _normalize_list(dataset_names) or []]
    )

    if not ds_ids and not ds_names:
        return feats

    mask = np.ones(feats.shape[0], dtype=bool)
    bundle_ids = (
        np.asarray(bundle.dataset_ids).astype(int, copy=False)
        if bundle.dataset_ids is not None
        else None
    )
    if ds_ids:
        if bundle_ids is None:
            raise ValueError(f"dataset_ids metadata missing in {path}")
        mask &= np.isin(bundle_ids, ds_ids)

    if ds_names:
        applied = False
        if getattr(bundle, "dataset_names", None):
            name_arr = np.asarray(bundle.dataset_names, dtype=object)
            mask &= np.isin(name_arr, ds_names)
            applied = True
        elif bundle.meta and bundle.meta.get("dataset_name_to_id") and bundle_ids is not None:
            mapping = bundle.meta["dataset_name_to_id"]
            target_ids = []
            for name in ds_names:
                if name not in mapping:
                    raise ValueError(f"dataset name '{name}' not found in {path}")
                target_ids.append(int(mapping[name]))
            mask &= np.isin(bundle_ids, target_ids)
            applied = True
        if not applied:
            raise ValueError(
                f"dataset_names filtering requested but metadata missing in {path}"
            )

    filtered = feats[mask]
    if filtered.size == 0:
        raise ValueError(
            f"No samples remaining after filtering {path} "
            f"(dataset_ids={ds_ids or None}, dataset_names={ds_names or None})"
        )
    return filtered


def fit_pca_on_concat(
    x1: Array,
    x2: Array,
    n_components: int,
    random_state: int | None = None,
) -> Tuple[PCA, Array, Array]:
    """Fit a single PCA on the concatenation of x1/x2 and return transformed copies."""
    a = np.asarray(x1)
    b = np.asarray(x2)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D (got {a.shape} and {b.shape})"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dims must match (got {a.shape[1]} and {b.shape[1]})"
        )
    total = np.concatenate([a, b], axis=0)
    n_samples, feat_dim = total.shape
    if n_components > min(n_samples, feat_dim):
        raise ValueError(
            f"n_components={n_components} exceeds allowable "
            f"min(n_samples={n_samples}, n_features={feat_dim})"
        )
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(total)
    return pca, pca.transform(a), pca.transform(b)


def compute_gaussian_stats(x: Array) -> Tuple[Array, Array]:
    """Return (mean, covariance) treating each row in x as a sample."""
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    mean = np.mean(arr, axis=0)
    cov = np.cov(arr, rowvar=False)
    return mean, cov


def compute_fid_from_features(x1: Array, x2: Array) -> float:
    """Compute Fréchet distance between two Gaussian stats derived from feature rows."""
    mu1, sigma1 = compute_gaussian_stats(x1)
    mu2, sigma2 = compute_gaussian_stats(x2)
    fid = float(
        calculate_frechet_distance(
            mu1.astype(np.float64, copy=False),
            sigma1.astype(np.float64, copy=False),
            mu2.astype(np.float64, copy=False),
            sigma2.astype(np.float64, copy=False),
        )
    )
    return fid


def lr_domain_separability(
    x1: Array,
    x2: Array,
    random_state: int | None = None,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """Train/test a logistic regressor to measure how linearly separable the domains are."""
    a = np.asarray(x1)
    b = np.asarray(x2)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Inputs must be 2D (got {a.shape} and {b.shape})")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Feature dims must match (got {a.shape[1]} and {b.shape[1]})")
    x = np.concatenate([a, b], axis=0)
    y = np.concatenate(
        [
            np.zeros(a.shape[0], dtype=np.int32),
            np.ones(b.shape[0], dtype=np.int32),
        ]
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )
    clf = LogisticRegression(
        penalty="l2",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)
    val_probs = clf.predict_proba(x_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_val, val_preds)),
        "roc_auc": float(roc_auc_score(y_val, val_probs)),
    }
    return metrics
