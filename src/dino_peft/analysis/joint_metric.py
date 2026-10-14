"""Local foreground/background directions and frozen-feature linear probes.

The compatibility diagnostic is the cosine between dataset directions. Probe
training is a separate check; it never updates or loads the backbone.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class MetricConfig:
    tau_fg: float = 0.8
    tau_bg: float = 0.2
    near_foreground_radius_patches: int = 2
    seed: int = 42
    l2_normalize: bool = True
    probe_patches_per_image_per_class: int = 32
    probe_train_fraction: float = 0.6
    probe_validation_fraction: float = 0.2
    probe_l2: float = 1e-3
    probe_max_iterations: int = 100
    direction_epsilon: float = 1e-12

    def __post_init__(self):
        if not 0 <= self.tau_bg < self.tau_fg <= 1:
            raise ValueError("Require 0 <= tau_bg < tau_fg <= 1")
        for field in ("near_foreground_radius_patches", "probe_patches_per_image_per_class", "probe_max_iterations"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 < self.probe_train_fraction < 1 or not 0 < self.probe_validation_fraction < 1:
            raise ValueError("Probe split fractions must be between zero and one")
        if self.probe_train_fraction + self.probe_validation_fraction >= 1:
            raise ValueError("Probe split must leave images for testing")
        if not np.isfinite(self.probe_l2) or self.probe_l2 < 0:
            raise ValueError("probe_l2 must be finite and nonnegative")
        if not np.isfinite(self.direction_epsilon) or self.direction_epsilon <= 0:
            raise ValueError("direction_epsilon must be positive and finite")


def _rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256((str(seed) + "\0" + "\0".join(parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _grid(occupancy: np.ndarray, grid_shape: Sequence[int]):
    height, width = map(int, grid_shape)
    q = np.asarray(occupancy)
    if height < 1 or width < 1 or q.shape != (height * width,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Invalid occupancy or patch grid")
    return q.reshape(height, width)


def near_foreground_background(occupancy: np.ndarray, grid_shape: Sequence[int],
                               *, tau_fg: float, tau_bg: float, radius: int) -> np.ndarray:
    """Clear background within Chebyshev radius of clear foreground, per image."""
    q = _grid(occupancy, grid_shape)
    if radius < 1:
        raise ValueError("Near-foreground radius must be positive")
    fg = (q >= tau_fg).astype(np.int32)
    padded = np.pad(fg, radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    diameter = 2 * radius + 1
    nearby = (integral[diameter:, diameter:] - integral[:-diameter, diameter:]
              - integral[diameter:, :-diameter] + integral[:-diameter, :-diameter]) > 0
    return ((q <= tau_bg) & nearby).reshape(-1)


def local_directions(embeddings: np.ndarray, occupancy: np.ndarray,
                     grid_shape: Sequence[int], config: MetricConfig) -> dict:
    """Compute one d(f)=f-mean(local clear BG) for every usable FG token."""
    q = _grid(occupancy, grid_shape)
    z = np.asarray(embeddings, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != q.size or not np.isfinite(z).all():
        raise ValueError("Embeddings do not match the patch grid")
    if config.l2_normalize:
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / np.where(norms > 0, norms, 1)
    height, width = q.shape
    radius = config.near_foreground_radius_patches
    fg = np.flatnonzero(q.reshape(-1) >= config.tau_fg)
    nearby_bg = near_foreground_background(occupancy, grid_shape, tau_fg=config.tau_fg,
                                          tau_bg=config.tau_bg, radius=radius)
    bg_grid = q <= config.tau_bg
    used, local_backgrounds, counts = [], [], []
    for token in fg:
        row, col = divmod(int(token), width)
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        local = np.argwhere(bg_grid[r0:r1, c0:c1])
        if not len(local):
            continue
        indices = (local[:, 0] + r0) * width + local[:, 1] + c0
        used.append(token)
        local_backgrounds.append(z[indices].mean(axis=0))
        counts.append(len(indices))
    used = np.asarray(used, dtype=np.int64)
    backgrounds = np.asarray(local_backgrounds, dtype=np.float32).reshape(-1, z.shape[1])
    vectors = z[used] - backgrounds
    return {"foreground_indices": used, "background_indices": np.flatnonzero(nearby_bg),
            "vectors": vectors, "local_background_centroids": backgrounds,
            "local_background_counts": np.asarray(counts, dtype=np.int32),
            "features": z, "n_foreground_all": int(len(fg)),
            "n_background_all": int(np.count_nonzero(q <= config.tau_bg)),
            "n_background_eligible": int(np.count_nonzero(nearby_bg)),
            "n_ambiguous": int(np.count_nonzero((q > config.tau_bg) & (q < config.tau_fg))),
            "n_skipped_no_local_background": int(len(fg) - len(used))}


def direction_alignment(image_summaries: Mapping[str, list[dict]], config: MetricConfig) -> dict:
    """Equal-image direction means; symmetric peer-to-peer cosine alignment."""
    names = sorted(image_summaries)
    if len(names) < 2:
        raise ValueError("At least two datasets are required")
    rows, unit = [], []
    for name in names:
        images = image_summaries[name]
        if not images:
            raise ValueError(f"{name}: no image has a usable local foreground/background pair")
        direction = np.mean([image["mean_vector"] for image in images], axis=0)
        magnitude = float(np.mean([image["mean_magnitude"] for image in images]))
        norm = float(np.linalg.norm(direction))
        if norm <= config.direction_epsilon:
            raise ValueError(f"{name}: mean local direction has zero magnitude; cosine is undefined")
        unit.append(direction / norm)
        rows.append({"dataset": name, "M_i": magnitude, "direction_norm": norm,
                     "n_images_used": len(images),
                     "n_local_vectors": sum(image["n_vectors"] for image in images),
                     "n_skipped_no_local_background": sum(image["n_skipped_no_local_background"] for image in images),
                     "mean_local_foreground": np.mean([image["mean_foreground"] for image in images], axis=0).tolist(),
                     "mean_local_background": np.mean([image["mean_background"] for image in images], axis=0).tolist()})
    matrix = np.clip(np.stack(unit) @ np.stack(unit).T, -1, 1)
    np.fill_diagonal(matrix, 1.0)
    for index, row in enumerate(rows):
        row["A_i"] = float((matrix[index].sum() - 1) / (len(names) - 1))
    upper = matrix[np.triu_indices(len(names), k=1)]
    return {"dataset_order": names, "per_dataset": rows,
            "pairwise_alignment": matrix.tolist(),
            "collection": {"A_collection": float(upper.mean()), "n_datasets": len(names)},
            "directions": {name: vector.tolist() for name, vector in zip(names, unit)}}


def image_split(names: list[str], *, seed: int, dataset: str,
                train_fraction: float, validation_fraction: float) -> dict[str, list[str]]:
    if len(names) < 3:
        raise ValueError(f"{dataset}: at least three images with both classes are needed for a probe")
    ordered = np.asarray(sorted(names), dtype=object)
    shuffled = ordered[_rng(seed, dataset, "probe_split").permutation(len(ordered))]
    n_train = max(1, min(len(names) - 2, round(len(names) * train_fraction)))
    n_val = max(1, min(len(names) - n_train - 1, round(len(names) * validation_fraction)))
    return {"train": shuffled[:n_train].tolist(),
            "validation": shuffled[n_train:n_train + n_val].tolist(),
            "test": shuffled[n_train + n_val:].tolist()}


def _balanced_accuracy(logits: np.ndarray, labels: np.ndarray, image_ids: np.ndarray) -> float:
    scores = []
    for image in np.unique(image_ids):
        chosen = image_ids == image
        y, predicted = labels[chosen], logits[chosen] >= 0
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError("Every scored image needs both patch classes")
        scores.append(float(((predicted[y == 1] == 1).mean() + (predicted[y == 0] == 0).mean()) / 2))
    return float(np.mean(scores))


def _probe_fit(features: np.ndarray, labels: np.ndarray, datasets: np.ndarray,
               images: np.ndarray, config: MetricConfig) -> tuple[np.ndarray, float]:
    import torch
    import torch.nn.functional as functional
    weights = np.zeros(len(labels), dtype=np.float64)
    dataset_names = np.unique(datasets)
    for dataset in dataset_names:
        for label in (0, 1):
            chosen = (datasets == dataset) & (labels == label)
            image_names = np.unique(images[chosen])
            if not len(image_names):
                raise ValueError(f"{dataset}: training probe has no class {label}")
            for image in image_names:
                selected = chosen & (images == image)
                weights[selected] = 1 / (len(dataset_names) * 2 * len(image_names) * selected.sum())
    x = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    sample_weight = torch.as_tensor(weights, dtype=torch.float64)
    coefficient = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficient, intercept], lr=1, max_iter=config.probe_max_iterations,
                                   line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (functional.binary_cross_entropy_with_logits(x @ coefficient + intercept, y,
                                                             reduction="none") * sample_weight).sum()
        loss = loss + config.probe_l2 * coefficient.square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    return coefficient.detach().numpy(), float(intercept.detach())


def run_linear_probes(samples: Mapping[str, Mapping[str, dict]], config: MetricConfig) -> tuple[dict, dict]:
    """Image-disjoint individual/joint probes with equal dataset/class/image weight."""
    names = sorted(samples)
    splits = {name: image_split(list(samples[name]), seed=config.seed, dataset=name,
                                train_fraction=config.probe_train_fraction,
                                validation_fraction=config.probe_validation_fraction) for name in names}

    def arrays(selected_datasets: list[str], part: str):
        x, y, ds, im = [], [], [], []
        for name in selected_datasets:
            for sample_id in splits[name][part]:
                record = samples[name][sample_id]
                for kind, label in (("background", 0), ("foreground", 1)):
                    values = record[kind]
                    x.append(values)
                    y.append(np.full(len(values), label, dtype=np.int8))
                    ds.append(np.full(len(values), name))
                    im.append(np.full(len(values), sample_id))
        return tuple(np.concatenate(parts) for parts in (x, y, ds, im))

    models = {}
    joint_x, joint_y, joint_ds, joint_im = arrays(names, "train")
    models["joint"] = _probe_fit(joint_x, joint_y, joint_ds, joint_im, config)
    for name in names:
        x, y, ds, im = arrays([name], "train")
        models[name] = _probe_fit(x, y, ds, im, config)
    rows = []
    for name in names:
        row = {"dataset": name, "n_train_images": len(splits[name]["train"]),
               "n_validation_images": len(splits[name]["validation"]),
               "n_test_images": len(splits[name]["test"])}
        for part in ("validation", "test"):
            x, y, _, im = arrays([name], part)
            for model_name, label in ((name, "individual"), ("joint", "joint")):
                coefficient, intercept = models[model_name]
                row[f"{part}_{label}_balanced_accuracy"] = _balanced_accuracy(x @ coefficient + intercept, y, im)
            row[f"{part}_joint_minus_individual"] = row[f"{part}_joint_balanced_accuracy"] - row[f"{part}_individual_balanced_accuracy"]
        rows.append(row)
    individual = np.stack([models[name][0] for name in names])
    norms = np.linalg.norm(individual, axis=1)
    denominator = norms[:, None] * norms[None, :]
    weight_cosine = np.divide(individual @ individual.T, denominator,
                              out=np.zeros_like(denominator), where=denominator > config.direction_epsilon)
    weight_cosine = [[float(value) if denominator[i, j] > config.direction_epsilon else None
                      for j, value in enumerate(line)] for i, line in enumerate(weight_cosine)]
    report = {"metric": "balanced_accuracy", "primary_split": "test", "per_dataset": rows,
              "individual_weight_cosine": weight_cosine,
              "split": splits, "method": "L2-regularized logistic regression; threshold 0; fixed regularization",
              "training_weighting": "equal dataset, class and image contribution"}
    weights = {"dataset_order": np.asarray(names), "joint_weight": models["joint"][0],
               "joint_intercept": np.asarray(models["joint"][1]),
               "individual_weights": individual,
               "individual_intercepts": np.asarray([models[name][1] for name in names])}
    return report, weights
