"""Training-free joint-training geometry; no backbone or cache I/O dependencies.

The barycenter is a free-support, fixed uniform-mass minimizer of the mean
debiased Sinkhorn divergence. Its support positions are optimized: this is
not a centroid or a pooled foreground distribution. Nonconvex optimization
supplies a reproducible approximation, not a global-optimum certificate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricConfig:
    tau_fg: float = 0.8
    tau_bg: float = 0.2
    n_samples_per_distribution: int = 512
    min_samples_per_distribution: int = 16
    seed: int = 0
    l2_normalize: bool = True
    pca_dim: int | None = None
    ot_epsilon: float = 0.05
    sinkhorn_scaling: float = 0.9
    barycenter_support: int = 64
    barycenter_max_iterations: int = 100
    barycenter_tolerance: float = 1e-5
    device: str = "cpu"
    dtype: str = "float64"
    risk_formula: str = "ratio"
    risk_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not 0 <= self.tau_bg < self.tau_fg <= 1:
            raise ValueError("Require 0 <= tau_bg < tau_fg <= 1.")
        for name in ("n_samples_per_distribution", "min_samples_per_distribution",
                     "barycenter_support", "barycenter_max_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.min_samples_per_distribution > self.n_samples_per_distribution:
            raise ValueError("min_samples_per_distribution exceeds the sampling cap.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for name in ("ot_epsilon", "barycenter_tolerance", "risk_epsilon"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not 0 < self.sinkhorn_scaling < 1:
            raise ValueError("sinkhorn_scaling must be between zero and one.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64.")
        if self.risk_formula not in ("ratio", "consensus"):
            raise ValueError("risk_formula must be ratio or consensus.")
        if self.pca_dim is not None and (
            not isinstance(self.pca_dim, int) or isinstance(self.pca_dim, bool)
            or self.pca_dim < 1
        ):
            raise ValueError("pca_dim must be a positive integer or null.")


def _rng(seed: int, dataset: str, purpose: str) -> np.random.Generator:
    """Stable independent streams; Python hash and caller order never enter seeds."""
    digest = hashlib.sha256(f"{seed}\0{dataset}\0{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _dataset_names(datasets: Mapping) -> list[str]:
    if len(datasets) < 2:
        raise ValueError("Joint compatibility requires at least two distinct datasets.")
    if any(not isinstance(name, str) or not name for name in datasets):
        raise ValueError("Dataset identifiers must be nonempty strings.")
    return sorted(datasets)


def sample_indices(
    occupancies: Mapping[str, np.ndarray], config: MetricConfig
) -> dict[str, Any]:
    """Return balanced FG/BG samples as original flattened cache indices.

    The smallest class across all datasets sets the common cap. Sampling is
    without replacement. Duplicating every atom preserves the full empirical
    measure, but capped random samples can differ through ordinary sampling
    variation. Exact duplication invariance of arbitrary capped samples is
    not claimed.
    """
    names = _dataset_names(occupancies)
    candidates, counts = {}, {}
    for name in names:
        q = np.asarray(occupancies[name])
        if q.ndim != 1 or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
            raise ValueError(f"{name}: occupancy must be a finite 1D array in [0, 1].")
        fg, bg = np.flatnonzero(q >= config.tau_fg), np.flatnonzero(q <= config.tau_bg)
        candidates[name] = {"foreground": fg, "background": bg}
        counts[name] = {
            "n_available": int(q.size), "n_foreground": int(fg.size),
            "n_background": int(bg.size), "n_ambiguous": int(q.size - fg.size - bg.size),
        }
        if min(fg.size, bg.size) < config.min_samples_per_distribution:
            raise ValueError(
                f"{name}: only {fg.size} foreground / {bg.size} background patches "
                f"at tau_fg={config.tau_fg}, tau_bg={config.tau_bg}; at least "
                f"{config.min_samples_per_distribution} of each are required. "
                "Inspect occupancy/mask alignment, add images, or explicitly adjust thresholds/minimum."
            )
    cap = min(config.n_samples_per_distribution,
              min(len(v) for dataset in candidates.values() for v in dataset.values()))
    sampled = {}
    for name in names:
        sampled[name] = {}
        for kind, values in candidates[name].items():
            sampled[name][kind] = np.sort(_rng(config.seed, name, kind).choice(
                values, size=cap, replace=False)).astype(np.int64)
            counts[name][f"n_sampled_{kind}"] = int(cap)
    return {"dataset_order": names, "indices": sampled, "counts": counts,
            "balanced_sample_count": int(cap)}


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name}: expected a nonempty [samples, dimensions] array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: embeddings contain NaN or infinity.")
    return array


def risk_from_components(
    consensus: Sequence[float], saliency: Sequence[float],
    formula: str = "ratio", epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Cheap isolated equation experiments from previously saved C_i and S_i."""
    c, s = np.asarray(consensus, dtype=np.float64), np.asarray(saliency, dtype=np.float64)
    if c.ndim != 1 or c.size < 2 or c.shape != s.shape:
        raise ValueError("Consensus and saliency must be equal-length vectors for at least two datasets.")
    if not np.isfinite(c).all() or not np.isfinite(s).all() or (c < 0).any() or (s < 0).any():
        raise ValueError("Consensus and saliency must contain finite nonnegative distances.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Risk epsilon must be positive and finite.")
    if formula == "ratio":
        r = c / (s + epsilon)
    elif formula == "consensus":
        r = c.copy()
    else:
        raise ValueError(f"Unknown risk formula {formula!r}; use ratio or consensus.")
    g, selective = float(r.mean()), float(r.std(ddof=0))
    if not np.isfinite(r).all() or not np.isfinite([g, selective]).all():
        raise FloatingPointError("Risk aggregation overflowed; inspect components and risk epsilon.")
    return {"r": r.tolist(), "G": g, "L": selective, "L_rel": selective / (g + epsilon)}


class SinkhornDistance:
    """GeomLoss divergence with the squared-Euclidean epsilon convention.

    GeomLoss p=2 uses C=|x-y|²/2 and epsilon=blur². Multiplying its
    divergence by two and setting blur=sqrt(ot_epsilon/2) gives our
    C=|x-y|² and ot_epsilon*KL(pi|a tensor b). All measures have unit mass.
    SamplesLoss uses epsilon scaling, not a marginal-residual tolerance.
    """
    def __init__(self, config: MetricConfig):
        import torch
        try:
            import geomloss
        except ImportError as exc:
            raise ImportError("Install the joint-metric extra: pip install -e '.[joint-metric]'.") from exc
        self.torch, self.geomloss, self.config = torch, geomloss, config
        self.dtype, self.blur = getattr(torch, config.dtype), math.sqrt(config.ot_epsilon / 2)
        self.clamped_negative_count, self.minimum_raw_distance = 0, 0.0

    def tensor(self, array: np.ndarray):
        return self.torch.as_tensor(np.array(array, copy=True), dtype=self.dtype, device=self.config.device)

    def tensor_distance(self, a, b):
        # Guard GeomLoss' log(diameter) schedule for identical point masses.
        with self.torch.no_grad():
            low = self.torch.minimum(a.amin(dim=0), b.amin(dim=0))
            high = self.torch.maximum(a.amax(dim=0), b.amax(dim=0))
            diameter = max(float((high - low).norm().item()), self.blur)
        loss = self.geomloss.SamplesLoss(
            loss="sinkhorn", p=2, blur=self.blur, diameter=diameter,
            scaling=self.config.sinkhorn_scaling, debias=True, backend="tensorized")
        return 2.0 * loss(a, b)

    def checked_value(self, value: float, label: str) -> float:
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite Sinkhorn divergence for {label}.")
        self.minimum_raw_distance = min(self.minimum_raw_distance, value)
        tolerance = 1e-5 if self.config.dtype == "float32" else 1e-8
        if value < -tolerance:
            raise FloatingPointError(
                f"Negative Sinkhorn divergence {value:g} for {label}. "
                "Increase sinkhorn_scaling toward 1, use float64, or increase ot_epsilon.")
        if value < 0:
            self.clamped_negative_count += 1
        return max(value, 0.0)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        x, y = _array(a, "distribution_a"), _array(b, "distribution_b")
        if x.shape[1] != y.shape[1]:
            raise ValueError("Distribution embedding dimensions differ.")
        with self.torch.no_grad():
            value = float(self.tensor_distance(self.tensor(x), self.tensor(y)).item())
        return self.checked_value(value, "distribution pair")

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "sinkhorn_divergence", "library": "geomloss",
            "library_version": getattr(self.geomloss, "__version__", "unknown"),
            "torch_version": self.torch.__version__, "backend": "tensorized",
            "ground_cost": "squared_euclidean", "ot_epsilon": self.config.ot_epsilon,
            "geomloss_blur": self.blur, "geomloss_output_multiplier": 2.0,
            "sinkhorn_scaling": self.config.sinkhorn_scaling,
            "inner_solver_convergence": "epsilon-scaling schedule; no marginal residual certificate",
            "empirical_mass": "uniform, total one per distribution",
            "dtype": self.config.dtype, "device": self.config.device,
            "negative_roundoff_clamps": self.clamped_negative_count,
            "minimum_raw_distance": self.minimum_raw_distance,
        }


def fit_barycenter(
    foreground: Mapping[str, np.ndarray], config: MetricConfig,
    distance: SinkhornDistance | None = None,
) -> dict[str, Any]:
    """Optimize equal-weight mean Sinkhorn divergence over uniform support atoms."""
    names = _dataset_names(foreground)
    arrays = {name: _array(foreground[name], name) for name in names}
    if len({a.shape[1] for a in arrays.values()}) != 1:
        raise ValueError("All foreground embedding dimensions must match.")
    solver = distance or SinkhornDistance(config)
    torch = solver.torch
    clouds = [solver.tensor(arrays[name]) for name in names]
    per_dataset = min(min(len(a) for a in arrays.values()), config.barycenter_support * 2)
    pool = np.concatenate([
        arrays[name][_rng(config.seed, name, "barycenter_pool").choice(
            len(arrays[name]), per_dataset, replace=False)] for name in names])
    size = min(config.barycenter_support, len(pool))
    # K-means++ seeding prevents repeated copies of an atom from initializing
    # every support point at the same position (an artificial symmetry trap).
    init_rng = _rng(config.seed, "", "barycenter_init")
    initial_indices = [int(init_rng.integers(len(pool)))]
    nearest_squared = np.full(len(pool), np.inf)
    for _ in range(1, size):
        nearest_squared = np.minimum(nearest_squared, np.square(pool - pool[initial_indices[-1]]).sum(axis=1))
        mass = nearest_squared.sum()
        if mass > 0:
            selected = int(init_rng.choice(len(pool), p=nearest_squared / mass))
        else:
            selected = int(init_rng.integers(len(pool)))
        initial_indices.append(selected)
    support = solver.tensor(pool[initial_indices]).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [support], lr=1.0, max_iter=1, max_eval=20,
        tolerance_grad=config.barycenter_tolerance / size,
        tolerance_change=0.0, history_size=10, line_search_fn="strong_wolfe")
    evaluations = 0

    def objective():
        return torch.stack([solver.tensor_distance(cloud, support) for cloud in clouds]).mean()

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        value = objective()
        if not torch.isfinite(value):
            raise FloatingPointError("Barycenter objective became nonfinite.")
        value.backward()
        if not torch.isfinite(support.grad).all():
            raise FloatingPointError("Barycenter gradient became nonfinite.")
        evaluations += 1
        return value

    objective_history, gradient_history = [], []
    converged, stop_reason, iterations = False, "max_iterations", 0
    for iteration in range(config.barycenter_max_iterations + 1):
        current = float(closure().detach().item())
        # Divide out atom mass so tolerance has comparable meaning across
        # support sizes. A stalled objective alone does not imply convergence.
        gradient_rms = float((support.grad.detach() * size).square().mean().sqrt().item())
        objective_history.append(current)
        gradient_history.append(gradient_rms)
        if gradient_rms <= config.barycenter_tolerance:
            converged, stop_reason = True, "mass_scaled_gradient_tolerance"
            break
        if iteration == config.barycenter_max_iterations:
            break
        optimizer.step(closure)
        iterations += 1
    return {
        "support": support.detach().cpu().numpy(), "weights": np.full(size, 1.0 / size),
        "diagnostics": {
            "method": "free_support_sinkhorn_divergence", "optimizer": "torch.optim.LBFGS",
            "objective": "mean_i S_epsilon(foreground_i, uniform_support)",
            "dataset_order": names, "dataset_weights": [1.0 / len(names)] * len(names),
            "support_size_requested": config.barycenter_support, "support_size": size,
            "support_mass": "fixed_uniform", "initialization": "seeded_kmeans_plus_plus_on_balanced_pool",
            "seed": config.seed, "iterations": iterations, "objective_evaluations": evaluations,
            "max_iterations": config.barycenter_max_iterations,
            "tolerance": config.barycenter_tolerance,
            "converged": converged, "stop_reason": stop_reason,
            "objective_history": objective_history, "gradient_rms_history": gradient_history,
            "initial_objective": objective_history[0], "final_objective": objective_history[-1],
            "final_mass_scaled_gradient_rms": gradient_history[-1],
            "global_optimum_certified": False,
        },
    }


def _transform_distributions(distributions: Mapping[str, Mapping], config: MetricConfig):
    names = _dataset_names(distributions)
    transformed, dimensions, zero_counts = {}, set(), {}
    for name in names:
        transformed[name], zero_counts[name] = {}, {}
        for kind in ("foreground", "background"):
            array = _array(distributions[name][kind], f"{name}/{kind}").copy()
            dimensions.add(array.shape[1])
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            zero_counts[name][kind] = int((norms[:, 0] == 0).sum())
            if config.l2_normalize:
                array /= np.where(norms > 0, norms, 1.0)
            transformed[name][kind] = array
    if len(dimensions) != 1:
        raise ValueError("All foreground/background embedding dimensions must match.")
    metadata = {"l2_normalize": config.l2_normalize, "zero_norm_counts": zero_counts,
                "input_dimension": dimensions.pop(), "pca_dim": config.pca_dim,
                "order": "L2 normalization, then shared balanced PCA; no post-PCA renormalization"}
    if config.pca_dim is not None:
        cap = min(len(a) for d in transformed.values() for a in d.values())
        pooled = np.concatenate([
            transformed[name][kind][_rng(config.seed, name, f"pca_{kind}").choice(
                len(transformed[name][kind]), cap, replace=False)]
            for name in names for kind in ("foreground", "background")])
        if config.pca_dim > min(pooled.shape[0] - 1, pooled.shape[1]):
            raise ValueError("pca_dim exceeds the common balanced sample's centered rank bound.")
        mean = pooled.mean(axis=0)
        _, singular, vectors = np.linalg.svd(pooled - mean, full_matrices=False)
        components = vectors[:config.pca_dim]
        pivots = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(len(components)), pivots])
        components = components * np.where(signs == 0, 1, signs)[:, None]
        for name in names:
            for kind in ("foreground", "background"):
                transformed[name][kind] = (transformed[name][kind] - mean) @ components.T
        total_variance = float(np.square(singular).sum())
        metadata.update({
            "pca_fit_samples_per_dataset_per_class": cap,
            "pca_fit_weighting": "equal datasets and equal foreground/background classes",
            "pca_mean": mean.tolist(), "pca_components": components.tolist(),
            "pca_explained_variance_ratio": (
                (np.square(singular[:config.pca_dim]) / total_variance).tolist()
                if total_variance > 0 else [0.0] * config.pca_dim),
        })
    metadata["output_dimension"] = next(iter(transformed.values()))["foreground"].shape[1]
    return transformed, metadata


def compute_metric(
    distributions: Mapping[str, Mapping[str, np.ndarray]], config: MetricConfig,
) -> dict[str, Any]:
    """Compute geometry on sampled arrays; never reads caches or runs models.

    Call sample_indices first for balanced sampling. Unequal lengths remain
    valid normalized measures with equal dataset weights and are reported.
    """
    names = _dataset_names(distributions)
    arrays, transform = _transform_distributions(distributions, config)
    solver = SinkhornDistance(config)
    foreground = {name: arrays[name]["foreground"] for name in names}
    pairwise = np.zeros((len(names), len(names)))
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            pairwise[i, j] = pairwise[j, i] = solver.distance(foreground[left], foreground[names[j]])
    barycenter = fit_barycenter(foreground, config, solver)
    c = [solver.distance(foreground[name], barycenter["support"]) for name in names]
    s = [solver.distance(foreground[name], arrays[name]["background"]) for name in names]
    c_pairwise = pairwise.sum(axis=1) / (len(names) - 1)
    risks = risk_from_components(c, s, config.risk_formula, config.risk_epsilon)
    pair_risks = risk_from_components(c_pairwise, s, config.risk_formula, config.risk_epsilon)
    warnings = []
    if not barycenter["diagnostics"]["converged"]:
        warnings.append("Barycenter reached the iteration limit without meeting its gradient tolerance; inspect diagnostics or increase iterations.")
    if any(value <= config.risk_epsilon for value in s):
        warnings.append("At least one saliency S_i is near zero; ratio risk is sensitive to risk_epsilon. Inspect C_i and S_i separately.")
    if any(count for ds in transform["zero_norm_counts"].values() for count in ds.values()):
        warnings.append("Zero-norm embeddings were retained as zeros; inspect the recorded counts and feature extraction.")
    if len({len(value) for ds in arrays.values() for value in ds.values()}) > 1:
        warnings.append("Unequal empirical sample counts supplied: normalized measures and equal dataset objective weights are used.")
    return {
        "dataset_order": names, "config": asdict(config), "transform": transform,
        "per_dataset": [
            {"dataset": name, "C_i": c[i], "S_i": s[i], "r_i": risks["r"][i],
             "C_pairwise": float(c_pairwise[i]), "r_pairwise": pair_risks["r"][i],
             "n_sampled_foreground": len(arrays[name]["foreground"]),
             "n_sampled_background": len(arrays[name]["background"])}
            for i, name in enumerate(names)],
        "collection": {
            "n_datasets": len(names), "G": risks["G"], "L": risks["L"], "L_rel": risks["L_rel"],
            "pairwise_G": pair_risks["G"], "pairwise_L": pair_risks["L"],
            "pairwise_L_rel": pair_risks["L_rel"]},
        "pairwise_foreground": pairwise.tolist(),
        "barycenter": {"support": barycenter["support"].tolist(),
                       "weights": barycenter["weights"].tolist(), **barycenter["diagnostics"]},
        "distance": solver.metadata(), "warnings": warnings,
    }
