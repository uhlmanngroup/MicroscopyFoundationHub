"""Stage B orchestration: stream cached arrays, compute metrics, and save provenance.

No backbone or dataset loader is used by this module. Unselected embeddings are
never concatenated: memory scales with occupancy arrays and selected OT samples.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from dino_peft.config import deep_merge, load_config, repo_root
from dino_peft.utils.paths import write_run_info
from .joint_cache import load_cache_manifest, read_shard
from .joint_metric import MetricConfig, compute_metric, risk_from_components, sample_indices
from .joint_report import join_performance, summarize_results, write_json, write_report


REPORT_VERSION = 1


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _safe_id(value: object, label: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*", value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a nonempty filename-safe identifier: {value!r}")
    return value


def resolve_experiment(cfg: dict, collection: str, backbone: str) -> dict:
    """Reuse the project's training config paths, pairing and backbone blocks."""
    if collection not in cfg["collections"] or backbone not in cfg["backbones"]:
        raise ValueError(f"Unknown collection/backbone: {collection}/{backbone}")
    group = cfg["collections"][collection]
    reference = Path(cfg["backbones"][backbone])
    model_cfg = load_config(reference if reference.is_absolute() else repo_root() / reference)
    model = model_cfg["backbone"]
    variant = model.get("variant", model.get("model", ""))
    backbone_id = backbone if backbone == "resnet50" else f"{backbone}_{str(variant).lower().replace('-', '')}"
    datasets = []
    for source in group["datasets"]:
        source = dict(source)
        if group.get("training_cfg"):
            source.setdefault("training_cfg", group["training_cfg"])
        datasets.append(source)
    return {
        "collection_id": collection, "backbone_id": backbone_id, "backbone": model,
        "cache_dir": str(Path(cfg["cache_root"]) / collection / backbone_id),
        "results_root": cfg["results_root"], "datasets": datasets,
        "preprocessing": deep_merge({"img_size": model_cfg["img_size"]}, group.get("preprocessing", {})),
        "seed": cfg.get("seed", 42), "extraction": cfg.get("extraction", {}),
        "metric": cfg.get("metric", {}),
        **({"performance_csv": cfg["performance_csv"]} if cfg.get("performance_csv") else {}),
    }


def load_occupancies(cache_dir: Path, entries: list[dict]) -> dict:
    return {entry["id"]: np.concatenate([
        read_shard(cache_dir, entry, shard, embeddings=False)[1]
        for shard in entry["shards"]]) for entry in entries}


def load_selected(cache_dir: Path, entries: list[dict], selection: dict) -> tuple[dict, dict]:
    distributions, identities = {}, {}
    for entry in entries:
        name, dim = entry["id"], entry["embedding_dim"]
        indices = selection["indices"][name]
        distributions[name] = {kind: np.empty((len(ix), dim), dtype=np.float32) for kind, ix in indices.items()}
        identities[name] = {kind: [None] * len(ix) for kind, ix in indices.items()}
        start = 0
        for shard in entry["shards"]:
            stop = start + shard["num_embeddings"]
            needed = {kind: np.flatnonzero((ix >= start) & (ix < stop)) for kind, ix in indices.items()}
            if any(len(pos) for pos in needed.values()):
                z, _ = read_shard(cache_dir, entry, shard)
                width = shard["grid_shape"][1]
                for kind, positions in needed.items():
                    local = indices[kind][positions] - start
                    distributions[name][kind][positions] = z[local]
                    for pos, token in zip(positions, local):
                        identities[name][kind][int(pos)] = {
                            "index": int(start + token), "sample_id": shard["sample_id"],
                            "shard": shard["path"], "token_index": int(token),
                            "row": int(token // width), "column": int(token % width)}
            start = stop
        if any(any(v is None for v in values) for values in identities[name].values()):
            raise ValueError(f"Sampled indices outside cache for {name}")
    return distributions, identities


def _implementation_hash() -> str:
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in ("joint_metric.py", "joint_run.py", "joint_report.py"))).hexdigest()


def run_one(cfg: dict, *, update_index: bool = True) -> Path:
    cache_dir = Path(cfg["cache_dir"]).expanduser().resolve()
    config = MetricConfig(**cfg.get("metric", {}))
    manifest = load_cache_manifest(cache_dir, verify_checksums=True)
    # Constructing the dataclass validates settings before reading large arrays.
    collection = _safe_id(cfg["collection_id"], "collection_id")
    backbone = _safe_id(cfg.get("backbone_id", manifest.get("backbone_id", "")), "backbone_id")
    if manifest.get("backbone_id") and manifest["backbone_id"] != backbone:
        raise ValueError(f"Requested backbone_id {backbone!r} does not match cache {manifest['backbone_id']!r}")
    all_entries = {d["id"]: d for d in manifest["datasets"]}
    requested = cfg.get("dataset_ids", sorted(all_entries))
    if not isinstance(requested, list) or len(requested) < 2 or len(set(requested)) != len(requested):
        raise ValueError("dataset_ids must select at least two distinct datasets")
    unknown = set(requested) - set(all_entries)
    if unknown:
        raise ValueError(f"Unknown datasets in collection: {sorted(unknown)}")
    entries = [all_entries[name] for name in sorted(requested)]
    perf_path = cfg.get("performance_csv")
    perf_hash = hashlib.sha256(Path(perf_path).read_bytes()).hexdigest() if perf_path else None
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "geomloss")}
    fingerprint = {"versions": versions, "report_version": REPORT_VERSION, "implementation_hash": _implementation_hash(), "metric": asdict(config), "cache_hash": manifest["extraction_hash"], "manifest_hash": digest(manifest), "collection_id": collection, "dataset_ids": sorted(requested), "backbone": backbone, "performance_hash": perf_hash}
    run_id = digest(fingerprint)[:16]
    root = Path(cfg["results_root"]).expanduser().resolve()
    output = root / collection / backbone / run_id
    if (output / "components.json").is_file():
        previous = json.loads((output / "components.json").read_text())
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Run identity collision at {output}")
        print(f"[joint-metric] reuse {output}", flush=True)
        if update_index:
            summarize_results(root)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[joint-metric] {collection}/{backbone}/{run_id}: counting cached occupancies", flush=True)
    occupancies = load_occupancies(cache_dir, entries)
    selection = sample_indices(occupancies, config)
    distributions, identities = load_selected(cache_dir, entries, selection)
    del occupancies
    for name, counts in selection["counts"].items():
        print(f"[joint-metric] {name}: {counts}", flush=True)
    print("[joint-metric] computing Sinkhorn components and equal-weight barycenter", flush=True)
    result = compute_metric(distributions, config)
    for row in result["per_dataset"]:
        row.update(selection["counts"][row["dataset"]])
    result.update({"schema_version": 1, "backbone": backbone, "collection_id": collection, "run_id": run_id, "cache_dir": str(cache_dir), "cache_hash": manifest["extraction_hash"], "fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(), "cache_metadata": {k: v for k, v in manifest.items() if k != "datasets"}, "balanced_sample_count": selection["balanced_sample_count"], "index_relative": "../../../index.html"})
    if perf_path:
        join_performance(result, perf_path)
    # Atomic directory publication: incomplete/failed runs never appear in indexes.
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-", dir=output.parent) as tmp:
        stage = Path(tmp)
        resolved = deepcopy(cfg)
        resolved["metric"] = asdict(config)
        resolved["dataset_ids"] = sorted(requested)
        resolved["cache_dir"] = str(cache_dir)
        resolved["results_root"] = str(root)
        (stage / "config_used.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
        write_json(stage / "samples.json", {"seed": config.seed, "cache_hash": manifest["extraction_hash"], "datasets": identities})
        bary = result.get("barycenter", {})
        if "support" in bary:
            support = bary.pop("support")
            weights = bary.pop("weights", None)
            np.savez_compressed(stage / "barycenter.npz", support=np.asarray(support), weights=np.asarray(weights))
            bary["artifact"] = "barycenter.npz"
        transform = result.get("transform", {})
        if "pca_components" in transform:
            np.savez_compressed(stage / "pca.npz", components=np.asarray(transform.pop("pca_components")), mean=np.asarray(transform.pop("pca_mean")))
            transform["artifact"] = "pca.npz"
        write_run_info(stage, {"cache_hash": manifest["extraction_hash"], "run_id": run_id})
        write_report(result, stage)
        if output.exists():
            raise FileExistsError(f"Run directory already exists; refusing to overwrite: {output}")
        os.rename(stage, output)
    print(f"[joint-metric] report: {output / 'index.html'}", flush=True)
    if update_index:
        summarize_results(root)
    return output


def expand_runs(cfg: dict, sweep: dict | None = None):
    sweep = deepcopy(sweep if sweep is not None else cfg.get("sweep", {}))
    if not isinstance(sweep, dict):
        raise ValueError("sweep must map metric.<parameter> to a nonempty list")
    keys = sorted(sweep)
    for key in keys:
        if not key.startswith("metric.") or key.count(".") != 1 or not isinstance(sweep[key], list) or not sweep[key]:
            raise ValueError(f"Invalid sweep axis: {key!r}; use metric.<parameter>: [values]")
    for values in itertools.product(*(sweep[k] for k in keys)):
        current = deepcopy(cfg)
        current.pop("sweep", None)
        for key, value in zip(keys, values):
            current.setdefault("metric", {})[key.split(".")[1]] = value
        yield current


def run_from_config(cfg: dict, *, sweep: dict | None = None, update_index: bool = True) -> list[Path]:
    runs = list(expand_runs(cfg, sweep))
    for run in runs:
        MetricConfig(**run.get("metric", {}))
    paths = []
    try:
        for run in runs:
            paths.append(run_one(run, update_index=False))
    finally:
        if update_index:
            summarize_results(Path(cfg["results_root"]))
    return paths


def recompute(components: Path, output: Path, *, formula: str = "ratio", epsilon: float = 1e-8, consensus: str = "barycenter", performance_csv: str | None = None) -> Path:
    source = json.loads(components.read_text())
    if consensus not in {"barycenter", "pairwise"}:
        raise ValueError("consensus must be barycenter or pairwise")
    result = deepcopy(source)
    rows = result["per_dataset"]
    # C_i is kept as the barycenter raw component even when a different consensus is selected.
    key = "C_i" if consensus == "barycenter" else "C_pairwise"
    risk = risk_from_components([r[key] for r in rows], [r["S_i"] for r in rows], formula=formula, epsilon=epsilon)
    pair = risk_from_components([r["C_pairwise"] for r in rows], [r["S_i"] for r in rows], formula=formula, epsilon=epsilon)
    for i, row in enumerate(rows):
        row["r_i"], row["r_pairwise"] = risk["r"][i], pair["r"][i]
    result["collection"].update({k: risk[k] for k in ("G", "L", "L_rel")})
    result["collection"].update({"pairwise_" + k: pair[k] for k in ("G", "L", "L_rel")})
    result["config"].update({"risk_formula": formula, "risk_epsilon": epsilon, "risk_consensus": consensus})
    source_path = components.resolve()
    result["index_relative"] = None
    result["postprocess_source"] = str(source_path)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["source_fingerprint"] = result.get("fingerprint")
    result["fingerprint"] = {"source_components_sha256": hashlib.sha256(components.read_bytes()).hexdigest(),
                             "formula": formula, "epsilon": epsilon, "consensus": consensus,
                             "implementation_hash": _implementation_hash(),
                             "performance_csv_sha256": hashlib.sha256(Path(performance_csv).read_bytes()).hexdigest() if performance_csv else None}
    result["run_id"] = digest(result["fingerprint"])[:16]
    if performance_csv:
        join_performance(result, performance_csv)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    import shutil
    for name in ("samples.json", "barycenter.npz", "pca.npz"):
        path = source_path.parent / name
        if path.is_file():
            shutil.copy2(path, output / name)
    (output / "config_used.yaml").write_text(yaml.safe_dump({"source_components": str(source_path), "metric": result["config"]}, sort_keys=True))
    write_report(result, output)
    return output
