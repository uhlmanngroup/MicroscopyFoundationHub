"""Stage B orchestration: stream cached arrays, compute metrics, and save provenance.

No backbone is loaded. Optional alignment previews reload a few source images.
Unselected embeddings are never concatenated: memory scales with occupancy
arrays and selected OT samples.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from dino_peft.config import deep_merge, load_config, repo_root
from dino_peft.utils.paths import write_run_info
from .joint_cache import _resolve_sources, _shard_path, load_cache_manifest, read_shard
from .joint_metric import MetricConfig, compute_metric, near_foreground_background, risk_from_components, sample_indices, _rng
from .joint_report import alignment_triptych, join_performance, save_cloud_projection, summarize_results, write_json, write_report


REPORT_VERSION = 3


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


def load_occupancies(cache_dir: Path, entries: list[dict], config: MetricConfig) -> tuple[dict, dict]:
    occupancies, near = {}, {}
    for entry in entries:
        chunks, nearby = [], []
        for shard in entry["shards"]:
            q = read_shard(cache_dir, entry, shard, embeddings=False)[1]
            chunks.append(q)
            if config.background_sampling == "near_foreground":
                nearby.append(near_foreground_background(
                    q, shard["grid_shape"], tau_fg=config.tau_fg,
                    tau_bg=config.tau_bg, radius=config.near_foreground_radius_patches))
        occupancies[entry["id"]] = np.concatenate(chunks)
        if nearby:
            near[entry["id"]] = np.concatenate(nearby)
    return occupancies, near


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


def save_alignment_previews(cache_dir: Path, manifest: dict, entries: list[dict], identities: dict,
                            config: MetricConfig, output: Path, current_sources: dict | None = None) -> dict:
    """Reload three reproducible random images per dataset; never run a model.

    Use the cache's extraction recipe, not today's training config. Verify file
    identity, geometry and occupancies before describing a preview as aligned.
    """
    from .joint_cache import _AlignedDataset, occupancy_on_grid
    report = {"seed": config.seed, "previews": [], "warnings": []}
    try:
        extraction = json.loads((cache_dir / "extraction_identity.json").read_text())
        sources = {source["id"]: source for source in extraction["sources"]}
        original_files = {p: (size, mtime) for p, size, mtime in extraction["inputs"]}
    except (OSError, KeyError, ValueError) as exc:
        report["warnings"].append(f"Alignment previews unavailable: {exc}")
        return report
    for entry in entries:
        name = entry["id"]
        try:
            source = sources[name]
            try:
                dataset = _AlignedDataset(source)
            except RuntimeError:
                current = (current_sources or {}).get(name)
                if not current or (current["image_dir"], current["mask_dir"]) == (source["image_dir"], source["mask_dir"]):
                    raise
                # Preserve the cache's preprocessing; only relocate its sources.
                dataset = _AlignedDataset({**source, "image_dir": current["image_dir"], "mask_dir": current["mask_dir"]})
            positions = {ip.relative_to(dataset.dataset.image_dir).as_posix(): i for i, (ip, _) in enumerate(dataset.dataset.pairs)}
            chosen = sorted(_rng(config.seed, name, "alignment_preview").choice(len(entry["shards"]), min(3, len(entry["shards"])), replace=False))
            for index in chosen:
                shard = entry["shards"][index]
                position = positions[shard["sample_id"]]
                for path, old_dir, new_dir in zip(dataset.dataset.pairs[position],
                                                   (source["image_dir"], source["mask_dir"]),
                                                   (dataset.dataset.image_dir, dataset.dataset.mask_dir)):
                    original = str(Path(old_dir) / path.relative_to(new_dir))
                    if original_files[original] != (path.stat().st_size, path.stat().st_mtime_ns):
                        raise ValueError(f"Source file changed since extraction: {path}")
                image, mask, meta = dataset[position]
                for key in ("sample_id", "original_image_shape", "processed_shape", "crop_box_xyxy"):
                    if meta[key] != shard[key]:
                        raise ValueError(f"Cached geometry differs for {shard['sample_id']}: {key}")
                grid = shard["grid_shape"]
                patch = None if manifest["backbone"]["name"] == "resnet50" else mask.shape[0] // grid[0]
                q = occupancy_on_grid(mask, grid, patch_size=patch)
                cached_q = read_shard(cache_dir, entry, shard, embeddings=False)[1]
                if not np.allclose(q, cached_q, rtol=0, atol=1e-6):
                    raise ValueError(f"Mask occupancy differs from cache: {shard['sample_id']}")
                selected = {kind: [p["token_index"] for p in points if p["shard"] == shard["path"]] for kind, points in identities[name].items()}
                relative = f"figs/alignment/{name}-{index:07d}.png"
                stats = alignment_triptych(output / relative, image, mask, q, grid,
                                          normalization=source["preprocessing"]["normalization"],
                                          tau_fg=config.tau_fg, tau_bg=config.tau_bg,
                                          title=f"{name} · {shard['sample_id']} · seed {config.seed}", selected=selected)
                report["previews"].append({"dataset": name, "sample_id": shard["sample_id"], "figure": relative,
                                           "source": "source_files", "relocated": str(dataset.dataset.image_dir) != source["image_dir"],
                                           "cache_occupancy_verified": True, "processed_shape": meta["processed_shape"],
                                           "crop_box_xyxy": meta["crop_box_xyxy"], **stats})
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            cached = 0
            for index, shard in enumerate(entry["shards"]):
                if "preview" not in shard:
                    continue
                path = _shard_path(cache_dir, shard["preview"])
                if not path.is_file():
                    continue
                relative = f"figs/alignment/{name}-cached-{index:07d}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                report["previews"].append({"dataset": name, "sample_id": shard["sample_id"],
                                           "figure": relative, "source": "cache_preview",
                                           "cache_occupancy_verified": False, "crop_box_xyxy": shard.get("crop_box_xyxy"),
                                           "near_black_pixel_fraction": None})
                cached += 1
            report["warnings"].append(
                f"{name}: could not verify source images for random alignment checks ({exc}); "
                + (f"showing {cached} triptychs saved during cache extraction, without sampled locations."
                   if cached else "no cached triptychs available."))
    return report


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
    occupancies, nearby = load_occupancies(cache_dir, entries, config)
    selection = sample_indices(occupancies, config, near_background=nearby)
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
        print("[joint-metric] plotting sampled clouds and checking random image/mask alignments", flush=True)
        save_cloud_projection(result, distributions, stage)
        current_sources = {source["id"]: source for source in _resolve_sources(cfg)}
        result["alignment"] = save_alignment_previews(cache_dir, manifest, entries, identities, config, stage, current_sources)
        result["warnings"].extend(result["alignment"]["warnings"])
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


def refresh_alignment_report(components: Path, joint_cfg: dict) -> Path:
    """Repair report previews from relocated sources without rerunning OT."""
    components = components.resolve()
    output = components.parent
    result = json.loads(components.read_text())
    cache_dir = Path(result["cache_dir"])
    manifest = load_cache_manifest(cache_dir)
    entries_by_id = {entry["id"]: entry for entry in manifest["datasets"]}
    entries = [entries_by_id[name] for name in result["dataset_order"]]
    identities = json.loads((output / "samples.json").read_text())["datasets"]
    collection = result["collection_id"].split("__", 1)[0]
    current_sources = {}
    if collection in joint_cfg["collections"]:
        for backbone in joint_cfg["backbones"]:
            experiment = resolve_experiment(joint_cfg, collection, backbone)
            if experiment["backbone_id"] == result["backbone"]:
                current_sources = {source["id"]: source for source in _resolve_sources(experiment)}
                break
    old_warnings = set(result.get("alignment", {}).get("warnings", []))
    result["warnings"] = [warning for warning in result.get("warnings", []) if warning not in old_warnings]
    result["alignment"] = save_alignment_previews(
        cache_dir, manifest, entries, identities, MetricConfig(**result["config"]), output,
        current_sources)
    result["warnings"].extend(result["alignment"]["warnings"])
    result["alignment_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(result, output)
    summarize_results(output.parents[2])
    return output


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
    for name in ("samples.json", "barycenter.npz", "pca.npz", "cloud_projection.npz"):
        path = source_path.parent / name
        if path.is_file():
            shutil.copy2(path, output / name)
    alignment_dir = source_path.parent / "figs" / "alignment"
    if alignment_dir.is_dir():
        shutil.copytree(alignment_dir, output / "figs" / "alignment")
    (output / "config_used.yaml").write_text(yaml.safe_dump({"source_components": str(source_path), "metric": result["config"]}, sort_keys=True))
    write_report(result, output)
    return output
