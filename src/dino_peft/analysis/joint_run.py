"""Analyze cached spatial features; the pretrained backbone is never loaded."""
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
from .joint_metric import MetricConfig, direction_alignment, local_directions, run_linear_probes, _rng
from .joint_report import alignment_triptych, join_performance, save_cloud_projection, summarize_results, write_json, write_report


REPORT_VERSION = 4


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


def analyze_cached_images(cache_dir: Path, entries: list[dict], config: MetricConfig):
    """One cache pass for exact local vectors and balanced per-image probe samples."""
    summaries, samples, identities, counts = {}, {}, {}, {}
    vectors, datasets, images, sample_ids, tokens, neighbor_counts = [], [], [], [], [], []
    for dataset_index, entry in enumerate(entries):
        name = entry["id"]
        summaries[name], samples[name] = [], {}
        identities[name] = {"foreground": [], "background": []}
        counts[name] = {"n_available": 0, "n_foreground_all": 0, "n_background_all": 0,
                        "n_background_eligible": 0, "n_ambiguous": 0,
                        "n_images_total": len(entry["shards"]), "n_images_skipped": 0,
                        "n_skipped_no_local_background": 0}
        offset = 0
        for image_index, shard in enumerate(entry["shards"]):
            z, q = read_shard(cache_dir, entry, shard)
            local = local_directions(z, q, shard["grid_shape"], config)
            for key in ("n_foreground_all", "n_background_all", "n_background_eligible", "n_ambiguous"):
                counts[name][key] += local[key]
            counts[name]["n_skipped_no_local_background"] += local["n_skipped_no_local_background"]
            counts[name]["n_available"] += len(q)
            fg, bg = local["foreground_indices"], local["background_indices"]
            if not len(fg) or not len(bg):
                counts[name]["n_images_skipped"] += 1
                offset += len(q)
                continue
            d = local["vectors"]
            summaries[name].append({"sample_id": shard["sample_id"], "n_vectors": len(d),
                                    "n_skipped_no_local_background": local["n_skipped_no_local_background"],
                                    "mean_vector": d.mean(axis=0, dtype=np.float64),
                                    "mean_magnitude": float(np.linalg.norm(d, axis=1).mean()),
                                    "mean_foreground": local["features"][fg].mean(axis=0, dtype=np.float64),
                                    "mean_background": local["local_background_centroids"].mean(axis=0, dtype=np.float64)})
            vectors.append(d)
            datasets.append(np.full(len(d), dataset_index, dtype=np.int16))
            images.append(np.full(len(d), image_index, dtype=np.int32))
            sample_ids.append(np.full(len(d), shard["sample_id"]))
            tokens.append(fg.astype(np.int32))
            neighbor_counts.append(local["local_background_counts"])
            samples[name][shard["sample_id"]] = {}
            for kind, candidates in (("foreground", fg), ("background", bg)):
                cap = min(config.probe_patches_per_image_per_class, len(candidates))
                chosen = np.sort(_rng(config.seed, name, shard["sample_id"], kind).choice(
                    candidates, size=cap, replace=False))
                samples[name][shard["sample_id"]][kind] = local["features"][chosen]
                width = shard["grid_shape"][1]
                identities[name][kind].extend({"index": int(offset + token),
                                               "sample_id": shard["sample_id"], "shard": shard["path"],
                                               "token_index": int(token), "row": int(token // width),
                                               "column": int(token % width)} for token in chosen)
            offset += len(q)
        if len(samples[name]) < 3:
            raise ValueError(f"{name}: only {len(samples[name])} images have local FG/BG pairs; three are needed for image-level probe splits")
    local_vectors = {"vectors": np.concatenate(vectors), "dataset_index": np.concatenate(datasets),
                     "image_index": np.concatenate(images), "foreground_token_index": np.concatenate(tokens),
                     "sample_id": np.concatenate(sample_ids),
                     "local_background_count": np.concatenate(neighbor_counts),
                     "dataset_order": np.asarray([entry["id"] for entry in entries])}
    return summaries, samples, identities, counts, local_vectors


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
                                          radius=config.near_foreground_radius_patches,
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
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch")}
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
    print(f"[joint-metric] {collection}/{backbone}/{run_id}: computing local directions from cached images", flush=True)
    summaries, samples, identities, counts, local_vectors = analyze_cached_images(cache_dir, entries, config)
    result = direction_alignment(summaries, config)
    print("[joint-metric] fitting frozen-feature individual and joint linear probes", flush=True)
    probe, probe_weights = run_linear_probes(samples, config)
    result["linear_probe"] = {key: value for key, value in probe.items() if key != "split"}
    result["config"] = asdict(config)
    result["warnings"] = []
    for row in result["per_dataset"]:
        name = row["dataset"]
        row.update(counts[name])
        row["foreground_coverage"] = row["n_local_vectors"] / row["n_foreground_all"] if row["n_foreground_all"] else 0.0
        row.update(next(probe_row for probe_row in probe["per_dataset"] if probe_row["dataset"] == name))
        row["n_probe_foreground"] = len(identities[name]["foreground"])
        row["n_probe_background"] = len(identities[name]["background"])
        print(f"[joint-metric] {name}: A={row['A_i']:.4f}, M={row['M_i']:.4f}, local vectors={row['n_local_vectors']}, skipped={row['n_skipped_no_local_background']}", flush=True)
    result.update({"schema_version": 2, "backbone": backbone, "collection_id": collection, "run_id": run_id,
                   "cache_dir": str(cache_dir), "cache_hash": manifest["extraction_hash"],
                   "fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "cache_metadata": {k: v for k, v in manifest.items() if k != "datasets"},
                   "index_relative": "../../../index.html",
                   "local_vectors_artifact": "local_vectors.npz", "probe_weights_artifact": "probe_weights.npz",
                   "probe_split_artifact": "probe_split.json"})
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
        write_json(stage / "probe_split.json", probe["split"])
        np.savez_compressed(stage / "local_vectors.npz", **local_vectors)
        np.savez_compressed(stage / "probe_weights.npz", **probe_weights)
        display = {}
        for name in sorted(samples):
            display[name] = {}
            for kind in ("foreground", "background"):
                cloud = np.concatenate([image[kind] for image in samples[name].values()])
                cap = min(512, len(cloud))
                display[name][kind] = cloud[_rng(config.seed, name, kind, "projection").choice(len(cloud), cap, replace=False)]
        print("[joint-metric] plotting sampled clouds and checking random image/mask alignments", flush=True)
        save_cloud_projection(result, display, stage)
        current_sources = {source["id"]: source for source in _resolve_sources(cfg)}
        result["alignment"] = save_alignment_previews(cache_dir, manifest, entries, identities, config, stage, current_sources)
        result["warnings"].extend(result["alignment"]["warnings"])
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
    """Repair current-format report previews from relocated source images."""
    components = components.resolve()
    output = components.parent
    result = json.loads(components.read_text())
    if result.get("schema_version") != 2:
        raise ValueError("Preview refresh supports only direction-alignment reports; keep historical reports unchanged")
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
