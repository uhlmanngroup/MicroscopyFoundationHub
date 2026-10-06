"""Frozen spatial vectors with continuous mask occupancy; no metric decisions.

A committed cache contains manifest.json and per-image NPZ shards. Vectors are
raw adapter outputs in row-major grid order. Reading caches requires only NumPy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import numpy as np

SCHEMA_VERSION = 1
EXTRACTION_VERSION = "joint-spatial-cache-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_path(cache_dir: Path, relative: str) -> Path:
    path = (cache_dir / relative).resolve()
    if not path.is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Cache shard escapes cache directory: {relative}")
    return path


def load_cache_manifest(cache_dir: str | Path, *, verify_checksums=False) -> dict:
    """Single validator shared by extraction reuse and metric runs."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or not manifest.get("complete"):
        raise ValueError(f"Unsupported or incomplete spatial cache at {cache_dir}")
    entries = manifest.get("datasets", [])
    if not entries or len({d["id"] for d in entries}) != len(entries):
        raise ValueError("Cache requires unique datasets")
    if len({d["embedding_dim"] for d in entries}) != 1:
        raise ValueError("Cache embedding dimensions differ")
    for entry in entries:
        shards = entry["shards"]
        if not shards or sum(s["num_embeddings"] for s in shards) != entry["num_embeddings"]:
            raise ValueError(f"Invalid shard counts for {entry['id']}")
        if len({s["sample_id"] for s in shards}) != len(shards):
            raise ValueError(f"Duplicate sample IDs for {entry['id']}")
        for shard in shards:
            path = _shard_path(cache_dir, shard["path"])
            if not path.is_file():
                raise ValueError(f"Missing cache shard: {path}")
            if verify_checksums and _sha256(path) != shard["sha256"]:
                raise ValueError(f"Cache checksum mismatch: {path}")
    return manifest


def read_shard(cache_dir, entry, shard, *, embeddings=True):
    """Load just occupancies, or one image's vectors as well; shared by Stage B."""
    with np.load(_shard_path(Path(cache_dir), shard["path"]), allow_pickle=False) as data:
        q, grid = data["occupancy"], data["grid_shape"].tolist()
        z = data["embeddings"] if embeddings else None
    n = shard["num_embeddings"]
    if q.shape != (n,) or grid != shard["grid_shape"] or int(np.prod(grid)) != n:
        raise ValueError(f"Invalid grid/occupancy shape: {shard['path']}")
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError(f"Invalid occupancy: {shard['path']}")
    if z is not None and (z.shape != (n, entry["embedding_dim"]) or not np.isfinite(z).all()):
        raise ValueError(f"Invalid embeddings: {shard['path']}")
    return z, q


def occupancy_on_grid(mask, grid_shape, *, patch_size: int | None = None):
    """Exact ViT patch averages or CNN adaptive spatial-bin area approximation.

    CNN bins describe spatial location, not overlapping receptive fields. ViT
    trailing pixels ignored by a patch convolution are ignored here too.
    """
    import torch
    import torch.nn.functional as functional
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim != 2 or not torch.isfinite(mask).all() or torch.any((mask < 0) | (mask > 1)):
        raise ValueError("Mask must be a finite HxW array in [0,1]")
    gh, gw = (int(v) for v in grid_shape)
    if gh <= 0 or gw <= 0:
        raise ValueError("Feature-grid dimensions must be positive")
    source = mask[None, None]
    if patch_size is not None:
        patch_size = int(patch_size)
        if patch_size <= 0 or (mask.shape[0] // patch_size, mask.shape[1] // patch_size) != (gh, gw):
            raise ValueError("Feature grid disagrees with exact patch footprint")
        result = functional.avg_pool2d(source, patch_size, patch_size)
    else:
        result = functional.adaptive_avg_pool2d(source, (gh, gw))
    return result[0, 0].reshape(-1).numpy()


def _resolve_sources(cfg: Mapping[str, Any]) -> list[dict]:
    from dino_peft.config import deep_merge, load_config, repo_root
    from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG
    sources = cfg.get("datasets")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config needs a nonempty datasets list")
    resolved = []
    for source in sources:
        source = dict(source)
        training = {}
        if source.get("training_cfg"):
            path = Path(source["training_cfg"]).expanduser()
            if not path.is_absolute():
                path = repo_root() / path
            if not path.is_file():
                raise FileNotFoundError(f"Training config not found: {path}")
            training = load_config(path)
        modality = str(source.get("modality", training.get("modality", cfg.get("modality", "em")))).lower()
        pp = {"img_size": training.get("img_size", cfg.get("img_size", DEFAULT_IMG_SIZE_CFG)),
              "center_crop_size": None, "clahe_norm": training.get("clahe_norm", False),
              "normalization": "training", "binarize": training.get("binarize", True),
              "binarize_threshold": training.get("binarize_threshold", 128)}
        if modality in ("deepbacs", "monusac"):
            pp.update(img_size={"mode": "native"}, center_crop_size=training.get("center_crop_size", 448))
        pp = deep_merge(deep_merge(pp, cfg.get("preprocessing", {})), source.get("preprocessing", {}))
        if pp["normalization"] not in ("training", "openclip_native"):
            raise ValueError("preprocessing.normalization must be training or openclip_native")
        dataset = deep_merge(training.get("dataset", {"type": "paired", "params": {}}), source.get("dataset", {}))
        image_dir = source.get("image_dir", source.get("img_dir", training.get("train_img_dir")))
        mask_dir = source.get("mask_dir", training.get("train_mask_dir"))
        if not image_dir or not mask_dir:
            raise ValueError(f"Dataset {source.get('id')} needs image_dir and mask_dir")
        identifier = str(source["id"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", identifier):
            raise ValueError(f"Unsafe dataset ID: {identifier!r}")
        resolved.append({"id": identifier, "image_dir": str(Path(image_dir).expanduser().resolve()),
                         "mask_dir": str(Path(mask_dir).expanduser().resolve()), "dataset": dataset,
                         "preprocessing": pp, "modality": modality})
    if len({d["id"] for d in resolved}) != len(resolved):
        raise ValueError("Dataset IDs must be unique")
    return sorted(resolved, key=lambda d: d["id"])


class _AlignedDataset:
    """Reuse training image preprocessing while propagating a fractional mask."""
    def __init__(self, source):
        import inspect
        from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset, _list_files
        from dino_peft.datasets.lucchi_seg import LucchiSegDataset
        from dino_peft.datasets.droso_seg import DrosoSegDataset
        from dino_peft.utils.transforms import em_seg_transforms, openclip_native_transforms
        classes = {"paired": PairedDirsSegDataset, "lucchi": LucchiSegDataset, "droso": DrosoSegDataset}
        kind = source["dataset"].get("type", "paired")
        if kind not in classes:
            raise ValueError(f"Unsupported dataset type {kind!r}")
        cls, pp = classes[kind], source["preprocessing"]
        params = dict(source["dataset"].get("params", {}))
        params.update(img_size=pp["img_size"], center_crop_size=pp["center_crop_size"], to_rgb=True,
                      transform=None, binarize=pp["binarize"], binarize_threshold=pp["binarize_threshold"])
        unknown = set(params) - set(inspect.signature(cls).parameters)
        if unknown:
            raise ValueError(f"Unsupported dataset parameters: {sorted(unknown)}")
        self.dataset = cls(source["image_dir"], source["mask_dir"], **params)
        recursive = params.get("recursive", kind == "droso")
        images = _list_files(Path(source["image_dir"]), recursive)
        masks = _list_files(Path(source["mask_dir"]), recursive)
        pairs = self.dataset.pairs
        if len(pairs) != len(images) or len(pairs) != len(masks) or len({m for _, m in pairs}) != len(pairs):
            raise ValueError(f"{source['id']}: incomplete or ambiguous pairing: {len(images)} images, {len(masks)} masks, {len(pairs)} pairs")
        self.source = source
        self.transform = em_seg_transforms(clahe_norm=bool(pp["clahe_norm"]))
        if pp["normalization"] == "openclip_native":
            if pp["clahe_norm"]:
                raise ValueError("openclip_native with clahe_norm is unsupported")
            self.transform = openclip_native_transforms(img_size={"mode": "native"})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        import torch.nn.functional as functional
        from dino_peft.utils.image_loading import open_first_frame, center_crop_box
        image_path, mask_path = self.dataset.pairs[index]
        with open_first_frame(image_path) as original_image:
            original_image_shape = [original_image.height, original_image.width]
        with open_first_frame(mask_path) as original_mask:
            original_mask_shape = [original_mask.height, original_mask.width]
            if original_mask_shape != original_image_shape:
                raise ValueError(f"Image/mask geometry mismatch: {image_path} vs {mask_path}")
            if original_mask.mode not in ("L", "I;16", "I"):
                original_mask = original_mask.convert("L")
            values = np.array(original_mask)
        pp = self.source["preprocessing"]
        foreground = values > (float(pp["binarize_threshold"]) if pp["binarize"] else 0)
        original_fraction = float(foreground.mean())
        crop_box = None
        if self.dataset.center_crop_size is not None:
            h, w = self.dataset.center_crop_size
            crop_box = center_crop_box(width=original_image_shape[1], height=original_image_shape[0], crop_w=w, crop_h=h)
            left, top, right, bottom = crop_box
            foreground = foreground[top:bottom, left:right]
        # Exactly training's RGB conversion, crop, bicubic image resize; no aug.
        image, _, _ = self.dataset[index]
        shape = [image.height, image.width]
        mask = torch.from_numpy(foreground.astype(np.float32))
        if list(mask.shape) != shape:
            mask = functional.interpolate(mask[None, None], size=shape, mode="area")[0, 0]
        metadata = {"sample_id": image_path.relative_to(self.dataset.image_dir).as_posix(),
                    "image_path": str(image_path), "mask_path": str(mask_path),
                    "original_image_shape": original_image_shape, "original_mask_shape": original_mask_shape,
                    "processed_shape": shape, "crop_box_xyxy": crop_box,
                    "foreground_fraction_original": original_fraction,
                    "foreground_fraction_processed": float(mask.mean())}
        return self.transform(image), mask, metadata


def _state_sha256(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_preview(path, image, mask, occupancy, grid_shape, normalization):
    from PIL import Image, ImageDraw
    from dino_peft.utils.transforms import IMAGENET_MEAN, IMAGENET_STD, OPENCLIP_MEAN, OPENCLIP_STD
    mean, std = (OPENCLIP_MEAN, OPENCLIP_STD) if normalization == "openclip_native" else (IMAGENET_MEAN, IMAGENET_STD)
    rgb = image.numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)
    rgb = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255))
    foreground = Image.fromarray(np.uint8(np.clip(mask.numpy(), 0, 1) * 255)).convert("RGB")
    q_image = Image.fromarray(np.uint8(occupancy.reshape(grid_shape) * 255)).resize(rgb.size, Image.Resampling.NEAREST).convert("RGB")
    width = min(512, rgb.width)
    height = max(1, round(rgb.height * width / rgb.width))
    canvas = Image.new("RGB", (width * 3, height + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip((rgb, foreground, q_image), ("Processed image", "Area-resized foreground", "Grid occupancy q [0,1]"))):
        canvas.paste(panel.resize((width, height), Image.Resampling.NEAREST), (i * width, 30))
        draw.text((i * width + 5, 8), label, fill="black")
    canvas.save(path)


def extract_joint_cache(cfg: Mapping[str, Any], *, backbone_factory=None) -> Path:
    """Save all raw spatial vectors once. Publish only a complete, valid cache."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from dino_peft.backbones import resolve_backbone_cfg, build_backbone
    if cfg.get("checkpoint") or cfg.get("use_lora") or cfg.get("full_finetune"):
        raise ValueError("Pretrained-only extraction does not accept trained checkpoints or tuning flags")
    cache_dir = Path(cfg["cache_dir"]).expanduser().resolve()
    sources = _resolve_sources(cfg)
    datasets = [_AlignedDataset(source) for source in sources]
    backbone = resolve_backbone_cfg(cfg)
    name = backbone["name"]
    if name in ("resnet50", "openclip") and not backbone.get("weights") and str(backbone.get("pretrained", "")).lower() in ("", "none", "null", "random"):
        raise ValueError(f"{name}: random initialization is not a pretrained representation")
    if name == "dinov2" and backbone.get("weights"):
        raise ValueError("The existing DINOv2 adapter cannot honor backbone.weights")
    extraction = {"device": "cuda", "preview_samples": 3, **cfg.get("extraction", {})}
    seed = int(cfg.get("seed", 42))
    # Track actual inputs and the local checkpoint, not metric thresholds/sampling.
    files = [p for ds in datasets for pair in ds.dataset.pairs for p in pair]
    checkpoint = backbone.get("weights") or backbone.get("pretrained")
    identity = {
        "version": EXTRACTION_VERSION, "code_hash": _sha256(Path(__file__)),
        "sources": sources, "backbone": backbone, "seed": seed,
        "inputs": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        "checkpoint_sha256": _sha256(Path(checkpoint).expanduser()) if checkpoint and Path(checkpoint).expanduser().is_file() else None,
        "torch_version": torch.__version__,
    }
    extraction_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache_dir.exists():
        manifest = load_cache_manifest(cache_dir, verify_checksums=True)
        if manifest["extraction_hash"] != extraction_hash:
            raise ValueError(f"Cache inputs/config changed at {cache_dir}; choose a new cache_dir")
        print(f"[joint cache] Reusing {cache_dir}")
        return cache_dir
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_dir.parent / f".{cache_dir.name}.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        with tempfile.TemporaryDirectory(prefix=f".{cache_dir.name}-", dir=cache_dir.parent) as tmp:
            stage = Path(tmp)
            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            model = (backbone_factory or build_backbone)(backbone, device=extraction["device"])
            model.eval().requires_grad_(False)
            layer = "layer4 spatial feature map" if name == "resnet50" else "final normalized spatial patch tokens, excluding special tokens"
            manifest = {
                "schema_version": SCHEMA_VERSION, "complete": True, "extraction_hash": extraction_hash,
                "backbone_id": cfg["backbone_id"], "collection_id": cfg["collection_id"],
                "backbone": {**backbone, "frozen": True, "layer": layer, "loaded_state_sha256": _state_sha256(model)},
                "extraction": {**extraction, "seed": seed, "dtype": "float32"},
                "preprocessing": {s["id"]: s["preprocessing"] for s in sources}, "datasets": [],
            }
            for source, dataset in zip(sources, datasets):
                folder = stage / source["id"]
                folder.mkdir()
                entry = {"id": source["id"], "num_images": len(dataset), "num_embeddings": 0, "embedding_dim": None, "shards": []}
                for i in range(len(dataset)):
                    image, mask, metadata = dataset[i]
                    with torch.inference_mode():
                        output = model(image.unsqueeze(0).to(extraction["device"]))
                    z = output.patch_tokens[0].float().cpu().numpy()
                    grid = list(output.grid_size)
                    q = occupancy_on_grid(mask, grid, patch_size=None if name == "resnet50" else model.patch_size)
                    if z.ndim != 2 or len(z) != len(q) or not np.isfinite(z).all():
                        raise ValueError("Backbone features do not match the spatial grid or contain invalid values")
                    if entry["embedding_dim"] not in (None, z.shape[1]):
                        raise ValueError("Embedding dimension changed during extraction")
                    entry["embedding_dim"] = z.shape[1]
                    path = folder / f"{i:07d}.npz"
                    np.savez_compressed(path, embeddings=z, occupancy=q, grid_shape=grid)
                    shard = {**metadata, "path": path.relative_to(stage).as_posix(), "sha256": _sha256(path),
                             "num_embeddings": len(q), "grid_shape": grid, "occupancy_mean": float(q.mean())}
                    if i < int(extraction["preview_samples"]):
                        preview = folder / f"{i:07d}.png"
                        _write_preview(preview, image, mask, q, grid, source["preprocessing"]["normalization"])
                        shard["preview"] = preview.relative_to(stage).as_posix()
                    entry["shards"].append(shard)
                    entry["num_embeddings"] += len(q)
                    print(f"[joint cache] {source['id']}: {i+1}/{len(dataset)} images", flush=True)
                manifest["datasets"].append(entry)
            for filename, value in (("manifest.json", manifest), ("extraction_identity.json", identity), ("config_used.json", dict(cfg))):
                (stage / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            previews = [f"![{entry['id']} / {s['sample_id']}]({s['preview']})" for entry in manifest["datasets"] for s in entry["shards"] if "preview" in s]
            (stage / "README.md").write_text("# Spatial cache alignment\n\nProcessed image | fractional foreground | grid occupancy. "
                "ViT: exact patch averages; CNN: adaptive spatial bins, not receptive fields.\n\n" + "\n\n".join(previews))
            load_cache_manifest(stage)
            stage.rename(cache_dir)
    finally:
        lock.unlink(missing_ok=True)
    return cache_dir
