from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ndi = None


@dataclass
class InstanceImageMetrics:
    name: str
    gt_instances: int
    pred_instances: int
    count_error: int
    abs_count_error: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float


def _neighbors(connectivity: int) -> List[Tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _label_connected_components_numpy(binary_mask: np.ndarray, connectivity: int) -> np.ndarray:
    h, w = binary_mask.shape
    out = np.zeros((h, w), dtype=np.int32)
    nbrs = _neighbors(connectivity)
    cur_label = 0

    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or out[y, x] != 0:
                continue
            cur_label += 1
            out[y, x] = cur_label
            q = deque([(y, x)])
            while q:
                cy, cx = q.popleft()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not binary_mask[ny, nx] or out[ny, nx] != 0:
                        continue
                    out[ny, nx] = cur_label
                    q.append((ny, nx))
    return out


def label_connected_components(
    binary_mask: np.ndarray,
    *,
    connectivity: int = 8,
    min_area: int = 0,
) -> np.ndarray:
    """
    Convert a binary mask (foreground > 0) into sequential instance ids.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary_mask.shape}")
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")

    binary = binary_mask.astype(bool, copy=False)

    if ndi is not None:
        if connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        elif connectivity == 8:
            structure = np.ones((3, 3), dtype=np.uint8)
        else:
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        labels, _ = ndi.label(binary, structure=structure)
        labels = labels.astype(np.int32, copy=False)
    else:
        labels = _label_connected_components_numpy(binary, connectivity=connectivity)

    if min_area > 0:
        ids, counts = np.unique(labels, return_counts=True)
        drop_ids = ids[(ids > 0) & (counts < min_area)]
        if drop_ids.size > 0:
            drop_mask = np.isin(labels, drop_ids)
            labels[drop_mask] = 0

    # Reindex to contiguous ids 1..N.
    out = np.zeros_like(labels, dtype=np.int32)
    ids = [int(i) for i in np.unique(labels) if int(i) > 0]
    for new_id, old_id in enumerate(ids, start=1):
        out[labels == old_id] = new_id
    return out


def _pairwise_overlaps(pred_labels: np.ndarray, gt_labels: np.ndarray) -> Dict[Tuple[int, int], int]:
    mask = (pred_labels > 0) & (gt_labels > 0)
    if not np.any(mask):
        return {}
    pred_vals = pred_labels[mask].astype(np.int64)
    gt_vals = gt_labels[mask].astype(np.int64)
    pairs = np.stack([pred_vals, gt_vals], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    return {(int(p), int(g)): int(c) for (p, g), c in zip(uniq, counts)}


def match_instances(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Greedy one-to-one matching of instances by IoU.
    """
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(
            f"pred/gt shape mismatch: {pred_labels.shape} vs {gt_labels.shape}"
        )
    if pred_labels.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {pred_labels.shape}")
    if not (0.0 <= float(iou_threshold) <= 1.0):
        raise ValueError(f"iou_threshold must be in [0,1], got {iou_threshold}")

    pred_ids = [int(v) for v in np.unique(pred_labels) if int(v) > 0]
    gt_ids = [int(v) for v in np.unique(gt_labels) if int(v) > 0]

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)
    if n_pred == 0 and n_gt == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_matched_iou": 1.0,
        }
    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
        }
    if n_gt == 0:
        return {
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
        }

    area_pred = {pid: int((pred_labels == pid).sum()) for pid in pred_ids}
    area_gt = {gid: int((gt_labels == gid).sum()) for gid in gt_ids}
    overlaps = _pairwise_overlaps(pred_labels, gt_labels)

    scored_pairs: List[Tuple[float, int, int]] = []
    for (pid, gid), inter in overlaps.items():
        union = area_pred[pid] + area_gt[gid] - int(inter)
        if union <= 0:
            continue
        iou = float(inter) / float(union)
        if iou >= float(iou_threshold):
            scored_pairs.append((iou, pid, gid))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    used_pred = set()
    used_gt = set()
    matched_ious: List[float] = []
    for iou, pid, gid in scored_pairs:
        if pid in used_pred or gid in used_gt:
            continue
        used_pred.add(pid)
        used_gt.add(gid)
        matched_ious.append(float(iou))

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(mean_iou),
    }


def evaluate_instance_image(
    *,
    name: str,
    pred_binary: np.ndarray,
    gt_instance_labels: np.ndarray,
    connectivity: int = 8,
    min_area: int = 0,
    iou_threshold: float = 0.5,
) -> Tuple[InstanceImageMetrics, np.ndarray]:
    if pred_binary.shape != gt_instance_labels.shape:
        raise ValueError(
            f"pred/gt shape mismatch for '{name}': "
            f"{pred_binary.shape} vs {gt_instance_labels.shape}"
        )
    pred_instances = label_connected_components(
        pred_binary.astype(bool),
        connectivity=connectivity,
        min_area=min_area,
    )
    gt_instances = gt_instance_labels.astype(np.int32, copy=False)
    gt_instances = np.where(gt_instances > 0, gt_instances, 0).astype(np.int32, copy=False)

    match = match_instances(
        pred_instances,
        gt_instances,
        iou_threshold=iou_threshold,
    )
    gt_count = int(sum(1 for v in np.unique(gt_instances) if int(v) > 0))
    pred_count = int(sum(1 for v in np.unique(pred_instances) if int(v) > 0))
    count_error = pred_count - gt_count

    row = InstanceImageMetrics(
        name=str(name),
        gt_instances=gt_count,
        pred_instances=pred_count,
        count_error=int(count_error),
        abs_count_error=int(abs(count_error)),
        tp=int(match["tp"]),
        fp=int(match["fp"]),
        fn=int(match["fn"]),
        precision=float(match["precision"]),
        recall=float(match["recall"]),
        f1=float(match["f1"]),
        mean_matched_iou=float(match["mean_matched_iou"]),
    )
    return row, pred_instances


def aggregate_instance_metrics(rows: Iterable[InstanceImageMetrics]) -> Dict[str, float]:
    rows_list = list(rows)
    if not rows_list:
        return {
            "num_images": 0,
            "mean_gt_instances": 0.0,
            "mean_pred_instances": 0.0,
            "count_mae": 0.0,
            "count_bias": 0.0,
            "instance_precision": 0.0,
            "instance_recall": 0.0,
            "instance_f1": 0.0,
            "mean_matched_iou": 0.0,
        }

    tp = sum(r.tp for r in rows_list)
    fp = sum(r.fp for r in rows_list)
    fn = sum(r.fn for r in rows_list)
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "num_images": int(len(rows_list)),
        "mean_gt_instances": float(np.mean([r.gt_instances for r in rows_list])),
        "mean_pred_instances": float(np.mean([r.pred_instances for r in rows_list])),
        "count_mae": float(np.mean([r.abs_count_error for r in rows_list])),
        "count_bias": float(np.mean([r.count_error for r in rows_list])),
        "instance_precision": float(precision),
        "instance_recall": float(recall),
        "instance_f1": float(f1),
        "mean_matched_iou": float(np.mean([r.mean_matched_iou for r in rows_list])),
    }


def rows_to_dicts(rows: Iterable[InstanceImageMetrics]) -> List[Dict[str, float | int | str]]:
    return [asdict(row) for row in rows]
