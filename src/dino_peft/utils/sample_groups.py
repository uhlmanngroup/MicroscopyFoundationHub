from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Sequence


def infer_expected_group_count(cfg: dict | None) -> Optional[int]:
    if not isinstance(cfg, dict):
        return None
    candidates = (
        cfg.get("train_img_dir"),
        cfg.get("test_img_dir"),
        cfg.get("experiment_id"),
        cfg.get("task_type"),
    )
    for value in candidates:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if "triplet" in text:
            return 3
        if "paired" in text:
            return 2
    return None


def _normalize_token(token: str) -> str:
    token = re.sub(r"\d+", "", token.lower())
    token = re.sub(r"[^a-z]+", "", token)
    return token


def _stem_tokens(name: str) -> List[str]:
    stem = Path(str(name)).stem
    raw_tokens = [tok for tok in re.split(r"[_\-\s]+", stem) if tok]
    tokens = [_normalize_token(tok) for tok in raw_tokens]
    tokens = [tok for tok in tokens if tok]
    if tokens:
        return tokens
    fallback = _normalize_token(stem)
    if fallback:
        return [fallback[:6]]
    return ["sample"]


@dataclass(frozen=True)
class SampleGrouping:
    labels: List[str]
    unique_labels: List[str]
    inferred_depth: int

    def label_for_index(self, idx: int) -> str:
        return self.labels[idx]


def infer_sample_grouping(
    names: Sequence[str],
    *,
    expected_groups: Optional[int] = None,
    max_depth: int = 3,
) -> SampleGrouping:
    tokens_per_name = [_stem_tokens(name) for name in names]
    best_labels = ["sample" for _ in names]
    best_unique = ["sample"]
    best_depth = 1
    best_score = None

    max_depth = max(1, int(max_depth))
    for depth in range(1, max_depth + 1):
        labels = [
            "_".join(tokens[:depth]) if len(tokens) >= depth else "_".join(tokens)
            for tokens in tokens_per_name
        ]
        unique_labels = list(dict.fromkeys(labels))
        n_groups = len(unique_labels)
        if n_groups <= 1:
            score = (2, depth, 1)
        elif expected_groups is not None and n_groups == expected_groups:
            score = (0, depth, -n_groups)
        elif expected_groups is None and 1 < n_groups <= 6:
            score = (1, depth, -n_groups)
        else:
            score = (3, abs((expected_groups or 2) - n_groups), depth)
        if best_score is None or score < best_score:
            best_score = score
            best_labels = labels
            best_unique = unique_labels
            best_depth = depth

    return SampleGrouping(
        labels=list(best_labels),
        unique_labels=list(best_unique),
        inferred_depth=int(best_depth),
    )


def select_balanced_preview_indices(
    grouping: SampleGrouping,
    *,
    seed: Optional[int] = None,
    expected_groups: Optional[int] = None,
) -> List[int]:
    if expected_groups not in (2, 3):
        return []
    group_labels = list(grouping.unique_labels)
    n_groups = len(group_labels)
    if n_groups != expected_groups:
        return []
    if n_groups == 2:
        per_group = 2
    elif n_groups == 3:
        per_group = 1
    else:
        return []

    grouped_indices: Dict[str, List[int]] = {label: [] for label in group_labels}
    for idx, label in enumerate(grouping.labels):
        grouped_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed) if seed is not None else None
    selected: List[int] = []
    for label in group_labels:
        candidates = list(grouped_indices.get(label, []))
        if rng is not None:
            rng.shuffle(candidates)
        selected.extend(candidates[:per_group])
    return selected
