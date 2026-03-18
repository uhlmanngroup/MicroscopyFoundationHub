"""Generate the Lucchi++ summary comparison plot.

Example (local; absolute paths are user-specific):
    python scripts/analysis/fancy_plot.py
    python scripts/analysis/fancy_plot.py --variant poster
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import re

# === CONFIG (edit here) ===
CONFIG = {
    "summary_dir": Path("/Users/cfuste/Documents/Results/DINO-LoRA/seg/summary-output"),
    "summary_files": {
        "DINOv2": "summary-dinov2.csv",
        "DINOv3": "summary-dinov3.csv",
        "OpenCLIP": "summary-openclip.csv",
    },
    "metric_col": "foreground_iou_mean",  # IoUf
}
metric_col = CONFIG["metric_col"]

STYLE_CONFIGS = {
    "default": {
        "figsize": (10, 6),
        "colors": {
            "Supervised": "#B1C7E6",
            "Zero-shot": "#5D91E3",
            "Ours": "#1f57ff",
            "Improve": "#35C759",
            "Worse": "#FF3B30",
        },
        "legend_labels": {
            "Supervised": "Supervised baseline",
            "Zero-shot": "Zero-shot baseline",
            "Ours": "Ours (zero-shot)",
            "Improve": "+LoRA",
        },
        "legend_kwargs": {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.12),
            "ncol": 2,
        },
        "subplot_adjust": {"bottom": 0.18},
        "title": "EM Segmentation – Foreground IoU comparison",
        "output_name": "fancyplot.png",
    },
    "poster": {
        "figsize": (11.5, 6),
        "colors": {
            "Supervised": "#9DA9BE",
            "Zero-shot": "#7888A5",
            "Ours": "#4F5D75",
            "Improve": "#EF8354",
            "Worse": "#FF3B30",
        },
        "legend_labels": {
            "Supervised": "Supervised",
            "Zero-shot": "Zero-shot",
            "Ours": "Ours (frozen)",
            "Improve": "+LoRA",
        },
        "legend_kwargs": {
            "loc": "center left",
            "bbox_to_anchor": (1.01, 0.5),
            "ncol": 1,
            "borderaxespad": 0.0,
        },
        "subplot_adjust": {"right": 0.78},
        "title": None,
        "output_name": "fancyplot_poster.png",
    },
}

SIZE_RANK = {
    "tiny": 0,
    "small": 1,
    "base": 2,
    "large": 3,
    "giant": 4,
    "huge": 5,
}
VIT_RANK = {
    "vit-s": 0,
    "vit-b": 1,
    "vit-l": 2,
    "vit-g": 3,
    "vit-h": 4,
    "vit-bigg": 5,
}

def _normalize_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return (
        series.astype(str).str.strip().str.lower().map(mapping, na_action="ignore").fillna(False)
    )

def _rank_size(value: str) -> int:
    s = str(value).strip().lower()
    if s in SIZE_RANK:
        return SIZE_RANK[s]
    match = re.match(r"vit[-_]?([a-z]+)", s)
    if match:
        key = f"vit-{match.group(1)}"
        return VIT_RANK.get(key, -1)
    return -1

def _load_summary(label: str, filename: str) -> pd.DataFrame:
    path = CONFIG["summary_dir"] / filename
    if not path.is_file():
        raise FileNotFoundError(f"{label} summary not found: {path}")
    df = pd.read_csv(path)
    df["dino_size"] = df["dino_size"].astype(str).str.lower()
    df["dataset_type"] = df["dataset_type"].astype(str).str.lower()
    df["use_lora"] = _normalize_bool(df["use_lora"])
    return df

def _select_largest_model(df: pd.DataFrame) -> pd.DataFrame:
    ranks = df["dino_size"].apply(_rank_size)
    if ranks.max() < 0:
        sizes = sorted(df["dino_size"].unique().tolist())
        raise RuntimeError(f"Unable to rank model sizes from: {sizes}")
    return df.loc[ranks == ranks.max()].copy()

def get_pair(df: pd.DataFrame, dataset_type: str) -> tuple[float, float]:
    dataset_type = dataset_type.lower()
    base_val = df[
        (df["dataset_type"] == dataset_type) &
        (df["use_lora"] == False)
    ][metric_col].iloc[0]

    lora_val = df[
        (df["dataset_type"] == dataset_type) &
        (df["use_lora"] == True)
    ][metric_col].iloc[0]

    return base_val, lora_val


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate the Lucchi++ comparison plot.")
    ap.add_argument(
        "--variant",
        choices=sorted(STYLE_CONFIGS),
        default="default",
        help="Visual variant to render.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to docs/media/<variant filename>.",
    )
    return ap.parse_args()

summary_data = {}
for label, filename in CONFIG["summary_files"].items():
    summary_df = _select_largest_model(_load_summary(label, filename))
    summary_data[label] = {
        "paired": get_pair(summary_df, "paired"),
        "lucchi": get_pair(summary_df, "lucchi"),
    }

# === 2) External baselines (from the papers) ===
# All values are foreground IoU in [0, 1].
external_rows = [
    # Supervised baselines on Lucchi++
    {"label": "2D U-Net (Casser 2020)",                     "group": "Supervised", "baseline": 0.888},
    {"label": "MLP (González-Marfil 2025)",                 "group": "Supervised", "baseline": 0.586},
    {"label": "2D U-Net (Franco-Barranco 2021)",            "group": "Supervised", "baseline": 0.893},
    {"label": "2D Attention U-Net (Franco-Barranco 2021)",  "group": "Supervised", "baseline": 0.893},

    # Zero-shot / few-shot baselines on Lucchi++
    {"label": "CLIPSeg-Img (Hinderling 2024)",              "group": "Zero-shot",  "baseline": 0.148},
    {"label": "CLIPSeg-Txt (Lüddecke 2022)",                "group": "Zero-shot",  "baseline": 0.067},
    {"label": "ConvPaint (Lüddecke 2022)",                  "group": "Zero-shot",  "baseline": 0.189},
    {"label": "DINOSim 1P, ViT-g (González-Marfil 2025)",   "group": "Zero-shot",  "baseline": 0.421},
    {"label": "DINOSim 3P, ViT-g (González-Marfil 2025)",   "group": "Zero-shot",  "baseline": 0.486},
]

external_df = pd.DataFrame(external_rows)
external_df["new"] = np.nan    # no LoRA overlay for these
external_df["is_ours"] = False

# === 3) Your methods (Lucchi++ & Paired, largest model per backbone) ===
our_rows = []
for label in ("DINOv2", "DINOv3", "OpenCLIP"):
    paired_base, paired_lora = summary_data[label]["paired"]
    lucchi_base, lucchi_lora = summary_data[label]["lucchi"]
    our_rows.extend([
        {
            "label": f"Paired - VNC+Lucchi++, {label} (Ours)",
            "group": "Ours",
            "baseline": paired_base,
            "new": paired_lora,
            "is_ours": True,
        },
        {
            "label": f"Lucchi++, {label} (Ours)",
            "group": "Ours",
            "baseline": lucchi_base,
            "new": lucchi_lora,
            "is_ours": True,
        },
    ])

our_df = pd.DataFrame(our_rows)

# === 4) Merge and set order: supervised -> zero-shot -> ours ===
plot_df = pd.concat([external_df, our_df], ignore_index=True)

group_order = {"Supervised": 0, "Zero-shot": 1, "Ours": 2}
plot_df["group_order"] = plot_df["group"].map(group_order)

# Sort by group, then by baseline performance
plot_df = plot_df.sort_values(
    by=["group_order", "baseline"],
    ascending=[True, False]
).reset_index(drop=True)

# For convenience, convert metric to %
plot_df["baseline_pct"] = plot_df["baseline"] * 100
plot_df["new_pct"] = plot_df["new"] * 100

# Absolute difference in percentage points
plot_df["delta_pct"] = plot_df["new_pct"] - plot_df["baseline_pct"]

def build_plot(style_name: str, output_path: Path) -> None:
    style = STYLE_CONFIGS[style_name]
    colors = style["colors"]

    fig, ax = plt.subplots(figsize=style["figsize"])
    y_pos = np.arange(len(plot_df))

    for i, row in plot_df.iterrows():
        y = y_pos[i]
        base_color = colors[row["group"]]

        ax.barh(
            y,
            row["baseline_pct"],
            color=base_color,
            edgecolor="none",
        )

        if row["is_ours"] and pd.notnull(row["new"]):
            if row["new"] >= row["baseline"]:
                ax.barh(
                    y,
                    row["new_pct"] - row["baseline_pct"],
                    left=row["baseline_pct"],
                    color=colors["Improve"],
                    edgecolor="none",
                )
            else:
                ax.barh(
                    y,
                    row["baseline_pct"] - row["new_pct"],
                    left=row["new_pct"],
                    color=colors["Worse"],
                    edgecolor="none",
                )

            delta = row["delta_pct"]
            if np.isfinite(delta):
                sign = "+" if delta >= 0 else "−"
                txt_color = colors["Improve"] if delta >= 0 else colors["Worse"]
                ax.text(
                    row["new_pct"] + 1,
                    y,
                    f"{sign}{abs(delta):.1f}%",
                    va="center",
                    ha="left",
                    fontsize=10,
                    color=txt_color,
                )

    labels = [
        ("★ " if is_ours else "") + lbl
        for lbl, is_ours in zip(plot_df["label"], plot_df["is_ours"])
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Foreground IoU (IoUf) [%]")
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for _, g_data in plot_df.groupby("group_order"):
        last_idx = g_data.index.max()
        ax.axhline(last_idx + 0.5, color="lightgray", linewidth=1.0)

    legend_handles = [
        Patch(color=colors["Supervised"], label=style["legend_labels"]["Supervised"]),
        Patch(color=colors["Zero-shot"], label=style["legend_labels"]["Zero-shot"]),
        Patch(color=colors["Ours"], label=style["legend_labels"]["Ours"]),
        Patch(color=colors["Improve"], label=style["legend_labels"]["Improve"]),
    ]
    ax.legend(handles=legend_handles, frameon=False, **style["legend_kwargs"])

    if style["title"]:
        ax.set_title(style["title"])

    plt.tight_layout()
    plt.subplots_adjust(**style["subplot_adjust"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_path = args.output
    if output_path is None:
        output_path = repo_root / "docs" / "media" / STYLE_CONFIGS[args.variant]["output_name"]
    build_plot(args.variant, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
