"""Generate the paper's Figure 2 from the current results workbook.

Example (local; absolute paths are user-specific):
    python scripts/analysis/fancy_plot.py
    python scripts/analysis/fancy_plot.py --variant poster
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from openpyxl import load_workbook
from pathlib import Path

# Keep exactly the six VFM rows of the published Figure 2, using the ViT-L
# variants reported in main Tables 1 and 2 for all three model families.
# Each tuple is (sheet, dataset row, header row, frozen-mean cell, LoRA-mean cell).
FIGURE_2_CELLS = {
    "DINOv2": {
        "lucchi": ("Individual", 9, 8, "J9", "L9"),
        "paired": ("Paired", 5, 2, "J5", "L5"),
    },
    "DINOv3": {
        "lucchi": ("Individual", 4, 2, "H4", "L4"),
        "paired": ("Paired", 11, 8, "J11", "L11"),
    },
    "OpenCLIP": {
        "lucchi": ("Individual", 4, 2, "N4", "R4"),
        "paired": ("Paired", 17, 14, "B17", "D17"),
    },
}
MODEL_VARIANTS = {
    "DINOv2": "ViT-L/14",
    "DINOv3": "ViT-L/16",
    "OpenCLIP": "ViT-L/14",
}
MODEL_LABELS = {
    "DINOv2": "DINOv2-L/14",
    "DINOv3": "DINOv3-L/16",
    "OpenCLIP": "OpenCLIP ViT-L/14",
}
WORKBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs/2026-11-04-results-workbook/Results_Report_All_Experiments.xlsx"
)

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
            "Ours": "Ours (frozen)",
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

def load_figure_results(workbook_path: Path) -> dict:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    results = {}
    try:
        for model, datasets in FIGURE_2_CELLS.items():
            results[model] = {}
            for dataset, (sheet_name, row, header_row, frozen_cell, lora_cell) in datasets.items():
                sheet = workbook[sheet_name]
                expected_row = "Lucchi++" if dataset == "lucchi" else "Macro"
                if sheet.cell(row=row, column=1).value != expected_row:
                    raise ValueError(f"Unexpected dataset at {sheet_name}!A{row}")
                for cell, mode in ((frozen_cell, "Frozen"), (lora_cell, "LoRA")):
                    header = str(sheet.cell(row=header_row, column=sheet[cell].column).value)
                    # The Individual sheet omits the DINOv3/OpenCLIP variant
                    # from its headers; the Paired sheet names all variants.
                    variant_in_header = model == "DINOv2" or dataset == "paired"
                    if (model not in header or mode not in header or "Mean" not in header
                            or (variant_in_header and MODEL_VARIANTS[model] not in header)):
                        raise ValueError(f"Unexpected column at {sheet_name}!{cell}: {header}")
                frozen = sheet[frozen_cell].value
                lora = sheet[lora_cell].value
                if not isinstance(frozen, (int, float)):
                    raise ValueError(f"Missing frozen score at {sheet_name}!{frozen_cell}")
                if lora is not None and not isinstance(lora, (int, float)):
                    raise ValueError(f"Invalid LoRA score at {sheet_name}!{lora_cell}")
                results[model][dataset] = (float(frozen), float(lora) if lora is not None else np.nan)
    finally:
        workbook.close()
    return results


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

summary_data = load_figure_results(WORKBOOK_PATH)

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

# === 3) Your methods (Lucchi++ & Paired, main-table ViT-L variants) ===
our_rows = []
for label in ("DINOv2", "DINOv3", "OpenCLIP"):
    paired_base, paired_lora = summary_data[label]["paired"]
    lucchi_base, lucchi_lora = summary_data[label]["lucchi"]
    model_label = MODEL_LABELS[label]
    our_rows.extend([
        {
            "label": f"Paired - VNC+Lucchi++, {model_label} (Ours)",
            "group": "Ours",
            "baseline": paired_base,
            "new": paired_lora,
            "is_ours": True,
        },
        {
            "label": f"Lucchi++, {model_label} (Ours)",
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

# Keep the published row order while the corrected paired bars move rightward.
ours_order = {
    "Lucchi++, DINOv2-L/14 (Ours)": 0,
    "Lucchi++, DINOv3-L/16 (Ours)": 1,
    "Lucchi++, OpenCLIP ViT-L/14 (Ours)": 2,
    "Paired - VNC+Lucchi++, DINOv3-L/16 (Ours)": 3,
    "Paired - VNC+Lucchi++, DINOv2-L/14 (Ours)": 4,
    "Paired - VNC+Lucchi++, OpenCLIP ViT-L/14 (Ours)": 5,
}
plot_df["display_order"] = plot_df.apply(
    lambda row: ours_order[row["label"]] if row["is_ours"] else -row["baseline"],
    axis=1,
)
plot_df = plot_df.sort_values(
    by=["group_order", "display_order"],
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
        elif row["is_ours"]:
            ax.text(
                row["baseline_pct"] + 1,
                y,
                "LoRA pending",
                va="center",
                ha="left",
                fontsize=10,
                color="dimgray",
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
