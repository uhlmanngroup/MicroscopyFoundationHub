"""Plot individual-training foreground IoU for EM and DeepBacs.

Reads only the two Notion-export blocks in the Individual worksheet. The
resulting SVG and PNG are intended for a 16:9 presentation slide.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "outputs/2026-11-04-results-workbook/Results_Report_All_Experiments.xlsx"
OUTPUT_DIR = ROOT / "outputs/2026-11-09-individual-training-scatter"

DATASETS = [
    ("Lucchi++", "EM", 4, 2),
    ("VNC", "EM", 3, 2),
    ("Kasthuri++", "EM", 5, 2),
    ("E. coli", "LM", 25, 23),
    ("S. aureus", "LM", 24, 23),
    ("B. subtilis", "LM", 26, 23),
]
MODELS = {
    "DINOv2": {"color": "#788AA5", "offset": -0.28, "mean_cols": {"Frozen": 2, "LoRA": 6, "Full fine-tuning": 4}},
    "DINOv3": {"color": "#EF8354", "offset": 0.00, "mean_cols": {"Frozen": 8, "LoRA": 12, "Full fine-tuning": 10}},
    "OpenCLIP": {"color": "#4F5D75", "offset": 0.28, "mean_cols": {"Frozen": 14, "LoRA": 18, "Full fine-tuning": 16}},
}
ADAPTATIONS = {
    "Frozen": {"marker": "o", "offset": -0.085, "source_name": "Frozen"},
    "LoRA": {"marker": "s", "offset": 0.0, "source_name": "LoRA"},
    "Full fine-tuning": {"marker": "^", "offset": 0.085, "source_name": "FT"},
}


def load_results() -> list[dict]:
    book = load_workbook(WORKBOOK, data_only=True, read_only=True)
    sheet = book["Individual"]
    records = []
    try:
        for dataset, group, row, header_row in DATASETS:
            if sheet.cell(row, 1).value != dataset:
                raise ValueError(f"Expected {dataset} at Individual!A{row}")
            for model, model_style in MODELS.items():
                for adaptation, adaptation_style in ADAPTATIONS.items():
                    mean_col = model_style["mean_cols"][adaptation]
                    std_col = mean_col + 1
                    mean_cell = sheet.cell(row, mean_col)
                    std_cell = sheet.cell(row, std_col)
                    expected = f"{model} {adaptation_style['source_name']}"
                    mean_header = str(sheet.cell(header_row, mean_col).value).replace("\n", " ")
                    std_header = str(sheet.cell(header_row, std_col).value).replace("\n", " ")
                    if not mean_header.startswith(expected) or not mean_header.endswith("Mean"):
                        raise ValueError(f"Unexpected header at Individual!{sheet.cell(header_row, mean_col).coordinate}")
                    if not std_header.startswith(expected) or not std_header.endswith("Std"):
                        raise ValueError(f"Unexpected header at Individual!{sheet.cell(header_row, std_col).coordinate}")
                    mean, std = mean_cell.value, std_cell.value
                    if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
                        raise ValueError(f"Missing numeric result at Individual!{mean_cell.coordinate}:{std_cell.coordinate}")
                    if not 0 <= mean <= 1 or std < 0 or mean - std < 0 or mean + std > 1:
                        raise ValueError(f"Invalid IoU or std at Individual!{mean_cell.coordinate}:{std_cell.coordinate}")
                    records.append({
                        "dataset": dataset,
                        "group": group,
                        "model": model,
                        "adaptation": adaptation,
                        "mean": float(mean),
                        "std": float(std),
                    })
    finally:
        book.close()
    if len(records) != 54:
        raise ValueError(f"Expected 54 individual-training results, found {len(records)}")
    return records


def plot(records: list[dict]) -> None:
    dark = "#111218"
    muted = "#4F5D75"
    light = "#DCE2E9"

    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",
        "text.color": dark,
        "axes.labelcolor": dark,
        "xtick.color": dark,
        "ytick.color": dark,
    })

    fig = plt.figure(figsize=(16, 9), facecolor="#FFFFFF")
    ax = fig.add_axes([0.075, 0.205, 0.705, 0.615], facecolor="#FFFFFF")

    for record in records:
        dataset_index = next(i for i, item in enumerate(DATASETS) if item[0] == record["dataset"])
        model_style = MODELS[record["model"]]
        adaptation_style = ADAPTATIONS[record["adaptation"]]
        x = dataset_index + model_style["offset"] + adaptation_style["offset"]
        ax.errorbar(
            x,
            record["mean"],
            yerr=record["std"],
            fmt=adaptation_style["marker"],
            linestyle="none",
            color=model_style["color"],
            ecolor=model_style["color"],
            elinewidth=1.25,
            capsize=2.7,
            capthick=1.1,
            markersize=8.5,
            markerfacecolor=model_style["color"],
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.0,
            zorder=4,
        )

    ax.set_xlim(-0.55, 5.55)
    ax.set_ylim(0.45, 0.96)
    ax.set_xticks(range(len(DATASETS)), [item[0] for item in DATASETS])
    ax.set_yticks(np.arange(0.5, 1.0, 0.1))
    ax.set_ylabel("Foreground IoU", fontsize=13, fontweight="bold", labelpad=14)
    ax.tick_params(axis="x", length=0, pad=13, labelsize=12)
    ax.tick_params(axis="y", length=0, pad=9, labelsize=11)
    ax.yaxis.grid(True, color="#E9EDF1", linewidth=0.8)
    ax.axvline(2.5, color=light, linewidth=1.2, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9D1D9")
        ax.spines[side].set_linewidth(0.9)

    ax.text(1, -0.16, "ELECTRON MICROSCOPY  ·  EM", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=muted)
    ax.text(4, -0.16, "LIGHT MICROSCOPY  ·  DEEPBACS", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=muted)

    fig.text(0.075, 0.925, "Individual-dataset segmentation performance",
             ha="left", va="center", fontsize=24, fontweight="bold", color=dark)
    fig.text(0.075, 0.875, "EM and DeepBacs  ·  Three models  ·  Three adaptation regimes",
             ha="left", va="center", fontsize=12.5, color=muted)
    fig.add_artist(Line2D([0.802, 0.802], [0.205, 0.82], transform=fig.transFigure,
                          color=light, linewidth=1.0))

    fig.text(0.83, 0.78, "MODEL", fontsize=11, fontweight="bold", color=dark)
    model_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=style["color"],
               markeredgecolor="#FFFFFF", markersize=11, label=model)
        for model, style in MODELS.items()
    ]
    fig.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.82, 0.755),
               frameon=False, fontsize=12, labelspacing=1.15, handletextpad=0.75,
               borderpad=0.0)

    fig.text(0.83, 0.535, "ADAPTATION", fontsize=11, fontweight="bold", color=dark)
    adaptation_handles = [
        Line2D([], [], marker=style["marker"], linestyle="none", markerfacecolor=dark,
               markeredgecolor="#FFFFFF", markersize=11, label=adaptation)
        for adaptation, style in ADAPTATIONS.items()
    ]
    fig.legend(handles=adaptation_handles, loc="upper left", bbox_to_anchor=(0.82, 0.51),
               frameon=False, fontsize=12, labelspacing=1.15, handletextpad=0.75,
               borderpad=0.0)
    fig.text(0.83, 0.265, "Point = mean\nWhisker = ±1 SD", fontsize=10.5,
             color=muted, linespacing=1.6, ha="left", va="top")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "individual_training_em_deepbacs_iouf"
    fig.savefig(stem.with_suffix(".svg"), facecolor="#FFFFFF")
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved {stem}.svg and {stem}.png ({len(records)} results)")


if __name__ == "__main__":
    plot(load_results())
