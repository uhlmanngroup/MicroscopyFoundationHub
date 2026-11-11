"""Make matching EM and DeepBacs slides from the individual-training plot data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_individual_em_deepbacs import ADAPTATIONS, MODELS, OUTPUT_DIR, load_results


GROUPS = {
    "EM": {
        "title": "Electron microscopy · individual training",
        "subtitle": "Lucchi++, VNC, Kasthuri++",
        "filename": "individual_training_em_iouf",
    },
    "LM": {
        "title": "DeepBacs · individual training",
        "subtitle": "E. coli, S. aureus, B. subtilis",
        "filename": "individual_training_deepbacs_iouf",
    },
}


def make_slide(group: str, all_results: list[dict]) -> None:
    dark = "#111218"
    muted = "#4F5D75"
    config = GROUPS[group]
    results = [row for row in all_results if row["group"] == group]
    datasets = list(dict.fromkeys(row["dataset"] for row in results))
    if len(results) != 27 or len(datasets) != 3:
        raise ValueError(f"Expected 27 results across 3 {group} datasets")

    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",
        "text.color": dark,
        "axes.labelcolor": dark,
        "xtick.color": dark,
        "ytick.color": dark,
    })
    fig = plt.figure(figsize=(16, 9), facecolor="#FFFFFF")
    ax = fig.add_axes([0.09, 0.21, 0.69, 0.61], facecolor="#FFFFFF")

    for row in results:
        model = MODELS[row["model"]]
        adaptation = ADAPTATIONS[row["adaptation"]]
        x = datasets.index(row["dataset"]) + model["offset"] + adaptation["offset"]
        ax.errorbar(
            x, row["mean"], yerr=row["std"],
            fmt=adaptation["marker"], linestyle="none",
            color=model["color"], ecolor=model["color"],
            elinewidth=1.45, capsize=3.2, capthick=1.2,
            markersize=10, markerfacecolor=model["color"],
            markeredgecolor="#FFFFFF", markeredgewidth=1.1,
            zorder=4,
        )

    ax.set_xlim(-0.58, 2.58)
    ax.set_ylim(0.45, 0.96)
    ax.set_xticks(range(3), datasets)
    ax.set_yticks(np.arange(0.5, 1.0, 0.1))
    ax.set_ylabel("Foreground IoU", fontsize=14, fontweight="bold", labelpad=15)
    ax.tick_params(axis="x", length=0, pad=16, labelsize=15)
    ax.tick_params(axis="y", length=0, pad=10, labelsize=12)
    ax.yaxis.grid(True, color="#E9EDF1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9D1D9")
        ax.spines[side].set_linewidth(0.9)

    fig.text(0.09, 0.925, config["title"], ha="left", va="center",
             fontsize=25, fontweight="bold", color=dark)
    fig.text(0.09, 0.875, config["subtitle"], ha="left", va="center",
             fontsize=13.5, color=muted)
    fig.add_artist(Line2D([0.805, 0.805], [0.21, 0.82], transform=fig.transFigure,
                          color="#DCE2E9", linewidth=1.0))

    fig.text(0.83, 0.78, "MODEL", fontsize=11.5, fontweight="bold", color=dark)
    model_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=style["color"],
               markeredgecolor="#FFFFFF", markersize=12, label=model)
        for model, style in MODELS.items()
    ]
    fig.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(0.82, 0.755),
               frameon=False, fontsize=12.5, labelspacing=1.25, handletextpad=0.75,
               borderpad=0.0)

    fig.text(0.83, 0.53, "ADAPTATION", fontsize=11.5, fontweight="bold", color=dark)
    adaptation_handles = [
        Line2D([], [], marker=style["marker"], linestyle="none", markerfacecolor=dark,
               markeredgecolor="#FFFFFF", markersize=12, label=adaptation)
        for adaptation, style in ADAPTATIONS.items()
    ]
    fig.legend(handles=adaptation_handles, loc="upper left", bbox_to_anchor=(0.82, 0.505),
               frameon=False, fontsize=12.5, labelspacing=1.25, handletextpad=0.75,
               borderpad=0.0)
    fig.text(0.83, 0.25, "Point = mean\nWhisker = ±1 SD\nShared IoU scale", fontsize=10.5,
             color=muted, linespacing=1.6, ha="left", va="top")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / config["filename"]
    fig.savefig(stem.with_suffix(".svg"), facecolor="#FFFFFF")
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved {stem}.svg and {stem}.png ({len(results)} results)")


if __name__ == "__main__":
    data = load_results()
    for name in GROUPS:
        make_slide(name, data)
