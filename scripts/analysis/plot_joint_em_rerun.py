#!/usr/bin/env python3
"""Plot the supplied joint EM scores before and after the corrected Lucchi++ rerun.

Run with the project environment::

    python scripts/analysis/plot_joint_em_rerun.py
    python scripts/analysis/plot_joint_em_rerun.py --output-dir /path/to/figures

The numbers below are transcribed from the two tables supplied for the comparison.
The ± values are drawn as given; their statistical meaning was not specified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


MODELS = (
    "DINOv3 Frozen",
    "DINOv3 FT",
    "OpenCLIP Frozen",
    "OpenCLIP FT",
    "ResNet50 FT",
)
DATASETS = ("VNC", "Kasthuri++", "Lucchi++", "Pooled aggregate")

# Each entry is (mean, supplied ± value), in MODELS order.
ORIGINAL = {
    "VNC": ((.872, .005), (.860, .009), (.535, .021), (.607, .051), (.172, .149)),
    "Kasthuri++": ((.727, .005), (.876, .002), (.672, .010), (.743, .061), (.281, .255)),
    "Lucchi++": ((.023, .006), (.002, .002), (.014, .004), (.036, .009), (.020, .014)),
    "Pooled aggregate": ((.168, .002), (.187, .001), (.144, .002), (.153, .016), (.070, .048)),
}
CORRECTED = {
    "VNC": ((.844, .005), (.873, .005), (.620, .008), (.627, .069), (.597, .095)),
    "Kasthuri++": ((.762, .003), (.879, .002), (.708, .009), (.817, .013), (.691, .045)),
    "Lucchi++": ((.756, .002), (.871, .001), (.622, .011), (.587, .035), (.524, .074)),
    "Pooled aggregate": ((.788, .001), (.874, .005), (.650, .005), (.677, .035), (.604, .029)),
}

OLD_COLOR = "#717985"
NEW_COLOR = "#087e8b"
GRID_COLOR = "#e7ebee"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_pair(ax: plt.Axes, dataset: str, *, show_values: bool = False) -> None:
    for i, model in enumerate(MODELS):
        old_mean, old_pm = ORIGINAL[dataset][i]
        new_mean, new_pm = CORRECTED[dataset][i]
        ax.plot((old_mean, new_mean), (i, i), color="#bac4ca", lw=1.5, zorder=1)
        ax.errorbar(old_mean, i, xerr=old_pm, fmt="s", color=OLD_COLOR,
                    markerfacecolor="white", markeredgewidth=1.5,
                    markersize=6, capsize=3, lw=1.3, zorder=3)
        ax.errorbar(new_mean, i, xerr=new_pm, fmt="o", color=NEW_COLOR,
                    markersize=6.5, capsize=3, lw=1.3, zorder=4)
        if show_values:
            ax.text(1.015, i, f"{new_mean - old_mean:+.3f}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=NEW_COLOR if new_mean >= old_mean else "#b35443",
                    fontweight="semibold")
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks((0, .2, .4, .6, .8, 1.0))
    ax.grid(axis="x", color=GRID_COLOR, lw=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c6cdd2")


def comparison_figure(output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        add_pair(ax, dataset)
        ax.set_title(dataset, loc="left", pad=12)
    for ax in axes[1]:
        ax.set_xlabel("Foreground IoU")
    fig.suptitle("Joint EM training | original vs corrected rerun", x=.065,
                 ha="left", fontsize=17, fontweight="bold", y=.995)
    fig.text(.065, .936, "Mean scores with the supplied ± values", fontsize=10,
             color="#59636c")
    legend = [
        Line2D([0], [0], marker="s", linestyle="", color=OLD_COLOR,
               markerfacecolor="white", label="Original"),
        Line2D([0], [0], marker="o", linestyle="", color=NEW_COLOR, label="Corrected"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.96, .966),
               frameon=False, ncol=2)
    fig.text(.065, .015, "Aggregate is pooled foreground IoU across test pixels; it is not the mean of the three dataset rows.",
             fontsize=9, color="#59636c")
    fig.subplots_adjust(left=.16, right=.955, top=.89, bottom=.095, hspace=.35, wspace=.33)
    save(fig, output_dir, "joint_em_before_after", dpi)


def lucchi_figure(output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    add_pair(ax, "Lucchi++", show_values=True)
    ax.set_xlabel("Lucchi++ foreground IoU", labelpad=10)
    ax.text(1.015, -.64, "CHANGE", transform=ax.get_yaxis_transform(),
            fontsize=9, fontweight="bold", color="#59636c", ha="left")
    fig.suptitle("Lucchi++ scores after the corrected rerun", x=.085,
                 ha="left", fontsize=17, fontweight="bold", y=.99)
    fig.text(.085, .89, "All five corrected scores increased; changes are absolute IoU points.",
             color="#59636c")
    fig.legend(handles=[
        Line2D([0], [0], marker="s", linestyle="", color=OLD_COLOR,
               markerfacecolor="white", label="Original"),
        Line2D([0], [0], marker="o", linestyle="", color=NEW_COLOR, label="Corrected"),
    ], loc="upper right", bbox_to_anchor=(.88, .95), frameon=False, ncol=2)
    fig.text(.085, .045, "Error bars reproduce the supplied ± values; their statistical definition was not provided.",
             color="#59636c", fontsize=9)
    fig.subplots_adjust(left=.205, right=.84, top=.80, bottom=.16)
    save(fig, output_dir, "lucchi_before_after", dpi)


def per_dataset_figure(output_dir: Path, dpi: int) -> None:
    """Compare every model within each individual dataset, excluding the pool."""
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12), sharex=True)
    for ax, dataset in zip(axes, DATASETS[:3]):
        if dataset == "Lucchi++":
            ax.set_facecolor("#f4fafb")
        add_pair(ax, dataset, show_values=True)
        ax.set_title(dataset, loc="left", pad=12,
                     color="#075e68" if dataset == "Lucchi++" else "#24313a")
        ax.text(1.015, -.65, "CHANGE", transform=ax.get_yaxis_transform(),
                fontsize=9, fontweight="bold", color="#59636c", ha="left")
    axes[-1].set_xlabel("Foreground IoU", labelpad=10)
    fig.suptitle("How each EM dataset changed", x=.085, ha="left",
                 fontsize=18, fontweight="bold", y=.985)
    fig.text(.085, .943, "Original vs corrected rerun | individual datasets only",
             color="#59636c", fontsize=10)
    fig.legend(handles=[
        Line2D([0], [0], marker="s", linestyle="", color=OLD_COLOR,
               markerfacecolor="white", label="Original"),
        Line2D([0], [0], marker="o", linestyle="", color=NEW_COLOR, label="Corrected"),
    ], loc="upper right", bbox_to_anchor=(.87, .973), frameon=False, ncol=2)
    fig.text(.085, .025, "Error bars reproduce the supplied ± values. Change = corrected mean − original mean.",
             color="#59636c", fontsize=9)
    fig.subplots_adjust(left=.21, right=.84, top=.88, bottom=.09, hspace=.38)
    save(fig, output_dir, "em_individual_datasets_before_after", dpi)


def delta_figure(output_dir: Path, dpi: int) -> None:
    deltas = [
        [CORRECTED[dataset][i][0] - ORIGINAL[dataset][i][0]
         for i in range(len(MODELS))]
        for dataset in DATASETS
    ]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    colors = plt.get_cmap("RdBu")
    im = ax.imshow(deltas, cmap=colors, norm=TwoSlopeNorm(vmin=-.9, vcenter=0, vmax=.9),
                   aspect="auto")
    for row in range(len(DATASETS)):
        for col in range(len(MODELS)):
            value = deltas[row][col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center",
                    fontsize=11, fontweight="semibold",
                    color="white" if abs(value) >= .45 else "#25313a")
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_yticks(range(len(DATASETS)), DATASETS)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.xaxis.tick_top()
    ax.spines[:].set_visible(False)
    ax.set_xticks([x - .5 for x in range(1, len(MODELS))], minor=True)
    ax.set_yticks([y - .5 for y in range(1, len(DATASETS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, shrink=.75, pad=.035, label="Change in foreground IoU")
    fig.suptitle("Change after corrected rerun", x=.09, ha="left",
                 fontsize=17, fontweight="bold", y=.98)
    fig.text(.09, .88, "Corrected minus original | positive values indicate higher scores",
             color="#59636c", fontsize=10)
    fig.subplots_adjust(left=.17, right=.96, top=.77, bottom=.13)
    save(fig, output_dir, "joint_em_change_matrix", dpi)


def lucchi_gain_figure(output_dir: Path, dpi: int) -> None:
    """Slide-friendly alternative to the paired-score plot: rank mean gains."""
    rows = sorted(
        ((CORRECTED["Lucchi++"][i][0] - ORIGINAL["Lucchi++"][i][0],
          model, ORIGINAL["Lucchi++"][i][0], CORRECTED["Lucchi++"][i][0])
         for i, model in enumerate(MODELS)),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for y, (gain, model, before, after) in enumerate(rows):
        color = "#075e68" if y == 0 else "#41a5ae"
        ax.barh(y, gain, height=.58, color=color, zorder=2)
        ax.text(gain + .012, y, f"+{gain:.3f}", va="center", ha="left",
                color="#24313a", fontweight="bold", fontsize=11)
        ax.text(1.015, y, f"{before:.3f} → {after:.3f}",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                color="#49545e", fontsize=10)
    ax.set_yticks(range(len(rows)), [row[1] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks((0, .2, .4, .6, .8, 1.0))
    ax.set_xlabel("Increase in Lucchi++ foreground IoU", labelpad=10)
    ax.grid(axis="x", color=GRID_COLOR, lw=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c6cdd2")
    ax.text(1.015, -.75, "ORIGINAL → CORRECTED",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            color="#59636c", fontsize=9, fontweight="bold")
    fig.suptitle("Lucchi++ gain by model", x=.085, ha="left",
                 fontsize=17, fontweight="bold", y=.98)
    fig.text(.085, .88, "Ranked change in mean score after the corrected rerun",
             color="#59636c", fontsize=10)
    fig.text(.085, .04, "Bars show differences between supplied means; uncertainties are shown in the paired-score figures.",
             color="#59636c", fontsize=9)
    fig.subplots_adjust(left=.24, right=.79, top=.78, bottom=.16)
    save(fig, output_dir, "lucchi_gain_ranking", dpi)


def write_data(output_dir: Path) -> None:
    with (output_dir / "joint_em_rerun_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "model", "original_mean", "original_pm",
                         "corrected_mean", "corrected_pm", "change"))
        for dataset in DATASETS:
            for i, model in enumerate(MODELS):
                old_mean, old_pm = ORIGINAL[dataset][i]
                new_mean, new_pm = CORRECTED[dataset][i]
                writer.writerow((dataset, model, f"{old_mean:.3f}", f"{old_pm:.3f}",
                                 f"{new_mean:.3f}", f"{new_pm:.3f}",
                                 f"{new_mean - old_mean:+.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "docs" / "media" / "joint_em_rerun")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    comparison_figure(args.output_dir, args.dpi)
    lucchi_figure(args.output_dir, args.dpi)
    per_dataset_figure(args.output_dir, args.dpi)
    delta_figure(args.output_dir, args.dpi)
    lucchi_gain_figure(args.output_dir, args.dpi)
    write_data(args.output_dir)
    print(f"Saved five figures (PNG + PDF) and values CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
