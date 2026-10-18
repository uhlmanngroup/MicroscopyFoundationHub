#!/usr/bin/env python3
"""Compare individual EM training with the corrected three-dataset joint run.

Run with the project environment::

    python scripts/analysis/plot_joint_vs_individual_em.py

Means and ± values are transcribed from the supplied results tables. The ±
values are plotted as given; their statistical definition was not specified.
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

from plot_joint_em_rerun import CORRECTED, DATASETS, MODELS, style, save


# (mean, supplied ± value), in MODELS order. Joint values come from CORRECTED.
INDIVIDUAL = {
    "VNC": ((.833, .003), (.856, .016), (.576, .015), (.649, .009), (.841, .013)),
    "Kasthuri++": ((.716, .006), (.864, .008), (.687, .005), (.754, .019), (.833, .018)),
    "Lucchi++": ((.739, .009), (.860, .009), (.549, .009), (.550, .066), (.766, .007)),
}
DATASET_ROWS = DATASETS[:3]
INDIVIDUAL_COLOR = "#736888"
JOINT_COLOR = "#087e8b"
NEGATIVE_COLOR = "#b35443"


def paired_figure(output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12), sharex=True)
    for ax, dataset in zip(axes, DATASET_ROWS):
        for i, model in enumerate(MODELS):
            individual_mean, individual_pm = INDIVIDUAL[dataset][i]
            joint_mean, joint_pm = CORRECTED[dataset][i]
            delta = joint_mean - individual_mean
            ax.plot((individual_mean, joint_mean), (i, i),
                    color="#bec7cb", lw=1.6, zorder=1)
            ax.errorbar(individual_mean, i, xerr=individual_pm, fmt="s",
                        color=INDIVIDUAL_COLOR, markerfacecolor="white",
                        markeredgewidth=1.5, markersize=6, capsize=3,
                        lw=1.3, zorder=3)
            ax.errorbar(joint_mean, i, xerr=joint_pm, fmt="o",
                        color=JOINT_COLOR, markersize=6.5, capsize=3,
                        lw=1.3, zorder=4)
            ax.text(1.015, i, f"{delta:+.3f}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=JOINT_COLOR if delta >= 0 else NEGATIVE_COLOR,
                    fontweight="semibold")
        ax.set_yticks(range(len(MODELS)), MODELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xticks((0, .2, .4, .6, .8, 1.0))
        ax.grid(axis="x", color="#e7ebee", lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#c6cdd2")
        ax.set_title(dataset, loc="left", pad=12,
                     color="#075e68" if dataset == "Lucchi++" else "#24313a")
        ax.text(1.015, -.65, "JOINT − INDIVIDUAL",
                transform=ax.get_yaxis_transform(), fontsize=9,
                fontweight="bold", color="#59636c", ha="left")
        if dataset == "Lucchi++":
            ax.set_facecolor("#f4fafb")
    axes[-1].set_xlabel("Foreground IoU", labelpad=10)
    fig.suptitle("Individual vs corrected joint EM training", x=.085,
                 ha="left", fontsize=18, fontweight="bold", y=.985)
    fig.text(.085, .943, "Each panel shows one dataset; differences are between supplied mean scores.",
             color="#59636c", fontsize=10)
    fig.legend(handles=[
        Line2D([0], [0], marker="s", linestyle="", color=INDIVIDUAL_COLOR,
               markerfacecolor="white", label="Individual training"),
        Line2D([0], [0], marker="o", linestyle="", color=JOINT_COLOR,
               label="Corrected joint training"),
    ], loc="upper right", bbox_to_anchor=(.94, .973), frameon=False, ncol=2)
    fig.text(.085, .025, "Error bars reproduce the supplied ± values; their statistical definition was not provided.",
             color="#59636c", fontsize=9)
    fig.subplots_adjust(left=.21, right=.82, top=.88, bottom=.09, hspace=.38)
    save(fig, output_dir, "em_individual_vs_joint", dpi)


def change_matrix(output_dir: Path, dpi: int) -> None:
    changes = [
        [CORRECTED[dataset][i][0] - INDIVIDUAL[dataset][i][0]
         for i in range(len(MODELS))]
        for dataset in DATASET_ROWS
    ]
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    im = ax.imshow(changes, cmap="RdBu", norm=TwoSlopeNorm(vmin=-.26,
                   vcenter=0, vmax=.26), aspect="auto")
    for row in range(len(DATASET_ROWS)):
        for col in range(len(MODELS)):
            value = changes[row][col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center",
                    fontsize=11, fontweight="semibold",
                    color="white" if abs(value) >= .16 else "#25313a")
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_yticks(range(len(DATASET_ROWS)), DATASET_ROWS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10)
    ax.spines[:].set_visible(False)
    ax.set_xticks([x - .5 for x in range(1, len(MODELS))], minor=True)
    ax.set_yticks([y - .5 for y in range(1, len(DATASET_ROWS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, shrink=.76, pad=.035,
                 label="Joint − individual foreground IoU")
    fig.suptitle("Joint vs individual training: score differences", x=.09,
                 ha="left", fontsize=17, fontweight="bold", y=.98)
    fig.text(.09, .88, "Corrected joint mean minus individual-training mean | blue = higher with joint training",
             color="#59636c", fontsize=10)
    fig.subplots_adjust(left=.16, right=.96, top=.77, bottom=.12)
    save(fig, output_dir, "em_joint_minus_individual", dpi)


def write_data(output_dir: Path) -> None:
    with (output_dir / "em_joint_vs_individual_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "model", "individual_mean", "individual_pm",
                         "corrected_joint_mean", "corrected_joint_pm", "joint_minus_individual"))
        for dataset in DATASET_ROWS:
            for i, model in enumerate(MODELS):
                individual_mean, individual_pm = INDIVIDUAL[dataset][i]
                joint_mean, joint_pm = CORRECTED[dataset][i]
                writer.writerow((dataset, model, f"{individual_mean:.3f}",
                                 f"{individual_pm:.3f}", f"{joint_mean:.3f}",
                                 f"{joint_pm:.3f}", f"{joint_mean - individual_mean:+.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "docs" / "media" / "joint_em_rerun")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    paired_figure(args.output_dir, args.dpi)
    change_matrix(args.output_dir, args.dpi)
    write_data(args.output_dir)
    print(f"Saved two figures (PNG + PDF) and values CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
