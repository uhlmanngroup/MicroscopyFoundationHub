import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# === 1) Load your summary and extract ViT-g (giant) results ===
summary_path = Path("/Users/cfuste/Documents/Results/20251201-cluster-backup/seg/summary/summary.csv")
df = pd.read_csv(summary_path)

metric_col = "foreground_iou_mean"  # IoUf

# Colors
supervised_color = "#B1C7E6"   # darker blue
zeroshot_color   = "#5D91E3"   # lighter blue
ours_color       = "#1f57ff"   # yellow 
improve_color    = "#35C759"   # green
worse_color      = "#FF3B30"   # red

# Filter to ViT-g only
df_g = df[df["dino_size"] == "giant"].copy()

def get_pair(dataset_type):
    base_val = df_g[
        (df_g["dataset_type"] == dataset_type) &
        (df_g["use_lora"] == False)
    ][metric_col].iloc[0]

    lora_val = df_g[
        (df_g["dataset_type"] == dataset_type) &
        (df_g["use_lora"] == True)
    ][metric_col].iloc[0]

    return base_val, lora_val

lucchi_base, lucchi_lora = get_pair("lucchi")
paired_base, paired_lora = get_pair("paired")

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

# === 3) Your methods (Lucchi++ & Paired, ViT-g) ===
our_rows = [
    {
        "label": "Paired - Droso+Lucchi++, ViT-g (Ours)",
        "group": "Ours",
        "baseline": paired_base,
        "new": paired_lora,
        "is_ours": True,
    },
    {
        "label": "Lucchi++, ViT-g (Ours)",
        "group": "Ours",
        "baseline": lucchi_base,
        "new": lucchi_lora,
        "is_ours": True,
    },
]

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

# === 5) Create the figure ===
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(plot_df))

text_positive = improve_color
text_negative = worse_color

for i, row in plot_df.iterrows():
    y = y_pos[i]

    # Choose base color by group
    if row["group"] == "Supervised":
        base_color = supervised_color
    elif row["group"] == "Zero-shot":
        base_color = zeroshot_color
    else:  # "Ours"
        base_color = ours_color

    # Base bar
    ax.barh(
        y,
        row["baseline_pct"],
        color=base_color,
        edgecolor="none",
    )

    # Overlay for our methods where we have LoRA
    if row["is_ours"] and pd.notnull(row["new"]):
        if row["new"] >= row["baseline"]:
            # Improvement: green segment to the right
            ax.barh(
                y,
                row["new_pct"] - row["baseline_pct"],
                left=row["baseline_pct"],
                color=improve_color,
                edgecolor="none",
            )
        else:
            # Worse: red segment on the left (new is smaller)
            ax.barh(
                y,
                row["baseline_pct"] - row["new_pct"],
                left=row["new_pct"],
                color=worse_color,
                edgecolor="none",
            )

        # Text with +X.X% / -X.X%
        delta = row["delta_pct"]
        if np.isfinite(delta):
            sign = "+" if delta >= 0 else "−"
            txt_color = text_positive if delta >= 0 else text_negative
            ax.text(
                row["new_pct"] + 1,  # a bit to the right of the bar
                y,
                f"{sign}{abs(delta):.1f}%",
                va="center",
                ha="left",
                fontsize=10,
                color=txt_color,
            )

# === 6) Formatting ===

# Y labels, with a star for our methods
labels = []
for lbl, is_ours in zip(plot_df["label"], plot_df["is_ours"]):
    labels.append(("★ " if is_ours else "") + lbl)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # top row = first method

ax.set_xlabel("Foreground IoU (IoUf) [%]")
ax.set_xlim(0, 102)  # a bit of margin for the +X.X% text
ax.set_xticks([0, 25, 50, 75, 100])

# Light vertical grid for readability
ax.xaxis.grid(True, linestyle="--", alpha=0.25)
ax.set_axisbelow(True)

# Stronger group separators + optional group labels on the right
for g_name, g_data in plot_df.groupby("group_order"):
    last_idx = g_data.index.max()
    ax.axhline(last_idx + 0.5, color="lightgray", linewidth=1.0)

# === Legend at the bottom ===
legend_handles = [
    Patch(color=supervised_color, label="Supervised baseline"),
    Patch(color=zeroshot_color,   label="Zero-shot baseline"),
    Patch(color=ours_color,       label="Ours (zero-shot)"),
    Patch(color=improve_color,    label="+LoRA"),
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),  # below the axes
    ncol=2,
    frameon=False,
)

ax.set_title("EM Segmentation – Foreground IoU comparison")

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)  # extra space for legend
repo_root = Path(__file__).resolve().parents[2]
out_path = repo_root / "docs" / "media" / "fancyplot.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300)
