import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "./visualizations/ablations.csv"
OUTPUT_PATH = "./visualizations/ablations.png"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ============================================================
# Plot style
# ============================================================

# Okabe-Ito colorblind-safe palette
# Blue, orange, vermillion
colors = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#D55E00"   # Vermillion
]

# Sans-serif font with fallbacks
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": 300
})

# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

# Rename the first unnamed column if needed
df = df.rename(
    columns={df.columns[0]: "Training Configuration"}
)

# Fill blank configuration names
df["Training Configuration"] = (
    df["Training Configuration"].ffill()
)

# ============================================================
# Short names for plotting
# ============================================================

config_names = {
    "Training Prompt Encoder":
        "Prompt Encoder",

    "Training Prompt Encoder & Image Encoder":
        "Prompt + Image Encoder",

    "Training Prompt Encoder & Mask Decoder":
        "Prompt Encoder + Mask Decoder"
}

df["Configuration"] = (
    df["Training Configuration"].map(config_names)
)

# ============================================================
# Settings
# ============================================================

datasets = [
    "ClinicDB",
    "ColonDB",
    "Kvasir"
]

metrics = [
    "Dice",
    "IoU",
    "Precision",
    "Recall"
]

configurations = [
    "Prompt Encoder",
    "Prompt + Image Encoder",
    "Prompt Encoder + Mask Decoder"
]

# ============================================================
# Plot
# ============================================================

x = np.arange(len(datasets))
width = 0.24

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 11)
)

axes = axes.flatten()

for ax, metric in zip(axes, metrics):

    for i, config in enumerate(configurations):

        values = []

        for dataset in datasets:

            selected = df.loc[
                (df["Dataset"] == dataset) &
                (df["Configuration"] == config),
                metric
            ]

            if selected.empty:
                raise ValueError(
                    f"Missing value for {config}, "
                    f"{dataset}, metric {metric}"
                )

            values.append(selected.iloc[0])

        offset = (i - 1) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=config,
            color=colors[i],
            edgecolor="none",
            zorder=3
        )

        # ----------------------------------------------------
        # Bar-value annotations
        # ----------------------------------------------------

        for bar, value in zip(bars, values):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=14
            )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax.set_title(
        metric,
        pad=12
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        datasets,
        rotation=0,
        ha="center"
    )

    ax.set_ylim(0.70, 1.02)

    ax.set_ylabel("Score")

    # Subtle horizontal grid
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
        zorder=0
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Lighten remaining axes
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")

    # Keep ticks subtle
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        color="#777777"
    )

# ============================================================
# Common legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=False,
    fontsize=18
)

# ============================================================
# Layout and save
# ============================================================

plt.tight_layout(
    rect=[0, 0.08, 1, 1],
    h_pad=2.5,
    w_pad=2.0
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print(f"[Saved] {OUTPUT_PATH}")