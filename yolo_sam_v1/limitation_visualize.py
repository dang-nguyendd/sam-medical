import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ============================================================
# Configuration
# ============================================================

DATASET = "CVC-ColonDB"
IOU_THRESHOLD = 0.8667
# CSV_PATH = f"./limitation_analysis/{DATASET}_run1.csv"
CSV_PATH = f"./limitation_analysis/all.csv"
OUTPUT_PATH = "./limitation_analysis/box_vs_mask_iou.png"


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Loaded data:")
print(df)

# Make sure values are numeric
df["SAM_IoU"] = pd.to_numeric(df["SAM_IoU"], errors="coerce")
df["YOLO_IoU"] = pd.to_numeric(df["YOLO_IoU"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["SAM_IoU", "YOLO_IoU"])


# ============================================================
# Identify low-SAM-IoU cases
# ============================================================

df["Low_SAM_IoU"] = df["SAM_IoU"] < IOU_THRESHOLD

low_sam = df[df["Low_SAM_IoU"]]
normal_sam = df[~df["Low_SAM_IoU"]]

print("\nLow SAM IoU cases:")
print(low_sam[["image", "SAM_IoU", "YOLO_IoU"]])


# ============================================================
# Correlation
# ============================================================

if len(df) >= 2:
    correlation, p_value = pearsonr(
        df["YOLO_IoU"],
        df["SAM_IoU"]
    )

    print("\nCorrelation analysis")
    print("--------------------")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"p-value:             {p_value:.4f}")


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 7))

# Normal cases
plt.scatter(
    normal_sam["YOLO_IoU"],
    normal_sam["SAM_IoU"],
    s=80,
    alpha=0.8,
    label=f"SAM IoU ≥ {IOU_THRESHOLD}"
)

# Low SAM IoU cases
plt.scatter(
    low_sam["YOLO_IoU"],
    low_sam["SAM_IoU"],
    s=100,
    alpha=0.9,
    marker="x",
    label=f"SAM IoU < {IOU_THRESHOLD}"
)


# ============================================================
# Label every image
# ============================================================

for _, row in df.iterrows():

    plt.annotate(
        row["image"],
        (
            row["YOLO_IoU"],
            row["SAM_IoU"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9
    )


# ============================================================
# SAM IoU threshold
# ============================================================

plt.axhline(
    IOU_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    label=f"SAM IoU threshold = {IOU_THRESHOLD}"
)


# ============================================================
# Reference diagonal
# ============================================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    linewidth=1,
    label="Box IoU = Mask IoU"
)


# ============================================================
# Labels and formatting
# ============================================================

plt.xlabel("YOLO Box IoU")
plt.ylabel("SAM Mask IoU")

plt.title(
    f"Relationship Between YOLO Box IoU and SAM Mask IoU"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()


# ============================================================
# Save and display
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nFigure saved to: {OUTPUT_PATH}")

plt.show()