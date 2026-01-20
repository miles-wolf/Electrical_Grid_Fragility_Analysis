import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/unsupervised/daily_with_clusters.parquet")
FIG_DIR = Path("reports/figures/storytelling/01_regimes_feature_space")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(DATA_PATH)

# Columns we will use
X_COL = "z_daily_peak_mw"
Y_COL = "z_daily_max_ramp_mw"
CLUSTER_COL = "cluster"
BA_COL = "balancing_authority"

# -----------------------------
# Plot function
# -----------------------------
def plot_feature_space(df_ba: pd.DataFrame, ba_name: str):
    # Display CAISO instead of CISO
    display_name = "CAISO" if ba_name == "CISO" else ba_name
    
    plt.figure(figsize=(8, 6))

    # ---- Conditional clipping policy (ONLY for CISO, ONLY if needed) ----
    clip_applied = False
    y_min, y_max = None, None

    # Choose a sensible viewing window for z-scores if clipping is needed
    CLIP_Y_MIN, CLIP_Y_MAX = -3, 4

    if ba_name.upper() == "CISO":
        # Apply clipping only if there are "stretch" points that would ruin readability
        if df_ba[Y_COL].max() > CLIP_Y_MAX:
            clip_applied = True
            y_min, y_max = CLIP_Y_MIN, CLIP_Y_MAX

    # ---- Plot points (and build a legend that matches what's actually visible) ----
    handles, labels = [], []
    clusters = sorted(df_ba[CLUSTER_COL].unique())

    for c in clusters:
        sub = df_ba[df_ba[CLUSTER_COL] == c]

        # If we're clipping, only plot points that fall inside the visible window.
        # This ensures the legend doesn't include clusters with no visible points.
        if clip_applied:
            sub = sub[(sub[Y_COL] >= y_min) & (sub[Y_COL] <= y_max)]
            if sub.empty:
                continue

        sc = plt.scatter(
            sub[X_COL],
            sub[Y_COL],
            s=40,
            alpha=0.7,
            label=str(c),
        )
        handles.append(sc)
        labels.append(str(c))

    # Apply axis limits ONLY when clipping is active
    if clip_applied:
        plt.ylim(y_min, y_max)

    # Reference axes
    plt.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    plt.axvline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    plt.title(f"{display_name}: Operational Regimes in Feature Space")
    plt.xlabel("Daily Peak Load (z-score)")
    plt.ylabel("Daily Max Ramp (z-score)")

    # Legend: if clipping is applied, legend only includes visible clusters.
    # If not clipped, legend shows all clusters.
    plt.legend(handles, labels, title="Cluster", loc="lower right")

    plt.tight_layout()

    outpath = FIG_DIR / f"regimes_feature_space_{ba_name.lower()}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Wrote {outpath}")

# -----------------------------
# Generate plots
# -----------------------------
for ba in sorted(df[BA_COL].unique()):
    plot_feature_space(df[df[BA_COL] == ba], ba)
