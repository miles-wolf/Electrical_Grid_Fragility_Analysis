import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/unsupervised/daily_with_clusters.parquet")

# Update these if your project uses a different figures folder
FIG_DIR = Path("reports/figures/storytelling/02_fragility_by_cluster")
FIG_DIR.mkdir(parents=True, exist_ok=True)

BA_COL = "balancing_authority"
CLUSTER_COL = "cluster"
FRAG_COL = "fragility_z"

def plot_fragility_by_cluster(df_ba: pd.DataFrame, ba_name: str):
    # Display CAISO instead of CISO
    display_name = "CAISO" if ba_name == "CISO" else ba_name
    
    # Order clusters by mean fragility (makes the plot instantly interpretable)
    cluster_order = (
        df_ba.groupby(CLUSTER_COL)[FRAG_COL]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    data = [df_ba[df_ba[CLUSTER_COL] == c][FRAG_COL].dropna().values for c in cluster_order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        data,
        showfliers=True,   # keep outliers visible here
    )
    
    # Set tick labels explicitly
    plt.xticks(range(1, len(cluster_order) + 1), [str(c) for c in cluster_order])

    # Add mean markers (small but helpful)
    means = [df_ba[df_ba[CLUSTER_COL] == c][FRAG_COL].mean() for c in cluster_order]
    plt.scatter(range(1, len(means) + 1), means, s=40, marker="D", label="Mean")

    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    plt.title(f"{display_name}: Fragility by Operational Regime (Cluster)")
    plt.xlabel("Cluster (ordered by mean fragility)")
    plt.ylabel("Fragility (z-score)")

    plt.legend(loc="upper left")
    plt.tight_layout()
    
    # Add slight padding to y-axis so markers near zero are visible
    # Must be after tight_layout() to avoid being reset
    current_ylim = plt.ylim()
    plt.ylim(current_ylim[0] - 0.5, current_ylim[1])

    outpath = FIG_DIR / f"02_fragility_by_cluster_{ba_name.lower()}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Wrote {outpath}")

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)

    for ba in sorted(df[BA_COL].unique()):
        plot_fragility_by_cluster(df[df[BA_COL] == ba], ba)
