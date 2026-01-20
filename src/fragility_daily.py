"""
Module 3 — Daily Fragility Score + Plots

Reads hourly EIA-930 processed parquet (Module 1/2 output),
engineers daily features, computes two fragility indices:
- fragility_z  (z-score composite, per BA)
- fragility_01 (0–1 scaled version of fragility_z, per BA)

Outputs:
- data/processed/fragility/daily_fragility_pjm_ciso_2025.parquet
- reports/figures/fragility_daily/fragility_timeseries.png
- reports/figures/fragility_daily/fragility_distribution.png
- reports/figures/fragility_daily/top_days_table.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# CONFIG: update paths here only
# -----------------------------
INPUT_PARQUET = Path("data/processed/eia930/eia930_balance_2025_pjm_ciso.parquet")
OUTPUT_PARQUET = Path("data/processed/fragility/daily_fragility_pjm_ciso_2025.parquet")

# ✅ Concept-based figures directory (your preference)
FIG_DIR = Path("reports/figures/fragility_daily")
TOP_DAYS_CSV = FIG_DIR / "top_days_table.csv"

# Core columns expected from Module 1/2 output
COL_BA = "balancing_authority"
COL_TS = "timestamp_utc"
COL_DEMAND = "demand_(mw)_(adjusted)"

# Fragility feature settings
NEAR_PEAK_THRESHOLD = 0.95  # hours at >= 95% of daily peak count as "near peak"
TOP_N_DAYS = 10


def ensure_dirs() -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_hourly() -> pd.DataFrame:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Could not find input parquet at: {INPUT_PARQUET}\n"
            f"Confirm Module 1 created it, or update INPUT_PARQUET in this script."
        )

    df = pd.read_parquet(INPUT_PARQUET)

    # Basic column checks
    missing = [c for c in [COL_BA, COL_TS, COL_DEMAND] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Ensure timestamp is datetime
    df[COL_TS] = pd.to_datetime(df[COL_TS], errors="coerce")
    if df[COL_TS].isna().any():
        bad = df[df[COL_TS].isna()].head(5)
        raise ValueError(
            "Some timestamps could not be parsed into datetime. Example rows:\n"
            f"{bad[[COL_BA, COL_TS, COL_DEMAND]]}"
        )

    # Sort for ramp calculation
    df = df.sort_values([COL_BA, COL_TS]).reset_index(drop=True)

    # Keep only rows with demand present (fragility relies on demand)
    df = df.dropna(subset=[COL_DEMAND]).copy()

    return df


def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    # Hourly ramp per BA
    df["hourly_ramp_mw"] = df.groupby(COL_BA)[COL_DEMAND].diff()

    # Daily key
    df["date"] = df[COL_TS].dt.date

    # Daily aggregates (core stats)
    daily = (
        df.groupby([COL_BA, "date"])
        .agg(
            daily_peak_mw=(COL_DEMAND, "max"),
            daily_min_mw=(COL_DEMAND, "min"),
            daily_max_ramp_mw=("hourly_ramp_mw", lambda x: np.nanmax(np.abs(x.values))),
            daily_ramp_std=("hourly_ramp_mw", "std"),
            hours_observed=(COL_DEMAND, "size"),
        )
        .reset_index()
    )

    daily["daily_range_mw"] = daily["daily_peak_mw"] - daily["daily_min_mw"]

    # Percent of hours near peak (merge daily peak back to hourly)
    df = df.merge(
        daily[[COL_BA, "date", "daily_peak_mw"]],
        on=[COL_BA, "date"],
        how="left",
        validate="many_to_one",
    )

    df["near_peak"] = df[COL_DEMAND] >= (NEAR_PEAK_THRESHOLD * df["daily_peak_mw"])

    pct_near_peak = (
        df.groupby([COL_BA, "date"])["near_peak"]
        .mean()
        .reset_index(name="pct_hours_near_peak")
    )

    daily = daily.merge(pct_near_peak, on=[COL_BA, "date"], how="left", validate="one_to_one")

    # Sanity clamp (should already be [0,1], but avoid float weirdness)
    daily["pct_hours_near_peak"] = daily["pct_hours_near_peak"].clip(0, 1)

    return daily


def add_fragility_scores(daily: pd.DataFrame) -> pd.DataFrame:
    # Features used for composite (simple + defensible)
    feat_cols = ["daily_peak_mw", "daily_max_ramp_mw", "pct_hours_near_peak"]

    # Z-scores within each BA
    for col in feat_cols:
        zcol = f"z_{col}"
        daily[zcol] = (
            daily.groupby(COL_BA)[col]
            .transform(
                lambda x: (x - x.mean())
                / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
            )
        )

    # Composite z-score fragility
    daily["fragility_z"] = (
        daily["z_daily_peak_mw"]
        + daily["z_daily_max_ramp_mw"]
        + daily["z_pct_hours_near_peak"]
    )

    # 0–1 scaled fragility (per BA)
    def minmax(s: pd.Series) -> pd.Series:
        smin, smax = s.min(), s.max()
        if pd.isna(smin) or pd.isna(smax) or smax == smin:
            return pd.Series(np.nan, index=s.index)
        return (s - smin) / (smax - smin)

    daily["fragility_01"] = daily.groupby(COL_BA)["fragility_z"].transform(minmax)

    return daily


def save_outputs(daily: pd.DataFrame) -> None:
    # Save to parquet
    daily.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Wrote daily fragility parquet: {OUTPUT_PARQUET}")
    
    # Save to CSV
    csv_path = OUTPUT_PARQUET.with_suffix('.csv')
    daily.to_csv(csv_path, index=False)
    print(f"Wrote daily fragility CSV: {csv_path}")
    
    # Save to JSON
    json_path = OUTPUT_PARQUET.with_suffix('.json')
    daily.to_json(json_path, orient='records', date_format='iso', indent=2)
    print(f"Wrote daily fragility JSON: {json_path}")


def plot_timeseries(daily: pd.DataFrame) -> None:
    """
    Writes a single stacked time series figure with two panels:
      (a) min–max scaled fragility_01 (per BA)
      (b) robust-scaled fragility using 5th–95th percentiles (per BA), clipped to [0,1]
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    bas = sorted(d[COL_BA].unique())

    # --- Robust scaling helper (5th–95th percentile, per BA) ---
    def robust_scale_5_95(s: pd.Series) -> pd.Series:
        arr = s.astype(float).to_numpy()
        p5, p95 = np.nanpercentile(arr, [5, 95])
        if not np.isfinite(p5) or not np.isfinite(p95) or p95 == p5:
            return pd.Series(np.nan, index=s.index)
        scaled = (s - p5) / (p95 - p5)
        return scaled.clip(0, 1)

    # Use fragility_z as the base signal for robust scaling
    d["fragility_robust_5_95"] = d.groupby(COL_BA)["fragility_z"].transform(robust_scale_5_95)

    # --- Create stacked figure ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True
    )

    # Panel (a): existing min–max scaled series (fragility_01)
    for ba in bas:
        sub = d[d[COL_BA] == ba].sort_values("date")
        ba_label = "CAISO" if ba == "CISO" else ba
        ax1.plot(sub["date"], sub["fragility_01"], label=ba_label)

    ax1.set_title("Daily Fragility (Min–max scaled 0–1) — Time Series")
    ax1.set_ylabel("Fragility (0–1)")
    ax1.legend(loc="upper right")

    # Panel (b): robust 5th–95th scaled series
    for ba in bas:
        sub = d[d[COL_BA] == ba].sort_values("date")
        ba_label = "CAISO" if ba == "CISO" else ba
        ax2.plot(sub["date"], sub["fragility_robust_5_95"], label=ba_label)

    ax2.set_title("Daily Fragility (Robust-scaled 5th–95th pct, clipped) — Time Series")
    ax2.set_ylabel("Fragility (robust 0–1)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    # Format x-axis as MM-YYYY (apply to bottom axis only)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    out = FIG_DIR / "fragility_timeseries_stacked.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote plot: {out}")



def plot_distribution(daily: pd.DataFrame) -> None:
    bas = sorted(daily[COL_BA].unique())

    # Distribution of fragility_z
    plt.figure(figsize=(10, 5))
    for ba in bas:
        sub = daily[daily[COL_BA] == ba]["fragility_z"].dropna()
        plt.hist(sub, bins=40, alpha=0.5, label=ba)
    plt.title("Distribution of Daily Fragility (z-score composite)")
    plt.xlabel("fragility_z")
    plt.ylabel("Count of days")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "fragility_distribution.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote plot: {out}")


def export_top_days(daily: pd.DataFrame) -> None:
    # Top fragile days by fragility_z for each BA
    top_rows = []
    for ba, sub in daily.groupby(COL_BA):
        sub = sub.sort_values("fragility_z", ascending=False).head(TOP_N_DAYS).copy()
        sub["rank"] = np.arange(1, len(sub) + 1)
        top_rows.append(sub)

    top = pd.concat(top_rows, ignore_index=True)

    # Keep the most relevant columns
    keep = [
        COL_BA,
        "rank",
        "date",
        "fragility_z",
        "fragility_01",
        "daily_peak_mw",
        "daily_max_ramp_mw",
        "pct_hours_near_peak",
        "daily_range_mw",
        "daily_ramp_std",
        "hours_observed",
    ]
    keep = [c for c in keep if c in top.columns]
    top = top[keep]

    top.to_csv(TOP_DAYS_CSV, index=False)
    print(f"Wrote top fragile days table: {TOP_DAYS_CSV}")


def print_sanity_summary(daily: pd.DataFrame) -> None:
    print("\n=== SANITY SUMMARY ===")
    print("Rows (BA-days):", len(daily))
    print("BAs:", sorted(daily[COL_BA].unique()))

    desc = daily.groupby(COL_BA)[["fragility_z", "fragility_01"]].describe()
    print("\nFragility stats by BA:")
    print(desc)

    # Quick checks
    bad01 = daily[(daily["fragility_01"] < -1e-9) | (daily["fragility_01"] > 1 + 1e-9)]
    if len(bad01) > 0:
        print("\nWARNING: fragility_01 outside [0,1] for some rows (should not happen).")
        print(bad01[[COL_BA, "date", "fragility_01"]].head(10))

    badpct = daily[(daily["pct_hours_near_peak"] < -1e-9) | (daily["pct_hours_near_peak"] > 1 + 1e-9)]
    if len(badpct) > 0:
        print("\nWARNING: pct_hours_near_peak outside [0,1] for some rows (should not happen).")
        print(badpct[[COL_BA, "date", "pct_hours_near_peak"]].head(10))


def main() -> None:
    ensure_dirs()
    df = load_hourly()
    daily = engineer_daily_features(df)
    daily = add_fragility_scores(daily)

    save_outputs(daily)
    print_sanity_summary(daily)

    plot_timeseries(daily)
    plot_distribution(daily)
    export_top_days(daily)

    try:
        print("\n✅ Module 3 complete.")
    except UnicodeEncodeError:
        print("\nModule 3 complete.")
    print(f"- Daily parquet: {OUTPUT_PARQUET}")
    print(f"- Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
