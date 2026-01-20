import pandas as pd

# -----------------------
# Load daily fragility data
# -----------------------
df = pd.read_parquet(
    "data/processed/fragility/daily_fragility_pjm_ciso_2025.parquet"
)

print("\n--- BASIC SHAPE ---")
print("Rows:", len(df))
print("Unique BA-days:", df[["balancing_authority", "date"]].drop_duplicates().shape[0])

print("\n--- BALANCING AUTHORITIES ---")
print(df["balancing_authority"].value_counts())

print("\n--- DATE RANGE ---")
print("Min date:", df["date"].min())
print("Max date:", df["date"].max())

print("\n--- COLUMN INVENTORY ---")
print(sorted(df.columns.tolist()))

feature_cols = [
    "daily_peak_mw",
    "daily_min_mw",
    "daily_range_mw",
    "daily_max_ramp_mw",
    "daily_ramp_std",
    "pct_hours_near_peak",
    "z_daily_peak_mw",
    "z_daily_max_ramp_mw",
    "z_pct_hours_near_peak",
    "fragility_z",
]


print("\n--- FEATURE MISSINGNESS ---")
print(df[feature_cols].isna().mean())

print("\n--- FEATURE RANGES ---")
print(df[feature_cols].describe().T[["min", "mean", "max"]])

print("\n--- DATA TYPES ---")
print(df[feature_cols + ["balancing_authority", "date"]].dtypes)

# Final integrity check
assert (
    df[["balancing_authority", "date"]].drop_duplicates().shape[0] == len(df)
), "Duplicate BA-days detected"

try:
    print("\n✅ Step 0 checks passed. Data is ML-ready.")
except UnicodeEncodeError:
    print("\nStep 0 checks passed. Data is ML-ready.")

