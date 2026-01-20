import numpy as np
import pandas as pd


EPS = 1e-9


def add_size_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds BA-size-normalized stress features.
    """
    df = df.copy()

    df["max_abs_ramp_pct_mean"] = df["max_abs_ramp_mw"] / (df["demand_mean_mw"] + EPS)
    df["ramp_p95_abs_pct_mean"] = df["ramp_p95_abs_mw"] / (df["demand_mean_mw"] + EPS)
    df["intra_day_range_pct_mean"] = df["intra_day_range_mw"] / (df["demand_mean_mw"] + EPS)
    df["demand_std_pct_mean"] = df["demand_std_mw"] / (df["demand_mean_mw"] + EPS)

    return df


def add_rolling_zscores(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    cols: list[str],
    window: int = 7,
    min_periods: int = 4,
) -> pd.DataFrame:
    """
    Adds rolling z-score columns per group.
    """
    df = df.sort_values([group_col, time_col]).copy()

    def _z(s: pd.Series) -> pd.Series:
        mu = s.rolling(window, min_periods=min_periods).mean()
        sd = s.rolling(window, min_periods=min_periods).std()
        return (s - mu) / (sd + EPS)

    for c in cols:
        df[f"{c}_z{window}"] = (
            df.groupby(group_col)[c]
              .apply(_z)
              .reset_index(level=0, drop=True)
        )

    return df
