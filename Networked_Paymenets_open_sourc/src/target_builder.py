"""Target variable construction and traditional feature engineering.

Implements:
- Eq. 9: Quarter-on-quarter growth rates
- Lagged growth features
- Seasonal indicators
- Industry fixed effects
"""

import pandas as pd
import numpy as np
from src.utils import parse_quarter


def compute_growth_rates(df):
    """Compute quarter-on-quarter growth rates for bilateral payment flows (Eq. 9).

    g_{ij}^{(t)} = (w_{ij}^{(t)} - w_{ij}^{(t-1)}) / w_{ij}^{(t-1)}

    Args:
        df: DataFrame with columns [source, target, value, quarter].

    Returns:
        DataFrame with columns: source, target, quarter, prev_quarter,
        value, prev_value, growth_rate.
    """
    quarters = sorted(df["quarter"].unique())
    quarter_idx = {q: i for i, q in enumerate(quarters)}

    # Aggregate to unique (source, target, quarter) level
    agg = df.groupby(["source", "target", "quarter"])["value"].sum().reset_index()

    # Pivot to get value per quarter for each pair
    records = []
    for i in range(1, len(quarters)):
        q_curr = quarters[i]
        q_prev = quarters[i - 1]

        curr = agg[agg["quarter"] == q_curr].set_index(["source", "target"])["value"]
        prev = agg[agg["quarter"] == q_prev].set_index(["source", "target"])["value"]

        # Only compute for pairs present in both quarters with positive values
        common_pairs = curr.index.intersection(prev.index)
        for pair in common_pairs:
            v_curr = curr[pair]
            v_prev = prev[pair]
            if v_prev > 0:
                growth = (v_curr - v_prev) / v_prev
                records.append({
                    "source": pair[0],
                    "target": pair[1],
                    "quarter": q_curr,
                    "prev_quarter": q_prev,
                    "value": v_curr,
                    "prev_value": v_prev,
                    "growth_rate": growth,
                    "quarter_order": quarter_idx[q_curr],
                })

    return pd.DataFrame(records)


def add_lagged_growth(growth_df, n_lags=2):
    """Add lagged growth rate features (g_{ij}^{t-1}, g_{ij}^{t-2}, ...).

    Args:
        growth_df: DataFrame from compute_growth_rates.
        n_lags: Number of lag periods.

    Returns:
        DataFrame with additional lag columns, rows without sufficient history dropped.
    """
    df = growth_df.copy()
    df = df.sort_values(["source", "target", "quarter_order"])

    for lag in range(1, n_lags + 1):
        df[f"growth_lag_{lag}"] = df.groupby(["source", "target"])["growth_rate"].shift(lag)

    # Drop rows missing any lag
    lag_cols = [f"growth_lag_{lag}" for lag in range(1, n_lags + 1)]
    df = df.dropna(subset=lag_cols)

    return df


def add_seasonal_indicators(df):
    """Add quarterly seasonal dummy variables (Q1, Q2, Q3, Q4).

    Args:
        df: DataFrame with 'quarter' column (format: 'YYYY-QN').

    Returns:
        DataFrame with added columns: season_Q1, season_Q2, season_Q3, season_Q4.
    """
    df = df.copy()
    df["_q_num"] = df["quarter"].apply(lambda x: parse_quarter(x)[1])
    for q in range(1, 5):
        df[f"season_Q{q}"] = (df["_q_num"] == q).astype(int)
    df = df.drop(columns=["_q_num"])
    return df


def add_industry_fixed_effects(df, max_industries=50):
    """Add industry fixed effects as binary indicators.

    To keep dimensionality manageable, uses the top N most frequent
    industries by occurrence.

    Args:
        df: DataFrame with 'source' and 'target' columns.
        max_industries: Maximum number of industry dummies per role.

    Returns:
        DataFrame with added industry indicator columns.
    """
    df = df.copy()

    # Get top industries
    all_industries = pd.concat([df["source"], df["target"]]).value_counts()
    top_industries = all_industries.head(max_industries).index.tolist()

    for ind in top_industries:
        safe_name = str(ind)[:30].replace(" ", "_")
        df[f"src_fe_{safe_name}"] = (df["source"] == ind).astype(int)
        df[f"tgt_fe_{safe_name}"] = (df["target"] == ind).astype(int)

    return df


def build_traditional_features(df):
    """Build the complete traditional feature set.

    Combines lagged growth rates, seasonal indicators, and industry fixed effects.

    Args:
        df: DataFrame from compute_growth_rates.

    Returns:
        DataFrame with all traditional features added.
    """
    df = add_lagged_growth(df, n_lags=2)
    df = add_seasonal_indicators(df)
    df = add_industry_fixed_effects(df)
    return df


def get_traditional_feature_columns(df):
    """Get column names that constitute the traditional feature set."""
    cols = []
    # Lagged growth
    cols += [c for c in df.columns if c.startswith("growth_lag_")]
    # Seasonal
    cols += [c for c in df.columns if c.startswith("season_")]
    # Industry fixed effects
    cols += [c for c in df.columns if c.startswith("src_fe_") or c.startswith("tgt_fe_")]
    return cols


def get_network_feature_columns(df):
    """Get column names that constitute the network feature set."""
    cols = []
    cols += [c for c in df.columns if c.startswith("src_") and not c.startswith("src_fe_")]
    cols += [c for c in df.columns if c.startswith("tgt_") and not c.startswith("tgt_fe_")]
    cols += [c for c in df.columns if c.startswith("net_")]
    if "multihop_strength" in df.columns:
        cols.append("multihop_strength")
    return cols
