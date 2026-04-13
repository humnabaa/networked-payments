"""Data loading and synthetic data generation for payment flow analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import SIC_INDUSTRY_NAMES, quarter_to_str


def _month_to_quarter(date_str):
    """Convert 'January 2019' style date string to '2019-Q1' format."""
    month_to_q = {
        "january": 1, "february": 1, "march": 1,
        "april": 2, "may": 2, "june": 2,
        "july": 3, "august": 3, "september": 3,
        "october": 4, "november": 4, "december": 4,
    }
    parts = str(date_str).strip().split()
    if len(parts) != 2:
        return None
    month_str, year_str = parts[0].lower(), parts[1]
    q = month_to_q.get(month_str)
    if q is None:
        return None
    return f"{year_str}-Q{q}"


def _sic_to_industry_name(sic_str):
    """Map a 2-digit SIC code string to its industry name."""
    try:
        sic_int = int(sic_str)
    except (ValueError, TypeError):
        return None
    return SIC_INDUSTRY_NAMES.get(sic_int)


def load_payment_data(filepath, config):
    """Load payment flow data from CSV or Excel, mapping columns via config.

    Handles the ONS Excel format (multiple header rows, monthly dates,
    suppressed values) as well as pre-processed CSV files.

    Args:
        filepath: Path to CSV or Excel file.
        config: Dict with 'schema' key containing column mappings.

    Returns:
        DataFrame with canonical columns: source, target, value, quarter
    """
    path = Path(filepath)
    schema = config["schema"]
    ons_cfg = schema.get("ons_excel", {})

    # --- Load raw data ---
    if path.suffix in (".xlsx", ".xls"):
        sheet = ons_cfg.get("sheet_name")
        skip = ons_cfg.get("skip_rows", 0)
        if sheet:
            df = pd.read_excel(filepath, sheet_name=sheet, skiprows=skip, header=None)
        else:
            df = pd.read_excel(filepath, skiprows=skip, header=None)
        # If the first data row is actually column names, use it as header
        if ons_cfg.get("extra_header_row"):
            df.columns = df.iloc[0].values
            df = df.iloc[1:].reset_index(drop=True)
    elif path.suffix == ".csv":
        df = pd.read_csv(filepath)
    elif path.suffix == ".tsv":
        df = pd.read_csv(filepath, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # --- Map columns to canonical names ---
    # Handle encoding differences (e.g. £ may read differently)
    def _find_col(df_cols, target):
        if target in df_cols:
            return target
        # Try substring match for encoding-mangled column names
        target_clean = target.lower().replace("£", "").replace("(", "").replace(")", "").strip()
        for c in df_cols:
            c_clean = c.lower().replace("£", "").replace("(", "").replace(")", "").strip()
            # Match on key words: e.g. "value" in "Value (£)"
            if target_clean.split()[0] in c_clean:
                return c
        return None

    src_col = _find_col(df.columns, schema["source_column"])
    dst_col = _find_col(df.columns, schema["destination_column"])
    val_col = _find_col(df.columns, schema["value_column"])
    time_col = _find_col(df.columns, schema["time_column"])

    resolved = {"source": src_col, "target": dst_col, "value": val_col, "date": time_col}
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise ValueError(
            f"Could not resolve columns for: {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Update config/settings.yaml schema section."
        )

    col_map = {src_col: "source", dst_col: "target", val_col: "value", time_col: "date"}

    df = df.rename(columns=col_map)

    # --- Clean suppressed / non-numeric values ---
    df = df[~df["value"].isin(["[-]", "[c]", "-", "c"])].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["source", "target", "value", "date"])
    df = df[df["value"] > 0].copy()

    # --- Convert SIC codes to industry names ---
    df["source"] = df["source"].apply(_sic_to_industry_name)
    df["target"] = df["target"].apply(_sic_to_industry_name)
    df = df.dropna(subset=["source", "target"])

    # --- Convert monthly dates to quarters and aggregate ---
    df["quarter"] = df["date"].apply(_month_to_quarter)
    df = df.dropna(subset=["quarter"])

    # Aggregate monthly values to quarterly totals per industry pair
    df = (
        df.groupby(["source", "target", "quarter"], as_index=False)["value"]
        .sum()
    )

    # Sort by quarter
    df = df.sort_values("quarter").reset_index(drop=True)

    print(f"Loaded {len(df):,} payment records (quarterly aggregated)")
    print(f"  Industries: {df['source'].nunique()} sources, {df['target'].nunique()} targets")
    print(f"  Quarters: {df['quarter'].nunique()} ({df['quarter'].min()} to {df['quarter'].max()})")
    print(f"  Total value: £{df['value'].sum():,.0f}")

    return df


def generate_sample_data(
    n_quarters=32,
    start_year=2017,
    output_path=None,
    random_state=42,
):
    """Generate synthetic payment flow data mimicking the paper's structure.

    Creates realistic inter-industry payment flows with:
    - 89 SIC-coded industry sectors
    - Financial Services as a high-centrality hub
    - COVID-19 disruption in 2020 (reduced density + volume shock)
    - Gradual network densification over time

    Args:
        n_quarters: Number of quarters to generate.
        start_year: Starting year.
        output_path: Optional path to save CSV.
        random_state: Random seed.

    Returns:
        DataFrame with columns: source_industry, destination_industry,
        payment_value, time_period, sic_code
    """
    rng = np.random.RandomState(random_state)

    # Use actual SIC codes from the mapping
    sic_codes = sorted(SIC_INDUSTRY_NAMES.keys())
    n_industries = len(sic_codes)

    # Define hub industries (higher connection probability and volume)
    hub_sics = {64, 65, 66, 46, 69, 70, 71, 62, 47}  # Financial, Wholesale, Professional, IT, Retail

    # Base connection probability matrix
    base_prob = np.full((n_industries, n_industries), 0.15)
    for i, sic_i in enumerate(sic_codes):
        for j, sic_j in enumerate(sic_codes):
            if i == j:
                base_prob[i, j] = 0.0  # No self-loops
                continue
            # Hub industries have higher connectivity
            if sic_i in hub_sics:
                base_prob[i, j] += 0.35
            if sic_j in hub_sics:
                base_prob[i, j] += 0.25
            # Same-sector affinity
            if abs(sic_i - sic_j) <= 3:
                base_prob[i, j] += 0.1

    base_prob = np.clip(base_prob, 0, 0.95)

    # Base volume weights (hub industries have much higher volumes)
    base_volume = np.ones(n_industries) * 0.5
    for i, sic in enumerate(sic_codes):
        if sic in {64, 65, 66}:
            base_volume[i] = 5.0  # Financial Services ~19% of total
        elif sic == 46:
            base_volume[i] = 3.5  # Wholesale ~14%
        elif sic in range(10, 34):
            base_volume[i] = 1.5  # Manufacturing ~11%
        elif sic == 68:
            base_volume[i] = 2.0  # Real Estate ~8%
        elif sic in {69, 70, 71}:
            base_volume[i] = 1.8  # Professional Services ~7%
        elif sic == 47:
            base_volume[i] = 1.5  # Retail ~6%
        elif sic in {41, 42, 43}:
            base_volume[i] = 1.3  # Construction ~5%
        elif sic in {58, 59, 60, 61, 62, 63}:
            base_volume[i] = 1.2  # IT & Communication ~5%

    records = []
    sic_to_name = {sic: SIC_INDUSTRY_NAMES[sic] for sic in sic_codes}

    for q_idx in range(n_quarters):
        year = start_year + q_idx // 4
        quarter = (q_idx % 4) + 1
        quarter_str = quarter_to_str(year, quarter)

        # Time-varying density (gradual increase + COVID dip)
        density_factor = 1.0 + 0.005 * q_idx  # Gradual increase
        if year == 2020:
            density_factor *= 0.85  # COVID reduction
        elif year == 2021 and quarter <= 2:
            density_factor *= 0.92  # Partial recovery

        # Time-varying volume (growth trend + COVID shock)
        volume_factor = 1.0 + 0.01 * q_idx
        if year == 2020 and quarter == 2:
            volume_factor *= 0.6  # Sharp Q2 2020 drop
        elif year == 2020 and quarter in (1, 3):
            volume_factor *= 0.8
        elif year == 2020 and quarter == 4:
            volume_factor *= 0.85

        # Seasonal effects
        seasonal = {1: 0.95, 2: 1.0, 3: 0.97, 4: 1.08}
        volume_factor *= seasonal[quarter]

        # Generate edges for this quarter
        connection_mask = rng.random((n_industries, n_industries)) < (base_prob * density_factor)
        np.fill_diagonal(connection_mask, False)

        for i in range(n_industries):
            for j in range(n_industries):
                if connection_mask[i, j]:
                    vol = (
                        base_volume[i]
                        * base_volume[j] ** 0.3
                        * volume_factor
                        * rng.lognormal(0, 0.5)
                        * 1e6  # Scale to millions
                    )
                    records.append({
                        "source_industry": sic_to_name[sic_codes[i]],
                        "destination_industry": sic_to_name[sic_codes[j]],
                        "payment_value": round(vol, 2),
                        "time_period": quarter_str,
                        "sic_code": sic_codes[i],
                    })

    df = pd.DataFrame(records)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved synthetic data: {len(df):,} records to {output_path}")

    print(f"Generated {len(df):,} synthetic payment records")
    print(f"  Industries: {df['source_industry'].nunique()}")
    print(f"  Quarters: {df['time_period'].nunique()} ({df['time_period'].min()} to {df['time_period'].max()})")

    return df
