"""Generate paper-style result tables (Tables 1-4)."""

import pandas as pd
import numpy as np
from pathlib import Path


def table1_top_industries(df, n_top=10):
    """Table 1: Top N industries by inter-industry payment volume.

    Args:
        df: Raw payment DataFrame with [source, target, value, quarter].
        n_top: Number of top industries to show.

    Returns:
        DataFrame with columns: Rank, Industry, Volume, Share(%).
    """
    # Total outgoing + incoming volume per industry
    out_vol = df.groupby("source")["value"].sum()
    in_vol = df.groupby("target")["value"].sum()
    total_vol = out_vol.add(in_vol, fill_value=0).sort_values(ascending=False)

    grand_total = total_vol.sum()
    top = total_vol.head(n_top)

    result = pd.DataFrame({
        "Rank": range(1, n_top + 1),
        "Industry": top.index,
        "Volume": top.values,
        "Share (%)": (top.values / grand_total * 100).round(1),
    })

    # Add total row
    top_total = top.sum()
    result.loc[len(result)] = {
        "Rank": "",
        "Industry": f"Top {n_top} Total",
        "Volume": top_total,
        "Share (%)": round(top_total / grand_total * 100, 1),
    }

    return result


def table2_model_comparison(spec_results, eval_results):
    """Table 2: Network-enhanced model performance comparison.

    Args:
        spec_results: Dict from model_trainer.train_all_specifications.
        eval_results: Dict mapping spec_name → metrics from evaluator.

    Returns:
        DataFrame matching Table 2 format.
    """
    rows = []
    traditional_r2 = None

    for spec_name in ["Traditional", "Network", "Combined"]:
        if spec_name not in eval_results:
            continue
        metrics = eval_results[spec_name]
        r2 = metrics["r2"]
        r2_std = metrics.get("r2_std", 0)
        rmse = metrics["rmse"] * 100  # Convert to percentage
        mae = metrics["mae"] * 100

        if spec_name == "Traditional":
            traditional_r2 = r2
            vs_trad = "Baseline"
        else:
            diff = (r2 - traditional_r2) * 100 if traditional_r2 is not None else 0
            vs_trad = f"{diff:+.1f} pp"

        rows.append({
            "Feature Set": f"{spec_name} Features Only" if spec_name != "Combined" else "Combined (Network + Traditional)",
            "R2": f"{r2:.3f} +/- {r2_std:.3f}",
            "RMSE (%)": f"{rmse:.2f}",
            "MAE (%)": f"{mae:.2f}",
            "vs. Traditional": vs_trad,
        })

    return pd.DataFrame(rows)


def table3_period_performance(period_results_by_spec):
    """Table 3: Network performance improvements by economic period.

    Args:
        period_results_by_spec: Dict[spec_name, Dict[period, metrics]].

    Returns:
        DataFrame matching Table 3 format.
    """
    rows = []
    period_order = ["pre_pandemic", "pandemic", "recovery"]
    period_labels = {
        "pre_pandemic": "Pre-Pandemic (2017-2019)",
        "pandemic": "Pandemic (2020-2021)",
        "recovery": "Recovery (2022-2024)",
    }

    trad = period_results_by_spec.get("Traditional", {})
    combined = period_results_by_spec.get("Combined", {})

    for period in period_order:
        if period in trad and period in combined:
            trad_r2 = trad[period]["r2"]
            comb_r2 = combined[period]["r2"]
            improvement = (comb_r2 - trad_r2) * 100
            rows.append({
                "Period": period_labels.get(period, period),
                "Traditional R2": f"{trad_r2:.3f}",
                "Enhanced R2": f"{comb_r2:.3f}",
                "Improvement": f"+{improvement:.1f} pp",
            })

    # Full sample
    if "Traditional" in period_results_by_spec and "Combined" in period_results_by_spec:
        # Use the overall spec results if available
        pass

    return pd.DataFrame(rows)


def table4_network_evolution(network_features_by_quarter):
    """Table 4: Evolution of payment network structure.

    Args:
        network_features_by_quarter: Dict[quarter_str, dict of network metrics].

    Returns:
        DataFrame matching Table 4 format.
    """
    rows = []
    quarters = sorted(network_features_by_quarter.keys())

    for q in quarters:
        nf = network_features_by_quarter[q]
        rows.append({
            "Quarter": q,
            "Density": round(nf.get("density", 0), 3),
            "Edges": nf.get("num_edges", 0),
            "Avg Path Length": round(nf.get("average_path_length", 0), 2),
            "Clustering": round(nf.get("average_clustering", 0), 2),
        })

    df = pd.DataFrame(rows)

    # Add change row if enough data
    if len(df) >= 2:
        first = df.iloc[0]
        last = df.iloc[-1]
        change = {
            "Quarter": f"Change {first['Quarter']}-{last['Quarter']}",
            "Density": f"+{((last['Density'] / first['Density']) - 1) * 100:.1f}%" if first["Density"] > 0 else "N/A",
            "Edges": f"+{((last['Edges'] / first['Edges']) - 1) * 100:.1f}%" if first["Edges"] > 0 else "N/A",
            "Avg Path Length": f"{((last['Avg Path Length'] / first['Avg Path Length']) - 1) * 100:.1f}%" if first["Avg Path Length"] > 0 else "N/A",
            "Clustering": f"+{((last['Clustering'] / first['Clustering']) - 1) * 100:.1f}%" if first["Clustering"] > 0 else "N/A",
        }
        df.loc[len(df)] = change

    return df


def save_tables(tables, output_dir="outputs/tables"):
    """Save all tables as CSV files.

    Args:
        tables: Dict mapping table name → DataFrame.
        output_dir: Output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {path}")
