"""Main pipeline orchestrator: data → graphs → features → models → evaluation → tables.

Usage:
    python run_pipeline.py --data data/sample/synthetic_payments.csv
    python run_pipeline.py --data data/raw/ons_payments.csv --config config/settings.yaml
    python run_pipeline.py --generate-sample  # Generate synthetic data and run
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, quarter_to_index, get_period_label
from src.data_loader import load_payment_data, generate_sample_data
from src.graph_builder import build_quarterly_graphs, get_node_list, build_adjacency_matrix
from src.feature_extractor import extract_all_features
from src.target_builder import (
    compute_growth_rates,
    build_traditional_features,
    get_traditional_feature_columns,
    get_network_feature_columns,
)
from src.model_trainer import train_all_specifications
from src.evaluator import compute_metrics, diebold_mariano_test, period_analysis
from src.table_generator import (
    table1_top_industries,
    table2_model_comparison,
    table3_period_performance,
    table4_network_evolution,
    save_tables,
)


def run_pipeline(data_path, config_path="config/settings.yaml", output_dir="outputs"):
    """Execute the full analysis pipeline."""
    print("=" * 70)
    print("Network Structure in Payment Flows — Analysis Pipeline")
    print("=" * 70)

    # 1. Load config
    print("\n[1/7] Loading configuration...")
    config = load_config(config_path)

    # 2. Load data
    print("\n[2/7] Loading payment data...")
    df = load_payment_data(data_path, config)

    # 3. Build quarterly graphs
    print("\n[3/7] Constructing quarterly payment networks...")
    graphs = build_quarterly_graphs(df)
    node_list = get_node_list(graphs)
    print(f"  Built {len(graphs)} quarterly graphs with {len(node_list)} industries")

    # Build adjacency matrices
    adjacency_matrices = {}
    for quarter, G in graphs.items():
        adjacency_matrices[quarter] = build_adjacency_matrix(G, node_list)

    # 4. Extract features
    print("\n[4/7] Extracting graph-theoretic features...")
    node_features, network_features, edge_features = extract_all_features(
        graphs, node_list, adjacency_matrices
    )
    print(f"  Node features: {list(node_features.values())[0].shape[1]} per node per quarter")
    print(f"  Network features: {len(list(network_features.values())[0])} global metrics per quarter")

    # 5. Build targets and merge features
    print("\n[5/7] Building prediction targets and merging features...")
    growth_df = compute_growth_rates(df)
    growth_df = build_traditional_features(growth_df)
    print(f"  Growth rate observations: {len(growth_df):,}")

    # Merge edge-level network features into growth_df
    all_edge_features = []
    for quarter, ef in edge_features.items():
        all_edge_features.append(ef)
    edge_feat_df = pd.concat(all_edge_features, ignore_index=True) if all_edge_features else pd.DataFrame()

    if not edge_feat_df.empty:
        merged = growth_df.merge(
            edge_feat_df,
            on=["source", "target", "quarter"],
            how="left",
            suffixes=("", "_net"),
        )
    else:
        merged = growth_df.copy()

    # Fill NaN features with 0
    merged = merged.fillna(0)

    traditional_cols = get_traditional_feature_columns(merged)
    network_cols = get_network_feature_columns(merged)
    print(f"  Traditional features: {len(traditional_cols)}")
    print(f"  Network features: {len(network_cols)}")

    # 6. Train models
    print("\n[6/7] Training models (expanding window CV)...")
    spec_results = train_all_specifications(merged, traditional_cols, network_cols, config)

    # 7. Evaluate and generate tables
    print("\n[7/7] Evaluating and generating result tables...")

    # Compute metrics for each specification
    eval_results = {}
    for spec_name, result in spec_results.items():
        if result is not None:
            metrics = compute_metrics(result["y_true"], result["y_pred"])
            eval_results[spec_name] = metrics
            print(f"  {spec_name}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    # Diebold-Mariano test
    if "Traditional" in spec_results and "Combined" in spec_results:
        dm = diebold_mariano_test(
            spec_results["Traditional"]["y_true"],
            spec_results["Traditional"]["y_pred"],
            spec_results["Combined"]["y_pred"],
        )
        print(f"  Diebold-Mariano: DM={dm['dm_stat']:.2f}, p={dm['p_value']:.4f}")

    # Period analysis
    print("\n  Period analysis:")
    period_results_by_spec = {}
    for spec_name, result in spec_results.items():
        if result is not None:
            # Build period map from time indices
            quarters = sorted(graphs.keys())
            quarter_order_map = {quarter_to_index(q): q for q in quarters}
            time_to_period = {}
            for t_idx in set(result["time_idx"]):
                q_str = quarter_order_map.get(t_idx, "")
                if q_str:
                    period = get_period_label(q_str, config)
                    time_to_period[t_idx] = period

            period_results = period_analysis(
                result["y_true"], result["y_pred"],
                result["time_idx"], time_to_period,
            )
            period_results_by_spec[spec_name] = period_results
            for period, m in period_results.items():
                print(f"    {spec_name} / {period}: R2={m['r2']:.4f}")

    # Generate tables
    print("\n  Generating tables...")
    tables = {}
    tables["table1_top_industries"] = table1_top_industries(df)
    tables["table2_model_comparison"] = table2_model_comparison(spec_results, eval_results)
    tables["table3_period_performance"] = table3_period_performance(period_results_by_spec)
    tables["table4_network_evolution"] = table4_network_evolution(network_features)

    save_tables(tables, os.path.join(output_dir, "tables"))

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)

    return {
        "graphs": graphs,
        "node_features": node_features,
        "network_features": network_features,
        "spec_results": spec_results,
        "eval_results": eval_results,
        "tables": tables,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Network Payment Flow Analysis Pipeline"
    )
    parser.add_argument(
        "--data", type=str, help="Path to payment data CSV/Excel file"
    )
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="Path to configuration YAML (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory (default: outputs/)"
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate synthetic sample data and run pipeline on it"
    )
    args = parser.parse_args()

    if args.generate_sample:
        print("Generating synthetic sample data...")
        sample_path = "data/sample/synthetic_payments.csv"
        generate_sample_data(output_path=sample_path)
        args.data = sample_path

    if not args.data:
        parser.error("Either --data or --generate-sample is required")

    run_pipeline(args.data, args.config, args.output)


if __name__ == "__main__":
    main()
