"""Model training with expanding window cross-validation.

Three specifications:
1. Traditional: lagged growth, seasonal indicators, industry fixed effects
2. Network: graph-theoretic features only
3. Combined: all features

Uses Random Forest and Gradient Boosting ensemble methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def expanding_window_cv(X, y, time_indices, model, min_train_periods=4):
    """Expanding window cross-validation respecting temporal ordering.

    For each time step t >= min_train_periods:
      Train on all data from periods 0..t-1, predict period t.

    Args:
        X: Feature matrix (DataFrame or array).
        y: Target values (Series or array).
        time_indices: Array of integer time indices per observation.
        model: Sklearn estimator (will be cloned per fold).
        min_train_periods: Minimum number of training periods before first prediction.

    Returns:
        Dict with keys:
        - y_true: array of actual values (all test folds concatenated)
        - y_pred: array of predicted values
        - time_idx: array of time indices for each prediction
        - fold_results: list of per-fold dicts {period, r2, rmse, n_samples}
    """
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    time_indices = np.array(time_indices)

    unique_periods = sorted(np.unique(time_indices))
    all_y_true, all_y_pred, all_time = [], [], []
    fold_results = []

    for t_idx in range(min_train_periods, len(unique_periods)):
        t = unique_periods[t_idx]
        train_periods = unique_periods[:t_idx]

        train_mask = np.isin(time_indices, train_periods)
        test_mask = time_indices == t

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Clone and fit model
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_test_s)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_time.extend([t] * len(y_test))

        # Per-fold metrics
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        fold_results.append({
            "period": t,
            "r2": r2,
            "rmse": rmse,
            "n_samples": len(y_test),
        })

    return {
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "time_idx": np.array(all_time),
        "fold_results": fold_results,
    }


def train_all_specifications(merged_df, traditional_cols, network_cols, config):
    """Train all three model specifications with both RF and GBM.

    Args:
        merged_df: DataFrame with all features, growth_rate target, and quarter_order.
        traditional_cols: List of traditional feature column names.
        network_cols: List of network feature column names.
        config: Config dict with model hyperparameters.

    Returns:
        Dict mapping spec_name → best results dict from expanding_window_cv.
    """
    y = merged_df["growth_rate"].values
    time_idx = merged_df["quarter_order"].values
    model_cfg = config.get("model", {})
    rs = model_cfg.get("random_state", 42)
    min_window = model_cfg.get("min_expanding_window", 4)

    # Define specifications
    specs = {
        "Traditional": traditional_cols,
        "Network": network_cols,
        "Combined": traditional_cols + network_cols,
    }

    # Define models
    rf_params = model_cfg.get("rf_params", {})
    gbm_params = model_cfg.get("gbm_params", {})
    models = {
        "RandomForest": RandomForestRegressor(random_state=rs, **rf_params),
        "GradientBoosting": GradientBoostingRegressor(random_state=rs, **gbm_params),
    }

    results = {}
    for spec_name, feature_cols in specs.items():
        # Filter to columns that exist
        available_cols = [c for c in feature_cols if c in merged_df.columns]
        if not available_cols:
            print(f"  Warning: No features available for {spec_name} specification")
            continue

        X = merged_df[available_cols].values
        print(f"  {spec_name}: {len(available_cols)} features, {len(y)} observations")

        best_result = None
        best_r2 = -np.inf

        for model_name, model in models.items():
            result = expanding_window_cv(X, y, time_idx, model, min_window)
            # Compute overall R2
            ss_res = np.sum((result["y_true"] - result["y_pred"]) ** 2)
            ss_tot = np.sum((result["y_true"] - np.mean(result["y_true"])) ** 2)
            overall_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            result["overall_r2"] = overall_r2
            result["model_name"] = model_name
            result["spec_name"] = spec_name
            result["n_features"] = len(available_cols)

            print(f"    {model_name}: R2={overall_r2:.4f}")

            if overall_r2 > best_r2:
                best_r2 = overall_r2
                best_result = result

        results[spec_name] = best_result

    return results
