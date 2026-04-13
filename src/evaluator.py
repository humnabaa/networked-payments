"""Model evaluation metrics and statistical tests.

Implements:
- R-squared, RMSE, MAE with bootstrap confidence intervals
- Eq. 10: Diebold-Mariano test for forecast comparison
- Period-wise performance analysis
"""

import numpy as np
from scipy import stats


def compute_metrics(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Compute R2, RMSE, MAE with bootstrap confidence intervals.

    Args:
        y_true, y_pred: Arrays of actual and predicted values.
        n_bootstrap: Number of bootstrap samples for CI.
        random_state: Random seed.

    Returns:
        Dict with metric values and confidence intervals.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)

    # Point estimates
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # Bootstrap CIs
    rng = np.random.RandomState(random_state)
    r2_boot, rmse_boot, mae_boot = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - np.mean(yt)) ** 2)
        r2_boot.append(1 - ss_r / ss_t if ss_t > 0 else 0)
        rmse_boot.append(np.sqrt(np.mean((yt - yp) ** 2)))
        mae_boot.append(np.mean(np.abs(yt - yp)))

    return {
        "r2": r2,
        "r2_ci": (np.percentile(r2_boot, 2.5), np.percentile(r2_boot, 97.5)),
        "r2_std": np.std(r2_boot),
        "rmse": rmse,
        "rmse_ci": (np.percentile(rmse_boot, 2.5), np.percentile(rmse_boot, 97.5)),
        "mae": mae,
        "mae_ci": (np.percentile(mae_boot, 2.5), np.percentile(mae_boot, 97.5)),
    }


def diebold_mariano_test(y_true, y_pred_1, y_pred_2):
    """Diebold-Mariano test for comparing forecast accuracy (Eq. 10).

    Tests H0: E[d_t] = 0 where d_t = e1_t^2 - e2_t^2.
    A positive DM statistic means model 2 is more accurate.

    Args:
        y_true: Actual values.
        y_pred_1: Predictions from model 1 (baseline).
        y_pred_2: Predictions from model 2 (challenger).

    Returns:
        Dict with DM statistic and p-value.
    """
    y_true = np.array(y_true)
    e1 = y_true - np.array(y_pred_1)
    e2 = y_true - np.array(y_pred_2)

    # Loss differential (squared errors)
    d = e1 ** 2 - e2 ** 2
    n = len(d)

    if n < 2:
        return {"dm_stat": 0.0, "p_value": 1.0}

    d_bar = np.mean(d)

    # Newey-West variance estimator (lag = floor(n^(1/3)))
    max_lag = max(1, int(np.floor(n ** (1 / 3))))
    gamma_0 = np.var(d, ddof=1)
    nw_var = gamma_0

    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_var += 2 * weight * gamma_lag

    nw_var = max(nw_var, 1e-10)  # Prevent division by zero
    dm_stat = d_bar / np.sqrt(nw_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {"dm_stat": dm_stat, "p_value": p_value}


def period_analysis(y_true, y_pred, time_indices, period_map):
    """Analyze performance by economic period (for Table 3).

    Args:
        y_true, y_pred: Arrays of actual and predicted values.
        time_indices: Array of time indices per observation.
        period_map: Dict mapping time_index → period_label.

    Returns:
        Dict mapping period_label → metrics dict.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    time_indices = np.array(time_indices)

    results = {}
    for period_label in sorted(set(period_map.values())):
        # Get time indices belonging to this period
        period_times = {t for t, p in period_map.items() if p == period_label}
        mask = np.isin(time_indices, list(period_times))

        if mask.sum() > 0:
            metrics = compute_metrics(y_true[mask], y_pred[mask])
            metrics["n_observations"] = int(mask.sum())
            results[period_label] = metrics

    return results
