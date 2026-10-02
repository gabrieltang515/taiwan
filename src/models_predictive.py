"""
Predictive models module.

For each target we:
  1. Use LAGGED features only (no lookahead).
  2. Split by time (75 / 25).
  3. Fit baseline OLS, Ridge, Lasso.
  4. Optionally fit Logistic Regression for directional prediction.
  5. Report RMSE, MAE, directional accuracy, and comparison vs naïve
     (naïve = predict zero every month, i.e. no-change benchmark).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.config import TARGETS, PRED_FEATURES, TRAIN_RATIO, OUT_TABLES
from src.utils import time_train_test_split, directional_accuracy, save_table


def _prepare_Xy(
    df: pd.DataFrame, target_col: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """Drop NaN, return X and y aligned."""
    avail = [f for f in features if f in df.columns]
    if not avail or target_col not in df.columns:
        return None, None
    sub = df[[target_col] + avail].dropna()
    if len(sub) < 30:
        return None, None
    return sub[avail], sub[target_col]


def _evaluate_regression(y_true, y_pred, y_train_mean) -> dict:
    """Compute RMSE, MAE, directional accuracy, and naïve RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    da = directional_accuracy(y_true.values, y_pred)
    naive_pred = np.full_like(y_true.values, 0.0)  # predict no change
    naive_rmse = np.sqrt(mean_squared_error(y_true, naive_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "dir_acc": da,
        "naive_rmse": naive_rmse,
        "rmse_vs_naive": rmse / naive_rmse if naive_rmse > 0 else np.nan,
    }


def run_predictive_for_target(
    df: pd.DataFrame,
    target_col: str,
    target_label: str,
) -> tuple[list[dict], dict]:
    """
    Run all predictive models for one target.

    Returns a list of metric dicts and a dict of predictions for plotting.
    """
    X, y = _prepare_Xy(df, target_col, PRED_FEATURES)
    if X is None:
        print(f"  ⚠ skipping {target_label}: insufficient data")
        return [], {}

    # Time-ordered split
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Standardise for regularised models (fit on train only)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                              index=X_train.index, columns=X_train.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test),
                             index=X_test.index, columns=X_test.columns)

    rows = []
    predictions = {}

    # -- Linear regression (unscaled) --
    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    metrics = _evaluate_regression(y_test, pred_lr, y_train.mean())
    metrics.update({"model": "OLS", "target": target_label})
    rows.append(metrics)
    predictions["OLS"] = pd.Series(pred_lr, index=y_test.index)

    # -- Ridge --
    ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
    pred_ridge = ridge.predict(X_test_sc)
    metrics = _evaluate_regression(y_test, pred_ridge, y_train.mean())
    metrics.update({"model": "Ridge", "target": target_label})
    rows.append(metrics)
    predictions["Ridge"] = pd.Series(pred_ridge, index=y_test.index)

    # -- Lasso --
    lasso = Lasso(alpha=0.1, max_iter=5000).fit(X_train_sc, y_train)
    pred_lasso = lasso.predict(X_test_sc)
    metrics = _evaluate_regression(y_test, pred_lasso, y_train.mean())
    metrics.update({"model": "Lasso", "target": target_label})
    rows.append(metrics)
    predictions["Lasso"] = pd.Series(pred_lasso, index=y_test.index)

    # Lasso feature selection summary
    lasso_coefs = pd.Series(lasso.coef_, index=X_train.columns)
    selected = lasso_coefs[lasso_coefs.abs() > 0]
    if len(selected) > 0:
        print(f"    Lasso selected {len(selected)}/{len(X_train.columns)} features")

    # -- Logistic regression (directional) --
    y_dir_train = (y_train > 0).astype(int)
    y_dir_test = (y_test > 0).astype(int)

    if y_dir_train.nunique() > 1:
        log_clf = LogisticRegression(max_iter=1000, C=1.0)
        log_clf.fit(X_train_sc, y_dir_train)
        pred_dir = log_clf.predict(X_test_sc)
        dir_acc = (pred_dir == y_dir_test.values).mean()
        rows.append({
            "model": "Logistic",
            "target": target_label,
            "dir_acc": dir_acc,
            "rmse": np.nan,
            "mae": np.nan,
            "naive_rmse": np.nan,
            "rmse_vs_naive": np.nan,
        })

    # Store actual for plotting
    predictions["actual"] = y_test

    return rows, predictions


def run_all_predictive(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run predictive models for all targets.

    Returns
    -------
    perf_table : pd.DataFrame   – performance metrics
    all_preds  : dict            – {target: {model: Series of predictions}}
    """
    print("── Predictive models ──")
    all_rows = []
    all_preds = {}

    for target_label, target_col in TARGETS.items():
        print(f"  Target: {target_label}")
        rows, preds = run_predictive_for_target(df, target_col, target_label)
        all_rows.extend(rows)
        if preds:
            all_preds[target_label] = preds

    perf = pd.DataFrame(all_rows)
    if not perf.empty:
        save_table(perf, OUT_TABLES / "predictive_performance.csv")
        print("\n  Predictive performance summary:")
        print(perf.to_string(index=False, float_format="%.4f"))

    return perf, all_preds


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_csv(DATA_PROC / "features_monthly.csv",
                     index_col=0, parse_dates=True)
    run_all_predictive(df)
