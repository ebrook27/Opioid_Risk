"""
Bootstrap inference for G-computation counterfactual estimates.
 
Provides confidence intervals for the estimated effect of dispensing
reductions on mortality by resampling counties (the cross-sectional
unit) and re-running the full pipeline: train -> predict -> intervene
-> counterfactual predict.
 
The resampling unit is the county (FIPS), not the county-year, because
within-county observations are correlated across years.
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
import polars as pl
import joblib
 
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass

@dataclass
class BootstrapResult:
    """Container for bootstrap inference results."""
    # Point estimates (from the full sample)
    point_estimates: pd.DataFrame
 
    # Bootstrap distribution
    boot_estimates: pd.DataFrame  # one row per (replicate, year)
 
    # Summary with confidence intervals
    summary: pd.DataFrame
 
    n_boot: int
    alpha: float
    reduction: float
 
 
def bootstrap_counterfactual_inference(
    df_raw: pl.DataFrame,
    df_enhanced: pl.DataFrame,
    model_template,
    feature_cols: list[str],
    reduction: float,
    build_features_fn: Callable,
    build_counterfactual_fn: Callable,
    *,
    target_col: str = "mortality_rate",
    id_col: str = "FIPS",
    time_col: str = "year",
    n_boot: int = 20,
    n_splits: int = 5,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> BootstrapResult:
    """
    Bootstrap confidence intervals for the counterfactual effect of
    reducing dispensing by `reduction` proportion.
 
    At each replicate:
      1. Resample counties (FIPS) with replacement
      2. Build enhanced features on the resampled panel
      3. Train year-by-year models with K-fold CV
      4. Build counterfactual panel (reduced RX) for same counties
      5. Predict baseline and counterfactual mortality
      6. Compute the effect: mean(baseline_pred) - mean(cf_pred) per year
 
    Parameters
    ----------
    df_raw : pl.DataFrame
        Original raw panel from the dataloader.
    df_enhanced : pl.DataFrame
        Enhanced panel (for computing point estimates on full sample).
    model_template : sklearn-compatible estimator
        Will be cloned for each fold/replicate.
    feature_cols : list[str]
        Feature columns to use.
    reduction : float
        Proportional RX reduction (e.g., 0.15 for 15%).
    build_features_fn : callable
        Function to build enhanced features from raw panel.
        Signature: (df_raw) -> pl.DataFrame
    build_counterfactual_fn : callable
        Function to build counterfactual panel.
        Signature: (df_raw, reduction=r) -> pl.DataFrame
    target_col, id_col, time_col : str
        Column names.
    n_boot : int
        Number of bootstrap replicates.
    n_splits : int
        Number of CV folds.
    alpha : float
        Significance level for CI (0.05 = 95% CI).
    seed : int
        Random seed.
    verbose : bool
        Print progress.
 
    Returns
    -------
    BootstrapResult
        Point estimates, bootstrap distribution, and summary with CIs.
    """
    rng = np.random.RandomState(seed)
 
    # ── Step 0: Compute point estimates from full sample ─────────
    if verbose:
        print("Computing point estimates on full sample...")
    point_estimates = _run_single_pipeline(
        df_raw=df_raw,
        df_enhanced=df_enhanced,
        model_template=model_template,
        feature_cols=feature_cols,
        reduction=reduction,
        build_counterfactual_fn=build_counterfactual_fn,
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        n_splits=n_splits,
    )
    if verbose:
        print("Point estimates computed.")
        print(point_estimates.to_string(index=False))
 
    # ── Get all unique FIPS for resampling ───────────────────────
    all_fips = df_raw[id_col].unique().to_list()
    n_counties = len(all_fips)
 
    # ── Bootstrap loop ───────────────────────────────────────────
    boot_records = []
 
    for b in range(n_boot):
        if verbose and (b + 1) % 10 == 0:
            print(f"  Bootstrap replicate {b + 1}/{n_boot}")
 
        # Resample counties with replacement
        boot_fips = rng.choice(all_fips, size=n_counties, replace=True)
        boot_fips_list = boot_fips.tolist()
 
        # Create resampled raw panel
        # (counties appearing multiple times get duplicated rows
        #  with unique FIPS suffixes to avoid deduplication)
        boot_raw = _resample_panel(df_raw, boot_fips_list, id_col, time_col)
 
        # Build features on resampled panel
        try:
            boot_enhanced = build_features_fn(boot_raw)
        except Exception as e:
            if verbose:
                print(f"    Replicate {b + 1} feature build failed: {e}")
            continue
 
        # Run pipeline on resampled data
        try:
            boot_est = _run_single_pipeline(
                df_raw=boot_raw,
                df_enhanced=boot_enhanced,
                model_template=model_template,
                feature_cols=feature_cols,
                reduction=reduction,
                build_counterfactual_fn=build_counterfactual_fn,
                target_col=target_col,
                id_col=id_col,
                time_col=time_col,
                n_splits=n_splits,
            )
            boot_est["Replicate"] = b
            boot_records.append(boot_est)
        except Exception as e:
            if verbose:
                print(f"    Replicate {b + 1} pipeline failed: {e}")
            continue
 
    boot_df = pd.concat(boot_records, ignore_index=True)
 
    # ── Compute confidence intervals ─────────────────────────────
    summary = _compute_ci(point_estimates, boot_df, alpha)
    summary["Reduction"] = reduction
 
    if verbose:
        print(f"\nBootstrap summary ({n_boot} replicates, {1 - alpha:.0%} CI):")
        print(summary.to_string(index=False))
 
    return BootstrapResult(
        point_estimates=point_estimates,
        boot_estimates=boot_df,
        summary=summary,
        n_boot=n_boot,
        alpha=alpha,
        reduction=reduction,
    )
 
 
# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────
def _run_single_pipeline(
    df_raw: pl.DataFrame,
    df_enhanced: pl.DataFrame,
    model_template,
    feature_cols: list[str],
    reduction: float,
    build_counterfactual_fn: Callable,
    target_col: str,
    id_col: str,
    time_col: str,
    n_splits: int,
) -> pd.DataFrame:
    """
    Run the full G-computation pipeline on a single dataset and return
    per-year effect estimates.
 
    Returns DataFrame with columns:
      Year, MeanPred_Baseline, MeanPred_CF, Effect, PctEffect
    """
    years = sorted(df_enhanced[time_col].unique().to_list())
    start_year, end_year = min(years), max(years)
 
    # Build counterfactual panel from raw
    df_cf = build_counterfactual_fn(df_raw, reduction=reduction)
 
    year_results = []
 
    for year in range(start_year, end_year):
        pred_year = year + 1
 
        # Get feature-year and target-year data
        df_train = df_enhanced.filter(pl.col(time_col) == year)
        df_target = df_enhanced.filter(pl.col(time_col) == pred_year)
        df_cf_train = df_cf.filter(pl.col(time_col) == year)
 
        if df_train.is_empty() or df_target.is_empty():
            continue
 
        # Align counties across actual and counterfactual
        fips_train = set(df_train[id_col].to_list())
        fips_target = set(df_target[id_col].to_list())
        fips_cf = set(df_cf_train[id_col].to_list())
        common = sorted(fips_train & fips_target & fips_cf)
 
        if len(common) == 0:
            continue
 
        df_train = df_train.filter(pl.col(id_col).is_in(common)).sort(id_col)
        df_target = df_target.filter(pl.col(id_col).is_in(common)).sort(id_col)
        df_cf_train = df_cf_train.filter(pl.col(id_col).is_in(common)).sort(id_col)
 
        X = df_train.select(feature_cols).to_pandas()
        X_cf = df_cf_train.select(feature_cols).to_pandas()
        y = df_target[target_col].to_numpy()
 
        # K-fold: collect baseline and CF predictions
        baseline_preds = np.zeros(len(common))
        cf_preds = np.zeros(len(common))
 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
 
        for train_idx, test_idx in kf.split(X):
            fold_model = clone(model_template)
            fold_model.fit(X.iloc[train_idx], y[train_idx])
 
            baseline_preds[test_idx] = fold_model.predict(X.iloc[test_idx])
            cf_preds[test_idx] = fold_model.predict(X_cf.iloc[test_idx])
 
        mean_base = baseline_preds.mean()
        mean_cf = cf_preds.mean()
        effect = mean_base - mean_cf
        pct_effect = (effect / mean_base * 100) if mean_base != 0 else 0
 
        year_results.append({
            "Year": pred_year,
            "MeanPred_Baseline": mean_base,
            "MeanPred_CF": mean_cf,
            "Effect": effect,
            "PctEffect": pct_effect,
        })
 
    return pd.DataFrame(year_results)
 
 
def _resample_panel(
    df: pl.DataFrame,
    boot_fips: list,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """
    Create a bootstrap-resampled panel by drawing counties with
    replacement.  Counties drawn multiple times get unique synthetic
    FIPS codes to maintain panel structure.
    """
    frames = []
    fips_counter = {}
 
    for fips in boot_fips:
        # Track how many times this FIPS has been drawn
        fips_counter[fips] = fips_counter.get(fips, 0) + 1
        count = fips_counter[fips]
 
        county_data = df.filter(pl.col(id_col) == fips)
 
        if count > 1:
            # Create synthetic FIPS for duplicate draws
            synthetic_fips = f"{fips}_b{count}"
            county_data = county_data.with_columns(
                pl.lit(synthetic_fips).alias(id_col)
            )
 
        frames.append(county_data)
 
    return pl.concat(frames).sort([id_col, time_col])
 
 
def _compute_ci(
    point_estimates: pd.DataFrame,
    boot_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    Compute percentile bootstrap confidence intervals.
    """
    lo = alpha / 2
    hi = 1 - alpha / 2
 
    summary_rows = []
    for _, row in point_estimates.iterrows():
        year = row["Year"]
        boot_year = boot_df[boot_df["Year"] == year]
 
        if boot_year.empty:
            continue
 
        summary_rows.append({
            "Year": year,
            "Effect_Point": row["Effect"],
            "Effect_CI_Lower": boot_year["Effect"].quantile(lo),
            "Effect_CI_Upper": boot_year["Effect"].quantile(hi),
            "PctEffect_Point": row["PctEffect"],
            "PctEffect_CI_Lower": boot_year["PctEffect"].quantile(lo),
            "PctEffect_CI_Upper": boot_year["PctEffect"].quantile(hi),
            "MeanPred_Baseline": row["MeanPred_Baseline"],
            "MeanPred_CF": row["MeanPred_CF"],
            "N_Boot_Successful": len(boot_year),
        })
 
    return pd.DataFrame(summary_rows)
