"""
Counterfactual prediction module for the G-computation pipeline.
 
Uses the fold-specific models saved during baseline training to
generate counterfactual predictions on modified data, preserving
out-of-sample integrity.
"""

from __future__ import annotations

import joblib
import numpy as np
import polars as pl

from pathlib import Path



def predict_counterfactual_cv_polars(
    df_modified: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    model_dir: Path,
    feature_cols: list[str],
    target_col: str = "mortality_rate",
) -> pl.DataFrame:
    """
    Produce counterfactual predictions that preserve OOS integrity:
    each county-year prediction uses the SAME fold model that was used
    for its baseline (out-of-sample) prediction.
 
    Parameters
    ----------
    df_modified : pl.DataFrame
        Enhanced counterfactual panel from build_counterfactual_panel()
        or build_counterfactual_panel_tiered().  Must contain FIPS,
        year, target_col, and all feature_cols.
    baseline_predictions : pl.DataFrame
        Baseline predictions from initial training.  Used to determine
        which counties belong to which fold for each year.
        Must contain columns: FIPS, Year, Fold.
    model_dir : Path
        Directory containing saved fold models with naming convention:
        model_year_{target_year}_fold{fold}.pkl
    feature_cols : list[str]
        Same feature columns used during training.
    target_col : str
        Name of the outcome variable in df_modified.
 
    Returns
    -------
    pl.DataFrame
        Counterfactual predictions with columns:
        FIPS, Year, True, Predicted, Fold, AbsError
    """
    preds_all = []
 
    # ── Validate feature columns exist in modified data ──────────
    missing_cols = set(feature_cols) - set(df_modified.columns)
    if missing_cols:
        raise KeyError(
            f"Feature columns not found in df_modified: {sorted(missing_cols)}. "
            f"Did you pass the enhanced counterfactual panel?"
        )
 
    base_pdf = baseline_predictions.to_pandas()
    years = sorted(base_pdf["Year"].unique())
 
    for target_year in years:
        feature_year = target_year - 1
 
        df_year = base_pdf[base_pdf["Year"] == target_year]
        folds = sorted(df_year["Fold"].unique())
 
        for fold in folds:
            fold_model_fp = model_dir / f"model_year_{target_year}_fold{fold}.pkl"
            if not fold_model_fp.exists():
                raise FileNotFoundError(
                    f"Missing model: {fold_model_fp}"
                )
 
            fold_model = joblib.load(fold_model_fp)
 
            # Counties belonging to this fold
            fold_fips = df_year[df_year["Fold"] == fold]["FIPS"].tolist()
            if len(fold_fips) == 0:
                continue
 
            # ── Extract features from the feature year ───────────
            df_X = (
                df_modified.filter(
                    (pl.col("year") == feature_year)
                    & (pl.col("FIPS").is_in(fold_fips))
                )
                .select(["FIPS"] + feature_cols)
                .sort("FIPS")
            )
 
            if df_X.height != len(fold_fips):
                missing = set(fold_fips) - set(df_X["FIPS"].to_list())
                raise ValueError(
                    f"Missing {len(missing)} counties in df_modified for "
                    f"feature_year={feature_year}, target_year={target_year}, "
                    f"Fold={fold}. Examples: {list(missing)[:5]}"
                )
 
            X_cf = df_X.select(feature_cols).to_pandas()
 
            # ── Extract true outcomes from the target year ───────
            df_Y = (
                df_modified.filter(
                    (pl.col("year") == target_year)
                    & (pl.col("FIPS").is_in(fold_fips))
                )
                .select(["FIPS", target_col])
                .sort("FIPS")
            )
 
            if df_Y.height != len(fold_fips):
                missing = set(fold_fips) - set(df_Y["FIPS"].to_list())
                raise ValueError(
                    f"Missing {len(missing)} counties in df_modified for "
                    f"target_year={target_year}, Fold={fold}. "
                    f"Examples: {list(missing)[:5]}"
                )
 
            y_true = df_Y.select(target_col).to_numpy().ravel()
            y_pred = fold_model.predict(X_cf)
            abs_err = np.abs(y_true - y_pred)
 
            preds_all.append(
                pl.DataFrame({
                    "FIPS": df_X["FIPS"],
                    "Year": [target_year] * len(y_pred),
                    "True": y_true,
                    "Predicted": y_pred.tolist(),
                    "Fold": [fold] * len(y_pred),
                    "AbsError": abs_err.tolist(),
                })
            )
 
    counterfact_df = pl.concat(preds_all)
 
    # ── Final completeness check ─────────────────────────────────
    baseline_keys = set(
        zip(
            base_pdf["FIPS"].astype(str),
            base_pdf["Year"].astype(int),
        )
    )
    counterfact_keys = set(
        zip(
            counterfact_df["FIPS"].to_list(),
            counterfact_df["Year"].to_list(),
        )
    )
 
    if baseline_keys != counterfact_keys:
        missing = baseline_keys - counterfact_keys
        extra = counterfact_keys - baseline_keys
        msg = f"Baseline and counterfactual keys do not match."
        if missing:
            msg += f" Missing {len(missing)} county-years."
        if extra:
            msg += f" Extra {len(extra)} county-years."
        raise ValueError(msg)
 
    return counterfact_df

