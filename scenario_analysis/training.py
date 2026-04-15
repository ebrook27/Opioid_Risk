"""
Training module for the G-computation counterfactual pipeline.
 
Produces out-of-sample predictions using year-by-year K-fold CV,
saves fold-specific models for later counterfactual prediction,
and is compatible with the enhanced feature set from
feature_engineering.build_opioid_panel_features.
"""
 
from __future__ import annotations
 
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
 
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
 
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 



# ─────────────────────────────────────────────────────────────────
# New model training and prediction function, that accomodates new features
# Additional new counterfactual prediction function as well.
# ─────────────────────────────────────────────────────────────────
def national_counterfact_initial_training(
    df: pl.DataFrame,
    model,
    feature_cols: list[str] | None = None,
    target_col: str = "mortality_rate",
    n_splits: int = 5,
    save_path: str | None = None,
) -> Tuple[
    pd.DataFrame,        # metrics_df
    pd.DataFrame,        # feature_importance_df
    pd.DataFrame,        # predictions_df
    List[float],         # all_errors
    Path | None          # save_dir
]:
    """
    TRAINING FUNCTION: Produces out-of-sample predictions using K-fold CV
    AND saves a fold-specific model for each prediction year inside:
        <save_path>/models/
 
    This version is compatible with the enhanced panel produced by
    build_opioid_panel_features().  The key assumption is:
      - Row for (FIPS, year=t) contains features from year t and earlier
        (lags, rolling means, changes are already computed).
      - The target for that row is mortality_rate at year t+1.
      - So we train on year-t rows and pull targets from year-(t+1) rows.
 
    Parameters
    ----------
    df : pl.DataFrame
        Enhanced panel from build_opioid_panel_features() — must contain
        id_col ("FIPS"), time_col ("year"), target_col, and all features.
    model : sklearn-compatible estimator
        Will be cloned for each fold.
    feature_cols : list[str] | None
        Columns to use as features.  If None, inferred by excluding
        FIPS, year, target_col, and urbanicity_class.
        Recommended: use get_feature_columns() from feature_engineering.
    target_col : str
        Name of the outcome variable.
    n_splits : int
        Number of CV folds.
    save_path : str | None
        If provided, models and artifacts are saved here.
 
    Returns
    -------
    metrics_df : pd.DataFrame
        Year-level aggregated metrics.
    feature_importance_df : pd.DataFrame
        Feature importances (if the model supports them).
    predictions_df : pd.DataFrame
        All out-of-sample predictions with FIPS, Year, True, Predicted,
        Fold, AbsError.
    all_errors : List[float]
        Flat list of all absolute errors.
    save_dir : Path | None
        Where artifacts were saved.
    """
    metrics_all_years = []
    feature_importance_all = []
    all_predictions = []
    all_errors = []
 
    df = df.drop_nulls(subset=[target_col])
 
    years = sorted(df["year"].unique().to_list())
    start_year, end_year = min(years), max(years)
 
    # ── Infer features if not provided ───────────────────────────
    if feature_cols is None:
        exclude = {"FIPS", "year", target_col, "urbanicity_class"}
        feature_cols = [c for c in df.columns if c not in exclude]
 
    # ── Validate that all feature columns exist ──────────────────
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Feature columns not found in dataframe: {sorted(missing_cols)}"
        )
 
    # ── Prepare save directory ───────────────────────────────────
    save_dir = None
    models_dir = None
 
    if save_path:
        # save_path is expected to be the run-level baseline directory:
        # .../national_counterfactual/<model>/<run_timestamp>/national_baseline/
        save_dir = Path(save_path)
        models_dir = save_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
 
        # Save feature list for reproducibility
        with (save_dir / "feature_cols.json").open("w") as f:
            json.dump(feature_cols, f, indent=2)
 
    # ── YEARLY CV LOOP ───────────────────────────────────────────
    for year in range(start_year, end_year):
 
        df_train = df.filter(pl.col("year") == year)
        df_target = df.filter(pl.col("year") == year + 1)
 
        if df_train.is_empty() or df_target.is_empty():
            print(f"Skipping {year}: missing data for year {year} or {year + 1}.")
            continue
 
        # ── Align counties present in both years ─────────────────
        # After feature engineering, some county-years may have been
        # dropped (incomplete history).  We need counties present in
        # BOTH the feature year and the target year.
        fips_train = set(df_train.select("FIPS").to_series().to_list())
        fips_target_set = set(df_target.select("FIPS").to_series().to_list())
        common_fips = fips_train & fips_target_set
 
        if len(common_fips) == 0:
            print(f"Skipping {year}: no overlapping counties.")
            continue
 
        df_train = df_train.filter(pl.col("FIPS").is_in(common_fips))
        df_target = df_target.filter(pl.col("FIPS").is_in(common_fips))
 
        # Sort both identically so row alignment is correct
        df_train = df_train.sort("FIPS")
        df_target = df_target.sort("FIPS")
 
        X = df_train.select(feature_cols).to_pandas()
        y = df_target.select(target_col).to_numpy().ravel()
        fips_list = df_target["FIPS"].to_list()
 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
 
        metrics_folds = []
 
        # ── FOLD LOOP ────────────────────────────────────────────
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
 
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fips_test = [fips_list[i] for i in test_idx]
 
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
 
            # Save model for later counterfactual predictions
            if models_dir:
                model_fp = models_dir / f"model_year_{year + 1}_fold{fold_idx + 1}.pkl"
                joblib.dump(fold_model, model_fp)
 
            # Predict + store out-of-sample predictions
            y_pred = fold_model.predict(X_test)
            abs_err = np.abs(y_test - y_pred)
 
            all_errors.extend(abs_err.tolist())
 
            fold_df = pd.DataFrame({
                "FIPS": fips_test,
                "Year": year + 1,
                "True": y_test,
                "Predicted": y_pred,
                "Fold": fold_idx + 1,
                "AbsError": abs_err,
            })
            all_predictions.append(fold_df)
 
            # Fold metrics
            metrics_folds.append({
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
                "Fold": fold_idx + 1,
                "Year": year + 1,
            })
 
            # Feature importance (tree-based models)
            if hasattr(fold_model, "feature_importances_"):
                fi = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": fold_model.feature_importances_,
                    "Fold": fold_idx + 1,
                    "Year": year + 1,
                })
                feature_importance_all.append(fi)
 
        # ── Aggregate year-level metrics ─────────────────────────
        metrics_df_year = (
            pd.DataFrame(metrics_folds)
            .drop(columns="Fold")
            .mean()
            .to_dict()
        )
        metrics_df_year["Year"] = year + 1
        metrics_all_years.append(metrics_df_year)
 
        n_counties = len(common_fips)
        avg_mae = metrics_df_year["MAE"]
        print(
            f"  Year {year}->{year + 1}: "
            f"{n_counties} counties, "
            f"MAE={avg_mae:.4f}"
        )
 
    # ── Combine all outputs ──────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_all_years)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    fold_assignments_df = predictions_df[["FIPS", "Year", "Fold"]].drop_duplicates()
    feature_importance_df = (
        pd.concat(feature_importance_all, ignore_index=True)
        if feature_importance_all else pd.DataFrame()
    )
 
    # ── Save artifacts ───────────────────────────────────────────
    if save_dir:
        metrics_df.to_csv(save_dir / "metrics.csv", index=False)
        predictions_df.to_csv(save_dir / "predictions.csv", index=False)
        fold_assignments_df.to_csv(save_dir / "fold_assignments.csv", index=False)
        if not feature_importance_df.empty:
            feature_importance_df.to_csv(
                save_dir / "feature_importance.csv", index=False
            )
 
    return (metrics_df, feature_importance_df, predictions_df, all_errors, save_dir)
