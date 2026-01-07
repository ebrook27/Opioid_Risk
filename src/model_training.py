### 11/06/25, EB: Cleaning up the codebase and moving to a new repo. This file will contain general utility functions for all models. Dataloader, model training loop, some evaluation functions, etc.

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



SVI_DATA = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment']


def yearly_mortality_prediction_polars(
    df: pl.DataFrame,
    model,
    feature_cols: list[str] | None = None,
    n_splits: int = 5,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float], Path]:
    """
    Train and evaluate a general regression model (sklearn-style) to predict
    next-year mortality using previous-year features, keeping Polars operations
    until just before model fitting.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format county-year dataset (SVI, rx_rate, mortality_rate, etc.)
    model : sklearn-like model
        Must implement .fit(X, y) and .predict(X)
    feature_cols : list[str] or None
        If None, automatically uses all numeric columns except FIPS, year, and known non-feature columns.
    n_splits : int
        Number of K-fold splits for within-year cross-validation.
    save_path : str or None
        If provided, saves results as CSVs in this folder.

    Returns
    -------
    metrics_df, feature_importance_df, predictions_df, all_errors
    """

    target_col = "mortality_rate"
    metrics_all_years = []
    feature_importance_all = []
    all_predictions = []
    all_errors = []

    # Drop rows missing the target
    df = df.drop_nulls(subset=[target_col])

    # --- Infer available year range dynamically ---
    years = df["year"].unique().to_list()
    start_year, end_year = min(years), max(years)

    print(f"📅 Year range detected: {start_year}–{end_year}")

    # --- Infer feature columns if not provided ---
    if feature_cols is None:
        exclude = {"FIPS", "year", target_col, "urbanicity_class"}
        feature_cols = [c for c in df.columns if c not in exclude]

    # --- Yearly training loop ---
    for year in range(start_year, end_year):
        print(f"\n🔁 Training on {year} → predicting {year + 1}")

        df_train = df.filter(pl.col("year") == year)
        df_target = df.filter(pl.col("year") == year + 1)

        if df_train.is_empty() or df_target.is_empty():
            print(f"⚠️ Skipping {year}: missing data.")
            continue

        # Align counties across years
        common_fips = set(df_train["FIPS"]) & set(df_target["FIPS"])
        df_train = df_train.filter(pl.col("FIPS").is_in(common_fips))
        df_target = df_target.filter(pl.col("FIPS").is_in(common_fips))

        X_train_pl = df_train.select(feature_cols)
        y_train_pl = df_target.select(target_col)
        fips_target = df_target.select("FIPS")

        # Convert just before fitting
        X = X_train_pl.to_pandas()
        y = y_train_pl.to_numpy().ravel()
        fips = fips_target.to_pandas().values.ravel()

        # --- KFold CV ---
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fips_test = fips[test_idx]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)

            abs_errors = np.abs(y_test - y_pred)
            all_errors.extend(abs_errors.tolist())

            fold_df = pd.DataFrame({
                "FIPS": fips_test,
                "Year": year + 1,
                "True": y_test,
                "Predicted": y_pred,
                "Fold": fold_idx + 1,
                "AbsError": abs_errors,
            })
            all_predictions.append(fold_df)

            fold_metrics.append({
                "fold": fold_idx + 1,
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            })

            if hasattr(fold_model, "feature_importances_"):
                fi = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": fold_model.feature_importances_,
                    "Year": year,
                    "Fold": fold_idx + 1,
                })
                feature_importance_all.append(fi)

        fold_df = pd.DataFrame(fold_metrics).drop(columns="fold")
        year_metrics = fold_df.mean().to_dict()
        year_metrics["Year"] = year + 1
        metrics_all_years.append(year_metrics)

    # --- Combine results ---
    metrics_df = pd.DataFrame(metrics_all_years)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_importance_df = (
        pd.concat(feature_importance_all, ignore_index=True)
        if feature_importance_all else pd.DataFrame()
    )

    # --- Optional save ---
    save_dir = None
    
    if save_path:
        # We create a timestamped filename to avoid overwriting.
        model_name = model.__class__.__name__
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        save_dir = Path(save_path) / model_name.lower() / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_fp = save_dir / "metrics.csv"
        feats_fp = save_dir / "feature_importance.csv"
        preds_fp = save_dir / "predictions.csv"

        metrics_df.to_csv(metrics_fp, index=False)
        predictions_df.to_csv(preds_fp, index=False)
        if not feature_importance_df.empty:
            feature_importance_df.to_csv(feats_fp, index=False)

        print(f"✅ Saved outputs to {save_path}")

    return metrics_df, feature_importance_df, predictions_df, all_errors, save_dir


import joblib
import json
from typing import Tuple, List, Optional

def national_counterfact_initial_training(
    df: pl.DataFrame,
    model,
    feature_cols: list[str] | None = None,
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
        <save_path>/<model_name>/<timestamp>/models/
    """

    target_col = "mortality_rate"
    metrics_all_years = []
    feature_importance_all = []
    all_predictions = []
    all_errors = []

    df = df.drop_nulls(subset=[target_col])

    years = df["year"].unique().to_list()
    start_year, end_year = min(years), max(years)

    # Infer features
    if feature_cols is None:
        exclude = {"FIPS", "year", target_col, "urbanicity_class"}
        feature_cols = [c for c in df.columns if c not in exclude]

    # Prepare save directory
    save_dir = None
    models_dir = None

    if save_path:
        model_name = model.__class__.__name__
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path(save_path) / model_name.lower() / timestamp
        models_dir = save_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save feature list
        with (save_dir / "feature_cols.json").open("w") as f:
            json.dump(feature_cols, f, indent=2)

    # YEARLY CV LOOP
    for year in range(start_year, end_year):

        df_train = df.filter(pl.col("year") == year)
        df_target = df.filter(pl.col("year") == year + 1)

        if df_train.is_empty() or df_target.is_empty():
            print(f"⚠️ Skipping {year}: missing data.")
            continue

        # Align counties
        # common_fips = set(df_train["FIPS"]) & set(df_target["FIPS"])
        common_fips = (
            set(df_train.select("FIPS").to_series().to_list())
            & set(df_target.select("FIPS").to_series().to_list())
        )
        df_train = df_train.filter(pl.col("FIPS").is_in(common_fips))
        df_target = df_target.filter(pl.col("FIPS").is_in(common_fips))

        X = df_train.select(feature_cols).to_pandas()
        y   = df_target.select(target_col).to_numpy().ravel()
        fips_target = df_target["FIPS"].to_list()

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        metrics_folds = []

        # FOLD LOOP
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fips_test = [fips_target[i] for i in test_idx]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            # SAVE MODEL LATER COUNTERFACTUAL PREDICTIONS
            if models_dir:
                model_fp = models_dir / f"model_year_{year+1}_fold{fold_idx+1}.pkl"
                joblib.dump(fold_model, model_fp)

            # Predict + store OOS predictions
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

            # Metrics
            metrics_folds.append({
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
                "Fold": fold_idx + 1,
                "Year": year + 1,
            })

            # Feature importance?
            if hasattr(fold_model, "feature_importances_"):
                fi = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": fold_model.feature_importances_,
                    "Fold": fold_idx + 1,
                    "Year": year + 1,
                })
                feature_importance_all.append(fi)

        # Aggregate metrics
        metrics_df_year = pd.DataFrame(metrics_folds).drop(columns="Fold").mean().to_dict()
        metrics_df_year["Year"] = year + 1
        metrics_all_years.append(metrics_df_year)

    # Combine outputs
    metrics_df = pd.DataFrame(metrics_all_years)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    fold_assignments_df = predictions_df[["FIPS", "Year", "Fold"]].drop_duplicates()
    feature_importance_df = (
        pd.concat(feature_importance_all, ignore_index=True)
        if feature_importance_all else pd.DataFrame()
    )

    # Save metrics/predictions
    if save_dir:
        metrics_df.to_csv(save_dir / "metrics.csv", index=False)
        predictions_df.to_csv(save_dir / "predictions.csv", index=False)
        fold_assignments_df.to_csv(save_dir / "fold_assignments.csv", index=False)
        if not feature_importance_df.empty:
            feature_importance_df.to_csv(save_dir / "feature_importance.csv", index=False)

    return (metrics_df, feature_importance_df, predictions_df, all_errors, save_dir)
