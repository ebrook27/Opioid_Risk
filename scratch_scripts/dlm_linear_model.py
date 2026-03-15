"""Distributed-lag linear model (DLNM) script.

Creates lagged predictors for unemployment, prescription dispensing (rx_rate),
and uninsured rates, uses SVI variables as linear terms, and fits an OLS
regression predicting year-ahead (`mortality_rate` at t+1) mortality.

Usage: run the script from the repository root. Example:
    python scripts/dlnm_linear_model.py --max-lag 3 --out-dir model_outputs/dlnm

The script uses `CountyDataLoader` from `src.data_processing` to build the panel.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm


# Ensure repo root is on sys.path so `src` imports work when running script
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_processing import CountyDataLoader


def build_lags_and_target(df: pd.DataFrame, predictors: list[str], group_col: str = "FIPS", max_lag: int = 3) -> pd.DataFrame:
    df = df.sort_values([group_col, "year"]).copy()
    for var in predictors:
        for lag in range(0, max_lag + 1):
            df[f"{var}_lag{lag}"] = df.groupby(group_col)[var].shift(lag)

    # Year-ahead target: mortality at t+1 for predictors observed at t
    df["mortality_ahead1"] = df.groupby(group_col)["mortality_rate"].shift(-1)
    return df


def fit_dlnm(df: pd.DataFrame, svi_vars: list[str], predictors: list[str], max_lag: int = 3, cluster_by: str | None = "FIPS"):
    lag_cols = []
    for var in predictors:
        for lag in range(0, max_lag + 1):
            lag_cols.append(f"{var}_lag{lag}")

    features = lag_cols + svi_vars
    df_model = df.dropna(subset=features + ["mortality_ahead1"]).copy()
    X = df_model[features]
    X = sm.add_constant(X)
    y = df_model["mortality_ahead1"]

    if cluster_by is not None and cluster_by in df_model.columns:
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_model[cluster_by]})
    else:
        res = sm.OLS(y, X).fit()
    return res, df_model


def main(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading panel data using CountyDataLoader...")
    loader = CountyDataLoader()
    panel_pl = loader.load()

    print("Converting to pandas DataFrame...")
    panel = panel_pl.to_pandas()

    predictors = ["unemp_rate", "rx_rate", "uninsured_rate"]
    max_lag = args.max_lag

    print(f"Creating lags up to {max_lag} for predictors: {predictors}")
    panel_with_lags = build_lags_and_target(panel, predictors=predictors, max_lag=max_lag)

    svi_vars = loader.svi_variables
    print(f"Using SVI linear terms: {svi_vars}")

    print("Fitting distributed-lag linear model (OLS) with clustered SEs by FIPS...")
    res, df_model = fit_dlnm(panel_with_lags, svi_vars=svi_vars, predictors=predictors, max_lag=max_lag, cluster_by="FIPS")

    print(res.summary())

    coef_path = out_dir / "dlnm_coefs.csv"
    print(f"Saving coefficients to {coef_path}")
    coefs = res.params.rename("coef").to_frame()
    coefs["std_err"] = res.bse
    coefs["t"] = res.tvalues
    coefs["pvalue"] = res.pvalues
    coefs.to_csv(coef_path)

    model_path = out_dir / "dlnm_results.pkl"
    print(f"Saving full results to {model_path}")
    with open(model_path, "wb") as fh:
        pickle.dump(res, fh)

    data_path = out_dir / "dlnm_model_data.parquet"
    print(f"Saving modeling dataset (trimmed) to {data_path}")
    df_model.to_parquet(data_path)

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Distributed-lag linear model for year-ahead mortality")
    p.add_argument("--max-lag", type=int, default=3, help="Maximum lag (in years) to include for predictors")
    p.add_argument("--out-dir", type=str, default="model_outputs/dlnm", help="Directory to save results")
    args = p.parse_args()
    main(args)
