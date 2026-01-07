### 11/13/25, EB: Contains helper functions for the counterfactual analysis.

import pandas as pd
import numpy as np
import random
import polars as pl
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


def get_high_risk_counties(risk_scores: pl.DataFrame, 
                           top_frac: float = 0.05,
                           error_col: str = 'AbsError_Risk'
                           ) -> pl.DataFrame:
    """
    Select top `top_frac` highest_risk counties in the most recent year.
    Assumes risk_scores has columns: ['FIPS', 'Year', error_col, ...]
    """
    # latest_year = risk_scores["Year"].max()
    # df_latest = risk_scores[risk_scores["Year"] == latest_year].copy()
    
    # cutoff = df_latest[error_col].quantile(1 - top_frac)
    # high_risk = df_latest[df_latest[error_col] >= cutoff].copy()
    
    # print(f"Selected {len(high_risk)} high-risk counties "
    #       f"({top_frac*100:.1f}%) for year {latest_year}, (cutoff = {cutoff:.4f})."
    #       )
    
    ### 11/21/25, EB: Refactored to use polars for consistency, keeping above until I can confirm this all works.
    latest_year = risk_scores['Year'].max()
    df_latest = risk_scores.filter(pl.col('Year') == latest_year)
    cutoff = df_latest.select(pl.col(error_col).quantile(1 - top_frac)).item()
    high_risk = df_latest.filter(pl.col(error_col) >= cutoff)
    
    print(
        f"Selected {high_risk.height} high-risk counties "
        f"({top_frac*100:.1f}%) for year {latest_year}, (cutoff = {cutoff:.4f})."        
    )
    
    return high_risk


def pick_counterfactual_counties(
    high_risk: pl.DataFrame,
    n: int = 2,
    seed: int = 1738
) -> List[str]:
    """
    Pick `n` counties from the high-risk subset for counterfactual intervention.

    Returns
    -------
    List of FIPS strings.
    """
    # if len(high_risk) < n:
    #     raise ValueError(f"Not enough high-risk counties to pick {n} of them.")

    # random.seed(seed)
    # selected = random.sample(
    #     list(high_risk["FIPS"].astype(str).str.zfill(5)),
    #     n
    # )
    # print(f"🎯 Selected {n} target counties for intervention: {selected}")
    
    ### 11/21/25, EB: Refactored to use polars for consistency, keeping above until I can confirm this all works.
    if high_risk.height < n:
        raise ValueError(f"Not enough high-risk counties to pick {n} of them.")
    
    # random.seed(seed)
    fips_list = high_risk["FIPS"].cast(str).str.zfill(5).to_list()
    selected = random.sample(fips_list, n)
    
    print(f" Selected {n} target counteis for intervention: {selected}")
    
    return selected

def apply_rx_reduction(df: pl.DataFrame, 
                       target_fips: List[str], 
                       adjust: float
                       ) -> pl.DataFrame:
    """
    Reduce prescription rate (rx_rate) for high-risk counties by the given adjustment factor.
    
    Parameters
    ----------
    df : pl.DataFrame
        The full long-format dataset (from CountyDataLoader).
    target_fips: List[str]
        Randomly selected high-risk counties, list of FIPS strings.
    adjust : float
        Multiplicative factor (e.g., 0.9 → 10% reduction).

    Returns
    -------
    pl.DataFrame
        Modified dataframe with reduced rx_rate in those counties.
    
    """
    
    target_set = set(str(f).zfill(5) for f in target_fips)
    if "rx_rate" not in df.columns:
        raise KeyError("Column 'rx_rate' not found in the dataframe. Cannot apply RX reduction.")

    print(f'Applying rx_rate *= {adjust:.3f} for target counties: {target_set}')

    return df.with_columns(
        pl.when(pl.col("FIPS").is_in(target_set))
        .then(pl.col("rx_rate") * adjust)
        .otherwise(pl.col("rx_rate"))
        .alias("rx_rate")
    )

def compare_predictions_and_save(
        actual_df: pl.DataFrame,
        baseline_pred: pl.DataFrame,
        cf_pred: pl.DataFrame,
        target_counties: list[str],
        out_dir: Path
    ):
    """
    Compare baseline vs counterfactual predicted mortality for selected counties,
    save merged comparison CSV, and generate line plots.

    Parameters
    ----------
    actual_df : pd.DataFrame
        Must contain ['FIPS', 'year', 'mortality_rate'].

    baseline_pred : pd.DataFrame
        Must contain ['FIPS', 'Year', 'Predicted'].

    cf_pred : pd.DataFrame
        Must contain ['FIPS', 'Year', 'Predicted'].

    target_counties : list[str]
        FIPS codes of counties selected for counterfactual intervention.

    out_dir : Path
        Directory to save outputs.
    """
    # out_dir.mkdir(parents=True, exist_ok=True)
    
    # # Normalize key columns
    # actual_df = actual_df.rename(columns={"year": "Year"})
    # actual_df["FIPS"] = actual_df["FIPS"].astype(str).str.zfill(5)
    # baseline_pred["FIPS"] = baseline_pred["FIPS"].astype(str).str.zfill(5)
    # cf_pred["FIPS"] = cf_pred["FIPS"].astype(str).str.zfill(5)

    # # Merge all three frames
    # merged = (
    #     actual_df.merge(baseline_pred, on=["FIPS", "Year"], how="left", suffixes=("", "_baseline"))
    #              .merge(cf_pred, on=["FIPS", "Year"], how="left", suffixes=("", "_cf"))
    # )

    # # Save merged comparison results
    # csv_path = out_dir / "counterfactual_comparison.csv"
    # merged.to_csv(csv_path, index=False)

    # print(f"💾 Saved merged comparison data → {csv_path}")

    # # Plot figures using the new function
    # plot_counterfactual_comparison(
    #     actual_df=actual_df,
    #     orig_pred_df=baseline_pred,
    #     cf_pred_df=cf_pred,
    #     target_counties=target_counties,
    #     save_dir=out_dir
    # )
    
    ### 11/21/25, EB: Refactored to use polars for consistency, keeping above until I can confirm this all works.
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #Normalize key coulmns
    actual_df = actual_df.with_columns(
        pl.col("FIPS").cast(str).str.zfill(5),
        pl.col("year").alias("Year"),
    )
    
    baseline_pred = baseline_pred.with_columns(
        pl.col("FIPS").cast(str).str.zfill(5)
    )
    
    cf_pred = cf_pred.with_columns(
        pl.col("FIPS").cast(str).str.zfill(5)
    )
    
    # Merge all three frames
    merged = (
        actual_df
        .join(baseline_pred, on=["FIPS", "Year"], how="left", suffix="_baseline")
        .join(cf_pred, on=["FIPS", "Year"], how="left", suffix="_cf")
    )
    
    merged.to_pandas().to_csv(out_dir / "counterfactual_comparison.csv", index=False)
    print(f" Saved merged comparison data → {out_dir / 'counterfactual_comparison.csv'}")
    
    # Plots require pandas unfortunately
    plot_counterfactual_comparison(
        actual_df=actual_df.to_pandas(),
        orig_pred_df=baseline_pred.to_pandas(),
        cf_pred_df=cf_pred.to_pandas(),
        target_counties=target_counties,
        save_dir=out_dir,
    )
    
    

def plot_counterfactual_comparison(
    actual_df: pd.DataFrame,
    orig_pred_df: pd.DataFrame,
    cf_pred_df: pd.DataFrame,
    target_counties: list[str],
    save_dir: str | Path,
    dpi: int = 300,
):
    """
    Plot time-series mortality trajectories for selected counties:
        - Actual mortality rate
        - Original model predictions
        - Counterfactual model predictions (after RX reduction)

    Parameters
    ----------
    actual_df : pd.DataFrame
        Must contain columns ['FIPS', 'year', 'mortality_rate'].

    orig_pred_df : pd.DataFrame
        Must contain columns ['FIPS', 'Year', 'Predicted'].

    cf_pred_df : pd.DataFrame
        Same format as orig_pred_df.

    target_counties : list[str]
        List of county FIPS codes to plot.

    save_dir : str or Path
        Directory in which to save plots.

    dpi : int
        Resolution of saved figures.
    """

    ### 11/21/25, EB: Refactored all functions to use polars for consistency, matplotlib uses pandas, so I've just commented out a couple lines here.

    save_dir = Path(save_dir)
    # save_dir.mkdir(parents=True, exist_ok=True)

    ### 11/21/25, EB: These ones
    # # Normalize column names
    # actual_df = actual_df.rename(columns={"year": "Year"})
    # actual_df["FIPS"] = actual_df["FIPS"].astype(str).str.zfill(5)
    # orig_pred_df["FIPS"] = orig_pred_df["FIPS"].astype(str).str.zfill(5)
    # cf_pred_df["FIPS"] = cf_pred_df["FIPS"].astype(str).str.zfill(5)

    for fips in target_counties:
        print(f"📉 Generating counterfactual plot for county {fips}...")

        # Extract series for the county
        df_act = actual_df[actual_df["FIPS"] == fips]
        df_orig = orig_pred_df[orig_pred_df["FIPS"] == fips]
        df_cf = cf_pred_df[cf_pred_df["FIPS"] == fips]

        ### 11/21/25, EB: And this one
        # # Merge years into a single frame
        # years = sorted(df_act["Year"].unique())

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot actual mortality
        ax.plot(
            df_act["Year"], df_act["mortality_rate"],
            label="Actual Mortality",
            color="black",
            linewidth=2,
        )

        # Original predicted mortality
        ax.plot(
            df_orig["Year"], df_orig["Predicted"],
            label="Original Prediction",
            color="blue",
            linewidth=2,
        )

        # Counterfactual predicted mortality
        ax.plot(
            df_cf["Year"], df_cf["Predicted"],
            label="Counterfactual Prediction",
            color="red",
            linewidth=2,
            linestyle="--",
        )

        ax.set_title(f"Mortality Trajectories — County {fips}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mortality Rate (per 100k)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Save
        outpath = save_dir / f"counterfactual_{fips}.png"
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {outpath}")


# def apply_risk_based_rx_modifier(
#     df: pl.DataFrame,
#     risk_scores: pl.DataFrame,
#     rx_col: str = "rx_rate",
#     risk_col: str = "AbsError_Risk",
#     lambda_val: float = 0.9,
#     rule: str = "linear",
# ) -> pl.DataFrame:
#     """
#     Apply a nationwide RX modification to ALL counties based on risk scores.

#     Parameters
#     ----------
#     df : pl.DataFrame
#         Full long-format dataset, including rx_rate values.
#     risk_scores : pl.DataFrame
#         Must include ['FIPS', 'Year', risk_col].
#         We will use the *latest year* risk scores for modification.
#     rx_col : str
#         Name of prescription rate column (default 'rx_rate').
#     risk_col : str
#         Column in risk_scores used to construct the adjustment.
#     lambda_val : float
#         Intervention strength. Example:
#             lambda_val = 0.9  → max 10% reduction 
#             lambda_val = 0.8  → max 20% reduction
#     rule : str
#         Currently supports:
#             - "linear": scale reductions linearly by normalized risk
#         Future options may include "threshold", "nonlinear", etc.

#     Returns
#     -------
#     pl.DataFrame
#         Modified dataframe with new rx_rate values.
#     """

#     if rx_col not in df.columns:
#         raise KeyError(f"Prescription column '{rx_col}' not found in df.")
    
#     if not (0 < lambda_val <= 1):
#         raise ValueError("lambda_val must be in (0,1].")

#     # ------------------------------------------------------------
#     # 1. Extract risk from the most recent year only
#     # ------------------------------------------------------------
#     latest_year = risk_scores["Year"].max()
#     rs_latest = (
#         risk_scores
#         .filter(pl.col("Year") == latest_year)
#         .select(["FIPS", risk_col])
#     )

#     # Normalize risk scores into [0, 1]
#     rs_norm = rs_latest.with_columns(
#         ((pl.col(risk_col) - pl.col(risk_col).min()) /
#          (pl.col(risk_col).max() - pl.col(risk_col).min()))
#         .alias("risk_norm")
#     )

#     # ------------------------------------------------------------
#     # 2. Join normalized risk back onto full df
#     # ------------------------------------------------------------
#     df_joined = df.join(rs_norm, on="FIPS", how="left")

#     if df_joined["risk_norm"].null_count() > 0:
#         print("Warning: Some counties missing risk_norm (likely no risk score). Filling with 0.")
#         df_joined = df_joined.with_columns(pl.col("risk_norm").fill_null(0))

#     # ------------------------------------------------------------
#     # 3. Compute county-specific RX modifier
#     # ------------------------------------------------------------
#     if rule == "linear":
#         # Counties with higher risk get larger reductions.
#         # rx_new = rx_old * (1 - (risk_norm * (1 - lambda_val)))
#         df_modified = df_joined.with_columns(
#             (pl.col(rx_col) * (1 - pl.col("risk_norm") * (1 - lambda_val)))
#             .alias(rx_col)
#         )

#     else:
#         raise NotImplementedError(f"Unknown RX modification rule: {rule}")

#     return df_modified

def apply_risk_based_rx_modifier(
    df: pl.DataFrame,
    risk_scores: pl.DataFrame,
    rx_col: str = "rx_rate",
    risk_col: str = "AbsError_Risk",
    lambda_val: float = 0.9,
    rule: str = "linear",
    clip_quantiles: tuple[float, float] = (0.05, 0.95),
    # --- piecewise params ---
    s: float = 0.5,
    t: float = 0.90,
    alpha: float | None = None,
    beta: float | None = None,
    alpha_delta: float = 0.05,
    beta_delta: float = 0.10,
) -> pl.DataFrame:
    """
    Apply a nationwide RX modification to ALL counties based on risk scores.

    Interpretation:
    - Uses the most recent year of risk_scores to define a national policy.
    - Higher-risk counties receive larger RX reductions.
    - Counties at maximum risk receive rx_new = lambda_val * rx_old.
    - Counties at minimum risk receive no change.
    
    
    rule="linear":
        rx_new = rx_old * (1 - risk_norm * (1 - lambda_val))
        - risk_norm=0 => no change
        - risk_norm=1 => rx_new = lambda_val * rx_old

    rule="piecewise":
        Define tiers on risk_norm:
            High:  risk_norm >= t      => multiplier = lambda_val
            Mid:   s <= risk_norm < t  => multiplier = alpha
            Low:   risk_norm < s       => multiplier = beta
        where multipliers should satisfy: lambda_val <= alpha <= beta <= 1.

    Parameters
    ----------
    df : pl.DataFrame
        Full long-format dataset.
    risk_scores : pl.DataFrame
        Must include ['FIPS', 'Year', risk_col].
    rx_col : str
        Prescription rate column.
    risk_col : str
        Risk metric used to scale the intervention.
    lambda_val : float
        Maximum RX multiplier for highest-risk counties (0 < lambda_val ≤ 1).
    rule : str
        Currently only "linear".
    clip_quantiles : tuple
        Quantiles used to robustly normalize risk scores.

    s, t
        Thresholds on risk_norm in [0,1], with 0 <= s < t <= 1.
    alpha, beta
        Optional explicit multipliers for mid/low tiers.
        If None, they are derived as:
            alpha = min(1, lambda_val + alpha_delta)
            beta  = min(1, lambda_val + beta_delta)

    Returns
    -------
    pl.DataFrame
        Dataset with modified rx_col.
    """

    if rx_col not in df.columns:
        raise KeyError(f"Prescription column '{rx_col}' not found in df.")

    if not (0 < lambda_val <= 1):
        raise ValueError("lambda_val must be in (0, 1].")
    if rule not in {"linear", "piecewise"}:
        raise ValueError(f"Unknown rule '{rule}'. Supported: 'linear', 'piecewise'.")

    # ------------------------------------------------------------
    # 1. Extract most recent risk scores
    # ------------------------------------------------------------
    latest_year = risk_scores["Year"].max()
    rs_latest = risk_scores.filter(pl.col("Year") == latest_year)

    # ------------------------------------------------------------
    # 2. Robust risk normalization
    # ------------------------------------------------------------
    q_lo, q_hi = clip_quantiles

    rs_norm = rs_latest.with_columns(
        pl.col(risk_col)
        .clip(
            pl.col(risk_col).quantile(q_lo),
            pl.col(risk_col).quantile(q_hi),
        )
        .alias("risk_clipped")
    ).with_columns(
        (
            (pl.col("risk_clipped") - pl.col("risk_clipped").min()) /
            (pl.col("risk_clipped").max() - pl.col("risk_clipped").min())
        ).alias("risk_norm")
    ).select(["FIPS", "risk_norm"])

    # ------------------------------------------------------------
    # 3. Join onto full dataset (uniform across years)
    # ------------------------------------------------------------
    df_joined = df.join(rs_norm, on="FIPS", how="left")

    if df_joined["risk_norm"].null_count() > 0:
        df_joined = df_joined.with_columns(
            pl.col("risk_norm").fill_null(0)
        )

    # ------------------------------------------------------------
    # 4. Apply RX modification
    # ------------------------------------------------------------
    if rule == "linear":
        rx_new = (
            pl.col(rx_col) *
            (1 - pl.col("risk_norm") * (1 - lambda_val))
        )

    # else:
    #     raise NotImplementedError(f"Unknown RX modification rule: {rule}")
    else:  # piecewise
        # derive alpha/beta if not provided
        if alpha is None:
            alpha = min(1.0, lambda_val + alpha_delta)
        if beta is None:
            beta = min(1.0, lambda_val + beta_delta)

        # validate tiers
        if not (0 <= s < t <= 1):
            raise ValueError("Thresholds must satisfy 0 <= s < t <= 1.")
        for name, v in [("alpha", alpha), ("beta", beta)]:
            if not (0 < v <= 1):
                raise ValueError(f"{name} must be in (0, 1].")

        # enforce monotonic targeting: high risk => strongest reduction (smallest multiplier)
        if not (lambda_val <= alpha <= beta <= 1.0):
            raise ValueError(
                "Piecewise multipliers must satisfy lambda_val <= alpha <= beta <= 1. "
                f"Got lambda_val={lambda_val}, alpha={alpha}, beta={beta}."
            )

        multiplier = (
            pl.when(pl.col("risk_norm") >= t).then(pl.lit(lambda_val))
              .when(pl.col("risk_norm") >= s).then(pl.lit(alpha))
              .otherwise(pl.lit(beta))
        )

        rx_new = pl.col(rx_col) * multiplier


    # Guard against negatives
    df_modified = df_joined.with_columns(
        pl.when(rx_new < 0)
        .then(0)
        .otherwise(rx_new)
        .alias(rx_col)
    )

    return df_modified



def predict_counterfactual_cv_polars(
    df_modified: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    model_dir: Path,
    feature_cols: list[str],
) -> pl.DataFrame:
    """
    Produce counterfactual predictions that preserve OOS integrity:
    each county-year prediction uses the SAME fold model that was used
    for its baseline (out-of-sample) prediction.
    """

    preds_all = []

    # Convert baseline to pandas to simplify merging
    base_pdf = baseline_predictions.to_pandas()
    years = sorted(base_pdf["Year"].unique())

    for year in years:
        df_year = base_pdf[base_pdf["Year"] == year]
        folds = sorted(df_year["Fold"].unique())

        for fold in folds: # 5-fold CV, inferred automatically
            fold_model_fp = model_dir / f"model_year_{year}_fold{fold}.pkl"
            if not fold_model_fp.exists():
                raise FileNotFoundError(
                    f"Missing model for Year={year}, Fold={fold}, skipping."
                )
                
            fold_model = joblib.load(fold_model_fp)

            # Counties belonging to this fold
            fold_fips = df_year[df_year["Fold"] == fold]["FIPS"].tolist()

            if len(fold_fips) == 0:
                continue

            # Extract their features from df_modified
            df_fold = (
                df_modified.filter(
                    (pl.col("year") == year) & 
                    (pl.col("FIPS").is_in(fold_fips))
                )
            ).sort("FIPS")

            # if df_fold.is_empty():
            #     continue
            if df_fold.height != len(fold_fips):
                missing = set(fold_fips) - set(df_fold["FIPS"].to_list())
                raise ValueError(
                    f"Missing counties in df_modified for Year={year}, "
                    f"Fold={fold}: {list(missing)[:5]}"
                )

            X_cf = df_fold.select(feature_cols).to_pandas()
            y_true = df_fold.select("mortality_rate").to_pandas().values.ravel()

            y_pred = fold_model.predict(X_cf)
            abs_err = np.abs(y_true - y_pred)

            preds_all.append(pl.DataFrame({
                "FIPS": df_fold["FIPS"],
                "Year": df_fold["year"],
                "True": y_true,
                "Predicted": y_pred.tolist(),
                "Fold": [fold]*len(y_pred),
                "AbsError": abs_err.tolist(),
            }))

    counterfact_df = pl.concat(preds_all)

    # Final completeness check
    baseline_keys = set(
        zip(
            base_pdf["FIPS"].astype(str),
            base_pdf["Year"].astype(int)
        )
    )
    
    counterfact_keys = set(
        zip(
            counterfact_df["FIPS"].to_list(),
            counterfact_df["Year"].to_list()
        )
    )
    
    if baseline_keys != counterfact_keys:
        raise ValueError(
            f"Baseline and counterfactual prediction keys do not match. "
            f"Missing: {baseline_keys - counterfact_keys} county-years" 
        )


    return counterfact_df


def compute_national_prediction_comparison(
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compare baseline vs counterfactual predicted mortality at the national level.

    Computes yearly national mean predicted mortality rates before and after
    the counterfactual intervention.

    Parameters
    ----------
    baseline_predictions : pl.DataFrame
        Must contain ['Year', 'Predicted'].
    counterfact_predictions : pl.DataFrame
        Must contain ['Year', 'Predicted'].

    Returns
    -------
    pl.DataFrame
        Columns:
            - Year
            - MeanPred_Baseline
            - MeanPred_Counterfactual
            - AbsChange
            - PctChange
    """

    # --- Aggregate baseline ---
    base_nat = (
        baseline_predictions
        .group_by("Year")
        .agg(pl.col("Predicted").mean().alias("MeanPred_Baseline"))
    )

    # --- Aggregate counterfactual ---
    cf_nat = (
        counterfact_predictions
        .group_by("Year")
        .agg(pl.col("Predicted").mean().alias("MeanPred_Counterfactual"))
    )

    # --- Join and compute differences ---
    summary = (
        base_nat
        .join(cf_nat, on="Year", how="inner")
        .with_columns([
            (pl.col("MeanPred_Counterfactual") - pl.col("MeanPred_Baseline"))
                .alias("AbsChange"),
            (
                (pl.col("MeanPred_Counterfactual") - pl.col("MeanPred_Baseline"))
                / pl.col("MeanPred_Baseline")
            ).alias("PctChange"),
        ])
        .sort("Year")
    )

    return summary


def plot_national_prediction_comparison(
    summary_df: pl.DataFrame,
    df: pl.DataFrame,
    ylabel: str = "Mean Predicted Mortality Rate",
):
    """
    Plot national mean predicted mortality before vs after counterfactual.
    """

    obs_nat = (
        df
        .group_by("year")
        .agg(pl.mean("mortality_rate").alias("MeanObserved"))
        .sort("year")
    )

    pdf = summary_df.to_pandas()

    plt.figure(figsize=(8,5))
    plt.plot(
        obs_nat["year"], obs_nat["MeanObserved"], 
        label="Observed", color="black", linewidth=2
    )
    plt.plot(pdf["Year"], pdf["MeanPred_Baseline"], label="Baseline Pred")
    plt.plot(pdf["Year"], pdf["MeanPred_Counterfactual"], label="Counterfactual Pred")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title("National Mean Predicted Mortality Rate Comparison, Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.show()
