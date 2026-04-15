### 11/13/25, EB: Contains helper functions for the counterfactual analysis.

import pandas as pd
import numpy as np
import random
import polars as pl
from typing import List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    
    rule="uniform":
    rx_new = lambda_val * rx_old  (applied to ALL counties, ALL years)
    
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
    
    if rule not in {"linear", "piecewise", "uniform"}:
        raise ValueError(f"Unknown rule '{rule}'. Supported: 'linear', 'piecewise', 'uniform'.")

    # ------------------------------------------------------------
    # Uniform RX Reduction (no risk_scores required)
    # ------------------------------------------------------------
    if rule == "uniform":
        rx_new = pl.col(rx_col) * pl.lit(lambda_val)

        return df.with_columns(
            pl.when(rx_new < 0).then(0).otherwise(rx_new).alias(rx_col)
        )
    
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


def apply_risk_based_rx_modifier_winsorize(
    df: pl.DataFrame,
    risk_scores: pl.DataFrame,
    rx_col: str = "rx_rate",
    risk_col: str = "AbsError_Risk",
    lambda_val: float = 0.9,
    rule: str = "linear",
    clip_quantiles: tuple[float, float] = (0.05, 0.95),
    winsorize: bool = True,
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
    
    rule="uniform":
    rx_new = lambda_val * rx_old  (applied to ALL counties, ALL years)
    
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
    
    if rule not in {"linear", "piecewise", "uniform"}:
        raise ValueError(f"Unknown rule '{rule}'. Supported: 'linear', 'piecewise', 'uniform'.")

    # ------------------------------------------------------------
    # Uniform RX Reduction (no risk_scores required)
    # ------------------------------------------------------------
    if rule == "uniform":
        rx_new = pl.col(rx_col) * pl.lit(lambda_val)

        return df.with_columns(
            pl.when(rx_new < 0).then(0).otherwise(rx_new).alias(rx_col)
        )
    else:
        if risk_scores is None:
            raise ValueError("risk_scores must be provided when rule is 'linear' or 'piecewise'.")

        latest_year = risk_scores["Year"].max()
        rs_latest = risk_scores.filter(pl.col("Year") == latest_year)

        # ------------------------------------------------------------
        # 2) Risk normalization (winsorized OR raw)
        # ------------------------------------------------------------
        if winsorize:
            q_lo, q_hi = clip_quantiles

            rs_work = rs_latest.with_columns(
                pl.col(risk_col)
                .clip(
                    pl.col(risk_col).quantile(q_lo),
                    pl.col(risk_col).quantile(q_hi),
                )
                .alias("risk_work")
            )
        else:
            rs_work = rs_latest.with_columns(
                pl.col(risk_col).alias("risk_work")
            )

        # Guard against degenerate range (all equal)
        rs_work = rs_work.with_columns(
            pl.col("risk_work").min().alias("_rmin"),
            pl.col("risk_work").max().alias("_rmax"),
        ).with_columns(
            pl.when(pl.col("_rmax") == pl.col("_rmin"))
            .then(pl.lit(0.0))
            .otherwise((pl.col("risk_work") - pl.col("_rmin")) / (pl.col("_rmax") - pl.col("_rmin")))
            .alias("risk_norm")
        )

        rs_norm = rs_work.select(["FIPS", "risk_norm"])
    
    # # ------------------------------------------------------------
    # # 1. Extract most recent risk scores
    # # ------------------------------------------------------------
    # latest_year = risk_scores["Year"].max()
    # rs_latest = risk_scores.filter(pl.col("Year") == latest_year)

    # # ------------------------------------------------------------
    # # 2. Robust risk normalization
    # # ------------------------------------------------------------
    # q_lo, q_hi = clip_quantiles

    # rs_norm = rs_latest.with_columns(
    #     pl.col(risk_col)
    #     .clip(
    #         pl.col(risk_col).quantile(q_lo),
    #         pl.col(risk_col).quantile(q_hi),
    #     )
    #     .alias("risk_clipped")
    # ).with_columns(
    #     (
    #         (pl.col("risk_clipped") - pl.col("risk_clipped").min()) /
    #         (pl.col("risk_clipped").max() - pl.col("risk_clipped").min())
    #     ).alias("risk_norm")
    # ).select(["FIPS", "risk_norm"])

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

    for target_year in years:
        feature_year = target_year - 1  # features from prior year
        
        df_year = base_pdf[base_pdf["Year"] == target_year]
        folds = sorted(df_year["Fold"].unique())

        for fold in folds: # 5-fold CV, inferred automatically
            fold_model_fp = model_dir / f"model_year_{target_year}_fold{fold}.pkl"
            if not fold_model_fp.exists():
                raise FileNotFoundError(
                    f"Missing model for Year={target_year}, Fold={fold}, skipping."
                )
                
            fold_model = joblib.load(fold_model_fp)

            # Counties belonging to this fold
            fold_fips = df_year[df_year["Fold"] == fold]["FIPS"].tolist()

            if len(fold_fips) == 0:
                continue

            # Extract their features from feature_year in df_modified
            df_X = (
                df_modified.filter(
                    (pl.col("year") == feature_year) & 
                    (pl.col("FIPS").is_in(fold_fips))
                )
                .select(["FIPS"] + feature_cols)
                .sort("FIPS")
            )

            # if df_fold.is_empty():
            #     continue
            if df_X.height != len(fold_fips):
                missing = set(fold_fips) - set(df_X["FIPS"].to_list())
                raise ValueError(
                    f"Missing counties in df_modified for Year={feature_year}, "
                    f"Target_year:{target_year}, Fold={fold}: {list(missing)[:5]}"
                )

            X_cf = df_X.select(feature_cols).to_pandas()
            # y_true = df_fold.select("mortality_rate").to_pandas().values.ravel()
            df_Y = (
                df_modified.filter(
                    (pl.col("year") == target_year) & 
                    (pl.col("FIPS").is_in(fold_fips))
                )
                .select(["FIPS", "mortality_rate"])
                .sort("FIPS")
            )
            
            if df_Y.height != len(fold_fips):
                missing = set(fold_fips) - set(df_Y["FIPS"].to_list())
                raise ValueError(
                    f"Missing counties in df_modified for Year={target_year}, "
                    f"Target_year:{target_year}, Fold={fold}: {list(missing)[:5]}"
                )
            y_true = df_Y.select("mortality_rate").to_numpy().ravel()

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

def plot_national_prediction_comparison_multi(
    summary_df: pl.DataFrame,
    df: pl.DataFrame,
    ylabel: str = "Mean Mortality Rate",
    title: str | None = None,
    legend_outside: bool = True,
    max_legend_entries: int | None = None,
):
    """
    Plot observed + baseline + multiple counterfactual scenario curves.

    Required columns in summary_df:
      - Year
      - MeanPred_Baseline
      - MeanPred_Counterfactual
      - Scenario
    Recommended (if available):
      - lambda_val
      - rx_rule
    """
    import matplotlib.pyplot as plt

    # Observed national mean
    obs_nat = (
        df.group_by("year")
          .agg(pl.mean("mortality_rate").alias("MeanObserved"))
          .sort("year")
          .rename({"year": "Year"})
    )

    plot_df = (
        summary_df
        .join(obs_nat, on="Year", how="left")
        .sort(["Year", "Scenario"])
        .to_pandas()
    )

    plt.figure(figsize=(10, 5))

    # Observed (plot once)
    obs = plot_df.drop_duplicates(subset=["Year"])[["Year", "MeanObserved"]]
    plt.plot(obs["Year"], obs["MeanObserved"], 'k', label="Observed", linestyle="--", linewidth=2)

    # Baseline (plot once)
    base = plot_df.drop_duplicates(subset=["Year"])[["Year", "MeanPred_Baseline"]]
    plt.plot(base["Year"], base["MeanPred_Baseline"], label="Baseline Prediction", linewidth=2)

    # Determine scenario order (prefer increasing lambda if present)
    if "lambda_val" in plot_df.columns:
        scen_order = (
            plot_df[["Scenario", "lambda_val"]]
            .drop_duplicates()
            .sort_values("lambda_val")["Scenario"]
            .tolist()
        )
    else:
        scen_order = sorted(plot_df["Scenario"].unique())

    # Optional: limit number of legend entries (still plot all lines if you want)
    if max_legend_entries is not None:
        scen_order_for_legend = set(scen_order[:max_legend_entries])
    else:
        scen_order_for_legend = None

    # Plot counterfactual scenarios
    for scenario in scen_order:
        g = plot_df[plot_df["Scenario"] == scenario]

        # concise label if lambda exists
        if "lambda_val" in g.columns:
            lam = float(g["lambda_val"].iloc[0])
            rx_rule = g["rx_rule"].iloc[0] if "rx_rule" in g.columns else None
            label = f"{rx_rule} λ={lam:.2f}" if rx_rule else f"λ={lam:.2f}"
        else:
            label = scenario

        # Optionally suppress legend entry to reduce clutter
        if scen_order_for_legend is not None and scenario not in scen_order_for_legend:
            label = "_nolegend_"

        plt.plot(g["Year"], g["MeanPred_Counterfactual"], label=label, linewidth=1.5)

    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title or "National Mean Mortality: Observed vs Baseline vs Counterfactual Scenarios")
    plt.tight_layout()

    if legend_outside:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        plt.tight_layout(rect=(0, 0, 0.80, 1))  # leave space for legend
    else:
        plt.legend(ncol=2, fontsize=9)

    plt.show()

def plot_single_scenario_cohort_means(
    *,
    df: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
    risk_scores: pl.DataFrame,
    scenario_label: str,
):
    """
    For risk-informed counterfactuals, we plot risk-stratified cohort means over time.
      - Observed national mean MR (df)
      - Baseline national mean prediction (baseline_predictions)
      - Counterfactual cohort mean predictions (counterfact_predictions), using fixed cohorts
        defined from final-year AbsError_Risk (min-max normalized): Low 50%, Mid 40%, High 10%.
    """

    # 1) Fixed cohort assignment using FINAL risk year
    final_year = risk_scores.select(pl.col("Year").max()).item()

    rs = (
        risk_scores
        .filter(pl.col("Year") == final_year)
        .group_by("FIPS")
        .agg(pl.mean("AbsError_Risk").cast(pl.Float64).alias("risk"))
    )

    rmin = rs.select(pl.col("risk").min()).item()
    rmax = rs.select(pl.col("risk").max()).item()
    denom = rmax - rmin

    if denom == 0:
        cohort_lut = rs.with_columns(pl.lit("Mid").alias("Cohort")).select(["FIPS", "Cohort"])
    else:
        cohort_lut = (
            rs.with_columns(((pl.col("risk") - rmin) / denom).alias("risk_norm"))
              .with_columns(
                  pl.when(pl.col("risk_norm") <= 0.50).then(pl.lit("Low"))
                   .when(pl.col("risk_norm") <= 0.90).then(pl.lit("Mid"))
                   .otherwise(pl.lit("High"))
                   .alias("Cohort")
              )
              .select(["FIPS", "Cohort"])
        )

    # 2) Series to plot
    obs_nat = (
        df.group_by("year")
          .agg(pl.mean("mortality_rate").alias("Observed"))
          .sort("year")
          .rename({"year": "Year"})
    )

    base_nat = (
        baseline_predictions.group_by("Year")
        .agg(pl.mean("Predicted").alias("Baseline"))
        .sort("Year")
    )

    cf_cohort = (
        counterfact_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(cohort_lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("Counterfactual"))
        .sort(["Year", "Cohort"])
    )

    # 3) Plot
    obs_pd = obs_nat.to_pandas()
    base_pd = base_nat.to_pandas()
    cf_pd = cf_cohort.to_pandas()

    plt.figure(figsize=(10, 5))
    plt.plot(obs_pd["Year"], obs_pd["Observed"], "k", linestyle="--", linewidth=2, label="Observed (National Mean)")
    plt.plot(base_pd["Year"], base_pd["Baseline"], linewidth=2, label="Baseline Prediction (National Mean)")

    for cohort in ["Low", "Mid", "High"]:
        g = cf_pd[cf_pd["Cohort"] == cohort]
        plt.plot(g["Year"], g["Counterfactual"], linewidth=2, label=f"Counterfactual (Cohort Mean) — {cohort}")

    plt.xlabel("Year")
    plt.ylabel("Mean Mortality Rate")
    plt.title(f"Observed vs Baseline vs Counterfactual Cohort Means — {scenario_label}")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout(rect=(0, 0, 0.80, 1))
    plt.show()

def plot_single_scenario_cohort_change(
    *,
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
    risk_scores: pl.DataFrame,
    scenario_label: str,
    use_percent: bool = False,   # False: absolute change, True: percent change
):
    """
    One figure, one scenario:
      - Cohort-wise change from baseline to counterfactual:
          Delta = MeanPred_CF - MeanPred_Base   (or percent if use_percent=True)

    Cohorts are fixed, defined from final-year AbsError_Risk (min-max normalized):
      Low: bottom 50%, Mid: 50–90%, High: top 10%.
    """

    # 1) Cohorts from FINAL risk year
    final_year = risk_scores.select(pl.col("Year").max()).item()

    rs = (
        risk_scores
        .filter(pl.col("Year") == final_year)
        .group_by("FIPS")
        .agg(pl.mean("AbsError_Risk").cast(pl.Float64).alias("risk"))
    )

    rmin = rs.select(pl.col("risk").min()).item()
    rmax = rs.select(pl.col("risk").max()).item()
    denom = rmax - rmin

    if denom == 0:
        cohort_lut = rs.with_columns(pl.lit("Mid").alias("Cohort")).select(["FIPS", "Cohort"])
    else:
        cohort_lut = (
            rs.with_columns(((pl.col("risk") - rmin) / denom).alias("risk_norm"))
              .with_columns(
                  pl.when(pl.col("risk_norm") <= 0.50).then(pl.lit("Low"))
                   .when(pl.col("risk_norm") <= 0.90).then(pl.lit("Mid"))
                   .otherwise(pl.lit("High"))
                   .alias("Cohort")
              )
              .select(["FIPS", "Cohort"])
        )

    # 2) Cohort means for baseline and counterfactual
    base_by = (
        baseline_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(cohort_lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("Base"))
    )

    cf_by = (
        counterfact_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(cohort_lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("CF"))
    )

    change = (
        base_by.join(cf_by, on=["Year", "Cohort"], how="inner")
        .with_columns(
            (pl.col("CF") - pl.col("Base")).alias("Delta"),
            (100.0 * (pl.col("CF") - pl.col("Base")) / pl.col("Base")).alias("PctDelta"),
        )
        .sort(["Year", "Cohort"])
    )

    ch_pd = change.to_pandas()

    # 3) Plot
    plt.figure(figsize=(10, 5))
    ycol = "PctDelta" if use_percent else "Delta"
    ylab = "% Change in Mean Predicted MR" if use_percent else "Change in Mean Predicted MR (CF − Baseline)"

    plt.axhline(0, color="k", linestyle="--", linewidth=1)

    for cohort in ["Low", "Mid", "High"]:
        g = ch_pd[ch_pd["Cohort"] == cohort]
        plt.plot(g["Year"], g[ycol], linewidth=2, label=f"{cohort} cohort")

    plt.xlabel("Year")
    plt.ylabel(ylab)
    plt.title(f"Cohort-Level Intervention Effect — {scenario_label}")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    plt.show()


### 1/30/26, EB: The following function is similar to the previous, plot_single_scenario_cohort_change
### but rather than plotting the percent change, it plots the actual MR values for all cohorts, 
### the baseline and counterfactual values.

def plot_single_scenario_cohort_actuals(
    *,
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
    risk_scores: pl.DataFrame,
    scenario_label: str,
):
    """
    One figure, one scenario:
      - Mean predicted MR by cohort over time
      - Solid lines: baseline predictions
      - Dashed lines: counterfactual predictions

    Cohorts are fixed using final-year AbsError_Risk:
      Low: bottom 50%, Mid: 50–90%, High: top 10%.
    """

    # --- 1) Define cohorts from FINAL risk year ---
    final_year = risk_scores.select(pl.col("Year").max()).item()

    rs = (
        risk_scores
        .filter(pl.col("Year") == final_year)
        .group_by("FIPS")
        .agg(pl.mean("AbsError_Risk").cast(pl.Float64).alias("risk"))
    )

    rmin = rs.select(pl.col("risk").min()).item()
    rmax = rs.select(pl.col("risk").max()).item()
    denom = rmax - rmin

    if denom == 0:
        cohort_lut = rs.with_columns(
            pl.lit("Mid").alias("Cohort")
        ).select(["FIPS", "Cohort"])
    else:
        cohort_lut = (
            rs.with_columns(((pl.col("risk") - rmin) / denom).alias("risk_norm"))
              .with_columns(
                  pl.when(pl.col("risk_norm") <= 0.50).then(pl.lit("Low"))
                   .when(pl.col("risk_norm") <= 0.90).then(pl.lit("Mid"))
                   .otherwise(pl.lit("High"))
                   .alias("Cohort")
              )
              .select(["FIPS", "Cohort"])
        )

    # --- 2) Cohort-wise means for baseline & counterfactual ---
    base_by = (
        baseline_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(cohort_lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("Base"))
    )

    cf_by = (
        counterfact_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(cohort_lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("CF"))
    )

    df = (
        base_by.join(cf_by, on=["Year", "Cohort"], how="inner")
        .sort(["Year", "Cohort"])
        .to_pandas()
    )

    # --- 3) Plot ---
    plt.figure(figsize=(10, 5))

    cohort_colors = {
        "Low":  "#1f77b4",
        "Mid":  "#ff7f0e",
        "High": "#2ca02c",
    }

    for cohort, color in cohort_colors.items():
        g = df[df["Cohort"] == cohort]

        # Baseline (solid)
        plt.plot(
            g["Year"], g["Base"],
            color=color,
            linewidth=2,
            linestyle="-",
            label=f"{cohort} cohort (baseline)"
        )

        # Counterfactual (dashed)
        plt.plot(
            g["Year"], g["CF"],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{cohort} cohort (counterfactual)"
        )

    plt.xlabel("Year")
    plt.ylabel("Mean Predicted Mortality Rate")
    plt.title(f"Cohort-Level Predicted MR — Baseline vs Counterfactual\n{scenario_label}")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    plt.show()


### 2/4/26, EB: Troubleshooting function to investigate why we get a spike in predicted MR for just high-risk cohort after RX intervention 

def diagnose_highrisk_spike(
    *,
    df_base: pl.DataFrame,                 # original df (unmodified), for RX multiplier check
    df_modified: pl.DataFrame,             # modified df (RX changed)
    baseline_predictions: pl.DataFrame,    # polars DF from baseline_predictions_df
    counterfact_predictions: pl.DataFrame, # polars DF from predict_counterfactual_cv_polars
    risk_scores: pl.DataFrame,             # polars DF from compute_all_risk_scores
    scenario_label: str,
    target_year: int = 2021,
    rx_col: str = "rx_rate",
    risk_col: str = "AbsError_Risk",
    cohort_s: float = 0.50,
    cohort_t: float = 0.90,
    top_k: int = 20,
):
    """
    Diagnose whether a cohort-level spike (e.g., high-risk in 2021) is driven by:
      - outliers vs broad-based sign flip,
      - RX modifier behaving oddly in feature_year,
      - missing/mismatched joins (implicitly via rx_mult sanity),
      - fold routing mismatch (optional extension).

    Prints summary tables and returns a delta DataFrame for deeper inspection.
    """

    feature_year = target_year - 1

    # --- 1) Build cohort LUT exactly like your plotting function (final-year risk) ---
    final_year = risk_scores.select(pl.col("Year").max()).item()

    rs = (
        risk_scores
        .filter(pl.col("Year") == final_year)
        .group_by("FIPS")
        .agg(pl.mean(risk_col).cast(pl.Float64).alias("risk"))
    )

    rmin = rs.select(pl.col("risk").min()).item()
    rmax = rs.select(pl.col("risk").max()).item()
    denom = rmax - rmin

    if denom == 0:
        cohort_lut = rs.with_columns(pl.lit("Mid").alias("Cohort")).select(["FIPS", "Cohort"])
    else:
        cohort_lut = (
            rs.with_columns(((pl.col("risk") - rmin) / denom).alias("risk_norm"))
              .with_columns(
                  pl.when(pl.col("risk_norm") <= cohort_s).then(pl.lit("Low"))
                   .when(pl.col("risk_norm") <= cohort_t).then(pl.lit("Mid"))
                   .otherwise(pl.lit("High"))
                   .alias("Cohort")
              )
              .select(["FIPS", "Cohort"])
        )

    # --- 2) Compute county-level deltas for the target_year ---
    base_y = (
        baseline_predictions
        .filter(pl.col("Year") == target_year)
        .select(["FIPS", "Year", pl.col("Predicted").alias("Base"), "Fold"])
    )

    cf_y = (
        counterfact_predictions
        .filter(pl.col("Year") == target_year)
        .select(["FIPS", "Year", pl.col("Predicted").alias("CF"), "Fold"])
    )

    delta = (
        base_y.join(cf_y, on=["FIPS", "Year", "Fold"], how="inner")
              .join(cohort_lut, on="FIPS", how="inner")
              .with_columns((pl.col("CF") - pl.col("Base")).alias("Delta"))
    )

    print(f"\n=== DIAGNOSE SPIKE: {scenario_label} | target_year={target_year} (feature_year={feature_year}) ===")

    # Summary by cohort
    summary = (
        delta.group_by("Cohort")
             .agg([
                 pl.len().alias("n"),
                 (pl.col("Delta") > 0).mean().alias("frac_positive"),
                 pl.col("Delta").mean().alias("mean_delta"),
                 pl.col("Delta").median().alias("median_delta"),
                 pl.col("Delta").quantile(0.95).alias("q95"),
                 pl.col("Delta").quantile(0.99).alias("q99"),
             ])
             .sort("Cohort")
    )
    print("\nDelta summary (CF - Base) by cohort:")
    print(summary)

    # High-risk outliers
    hi_outliers = (
        delta.filter(pl.col("Cohort") == "High")
             .sort("Delta", descending=True)
             .select(["FIPS", "Fold", "Base", "CF", "Delta"])
             .head(top_k)
    )
    print(f"\nTop {top_k} positive Delta counties in High cohort:")
    print(hi_outliers)

    # --- 3) RX multiplier sanity check in feature_year (this is where CF inputs come from) ---
    # Join base vs modified RX at feature_year and compare multipliers
    base_rx = df_base.filter(pl.col("year") == feature_year).select(["FIPS", pl.col(rx_col).alias("rx_base")])
    mod_rx  = df_modified.filter(pl.col("year") == feature_year).select(["FIPS", pl.col(rx_col).alias("rx_mod")])

    rx_cmp = (
        base_rx.join(mod_rx, on="FIPS", how="inner")
               .join(cohort_lut, on="FIPS", how="inner")
               .with_columns(
                   pl.when(pl.col("rx_base") > 0)
                     .then(pl.col("rx_mod") / pl.col("rx_base"))
                     .otherwise(None)
                     .alias("rx_mult")
               )
    )

    print("\nRX multiplier sanity (feature_year) by cohort:")
    rx_sum = (
        rx_cmp.group_by("Cohort")
              .agg([
                  pl.len().alias("n"),
                  (pl.col("rx_mod") > pl.col("rx_base")).sum().alias("rx_increased_count"),
                  pl.col("rx_mult").drop_nulls().min().alias("min_mult"),
                  pl.col("rx_mult").drop_nulls().max().alias("max_mult"),
                  pl.col("rx_mult").drop_nulls().mean().alias("mean_mult"),
                  pl.col("rx_mult").drop_nulls().n_unique().alias("n_unique_mult"),
              ])
              .sort("Cohort")
    )
    print(rx_sum)

    # Specifically inspect High cohort RX multipliers
    rx_hi = rx_cmp.filter(pl.col("Cohort") == "High").select(["FIPS", "rx_base", "rx_mod", "rx_mult"])
    rx_hi_out = rx_hi.sort("rx_mult", descending=True).head(10)
    print("\nHigh cohort: largest rx_mult (should not exceed 1; and should cluster around tier multiplier):")
    print(rx_hi_out)

    return delta, summary, hi_outliers, rx_sum

################################################################
### Testing new functions to modify rx rates based on risk scores
### The one above doesn't do what I thought it did, which was splitting 
### counties into risk cohorts based on quantiles of risk scores, then applying
### fixed multipliers per cohort.
### Additionally added updated plotting functions to carry through the correct cohorts.

from typing import Tuple, Optional

def apply_risk_based_rx_modifier_clean(
    df: pl.DataFrame,
    risk_scores: pl.DataFrame,
    *,
    rx_col: str = "rx_rate",
    risk_col: str = "AbsError_Risk",
    lambda_val: float = 0.80,          # high-risk multiplier
    quantiles: tuple[float, float] = (0.50, 0.90), # risk cohort cutoffs
    # q_low: float = 0.50,               # Low: <= q_low
    # q_mid: float = 0.90,               # Mid: (q_low, q_mid], High: > q_mid
    delta: tuple[float, float] = (0.10, 0.05),  # (low_delta, mid_delta)
    # mid_delta: float = 0.05,           # Mid multiplier = min(1, lambda_val + mid_delta)
    # low_delta: float = 0.10,           # Low multiplier = min(1, lambda_val + low_delta)
) -> Tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    
    """
    Quantile-tier RX modification based on latest-year risk.

    Tiers are defined on the distribution of `risk_col` in the latest year:
      - Low:  risk <= quantile(q_low)
      - Mid:  quantile(q_low) < risk <= quantile(q_mid)
      - High: risk > quantile(q_mid)

    Multipliers:
      - High: lambda_val
      - Mid:  min(1, lambda_val + mid_delta)
      - Low:  min(1, lambda_val + low_delta)

    The tier/multiplier is constant across all years for each county (static policy),
    because it is computed from the latest-year risk only.
    """
    q_low, q_mid = quantiles
    low_delta, mid_delta = delta

    # --- validation ---
    if rx_col not in df.columns:
        raise KeyError(f"RX column '{rx_col}' not found in df.")
    required = {"FIPS", "Year", risk_col}
    missing = required - set(risk_scores.columns)
    if missing:
        raise KeyError(f"risk_scores missing columns: {sorted(missing)}")

    if not (0 < lambda_val <= 1):
        raise ValueError("lambda_val must be in (0, 1].")
    if not (0 < q_low < q_mid < 1):
        raise ValueError("Require 0 < q_low < q_mid < 1.")
    if mid_delta < 0 or low_delta < 0:
        raise ValueError("mid_delta and low_delta must be >= 0.")

    mid_mult = min(1.0, lambda_val + mid_delta)
    low_mult = min(1.0, lambda_val + low_delta)
    if not (lambda_val <= mid_mult <= low_mult <= 1.0):
        raise ValueError(
            "Expected lambda_val <= mid_mult <= low_mult <= 1. "
            f"Got lambda_val={lambda_val}, mid_mult={mid_mult}, low_mult={low_mult}"
        )

    # --- latest-year risk, aggregated to one risk per FIPS ---
    latest_year = risk_scores.select(pl.col("Year").max()).item()

    rs = (
        risk_scores
        .filter(pl.col("Year") == latest_year)
        .group_by("FIPS")
        .agg(pl.mean(risk_col).cast(pl.Float64).alias("risk"))
    )

    # --- quantile cutpoints ---
    q_lo_val = rs.select(pl.col("risk").quantile(q_low)).item()
    q_mid_val = rs.select(pl.col("risk").quantile(q_mid)).item()

    # --- cohort lookup table: FIPS -> (risk, cohort, mult) ---
    cohort_lookup_table = (
        rs.with_columns(
            pl.when(pl.col("risk") <= q_lo_val).then(pl.lit("Low"))
              .when(pl.col("risk") <= q_mid_val).then(pl.lit("Mid"))
              .otherwise(pl.lit("High"))
              .alias("Cohort")
        )
        .with_columns(
            pl.when(pl.col("Cohort") == "High").then(pl.lit(lambda_val))
              .when(pl.col("Cohort") == "Mid").then(pl.lit(mid_mult))
              .otherwise(pl.lit(low_mult))
              .alias("rx_mult")
        )
        .select(["FIPS", "risk", "Cohort", "rx_mult"])
    )

    # Optional: print tier counts for transparency
    counts = cohort_lookup_table.group_by("Cohort").len().sort("Cohort")
    print(f"\n[apply_risk_based_rx_modifier_clean] latest_year={latest_year}, cutpoints: "
          f"q{int(q_low*100)}={q_lo_val:.6g}, q{int(q_mid*100)}={q_mid_val:.6g}")
    print(counts)

    # --- join + apply ---
    df_joined = df.join(cohort_lookup_table.select(["FIPS", "rx_mult"]), on="FIPS", how="left")

    # If any missing multipliers, treat as no change (mult=1)
    if df_joined["rx_mult"].null_count() > 0:
        df_joined = df_joined.with_columns(pl.col("rx_mult").fill_null(1.0))

    df_modified = df_joined.with_columns(
        (pl.col(rx_col) * pl.col("rx_mult")).alias(rx_col)
    ).drop("rx_mult")
    
    ### For plotting purposes, return lambda values and quantiles as well
    lambda_and_quantiles = {
        "lambda_vals": {"High": lambda_val, "Mid": mid_mult, "Low": low_mult},
        "quantiles": (q_lo_val, q_mid_val),
    }

    return (df_modified, cohort_lookup_table, lambda_and_quantiles)


### Helper function to compute cohort means for plotting:
def cohort_mean_predictions_base_cf(
    *,
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
    cohort_lookup_table: pl.DataFrame,
) -> pl.DataFrame:
    """
    Returns a Polars DF with columns:
      ["Year", "Cohort", "Base", "CF"]
    where Base/CF are mean Predicted MR within cohort for each Year.

    Requires:
      baseline_predictions:  ["FIPS","Year","Predicted"]
      counterfact_predictions:["FIPS","Year","Predicted"]
      cohort_lookup_table:            ["FIPS","Cohort"]
    """
    # --- validation ---
    for name, df_ in [
        ("baseline_predictions", baseline_predictions),
        ("counterfact_predictions", counterfact_predictions),
    ]:
        missing = {"FIPS", "Year", "Predicted"} - set(df_.columns)
        if missing:
            raise KeyError(f"{name} missing columns: {sorted(missing)}")

    missing = {"FIPS", "Cohort"} - set(cohort_lookup_table.columns)
    if missing:
        raise KeyError(f"cohort_lookup_table missing columns: {sorted(missing)}")

    lut = cohort_lookup_table.select(["FIPS", "Cohort"])
    base_by = (
        baseline_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("Base"))
    )

    cf_by = (
        counterfact_predictions
        .select(["FIPS", "Year", "Predicted"])
        .join(lut, on="FIPS", how="inner")
        .group_by(["Year", "Cohort"])
        .agg(pl.mean("Predicted").alias("CF"))
    )

    out = (
        base_by.join(cf_by, on=["Year", "Cohort"], how="inner")
              .sort(["Year", "Cohort"])
    )
    return out

### Helper function to add extra legend describing interventions to plots
def _add_lambda_legend(ax, *, lambda_vals: dict[str, float], quantiles: tuple[float, float] | None = None):
    """
    Adds a second legend describing intervention multipliers.
    lambda_vals keys expected: {"High","Mid","Low"} (order handled).
    """
    q_text = ""
    if quantiles is not None:
        q_low, q_mid = quantiles
        q_text = f"\nQuantile cutoffs: ({q_low:.2f}, {q_mid:.2f})"

    text = (
        r"$\lambda_{\mathrm{high}}$" + f" = {lambda_vals['High']:.2f}\n"
        r"$\lambda_{\mathrm{mid}}$"  + f"  = {lambda_vals['Mid']:.2f}\n"
        r"$\lambda_{\mathrm{low}}$"  + f"  = {lambda_vals['Low']:.2f}"
        # + q_text
    )

    # "Proxy" handle so legend renders text block nicely
    handle = Line2D([], [], linestyle="none", marker=None)
    leg2 = ax.legend(
        [handle],
        [text],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        handlelength=0,
        handletextpad=0,
        borderaxespad=0.0,
        fontsize=9,
    )
    # ax.add_artist(leg2)  # keep this legend when adding the main legend

### Helper function to compute actual cohort MR means to compare to baseline predictions and cf predictions
def cohort_mean_observed_mortality(
    *,
    df_actual: pl.DataFrame,
    cohort_lookup_table: pl.DataFrame,
    year_col: str = "year",
    fips_col: str = "FIPS",
    mort_col: str = "mortality_rate",
) -> pl.DataFrame:
    """
    Returns: Year, Cohort, Observed (mean mortality_rate)
    """
    required_df = {fips_col, year_col, mort_col}
    missing_df = required_df - set(df_actual.columns)
    if missing_df:
        raise KeyError(f"df_actual missing columns: {sorted(missing_df)}")

    required_lookup_table = {"FIPS", "Cohort"}
    missing_lookup_table = required_lookup_table - set(cohort_lookup_table.columns)
    if missing_lookup_table:
        raise KeyError(f"cohort_lookup_table missing columns: {sorted(missing_lookup_table)}")

    return (
        df_actual
        .select([pl.col(fips_col).cast(pl.Utf8), pl.col(year_col).cast(pl.Int64), pl.col(mort_col).cast(pl.Float64)])
        .join(cohort_lookup_table.select(["FIPS", "Cohort"]), on="FIPS", how="inner")
        .group_by([year_col, "Cohort"])
        .agg(pl.mean(mort_col).alias("Observed"))
        .rename({year_col: "Year"})
        .sort(["Year", "Cohort"])
    )

### Color consistency across plots for the cohorts.
COHORT_COLORS = {
    "Low":  "#1f77b4",  # blue
    "Mid":  "#ff7f0e",  # orange
    "High": "#2ca02c",  # green
}

### 2/6/26, EB: Added actual observed mean MR for cohorts, as comparison
def plot_single_scenario_cohort_actuals_lookup_table(
    *,
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
    cohort_lookup_table: pl.DataFrame,
    df_actual: pl.DataFrame,                    # NEW
    model_name: str,
    lambda_vals: dict[str, float],
    quantiles: tuple[float, float] | None = None,
):
    # --- Predicted cohort means (baseline vs counterfactual)
    df_plot = (
        cohort_mean_predictions_base_cf(
            baseline_predictions=baseline_predictions,
            counterfact_predictions=counterfact_predictions,
            cohort_lookup_table=cohort_lookup_table,
        )
        .to_pandas()
    )
    
    # --- Observed cohort means
    obs = (
        cohort_mean_observed_mortality(
            df_actual=df_actual,
            cohort_lookup_table=cohort_lookup_table,
        )
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for cohort in ["Low", "Mid", "High"]:
        g = df_plot[df_plot["Cohort"] == cohort]
        if g.empty:
            continue

        ax.plot(g["Year"], g["Base"], linewidth=2, linestyle="-",
                 color=COHORT_COLORS[cohort],
                 label=f"{cohort} cohort (baseline)")
        ax.plot(g["Year"], g["CF"], linewidth=2, linestyle="--",
                 color=COHORT_COLORS[cohort],
                 label=f"{cohort} cohort (counterfactual)")

        # Add observed (dotted)
        g_obs = obs[obs["Cohort"] == cohort]
        if not g_obs.empty:
            ax.plot(g_obs["Year"], g_obs["Observed"], linewidth=2, linestyle=":",
                     color=COHORT_COLORS[cohort],
                     label=f"{cohort} cohort (observed)")


    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Predicted Mortality Rate")
    ax.set_title(f"Cohort-Level Predicted MR — Baseline vs Counterfactual\n{model_name}")
    
    # Main legend (cohort lines+colors)
    leg1 = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    # Make sure we keep the first legend when we add a second legend
    ax.add_artist(leg1)
    
    # Secondary legend (lambda values)
    _add_lambda_legend(ax, lambda_vals=lambda_vals, quantiles=quantiles)
    
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    plt.show()

def plot_single_scenario_cohort_change_lookup_table(
    *,
    baseline_predictions: pl.DataFrame,
    counterfact_predictions: pl.DataFrame,
    cohort_lookup_table: pl.DataFrame,
    model_name: str,
    lambda_vals: dict[str, float],          # {"High":0.55,"Mid":0.60,"Low":0.65}
    quantiles: tuple[float, float] | None = None,
    use_percent: bool = False,
):
    df_means = cohort_mean_predictions_base_cf(
        baseline_predictions=baseline_predictions,
        counterfact_predictions=counterfact_predictions,
        cohort_lookup_table=cohort_lookup_table,
    )

    change = (
        df_means
        .with_columns(
            (pl.col("CF") - pl.col("Base")).alias("Delta"),
            pl.when(pl.col("Base") == 0)
              .then(None)
              .otherwise(100.0 * (pl.col("CF") - pl.col("Base")) / pl.col("Base"))
              .alias("PctDelta"),
        )
        .sort(["Year", "Cohort"])
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ycol = "PctDelta" if use_percent else "Delta"
    ylab = "% Change in Mean Predicted MR" if use_percent else "Change in Mean Predicted MR (CF − Baseline)"

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    
    for cohort in ["Low", "Mid", "High"]:
        g = change[change["Cohort"] == cohort]
        if g.empty:
            continue
        ax.plot(g["Year"], g[ycol], linewidth=2, color=COHORT_COLORS[cohort], label=f"{cohort} cohort")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylab)
    ax.set_title(f"Cohort-Level Intervention Effect — {model_name}")
    
    # Main legend (cohort lines+colors)
    leg1 = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    # Make sure we keep the first legend when we add a second legend
    ax.add_artist(leg1)
    
    # Secondary legend (lambda values)
    _add_lambda_legend(ax, lambda_vals=lambda_vals, quantiles=quantiles)
    
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    plt.show()



### -------------------------------------------------------------------------------------------------------------
### ------------------------------ G-Computation Functions ------------------------------------------------------
### -------------------------------------------------------------------------------------------------------------

### 3/15/26, EB: After reading through Clarke & Polselli (2026), I think using DML techniques is kind of doomed for problem
### setting.