### 11/13/25, EB: Contains helper functions for the counterfactual analysis.

import pandas as pd
import random
import polars as pl
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path


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
    
    random.seed(seed)
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
