"""
Sensitivity analysis for the scenario prediction pipeline.
 
Provides functions for:
  1. Dose-response curves: how do predicted mortality shifts scale
     with the magnitude of the dispensing reduction?
  2. Temporal profiles: how does the predicted shift evolve over
     the study period, with bootstrap uncertainty bands?
  3. Subgroup analysis: how does the predicted shift vary across
     county types (urbanicity, SVI level, etc.)?
 
All analyses are framed as descriptive characterizations of the
learned association between dispensing and mortality — no causal
claims are made.
"""

from __future__ import annotations
 
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.colors import TwoSlopeNorm
 
from pathlib import Path
from typing import Callable, Optional



# ─────────────────────────────────────────────────────────────────
# 1. Dose-Response Curves
# ─────────────────────────────────────────────────────────────────

def compute_dose_response(
    df_raw: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    model_dir: Path,
    feature_cols: list[str],
    reductions: list[float],
    build_counterfactual_fn: Callable,
    predict_counterfactual_fn: Callable,
    *,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """
    Compute mean predicted mortality at each reduction level and year.
 
    Returns a DataFrame with columns:
      Reduction, Year, MeanPred_Baseline, MeanPred_Scenario,
      PredictionShift, PctShift, Model
 
    Parameters
    ----------
    df_raw : pl.DataFrame
        Original raw panel (for building scenario panels).
    baseline_predictions : pl.DataFrame
        Baseline predictions from training.
    model_dir : Path
        Directory with saved fold models.
    feature_cols : list[str]
        Feature columns used in training.
    reductions : list[float]
        Reduction levels to evaluate (e.g., [0.05, 0.10, 0.15, 0.20, 0.25]).
    build_counterfactual_fn : callable
        Signature: (df_raw, reduction=r) -> pl.DataFrame
    predict_counterfactual_fn : callable
        Signature: (df_modified, baseline_predictions, model_dir, feature_cols) -> pl.DataFrame
    model_name : str
        Label for the model (for multi-model plots).
    """
    base_pdf = _to_pandas(baseline_predictions)
    years = sorted(base_pdf["Year"].unique())
 
    # Baseline means per year
    baseline_means = (
        base_pdf.groupby("Year")["Predicted"]
        .mean()
        .reset_index()
        .rename(columns={"Predicted": "MeanPred_Baseline"})
    )
 
    records = []
 
    # Reduction = 0 is the baseline
    for year in years:
        bm = baseline_means[baseline_means["Year"] == year]["MeanPred_Baseline"].values[0]
        records.append({
            "Reduction": 0.0,
            "Year": year,
            "MeanPred_Baseline": bm,
            "MeanPred_Scenario": bm,
            "PredictionShift": 0.0,
            "PctShift": 0.0,
            "Model": model_name,
        })
 
    for reduction in reductions:
        print(f"  Dose-response: reduction={reduction:.0%} ({model_name})")
 
        df_cf = build_counterfactual_fn(df_raw, reduction=reduction)
        cf_preds = predict_counterfactual_fn(
            df_modified=df_cf,
            baseline_predictions=baseline_predictions,
            model_dir=model_dir,
            feature_cols=feature_cols,
        )
 
        cf_pdf = _to_pandas(cf_preds)
 
        for year in years:
            bm = baseline_means[baseline_means["Year"] == year]["MeanPred_Baseline"].values[0]
            cf_mean = cf_pdf[cf_pdf["Year"] == year]["Predicted"].mean()
            shift = bm - cf_mean
            pct = (shift / bm * 100) if bm != 0 else 0.0
 
            records.append({
                "Reduction": reduction,
                "Year": year,
                "MeanPred_Baseline": bm,
                "MeanPred_Scenario": cf_mean,
                "PredictionShift": shift,
                "PctShift": pct,
                "Model": model_name,
            })
 
    return pd.DataFrame(records)
 

def plot_dose_response(
    dose_response_df: pd.DataFrame,
    *,
    target_col: str = "rx_rate",
    years: list[int] | None = None,
    figsize: tuple = (10, 6),
    title: str | None = None,
    model_name: str = "XGBoost",
):
    """
    Plot dose-response curves: predicted mortality vs reduction level.
 
    One curve per year (or per model if multi-model data is provided).
    """
    df = dose_response_df.copy()
    if years is not None:
        df = df[df["Year"].isin(years)]
 
    models = df["Model"].unique()
    plot_years = sorted(df["Year"].unique())
 
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if target_col == "rx_rate":
        xlabel = "RX Dispensing Reduction (%)"
        var_name = "RX Dispensing Rate"
    elif target_col == "unemp_rate":
        xlabel = "Unemployment Rate Reduction (%)"
        var_name = "Unemployment Rate"
    else:
        xlabel = "Uninsured Rate Reduction (%)"
        var_name = "Uninsured Rate"
 
    # Left panel: absolute predicted mortality
    ax = axes[0]
    for year in plot_years:
        for model in models:
            subset = df[(df["Year"] == year) & (df["Model"] == model)]
            subset = subset.sort_values("Reduction")
            label = f"{year}" if len(models) == 1 else f"{year} ({model})"
            if target_col == "rx_rate":
                ax.plot(
                    subset["Reduction"] * 100,
                    subset["MeanPred_Scenario"],
                    marker="o", markersize=4, label=label,
                )
            else:
                ax.plot(
                    subset["Reduction"],
                    subset["MeanPred_Scenario"],
                    marker="o", markersize=4, label=label,
                )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Predicted Mortality Rate")
    ax.set_title("Predicted Mortality by Reduction Level")
    ax.legend(fontsize=8, ncol=2)
 
    # Right panel: percentage shift from baseline
    ax = axes[1]
    for year in plot_years:
        for model in models:
            subset = df[(df["Year"] == year) & (df["Model"] == model)]
            subset = subset.sort_values("Reduction")
            label = f"{year}" if len(models) == 1 else f"{year} ({model})"
            if target_col == "rx_rate":
                ax.plot(
                    subset["Reduction"] * 100,
                    subset["PctShift"],
                    marker="o", markersize=4, label=label,
                )
            else:
                ax.plot(
                    subset["Reduction"],
                    subset["PctShift"],
                    marker="o", markersize=4, label=label,
                )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prediction Shift (%)")
    ax.set_title("% Change in Predicted Mortality")
    ax.legend(fontsize=8, ncol=2)
 
    fig.suptitle(
        title or f"Dose-Response: Predicted Mortality Under {var_name} Reductions, {model_name}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    # plt.show()
 
    return fig

def plot_dose_response_multi_model(
    dose_response_dfs: list[pd.DataFrame],
    *,
    target_col: str = "rx_rate",
    year: int,
    figsize: tuple = (8, 5),
    title: str | None = None,
):
    """
    Plot dose-response curves for a single year, comparing multiple models.
    Each DataFrame in the list should have a 'Model' column.
    """
    df = pd.concat(dose_response_dfs, ignore_index=True)
    df = df[df["Year"] == year].sort_values(["Model", "Reduction"])
 
    fig, ax = plt.subplots(figsize=figsize)
    if target_col == "rx_rate":
        xlabel = "RX Dispensing Reduction (%)"
    elif target_col == "unemp_rate":
        xlabel = "Unemployment Rate Reduction (%)"
    else:
        xlabel = "Uninsured Rate Reduction (%)"
 
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        ax.plot(
            subset["Reduction"] * 100,
            subset["MeanPred_Scenario"],
            marker="o", markersize=5, linewidth=2, label=model,
        )
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Predicted Mortality Rate")
    ax.set_title(title or f"Multi-Model Dose-Response ({year})")
    ax.legend()
    plt.tight_layout()
    # plt.show()
 
    return fig


# ─────────────────────────────────────────────────────────────────
# 2. Temporal Profiles
# ─────────────────────────────────────────────────────────────────

def plot_temporal_profile(
    baseline_predictions: pl.DataFrame,
    scenario_predictions: pl.DataFrame,
    df_actual: pl.DataFrame,
    *,
    reduction: float,
    model_name: str = "XGBoost",
    boot_summary: pd.DataFrame | None = None,
    figsize: tuple = (10, 5),
    title: str | None = None,
):
    """
    Plot predicted mortality over time: observed, baseline prediction,
    and scenario prediction, optionally with bootstrap CI bands.
 
    Parameters
    ----------
    baseline_predictions : pl.DataFrame
        Baseline model predictions (FIPS, Year, Predicted).
    scenario_predictions : pl.DataFrame
        Scenario model predictions (FIPS, Year, Predicted).
    df_actual : pl.DataFrame
        Raw data with actual mortality_rate and year columns.
    reduction : float
        The reduction level (for labeling).
    boot_summary : pd.DataFrame | None
        If provided, should have columns:
        Year, MeanPred_CF, Effect_CI_Lower, Effect_CI_Upper, MeanPred_Baseline
        Used to draw uncertainty bands.
    """
    base_pdf = _to_pandas(baseline_predictions)
    cf_pdf = _to_pandas(scenario_predictions)
 
    # Observed national means
    obs = (
        df_actual.group_by("year")
        .agg(pl.mean("mortality_rate").alias("MeanObserved"))
        .sort("year")
        .to_pandas()
        .rename(columns={"year": "Year"})
    )
 
    base_means = base_pdf.groupby("Year")["Predicted"].mean().reset_index()
    base_means.columns = ["Year", "MeanPred_Baseline"]
 
    cf_means = cf_pdf.groupby("Year")["Predicted"].mean().reset_index()
    cf_means.columns = ["Year", "MeanPred_Scenario"]
 
    fig, ax = plt.subplots(figsize=figsize)
 
    # Observed
    ax.plot(
        obs["Year"], obs["MeanObserved"],
        "k--", linewidth=2, label="Observed",
    )
 
    # Baseline prediction
    ax.plot(
        base_means["Year"], base_means["MeanPred_Baseline"],
        linewidth=2, label="Baseline Prediction",
    )
 
    # Scenario prediction
    ax.plot(
        cf_means["Year"], cf_means["MeanPred_Scenario"],
        linewidth=2, label=f"Scenario ({reduction:.0%} reduction)",
    )
 
    # Bootstrap CI bands (if available)
    if boot_summary is not None:
        # CI is on the shift (Effect), so scenario CI =
        # baseline - Effect_CI_Upper, baseline - Effect_CI_Lower
        merged = base_means.merge(boot_summary[["Year", "Effect_CI_Lower", "Effect_CI_Upper"]], on="Year")
        ci_upper = merged["MeanPred_Baseline"] - merged["Effect_CI_Lower"]
        ci_lower = merged["MeanPred_Baseline"] - merged["Effect_CI_Upper"]
        ax.fill_between(
            merged["Year"], ci_lower, ci_upper,
            alpha=0.2, label="95% Bootstrap CI",
        )
 
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Mortality Rate")
    ax.set_title(title or f"Temporal Profile: {reduction:.0%} Reduction ({model_name})")
    ax.legend()
    plt.tight_layout()
    # plt.show()
 
    return fig


# ─────────────────────────────────────────────────────────────────
# 3. Subgroup Analysis
# ─────────────────────────────────────────────────────────────────

def compute_subgroup_shifts(
    baseline_predictions: pl.DataFrame,
    scenario_predictions: pl.DataFrame,
    df_enhanced: pl.DataFrame,
    *,
    group_col: str = "urbanicity_class",
    reduction: float,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """
    Compute prediction shifts within subgroups defined by `group_col`.
 
    Returns a DataFrame with columns:
      Group, Year, N_Counties, MeanPred_Baseline, MeanPred_Scenario,
      PredictionShift, PctShift, Reduction, Model
 
    Parameters
    ----------
    baseline_predictions, scenario_predictions : pl.DataFrame
        Must contain FIPS, Year, Predicted.
    df_enhanced : pl.DataFrame
        Enhanced panel containing the grouping column.
    group_col : str
        Column to group by (e.g., "urbanicity_class").
    reduction : float
        For labeling.
    model_name : str
        For labeling.
    """
    base_pdf = _to_pandas(baseline_predictions)
    cf_pdf = _to_pandas(scenario_predictions)
 
    # Get FIPS -> group mapping (use most recent year to avoid
    # issues with counties changing group over time)
    latest_year = df_enhanced["year"].max()
    group_map = (
        df_enhanced.filter(pl.col("year") == latest_year)
        .select(["FIPS", group_col])
        .unique()
        .to_pandas()
    )
 
    # Merge group labels into predictions
    base_merged = base_pdf.merge(group_map, on="FIPS", how="left")
    cf_merged = cf_pdf.merge(group_map, on="FIPS", how="left")
 
    records = []
    years = sorted(base_pdf["Year"].unique())
 
    for year in years:
        by = base_merged[base_merged["Year"] == year]
        cy = cf_merged[cf_merged["Year"] == year]
 
        for group in sorted(by[group_col].dropna().unique()):
            bg = by[by[group_col] == group]
            cg = cy[cy[group_col] == group]
 
            if len(bg) == 0 or len(cg) == 0:
                continue
 
            bm = bg["Predicted"].mean()
            cm = cg["Predicted"].mean()
            shift = bm - cm
            pct = (shift / bm * 100) if bm != 0 else 0.0
 
            records.append({
                "Group": group,
                "Year": year,
                "N_Counties": len(bg),
                "MeanPred_Baseline": bm,
                "MeanPred_Scenario": cm,
                "PredictionShift": shift,
                "PctShift": pct,
                "Reduction": reduction,
                "Model": model_name,
            })
 
    return pd.DataFrame(records)

def plot_subgroup_shifts(
    subgroup_df: pd.DataFrame,
    *,
    metric: str = "PredictionShift",
    figsize: tuple = (10, 6),
    title: str | None = None,
    target_col: str = "rx_rate",
):
    """
    Plot prediction shifts by subgroup over time.
    One line per subgroup.
    """
    fig, ax = plt.subplots(figsize=figsize)
 
    groups = sorted(subgroup_df["Group"].unique())
    for group in groups:
        subset = subgroup_df[subgroup_df["Group"] == group].sort_values("Year")
        ax.plot(
            subset["Year"], subset[metric],
            marker="o", markersize=4, linewidth=2, label=group,
        )
 
    reduction = subgroup_df["Reduction"].iloc[0]
    model = subgroup_df["Model"].iloc[0]
 
    ylabel = "Prediction Shift (mortality rate)" if metric == "PredictionShift" else "Prediction Shift (%)"
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    if target_col == "rx_rate":
        ax.set_title(title or f"Prediction Shifts by Subgroup: {reduction:.0%} Reduction ({model})")
    else:
        ax.set_title(title or f"Prediction Shifts by Subgroup: {reduction:.1f} Reduction ({model})")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    # plt.show()
 
    return fig


def plot_subgroup_dose_response(
    df_raw: pl.DataFrame,
    df_enhanced: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    model_dir: Path,
    feature_cols: list[str],
    reductions: list[float],
    build_counterfactual_fn: Callable,
    predict_counterfactual_fn: Callable,
    *,
    group_col: str = "urbanicity_class",
    year: int,
    model_name: str = "XGBoost",
    figsize: tuple = (8, 5),
    title: str | None = None,
) -> tuple[matplotlib.figure.Figure, pd.DataFrame]:
    """
    Plot dose-response curves by subgroup for a single year.
 
    Returns the figure and the underlying data.
    """
    all_subgroup_records = []
 
    # Baseline subgroup means
    base_pdf = _to_pandas(baseline_predictions)
    latest_year = df_enhanced["year"].max()
    group_map = (
        df_enhanced.filter(pl.col("year") == latest_year)
        .select(["FIPS", group_col])
        .unique()
        .to_pandas()
    )
 
    base_merged = base_pdf.merge(group_map, on="FIPS", how="left")
    base_year = base_merged[base_merged["Year"] == year]
 
    for group in sorted(base_year[group_col].dropna().unique()):
        bm = base_year[base_year[group_col] == group]["Predicted"].mean()
        all_subgroup_records.append({
            "Group": group, "Reduction": 0.0,
            "MeanPred_Scenario": bm, "Year": year,
        })
 
    for reduction in reductions:
        df_cf = build_counterfactual_fn(df_raw, reduction=reduction)
        cf_preds = predict_counterfactual_fn(
            df_modified=df_cf,
            baseline_predictions=baseline_predictions,
            model_dir=model_dir,
            feature_cols=feature_cols,
        )
        cf_pdf = _to_pandas(cf_preds)
        cf_merged = cf_pdf.merge(group_map, on="FIPS", how="left")
        cf_year = cf_merged[cf_merged["Year"] == year]
 
        for group in sorted(cf_year[group_col].dropna().unique()):
            cm = cf_year[cf_year[group_col] == group]["Predicted"].mean()
            all_subgroup_records.append({
                "Group": group, "Reduction": reduction,
                "MeanPred_Scenario": cm, "Year": year,
            })
 
    result_df = pd.DataFrame(all_subgroup_records)
 
    fig, ax = plt.subplots(figsize=figsize)
    for group in sorted(result_df["Group"].unique()):
        subset = result_df[result_df["Group"] == group].sort_values("Reduction")
        ax.plot(
            subset["Reduction"] * 100,
            subset["MeanPred_Scenario"],
            marker="o", markersize=5, linewidth=2, label=group,
        )
 
    ax.set_xlabel("RX Dispensing Reduction (%)")
    ax.set_ylabel("Mean Predicted Mortality Rate")
    ax.set_title(title or f"Dose-Response by {group_col} ({year}, {model_name})")
    ax.legend()
    plt.tight_layout()
    # plt.show()
 
    return fig, result_df

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _to_pandas(df) -> pd.DataFrame:
    """Convert Polars or Pandas DataFrame to Pandas."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


# ------------------------------------------------------------------
# Risk-tiered temporal summary + plotting helpers (Polars + matplotlib)
# ------------------------------------------------------------------
def load_risk_tier_lookup(training_run_dir, reduction: float) -> pl.DataFrame:
    """Load cohort lookup for a given reduction as a Polars DataFrame.

    The file is expected at: training_run_dir / 'scenarios' /
    f'cohort_lookup_r{reduction:.2f}.csv'. Ensures `FIPS` is a
    5-character zero-padded string.
    """
    p = Path(training_run_dir) / "scenarios" / f"cohort_lookup_r{reduction:.2f}.csv"
    df = pl.read_csv(p, try_parse_dates=False)
    # Ensure FIPS preserved as text and zero-padded to 5 chars
    if "FIPS" in df.columns:
        df = df.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
    return df


# def compute_risk_tier_temporal_shifts(
#     baseline_predictions: pl.DataFrame,  
#     scenario_predictions: pl.DataFrame,
#     cohort_lookup: pl.DataFrame,
# ) -> tuple[pl.DataFrame, pl.DataFrame]:
#     """Compute county-level deltas and Year x Cohort summaries.

#     Steps:
#       - attach `Cohort` to predictions via `FIPS`
#       - average predictions across `Fold` to obtain one value per
#         (FIPS, Year, Cohort)
#       - compute `delta = Predicted_baseline - Predicted_scenario`
#       - summarize by Year x Cohort: median, 25th, 75th, n_counties

#     Returns (county_level_df, summary_df) as Polars DataFrames.
#     """
#     # ensure Polars inputs
#     if not isinstance(baseline_predictions, pl.DataFrame):
#         baseline = pl.from_pandas(baseline_predictions)
#     else:
#         baseline = baseline_predictions
#     if not isinstance(scenario_predictions, pl.DataFrame):
#         scenario = pl.from_pandas(scenario_predictions)
#     else:
#         scenario = scenario_predictions
#     cohort = cohort_lookup

#     # preserve FIPS formatting
#     baseline = baseline.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
#     scenario = scenario.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
#     cohort = cohort.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))

#     # attach Cohort to each prediction table
#     baseline = baseline.join(cohort.select(["FIPS", "Cohort"]), on="FIPS", how="left")
#     scenario = scenario.join(cohort.select(["FIPS", "Cohort"]), on="FIPS", how="left")

#     # average across folds to obtain one value per county-year
#     base_mean = (
#         baseline
#         .groupby(["FIPS", "Year", "Cohort"], maintain_order=True)
#         .agg(pl.col("Predicted").mean().alias("Predicted_baseline"))
#     )
#     scen_mean = (
#         scenario
#         .groupby(["FIPS", "Year", "Cohort"], maintain_order=True)
#         .agg(pl.col("Predicted").mean().alias("Predicted_scenario"))
#     )

#     # join baseline vs scenario on FIPS, Year, Cohort
#     merged = base_mean.join(scen_mean, on=["FIPS", "Year", "Cohort"], how="inner")

#     # delta per county-year
#     merged = merged.with_columns((pl.col("Predicted_baseline") - pl.col("Predicted_scenario")).alias("delta"))

#     # summary by Year x Cohort: median, q25, q75, n_counties
#     summary = (
#         merged
#         .groupby(["Year", "Cohort"], maintain_order=True)
#         .agg(
#             pl.col("delta").median().alias("median_delta"),
#             pl.col("delta").quantile(0.25).alias("q25_delta"),
#             pl.col("delta").quantile(0.75).alias("q75_delta"),
#             pl.col("FIPS").n_unique().alias("n_counties"),
#         )
#         .sort(["Cohort", "Year"])
#     )

#     return merged, summary

### 4/5/26, EB: This new compute_risk_tier_temporal_shifts function merges baseline and scenario at the row level before averaging across folds,
### which makes more sense conceptually. We want the average of the differences, not the difference of the averages.
def compute_risk_tier_temporal_shifts(
    baseline_predictions: pl.DataFrame,
    scenario_predictions: pl.DataFrame,
    cohort_lookup: pl.DataFrame,
    rel_eps: float = 0.5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute county-level counterfactual shifts and Year x Cohort summaries.

    Logic:
      1. Attach cohort labels by FIPS
      2. Merge baseline and scenario predictions at the row level
         using FIPS, Year, and Fold
      3. Compute delta = Predicted_baseline - Predicted_scenario
      4. Average delta across folds for each county-year-cohort
      5. Summarize county-level deltas by Year x Cohort

    Returns
    -------
    county_level_df : pl.DataFrame
        One row per county-year-cohort, with fold-averaged delta.
    summary_df : pl.DataFrame
        Year x Cohort summary with mean, median, q25, q75, and n_counties.
    """
    # Ensure Polars inputs
    baseline = (
        baseline_predictions
        if isinstance(baseline_predictions, pl.DataFrame)
        else pl.from_pandas(baseline_predictions)
    )
    scenario = (
        scenario_predictions
        if isinstance(scenario_predictions, pl.DataFrame)
        else pl.from_pandas(scenario_predictions)
    )
    cohort = (
        cohort_lookup
        if isinstance(cohort_lookup, pl.DataFrame)
        else pl.from_pandas(cohort_lookup)
    )

    # Standardize FIPS
    baseline = baseline.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
    scenario = scenario.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
    cohort = cohort.with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))

    # Attach cohort labels
    cohort_labels = cohort.select(["FIPS", "Cohort"]).unique()

    baseline = baseline.join(cohort_labels, on="FIPS", how="left")
    scenario = scenario.join(cohort_labels, on="FIPS", how="left")

    # Merge baseline and scenario at the most granular level
    merged = baseline.join(
        scenario.select(["FIPS", "Year", "Fold", "Predicted"]),
        on=["FIPS", "Year", "Fold"],
        how="inner",
        suffix="_scenario",
    ).rename({"Predicted": "Predicted_baseline"})

    # After join, scenario Predicted column will be named Predicted_scenario
    if "Predicted_scenario" not in merged.columns:
        # Depending on Polars join behavior/version, rename explicitly
        pred_cols = [c for c in merged.columns if c.startswith("Predicted")]
        if len(pred_cols) >= 2:
            merged = merged.rename({pred_cols[1]: "Predicted_scenario"})

    # Row-level delta
    merged = merged.with_columns(
        (pl.col("Predicted_baseline") - pl.col("Predicted_scenario")).alias("delta")
    )

    # Average delta across folds to get one county-level shift per year
    county_level = (
        merged
        .group_by(["FIPS", "Year", "Cohort"], maintain_order=True)
        .agg(
            pl.col("delta").mean().alias("delta"),
            pl.col("Predicted_baseline").mean().alias("Predicted_baseline"),
            pl.col("Predicted_scenario").mean().alias("Predicted_scenario"),
            pl.col("True").mean().alias("Observed"),
        )
        .with_columns(
            pl.when(pl.col("Predicted_baseline").abs() > rel_eps)
            .then(pl.col("delta") / pl.col("Predicted_baseline"))
            .otherwise(None)
            .alias("relative_change")
        )
        .sort(["Cohort", "Year", "FIPS"])
    )

    # Summarize across counties within each Year x Cohort
    summary = (
        county_level
        .group_by(["Year", "Cohort"], maintain_order=True)
        .agg(
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").median().alias("median_delta"),
            pl.col("delta").quantile(0.25).alias("q25_delta"),
            pl.col("delta").quantile(0.75).alias("q75_delta"),
            pl.col("FIPS").n_unique().alias("n_counties"),
            pl.col("relative_change").mean().alias("mean_relative_change"),
            pl.col("relative_change").median().alias("median_relative_change"),
            pl.col("relative_change").quantile(0.25).alias("q25_relative_change"),
            pl.col("relative_change").quantile(0.75).alias("q75_relative_change"),
        )
        .sort(["Cohort", "Year"])
    )

    return county_level, summary


def plot_risk_tier_temporal_profile(
    summary_df,
    reduction: float,
    target_col: str,
    model_name: str,
    *,
    center_col: str = "mean_delta",
    cohort_order: list[str] | None = None,
    cohort_labels: dict | None = None,
    uncertainty: str = "bars",  # "bands" for IQR ribbons, "bars" for pointwise error bars
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """Plot mean/median delta with uncertainty for each cohort over time.
 
    Parameters
    ----------
    uncertainty : str
        "bands"  — continuous IQR ribbon (original behaviour)
        "bars"   — pointwise IQR error bars with caps
    `summary_df` may be a Polars or Pandas DataFrame with columns:
    Year, Cohort, mean_delta, median_delta, q25_delta, q75_delta, n_counties.
    """
    if isinstance(summary_df, pl.DataFrame):
        df = summary_df.to_pandas()
    else:
        df = summary_df.copy()
    df = df.sort_values(["Cohort", "Year"]).reset_index(drop=True)
 
    if cohort_order is None:
        cohort_order = list(df["Cohort"].unique())
    labels = cohort_labels or {c: c for c in cohort_order}
 
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color")
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(cohort_order)}
 
    for cohort in cohort_order:
        sub = df[df["Cohort"] == cohort].sort_values("Year")
        if sub.empty:
            continue
 
        x = sub["Year"].values
        y = sub[center_col].values
        lo = sub["q25_delta"].values
        hi = sub["q75_delta"].values
        c = color_map.get(cohort)
 
        if uncertainty == "bands":
            ax.plot(x, y, label=labels.get(cohort, cohort), color=c)
            ax.fill_between(x, lo, hi, alpha=0.25, color=c)
        elif uncertainty == "bars":
            yerr_lower = y - lo
            yerr_upper = hi - y
            ax.errorbar(
                x, y,
                yerr=[yerr_lower, yerr_upper],
                label=labels.get(cohort, cohort),
                color=c,
                marker="o",
                markersize=4,
                capsize=3,
                capthick=1.2,
                elinewidth=1.2,
                linewidth=1.8,
                ecolor=c,
                alpha=0.85,
            )
        else:
            raise ValueError(f"uncertainty must be 'bands' or 'bars', got '{uncertainty}'")
 
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Delta (baseline - scenario)\n(predicted mortality rate)")
    ax.set_title(f"Risk-tier temporal effect ({model_name}) — r{reduction:.2f}")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_risk_tier_delta_heatmap(
    county_level_df,
    reduction: float,
    target_col: str,
    model_name: str,
    *,
    agg_col: str = "delta",
    agg_func: str = "mean",   # "mean" or "median"
    cohort_order: list[str] | None = None,
    cohort_labels: dict[str, str] | None = None,
    # figsize: tuple[float, float] = (8, 3.5),
    # dpi: int = 300,
    # cmap: str = "RdBu_r",
    annotate: bool = True,
    # value_fmt: str = ".2f",
) -> matplotlib.figure.Figure:
    """
    Heatmap of prediction shifts by Year x Cohort.

    Expected columns in county_level_df:
        FIPS, Year, Cohort, delta, Predicted_baseline, Predicted_scenario
    """
    if isinstance(county_level_df, pl.DataFrame):
        df = county_level_df.to_pandas()
    else:
        df = county_level_df.copy()

    required = {"Year", "Cohort", agg_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"county_level_df missing required columns: {sorted(missing)}")

    if cohort_order is None:
        cohort_order = list(pd.unique(df["Cohort"]))

    labels = cohort_labels or {c: c for c in cohort_order}

    if agg_func not in {"mean", "median"}:
        raise ValueError("agg_func must be 'mean' or 'median'")

    grouped = (
        df.groupby(["Cohort", "Year"], observed=False)[agg_col]
        .agg(agg_func)
        .reset_index()
    )

    heatmap_df = (
        grouped.pivot(index="Cohort", columns="Year", values=agg_col)
        .reindex(cohort_order)
    )
    heatmap_df.index = [labels.get(c, c) for c in heatmap_df.index]

    values = heatmap_df.to_numpy(dtype=float)

    vmax = np.nanmax(np.abs(values))
    norm = (
        TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        if np.isfinite(vmax) and vmax > 0
        else None
    )

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    ax.set_xlabel("Year")
    ax.set_ylabel("Risk cohort")
    ax.set_title(
        f"Prediction shift by risk tier ({model_name}) — r{reduction:.2f}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta = baseline prediction - scenario prediction")

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        format(val, ".2f"),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

    plt.tight_layout()
    return fig

def _resolve_effect_col(effect_scale: str) -> tuple[str, str]:
    """
    Map effect_scale to the dataframe column and axis label.

    Parameters
    ----------
    effect_scale : {"absolute", "relative"}

    Returns
    -------
    effect_col : str
    effect_label : str
    """
    if effect_scale == "absolute":
        return (
            "delta",
            "Delta = baseline prediction - scenario prediction",
        )
    elif effect_scale == "relative":
        return (
            "relative_change",
            r"Relative change = $\Delta / \hat{Y}_{\mathrm{baseline}}$",
        )
    else:
        raise ValueError("effect_scale must be 'absolute' or 'relative'")


def plot_effect_vs_observed_scatter(
    county_level_df,
    reduction: float,
    target_col: str,
    model_name: str,
    *,
    effect_scale: str = "absolute",   # "absolute" or "relative"
    cohort_order: list[str] | None = None,
    cohort_labels: dict[str, str] | None = None,
    # alpha: float = 0.30,
    # s: float = 16,
    # figsize: tuple[float, float] = (8, 6),
    # dpi: int = 300,
    add_cohort_trend: bool = True,
) -> matplotlib.figure.Figure:
    """
    Scatter of observed mortality vs intervention effect, colored by risk cohort.

    effect_scale="absolute" plots:
        delta = baseline_pred - scenario_pred

    effect_scale="relative" plots:
        relative_change = delta / baseline_pred
    """
    if isinstance(county_level_df, pl.DataFrame):
        df = county_level_df.to_pandas()
    else:
        df = county_level_df.copy()

    effect_col, effect_label = _resolve_effect_col(effect_scale)

    required = {"Observed", effect_col, "Cohort", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"county_level_df missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["Observed", effect_col, "Cohort", "Year"]).copy()

    if cohort_order is None:
        cohort_order = list(pd.unique(df["Cohort"]))

    labels = cohort_labels or {c: c for c in cohort_order}
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["C0", "C1", "C2"])
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(cohort_order)}

    fig, ax = plt.subplots(figsize=(8,6), dpi=300)

    for cohort in cohort_order:
        sub = df.loc[df["Cohort"] == cohort]
        if sub.empty:
            continue

        x = sub["Observed"].to_numpy(dtype=float)
        y = sub[effect_col].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            alpha=0.30,
            s=16,
            color=color_map[cohort],
            label=labels.get(cohort, cohort),
        )

        if add_cohort_trend:
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
                x_line = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
                y_line = intercept + slope * x_line
                ax.plot(
                    x_line,
                    y_line,
                    color=color_map[cohort],
                    linewidth=2,
                )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Observed mortality rate")
    ax.set_ylabel(effect_label)

    scale_name = "absolute shift" if effect_scale == "absolute" else "relative shift"
    ax.set_title(
        f"Prediction {scale_name} vs observed mortality ({model_name}) — r{reduction:.2f}"
    )
    ax.legend()
    plt.tight_layout()
    return fig

import math

def plot_effect_vs_observed_scatter_by_year(
    county_level_df,
    reduction: float,
    target_col: str,
    model_name: str,
    *,
    effect_scale: str = "absolute",   # "absolute" or "relative"
    cohort_order: list[str] | None = None,
    cohort_labels: dict[str, str] | None = None,
    alpha: float = 0.28,
    s: float = 12,
    add_cohort_trend: bool = True,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 3.2),
    dpi: int = 300,
    sharex: bool = True,
    sharey: bool = True,
) -> matplotlib.figure.Figure:
    """
    Faceted scatter of observed mortality vs intervention effect, one panel per year.
    """
    if isinstance(county_level_df, pl.DataFrame):
        df = county_level_df.to_pandas()
    else:
        df = county_level_df.copy()

    effect_col, effect_label = _resolve_effect_col(effect_scale)

    required = {"Observed", effect_col, "Cohort", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"county_level_df missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["Observed", effect_col, "Cohort", "Year"]).copy()

    if cohort_order is None:
        cohort_order = list(pd.unique(df["Cohort"]))

    labels = cohort_labels or {c: c for c in cohort_order}
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["C0", "C1", "C2"])
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(cohort_order)}

    years = sorted(pd.unique(df["Year"]))
    n_panels = len(years)
    nrows = math.ceil(n_panels / ncols)

    figsize = (4.0 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=300,
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, year in zip(axes, years):
        sub_year = df.loc[df["Year"] == year]

        for cohort in cohort_order:
            sub = sub_year.loc[sub_year["Cohort"] == cohort]
            if sub.empty:
                continue

            x = sub["Observed"].to_numpy(dtype=float)
            y = sub[effect_col].to_numpy(dtype=float)

            ax.scatter(
                x,
                y,
                alpha=0.30,
                s=16,
                color=color_map[cohort],
                label=labels.get(cohort, cohort),
            )

            if add_cohort_trend:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 2:
                    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
                    x_line = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
                    y_line = intercept + slope * x_line
                    ax.plot(
                        x_line,
                        y_line,
                        color=color_map[cohort],
                        linewidth=1.8,
                    )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(str(year))
        ax.set_xlabel("Observed mortality")
        ax.set_ylabel(effect_label)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=len(handles),
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
        )

    scale_name = "absolute shift" if effect_scale == "absolute" else "relative shift"
    fig.suptitle(
        f"Prediction {scale_name} vs observed mortality by year ({model_name}) — r{reduction:.2f}",
        y=1.06,
    )
    plt.tight_layout()
    return fig


def plot_mortality_decomposition(
    county_level_df: pl.DataFrame | pd.DataFrame,
    *,
    year: int,
    reduction: float,
    target_col: str,
    model_name: str = "XGBoost",
    cohort_order: list[str] | None = None,
    cohort_labels: dict | None = None,
    orientation: str = "vertical",  # "vertical" or "horizontal"
    figsize: tuple = (8, 5),
    dpi: int = 300,
    title: str | None = None,
) -> tuple[matplotlib.figure.Figure, pd.DataFrame]:
    """
    Waterfall decomposition of observed mortality into three components
    per risk tier for a single year and intervention scenario:

        observed = counterfactual_pred + intervention_effect + residual

    where:
        counterfactual_pred  = ŷ_cf        (model prediction under intervention)
        intervention_effect  = ŷ - ŷ_cf    (what the policy can reach)
        residual             = y - ŷ        (what the model cannot explain)

    Parameters
    ----------
    county_level_df : DataFrame
        Output of compute_risk_tier_temporal_shifts, with columns:
        FIPS, Year, Cohort, delta, Predicted_baseline, Predicted_scenario,
        Observed, relative_change.
    year : int
        Focal year to display.
    reduction : float
        Reduction level applied (for labelling only).
    target_col : str
        Name of the intervention variable (for labelling only).
    model_name : str
        Model name (for labelling only).
    cohort_order : list[str], optional
        Display order for cohorts. If None, sorted unique values are used.
    cohort_labels : dict, optional
        Display labels for cohorts.
    orientation : str
        "vertical"   — cohorts on x-axis, stacked bars going up
        "horizontal" — cohorts on y-axis, stacked bars going right

    Returns
    -------
    fig : matplotlib.figure.Figure
    decomp_df : pd.DataFrame
        Underlying decomposition data with one row per cohort.
    """
    # --- Normalise to pandas ------------------------------------------
    if isinstance(county_level_df, pl.DataFrame):
        df = county_level_df.to_pandas()
    else:
        df = county_level_df.copy()

    # --- Filter to focal year -----------------------------------------
    df_yr = df[df["Year"] == year].copy()

    # --- Components per county ----------------------------------------
    # intervention_effect = ŷ - ŷ_cf  (already stored as "delta")
    # residual            = y - ŷ
    df_yr["residual"] = df_yr["Observed"] - df_yr["Predicted_baseline"]

    # --- Aggregate by cohort ------------------------------------------
    if cohort_order is None:
        cohort_order = sorted(df_yr["Cohort"].dropna().unique())
    labels = cohort_labels or {c: c for c in cohort_order}

    records = []
    for cohort in cohort_order:
        sub = df_yr[df_yr["Cohort"] == cohort]
        if sub.empty:
            continue
        records.append({
            "Cohort": labels.get(cohort, cohort),
            "Counterfactual Prediction": sub["Predicted_scenario"].mean(),
            "Intervention Effect": sub["delta"].mean(),
            "Residual": sub["residual"].mean(),
            "Observed": sub["Observed"].mean(),
            "n_counties": len(sub),
        })
    decomp_df = pd.DataFrame(records)

    # --- Plot ---------------------------------------------------------
    component_cols = [
        "Counterfactual Prediction",
        "Intervention Effect",
        "Residual",
    ]
    component_colors = ["#5a9bd5", "#ed7d31", "#a5a5a5"]
    component_hatches = [None, "//", ".."]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n = len(decomp_df)
    cohort_labels_display = decomp_df["Cohort"].values

    if orientation == "vertical":
        x = np.arange(n)
        bottoms = np.zeros(n)
        for col, color, hatch in zip(
            component_cols, component_colors, component_hatches
        ):
            vals = decomp_df[col].values
            ax.bar(
                x, vals, bottom=bottoms, color=color,
                edgecolor="white", linewidth=0.8,
                hatch=hatch, label=col, width=0.6,
            )
            bottoms += vals

        ax.scatter(
            x, decomp_df["Observed"].values,
            color="black", marker="_", s=200, linewidths=2,
            zorder=5, label="Observed Mortality",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(cohort_labels_display)
        ax.set_ylabel("Mortality Rate (per 100k)")
        ax.set_xlabel("Risk Tier")

    elif orientation == "horizontal":
        y = np.arange(n)
        lefts = np.zeros(n)
        for col, color, hatch in zip(
            component_cols, component_colors, component_hatches
        ):
            vals = decomp_df[col].values
            ax.barh(
                y, vals, left=lefts, color=color,
                edgecolor="white", linewidth=0.8,
                hatch=hatch, label=col, height=0.6,
            )
            lefts += vals

        ax.scatter(
            decomp_df["Observed"].values, y,
            color="black", marker="|", s=200, linewidths=2,
            zorder=5, label="Observed Mortality",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(cohort_labels_display)
        ax.set_xlabel("Mortality Rate (per 100k)")
        ax.set_ylabel("Risk Tier")

    else:
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got '{orientation}'"
        )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(
        title or (
            f"Mortality Decomposition — {target_col} "
            f"reduction {reduction*100:.0f}% "
            f"({model_name}, {year})"
        )
    )
    plt.tight_layout()

    return fig, decomp_df

# ------------------------------------------------------------------
# Example code block to plug into `analyze_results()`
# Insert this snippet inside the reductions loop in analyze_results():
# ------------------------------------------------------------------
#
# # risk-tiered temporal plots and CSVs
# cohort = load_risk_tier_lookup(training_run_dir, reduction)
# merged_county_df, summary_df = compute_risk_tier_temporal_shifts(
#     baseline_predictions, scenario_preds[reduction], cohort
# )
# fig = plot_risk_tier_temporal_profile(
#     summary_df, reduction, target_col, model_name
# )
# out_png = Path(plots_dir) / f"risk_tier_temporal_{target_col}_r{reduction:.2f}.png"
# fig.savefig(out_png, dpi=300)
# out_csv = Path(plots_dir) / f"risk_tier_temporal_{target_col}_r{reduction:.2f}.csv"
# # save summary as CSV (use pandas)
# summary_df.to_pandas().to_csv(out_csv, index=False)