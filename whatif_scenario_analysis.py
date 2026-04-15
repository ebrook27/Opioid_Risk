"""
run_analysis.py — Main orchestration script for scenario analysis.

Four pipeline functions:
  1. train_and_predict:  Train models from scratch, then run scenario predictions
  2. predict_from_existing:  Load a previous training run, run scenario predictions
  3. bootstrap_from_existing:  Load a previous training run, run bootstrap inference
  4. analyze_results:  Load scenario predictions, run sensitivity analyses

Each function is self-contained and can be called independently.
"""

from __future__ import annotations

import json
import polars as pl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from functools import partial

# ── Your existing source modules ─────────────────────────────────
import src.data_processing as data_proc
import src.metrics as metrics
import src.models.xgboost as xgb
import src.models.randomforest as rf
import src.models.mlp as mlp

# ── Scenario analysis modules ────────────────────────────────────
from scenario_analysis.feature_engineering import (
    build_opioid_panel_features,
    get_feature_columns,
    build_counterfactual_panel,
    build_cohort_lookup,
    build_counterfactual_panel_tiered,
)
from scenario_analysis.training import (
    national_counterfact_initial_training,
)
from scenario_analysis.counterfactual_prediction import (
    predict_counterfactual_cv_polars,
)
from scenario_analysis.sensitivity_analysis import (
    compute_dose_response,
    plot_dose_response,
    plot_dose_response_multi_model,
    plot_temporal_profile,
    compute_subgroup_shifts,
    plot_subgroup_shifts,
    # plot_feature_importance_by_year,
    load_risk_tier_lookup,
    compute_risk_tier_temporal_shifts,
    plot_risk_tier_temporal_profile,
    plot_risk_tier_delta_heatmap,
    plot_effect_vs_observed_scatter,
    plot_effect_vs_observed_scatter_by_year,
    plot_mortality_decomposition,
)
from scenario_analysis.bootstrap_counterfactual import (
    bootstrap_counterfactual_inference,
)

# ── Model registry (calls your src.models configs) ───────────────
MODEL_REGISTRY = {
    "xgboost": xgb.get_model,
    "random_forest": rf.get_model,
    "mlp": mlp.get_model,
}


def _get_model(model_name: str):
    """Instantiate a model from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _load_data() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Load raw data, build enhanced features, return both plus feature list."""
    data = data_proc.CountyDataLoader()
    df_raw = data.load()

    df_enhanced = build_opioid_panel_features(df_raw)
    feature_cols = get_feature_columns(
        df_enhanced,
        exclude_extra=["urbanicity_class"],
    )

    print(f"Loaded {df_raw.shape[0]} raw rows -> {df_enhanced.shape[0]} enhanced rows, {len(feature_cols)} features")
    return df_raw, df_enhanced, feature_cols


def _load_predictions(run_dir: Path) -> pl.DataFrame:
    """Load baseline predictions from a training run, with FIPS as zero-padded string."""
    return (
        pl.read_csv(run_dir / "baseline_predictions.csv")
        .with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
    )


def _get_models_dir(run_dir: Path) -> Path:
    """Get the models subdirectory from a training run."""
    models_dir = run_dir / "training" / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    return models_dir


def _get_feature_cols(run_dir: Path) -> list[str]:
    """Load feature column list from a training run."""
    with open(run_dir / "feature_cols.json") as f:
        return json.load(f)


def _run_scenario_predictions(
    df_raw: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    risk_scores: pl.DataFrame | None,
    reduction_type: str,
    models_dir: Path,
    feature_cols: list[str],
    target_col: str,
    reductions: list[float],
    output_dir: Path,
) -> dict[float, pl.DataFrame]:
    """
    Generate scenario predictions for one variable at multiple reduction levels.
    Returns dict mapping reduction -> predictions DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if reduction_type == "uniform":
        print(f"Running uniform reduction scenarios for {target_col}...")
        for reduction in reductions:
            print(f"  {target_col} reduction={reduction:.0%}")

            df_scenario = build_counterfactual_panel(
                df_raw, treatment_col=target_col, reduction=reduction,
            )

            preds = predict_counterfactual_cv_polars(
                df_modified=df_scenario,
                baseline_predictions=baseline_predictions,
                model_dir=models_dir,
                feature_cols=feature_cols,
            )

            save_path = output_dir / f"{target_col}_r{reduction:.2f}.csv"
            preds.write_csv(save_path)
            results[reduction] = preds
            
    # elif reduction_type == "risk_tiered":
    #     print(f"Running risk-tiered reduction scenarios for {target_col}...")
    #     # For risk-tiered, we can define custom reductions per county based on risk scores
    #     # Here we just use the same function but with a different df_scenario builder that applies tiered reductions
    #     for reduction in reductions:
    #         print(f"  {target_col} reduction={reduction:.0%} (risk-tiered)")

    #         # RX_rate cohort_lookup template
    #         # cohort_lookup, summary_info = build_cohort_lookup(
    #         #     risk_scores=risk_scores,
    #         #     risk_col="AbsError_Risk",
    #         #     modifier_type="multiplier",
    #         #     base_value=0.80,              # High-risk gets 20% reduction
    #         #     quantiles=(0.50, 0.90),
    #         #     delta=(0.10, 0.05),           # Low=0.90, Mid=0.85, High=0.80
    #         # )
            
    #         #Unempl/Unins cohort lookup template
    #         cohort_lookup, info = build_cohort_lookup(
    #             risk_scores=risk_scores,
    #             risk_col="AbsError_Risk",
    #             modifier_type="absolute_drop",
    #             base_value=1.00,              # High-risk gets 1.0-point drop
    #             quantiles=(0.50, 0.90),
    #             delta=(0.75, 0.50),           # Low=0.25, Mid=0.50, High=1.00
    #         )
            
    #         df_scenario = build_counterfactual_panel_tiered(
    #             df_raw, 
    #             cohort_lookup=cohort_lookup,
    #             treatment_col=target_col,
    #         )

    #         preds = predict_counterfactual_cv_polars(
    #             df_modified=df_scenario,
    #             baseline_predictions=baseline_predictions,
    #             model_dir=models_dir,
    #             feature_cols=feature_cols,
    #         )

    #         save_path = output_dir / f"{target_col}_r{reduction:.2f}_risk_tiered.csv"
    #         cohort_lookup.write_csv(output_dir / f"cohort_lookup_r{reduction:.2f}.csv")
    #         preds.write_csv(save_path)
    #         results[reduction] = preds
    
    elif reduction_type == "risk_tiered":
        print(f"Running risk-tiered reduction scenarios for {target_col}...")

        for reduction in reductions:
            print(f"  {target_col} reduction={reduction} (risk-tiered)")

            if target_col == "rx_rate":
                # reduction is proportional, so convert to multiplier
                high_risk_base = 1.0 - reduction

                cohort_lookup, info = build_cohort_lookup(
                    risk_scores=risk_scores,
                    risk_col="AbsError_Risk",
                    modifier_type="multiplier",
                    base_value=high_risk_base,
                    quantiles=(0.50, 0.90),
                    delta=(0.10, 0.05),
                )

            else:
                # reduction is an absolute drop
                high_risk_base = reduction

                cohort_lookup, info = build_cohort_lookup(
                    risk_scores=risk_scores,
                    risk_col="AbsError_Risk",
                    modifier_type="absolute_drop",
                    base_value=high_risk_base,
                    quantiles=(0.50, 0.90),
                    delta=(0.75, 0.50),
                )

            df_scenario = build_counterfactual_panel_tiered(
                df_raw,
                cohort_lookup=cohort_lookup,
                treatment_col=target_col,
            )

            preds = predict_counterfactual_cv_polars(
                df_modified=df_scenario,
                baseline_predictions=baseline_predictions,
                model_dir=models_dir,
                feature_cols=feature_cols,
            )

            save_path = output_dir / f"{target_col}_r{reduction:.2f}_risk_tiered.csv"
            cohort_lookup.write_csv(output_dir / f"cohort_lookup_r{reduction:.2f}.csv")
            preds.write_csv(save_path)
            results[reduction] = preds

    return results


# ═════════════════════════════════════════════════════════════════
# PIPELINE 1: Train from scratch + scenario predictions
# ═════════════════════════════════════════════════════════════════
def train_and_predict(
    model_name: str,
    target_col: str = "rx_rate",
    reduction_type: str = "uniform", # or "risk_tiered"
    reductions: list[float] | None = None,
    output_base: str = "model_outputs/scenario_analysis",
) -> Path:
    """
    Full pipeline: load data, train model, run scenario predictions.

    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY ("xgboost", "rf", "mlp").
    target_col : str
        Variable to modify in scenarios.
    reductions : list[float]
        Reduction levels to test. Default: [0.05, 0.10, 0.15, 0.20, 0.25].
    output_base : str
        Base directory for outputs.

    Returns
    -------
    Path to the run directory (for use with other pipeline functions).
    """
    if reductions is None:
        reductions = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\n{'='*60}")
    print(f"PIPELINE 1: Train + Predict ({model_name}, {target_col})")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────
    df_raw, df_enhanced, feature_cols = _load_data()

    # ── Create run directory ─────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_base) / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Train ────────────────────────────────────────────────────
    print(f"\nTraining {model_name}...")
    model = _get_model(model_name)

    metrics_df, fi_df, predictions_df, errors, save_dir = (
        national_counterfact_initial_training(
            df=df_enhanced,
            model=model,
            feature_cols=feature_cols,
            save_path=str(run_dir / "training"),
        )
    )

    baseline_predictions = pl.from_pandas(predictions_df).with_columns(
        pl.col("FIPS").cast(pl.Utf8).str.zfill(5)
    )
    models_dir = save_dir / "models" #type: ignore

    # Save enhanced data info for reproducibility
    baseline_predictions.write_csv(run_dir / "baseline_predictions.csv")
    with open(run_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    if not fi_df.empty:
        fi_df.to_csv(run_dir / "feature_importance.csv", index=False)
    # Saving risk scores, can be used in scenario predictions based on reduction_type
    risk_scores = metrics.compute_all_risk_scores(predictions_df, alpha=0.14) # 3/23/26, EB: alpha=0.14 was identified as best in scratch_scripts/ewma_alpha_sweep.py
    risk_scores_path = Path(save_dir) / "risk_scores.csv"
    risk_scores.to_csv(risk_scores_path, index=False)

    # ── Scenario predictions ─────────────────────────────────────
    if reduction_type == "uniform":
        print(f"\nRunning uniform reduction scenario predictions ({target_col})...")
        scenario_results = _run_scenario_predictions(
            df_raw, baseline_predictions, 
            risk_scores=None, # not needed for uniform reduction
            reduction_type=reduction_type,
            models_dir=models_dir, 
            feature_cols=feature_cols,
            target_col=target_col, reductions=reductions,
            output_dir=run_dir / "scenarios",
        )
        
    elif reduction_type == "risk_tiered":
        print(f"\nRunning risk-tiered reduction scenario predictions ({target_col})...")
        # For risk-tiered, we can define custom reductions per county based on risk scores
        # Here we just use the same function but with a different df_scenario builder that applies tiered reductions
        scenario_results = _run_scenario_predictions(
            df_raw, baseline_predictions, 
            risk_scores=pl.from_pandas(risk_scores), # needed for risk-tiered reduction
            reduction_type=reduction_type,
            models_dir=models_dir, 
            feature_cols=feature_cols,
            target_col=target_col, reductions=reductions,
            output_dir=run_dir / "scenarios",
        )

    print(f"\nPipeline 1 complete. Run directory: {run_dir}")
    return run_dir


# ═════════════════════════════════════════════════════════════════
# PIPELINE 2: Load existing training + scenario predictions
# ═════════════════════════════════════════════════════════════════
def predict_from_existing(
    training_run_dir: str | Path,
    target_col: str = "rx_rate",
    reductions: list[float] | None = None,
) -> dict[float, pl.DataFrame]:
    """
    Load a previous training run and generate scenario predictions.

    Parameters
    ----------
    training_run_dir : str or Path
        Path to the training run directory (contains models/, 
        predictions.csv, feature_cols.json).
    target_col : str
        Variable to modify.
    reductions : list[float]
        Reduction levels.

    Returns
    -------
    dict mapping reduction -> scenario predictions DataFrame.
    """
    if reductions is None:
        reductions = [0.05, 0.10, 0.15, 0.20, 0.25]

    training_run_dir = Path(training_run_dir)

    print(f"\n{'='*60}")
    print(f"PIPELINE 2: Predict from existing ({target_col})")
    print(f"  Training run: {training_run_dir}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────
    df_raw, _, _ = _load_data()

    # ── Load training artifacts ──────────────────────────────────
    baseline_predictions = _load_predictions(training_run_dir)
    models_dir = _get_models_dir(training_run_dir)
    feature_cols = _get_feature_cols(training_run_dir)

    print(f"  Loaded {len(feature_cols)} features, models from {models_dir}")

    # ── Scenario predictions ─────────────────────────────────────
    output_dir = training_run_dir / "scenarios"
    results = _run_scenario_predictions(
        df_raw, baseline_predictions, models_dir, feature_cols,
        target_col, reductions,
        output_dir=output_dir,
    )

    print(f"\nPipeline 2 complete. Scenarios saved to {output_dir}")
    return results


# ═════════════════════════════════════════════════════════════════
# PIPELINE 3: Load existing training + bootstrap
# ═════════════════════════════════════════════════════════════════
def bootstrap_from_existing(
    training_run_dir: str | Path,
    model_name: str,
    target_col: str = "rx_rate",
    reduction: float = 0.15,
    n_boot: int = 200,
):
    """
    Load a previous training run and run bootstrap inference.

    Parameters
    ----------
    training_run_dir : str or Path
        Path to the training run directory.
    model_name : str
        Key in MODEL_REGISTRY (needed to recreate the model template).
    target_col : str
        Variable to modify.
    reduction : float
        Single reduction level for bootstrapping.
    n_boot : int
        Number of bootstrap replicates.

    Returns
    -------
    BootstrapResult
    """
    training_run_dir = Path(training_run_dir)

    print(f"\n{'='*60}")
    print(f"PIPELINE 3: Bootstrap ({model_name}, {target_col}, {reduction:.0%})")
    print(f"  Training run: {training_run_dir}")
    print(f"  Replicates: {n_boot}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────
    df_raw, df_enhanced, _ = _load_data()
    feature_cols = _get_feature_cols(training_run_dir)

    # ── Run bootstrap ────────────────────────────────────────────
    model = _get_model(model_name)

    result = bootstrap_counterfactual_inference(
        df_raw=df_raw,
        df_enhanced=df_enhanced,
        model_template=model,
        feature_cols=feature_cols,
        reduction=reduction,
        build_features_fn=build_opioid_panel_features,
        build_counterfactual_fn=partial(
            build_counterfactual_panel, treatment_col=target_col,
        ),
        n_boot=n_boot,
        verbose=True,
    )

    # ── Save ─────────────────────────────────────────────────────
    boot_dir = training_run_dir / "bootstrap"
    boot_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(
        boot_dir / f"bootstrap_{target_col}_r{reduction:.2f}_{n_boot}reps.csv",
        index=False,
    )
    result.boot_estimates.to_csv(
        boot_dir / f"bootstrap_{target_col}_r{reduction:.2f}_{n_boot}reps_all.csv",
        index=False,
    )

    print(f"\nPipeline 3 complete. Results saved to {boot_dir}")
    return result


# ═════════════════════════════════════════════════════════════════
# PIPELINE 4: Analyze existing results
# ═════════════════════════════════════════════════════════════════
def analyze_results(
    training_run_dir: str | Path,
    model_name: str,
    target_col: str = "rx_rate",
    reduction_type: str = "uniform", # or "risk_tiered"
    reductions: list[float] | None = None,
    boot_summary_path: str | Path | None = None,
    plots_dir: str | Path | None = None,
):
    """
    Load scenario predictions and run sensitivity analyses.

    Parameters
    ----------
    training_run_dir : str or Path
        Path to the training run directory.
    model_name : str
        For labeling plots.
    target_col : str
        Variable that was modified.
    reduction_type : str
        "uniform" or "risk_tiered", affects how scenarios were generated.
    reductions : list[float]
        Reduction levels that were run (to load the right files).
    boot_summary_path : str or Path or None
        Path to bootstrap summary CSV (for temporal profile CI bands).
    plots_dir : str or Path or None
        Where to save plots. Default: training_run_dir/plots/
    """
    if reductions is None:
        reductions = [0.05, 0.10, 0.15, 0.20, 0.25]

    training_run_dir = Path(training_run_dir)
    scenarios_dir = training_run_dir / "scenarios"

    if plots_dir is None:
        plots_dir = training_run_dir / "plots"
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PIPELINE 4: Analyze results ({model_name}, {target_col})")
    print(f"  Training run: {training_run_dir}")
    print(f"{'='*60}")

    # ── Load data and artifacts ──────────────────────────────────
    df_raw, df_enhanced, _ = _load_data()
    baseline_predictions = _load_predictions(training_run_dir)
    feature_cols = _get_feature_cols(training_run_dir)
    models_dir = _get_models_dir(training_run_dir)

    # ── Load scenario predictions ────────────────────────────────
    scenario_preds = {}
    for reduction in reductions:
        if reduction_type == "risk_tiered":
            # risk-tiered temporal plots and CSVs
            cohort = load_risk_tier_lookup(training_run_dir, reduction)
            merged_county_df, summary_df = compute_risk_tier_temporal_shifts(
                baseline_predictions, scenario_preds[reduction], cohort
            )
            fig = plot_risk_tier_temporal_profile(
                summary_df, reduction, target_col, model_name
            )
            out_png = Path(plots_dir) / f"risk_tier_temporal_{target_col}_r{reduction:.2f}.png"
            fig.savefig(out_png, dpi=300)
            out_csv = Path(plots_dir) / f"risk_tier_temporal_{target_col}_r{reduction:.2f}.csv"
            # save summary as CSV
            summary_df.to_pandas().to_csv(out_csv, index=False)
                    
        elif reduction_type == "uniform":
            csv_path = scenarios_dir / f"{target_col}_r{reduction:.2f}.csv"
            if csv_path.exists():
                scenario_preds[reduction] = (
                    pl.read_csv(csv_path)
                    .with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
                )
                print(f"  Loaded scenario: {target_col} r={reduction:.0%}")
            else:
                print(f"  Warning: not found {csv_path}")

    if not scenario_preds:
        print("No scenario predictions found. Run predict_from_existing first.")
        return

    # ── Load bootstrap summary if available ──────────────────────
    boot_summary = None
    if boot_summary_path is not None:
        boot_summary = pd.read_csv(boot_summary_path)
        print(f"  Loaded bootstrap summary from {boot_summary_path}")

    # ── 4a. Dose-response ────────────────────────────────────────
    print("\n  Dose-response curves...")
    dr_df = compute_dose_response(
        df_raw=df_raw,
        baseline_predictions=baseline_predictions,
        model_dir=models_dir,
        feature_cols=feature_cols,
        reductions=reductions,
        build_counterfactual_fn=partial(
            build_counterfactual_panel, treatment_col=target_col,
        ),
        predict_counterfactual_fn=predict_counterfactual_cv_polars,
        model_name=model_name,
    )
    fig = plot_dose_response(
        dr_df, 
        target_col=target_col,
        # title=f"Dose-Response: {target_col} ({model_name})",
        model_name=model_name,
    )
    fig.savefig(plots_dir / f"dose_response_{target_col}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    dr_df.to_csv(plots_dir / f"dose_response_{target_col}.csv", index=False)

    # ── 4b. Temporal profiles ────────────────────────────────────
    print("\n  Temporal profiles...")
    for reduction, preds in scenario_preds.items():
        fig = plot_temporal_profile(
            baseline_predictions, preds, df_raw,
            reduction=reduction,
            model_name=model_name,
            boot_summary=boot_summary if boot_summary is not None else None,
        )
        fig.savefig(
            plots_dir / f"temporal_{target_col}_r{reduction:.2f}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # ── 4c. Subgroup analysis ────────────────────────────────────
    print("\n  Subgroup analysis...")
    # Use a middle reduction level
    mid_reduction = reductions[len(reductions) // 2]
    if mid_reduction in scenario_preds:
        sg_df = compute_subgroup_shifts(
            baseline_predictions,
            scenario_preds[mid_reduction],
            df_enhanced,
            group_col="urbanicity_class",
            reduction=mid_reduction,
            model_name=model_name,
        )
        fig = plot_subgroup_shifts(sg_df)
        fig.savefig(
            plots_dir / f"subgroup_{target_col}_r{mid_reduction:.2f}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        sg_df.to_csv(
            plots_dir / f"subgroup_{target_col}_r{mid_reduction:.2f}.csv",
            index=False,
        )

    # # ── 4d. Feature importance ───────────────────────────────────
    # fi_path = training_run_dir / "feature_importance.csv"
    # if fi_path.exists():
    #     print("\n  Feature importance...")
    #     fi_df = pd.read_csv(fi_path)
    #     fig = plot_feature_importance_by_year(fi_df, top_n=15)
    #     fig.savefig(plots_dir / f"feature_importance.png",
    #                 dpi=150, bbox_inches="tight")
    #     plt.close(fig)

    # print(f"\nPipeline 4 complete. Plots saved to {plots_dir}")
    return dr_df, scenario_preds


def analyze_results_v2(
    training_run_dir: str | Path,
    model_name: str,
    target_col: str = "rx_rate",
    reduction_type: str = "uniform",  # "uniform" or "risk_tiered"
    reductions: list[float] | None = None,
    boot_summary_path: str | Path | None = None,
    plots_dir: str | Path | None = None,
):
    if reductions is None:
        reductions = [0.05, 0.10, 0.15, 0.20, 0.25]

    training_run_dir = Path(training_run_dir)
    scenarios_dir = training_run_dir / "scenarios"

    if plots_dir is None:
        plots_dir = training_run_dir / "plots"
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PIPELINE 4: Analyze results ({model_name}, {target_col}, {reduction_type})")
    print(f"  Training run: {training_run_dir}")
    print(f"{'='*60}")

    # ── Load shared artifacts ─────────────────────────────────────
    df_raw, df_enhanced, _ = _load_data()
    baseline_predictions = _load_predictions(training_run_dir)
    feature_cols = _get_feature_cols(training_run_dir)
    models_dir = _get_models_dir(training_run_dir)

    # ── Load bootstrap summary if available ───────────────────────
    boot_summary = None
    if boot_summary_path is not None:
        boot_summary = pd.read_csv(boot_summary_path)
        print(f"  Loaded bootstrap summary from {boot_summary_path}")

    # ============================================================
    # UNIFORM-REDUCTION WORKFLOW
    # ============================================================
    if reduction_type == "uniform":
        scenario_preds = {}

        for reduction in reductions:
            csv_path = scenarios_dir / f"{target_col}_r{reduction:.2f}.csv"
            if csv_path.exists():
                scenario_preds[reduction] = (
                    pl.read_csv(csv_path)
                    .with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
                )
                print(f"  Loaded scenario: {target_col} r={reduction:.0%}")
            else:
                print(f"  Warning: not found {csv_path}")

        if not scenario_preds:
            print("No uniform scenario predictions found. Run predict_from_existing first.")
            return

        # ── 4a. Dose-response ────────────────────────────────────
        print("\n  Dose-response curves...")
        dr_df = compute_dose_response(
            df_raw=df_raw,
            baseline_predictions=baseline_predictions,
            model_dir=models_dir,
            feature_cols=feature_cols,
            reductions=reductions,
            build_counterfactual_fn=partial(
                build_counterfactual_panel, treatment_col=target_col,
            ),
            predict_counterfactual_fn=predict_counterfactual_cv_polars,
            model_name=model_name,
        )
        fig = plot_dose_response(
            dr_df,
            target_col=target_col,
            model_name=model_name,
        )
        fig.savefig(
            plots_dir / f"dose_response_{target_col}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        dr_df.to_csv(plots_dir / f"dose_response_{target_col}.csv", index=False)

        # ── 4b. Temporal profiles ────────────────────────────────
        print("\n  Temporal profiles...")
        for reduction, preds in scenario_preds.items():
            fig = plot_temporal_profile(
                baseline_predictions,
                preds,
                df_raw,
                reduction=reduction,
                model_name=model_name,
                boot_summary=boot_summary,
            )
            fig.savefig(
                plots_dir / f"temporal_{target_col}_r{reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        # ── 4c. Subgroup analysis ────────────────────────────────
        print("\n  Subgroup analysis...")
        mid_reduction = reductions[len(reductions) // 2]
        if mid_reduction in scenario_preds:
            sg_df = compute_subgroup_shifts(
                baseline_predictions,
                scenario_preds[mid_reduction],
                df_enhanced,
                group_col="urbanicity_class",
                reduction=mid_reduction,
                model_name=model_name,
            )
            fig = plot_subgroup_shifts(sg_df)
            fig.savefig(
                plots_dir / f"subgroup_{target_col}_r{mid_reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            sg_df.to_csv(
                plots_dir / f"subgroup_{target_col}_r{mid_reduction:.2f}.csv",
                index=False,
            )

        return dr_df, scenario_preds

    # ============================================================
    # RISK-TIERED WORKFLOW
    # ============================================================
    elif reduction_type == "risk_tiered":
        scenario_preds = {}

        for reduction in reductions:
            csv_path = scenarios_dir / f"{target_col}_r{reduction:.2f}_risk_tiered.csv"
            if csv_path.exists():
                scenario_preds[reduction] = (
                    pl.read_csv(csv_path)
                    .with_columns(pl.col("FIPS").cast(pl.Utf8).str.zfill(5))
                )
                print(f"  Loaded risk-tiered scenario: {target_col} r={reduction:.0%}")
            else:
                print(f"  Warning: not found {csv_path}")

        if not scenario_preds:
            print("No risk-tiered scenario predictions found.")
            return

        print("\n  Risk-tier temporal profiles...")
        for reduction, preds in scenario_preds.items():
            cohort_lookup = load_risk_tier_lookup(training_run_dir, reduction)

            merged_county_df, summary_df = compute_risk_tier_temporal_shifts(
                baseline_predictions,
                preds,
                cohort_lookup,
            )

            fig = plot_risk_tier_temporal_profile(
                summary_df,
                reduction=reduction,
                target_col=target_col,
                model_name=model_name,
                center_col="mean_delta",
                cohort_order=["Low", "Mid", "High"],  # adjust if needed
                
            )
            fig.savefig(
                plots_dir / f"risk_tier_temporal_{target_col}_r{reduction:.2f}_bars.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            
            fig = plot_risk_tier_delta_heatmap(
                merged_county_df,
                reduction=reduction,
                target_col=target_col,
                model_name=model_name,
                cohort_order=["Low", "Mid", "High"],
                agg_func="mean",
            )
            fig.savefig(
                plots_dir / f"risk_tier_heatmap_{target_col}_r{reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


            # Overall scatter: relative change
            fig = plot_effect_vs_observed_scatter(
                merged_county_df,
                reduction=reduction,
                target_col=target_col,
                model_name=model_name,
                effect_scale="relative",
                cohort_order=["Low", "Mid", "High"],
            )
            fig.savefig(
                plots_dir / f"risk_tier_scatter_relative_{target_col}_r{reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Yearly faceted scatter: absolute change
            fig = plot_effect_vs_observed_scatter_by_year(
                merged_county_df,
                reduction=reduction,
                target_col=target_col,
                model_name=model_name,
                effect_scale="absolute",
                cohort_order=["Low", "Mid", "High"],
            )
            fig.savefig(
                plots_dir / f"risk_tier_scatter_by_year_{target_col}_r{reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Yearly faceted scatter: relative change
            fig = plot_effect_vs_observed_scatter_by_year(
                merged_county_df,
                reduction=reduction,
                target_col=target_col,
                model_name=model_name,
                effect_scale="relative",
                cohort_order=["Low", "Mid", "High"],
            )
            fig.savefig(
                plots_dir / f"risk_tier_scatter_by_year_relative_{target_col}_r{reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            
            # Single year Mortality decomposition plot (e.g. 2022)
            fig, decomp = plot_mortality_decomposition(
                merged_county_df,
                year=2022,
                reduction=0.10,
                target_col=target_col,
                model_name="XGBoost",
                cohort_order=["Low", "Mid", "High"],
            )
            fig.savefig(
                plots_dir / f"risk_tier_decomposition_{target_col}_r{reduction:.2f}.png",
                dpi=150,
                bbox_inches="tight",
            )

            summary_df.write_csv(
                plots_dir / f"risk_tier_temporal_{target_col}_r{reduction:.2f}.csv"
            )
            merged_county_df.write_csv(
                plots_dir / f"risk_tier_county_{target_col}_r{reduction:.2f}.csv"
            )

        return None, scenario_preds

    else:
        raise ValueError("reduction_type must be 'uniform' or 'risk_tiered'")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════
def parse_args():
    """Build and return the argument parser for the scenario analysis CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Scenario analysis pipeline")
    subparsers = parser.add_subparsers(dest="command")

    model_choices = list(MODEL_REGISTRY.keys())

    # ── train ────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train + predict")
    p_train.add_argument("--model", default="xgboost", choices=model_choices)
    p_train.add_argument("--target", default="rx_rate")
    p_train.add_argument("--reduction_type", default="uniform", choices=["uniform", "risk_tiered"])
    p_train.add_argument("--reductions", nargs="+", type=float,
                         default=[0.05, 0.10, 0.15, 0.20, 0.25])

    # ── predict ──────────────────────────────────────────────────
    p_pred = subparsers.add_parser("predict", help="Predict from existing training")
    p_pred.add_argument("--run-dir", required=True)
    p_pred.add_argument("--target", default="rx_rate")
    p_pred.add_argument("--reductions", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.25])

    # ── bootstrap ────────────────────────────────────────────────
    p_boot = subparsers.add_parser("bootstrap", help="Bootstrap from existing training")
    p_boot.add_argument("--run-dir", required=True)
    p_boot.add_argument("--model", default="xgboost", choices=model_choices)
    p_boot.add_argument("--target", default="rx_rate")
    p_boot.add_argument("--reduction", type=float, default=0.15)
    p_boot.add_argument("--n-boot", type=int, default=200)

    # ── analyze ──────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Analyze existing results")
    p_analyze.add_argument("--run-dir", required=True)
    p_analyze.add_argument("--model", default="xgboost", choices=model_choices)
    p_analyze.add_argument("--target", default="rx_rate")
    p_analyze.add_argument("--reduction_type", default="uniform", choices=["uniform", "risk_tiered"])
    p_analyze.add_argument("--reductions", nargs="+", type=float,
                           default=[0.05, 0.10, 0.15, 0.20, 0.25])
    p_analyze.add_argument("--boot-summary", default=None,
                           help="Path to bootstrap summary CSV")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        train_and_predict(args.model, args.target, args.reduction_type, args.reductions)

    elif args.command == "predict":
        predict_from_existing(args.run_dir, args.target, args.reductions)

    elif args.command == "bootstrap":
        bootstrap_from_existing(
            args.run_dir, args.model, args.target,
            args.reduction, args.n_boot,
        )

    elif args.command == "analyze":
        analyze_results_v2(
            args.run_dir, args.model, args.target,
            args.reduction_type,args.reductions,
            args.boot_summary,
        )

    else:
        print("No command specified. Use --help for usage.")


if __name__ == "__main__":
    main()