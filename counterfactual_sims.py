### 11/12/25, EB: Here I am running counterfactual simulations to see how changing the RX dispensing rates of high-risk counties affects mortality predictions.

"""
counterfactual_sims.py
----------------------

Run counterfactual experiments on the opioid mortality model outputs.

Workflow:
1. Load saved model predictions + risk scores from `model_outputs/`.
2. Identify high-risk counties (e.g., top 5% by AbsError_Risk).
3. Apply simulated interventions (e.g., reduce prescription rates by 10%).
4. Re-run model predictions with modified data.
5. Compare baseline vs counterfactual mortality predictions.

Usage:
    uv run python counterfactual_sims.py --model xgboost --adjust 0.9 --top 0.05 
"""

from pathlib import Path
import pandas as pd
import polars as pl
import argparse
import src.data_processing as data_proc
import src.model_training as train
import src.metrics as metrics
import src.visualizations as viz
import src.models.xgboost as xgb
import src.models.randomforest as rf
import src.models.mlp as mlp
import src.counterfactual as cf
from datetime import datetime

# ---------------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Run counterfactual simulation on model outputs.")

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "mlp"],
        help="Model to use for counterfactual simulation."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="model_outputs",
        help="Base directory where model_outputs/<model>/<timestamp>/ live."
    )

    parser.add_argument(
        "--rx_adjust",
        type=float,
        default=0.9,
        help="Multiplicative adjustment factor for prescription rate (e.g., 0.9 = 10%% reduction)."
    )

    parser.add_argument(
        "--top_risk",
        type=float,
        default=0.05,
        help="Fraction of counties to select as high-risk (e.g., 0.05 = top 5%%)."
    )

    parser.add_argument(
        "--n_counties",
        type=int,
        default=2,
        help="Number of high-risk counties to select for intervention."
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate comparison plots for counterfactual results."
    )

    # National-level counterfactual flags below
    parser.add_argument(
        "--national",
        action="store_true",
        help="Apply national-level counterfactual (RX adjustment for ALL counties, based on risk)."
    )
    
    # Choose intervention rule
    parser.add_argument(
        "--rx_rule",
        type=str,
        default="linear",
        choices=["linear", "piecewise"],
        help="RX intervention strategy: 'linear' or 'piecewise'."
    )

    # Piecewise-only parameters (tier thresholds on risk_norm in [0,1])
    parser.add_argument(
        "--tier_s",
        type=float,
        default=0.50,
        help="Piecewise threshold s on risk_norm (low vs mid). Used only if --rx_rule piecewise."
    )
    parser.add_argument(
        "--tier_t",
        type=float,
        default=0.90,
        help="Piecewise threshold t on risk_norm (mid vs high). Used only if --rx_rule piecewise."
    )

    # How to derive alpha/beta from lambda (simple + explainable defaults)
    parser.add_argument(
        "--alpha_delta",
        type=float,
        default=0.05,
        help="If alpha not provided, set alpha = min(1, lambda + alpha_delta). Used only if --rx_rule piecewise."
    )
    parser.add_argument(
        "--beta_delta",
        type=float,
        default=0.10,
        help="If beta not provided, set beta = min(1, lambda + beta_delta). Used only if --rx_rule piecewise."
    )

    # OPTIONAL: allow explicit alpha/beta overrides (recommended to include)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional explicit mid-tier RX multiplier. Used only if --rx_rule piecewise."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Optional explicit low-tier RX multiplier. Used only if --rx_rule piecewise."
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# MODEL REGISTRY (same as in main.py)
# ---------------------------------------------------------------------
MODEL_REGISTRY = {
    "xgboost": xgb.get_model,
    "random_forest": rf.get_model,
    "mlp": mlp.get_model,
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_latest_run_dir(base_dir: str, model_name: str) -> Path:
    """
    Return the most recently modified run directory for a given model.
    Expects structure:
        base_dir/
        └─ <model>/
            └─ <timestamp>/  # e.g., 2024-11-01_15-30-00/
                ├─ predictions.csv
                ├─ risk_scores.csv

    ### 11/21/25, EB: Realized that there's a small discrepancy in the user input for model name vs folder name, so I made sure these match up correctly.
    ### There's probably a cleaner way to do this later, but BA is refactoring the entire dataloader and model training pipeline, so I'll leave this for now.
    """
    MODEL_FOLDER_MAP = {
        "xgboost": "xgbregressor",
        "random_forest": "randomforestregressor",
        "mlp": "mlpregressor",
    }
    
    if model_name not in MODEL_FOLDER_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Expected one of: {list(MODEL_FOLDER_MAP.keys())}"
        )
    
    folder_name = MODEL_FOLDER_MAP[model_name]
        
    model_dir = Path(base_dir) / folder_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    run_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped run directories found in {model_dir}")
    
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Latest run directory detected for {model_name}: {latest}")
    return latest


def create_new_run_dir(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        Path(args.base_dir)
        / "national_counterfactual"
        / args.model
        / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_shared_inputs(args):
    """
    Load base dataset, baseline predictions, and risk scores.
    Shared by both the case-study branch and the national branch.
    """
    data = data_proc.CountyDataLoader()
    df = data.load()
    
    run_dir = get_latest_run_dir(args.base_dir, args.model)
    print(f"Using baseline results from: {run_dir}")
    
    preds_path = run_dir / "predictions.csv"
    risk_path = run_dir / "risk_scores.csv"
    
    if not preds_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found at {preds_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Risk scores not found at {risk_path}")
    
    baseline_predictions = pl.read_csv(preds_path)
    risk_scores = pl.read_csv(risk_path)
    
    return df, baseline_predictions, risk_scores, run_dir


# ---------------------------------------------------------------------
# CASE-STUDY COUNTERFACTUALS PIPELINE
# ---------------------------------------------------------------------
def run_case_study_counterfactual(args, df, baseline_predictions, risk_scores, run_dir):
    """
    Run counterfactual simulations for few selected high-risk counties.
    """
    
    # Select top high-risk counties
    high_risk = cf.get_high_risk_counties(
        risk_scores,
        top_frac=args.top_risk,
        error_col='AbsError_Risk',
        )

    # Choose 2 counties from the high-risk set for intervention
    target_fips = cf.pick_counterfactual_counties(high_risk, n=args.n_counties)
    # target_fips = ['21031', '51690']  # Example FIPS codes for testing
    
    # Save selected counties for record-keeping
    (run_dir / "counterfactual_targets.txt").write_text(
        "Selected counterfactual counties:\n" + "\n".join(target_fips)
    )
    
    # Apply intervention (reduce rx_rate)
    df_modified = cf.apply_rx_reduction(df=df, 
                                     target_fips=target_fips,
                                     adjust=args.rx_adjust
                                     )

    # --- Re-run model on modified data ---
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY.keys())}")

    # Initialize model
    model = MODEL_REGISTRY[args.model]()

    # Run model again on modified data
    ###---*** QUICK FIX, MAKE SAVING LOGOC MORE ROBUST LATER ***---###
    cf_dir = run_dir / "counterfact"
    cf_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running counterfactual model training with modified prescription rates...")
    (
        counterfact_model_metrics, 
        counterfact_feature_importances, 
        counterfact_predictions, 
        counterfact_all_errors, 
        counterfact_save_dir
    ) = train.yearly_mortality_prediction_polars(
        df_modified, 
        model,
        save_path=str(cf_dir)
    ) #f"{args.base_dir}/{args.model}")


    # Compare results
    if args.plot:
        counterfact_predictions = pl.from_pandas(counterfact_predictions) ### FIX LATER: keep as polars if possible
        cf.compare_predictions_and_save(actual_df=df,
                                     baseline_pred=baseline_predictions,
                                     cf_pred=counterfact_predictions,
                                     target_counties=target_fips,
                                     out_dir=counterfact_save_dir)

    print("✅ Case-study counterfactual simulation complete.")


# ---------------------------------------------------------------------
# NATIONAL COUNTERFACTUALS PIPELINE
# ---------------------------------------------------------------------
# def run_national_counterfactual(args, df, baseline_predictions, risk_scores, run_dir):
    
#     print("Running national counterfactual simulation: adjusting RX rates for all counties based on risk scores.")
    
#     df_modified = cf.apply_risk_based_rx_modifier(df=df,
#                                                   risk_scores=risk_scores,
#                                                   rx_col='rx_rate',
#                                                   risk_col='AbsError_Risk',
#                                                   lambda_val=args.rx_adjust,
#                                                   rule='linear'
#                                                   )

#     if args.model not in MODEL_REGISTRY:
#         raise ValueError(f"Unknown model: {args.model}")
    
#     model = MODEL_REGISTRY[args.model]()
#     cf_dir = run_dir / "counterfact_national"
#     cf_dir.mkdir(parents=True, exist_ok=True)

#     print("Running counterfactual model training with modified prescription rates (national)...")
#     (
#         counterfact_model_metrics,
#         counterfact_feature_importances,
#         counterfact_predictions,
#         counterfact_all_errors,
#         counterfact_save_dir,
#     ) = train.yearly_mortality_prediction_polars(
#         df_modified,
#         model,
#         save_path=str(cf_dir)
#     )

#     # Compute national mortality summaries (before vs. after)
#     nat_summary = metrics.compute_national_summary(
#         baseline_predictions=baseline_predictions,
#         counterfact_predictions=pl.from_pandas(counterfact_predictions),
#         df=df
#     )

#     pl.DataFrame(nat_summary).write_csv(cf_dir / "national_summary.csv")
#     print("Saved national summary to:", cf_dir / "national_summary.csv")

#     print("National counterfactual complete.")

def run_national_counterfactual(args, df, run_dir):
    """
    Full NATIONAL counterfactual pipeline.
    This version performs BOTH:
        (1) Baseline model training (including saving fold models)
        (2) Counterfactual prediction using saved fold models

    No external prior run is needed.
    """

    print("\n=== NATIONAL COUNTERFACTUAL SIMULATION (SELF-CONTAINED) ===")

    # ---------------------------------------------------------------
    # STEP 1. Train baseline models and save fold-models + predictions
    # ---------------------------------------------------------------

    print("\nTraining baseline models (K-fold) and saving all required artifacts...")
    model = MODEL_REGISTRY[args.model]()

    cf_train_dir = run_dir / "national_baseline"
    cf_train_dir.mkdir(parents=True, exist_ok=True)

    (
        baseline_metrics,
        baseline_feature_importances,
        baseline_predictions_df,
        baseline_errors,
        baseline_save_dir,
    ) = train.national_counterfact_initial_training(
        df=df,
        model=model,
        save_path=str(cf_train_dir),
    )

    if baseline_save_dir is None:
        raise RuntimeError("baseline_save_dir is None — baseline training did not save outputs.")

    # # Reload as Polars for uniformity
    # baseline_predictions = pl.read_csv(baseline_save_dir / "predictions.csv")

    print("\nBaseline model training complete.")
    print("Baseline predictions stored at:", baseline_save_dir)
    
    # ---------------------------------------------------------------
    # STEP 1b. Compute risk scores from baseline predictions
    # ---------------------------------------------------------------

    print("\nComputing risk scores from baseline predictions...")

    baseline_predictions = pl.from_pandas(baseline_predictions_df)

    risk_scores_df = metrics.compute_all_risk_scores(
        baseline_predictions_df
    )

    risk_scores = pl.from_pandas(risk_scores_df)

    risk_scores_path = baseline_save_dir / "risk_scores.csv"
    risk_scores_df.to_csv(risk_scores_path, index=False)

    print("Saved baseline risk scores to:", risk_scores_path)

    # ---------------------------------------------------------------
    # STEP 2. Load feature columns from the training run
    # ---------------------------------------------------------------

    feature_cols_path = baseline_save_dir / "feature_cols.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(
            "ERROR: feature_cols.json was not found in training directory.\n"
            "This file is required for counterfactual prediction."
        )

    import json
    feature_cols = json.load(open(feature_cols_path, "r"))
    models_dir = baseline_save_dir / "models"

    if not models_dir.exists():
        raise FileNotFoundError(
            "ERROR: models/ directory missing — baseline training did not save fold models."
        )

    # ---------------------------------------------------------------
    # STEP 3. Apply nationwide RX modification
    # ---------------------------------------------------------------

    print("\nModifying RX rates across all counties...")
    # df_modified = cf.apply_risk_based_rx_modifier(
    #     df=df,
    #     risk_scores=risk_scores,
    #     rx_col="rx_rate",
    #     risk_col="AbsError_Risk",
    #     lambda_val=args.rx_adjust,
    #     rule="linear",
    # )
    df_modified = cf.apply_risk_based_rx_modifier(
        df=df,
        risk_scores=risk_scores,
        rx_col="rx_rate",
        risk_col="AbsError_Risk",
        lambda_val=args.rx_adjust,
        rule=args.rx_rule,
        s=args.tier_s,
        t=args.tier_t,
        alpha=args.alpha,
        beta=args.beta,
        alpha_delta=args.alpha_delta,
        beta_delta=args.beta_delta,
    )

    # ---------------------------------------------------------------
    # STEP 4. Predict counterfactual mortality (NO retraining)
    # ---------------------------------------------------------------

    print("\nPredicting counterfactual outcomes (CV-consistent, no retraining)...")

    cf_predictions = cf.predict_counterfactual_cv_polars(
        df_modified=df_modified,
        baseline_predictions=baseline_predictions,
        model_dir=models_dir,
        feature_cols=feature_cols,
    )

    cf_out_dir = run_dir / "counterfact_national"
    cf_out_dir.mkdir(parents=True, exist_ok=True)

    cf_predictions_path = cf_out_dir / "counterfactual_predictions.csv"
    cf_predictions.write_csv(cf_predictions_path)

    print("Saved counterfactual predictions to:", cf_predictions_path)

    # ---------------------------------------------------------------
    # STEP 5. Compute national mortality summary
    # ---------------------------------------------------------------

    print("\nComputing national mortality summary...")

    nat_summary = cf.compute_national_prediction_comparison(
        baseline_predictions=baseline_predictions,
        counterfact_predictions=cf_predictions
    )

    nat_summary_path = cf_out_dir / "national_counterfact_summary.csv"
    pl.DataFrame(nat_summary).write_csv(nat_summary_path)

    print("Saved national summary to:", nat_summary_path)
    
    cf.plot_national_prediction_comparison(
        summary_df=nat_summary,
        df = df
    )

    print("\n=== NATIONAL COUNTERFACTUAL COMPLETE ===")

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def original_case_study_main():
    args = get_args()

    # Load the base dataset
    data = data_proc.CountyDataLoader()
    df = data.load()

    # Load latest model outputs (baseline predictions + risk scores)
    run_dir = get_latest_run_dir(args.base_dir, args.model)
    print(f"📂 Using baseline results from: {run_dir}")

    preds_path = run_dir / "predictions.csv"
    risk_path = run_dir / "risk_scores.csv"

    if not preds_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found at {preds_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Risk scores not found at {risk_path} — "
                                f"make sure main.py saves risk_scores.csv")

    # risk_scores = pd.read_csv(risk_path, dtype={'FIPS': str})
    # baseline_predictions = pd.read_csv(preds_path, dtype={'FIPS': str})
    ### 11/21/25, EB: Refactored to use polars for consistency, keeping above until I can confirm this all works.
    risk_scores = pl.read_csv(risk_path)
    baseline_predictions = pl.read_csv(preds_path)


    # Select top high-risk counties
    high_risk = cf.get_high_risk_counties(
        risk_scores,
        top_frac=args.top_risk,
        error_col='AbsError_Risk',
        )

    # Choose 2 counties from the high-risk set for intervention
    # target_fips = cf.pick_counterfactual_counties(high_risk, n=args.n_counties)
    ###12/3/25, EB: Testing particular counterfactual counties for consistency
    target_fips = ['21031', '51690']  # Example FIPS codes for testing
    
    
    # Save selected counties for record-keeping
    (run_dir / "counterfactual_targets.txt").write_text(
        "Selected counterfactual counties:\n" + "\n".join(target_fips)
    )

    
    # Apply intervention (reduce rx_rate)
    df_modified = cf.apply_rx_reduction(df=df, 
                                     target_fips=target_fips,
                                     adjust=args.rx_adjust
                                     )

    # --- Re-run model on modified data ---
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY.keys())}")

    # Initialize model
    model = MODEL_REGISTRY[args.model]()

    # Run model again on modified data
    ###---*** QUICK FIX, MAKE SAVING LOGOC MORE ROBUST LATER ***---###
    cf_dir = run_dir / "counterfact"
    cf_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running counterfactual model training with modified prescription rates...")
    (
        counterfact_model_metrics, 
        counterfact_feature_importances, 
        counterfact_predictions, 
        counterfact_all_errors, 
        counterfact_save_dir
    ) = train.yearly_mortality_prediction_polars(
        df_modified, 
        model,
        save_path=str(cf_dir)
    ) #f"{args.base_dir}/{args.model}")


    # Compare results
    if args.plot:
        # cf_save_dir_path = Path(counterfact_save_dir)
        # compare_dir = cf_dir / counterfact_save_dir.name
        
        # Create timestamped folder inside fixed counterfactual_comparison
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # out_dir = Path("counterfactual_comparison") / timestamp
        # out_dir.mkdir(parents=True, exist_ok=True)
        
        # df = df.to_pandas() ### FIX LATER: keep as polars if possible
        counterfact_predictions = pl.from_pandas(counterfact_predictions) ### FIX LATER: keep as polars if possible
        cf.compare_predictions_and_save(actual_df=df,
                                     baseline_pred=baseline_predictions,
                                     cf_pred=counterfact_predictions,
                                     target_counties=target_fips,
                                     out_dir=counterfact_save_dir)

    print("✅ Counterfactual simulation complete.")


# def main():
#     args = get_args()
    
#     df, baseline_predictions, risk_scores, run_dir = load_shared_inputs(args)
    
#     if args.national:
#         run_national_counterfactual(args, df, run_dir)
#     else:
#         run_case_study_counterfactual(args, df, baseline_predictions, risk_scores, run_dir)


def main():
    args = get_args()

    if args.national:
        # NATIONAL: self-contained pipeline
        data = data_proc.CountyDataLoader()
        df = data.load()

        run_dir = create_new_run_dir(args)
        run_national_counterfactual(args, df, run_dir)

    else:
        # CASE STUDY: reuse latest trained model
        df, baseline_predictions, risk_scores, run_dir = load_shared_inputs(args)
        run_case_study_counterfactual(
            args, df, baseline_predictions, risk_scores, run_dir
        )

if __name__ == "__main__":
    main()
