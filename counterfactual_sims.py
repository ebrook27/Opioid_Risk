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

    # parser.add_argument(
    #     "--region",
    #     type=str,
    #     default=None,
    #     help="Optional state abbreviation filter (e.g., 'TN' to restrict to Tennessee)."
    # )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate comparison plots for counterfactual results."
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


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
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
    target_fips = cf.pick_counterfactual_counties(high_risk, n=args.n_counties)
    
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


if __name__ == "__main__":
    main()
