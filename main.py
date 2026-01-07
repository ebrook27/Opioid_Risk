### 11/09/25, EB: Runs the risk score modelling pipeline, and produces maps of the risk scores for each predicted year.

import src.model_training as train
import src.data_processing as data_proc
import src.visualizations as viz
import src.metrics as metrics
import src.models.xgboost as xgb
import src.models.randomforest as rf
import src.models.mlp as mlp
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="Run opioid risk modeling pipeline."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "mlp"],
        help="Which model to use for training."
    )

    parser.add_argument(
        "--plot",
        type=str,
        default="risk",
        choices=["risk", "features", "mortality", "triple_map"],
        help="Which plot to generate after training."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save plots (if not provided, plots are displayed interactively)."
    )

    parser.add_argument(
        "--model_args",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Optional model hyperparameters, e.g. --model_args max_depth=10 learning_rate=0.05",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config file with model hyperparameters (overrides defaults)."
    )

    return parser.parse_args()

MODEL_REGISTRY = {
    "xgboost": xgb.get_model,
    "random_forest": rf.get_model,
    "mlp": mlp.get_model,
}

def main():
    # Parse command-line arguments
    args = get_args()

    # Load data
    data = data_proc.CountyDataLoader()
    df = data.load()

    # Load model defaults and overrides
    model_kwargs = {}
    
    # Load from YAML config if provided
    if args.config:
        print(f" Loading model config from {args.config}")
        model_kwargs.update(data_proc.load_yaml_config(args.config))

    # Parse command line model kwargs if provided
    if args.model_args:
        model_kwargs.update(data_proc.parse_model_args(args.model_args))

    # Check if model template exists
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[args.model](**model_kwargs)  # dynamically pick model

    # Run model training and prediction, save results
    model_metrics, feature_importances, predictions, all_errors, save_dir = (
        train.yearly_mortality_prediction_polars(df, model, save_path=args.save_dir)
    )

    risk_scores = metrics.compute_all_risk_scores(predictions)
    risk_scores_path = Path(save_dir) / "risk_scores.csv"
    risk_scores.to_csv(risk_scores_path, index=False)

    PLOT_DISPATCH = {
        "risk": lambda: viz.plot_county_metric_maps(risk_scores, "AbsError_Risk", save_dir=save_dir),
        "features": lambda: viz.plot_yearly_feature_importances(feature_importances, save_dir=save_dir),
        "mortality": lambda: viz.plot_county_metric_maps(df, "mortality_rate", save_dir=save_dir),
        "triple_map": lambda: viz.plot_triple_metric_maps(df, risk_scores, save_dir=save_dir, cmap_risk='Blues', error_col="AbsError", model_name=args.model),
    }

    if args.plot not in PLOT_DISPATCH:
        raise ValueError(f"Unknown plot type: {args.plot}. "
                         f"Available: {list(PLOT_DISPATCH.keys())}")

    PLOT_DISPATCH[args.plot]()  # run selected plotting function


if __name__ == "__main__":
    main()