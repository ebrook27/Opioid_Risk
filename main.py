### 11/09/25, EB: Runs the risk score modelling pipeline, and produces maps of the risk scores for each predicted year.

from src.model_training import yearly_mortality_prediction_polars
from src.data_processing import CountyDataLoader, load_yaml_config, parse_model_args
from src.visualizations import plot_county_metric_maps, plot_yearly_feature_importances
from src.metrics import compute_all_risk_scores
from src.models.xgboost import get_model as xgb_model
from src.models.randomforest import get_model as rf_model
from src.models.mlp import get_model as mlp_model
import argparse


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
        choices=["risk", "features", "mortality"],
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
    "xgboost": xgb_model,
    "random_forest": rf_model,
    "mlp": mlp_model
}

PLOT_DISPATCH = {
    "risk": plot_county_metric_maps,
    "features": plot_yearly_feature_importances,
    "mortality": plot_county_metric_maps,
}

def main():
    # Parse command-line arguments
    args = get_args()

    # Check if model template exists
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    if args.plot not in PLOT_DISPATCH:
        raise ValueError(f"Unknown plot type: {args.plot}. "
                         f"Available: {list(PLOT_DISPATCH.keys())}")

    # Load data
    data = CountyDataLoader()
    df = data.load()

    # Load model defaults and overrides
    model_kwargs = {}

    # Load from YAML config if provided
    if args.config:
        print(f" Loading model config from {args.config}")
        model_kwargs.update(load_yaml_config(args.config))

    # Parse command line model kwargs if provided
    if args.model_args:
        model_kwargs.update(parse_model_args(args.model_args))

    model = MODEL_REGISTRY[args.model](**model_kwargs)  # dynamically pick model

    # Run model training and prediction, save results
    metrics, feature_importances, predictions, all_errors, save_dir = (
        yearly_mortality_prediction_polars(df, model, save_path=args.save_dir)
    )

    risk_scores = compute_all_risk_scores(predictions)

    match args.plot:
        case "risk":
            PLOT_DISPATCH["risk"](risk_scores, "AbsError_Risk", save_dir=save_dir)  # run selected plotting function
        case "features":
            PLOT_DISPATCH["features"](feature_importances, save_dir=save_dir)  # run selected plotting function
        case "mortality":
            PLOT_DISPATCH["mortality"](df, "mortality_rate", save_dir=save_dir)  # run selected plotting function


if __name__ == "__main__":
    main()
