from sklearn.neural_network import MLPRegressor
from typing import Any, Dict

def get_model(**kwargs: Any) -> MLPRegressor:
    """
    Return a configured MLPRegressor instance for opioid mortality prediction.

    Parameters
    ----------
    **kwargs : Any
        Optional keyword arguments to override default hyperparameters.

    Returns
    -------
    sklearn.neural_network.MLPRegressor
    """
    default_params: Dict[str, Any] = {
        # Architecture
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,          # L2 regularization term
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "batch_size": 128,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": 42,
        "verbose": False,
    }

    # Allow user overrides
    default_params.update(kwargs)

    return MLPRegressor(**default_params)
