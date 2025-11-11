from sklearn.ensemble import RandomForestRegressor
from typing import Any, Dict

def get_model(**kwargs: Any) -> RandomForestRegressor:
    """
    Return a configured RandomForestRegressor instance.
    """
    default_params: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "criterion": "squared_error",
        "bootstrap": True,
        "oob_score": False,
        "warm_start": False,
    }

    default_params.update(kwargs)
    return RandomForestRegressor(**default_params)
