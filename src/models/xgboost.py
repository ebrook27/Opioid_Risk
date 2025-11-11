### 11/10/25, EB: Need to verify the best model, but here's a starting model.
import xgboost as xgb

def get_model(**kwargs):
    """
    Return a configured XGBoostRegressor instance.
    
    Parameters
    ----------
    **kwargs : dict
        Optional keyword arguments to override default parameters.

    Returns
    -------
    xgb.XGBRegressor
    """
    default_params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    default_params.update(kwargs)
    return xgb.XGBRegressor(**default_params)
