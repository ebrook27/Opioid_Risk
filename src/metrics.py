import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def compute_all_risk_scores(predictions_df, alpha=0.3):
    """
    Compute both equal-weight (expanding mean) and EWMA risk scores
    for multiple error types (AbsError, SqError, RawError).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns ['FIPS', 'Year', 'True', 'Predicted', 'AbsError'].
    alpha : float, optional (default=0.3)
        Smoothing factor for EWMA (higher = more recent years weighted more).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['FIPS', 'Year',
         'AbsError_Risk', 'SqError_Risk', 'RawError_Risk',
         'AbsError_EWMA_Risk', 'SqError_EWMA_Risk', 'RawError_EWMA_Risk']
    """

    df = predictions_df.copy()

    # --- Ensure required columns exist ---
    if 'SqError' not in df.columns:
        df['SqError'] = (df['True'] - df['Predicted']) ** 2
    if 'RawError' not in df.columns:
        df['RawError'] = df['True'] - df['Predicted']

    # --- Ensure proper ordering ---
    df = df.sort_values(['FIPS', 'Year']).copy()

    errors = ['AbsError', 'SqError', 'RawError']

    for error in errors:
        # Equal-weight expanding mean risk
        df[f'{error}_Risk'] = (
            df.groupby('FIPS')[error]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
        )

        # Exponentially weighted moving average (EWMA) risk
        df[f'{error}_EWMA_Risk'] = (
            df.groupby('FIPS')[error]
              .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
              .reset_index(level=0, drop=True)
        )

    # --- Select just the risk columns and identifiers ---
    risk_cols = (
        ['FIPS', 'Year'] +
        [f'{err}_Risk' for err in errors] +
        [f'{err}_EWMA_Risk' for err in errors]
    )

    return df[risk_cols]


def shap_coclustering_persistence(
    shap_long_df: pd.DataFrame,
    *,
    linkage_method: str = "average",
    cutoff: float = 0.7,
    min_obs: int = 50,
    use_abs_corr: bool = True,
    denom: str = "total",
):
    """
    Compute feature co-clustering persistence from a long SHAP table.

    For each year t:
      - pivot long SHAP -> wide matrix M_t (rows=FIPS, cols=Feature)
      - compute correlation C_t = corr(M_t)
      - convert to distances D_t = 1 - |C_t| (or 1 - C_t if use_abs_corr=False)
      - run hierarchical clustering on D_t and cut at distance `cutoff`
      - record which feature pairs are in the same cluster

    Finally returns a persistence matrix P where
      P[i,j] = (1 / T) * sum_t 1[z_i,t == z_j,t]
    where T is the number of years considered (or number of years where both
    features are present when `denom='pairwise'`).

    Parameters
    ----------
    shap_long_df : pd.DataFrame
        Long-format SHAP table with columns ['FIPS','Year','Feature','SHAP']
        (other columns are ignored).
    linkage_method : str
        Method passed to `scipy.cluster.hierarchy.linkage`.
    cutoff : float
        Distance threshold at which to cut the dendrogram (in the same units
        as the distance matrix, typically in [0,1] when using 1-|corr|).
    min_obs : int
        Minimum number of counties required in a year to include that year.
    use_abs_corr : bool
        If True, use D = 1 - |corr|. If False, use D = 1 - corr (signed).
    denom : {'total','pairwise'}
        Whether to divide by total number of years considered ('total') or by
        number of years where both features were present ('pairwise').

    Returns
    -------
    persistence_df : pd.DataFrame
        DataFrame (features x features) with co-clustering frequencies in [0,1].
    details : dict
        Metadata with keys: 'years_used' (list), 'cluster_labels' (DataFrame: years x features)
    """

    required = {"FIPS", "Year", "Feature", "SHAP"}
    missing = required - set(shap_long_df.columns)
    if missing:
        raise ValueError(f"shap_long_df missing columns: {missing}")

    df = shap_long_df.copy()
    df = df.dropna(subset=["FIPS", "Year", "Feature"])
    df["FIPS"] = df["FIPS"].astype(str)
    df["Year"] = df["Year"].astype(int)

    years = sorted(df["Year"].unique())
    years_used = []
    pair_counts = {}  # (f1,f2) -> count of co-cluster occurrences
    pair_years = {}   # (f1,f2) -> number of years both present (for pairwise denom)
    features_union = set()
    cluster_labels_by_year = {}

    for yr in years:
        sub = df[df["Year"] == yr]
        # pivot to (FIPS x Feature) wide matrix, averaging across folds if present
        M = sub.pivot_table(index="FIPS", columns="Feature", values="SHAP", aggfunc="mean")
        M = M.dropna(axis=1, how="all")
        if M.shape[0] < min_obs or M.shape[1] < 2:
            continue

        years_used.append(yr)
        feats = list(M.columns)
        features_union.update(feats)

        X = M.to_numpy()
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        if use_abs_corr:
            C = np.abs(C)

        D = 1.0 - C

        # condensed distance for linkage
        if D.shape[0] == 1:
            # only one feature (shouldn't happen because M.shape[1] >=2), skip
            continue
        d_condensed = squareform(D, checks=False)
        Z = linkage(d_condensed, method=linkage_method)
        labels = fcluster(Z, t=cutoff, criterion="distance")

        # record labels per feature
        cluster_labels_by_year[yr] = pd.Series(labels, index=feats)

        # update pair counts
        for i, f1 in enumerate(feats):
            for j, f2 in enumerate(feats):
                if j <= i:
                    continue
                key = (f1, f2)
                pair_years[key] = pair_years.get(key, 0) + 1
                same = int(labels[i] == labels[j])
                if same:
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    if not years_used:
        raise ValueError("No years met the `min_obs` and feature-count requirements.")

    features = sorted(features_union)
    P = pd.DataFrame(0.0, index=features, columns=features)

    T = len(years_used)
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if j < i:
                P.iat[i, j] = P.iat[j, i]
                continue
            if f1 == f2:
                P.iat[i, j] = 1.0
                continue
            key = (f1, f2) if (f1, f2) in pair_counts else (f2, f1)
            count = pair_counts.get(key, 0)
            if denom == "total":
                P.iat[i, j] = count / T
            elif denom == "pairwise":
                denom_count = pair_years.get(key, 0)
                P.iat[i, j] = (count / denom_count) if denom_count > 0 else np.nan
            else:
                raise ValueError("denom must be 'total' or 'pairwise'")
            P.iat[j, i] = P.iat[i, j]

    cluster_df = pd.DataFrame(index=years_used, columns=features)
    for yr, s in cluster_labels_by_year.items():
        cluster_df.loc[yr, s.index] = s.values

    details = {"years_used": years_used, "cluster_labels": cluster_df}
    return P, details


