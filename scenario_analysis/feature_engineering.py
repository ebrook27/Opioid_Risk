"""
Feature engineering for What-if? counterfactual pipeline.
 
This module builds lagged, change, and rolling-mean features from a
county-year panel.  It is designed to be called **twice**:
 
  1. On the actual (observed) data before model training.
  2. On the counterfactual-modified data (after RX rates have been
     reduced) before counterfactual prediction.
 
Calling the same function in both places guarantees that derived
features (lags, rolling means, changes) are internally consistent
with whatever Treatment Column is passed in.
"""
 
from __future__ import annotations
import polars as pl
 
# ─────────────────────────────────────────────────────────────────
# Column name constants (match your dataloader output)
# ─────────────────────────────────────────────────────────────────
ID_COL = "FIPS"
TIME_COL = "year"
TARGET_COL = "mortality_rate"
TREATMENT_COL = "rx_rate"
 
SVI_COLS = [
    "Aged 17 or Younger",
    "Aged 65 or Older",
    "Below Poverty",
    "Crowding",
    "Group Quarters",
    "Limited English Ability",
    "Minority Status",
    "Mobile Homes",
    "Multi-Unit Structures",
    "No High School Diploma",
    "No Vehicle",
    "Single-Parent Household",
]
 
ECON_COLS = ["unemp_rate", "uninsured_rate"]
 
URBANICITY_COLS = ["urbanicity_class"]
 
# Columns whose temporal dynamics matter most — lag + change these
DYNAMIC_COLS = [TREATMENT_COL] + ECON_COLS + ["Below Poverty", "No High School Diploma"]
 
# Columns to compute year-over-year changes for
CHANGE_COLS = [TREATMENT_COL, "unemp_rate", "Below Poverty"]
 
# Rolling mean specs: (column, window_size)
ROLLING_SPECS = [(TREATMENT_COL, 2)]



# ─────────────────────────────────────────────────────────────────
# Core feature engineering function
# ─────────────────────────────────────────────────────────────────
def build_panel_features(
    df: pl.DataFrame,
    *,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
    lag_cols: list[str] | None = None,
    n_lags: int = 1,
    change_cols: list[str] | None = None,
    rolling_specs: list[tuple[str, int]] | None = None,
    urbanicity_cols: list[str] | None = None,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    """
    Augment a county-year panel with history-based features.
 
    Parameters
    ----------
    df : pl.DataFrame
        Raw panel with at least `id_col`, `time_col`, and the columns
        referenced by the other arguments.
    id_col, time_col : str
        Column names for the cross-sectional unit and time index.
    target_col : str
        Name of the outcome variable (excluded from feature derivation
        but kept in the output for convenience).
    lag_cols : list[str] | None
        Columns for which to create lagged values.  If None, defaults
        to all numeric columns except id_col, time_col, and target_col.
    n_lags : int
        Number of lags to create (1 = one-year lag, 2 = two-year lag, etc.).
    change_cols : list[str] | None
        Columns for which to compute year-over-year change (val_t - val_{t-1}).
        If None, defaults to the same set as `lag_cols`.
    rolling_specs : list[tuple[str, int]] | None
        Each entry is (column_name, window_size).  A rolling mean of
        `window_size` years (inclusive of the current year) is computed.
    urbanicity_cols : list[str] | None
        Categorical urbanicity columns to integer-encode.
    drop_incomplete : bool
        If True, drop rows where any newly created feature is null
        (i.e., the first years per county that lack sufficient history).
 
    Returns
    -------
    pl.DataFrame
        The original columns plus all derived features, sorted by
        (id_col, time_col).
    """
    # ── resolve defaults ─────────────────────────────────────────
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32)
        and c not in {id_col, time_col, target_col}
    ]
 
    if lag_cols is None:
        lag_cols = numeric_cols
    if change_cols is None:
        change_cols = lag_cols
    if rolling_specs is None:
        rolling_specs = []
    if urbanicity_cols is None:
        urbanicity_cols = []
 
    # ── validate ─────────────────────────────────────────────────
    all_referenced = set(lag_cols) | set(change_cols) | {c for c, _ in rolling_specs}
    missing = all_referenced - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in dataframe: {sorted(missing)}")
 
    # ── sort for correct lag/rolling computation ─────────────────
    out = df.sort([id_col, time_col])
 
    # ── lagged features ──────────────────────────────────────────
    for lag in range(1, n_lags + 1):
        for col in lag_cols:
            out = out.with_columns(
                pl.col(col)
                .shift(lag)
                .over(id_col)
                .alias(f"{col}_lag{lag}")
            )
 
    # ── year-over-year change features ───────────────────────────
    for col in change_cols:
        out = out.with_columns(
            (pl.col(col) - pl.col(col).shift(1).over(id_col))
            .alias(f"{col}_chg")
        )
 
    # ── rolling mean features ────────────────────────────────────
    for col, window in rolling_specs:
        out = out.with_columns(
            pl.col(col)
            .rolling_mean(window_size=window, min_samples=window)
            .over(id_col)
            .alias(f"{col}_rmean{window}")
        )
 
    # ── urbanicity encoding ──────────────────────────────────────
    for col in urbanicity_cols:
        if col not in df.columns:
            print(f"Warning: Urbanicity column '{col}' not found, skipping.")
            continue
        if out[col].dtype in (pl.Utf8, pl.Categorical):
            categories = out[col].unique().drop_nulls().sort().to_list()
            cat_map = {cat: idx for idx, cat in enumerate(categories)}
            out = out.with_columns(
                pl.col(col)
                .map_elements(lambda x, _m=cat_map: _m.get(x), return_dtype=pl.Int32)
                .alias(f"{col}_enc")
            )
        else:
            out = out.with_columns(
                pl.col(col).cast(pl.Int32).alias(f"{col}_enc")
            )
 
    # ── drop rows with incomplete history ────────────────────────
    if drop_incomplete:
        derived_cols = (
            [f"{c}_lag{l}" for l in range(1, n_lags + 1) for c in lag_cols]
            + [f"{c}_chg" for c in change_cols]
            + [f"{c}_rmean{w}" for c, w in rolling_specs]
        )
        derived_cols = [c for c in derived_cols if c in out.columns]
        if derived_cols:
            out = out.drop_nulls(subset=derived_cols)
 
    return out
 
 
# ─────────────────────────────────────────────────────────────────
# Convenience wrapper with recommended defaults
# ─────────────────────────────────────────────────────────────────
def build_opioid_panel_features(
    df: pl.DataFrame,
    *,
    n_lags: int = 1,
    include_rolling: bool = True,
    drop_incomplete: bool = True,
) -> pl.DataFrame:
    """
    Wrapper around build_panel_features with defaults tuned for
    the opioid mortality G-computation pipeline.
 
    Lags:    rx_rate, unemp_rate, uninsured_rate, Below Poverty,
             No High School Diploma  (1 year by default)
    Changes: rx_rate, unemp_rate, Below Poverty
    Rolling: 3-year rolling mean of rx_rate (optional)
    Urbanicity: urbanicity_class -> integer-encoded
 
    Effective estimation windows (with RX data starting 2014):
      include_rolling=True:   first usable training year = 2016
      include_rolling=False:  first usable training year = 2015
    """
    rolling = ROLLING_SPECS if include_rolling else []
 
    return build_panel_features(
        df,
        lag_cols=DYNAMIC_COLS,
        n_lags=n_lags,
        change_cols=CHANGE_COLS,
        rolling_specs=rolling,
        urbanicity_cols=URBANICITY_COLS,
        drop_incomplete=drop_incomplete,
    )
 
 
# ─────────────────────────────────────────────────────────────────
# Feature column selector
# ─────────────────────────────────────────────────────────────────
def get_feature_columns(
    df: pl.DataFrame,
    *,
    id_col: str = ID_COL,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
    exclude_extra: list[str] | None = None,
) -> list[str]:
    """
    Return the list of feature column names from an enhanced panel,
    excluding identifiers, the target, and any extra columns the
    caller wants to drop (e.g., raw urbanicity strings).
    """
    always_exclude = {id_col, time_col, target_col}
    if exclude_extra:
        always_exclude |= set(exclude_extra)
    return [c for c in df.columns if c not in always_exclude]
 
 
# ─────────────────────────────────────────────────────────────────
# Counterfactual data construction helpers
# ─────────────────────────────────────────────────────────────────
def build_counterfactual_panel(
    df_raw: pl.DataFrame,
    *,
    reduction: float = 0.15,
    treatment_col: str = TREATMENT_COL,
    n_lags: int = 1,
    include_rolling: bool = True,
) -> pl.DataFrame:
    """
    Apply a UNIFORM reduction on {treatment_col} to the raw panel, 
    then rebuild all derived features from the modified series.
 
    Parameters
    ----------
    df_raw : pl.DataFrame
        The ORIGINAL (unmodified) raw panel from the dataloader.
        Do NOT pass an already-enhanced dataframe.
    reduction : float
        Proportional reduction in rx_rate.  0.15 means 15% lower.
        Absolute reductions .
    """
    # if not 0.0 <= reduction < 1.0:
    #     raise ValueError(f"reduction must be in [0, 1), got {reduction}")
 
    # df_modified = df_raw.with_columns(
    #     (pl.col(treatment_col) * (1.0 - reduction)).alias(treatment_col)
    # )
    if treatment_col == "rx_rate":
        df_modified = df_raw.with_columns(
            (pl.col(treatment_col) * (1.0 - reduction)).alias(treatment_col)
        )
    else:
        # For non-RX variables (Unemp/Unins), a reduction means an absolute drop 
        df_modified = df_raw.with_columns(
            (pl.col(treatment_col) - reduction).alias(treatment_col)
        )
 
    return build_opioid_panel_features(
        df_modified,
        n_lags=n_lags,
        include_rolling=include_rolling,
    )

 
# def build_cohort_lookup(
#     risk_scores: pl.DataFrame,
#     *,
#     risk_col: str = "AbsError_Risk",
#     lambda_val: float = 0.80,
#     quantiles: tuple[float, float] = (0.50, 0.90),
#     delta: tuple[float, float] = (0.10, 0.05),
# ) -> tuple[pl.DataFrame, dict[str, object]]:
#     """
#     Build a cohort lookup table assigning each county to a risk tier
#     with a corresponding RX multiplier.
 
#     This is the cohort-creation logic extracted from your original
#     apply_risk_based_rx_modifier_clean, separated so the lookup can
#     be reused for plotting, analysis, and passed into
#     build_counterfactual_panel_tiered.
 
#     Tiers (based on latest-year risk distribution):
#       - Low:  risk <= quantile(q_low)       -> mult = min(1, lambda_val + low_delta)
#       - Mid:  q_low < risk <= quantile(q_mid) -> mult = min(1, lambda_val + mid_delta)
#       - High: risk > quantile(q_mid)        -> mult = lambda_val
 
#     Parameters
#     ----------
#     risk_scores : pl.DataFrame
#         Must contain columns ["FIPS", "Year", risk_col].
#     risk_col : str
#         Column name in risk_scores containing the risk metric.
#     lambda_val : float
#         Multiplier for the high-risk tier (e.g., 0.80 = 20% reduction).
#     quantiles : tuple[float, float]
#         (q_low, q_mid) cutpoints on the risk distribution.
#     delta : tuple[float, float]
#         (low_delta, mid_delta) added to lambda_val for lower tiers.
 
#     Returns
#     -------
#     cohort_lookup : pl.DataFrame
#         Columns: [FIPS, risk, Cohort, rx_mult]
#     lambdas_and_quantiles : dict
#         For plotting: {"lambda_vals": {...}, "quantiles": (q_lo_val, q_mid_val)}
#     """
#     q_low, q_mid = quantiles
#     low_delta, mid_delta = delta
 
#     # ── validation ───────────────────────────────────────────────
#     required = {ID_COL, "Year", risk_col}
#     missing = required - set(risk_scores.columns)
#     if missing:
#         raise KeyError(f"risk_scores missing columns: {sorted(missing)}")
#     if not (0 < lambda_val <= 1):
#         raise ValueError("lambda_val must be in (0, 1].")
#     if not (0 < q_low < q_mid < 1):
#         raise ValueError("Require 0 < q_low < q_mid < 1.")
#     if mid_delta < 0 or low_delta < 0:
#         raise ValueError("mid_delta and low_delta must be >= 0.")
 
#     mid_mult = min(1.0, lambda_val + mid_delta)
#     low_mult = min(1.0, lambda_val + low_delta)
#     if not (lambda_val <= mid_mult <= low_mult <= 1.0):
#         raise ValueError(
#             f"Expected lambda_val <= mid_mult <= low_mult <= 1. "
#             f"Got lambda_val={lambda_val}, mid_mult={mid_mult}, low_mult={low_mult}"
#         )
 
#     # ── latest-year risk, one value per FIPS ─────────────────────
#     latest_year = risk_scores.select(pl.col("Year").max()).item()
#     rs = (
#         risk_scores
#         .filter(pl.col("Year") == latest_year)
#         .group_by(ID_COL)
#         .agg(pl.mean(risk_col).cast(pl.Float64).alias("risk"))
#     )
 
#     # ── quantile cutpoints ───────────────────────────────────────
#     q_lo_val = rs.select(pl.col("risk").quantile(q_low)).item()
#     q_mid_val = rs.select(pl.col("risk").quantile(q_mid)).item()
 
#     # ── assign tiers and multipliers ─────────────────────────────
#     cohort_lookup = (
#         rs.with_columns(
#             pl.when(pl.col("risk") <= q_lo_val).then(pl.lit("Low"))
#               .when(pl.col("risk") <= q_mid_val).then(pl.lit("Mid"))
#               .otherwise(pl.lit("High"))
#               .alias("Cohort")
#         )
#         .with_columns(
#             pl.when(pl.col("Cohort") == "High").then(pl.lit(lambda_val))
#               .when(pl.col("Cohort") == "Mid").then(pl.lit(mid_mult))
#               .otherwise(pl.lit(low_mult))
#               .alias("rx_mult")
#         )
#         .select([ID_COL, "risk", "Cohort", "rx_mult"])
#     )
 
#     counts = cohort_lookup.group_by("Cohort").len().sort("Cohort")
#     print(
#         f"\n[build_cohort_lookup] latest_year={latest_year}, "
#         f"cutpoints: q{int(q_low*100)}={q_lo_val:.6g}, q{int(q_mid*100)}={q_mid_val:.6g}"
#     )
#     print(counts)
 
#     lambdas_and_quantiles = {
#         "lambda_vals": {"High": lambda_val, "Mid": mid_mult, "Low": low_mult},
#         "quantiles": (q_lo_val, q_mid_val),
#     }
 
#     return cohort_lookup, lambdas_and_quantiles

def build_cohort_lookup(
    risk_scores: pl.DataFrame,
    *,
    risk_col: str = "AbsError_Risk",
    modifier_type: str = "multiplier",   # "multiplier" or "absolute_drop"
    base_value: float = 0.80,
    quantiles: tuple[float, float] = (0.50, 0.90),
    delta: tuple[float, float] = (0.10, 0.05),
) -> tuple[pl.DataFrame, dict[str, object]]:
    """
    Build a cohort lookup assigning each county to a risk tier with a
    cohort-specific intervention modifier.

    Two supported intervention types:

    1. modifier_type="multiplier"
       Used for rx_rate. The counterfactual is:
           x_cf = x * modifier_value
       where lower modifier_value means stronger reduction.

    2. modifier_type="absolute_drop"
       Used for unemployment/uninsured. The counterfactual is:
           x_cf = x - modifier_value
       where larger modifier_value means stronger reduction.

    Tier logic (based on latest-year risk distribution):
      - Low  : risk <= quantile(q_low)
      - Mid  : q_low < risk <= quantile(q_mid)
      - High : risk > quantile(q_mid)

    Modifier assignment:
      - For multiplier:
            High = base_value
            Mid  = min(1, base_value + mid_delta)
            Low  = min(1, base_value + low_delta)
      - For absolute_drop:
            High = base_value
            Mid  = max(0, base_value - mid_delta)
            Low  = max(0, base_value - low_delta)

    Parameters
    ----------
    risk_scores : pl.DataFrame
        Must contain columns [FIPS, Year, risk_col].
    risk_col : str
        Risk metric column.
    modifier_type : {"multiplier", "absolute_drop"}
        Type of intervention.
    base_value : float
        Baseline cohort-specific intervention value for the high-risk tier.
        For multiplier, e.g. 0.80 means 20% reduction.
        For absolute_drop, e.g. 1.00 means subtract 1.0 units.
    quantiles : tuple[float, float]
        (q_low, q_mid) cutpoints on the risk distribution.
    delta : tuple[float, float]
        Tier spacing relative to the high-risk tier.
        Interpreted differently by modifier_type:
          - multiplier: added upward for lower-risk tiers
          - absolute_drop: subtracted downward for lower-risk tiers

    Returns
    -------
    cohort_lookup : pl.DataFrame
        Columns: [FIPS, risk, Cohort, modifier_value, modifier_type]
    modifier_info : dict
        Includes modifier values by cohort and quantile cutpoints.
    """
    q_low, q_mid = quantiles
    low_delta, mid_delta = delta

    required = {ID_COL, "Year", risk_col}
    missing = required - set(risk_scores.columns)
    if missing:
        raise KeyError(f"risk_scores missing columns: {sorted(missing)}")

    if modifier_type not in {"multiplier", "absolute_drop"}:
        raise ValueError("modifier_type must be 'multiplier' or 'absolute_drop'.")

    if not (0 < q_low < q_mid < 1):
        raise ValueError("Require 0 < q_low < q_mid < 1.")

    if low_delta < 0 or mid_delta < 0:
        raise ValueError("delta values must be >= 0.")

    # ---- modifier values by tier ----
    if modifier_type == "multiplier":
        if not (0 < base_value <= 1):
            raise ValueError("For multiplier mode, base_value must be in (0, 1].")

        high_val = base_value
        mid_val = min(1.0, base_value + mid_delta)
        low_val = min(1.0, base_value + low_delta)

        if not (high_val <= mid_val <= low_val <= 1.0):
            raise ValueError(
                f"Expected High <= Mid <= Low <= 1 for multiplier mode. "
                f"Got High={high_val}, Mid={mid_val}, Low={low_val}"
            )

    else:  # absolute_drop
        if base_value < 0:
            raise ValueError("For absolute_drop mode, base_value must be >= 0.")

        high_val = base_value
        mid_val = max(0.0, base_value - mid_delta)
        low_val = max(0.0, base_value - low_delta)

        if not (0.0 <= low_val <= mid_val <= high_val):
            raise ValueError(
                f"Expected Low <= Mid <= High for absolute_drop mode. "
                f"Got High={high_val}, Mid={mid_val}, Low={low_val}"
            )

    # ---- latest-year risk, one value per FIPS ----
    latest_year = risk_scores.select(pl.col("Year").max()).item()
    rs = (
        risk_scores
        .filter(pl.col("Year") == latest_year)
        .group_by(ID_COL)
        .agg(pl.mean(risk_col).cast(pl.Float64).alias("risk"))
    )

    # ---- quantile cutpoints ----
    q_lo_val = rs.select(pl.col("risk").quantile(q_low)).item()
    q_mid_val = rs.select(pl.col("risk").quantile(q_mid)).item()

    # ---- assign cohorts ----
    cohort_lookup = (
        rs.with_columns(
            pl.when(pl.col("risk") <= q_lo_val).then(pl.lit("Low"))
              .when(pl.col("risk") <= q_mid_val).then(pl.lit("Mid"))
              .otherwise(pl.lit("High"))
              .alias("Cohort")
        )
        .with_columns(
            pl.when(pl.col("Cohort") == "High").then(pl.lit(high_val))
              .when(pl.col("Cohort") == "Mid").then(pl.lit(mid_val))
              .otherwise(pl.lit(low_val))
              .alias("modifier_value"),
            pl.lit(modifier_type).alias("modifier_type"),
        )
        .select([ID_COL, "risk", "Cohort", "modifier_value", "modifier_type"])
    )

    counts = cohort_lookup.group_by("Cohort").len().sort("Cohort")
    print(
        f"\n[build_cohort_lookup] latest_year={latest_year}, "
        f"cutpoints: q{int(q_low*100)}={q_lo_val:.6g}, q{int(q_mid*100)}={q_mid_val:.6g}"
    )
    print(counts)

    modifier_info = {
        "modifier_type": modifier_type,
        "modifier_vals": {"High": high_val, "Mid": mid_val, "Low": low_val},
        "quantiles": (q_lo_val, q_mid_val),
    }

    return cohort_lookup, modifier_info


# def build_counterfactual_panel_tiered(
#     df_raw: pl.DataFrame,
#     cohort_lookup: pl.DataFrame,
#     *,
#     treatment_col: str = TREATMENT_COL,
#     n_lags: int = 1,
#     include_rolling: bool = True,
# ) -> pl.DataFrame:
#     """
#     Apply RISK-TIERED RX reductions to the raw panel, then rebuild
#     all derived features from the modified series.
 
#     The cohort_lookup table should have columns [FIPS, rx_mult]
#     mapping each county to its multiplier (e.g., 0.80 for a 20%
#     reduction in the high-risk tier).
 
#     Parameters
#     ----------
#     df_raw : pl.DataFrame
#         The ORIGINAL raw panel from the dataloader.
#     cohort_lookup : pl.DataFrame
#         Must contain columns ["FIPS", "rx_mult"].
#     """
#     df_modified = df_raw.join(
#         cohort_lookup.select([ID_COL, "rx_mult"]),
#         on=ID_COL,
#         how="left",
#     )
 
#     if df_modified["rx_mult"].null_count() > 0:
#         df_modified = df_modified.with_columns(
#             pl.col("rx_mult").fill_null(1.0)
#         )
 
#     df_modified = (
#         df_modified
#         .with_columns((pl.col(treatment_col) * pl.col("rx_mult")).alias(treatment_col))
#         .drop("rx_mult")
#     )
 
#     return build_opioid_panel_features(
#         df_modified,
#         n_lags=n_lags,
#         include_rolling=include_rolling,
#     )

def build_counterfactual_panel_tiered(
    df_raw: pl.DataFrame,
    cohort_lookup: pl.DataFrame,
    *,
    treatment_col: str,
    n_lags: int = 1,
    include_rolling: bool = True,
) -> pl.DataFrame:
    """
    Apply cohort-specific counterfactual modifications, then rebuild features.
    """
    df = df_raw.join(
        cohort_lookup.select([ID_COL, "modifier_value", "modifier_type"]),
        on=ID_COL,
        how="left",
    )

    modifier_type = cohort_lookup.select("modifier_type").unique().to_series().to_list()
    if len(modifier_type) != 1:
        raise ValueError("cohort_lookup must contain exactly one modifier_type.")
    modifier_type = modifier_type[0]

    if modifier_type == "multiplier":
        df = df.with_columns(
            (pl.col(treatment_col) * pl.col("modifier_value")).alias(treatment_col)
        )
    elif modifier_type == "absolute_drop":
        df = df.with_columns(
            (pl.col(treatment_col) - pl.col("modifier_value")).clip(lower_bound=0.0).alias(treatment_col)
        )
    else:
        raise ValueError(f"Unknown modifier_type: {modifier_type}")

    df = df.drop(["modifier_value", "modifier_type"])

    return build_opioid_panel_features(
        df,
        n_lags=n_lags,
        include_rolling=include_rolling,
    )

