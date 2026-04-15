### 3/9/26, EB: Here I'm beginning the script that will compute the estimated causal effect of unemployment on opioid overdose mortality.
### There are several assumptions in this, i.e. what the plausible confounding set is, that the relationship is $t -> t+1$, and that I've constructed
### the DAG correctly, but this is a starting point. We can complexify from here.

import src.data_processing as data_proc
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import doubleml as dml
from doubleml import DoubleMLPLR, DoubleMLPLPR
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

UNEMP_TREATMENT = "unemp_rate"
RX_TREATMENT = "rx_rate"
OUTCOME_RAW = "mortality_rate"
OUTCOME_LEAD = "mortality_rate_lead"


treatments = {
    "unemployment": "unemp_rate",
    "rx_dispensing": "rx_rate"
}

UNEMP_CONFOUNDS = [
    # # "Below Poverty",#
    # # "No High School Diploma",#
    # "Minority Status",
    # # "No Vehicle",#
    # "Aged 17 or Younger",
    # "Aged 65 or Older",
    'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
    'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
    'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
    'Single-Parent Household', 'uninsured_rate'
]

RX_CONFOUNDS = [
    "unemp_rate",
    "uninsured_rate",
    # "Below Poverty",
    # # "No High School Diploma",
    # "Aged 65 or Older",
    # "Minority Status",
    'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
    'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
    'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
    'Single-Parent Household'
]

treatment_confounds = {
    "unemployment": UNEMP_CONFOUNDS,
    "rx_dispensing": RX_CONFOUNDS,
}

ID_COLS = ["FIPS", "year"]


### Functions for constructing the analysis dataframe and DoubleMLData objects for both PLR and PLPR approaches. 
### The main difference is that if we want to include fixed effects, PLR requires explicit residualization of the outcome and treatment, 
### while PLPR relies on the partialling out score function to do that implicitly.
### PLPR is more robust to misspecification of the fixed effects structure, but less interpretable in terms of the magnitude of the estimated effect.


### ----- PLR Functions ----- ###

def make_dml_panel_PLR(df: pl.DataFrame, two_way: bool, treatment: str, confounds: list[str]) -> pd.DataFrame:
    """
    Construct a county-year analysis dataframe for PLR-DML.

    Input:
        df: Polars dataframe with at least the columns in ID_COLS,
            OUTCOME_RAW, treatment, and confounds.

    Output:
        pandas.DataFrame ready for DoubleML.
    """
    required = set(ID_COLS + [OUTCOME_RAW, treatment] + confounds)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = (
        df.sort(["FIPS", "year"])
        .with_columns(
            pl.col(OUTCOME_RAW)
            .shift(-1)
            .over("FIPS")
            .alias(OUTCOME_LEAD)
        )
        .select(
            ID_COLS
            + [OUTCOME_LEAD, treatment]
            + confounds
        )
        .drop_nulls()
    )

    twfe_cols = [OUTCOME_LEAD, treatment] + confounds
    out = fixed_effect_residualize(out, twfe_cols, two_way=two_way)

    pdf = out.to_pandas()

    # Basic sanity checks
    if pdf.empty:
        raise ValueError("Analysis dataframe is empty after lead construction and drop_nulls().")

    if pdf[OUTCOME_LEAD].isna().any():
        raise ValueError("Outcome contains NaN values after TWFE residualization.")

    if pdf[treatment].isna().any():
        raise ValueError("Treatment contains NaN values after preprocessing.")

    return pdf

def fixed_effect_residualize(df: pl.DataFrame, cols: list[str], two_way: bool) -> pl.DataFrame:
    """
    Apply one- or two-way fixed effect residualization to the specified columns.

    For each column Z:
        Z* = Z - mean_i(Z) - mean_t(Z) + mean(Z)
    where:
        mean_i(Z) = county mean
        mean_t(Z) = year mean
        mean(Z)   = overall mean
    """
    out = df.clone()

    for col in cols:
        overall_mean = out.select(pl.col(col).mean()).item()

        if two_way:
            out = out.with_columns(
                (
                    pl.col(col)
                    - pl.col(col).mean().over("FIPS")
                    - pl.col(col).mean().over("year")
                    + overall_mean
                ).alias(col)
            )
        else:
            out = out.with_columns(
                (
                    pl.col(col)
                    - pl.col(col).mean().over("FIPS")
                    + overall_mean
                ).alias(col)
            )

    return out

def make_doubleml_data_PLR(pdf: pd.DataFrame, treatment: str, confounds: list[str]) -> dml.DoubleMLData:
    """
    Create a DoubleMLData object for unemployment -> next-year mortality.
    """
    dml_data = dml.DoubleMLData(
                    data=pdf,
                    y_col=OUTCOME_LEAD,
                    d_cols=treatment,
                    x_cols=confounds,
                    cluster_cols="FIPS"
                )
    
    return dml_data

def fit_plr(dml_data: dml.DoubleMLData, learner_type: str, random_state: int = 42) -> DoubleMLPLR:
    outcome_model, treatment_model  = make_learners(learner_type=learner_type)#, random_state=random_state)

    dml_plr = DoubleMLPLR(
        obj_dml_data=dml_data,
        ml_l=outcome_model,     # E[Y | X]
        ml_m=treatment_model ,  # E[D | X]
        n_folds=5,
        n_rep=1,
        score="partialling out",
    )

    dml_plr.fit()
    return dml_plr



### ----- PLPR Functions ----- ###

def make_dml_panel_PLPR(df: pl.DataFrame, two_way: bool, treatment: str, confounds: list[str]) -> pd.DataFrame:
    '''
    Construct a county-year analysis dataframe for partially linear panel regression DML.

    Unlike the manually residualized PLR workflow, PLPR handles county-level panel structure
    internally through the panel score construction. If `two_way=True`, year fixed effects are
    included explicitly as year dummy variables in the control set.
    
    Input:
    df: Polars dataframe with at least the columns in ID_COLS,
        OUTCOME_RAW, treatment, and confounds.

    Output:
        pandas.DataFrame ready for DoubleML.
    '''
    required = set(ID_COLS + [OUTCOME_RAW, treatment] + confounds)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = (
        df.sort(["FIPS", "year"])
        .with_columns(
            pl.col(OUTCOME_RAW)
            .shift(-1)
            .over("FIPS")
            .alias(OUTCOME_LEAD)
        )
        .select(
            ID_COLS
            + [OUTCOME_LEAD, treatment]
            + confounds
        )
        .drop_nulls()
    )

    pdf = out.to_pandas()

    if pdf.empty:
        raise ValueError("Analysis dataframe is empty after lead construction and drop_nulls().")

    if two_way:
        print("Two-way fixed effects will be included in the PLPR-DML analysis, but not explicitly residualized out of the outcome and treatment.")
        # # Add year fixed effects explicitly, if desired.
        year_dummies = pd.get_dummies(pdf["year"], prefix="year", drop_first=True)
        pdf = pd.concat([pdf, year_dummies], axis=1)

    return pdf

def make_doubleml_panel_data(pdf: pd.DataFrame, treatment: str, confounds: list[str]) -> dml.DoubleMLPanelData:
    x_cols = confounds + [c for c in pdf.columns if c.startswith("year_")]

    return dml.DoubleMLPanelData(
        data=pdf,
        y_col=OUTCOME_LEAD,     # Outcome variable (next-year mortality)
        d_cols=treatment,       # Treatment variable(s)
        t_col="year",           # Time variable for panel structure
        id_col="FIPS",          # Unit identifier for panel structure (i.e. county FIPS code)
        x_cols=x_cols,          # Baseline confounders + optional year fixed effects
        static_panel=True,      # 
    )

def fit_plpr(obj_dml_data: dml.DoubleMLPanelData, learner_type: str, random_state: int = 42) -> DoubleMLPLPR:
    outcome_regressor, treatment_regressor = make_learners(learner_type=learner_type)#, random_state=random_state)

    dml_plpr = DoubleMLPLPR(
        obj_dml_data=obj_dml_data,
        ml_l=outcome_regressor,
        ml_m=treatment_regressor,
        n_folds=5,
        n_rep=10,
        score="partialling out",
        approach="fd_exact",
        # approach="wg_approx",
    )

    dml_plpr.fit()
    return dml_plpr

### ----- Common Functions ----- ###

def make_learners(learner_type: str = "random_forest"):
    random_state = 42
    if learner_type == "random_forest":
        base_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            )
    elif learner_type == "lasso":
        base_model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lasso", LassoCV(cv=5, max_iter=10000, n_jobs=-1, verbose=0))
        ])
    else:
        raise ValueError(f"Unknown learner type: {learner_type}. Supported: 'random_forest', 'lasso'.")
    
    outcome_regressor = clone(base_model)
    treatment_regressor = clone(base_model)

    return outcome_regressor, treatment_regressor



def main_PLR():
    print("Running PLR-DML analysis...")
    data = data_proc.CountyDataLoader()
    df = data.load()

    pdf = make_dml_panel_PLR(df, two_way=False, treatment=RX_TREATMENT, confounds=RX_CONFOUNDS)
    # print(pdf[["FIPS","year","unemp_rate","mortality_rate_lead"]].head(20))
    # print(pdf.groupby("FIPS")["unemp_rate"].std().describe())
    # print(pdf.groupby("year")["mortality_rate_lead"].mean())
    # print(pdf.groupby("FIPS")[UNEMP_TREATMENT].mean().describe())
    # print(pdf.groupby("year")[UNEMP_TREATMENT].mean().describe())
    
    dml_data = make_doubleml_data_PLR(pdf, treatment=RX_TREATMENT, confounds=RX_CONFOUNDS)
    dml_plr = fit_plr(dml_data, learner_type="lasso")  # You can switch to "random_forest" if desired.

    print("PLR-DML results:")
    print(dml_plr.summary)
    # Clustering the cols by FIPS to get cluster robust standard errors, but bootstrapping is not supported with that just yet.
    # dml_plr.bootstrap()
    # print('')
    # print("Cluster-bootstrapped results:")
    # print(dml_plr.summary)
    
    dml_plr.sensitivity_analysis(cf_y=0.05, cf_d=0.05)
    print('')
    print("Sensitivity analysis: cf_y=0.05, cf_d=0.05")
    print(dml_plr.sensitivity_summary)
    # dml_plr.sensitivity_analysis(cf_y=0.01, cf_d=0.01)
    # dml_plr.sensitivity_analysis(cf_y=0.05, cf_d=0.05)
    # dml_plr.sensitivity_analysis(cf_y=0.10, cf_d=0.10)
    print('----------------------------------------------')


def main_PLPR():
    data = data_proc.CountyDataLoader()
    df = data.load()
    # print("Running PLPR-DML analysis: fd_exact, two-way fixed effects included as year dummies, and full time period...")
    # earlier_years = df.filter(pl.col("year") < 2017)
    # pdf = make_dml_panel_PLPR(earlier_years, two_way=True, treatment=RX_TREATMENT, confounds=RX_CONFOUNDS)
    # pdf = make_dml_panel_PLPR(df, two_way=True, treatment=RX_TREATMENT, confounds=RX_CONFOUNDS)
    
    # dml_panel_data = make_doubleml_panel_data(pdf, treatment=RX_TREATMENT, confounds=RX_CONFOUNDS)
    # dml_plpr = fit_plpr(dml_panel_data, learner_type="random_forest")

    # print("PLPR-DML results:")
    # print(dml_plpr.summary)
    # print('----------------------------------------------')
    # print('')
    # print('')
    # print('-----------------------------------------------')
    # Re-running, with just earlier years now, for sensitivity.
    print("Running PLPR-DML analysis Unemploymeny: fd_exact, **NO year fixed effects**, and just years > 201...")

    earlier_years = df.filter(pl.col("year") > 2018)
    pdf = make_dml_panel_PLPR(earlier_years, two_way=False, treatment=UNEMP_TREATMENT, confounds=UNEMP_CONFOUNDS)
    # pdf = make_dml_panel_PLPR(df, two_way=False, treatment=RX_TREATMENT, confounds=RX_CONFOUNDS)
    
    dml_panel_data = make_doubleml_panel_data(pdf, treatment=UNEMP_TREATMENT, confounds=UNEMP_CONFOUNDS)
    dml_plpr = fit_plpr(dml_panel_data, learner_type="random_forest")

    print("PLPR-DML results:")
    print(dml_plpr.summary)
    print('----------------------------------------------')
    # Sensitivity analysis is not currently supported for the PLPR implementation in DoubleML.
    # dml_plpr.sensitivity_analysis(cf_y=0.05, cf_d=0.05)
    # print('')
    # print("Sensitivity analysis: cf_y=0.05, cf_d=0.05")
    # print(dml_plpr.sensitivity_summary)


# def main2():
#     ## Quick check on a simple OLS regression to check whether sign and magnitude of DML results in main() are even reasonable.
#     import statsmodels.api as sm
#     data = data_proc.CountyDataLoader()
#     df = data.load()

#     pdf = make_dml_panel(df)

#     X = pdf[["unemp_rate"] + BASELINE_CONFOUNDS]
#     X = sm.add_constant(X)

#     y = pdf["mortality_rate_lead"]

#     ols = sm.OLS(y, X).fit()
#     print(ols.summary())


if __name__ == "__main__":
    # main_PLR()
    main_PLPR()