### 11/26/25, EB: Contains a few functions that are for testing.

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns



# xgb_risks = pd.read_csv('model_outputs/xgbregressor/2025-11-20_21-47-27/risk_scores.csv')
# rf_risks = pd.read_csv('model_outputs/randomforestregressor/2025-12-02_20-36-07/risk_scores.csv')
# mlp_risks = pd.read_csv('model_outputs/mlpregressor/2025-12-02_20-40-45/risk_scores.csv')

# def compute_risk_mean_and_variance(df, risk_cols):
#     """
#     Compute per-county mean risk and variance of yearly risk values 
#     for each risk column in `risk_cols`.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain columns ['FIPS', 'Year'] + risk columns.
#     risk_cols : list of str
#         The risk column names to summarize (e.g., AbsError_Risk, SqError_Risk, ...)

#     Returns
#     -------
#     summary_df : pd.DataFrame
#         Columns:
#             FIPS,
#             <risk_col>_mean,
#             <risk_col>_var
#     """
    
#     # Group by county
#     grouped = df.groupby('FIPS')

#     # For each risk column, compute mean and variance across years
#     summary_dict = {}

#     for col in risk_cols:
#         summary_dict[f"{col}_mean"] = grouped[col].mean()
#         summary_dict[f"{col}_var"]  = grouped[col].var(ddof=1)  # sample variance
    
#     # Combine into a single DataFrame
#     summary_df = pd.concat(summary_dict, axis=1).reset_index()
    
#     return summary_df


# # def plot_county_risk_timeseries(df, fips, risk_col, show_ci=True):
# #     """
# #     Plot the risk time series for a single county, optionally with 
# #     a 95% confidence interval around the estimated mean risk.

# #     Parameters
# #     ----------
# #     df : pd.DataFrame
# #         Must contain columns ['FIPS', 'Year', risk_col].
# #     fips : str or int
# #         The county FIPS code to plot.
# #     risk_col : str
# #         The name of the risk column to plot.
# #     show_ci : bool
# #         Whether to show mean ± 1.96 * SE band.
# #     """

# #     # Subset for the county
# #     county_df = df[df['FIPS'] == fips].sort_values('Year')

# #     # Extract time series
# #     years = county_df['Year']
# #     risks = county_df[risk_col]

# #     # Compute mean and sample variance
# #     mean_risk = risks.mean()
# #     var_risk  = risks.var(ddof=1)
# #     T = len(risks)

# #     # Standard error of the mean
# #     SE = (var_risk ** 0.5) / (T ** 0.5)

# #     fig, ax = plt.subplots(figsize=(10, 5))

# #     # Time series line
# #     ax.plot(years, risks, marker='o', label=f'Yearly {risk_col}')

# #     # Mean line
# #     ax.axhline(mean_risk, color='orange', linestyle='--', label='Mean risk')

# #     # Confidence band
# #     if show_ci:
# #         upper = mean_risk + 1.96 * SE
# #         lower = mean_risk - 1.96 * SE
# #         ax.fill_between(years, lower, upper, color='orange', alpha=0.2,
# #                         label='95% CI of mean risk')

# #     ax.set_title(f"Risk Time Series for County FIPS {fips}")
# #     ax.set_xlabel("Year")
# #     ax.set_ylabel(f"{risk_col}")
# #     ax.legend()
# #     plt.show()


# def plot_county_risk_timeseries(df, fips, risk_col, show_ci=True):
#     """
#     Plot the risk time series for a single county, optionally with 
#     a 95% confidence interval around the estimated mean risk.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain columns ['FIPS', 'Year', risk_col].
#     fips : str or int
#         The county FIPS code to plot (e.g., '01001' or 1001).
#     risk_col : str
#         The name of the risk column to plot.
#     show_ci : bool
#         Whether to show mean ± 1.96 * SE band.
#     """

#     # Work on a copy so we don't mutate the original df
#     df_local = df.copy()

#     # Normalize FIPS to 5-character strings with leading zeros
#     df_local['FIPS'] = df_local['FIPS'].astype(str).str.zfill(5)
#     fips_str = str(fips).zfill(5)

#     # Subset for the county
#     county_df = df_local[df_local['FIPS'] == fips_str].sort_values('Year')

#     if county_df.empty:
#         raise ValueError(
#             f"No rows found for FIPS={fips_str}. "
#             "Check the FIPS format in your DataFrame."
#         )

#     # Extract time series
#     years = county_df['Year']
#     risks = county_df[risk_col]

#     # Number of years
#     T = len(risks)
#     if T < 2:
#         raise ValueError(
#             f"Not enough time points (T={T}) for FIPS={fips_str} "
#             "to compute variance/SE."
#         )

#     # Compute mean and sample variance
#     mean_risk = risks.mean()
#     var_risk  = risks.var(ddof=1)  # sample variance

#     # Standard error of the mean
#     SE = (var_risk ** 0.5) / (T ** 0.5)

#     fig, ax = plt.subplots(figsize=(10, 5))

#     # Time series line
#     ax.plot(years, risks, marker='o', label=f'Yearly {risk_col}')

#     # Mean line
#     ax.axhline(mean_risk, linestyle='--', label='Mean risk')

#     # Confidence band around the *mean*, not the series
#     if show_ci:
#         upper = mean_risk + 1.96 * SE
#         lower = mean_risk - 1.96 * SE
#         ax.fill_between(years, lower, upper, alpha=0.2,
#                         label='95% CI of mean risk')

#     ax.set_title(f"Risk Time Series for County FIPS {fips_str}")
#     ax.set_xlabel("Year")
#     ax.set_ylabel(risk_col)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()



# def plot_mean_variance_scatter(summary_df, risk_prefix, color_col=None):
#     """
#     Create a mean vs. variance scatterplot for county risk.

#     Parameters
#     ----------
#     summary_df : pd.DataFrame
#         Output of compute_risk_mean_and_variance(), one row per county.
#     risk_prefix : str
#         Prefix of the risk type, e.g. "AbsError_Risk", "SqError_Risk".
#         The function expects columns named f"{risk_prefix}_mean" and f"{risk_prefix}_var".
#     color_col : str or None
#         Optional column name indicating a categorical grouping (e.g. NCHS code).
#         If None, all points are blue.
#     """

#     mean_col = f"{risk_prefix}_mean"
#     var_col  = f"{risk_prefix}_var"

#     if mean_col not in summary_df.columns or var_col not in summary_df.columns:
#         raise ValueError(
#             f"Required columns {mean_col} and/or {var_col} not found in summary_df."
#         )

#     plt.figure(figsize=(10, 6))

#     if color_col is not None:
#         sns.scatterplot(
#             data=summary_df,
#             x=mean_col,
#             y=var_col,
#             hue=color_col,
#             palette="tab10",
#             alpha=0.7
#         )
#     else:
#         sns.scatterplot(
#             data=summary_df,
#             x=mean_col,
#             y=var_col,
#             color="steelblue",
#             alpha=0.7
#         )

#     # Add reference lines (median or mean)
#     x_med = summary_df[mean_col].median()
#     y_med = summary_df[var_col].median()

#     plt.axvline(x=x_med, color='gray', linestyle='--', alpha=0.6)
#     plt.axhline(y=y_med, color='gray', linestyle='--', alpha=0.6)

#     plt.title(f"Mean vs Variance of {risk_prefix} Across Counties")
#     plt.xlabel("Mean Risk Across Years")
#     plt.ylabel("Variance of Yearly Risk")
#     plt.grid(alpha=0.2)
#     plt.tight_layout()
#     plt.show()



# risk_cols = [
#     "AbsError_Risk",
#     "SqError_Risk",
#     "RawError_Risk",
#     "AbsError_EWMA_Risk",
#     "SqError_EWMA_Risk",
#     "RawError_EWMA_Risk",
# ]

# xgb_risks = compute_risk_mean_and_variance(xgb_risks, risk_cols=risk_cols)

# plot_county_risk_timeseries(xgb_risks, fips='27001', risk_col='AbsError_Risk', show_ci=True)
# # plot_mean_variance_scatter(xgb_risks, risk_prefix='AbsError_Risk', color_col=None)


# ### 12/1/25, EB: National Level Computations

# def compute_national_risk_band(df, risk_col):
#     """
#     Compute national (cross-sectional) mean risk, variance across counties,
#     SE of the mean, and 95% CI for each year.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain columns ['FIPS', 'Year', risk_col].
#     risk_col : str
#         Risk column to analyze, e.g. 'AbsError_Risk'.
    
#     Returns
#     -------
#     band_df : pd.DataFrame with:
#         Year, mean_risk, var_across_counties, se_mean, ci_lower, ci_upper
#     """
    
#     # Number of counties (should be constant per year, but we compute globally)
#     N = df['FIPS'].nunique()
    
#     # Group by year
#     grouped = df.groupby('Year')[risk_col]
    
#     mean_risk = grouped.mean()
#     var_across = grouped.var(ddof=1)     # sample variance across counties
#     se_mean = (var_across ** 0.5) / (N ** 0.5)
    
#     # 95% CI
#     ci_upper = mean_risk + 1.96 * se_mean
#     ci_lower = mean_risk - 1.96 * se_mean
    
#     band_df = pd.DataFrame({
#         'Year': mean_risk.index,
#         'mean_risk': mean_risk.values,
#         'var_across_counties': var_across.values,
#         'se_mean': se_mean.values,
#         'ci_lower': ci_lower.values,
#         'ci_upper': ci_upper.values
#     })
    
#     return band_df

# def plot_national_risk_band(band_df, risk_col_label="Risk"):
#     """
#     Plot national mean risk over time with 95% CI band.
    
#     Parameters
#     ----------
#     band_df : pd.DataFrame
#         Output of compute_national_risk_band().
#     risk_col_label : str
#         Label for y-axis (e.g., 'AbsError_Risk')
#     """
    
#     years = band_df['Year']
#     mean_risk = band_df['mean_risk']
#     ci_lower = band_df['ci_lower']
#     ci_upper = band_df['ci_upper']

#     plt.figure(figsize=(12, 6))
    
#     # Plot CI band
#     plt.fill_between(years, ci_lower, ci_upper, alpha=0.2, color='steelblue',
#                      label='95% CI of national mean risk')
    
#     # Plot mean risk line
#     plt.plot(years, mean_risk, color='steelblue', marker='o',
#              label='National mean risk')

#     plt.title(f"National Mean {risk_col_label} Over Time\nwith 95% Confidence Band, XGBoost Model", fontsize=16)
#     plt.xlabel("Year")
#     plt.ylabel(risk_col_label)
    
#     plt.grid(alpha=0.2)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_national_risk_band_with_trajectories(df, band_df, risk_col_label="Risk"):
#     """
#     Plot national mean risk over time with 95% CI band, and overlay risk
#     trajectories for all counties in light grey.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Long-format risk data: columns [FIPS, Year, <risk_col>].
#     band_df : pd.DataFrame
#         Output of compute_national_risk_band().
#     risk_col_label : str
#         Name of the risk column, e.g. 'AbsError_Risk'.
#     """

#     # Extract info
#     years = band_df["Year"]
#     mean_risk = band_df["mean_risk"]
#     ci_lower = band_df["ci_lower"]
#     ci_upper = band_df["ci_upper"]

#     plt.figure(figsize=(14, 7))

#     # --- 1. Plot all county trajectories ---
#     for fips, county_data in df.groupby("FIPS"):
#         plt.plot(
#             county_data["Year"],
#             county_data[risk_col_label],
#             color="lightgrey",
#             linewidth=0.6,
#             alpha=0.4
#         )

#     # --- 2. Plot CI band ---
#     plt.fill_between(
#         years,
#         ci_lower,
#         ci_upper,
#         color="steelblue",
#         alpha=0.2,
#         label="95% CI of national mean risk"
#     )

#     # --- 3. Plot national mean ---
#     plt.plot(
#         years,
#         mean_risk,
#         color="steelblue",
#         linewidth=2.5,
#         marker="o",
#         label="National mean risk"
#     )

#     # --- 4. Plot formatting ---
#     plt.title(
#         f"National {risk_col_label} with Confidence Band\n"
#         f"And County-Level Risk Trajectories",
#         fontsize=16
#     )
#     plt.xlabel("Year", fontsize=14)
#     plt.ylabel(risk_col_label, fontsize=14)

#     plt.grid(alpha=0.2)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_three_national_risk_bands(
#     band_xgb: pd.DataFrame,
#     band_rf: pd.DataFrame,
#     band_mlp: pd.DataFrame,
#     risk_col_label: str = "Risk",
#     model_labels: dict = None
# ):
#     """
#     Plot national mean risk over time for three models with 95% CI bands.

#     Parameters
#     ----------
#     band_xgb, band_rf, band_mlp : pd.DataFrame
#         Each must have ['Year', 'mean_risk', 'ci_lower', 'ci_upper'].
#     risk_col_label : str
#         Y-axis label.
#     model_labels : dict
#         Optional mapping for naming curves, e.g.:
#             {'xgb': 'XGBoost', 'rf': 'Random Forest', 'mlp': 'Neural Net'}
#     """

#     if model_labels is None:
#         model_labels = {
#             'xgb': 'XGBoost',
#             'rf': 'Random Forest',
#             'mlp': 'MLP'
#         }

#     # Bundle the inputs for iteration
#     models = {
#         'xgb': band_xgb,
#         'rf': band_rf,
#         'mlp': band_mlp
#     }

#     # Color palette chosen to avoid overlapping confusion
#     colors = {
#         'xgb': 'steelblue',
#         'rf': 'darkorange',
#         'mlp': 'seagreen'
#     }

#     plt.figure(figsize=(12, 6))

#     for key, df in models.items():
#         years = df['Year']
#         mean_risk = df['mean_risk']
#         ci_lower = df['ci_lower']
#         ci_upper = df['ci_upper']

#         # CI band
#         plt.fill_between(
#             years,
#             ci_lower,
#             ci_upper,
#             alpha=0.15,
#             color=colors[key],
#             # label=f"{model_labels[key]} 95% CI"
#         )

#         # Mean line
#         plt.plot(
#             years,
#             mean_risk,
#             color=colors[key],
#             marker='o',
#             linewidth=2,
#             label=f"{model_labels[key]}"
#         )

#     plt.title(f"National Mean Risk Over Time", fontsize=16)
#     plt.xlabel("Year")
#     plt.ylabel(risk_col_label)
#     plt.grid(alpha=0.2)
#     plt.legend(ncol=2)
#     plt.tight_layout()
#     plt.show()








# national_risk_band = compute_national_risk_band(xgb_risks, risk_col='AbsError_Risk')
# # print(national_risk_band.head(10))
# # plot_national_risk_band(national_risk_band, risk_col_label='AbsError_Risk')
# plot_three_national_risk_bands(
#     band_xgb=national_risk_band,
#     band_rf=compute_national_risk_band(rf_risks, risk_col='AbsError_Risk'),
#     band_mlp=compute_national_risk_band(mlp_risks, risk_col='AbsError_Risk'),
#     risk_col_label='AbsError_Risk',
#     model_labels={
#         'xgb': 'XGBoost',
#         'rf': 'Random Forest',
#         'mlp': 'Neural Network'
#     }
# )
# # plot_national_risk_band_with_trajectories(xgb_risks, national_risk_band, risk_col_label='AbsError_Risk')


##############################################
# ### 2/2/26, EB: Testing the new dataloader, added uninsured rates
# import src.data_processing as data_proc
# data = data_proc.CountyDataLoader()
# df = data.load()

# # Checking CT counties are present
# ct_rows = df.filter(pl.col("FIPS").str.starts_with("09")).height
# print("CT rows in merged:", ct_rows)
# assert ct_rows > 0

# # Checking for nulls in key variables
# required = ["mortality_rate", "rx_rate", "unemp_rate", "uninsured_rate"] + data.svi_variables

# ct_nulls = (
#     df.filter(pl.col("FIPS").str.starts_with("09"))
#           .select([pl.col(c).is_null().sum().alias(c) for c in required])
# )

# print(ct_nulls)
# print("Total nulls in CT data:", ct_nulls.sum_horizontal().item())

# # Checking for panel completeness in CT counties
# ct = df.filter(pl.col("FIPS").str.starts_with("09"))
# n_ct = ct.select(pl.col("FIPS").n_unique()).item()
# yrs = ct.select(pl.col("year").n_unique()).item()
# expected = n_ct * yrs
# assert ct.height == expected, f"CT panel has holes: rows={ct.height}, expected={expected}"

# print("")
# print("")
# print(df.glimpse())
# print("Total rows in merged data:", df.height)
# print(df.describe())
# print(df.schema)


### 2/9/26, EB: Investigating historical Unemployment rate trends

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

unemp_rates = pl.scan_csv('data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv')


def plot_state_unemployment_trends(unemp_rates, state_fips_path='data/Raw/STATE_FIPS_CODES.txt', 
                                    save=False, save_path=None):
    """
    Plot unemployment rate trends for each state, showing all counties in light grey
    with a black overlay line for the state mean.
    
    Parameters:
    -----------
    unemp_rates : polars.LazyFrame or DataFrame
        Data with FIPS codes (5-digit) and columns named "20XX Unemployment" for years 10-22
    state_fips_path : str
        Path to the STATE_FIPS_CODES.txt file for mapping 2-digit FIPS to state names
    save : bool
        Whether to save the plots as PNG files
    save_path : str or Path
        Directory path to save plots if save=True
    """
    # Collect the lazy frame if needed
    if hasattr(unemp_rates, 'collect'):
        df = unemp_rates.collect()
    else:
        df = unemp_rates
    
    # Read state FIPS codes
    state_fips = pl.read_csv(state_fips_path)
    state_fips = state_fips.with_columns(
        pl.col('FIPS').cast(pl.Utf8).str.pad_start(2, '0')
    )
    
    # Extract years from column names
    unemployment_cols = [col for col in df.columns if 'Unemployment' in col]
    years = [int(col.split()[0]) for col in unemployment_cols]
    years_sorted = sorted(zip(years, unemployment_cols), key=lambda x: x[0])
    years = [y[0] for y in years_sorted]
    unemployment_cols = [c[1] for c in years_sorted]
    
    # Extract 2-digit FIPS from 5-digit FIPS
    df = df.with_columns(
        pl.col('FIPS').cast(pl.Utf8).str.pad_start(5, '0').str.slice(0, 2).alias('state_fips')
    )
    
    # Calculate min and max across all data for consistent y-axis
    all_values = df.select(unemployment_cols).to_numpy().flatten()
    all_values = all_values[~np.isnan(all_values)]
    y_min = np.floor(np.min(all_values) * 2) / 2  # Round down to nearest 0.5
    y_max = np.ceil(np.max(all_values) * 2) / 2   # Round up to nearest 0.5
    
    # Get unique states
    states = df.select('state_fips').unique().sort('state_fips').to_series().to_list()
    
    # Create plots for each state
    for state_code in states:
        state_data = df.filter(pl.col('state_fips') == state_code)
        
        # Get state name
        state_name_df = state_fips.filter(pl.col('FIPS') == state_code)
        if state_name_df.height == 0:
            continue
        state_name = state_name_df.select('STATE').item()
        
        # Reshape data for plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot all counties in light blue
        first_county = True
        for row in state_data.iter_rows(named=True):
            fips_code = str(row['FIPS']).zfill(5)
            county_rates = [row[col] for col in unemployment_cols]
            if first_county:
                ax.plot(years, county_rates, color='#87CEEB', alpha=0.5, linewidth=0.8, label='County-level rates')
                first_county = False
            else:
                ax.plot(years, county_rates, color='#87CEEB', alpha=0.5, linewidth=0.8)
        
        # Calculate and plot state mean
        state_mean = state_data.select(unemployment_cols).mean()
        state_mean_values = state_mean.to_numpy().flatten()
        ax.plot(years, state_mean_values, color='#1f77b4', linewidth=2.5, label='State Mean')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
        ax.set_title(f'{state_name} Historical Unemployment Rates', fontsize=14, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save or show
        if save and save_path is not None:
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            filename = f'{state_name.lower().replace(" ", "_")}_{state_code}_unemployment_rate_trend.png'
            filepath = save_path_obj / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f'Saved: {filepath}')
        else:
            plt.show()
        
        plt.close()


def analyze_state_unemployment_summary(unemp_rates, state_fips_path='data/Raw/STATE_FIPS_CODES.txt', 
                                        save_path=None):
    """
    Generate state-level summary statistics for unemployment rates.
    
    Parameters:
    -----------
    unemp_rates : polars.LazyFrame or DataFrame
        Data with FIPS codes (5-digit) and columns named "20XX Unemployment" for years 10-22
    state_fips_path : str
        Path to the STATE_FIPS_CODES.txt file for mapping 2-digit FIPS to state names
    save_path : str or Path, optional
        File path to save the summary table as CSV. If None, just returns the DataFrame.
        
    Returns:
    --------
    polars.DataFrame
        Summary statistics table with one row per state
    """
    # Collect the lazy frame if needed
    if hasattr(unemp_rates, 'collect'):
        df = unemp_rates.collect()
    else:
        df = unemp_rates
    
    # Read state FIPS codes
    state_fips = pl.read_csv(state_fips_path)
    state_fips = state_fips.with_columns(
        pl.col('FIPS').cast(pl.Utf8).str.pad_start(2, '0')
    )
    
    # Extract years from column names
    unemployment_cols = [col for col in df.columns if 'Unemployment' in col]
    years = sorted([int(col.split()[0]) for col in unemployment_cols])
    unemployment_cols_sorted = [f'{year} Unemployment' for year in years]
    
    # Extract 2-digit FIPS from 5-digit FIPS
    df = df.with_columns(
        pl.col('FIPS').cast(pl.Utf8).str.pad_start(5, '0').str.slice(0, 2).alias('state_fips')
    )
    
    # Calculate statistics by state
    summary_list = []
    
    for state_code in sorted(df.select('state_fips').unique().to_series().to_list()):
        state_data = df.filter(pl.col('state_fips') == state_code)
        
        # Get state name
        state_name_df = state_fips.filter(pl.col('FIPS') == state_code)
        if state_name_df.height == 0:
            continue
        state_name = state_name_df.select('STATE').item()
        
        # Extract unemployment values
        unemp_values = state_data.select(unemployment_cols_sorted).to_numpy()
        
        # Overall statistics
        overall_mean = np.nanmean(unemp_values)
        overall_std = np.nanstd(unemp_values)
        overall_min = np.nanmin(unemp_values)
        overall_max = np.nanmax(unemp_values)
        overall_range = overall_max - overall_min
        
        # Yearly averages across all counties
        yearly_means = []
        for col in unemployment_cols_sorted:
            yearly_means.append(state_data.select(col).mean().item())
        
        # Average yearly spread (difference between max and min county each year)
        yearly_spreads = []
        for col in unemployment_cols_sorted:
            col_data = state_data.select(col).to_series()
            yearly_spreads.append(col_data.max() - col_data.min())
        avg_yearly_spread = np.mean(yearly_spreads)
        max_yearly_spread = np.max(yearly_spreads)
        
        # Trend: change from first to last year in state mean
        trend = yearly_means[-1] - yearly_means[0]
        
        # Peak year (year with highest state average)
        peak_year_idx = np.argmax(yearly_means)
        peak_year = years[peak_year_idx]
        peak_rate = yearly_means[peak_year_idx]
        
        # Count of counties
        n_counties = state_data.height
        
        summary_list.append({
            'State_FIPS': state_code,
            'State': state_name,
            'N_Counties': n_counties,
            'Mean_Unemployment': round(overall_mean, 2),
            'Std_Dev': round(overall_std, 2),
            'Min_Rate': round(overall_min, 2),
            'Max_Rate': round(overall_max, 2),
            'Overall_Range': round(overall_range, 2),
            'Avg_Yearly_Spread': round(avg_yearly_spread, 2),
            'Max_Yearly_Spread': round(max_yearly_spread, 2),
            'Trend_2010_to_2022': round(trend, 2),
            'Peak_Year': peak_year,
            'Peak_Rate': round(peak_rate, 2)
        })
    
    # Create DataFrame
    summary_df = pl.DataFrame(summary_list)
    
    # Save if path provided
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        summary_df.write_csv(save_path)
        print(f'Summary table saved to: {save_path}')
    
    return summary_df

# Just view the plots
# plot_state_unemployment_trends(unemp_rates)

# Save the plots to a directory
# plot_state_unemployment_trends(unemp_rates, save=True, save_path='unemployment_plots/')

# Just return the summary table
summary = analyze_state_unemployment_summary(unemp_rates)
print(summary)

# Save to CSV
# summary = analyze_state_unemployment_summary(unemp_rates, save_path='state_unemployment_summary.csv')