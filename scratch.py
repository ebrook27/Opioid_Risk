### 3/4/26, EB: Treating this a scratch notebook file for quick computations and visualizations. 
### I'm trying to understand the relationships between SHAP values for different features.
### I'm going to compute the correlation matrix of SHAP values across features, and then compute a distance matrix based on that correlation.
### This will help me understand which features have similar SHAP patterns across counties and years, 
### which could indicate that they are capturing similar underlying effects in the model.

# import pandas as pd
# import numpy as np
# from pathlib import Path

# # ---- load shap table ----
# fp = Path('model_outputs/xgbregressor/2026-03-04_14-15-39/shap_values.parquet')
# df = pd.read_parquet(fp)
# print(df.head(20))
# print(df.columns)

# # choose one year to inspect
# year = 2015
# sub = df[df["Year"] == year]

# # ---- pivot to SHAP matrix ----
# M = sub.pivot_table(
#     index=["Year", "Fold", "FIPS"],
#     columns="Feature",
#     values="SHAP",
#     aggfunc="mean"
# )
# print("SHAP matrix shape, no dropped NA:", M.shape)

# M = M.dropna(axis=1, how="all")

# print("SHAP matrix shape, dropped NA:", M.shape)

# # ---- feature correlation matrix ----
# corr = np.corrcoef(M.to_numpy(), rowvar=False)

# corr_df = pd.DataFrame(
#     corr,
#     index=M.columns,
#     columns=M.columns
# )

# print("\nFeature–feature SHAP correlations:")
# print(corr_df.round(2))

# # ---- compute clustering distance ----
# dist = 1 - np.abs(corr)

# dist_df = pd.DataFrame(
#     dist,
#     index=M.columns,
#     columns=M.columns
# )

# # ---- show strongest correlations ----
# pairs = []

# features = list(M.columns)

# for i in range(len(features)):
#     for j in range(i + 1, len(features)):
#         pairs.append(
#             (features[i], features[j], corr[i, j])
#         )

# pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

# print("\nTop SHAP correlations:")
# for p in pairs[:10]:
#     print(f"{p[0]:25s} {p[1]:25s} {p[2]:.3f}")

# # ---- distance summary ----
# d = dist[np.triu_indices(len(features), 1)]

# print("\nDistance statistics:")
# print("min distance:", d.min())
# print("mean distance:", d.mean())
# print("max distance:", d.max())

# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

### 3/7/26, EB: Got weird results when using DLNM with county fixed effects. Trying to understand why.
### I'm going to plot MR vs the four prioir years' unemployment rates to see if there's any obvious patterns.

import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import src.data_processing as data_proc

# Load data
data = data_proc.CountyDataLoader()
df = data.load()
# Precovid period only for now
# df = df.filter(pl.col("year") < 2020)
df = df.sort(["FIPS", "year"])
# print(df.select(["unemp_rate", "mortality_rate"]).describe())


print(df.head(10))
print(df.columns)


# mort_unemp = df.select("mortality_rate", "unemp_rate", "year", "FIPS")


# # Prepare list of years present
# years = sorted(mort_unemp["year"].unique())
# if years:
#     min_year = min(years)
#     max_year = max(years)
# else:
#     min_year = 0
#     max_year = 0

# # For each target year >= 2017, plot mortality (target year) vs unemployment
# # from the three prior years (target_year-3, -2, -1), colored by unemployment year.
# for target_year in range(2017, max_year + 1):
#     records = []
#     for lag in (3, 2, 1):
#         unemp_year = target_year - lag
#         if unemp_year < min_year:
#             continue

#         un = mort_unemp.filter(pl.col("year") == unemp_year).select(["FIPS", "unemp_rate"])
#         mort = mort_unemp.filter(pl.col("year") == target_year).select(["FIPS", "mortality_rate"])

#         joined = mort.join(un, on="FIPS", how="inner")
#         if joined.height == 0:
#             continue
#         joined = joined.with_columns(pl.lit(unemp_year).alias("unemp_year"))
#         records.append(joined)

#     if not records:
#         continue

#     plot_df = pl.concat(records).to_pandas()
#     print(plot_df.head(10))
#     sns.scatterplot(data=plot_df, x="unemp_rate", y="mortality_rate", hue="unemp_year", alpha=0.6)
#     plt.title(f"Mortality vs prior-year unemployment — target year {target_year}")
#     plt.xlabel("Unemployment rate (prior year)")
#     plt.ylabel("Mortality rate")
#     plt.show()
### The above worked, but didn't offer a great insight. I'm going to plot something different now,
### rather than plotting mortality vs unemployment directly, I'm going to demean the variables at the county
### level, and plot that for each year.

df = df.with_columns(
    pl.col("mortality_rate").mean().over("FIPS").alias("mortality_rate_county_mean"),
    pl.col("unemp_rate").mean().over("FIPS").alias("unemp_rate_county_mean")
)

df = df.with_columns(
    (pl.col("mortality_rate") - pl.col("mortality_rate_county_mean")).alias("mortality_rate_demeaned"),
    (pl.col("unemp_rate") - pl.col("unemp_rate_county_mean")).alias("unemp_rate_demeaned")
)

df = df.with_columns(
    pl.col("mortality_rate_demeaned").shift(-1).over("FIPS").alias("mortality_rate_demeaned_lead")
)

# plot_df = df.drop_nulls(["unemp_rate_demeaned", "mortality_rate_demeaned_lead"]).to_pandas()

# sns.scatterplot(
#     data=plot_df,
#     x="unemp_rate_demeaned",
#     y="mortality_rate_demeaned_lead",
#     alpha=0.4
# )

# plt.axhline(0,color="black",linewidth=1)
# plt.axvline(0,color="black",linewidth=1)

# plt.xlabel("Within-county unemployment deviation")
# plt.ylabel("Within-county mortality deviation (t+1)")
# plt.title("Within-county relationship (FE equivalent)")
# plt.show()


for lag in [0, 1, 2, 3]:
    df = df.with_columns(
        pl.col("unemp_rate_demeaned")
        .shift(lag)
        .over("FIPS")
        .alias(f"unemp_rate_demeaned{lag}")
    )

    plotting_df = (
        df.select(
            f"unemp_rate_demeaned{lag}",
            "mortality_rate_demeaned_lead"
        )
        .drop_nulls()
        .to_pandas()
    )

    plt.figure(figsize=(8,6))
    sns.regplot(
        data=plotting_df,
        x=f"unemp_rate_demeaned{lag}",
        y="mortality_rate_demeaned_lead",
        scatter_kws={"alpha":0.2},
        line_kws={"color":"red"}
    )
    plt.title(f"Within-county Unemployment lag {lag+1} vs next-year Mortality deviation")#, Pre-Covid")
    plt.xlabel(f"Demeaned unemployment, lag {lag+1}")
    plt.ylabel("Demeaned mortality, lead 1")
    plt.show()
    
### 3/9/26, EB: The above loop plots the county demeaned mortality rate at lead 1 (i.e., next year) against the county demeaned unemployment rate at lags 0, 1, 2, and 3.
### Now I'm going to plot just plain mean mortality vs mean unemployment for each lag, to see if there's any obvious patterns there. This checks between counties,
### now within like we did above.
df_means = df.group_by("FIPS").agg(
    unemp_mean = pl.col("unemp_rate").mean(),
    mort_mean = pl.col("mortality_rate").mean()
)

plot_df = df_means.to_pandas()
sns.regplot(
    data=plot_df,
    x="unemp_mean",
    y="mort_mean",
    scatter_kws={"alpha":0.5},
    line_kws={"color":"red"}
)

plt.xlabel("Mean unemployment rate (county)")
plt.ylabel("Mean mortality rate (county)")
plt.title("Between-county relationship: unemployment vs mortality")#, Pre-Covid")
plt.show()
### As expected, the between-relationship is stronger, but also what we might expect: higher unemployment counties have higher mortality rates.
### This suggests that our DLNM results aren't just picking up on some weird artifact of the data, but are likely picking up on a real relationship 
### between unemployment and mortality that exists both within and between counties. This means that the story we have to tell about unemployment and mortality
### is more complicated than the usual unemployment -> mortality story, and that the story is a matter of scale. A county's unemployment rate, relative to its own
### long-run mean (i.e., the within-county relationship) has a different relationship to mortality than 





### Ok the lag plots just above seem to show that the DLNM is picking up on legit behavior between unemployment and mortality, which is reassuring if strange.
### What I'm going to do now is fit a linear fixed effects regression for each lag, and plot the coefficients as a function of lag.
### This will help me understand the temporal pattern of the relationship between a variable and mortality.


def plot_within_county_lags(
    df,
    var: str,
    outcome: str = "mortality_rate",
    lags=(0, 1, 2, 3),
    county_col: str = "FIPS",
    lead: int = 1,
    figsize=(8, 6),
):
    # compute county means and demean the variable and outcome
    df = df.with_columns(
        pl.col(outcome).mean().over(county_col).alias(f"{outcome}_county_mean"),
        pl.col(var).mean().over(county_col).alias(f"{var}_county_mean"),
    )

    df = df.with_columns(
        (pl.col(outcome) - pl.col(f"{outcome}_county_mean")).alias(f"{outcome}_demeaned"),
        (pl.col(var) - pl.col(f"{var}_county_mean")).alias(f"{var}_demeaned"),
    )

    # create lead for outcome
    df = df.with_columns(
        pl.col(f"{outcome}_demeaned").shift(-lead).over(county_col).alias(f"{outcome}_demeaned_lead")
    )

    for lag in lags:
        df = df.with_columns(
            pl.col(f"{var}_demeaned").shift(lag).over(county_col).alias(f"{var}_demeaned{lag}")
        )

        plotting_df = (
            df.select(
                f"{var}_demeaned{lag}",
                f"{outcome}_demeaned_lead",
            )
            .drop_nulls()
            .to_pandas()
        )

        plt.figure(figsize=figsize)
        sns.regplot(
            data=plotting_df,
            x=f"{var}_demeaned{lag}",
            y=f"{outcome}_demeaned_lead",
            scatter_kws={"alpha": 0.2},
            line_kws={"color": "red"},
        )
        plt.title(f"Within-county {var} lag {lag+1} vs next-year {outcome} deviation")
        plt.xlabel(f"Demeaned {var}, lag {lag+1}")
        plt.ylabel(f"Demeaned {outcome}, lead {lead}")
        plt.show()


# Call the helper for unemployment rate
# plot_within_county_lags(df, "unemp_rate")
plot_within_county_lags(df, "rx_rate")
plot_within_county_lags(df, "uninsured_rate")



#--------------------------------------------------------------------------------------------------------------------------------------------
### Trying to fit a simple distributed lag regression model, to more adequately justify my use of DLNM to model the relationship
### between unemployment and mortality.