### 11/07/25, EB: Here I am adding plotting utilities functions for maps, feature importance plots, etc.

import os
from typing import Union
import pandas as pd
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, cm
import seaborn as sns
from pathlib import Path



def plot_county_metric_maps(
    df,
    value_col,
    save_dir=None,
    cmap="Reds",
    center_zero=False,
    filter_CONUS=True,
    title_prefix=None,
    dpi=300,
):
    """
    Plot county-level choropleth maps for any metric (e.g. mortality rate, prediction error, risk score).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ['FIPS', 'Year', value_col].
    shapefile_path : str
        Path to county shapefile (e.g., 'Data/cb_2022_us_county_20m.geojson').
    value_col : str
        Column to visualize (e.g., 'mortality_rate', 'AbsError_Risk', 'rx_rate').
    out_dir : str or None
        Directory to save output images. If None, maps will be shown interactively.
    cmap : str
        Matplotlib colormap name (e.g. 'Reds', 'coolwarm', 'bwr', 'viridis').
    center_zero : bool
        If True, centers the colorbar at zero (useful for signed quantities like RawError).
    filter_CONUS : bool
        Whether to exclude Alaska, Hawaii, and Puerto Rico.
    title_prefix : str or None
        Optional prefix for plot titles (e.g., "Model Error", "Risk Score").
    dpi : int
        Resolution for saved figures.

    Returns
    -------
    None
    """

    # --- Load shapefile ---
    print(f"📂 Loading shapefile")
    gdf = gpd.read_file("data/Processed/2022_County_Shapefile/2022_filtered_shapefile.shp")
    gdf["FIPS"] = gdf["GEOID"].astype(str).str.zfill(5)

    # --- Optional: filter to CONUS only (exclude AK, HI, PR) ---
    if filter_CONUS:
        exclude_prefixes = ("02", "15", "72")  # AK, HI, PR
        gdf = gdf[~gdf["FIPS"].str.startswith(exclude_prefixes)].copy()

    # --- Ensure data has FIPS + Year + value_col ---
    df = df.copy()
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)

    if "Year" not in df.columns:
        raise ValueError("Input dataframe must contain a 'Year' column.")

    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in dataframe.")

    # --- Merge metric onto shapefile ---
    merged = gdf.merge(df[["FIPS", "Year", value_col]], on="FIPS", how="left")

    years = sorted(merged["Year"].dropna().unique())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- Iterate over years ---
    for yr in years:
        subset = merged[merged["Year"] == yr].copy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        title = f"{title_prefix + ' — ' if title_prefix else ''}{value_col} ({yr})"
        ax.set_title(title, fontsize=14)

        # --- Compute color scaling ---
        if center_zero:
            vmax = subset[value_col].abs().max()
            vmin = -vmax
        else:
            vmin, vmax = subset[value_col].min(), subset[value_col].max()

        # --- Plot choropleth ---
        subset.plot(
            column=value_col,
            cmap=cmap,
            linewidth=0,
            edgecolor="none",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            legend=False,
        )

        # --- Add state outlines ---
        if "STATEFP" in subset.columns:
            states = subset.dissolve(by="STATEFP", as_index=False)
            states.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=2)

        # --- Add colorbar ---
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(value_col, fontsize=10)

        ax.axis("off")
        ax.set_aspect("equal")
        plt.tight_layout()

        # --- Save or show ---
        if save_dir:
            plots_dir = Path(save_dir) / "maps"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fname = plots_dir / f"{value_col}_Map_{yr}.png"
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")
            print(f"✅ Saved map: {fname}")
            plt.close()
        else:
            plt.show()

def plot_yearly_feature_importances(
    feature_importance_df,
    save_dir=None,
    top_n=None,
    figsize=(8, 6),
    dpi=300,
    palette="viridis",
    model_name=None
):
    """
    Plot ranked bar charts of average feature importances for each year.

    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        Must contain ['Feature', 'Importance', 'Year', 'Fold'].
    out_dir : str or None
        Directory to save PNGs. If None, shows interactively.
    top_n : int or None
        If provided, limits to the top N most important features per year.
    figsize : tuple
        Figure size for each yearly plot.
    dpi : int
        Image resolution.
    palette : str
        Seaborn/Matplotlib color palette (e.g. 'viridis', 'crest', 'mako', 'coolwarm').
    model_name : str or None
        Optional model name for titles / filenames.
    """

    # --- Validate dataframe ---
    required_cols = {"Feature", "Importance", "Year", "Fold"}
    if not required_cols.issubset(feature_importance_df.columns):
        raise ValueError(
            f"DataFrame must contain columns {required_cols}, got {feature_importance_df.columns.tolist()}"
        )

    if feature_importance_df.empty:
        print("⚠️ Feature importance dataframe is empty. Skipping plot.")
        return

    # --- Average over folds ---
    avg_importance = (
        feature_importance_df
        .groupby(["Year", "Feature"], as_index=False)["Importance"]
        .mean()
    )

    years = sorted(avg_importance["Year"].unique())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for yr in years:
        df_year = avg_importance[avg_importance["Year"] == yr].copy()
        df_year = df_year.sort_values("Importance", ascending=False)

        if top_n is not None:
            df_year = df_year.head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(
            data=df_year,
            y="Feature",
            x="Importance",
            palette=palette,
            order=df_year["Feature"]
        )

        model_str = f" ({model_name})" if model_name else ""
        plt.title(f"Average Feature Importance — {yr}{model_str}", fontsize=14)
        plt.xlabel("Mean Importance (across folds)", fontsize=12)
        plt.ylabel("")
        plt.tight_layout()

        if save_dir:
            plots_dir = Path(save_dir) / "feature_importances"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fname = plots_dir / f"Feature_Importance_{yr}{'_' + model_name if model_name else ''}.png"
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {fname}")
        else:
            plt.show()


def plot_triple_metric_maps(
    df: Union[pl.DataFrame, pd.DataFrame],
    risk_scores: pd.DataFrame,
    save_dir: str | None = None,
    error_col: str = "AbsError",
    cmap_risk: str = "Reds",
    dpi: int = 300,
    filter_CONUS: bool = True,
    model_name: str | None = None,
):
    """
    Plot 3 side-by-side maps per year:
        (1) Mortality rate
        (2) Equal-weighted risk
        (3) EWMA risk

    Parameters
    ----------
    df : pd.DataFrame or Polars DataFrame
        Dataset containing ['FIPS', 'year', 'mortality_rate'].
    risk_scores : pd.DataFrame
        Output from `compute_all_risk_scores()`; must include ['FIPS', 'Year', '{error_col}_Risk', '{error_col}_EWMA_Risk'].
    save_dir : str or None, optional
        Directory to save plots. If None, maps are displayed interactively.
    error_col : str, default="AbsError"
        Base name of error column used for risk scores.
    cmap_risk : str, default="Reds"
        Colormap for risk visualizations.
    dpi : int, default=300
        Resolution for saved figures.
    filter_CONUS : bool, default=True
        Exclude Alaska, Hawaii, and Puerto Rico.

    Notes
    -----
    - Shapefile path is fixed to your repo convention.
    - start_year and end_year inferred automatically from risk_scores['Year'].
    """

    # --- Load shapefile (fixed path for this repo) ---
    shapefile_path = Path("data/Processed/2022_County_Shapefile/2022_filtered_shapefile.shp")
    print(f"📂 Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    gdf["FIPS"] = gdf["GEOID"].astype(str).str.zfill(5)

    # --- Optional: filter to CONUS only ---
    if filter_CONUS:
        exclude_prefixes = ("02", "15", "72")  # AK, HI, PR
        gdf = gdf[~gdf["FIPS"].str.startswith(exclude_prefixes)].copy()

    # --- Compute state boundaries once for overlay ---
    if "STATEFP" in gdf.columns:
        state_boundaries = gdf.dissolve(by="STATEFP", as_index=False)
    else:
        state_boundaries = None

    # --- Prepare mortality + risk data ---
    df = df.to_pandas() if not isinstance(df, pd.DataFrame) else df
    
    # --- Normalize column names for consistency ---
    df.columns = [c.capitalize() if c.lower() == "year" else c for c in df.columns]
    risk_scores.columns = [c.capitalize() if c.lower() == "year" else c for c in risk_scores.columns]
    
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    risk_scores["FIPS"] = risk_scores["FIPS"].astype(str).str.zfill(5)

    merged = (
        gdf.merge(df[["FIPS", "Year", "mortality_rate"]], on="FIPS", how="left")
           .merge(risk_scores, on=["FIPS", "Year"], how="left")
    )

    # --- Infer year range automatically ---
    years = sorted(merged["Year"].dropna().unique())
    if not years:
        raise ValueError("No valid 'Year' values found in merged dataset.")
    start_year, end_year = years[0], years[-1]

    print(f"🗺️ Generating triple maps for {start_year}–{end_year} ({len(years)} years)")

    # --- Compute global colorbar scales for consistency ---
    vmin_mort, vmax_mort = merged["mortality_rate"].min(), merged["mortality_rate"].max()
    vmin_risk, vmax_risk = merged[f"{error_col}_Risk"].min(), merged[f"{error_col}_Risk"].max()
    vmin_ewma, vmax_ewma = merged[f"{error_col}_EWMA_Risk"].min(), merged[f"{error_col}_EWMA_Risk"].max()

    # --- Ensure save directory exists if requested ---
    save_path: Path | None = None
    if save_dir is not None:
        save_path = Path(save_dir) / "maps"
        save_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        subset = merged[merged["Year"] == year].copy()
        if subset.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # (1) Mortality rate (always Reds)
        subset.plot(
            column="mortality_rate",
            cmap="Reds",
            ax=axes[0],
            linewidth=0.075,
            edgecolor="lightgray",
            legend=True,
            vmin=vmin_mort,
            vmax=vmax_mort,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.05,
                "label": "Mortality Rate",
            },
        )
        if state_boundaries is not None:
            state_boundaries.boundary.plot(ax=axes[0], color="black", linewidth=0.4, zorder=2)
        axes[0].set_title(f"Mortality Rate — {year}")
        axes[0].axis("off")

        # (2) Equal-weighted risk
        subset.plot(
            column=f"{error_col}_Risk",
            cmap=cmap_risk,
            ax=axes[1],
            linewidth=0.075,
            edgecolor="lightgray",
            legend=True,
            vmin=vmin_risk,
            vmax=vmax_risk,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.05,
                "label": f"Equal-weight {error_col} Risk",
            },
        )
        if state_boundaries is not None:
            state_boundaries.boundary.plot(ax=axes[1], color="black", linewidth=0.4, zorder=2)
        # axes[1].set_title(f"Equal-weight {error_col} Risk — {year}")
        axes[1].set_title(f"Equal-weight Risk — {year}, Model: {model_name}" if model_name else f"Equal-weight Risk — {year}")
        axes[1].axis("off")

        # (3) EWMA risk
        subset.plot(
            column=f"{error_col}_EWMA_Risk",
            cmap='Greens',
            ax=axes[2],
            linewidth=0.075,
            edgecolor="lightgray",
            legend=True,
            vmin=vmin_ewma,
            vmax=vmax_ewma,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.05,
                "label": f"EWMA {error_col} Risk",
            },
        )
        if state_boundaries is not None:
            state_boundaries.boundary.plot(ax=axes[2], color="black", linewidth=0.4, zorder=2)
        # axes[2].set_title(f"EWMA {error_col} Risk — {year}")
        axes[2].set_title(f"Exponentially-Weighted Risk — {year}, Model: {model_name}" if model_name else f"Exponentially-Weighted Risk — {year}")
        axes[2].axis("off")

        # This was causing an issue with the initial 'constrained_layout=True' above
        # plt.tight_layout() 

        # --- Save or show ---
        if save_path is not None:
            fname = save_path / f"TripleMap_{error_col}_{year}.png"
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {fname}")
        else:
            plt.show()

