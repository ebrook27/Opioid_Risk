### 11/07/25, EB: Here I am adding plotting utilities functions for maps, feature importance plots, etc.
### 3/4/26, EB: Added the shapley value bar plot+dendrogram function and integrated it into the main pipeline.

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
import shap
from scipy.cluster.hierarchy import linkage
import numpy as np



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
        
    # --- Better titleing ---
    if value_col.lower() in ["mortality_rate", "mortality"]:
        title_suffix = "Mortality Rates"
    elif "abserror" in value_col.lower():
        if "EWMA" in value_col:
            title_suffix = "Exponentially-Weighted Absolute Error Risk Scores"
        else: 
            title_suffix = "Absolute Error Risk Scores"
    elif "RawError" in value_col.lower():
            if "EWMA" in value_col:
                title_suffix = "Exponentially-Weighted Signed-Difference Risk Scores"
            else: 
                title_suffix = "Signed-Difference Risk Scores"
    elif "SqError" in value_col.lower():
        if "EWMA" in value_col.lower():
            title_suffix = "Exponentially-Weighted Squared Error Risk Scores"
        else:
            title_suffix = "Squared Error Risk Scores"

    # --- Compute global color scaling across all years ---
    vals = merged[value_col]
    if vals.dropna().empty:
        raise ValueError(f"Column '{value_col}' contains no valid numeric values to plot.")

    if center_zero:
        global_max = vals.abs().max()
        vmin_global, vmax_global = -global_max, global_max
    else:
        vmin_global, vmax_global = vals.min(), vals.max()

    # --- Iterate over years ---
    for yr in years:
        subset = merged[merged["Year"] == yr].copy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        title = f"{title_prefix + ' — ' if title_prefix else ''}{title_suffix} ({yr})"
        ax.set_title(title, fontsize=14)

        # --- Use global color scaling for consistent comparison across years ---
        vmin, vmax = vmin_global, vmax_global

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
        # Use the human-friendly title suffix for the colorbar label
        cbar.set_label(title_suffix, fontsize=10)

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

    # Helper to choose a normalization. If a diverging cmap is requested
    # and the series spans negative and positive values, create a
    # TwoSlopeNorm centered at 0 so the neutral (white) color falls at 0.
    def _norm_for_series(series, cmap_name: str):
        s = series.dropna()
        if s.empty:
            return None, None, colors.Normalize(vmin=0.0, vmax=1.0)
        mn, mx = s.min(), s.max()
        if mn == mx:
            # avoid zero-range norm
            return mn, mx, colors.Normalize(vmin=mn - 1e-6, vmax=mx + 1e-6)

        diverging_cmaps = {
            "bwr",
            "seismic",
            "coolwarm",
            "RdBu",
            "PiYG",
            "PuOr",
            "BrBG",
        }

        if cmap_name in diverging_cmaps and mn < 0 and mx > 0:
            gmax = max(abs(mn), abs(mx))
            return -gmax, gmax, colors.TwoSlopeNorm(vmin=-gmax, vcenter=0.0, vmax=gmax)
        else:
            return mn, mx, colors.Normalize(vmin=mn, vmax=mx)

    # risk maps use the provided cmap_risk; EWMA map uses a sequential green cmap
    # by default, but if the user passed a diverging cmap (e.g., 'bwr') we
    # use the same diverging cmap for EWMA so centering at zero applies.
    vmin_risk, vmax_risk, norm_risk = _norm_for_series(merged[f"{error_col}_Risk"], cmap_risk)
    ewma_cmap = cmap_risk if cmap_risk in {"bwr", "seismic", "coolwarm", "RdBu"} else "Greens"
    vmin_ewma, vmax_ewma, norm_ewma = _norm_for_series(merged[f"{error_col}_EWMA_Risk"], ewma_cmap)

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
            norm=norm_risk,
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
            cmap=ewma_cmap,
            ax=axes[2],
            linewidth=0.075,
            edgecolor="lightgray",
            legend=True,
            norm=norm_ewma,
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


def plot_shap_bar_dendrogram(
    shap_df: pd.DataFrame,
    save_dir=None,
    *,
    by_year: bool = True,
    max_display: int = 25,
    linkage_method: str = "average",
    use_abs_corr: bool = True,
    dpi: int = 300,
):
    """
    Plot SHAP bar plot with clustered feature dendrogram.

    Assumes shap_df has EXACTLY these columns:
      ['FIPS','Year','Fold','BaseValue','Predicted','Feature','SHAP']

    Saves to: {save_dir}/shap/shap_bar_dendrogram_<YEAR>.png (or AllYears)
    """

    required = {"FIPS", "Year", "Fold", "BaseValue", "Predicted", "Feature", "SHAP"}
    missing = required - set(shap_df.columns)
    if missing:
        raise ValueError(f"shap_df missing columns: {missing}")

    df = shap_df.copy()
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["Year"] = df["Year"].astype(int)

    sns.set_theme(style="whitegrid")

    def _plot_one(sub: pd.DataFrame, label: str):
        # Pivot to wide: rows=(Year,Fold,FIPS), cols=Feature, values=SHAP
        wide = sub.pivot_table(
            index=["Year", "Fold", "FIPS"],
            columns="Feature",
            values="SHAP",
            aggfunc="mean",
        ).dropna(axis=1, how="all")

        if wide.shape[1] == 0:
            raise ValueError(f"No SHAP features found for label={label}")

        M = wide.to_numpy()  # (n_obs, n_feat)

        # Feature-feature correlation in SHAP space
        C = np.corrcoef(M, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

        D = 1.0 - (np.abs(C) if use_abs_corr else C)

        # condensed distance vector for scipy linkage
        iu = np.triu_indices(D.shape[0], k=1)
        d_condensed = D[iu]
        Z = linkage(d_condensed, method=linkage_method)

        expl = shap.Explanation(values=M, feature_names=list(wide.columns))

        plt.figure(figsize=(10, 6))
        shap.plots.bar(expl, max_display=max_display, clustering=Z, clustering_cutoff=1, show=False)
        plt.title(f"SHAP |mean| + dendrogram — {label}", fontsize=12)
        plt.tight_layout()

        if save_dir:
            out_dir = Path(save_dir) / "shap"
            out_dir.mkdir(parents=True, exist_ok=True)
            fp = out_dir / f"shap_bar_dendrogram_{label}.png"
            plt.savefig(fp, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {fp}")
        else:
            plt.show()

    if by_year:
        for yr in sorted(df["Year"].unique()):
            sub = df[df["Year"] == yr]
            if not sub.empty:
                _plot_one(sub, label=str(yr))
    else:
        _plot_one(df, label="AllYears")


def plot_temporal_shap_corr_heatmap(
    shap_source: str | Path | pd.DataFrame,
    save_dir: str | Path | None = None,
    *,
    years: list[int] | None = None,
    top_k_pairs: int = 20,
    use_abs_corr: bool = True,
    min_obs: int = 500,
    dpi: int = 300,
):
    """
    Temporal heatmap of feature-pair SHAP correlations.

    Expects SHAP long table with columns:
      ['FIPS','Year','Fold','Feature','SHAP'] (+ optionally BaseValue, Predicted)

    Output heatmap:
      rows = feature pairs (i|j)
      cols = years
      values = corr (or abs(corr)) between SHAP columns across counties

    Parameters
    ----------
    shap_source : path or DataFrame
        Path to shap_values.parquet OR the loaded shap_df.
    years : list[int] or None
        Restrict to these years.
    top_k_pairs : int
        Show only the K feature-pairs with largest mean |corr| across years.
        (Keeps the heatmap readable.)
    use_abs_corr : bool
        If True, heatmap shows |corr|. If False, shows signed corr.
    min_obs : int
        Minimum number of observations (counties) required to compute a year's corr matrix.
    """

    # ---- load ----
    if isinstance(shap_source, pd.DataFrame):
        df = shap_source.copy()
    else:
        p = Path(shap_source)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

    required = {"Year", "Fold", "FIPS", "Feature", "SHAP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SHAP table missing columns: {missing}")

    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["Year"] = df["Year"].astype(int)

    if years is not None:
        df = df[df["Year"].isin(years)].copy()

    yrs = sorted(df["Year"].unique())
    if not yrs:
        raise ValueError("No years available after filtering.")

    # ---- compute Ct per year, then stack upper triangles into (pair, year) ----
    pair_series = []
    pair_index = None

    for yr in yrs:
        sub = df[df["Year"] == yr]
        wide = sub.pivot_table(
            index=["Year", "Fold", "FIPS"],
            columns="Feature",
            values="SHAP",
            aggfunc="mean",
        ).dropna(axis=1, how="all")

        if wide.shape[0] < min_obs or wide.shape[1] < 2:
            continue

        M = wide.to_numpy()
        C = np.corrcoef(M, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

        if use_abs_corr:
            C = np.abs(C)

        feats = list(wide.columns)
        iu = np.triu_indices(len(feats), k=1)

        pairs = [f"{feats[i]} | {feats[j]}" for i, j in zip(iu[0], iu[1])]
        vals = C[iu]

        s = pd.Series(vals, index=pairs, name=yr)
        pair_series.append(s)

        if pair_index is None:
            pair_index = pairs

    if not pair_series:
        raise ValueError("No years met min_obs requirement; heatmap not created.")

    corr_by_year = pd.concat(pair_series, axis=1).sort_index()
    corr_by_year = corr_by_year.reindex(sorted(corr_by_year.columns), axis=1)

    # ---- select top K pairs by mean value across years ----
    mean_strength = corr_by_year.mean(axis=1).sort_values(ascending=False)
    keep = mean_strength.head(top_k_pairs).index
    heat = corr_by_year.loc[keep]

    # ---- plot ----
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(max(8, 0.8 * len(heat.columns)), max(6, 0.22 * len(heat))))
    # ax = sns.heatmap(
    #     heat,
    #     cmap="viridis",
    #     vmin=0.0,
    #     vmax=1.0,
    #     linewidths=0.2,
    #     linecolor="white",
    # )
    if use_abs_corr:
        ax = sns.heatmap(
            heat,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.2,
            linecolor="white",
        )
    else:
        ax = sns.heatmap(
            heat,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0,
            linewidths=0.2,
            linecolor="white",
        )
    title = "Temporal SHAP feature–feature correlation"
    title += " (abs)" if use_abs_corr else " (signed)"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Feature pair (i | j)")
    plt.tight_layout()

    if save_dir is not None:
        out_dir = Path(save_dir) / "shap"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / f"shap_temporal_corr_heatmap_top{top_k_pairs}.png"
        plt.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {fp}")
    else:
        plt.show()


def plot_shap_importance_trajectories(
    shap_source: str | Path | pl.DataFrame,
    save_dir: str | Path | None = None,
    *,
    top_k: int | None = None,
    dpi: int = 300,
):
    """
    Plot feature importance trajectories over time:
        importance(feature, year) = mean(|SHAP|) across counties.

    Expects long SHAP table with at least columns:
      ['Year','Feature','SHAP']
    """

    # ---- Load ----
    if isinstance(shap_source, pl.DataFrame):
        df = shap_source.clone()
    elif isinstance(shap_source, pd.DataFrame):
        df = pl.from_pandas(shap_source)
    else:
        p = Path(shap_source)
        if p.suffix.lower() == ".parquet":
            df = pl.read_parquet(p)
        else:
            df = pl.read_csv(p)

    df = df.with_columns([
        pl.col("Year").cast(pl.Int32),
        pl.col("Feature").cast(pl.Utf8),
        pl.col("SHAP").cast(pl.Float64),
        pl.col("SHAP").abs().alias("ABS_SHAP"),
    ])

    # importance per (year, feature)
    imp = (
        df.group_by(["Year", "Feature"])
          .agg(pl.col("ABS_SHAP").mean().alias("mean_abs_shap"))
          .sort(["Year", "mean_abs_shap"], descending=[False, True])
    )

    # Convert to pandas for seaborn lineplot (seaborn expects pandas)
    imp_pd = imp.to_pandas()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))

    ax = sns.lineplot(
        data=imp_pd,
        x="Year",
        y="mean_abs_shap",
        hue="Feature",
        marker="o",
    )
    ax.set_title("SHAP feature importance trajectories: mean(|SHAP|) vs year", fontsize=12)
    ax.set_ylabel("mean(|SHAP|)")
    ax.set_xlabel("Year")
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    if save_dir is not None:
        out_dir = Path(save_dir) / "shap"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / f"shap_importance_trajectories{'_top'+str(top_k) if top_k else ''}.png"
        plt.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {fp}")
    else:
        plt.show()


def plot_cohort_mean_variable_over_time(
    risk_scores: pl.DataFrame,
    df: pl.DataFrame,
    risk_col: str = "AbsError_Risk",
    target_col: str = "mortality_rate",
    cohorts: tuple = (0.5, 0.9, 1.0),
    labels: tuple | None = None,
    save_dir: str | None = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
):
    """
    Split counties into annual cohorts by risk score and plot mean target variable over time.

    Parameters
    ----------
    risk_scores : pl.DataFrame
        Polars DataFrame with at least ['FIPS', 'Year', risk_col].
    df : pl.DataFrame
        Polars DataFrame containing observed `target_col` by ['FIPS','Year'].
    risk_col : str
        Column name in `risk_scores` containing the risk metric to rank counties.
    target_col : str
        Column name in `df` containing the observed target variable.
    cohorts : tuple
        Increasing quantile cutpoints (e.g., (0.5, 0.9, 1.0) -> bottom 50%, mid 40%, top 10%).
    labels : tuple
        Labels for the resulting cohorts (length must be len(cohorts)).
    save_dir : str or None
        Directory to save the plot PNG. If None, shows interactively.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution for saved PNG.

    Notes
    -----
    - Cohorts are computed separately for each year (counties may shift cohorts over time).
    """

    # --- Convert to pandas for convenience with q-cut and plotting ---
    rs = risk_scores.to_pandas() if isinstance(risk_scores, pl.DataFrame) else risk_scores.copy()
    df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()

    # --- Normalize column names and dtypes ---
    for d in (rs, df_pd):
        if "FIPS" in d.columns:
            d["FIPS"] = d["FIPS"].astype(str).str.zfill(5)
        # Accept either 'Year' or 'year'
        if "year" in d.columns and "Year" not in d.columns:
            d.rename(columns={"year": "Year"}, inplace=True)

    # Validate required columns
    if "FIPS" not in rs.columns or "Year" not in rs.columns or risk_col not in rs.columns:
        raise ValueError("risk_scores must contain 'FIPS', 'Year', and the specified risk_col")
    if "FIPS" not in df_pd.columns or "Year" not in df_pd.columns or target_col not in df_pd.columns:
        raise ValueError("df must contain 'FIPS', 'Year', and the specified target_col")

    # --- Infer labels from cohorts if not provided ---
    if labels is None:
        # cohorts is a tuple of increasing quantiles (e.g., (0.5, 0.9, 1.0))
        try:
            cohort_qs = [float(q) for q in cohorts]
        except Exception:
            raise ValueError("`cohorts` must be numeric quantiles between 0 and 1")

        if any(q <= 0 or q > 1 for q in cohort_qs):
            raise ValueError("Quantile values in `cohorts` must be in the interval (0, 1].")

        if any(cohort_qs[i] <= cohort_qs[i - 1] for i in range(1, len(cohort_qs))):
            raise ValueError("`cohorts` must be strictly increasing (e.g., (0.5, 0.9, 1.0)).")

        # widths in percentage for each cohort
        widths = []
        prev = 0.0
        for q in cohort_qs:
            widths.append((q - prev) * 100.0)
            prev = q

        # default names: try to be descriptive for common cases
        n = len(widths)
        if n == 1:
            names = ["Top"]
        elif n == 2:
            names = ["Bottom", "Top"]
        elif n == 3:
            names = ["Bottom", "Middle", "Top"]
        else:
            names = [f"Cohort {i+1}" for i in range(n)]

        def fmt_pct(p):
            return f"{p:.0f}%" if p >= 1.0 else f"{p:.1f}%"

        labels = tuple(f"{names[i]} {fmt_pct(widths[i])}" for i in range(n))

    # Merge risk + outcome
    merged = rs[["FIPS", "Year", risk_col]].merge(
        df_pd[["FIPS", "Year", target_col]], on=["FIPS", "Year"], how="left"
    )

    years = sorted(merged["Year"].dropna().unique())
    if not years:
        raise ValueError("No valid Year values found after merging risk_scores and df")

    records = []
    for yr in years:
        sub = merged[merged["Year"] == yr].dropna(subset=[risk_col, target_col]).copy()
        if sub.empty:
            continue

        # compute quantiles
        q_low = sub[risk_col].quantile(cohorts[0]) # Seems to think the tuple might not have index 0? But I'll never pass an empty tuple, so this should be fine.
        q_mid = sub[risk_col].quantile(cohorts[1]) if len(cohorts) > 1 else q_low

        bins = [-float("inf"), q_low, q_mid, float("inf")]
        # assign cohorts: expect len(labels)==3 for default; allow labels length to match bins-1
        if len(labels) != (len(bins) - 1):
            raise ValueError("Number of labels must equal number of cohort bins")

        sub["cohort"] = pd.cut(sub[risk_col], bins=bins, labels=labels, include_lowest=True)

        grp = sub.groupby("cohort")[target_col].mean().reset_index()
        grp["Year"] = yr
        records.append(grp)

    if not records:
        raise ValueError("No cohort records computed — check input data and column names")

    result = pd.concat(records, axis=0, ignore_index=True)

    # Pivot for plotting
    pivot = result.pivot(index="Year", columns="cohort", values=target_col)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=pivot, markers=True)
    ax.set_title(f"Mean {target_col.replace('_', ' ').title()} by Risk Cohort: {labels} of {risk_col}", fontsize=12)
    ax.set_ylabel(f"Mean {target_col.replace('_', ' ').title()}")
    ax.set_xlabel("Year")
    ax.legend(title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / f"cohort_mean_{target_col.replace('_', ' ').title().lower().replace(' ', '_')}_over_time.png"
        plt.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved cohort mean {target_col.replace('_', ' ').title().lower().replace(' ', '_')} plot: {fp}")
    else:
        plt.show()
