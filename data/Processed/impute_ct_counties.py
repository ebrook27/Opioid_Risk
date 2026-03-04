### 2/3/26, EB: Need to align new datasets to the correct CT county structure for our modelling to be correct.
### This script imputes missing counties in Connecticut datasets to ensure consistency, using the Tobler package for spatial operations.

import re
import pandas as pd
import polars as pl
import geopandas as gpd
from tobler.area_weighted import area_interpolate

# ============================================================
# ========================= CONFIG ===========================
# ============================================================

UNEMP_INPUT_CSV  = "data/Processed/Unemployment/Unemployment_rates.csv"
UNEMP_OUTPUT_CSV = "data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv"
UNEMP_VALUE_NAME = "Unemployment"     # e.g. "Unemployment", "RX Rate", "Uninsured"

RX_INPUT_CSV = "data/Processed/Prescriptions/Prescription_dispensing_rates.csv"
RX_OUTPUT_CSV = "data/Processed/Prescriptions/Prescription_dispensing_rates_CT_fixed.csv"
RX_VALUE_NAME = "DR"

FIPS_COL   = "FIPS"

# Shapefiles
OLD_SHAPEFILE_PATH =  "data/Processed/2020_USA_County_Shapefile/Filtered Files/2020_filtered_shapefile.shp"
NEW_SHAPEFILE_PATH = "data/Processed/2022_County_Shapefile/2022_filtered_shapefile.shp"

SHAPE_ID_COL = "FIPS"           # change to "GEOID" if needed
TOBLER_EPSG  = 26918             # UTM 18N (good for CT)

# Historic CT counties (old structure)
CT_OLD_FIPS = ["09001","09003","09005","09007","09009","09011","09013","09015"]
CT_NEW_FIPS = ["09110", "09120", "09130", "09140", "09150", "09160", "09170", "09180", "09190"]

# ============================================================


def fix_connecticut_rate_only(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    df_year columns:
      - FIPS
      - value   (intensive variable)
    """
    old_shp = gpd.read_file(OLD_SHAPEFILE_PATH)
    new_shp = gpd.read_file(NEW_SHAPEFILE_PATH)

    # Normalize IDs
    old_shp[SHAPE_ID_COL] = old_shp[SHAPE_ID_COL].astype(str).str.zfill(5)
    new_shp[SHAPE_ID_COL] = new_shp[SHAPE_ID_COL].astype(str).str.zfill(5)

    # CT only
    old_ct = old_shp[old_shp[SHAPE_ID_COL].str.startswith("09")].to_crs(epsg=TOBLER_EPSG)
    new_ct = new_shp[new_shp[SHAPE_ID_COL].str.startswith("09")].to_crs(epsg=TOBLER_EPSG)

    # Merge data onto old CT geometry
    ct_old = df_year[df_year["FIPS"].isin(CT_OLD_FIPS)].copy()    
    if ct_old.empty:
        return df_year.sort_values("FIPS").reset_index(drop=True)
    
    old_ct = old_ct.merge(ct_old, how="left", left_on=SHAPE_ID_COL, right_on="FIPS")

    # Tobler interpolation (INTENSIVE variable)
    interp = area_interpolate(
        source_df=old_ct,
        target_df=new_ct,
        intensive_variables=["value"]
    )
    ### Try extensive, because the rates depend on population?
    
    # Attach new FIPS
    interp["FIPS"] = new_ct[SHAPE_ID_COL].values
    fixed_ct = interp[["FIPS", "value"]].copy()

    # Replace CT rows in original
    rest = df_year[~df_year["FIPS"].str.startswith("09")].copy()
    out = pd.concat([rest, fixed_ct], ignore_index=True)

    return out.sort_values("FIPS").reset_index(drop=True)


def CT_cleaning_imputing(
    input_csv: str,
    output_csv: str,
    value_name: str,
    fips_col: str = "FIPS",
    round_decimals: int | None = None,
    missing_values: list | None = None,
) -> None:
    """
    Generic CT fixer for wide-format CSVs:
      - one row per FIPS
      - columns like "2010 Unemployment" or "Unemployment 2010" (also tolerates underscores)
    Interprets the measurement as INTENSIVE and interpolates via Tobler.
    """
    
    # Load data
    df = pd.read_csv(input_csv, dtype={fips_col: str})
    # Normalize FIPS to 5-char strings
    df[fips_col] = df[fips_col].str.zfill(5)

    # Identify year columns (handles "2010 Unemployment" OR "Unemployment 2010")
    year_cols = []
    for c in df.columns:
        if c == fips_col:
            continue
        c_norm = re.sub(r"\s+", " ", str(c).strip().replace("_", " "))
        # Check if column name contains a year and the variable name
        has_year = re.search(r"(19|20)\d{2}", c_norm) is not None
        has_var  = value_name.lower() in c_norm.lower()
        
        if has_year and has_var:
            year_cols.append(c) # keep original column name
            
    if not year_cols:
        print("DEBUG: value_name =", value_name)
        print("DEBUG: columns =", list(df.columns))
        raise ValueError("No matching year columns found.")


    fixed_series = {}

    for col in sorted(year_cols):
        # Extract year (first 4-digit number)
        year = int(re.search(r"(19|20)\d{2}", col).group())

        # Build a year-specific long df
        df_year = df[[fips_col, col]].copy()
        df_year.columns = ["FIPS", "value"] # <- "value" refers to the variable we're fixing/imputing
        
        # Clean / numeric conversion
        if missing_values:
            df_year["value"] = df_year["value"].replace(missing_values, pd.NA)
        
        # Force numeric
        df_year["value"] = pd.to_numeric(df_year["value"], errors="coerce")

        # Apply CT fix
        df_fixed = fix_connecticut_rate_only(df_year)
        
        # Optional rounding (formatting, not modeling)
        if round_decimals is not None:
            df_fixed["value"] = df_fixed["value"].round(round_decimals)
        
        # Store as Series keyed by FIPS
        fixed_series[col] = df_fixed.set_index("FIPS")["value"]

        print(f"CT fix complete for {value_name}, year {year}")

    # Rebuild wide dataframe with union of all FIPS from fixed years
    all_fips = sorted(set().union(*[s.index for s in fixed_series.values()]))
    out = pd.DataFrame({"FIPS": all_fips})

    for col, s in fixed_series.items():
        out[col] = out["FIPS"].map(s)

    print(out[out["FIPS"].isin(CT_NEW_FIPS)][["FIPS"]])
    print(out[out["FIPS"].isin(CT_OLD_FIPS)][["FIPS"]])
    print(out[out["FIPS"].str.startswith("09")].sort_values("FIPS").head(20))

    out.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"CT rows in output: {sum(out['FIPS'].str.startswith('09'))}")

def ct_fix_long_parquet(
    parquet_in: str,
    parquet_out: str,
    fips_col: str,
    year_col: str,
    value_col: str,
    round_decimals: int | None = None,
):
    '''
    CT fixing for long-format parquet files:
      - columns: FIPS, year, value_col
    Specifically created for Uninsured Rates, which are stored as a long-format parquet,
    not a wide-format CSV (as most other data is).
    '''
    
    df = pl.read_parquet(parquet_in)

    # normalize FIPS to 5-char strings (important for joins / CT detection)
    df = df.with_columns(
        pl.col(fips_col).cast(pl.Utf8).str.zfill(5).alias(fips_col),
        pl.col(year_col).cast(pl.Int32).alias(year_col),
        pl.col(value_col).cast(pl.Float64).alias(value_col),
    )

    years = df.select(pl.col(year_col).unique()).to_series().to_list()
    years = sorted([int(y) for y in years])

    out_parts = []

    CT_SWITCH_YEAR = 2022  # SAHIE uses planning regions starting in 2022

    for y in years:
        df_y = df.filter(pl.col(year_col) == y).select([fips_col, value_col])

        # to pandas for Tobler
        pdf_y = df_y.to_pandas()
        pdf_y = pdf_y.rename(columns={fips_col: "FIPS", value_col: "value"})

        if y < CT_SWITCH_YEAR:
            pdf_fixed = fix_connecticut_rate_only(pdf_y)
        else:
            # For 2022 and later, CT counties are already in new structure; no fix needed
            pdf_fixed = pdf_y

        if round_decimals is not None:
            pdf_fixed["value"] = pdf_fixed["value"].round(round_decimals)

        # back to polars, restore schema
        pl_fixed = pl.from_pandas(pdf_fixed).with_columns(
            pl.lit(y).alias(year_col),
        ).rename({"FIPS": fips_col, "value": value_col})

        out_parts.append(pl_fixed)

        print(f"CT fix done for year {y}" if y < CT_SWITCH_YEAR else f"CT passthrough for year {y}")

    df_out = pl.concat(out_parts, how="vertical").select([fips_col, year_col, value_col])

    # Checks to make sure old counties gone and new counties added.
    # 1. Old CT counties should be gone
    print(df_out.filter(pl.col(fips_col).is_in(CT_OLD_FIPS)).height)

    # 2. CT should now have 9 county-equivalents
    print(
        df_out.filter(pl.col(fips_col).str.starts_with("09"))
        .select(fips_col)
        .unique()
        .sort(fips_col)
    )
    # Want to inspect the values for the new county structures:
    print(
        df_out
        .filter(pl.col(fips_col).str.starts_with("09"))
        .sort(fips_col)
        .head(20)
    )

    # If your original long file had other columns (state, etc.), you can join them back here if needed.
    # For now, we just write the cleaned long series.    
    df_out.write_parquet(parquet_out)
    print(f"Saved: {parquet_out}")



def main():
    # Prescription rates
    CT_cleaning_imputing(
        input_csv=RX_INPUT_CSV,
        output_csv=RX_OUTPUT_CSV,
        value_name=RX_VALUE_NAME,
        fips_col=FIPS_COL,
        round_decimals=1,
        missing_values=None,
    )
    # Unemployment rates
    CT_cleaning_imputing(
        input_csv=UNEMP_INPUT_CSV,
        output_csv=UNEMP_OUTPUT_CSV,
        value_name=UNEMP_VALUE_NAME,
        fips_col=FIPS_COL,
        round_decimals=1,
        missing_values=None,
    )
    # Uninsured rates (long parquet)
    ct_fix_long_parquet(
        parquet_in="data/Processed/Uninsured/SAHIE_Uninsured_rates.parquet",
        parquet_out="data/Processed/Uninsured/Uninsured_rates_CT_fixed.parquet",
        fips_col="FIPS",
        year_col="year",
        value_col="uninsured_rate",
        round_decimals=1,
    )


if __name__ == "__main__":
    main()
