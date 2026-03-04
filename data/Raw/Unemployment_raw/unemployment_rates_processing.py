### 1/21/26, EB: I want to do interventions on the unemployment rates, but I can't use the SVI data to do this.
### Those are percentile rank-ordered rates, not actual unemployment rates.
### Here I'm extracting the actual unemployment rates from the raw data file, sourced from the USDA ERS.

import polars as pl
from pathlib import Path

UNEMPLOYMENT_RATES_RAW_PATH = Path("data/Raw/Unemployment_raw/Unemployment2023.xlsx")
UNEMPLOYMENT_RATES_OUT_PATH = Path("data/Processed/Unemployment/Unemployment_rates.csv")

unemp_rates = pl.read_excel(UNEMPLOYMENT_RATES_RAW_PATH,
                sheet_id=1,        # first sheet only
                infer_schema_length=1000
            )

# ---- build the list of year columns we want (2010–2022) ----
years = list(range(2010, 2023))
year_cols = [f"Unemployment_rate_{y}" for y in years]

# keep only those that actually exist (guards against naming mismatches)
existing_year_cols = [c for c in year_cols if c in unemp_rates.columns]
missing_year_cols = sorted(set(year_cols) - set(existing_year_cols))
if missing_year_cols:
    print("Warning: missing expected columns:", missing_year_cols)

df_out = (
    unemp_rates
    # ---- drop non-county rows ----
    # County FIPS are 5-digit; USDA often includes: 00000 (US), 01000 (state), etc.
    .filter(
        (pl.col("FIPS_Code").str.len_chars() == 5) &
        (pl.col("FIPS_Code") != "00000") &
        (~pl.col("FIPS_Code").str.ends_with("000")) & #Drop state-level averages
        (~pl.col("FIPS_Code").str.starts_with("72"))   # drop Puerto Rico
    )
    # ---- select and rename ----
    .select(
        pl.col("FIPS_Code").alias("FIPS"),
        *[pl.col(c)
          .cast(pl.Float64)
          .round(2)
          .alias(f"{c.split('_')[-1]} Unemployment")
          for c in existing_year_cols
          ],
    )
    .sort("FIPS")
)

# optional: enforce uniqueness (will throw if duplicates exist)
df_out = df_out.unique(subset=["FIPS"], keep="first")
print(df_out.head(10))
print(f"Rows: {df_out.height:,}  Cols: {df_out.width}")

df_out.write_csv(UNEMPLOYMENT_RATES_OUT_PATH)
print(f"Wrote: {UNEMPLOYMENT_RATES_OUT_PATH}  | rows={df_out.height:,} cols={df_out.width}")