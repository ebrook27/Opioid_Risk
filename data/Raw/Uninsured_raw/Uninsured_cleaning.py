### 1/26/26, EB: In the file SAHIE_Uninsured_ETL.py, I wrote code to extract uninsured rates from the Census API.
### It looks like it pulled everything we wanted, but there are 10 more rows in FIPS and year than there are in the 
### uninsured_rate column. I suspect missing data for some counties/years.
### To keep things simple, I will inspect and clean the data in this file.

### The only missing county in the full dataset is Kalawao County, HI (FIPS 15005).
### I'll be dropping it from the dataset at this point, but we restrict to CONUS later on anyway, so it won't matter.

import polars as pl

UNINSURED_PARQUET_PATH = "data/Raw/Uninsured_raw/sahie_uninsured_2014_2023.parquet"
df = pl.scan_parquet(UNINSURED_PARQUET_PATH)

missing = (
    df
    .filter(
        pl.col("uninsured_rate").is_null()
        | pl.col("uninsured_rate_moe90").is_null()
    )
    .select(["FIPS", "year", "uninsured_rate", "uninsured_rate_moe90"])
    .sort(["FIPS", "year"])
    .collect()
)

print(missing)
print("Missing rows:", missing.height)
### Only missing county is Kalawao County, HI (FIPS 15005)

### Writing cleaned dataset to new parquet file

df_clean = (
    df
    .filter(pl.col("FIPS") != "15005")  # Kalawao County, HI (structurally suppressed)
    .with_columns(pl.col("year").cast(pl.Int32))
    .collect()
)

print("")
print(df_clean.glimpse())
print("")
print(df_clean.describe())
print("")
print(df_clean.schema)
# df_clean.write_parquet("data/Processed/Uninsured/Uninsured_rates.parquet")
# print("Cleaned uninsured rates data written to data/Processed/Uninsured/Uninsured_rates.parquet")