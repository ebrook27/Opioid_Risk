### 1/21/26, EB: Realized the population data file had some missing and extra values. Going to start from scratch to re-process from Census files.

import polars as pl

POPULATION_2010S_PATH = 'data/Raw/Population_raw/2010s_population_data.csv'
POPULATION_2020S_PATH = 'data/Raw/Population_raw/2020s_population_data.csv'
OUTPUT_PATH = "data/Processed/Population/population_long_cleaned.csv"


def read_pop_csv(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, encoding='windows-1252')

    if "County" not in df.columns:
        raise ValueError(f"'County' column not found in {path}. Columns: {df.columns}")

    # Clean join key: ".Autauga County, Alabama" -> "Autauga County, Alabama"
    df = df.with_columns(
        pl.col("County")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace(r"^\.", "", literal=False)
        .str.strip_chars()
        .alias("county_key")
    )

    # Optionally: ensure year columns are numeric
    year_cols = [c for c in df.columns if c != "County" and c != "county_key"]
    df = df.with_columns([
        pl.col(c).cast(pl.Int64, strict=False) for c in year_cols
    ])

    # Keep original County for audit, but join on county_key
    return df

def combine_population(df_2010s: pl.DataFrame, df_2020s: pl.DataFrame) -> pl.DataFrame:
    # Outer join keeps mismatched counties (Alaska, Hawaii, and North Dakota differences)
    out = df_2010s.join(df_2020s, on="county_key", how="full", suffix="_2020s")

    # Coalesce County labels into one column (prefer 2010s, else 2020s)
    out = out.with_columns(
        pl.coalesce(
            [pl.col("County"), pl.col("County_2020s")]
        ).alias("County")
    ).drop(["County_2020s"])

    # Optional: reorder columns nicely
    year_cols = sorted([c for c in out.columns if c.isdigit()], key=int)
    out = out.select(["County", "county_key"] + year_cols)

    return out

def mismatch_diagnostics(df_2010s: pl.DataFrame, df_2020s: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    s10 = df_2010s.select("county_key").unique()
    s20 = df_2020s.select("county_key").unique()

    only_2010s = (
        df_2010s.select(["County", "county_key"])
        .unique()
        .join(s20, on="county_key", how="anti")
        .sort("county_key")
    )

    only_2020s = (
        df_2020s.select(["County", "county_key"])
        .unique()
        .join(s10, on="county_key", how="anti")
        .sort("county_key")
    )

    return only_2010s, only_2020s

def read_fips_txt(path: str) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={"FIPS": pl.Utf8},  # preserve leading zeros
        encoding="windows-1252",   # safe on Windows exports
    )

    # Defensive: trim column names
    df = df.rename({c: c.strip() for c in df.columns})

    required = {"State", "FIPS", "County"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FIPS file missing columns {missing}. Columns: {df.columns}")

    df = df.with_columns([
        pl.col("State").cast(pl.Utf8).str.strip_chars(),
        pl.col("County").cast(pl.Utf8).str.strip_chars(),
        pl.col("FIPS").cast(pl.Utf8).str.strip_chars().str.zfill(5),
        (pl.col("County") + ", " + pl.col("State")).alias("county_key"),
    ])

    # Keep one row per county_key (should be unique)
    df = df.unique(subset=["county_key"])

    return df.select(["county_key", "FIPS"])


pop_2010s = read_pop_csv(POPULATION_2010S_PATH)
pop_2020s = read_pop_csv(POPULATION_2020S_PATH)

only_2010s, only_2020s = mismatch_diagnostics(pop_2010s, pop_2020s)
# print(only_2010s.head(10))
# print(only_2020s.head(20))

pop_wide = combine_population(pop_2010s, pop_2020s)

# # We want long format: FIPS, year, population
pop_long = pop_wide.unpivot(
    index=["county_key", "County"],
    on=[c for c in pop_wide.columns if c.isdigit()],
    variable_name="year",
    value_name="population",
).with_columns(
    pl.col("year").cast(pl.Int32),

    # Rebuild county_key deterministically from County
    pl.col("County")
      .cast(pl.Utf8)
      .str.strip_chars()
      .str.replace(r"^\.", "", literal=False)
      .str.strip_chars()
      .alias("county_key"),
)


# keep only actual observations
pop_long = pop_long.filter(pl.col("population").is_not_null())
# print(f"Total population records: {pop_long.height}")
# print(pop_long.head(10))
# print(pop_long.group_by("year").len().sort("year"))

fips_codes = read_fips_txt('data/Raw/Population_raw/county_fips_codes.txt')
# print(fips_codes.head(10))
pop_long_with_fips = pop_long.join(fips_codes, on="county_key", how="left")
print(pop_long_with_fips.head(10))

missing_fips = (
    pop_long_with_fips
    .filter(pl.col("FIPS").is_null())
    .select(["county_key", "County"])
    .unique()
    .sort("county_key")
)

print("Counties missing FIPS:", missing_fips.height)
print(missing_fips.head(50))



# # Write output:
pop_long_with_fips.write_csv(OUTPUT_PATH)

# print(pop_long.head(10))