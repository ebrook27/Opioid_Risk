### 1/26, 26, EB: To round out my counterfactuals, I want to have one more variable that I can intervene on.
### So far I have RX dispensing rates (CDC), Unemployment rates (BLS), and now I want to add uninsured rates (SAHIE).
### This script should do the ETL for the data, extracting the necessary columns and years.
### I've never pulled directly from an API before, so this will initially be a test run.

from __future__ import annotations

import time
import requests
import polars as pl

BASE = "https://api.census.gov/data/timeseries/healthins/sahie"

SLICE = {
    "AGECAT": "0",   # ages 0–64
    "RACECAT": "0",  # all races
    "SEXCAT": "0",   # both sexes
    "IPRCAT": "0",   # all incomes
}

GET_VARS = ["YEAR", "GEOID", "PCTUI_PT", "PCTUI_MOE"]


def _call_sahie(params: dict, retries: int = 4, backoff: float = 1.5) -> list:
    for attempt in range(retries + 1):
        r = requests.get(BASE, params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(backoff ** attempt)
            continue
        raise RuntimeError(
            f"SAHIE API error {r.status_code}: {r.text[:300]}"
        )
    raise RuntimeError("Unreachable")


import polars as pl

def _parse_lt_to_float(col: str) -> pl.Expr:
    """
    Convert values like "<0.1" -> "0.1" before float casting.
    Leaves normal numeric strings unchanged.
    """
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    # If it starts with "<", remove it; otherwise keep as-is
    s2 = pl.when(s.str.starts_with("<")).then(s.str.replace("^<", "")).otherwise(s)
    return s2.cast(pl.Float64)

def _json_to_df(data: list) -> pl.DataFrame:
    header, rows = data[0], data[1:]
    return (
        pl.DataFrame(rows, schema=header, orient="row")
        .with_columns(
            [
                pl.col("YEAR").cast(pl.Int32).alias("year"),
                pl.col("GEOID").cast(pl.Utf8).alias("FIPS"),

                _parse_lt_to_float("PCTUI_PT").alias("uninsured_rate"),
                _parse_lt_to_float("PCTUI_MOE").alias("uninsured_rate_moe90"),
            ]
        )
        .with_columns(
            (pl.col("uninsured_rate_moe90") / 1.645).alias("uninsured_rate_se")
        )
        .select(
            ["FIPS", "year",
             "uninsured_rate", "uninsured_rate_moe90", "uninsured_rate_se"]
        )
        .sort(["FIPS", "year"])
    )



# def _json_to_df(data: list) -> pl.DataFrame:
#     header, rows = data[0], data[1:]
#     return (
#         pl.DataFrame(rows, schema=header, orient="row")
#         .with_columns(
#             [
#                 pl.col("YEAR").cast(pl.Int32).alias("year"),
#                 pl.col("GEOID").cast(pl.Utf8).alias("FIPS"),
#                 pl.col("PCTUI_PT").cast(pl.Float64).alias("uninsured_rate"),
#                 pl.col("PCTUI_MOE").cast(pl.Float64).alias("uninsured_rate_moe90"),
#             ]
#         )
#         .with_columns(
#             (pl.col("uninsured_rate_moe90") / 1.645).alias("uninsured_rate_se")
#         )
#         .select(
#             ["FIPS", "year",
#              "uninsured_rate", "uninsured_rate_moe90", "uninsured_rate_se"]
#         )
#         .sort(["FIPS", "year"])
#     )


def fetch_panel_by_year(
    year_start: int,
    year_end: int,
    census_key: str | None = None,
) -> pl.DataFrame:
    frames = []
    for y in range(year_start, year_end + 1):
        params = {
            "get": ",".join(GET_VARS),
            "for": "county:*",
            "time": str(y),
            **SLICE,
        }
        if census_key:
            params["key"] = census_key

        data = _call_sahie(params)
        frames.append(_json_to_df(data))

    return pl.concat(frames, how="vertical").sort(["FIPS", "year"])


if __name__ == "__main__":
    df = fetch_panel_by_year(2014, 2023)
    df.write_parquet("data/Raw/Uninsured_raw/sahie_uninsured_2014_2023.parquet")
    print(f"Wrote {df.height:,} rows")
    # print(df.head(20))
    # print(df.describe())
