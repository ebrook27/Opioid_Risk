from dataclasses import dataclass, field
from pathlib import Path
import polars as pl
import yaml

@dataclass
class CountyDataLoader:
    """
    Dataclass for loading and merging SVI, mortality, prescription,
    and urbanicity datasets into a unified long-format Polars DataFrame.
    """

    # === Configuration parameters ===
    svi_dir: Path = Path("data/Processed/SVI")
    mort_path: Path = Path("data/Processed/Mortality/Mortality_final_rates.csv")
    rx_path: Path = Path("data/Processed/Prescriptions/Prescription_dispensing_rates_CT_fixed.csv")
    urb_path: Path = Path("data/Processed/Urbanicity/RUCC_urbrur_2013_2023.csv")
    unemp_path: Path = Path("data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv")
    uninsured_path: Path = Path("data/Processed/Uninsured/SAHIE_Uninsured_rates_CT_fixed.parquet")

    urb_code: str = "RUCC_2023"
    urb_mapping: dict | None = None
    svi_variables: list[str] = field(default_factory=lambda: [
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household'#, 'Unemployment' #Using new unemployment data separately
    ])
    start_year: int = 2010
    end_year: int = 2022
    rx_start_year: int = 2014  # Prescription data starts in 2014
    uninsured_start_year: int = 2014  # SAHIE uninsured data starts in 2014
    
    # 2/4/26, EB: Added helper function to reduce code duplication (mort_long, rx_long, and unemp_long all similar)
    def _wide_to_long(self, path: Path, value_name: str, cast_float: bool = False) -> pl.DataFrame:
        df = (
            pl.read_csv(path, schema_overrides={"FIPS": pl.Utf8})
            .with_columns(pl.col("FIPS").str.zfill(5))
            .unpivot(index="FIPS", variable_name="year_str", value_name=value_name)
            .with_columns(
                pl.col("year_str").str.extract(r"(\d{4})").cast(pl.Int64).alias("year")
            )
            .filter(pl.col("year").is_between(self.start_year, self.end_year))
            .select(["FIPS", "year", value_name])
        )
        if cast_float:
            df = df.with_columns(pl.col(value_name).cast(pl.Float64))
        return df

    def load(self) -> pl.DataFrame:
        """Load, clean, and merge all datasets into a unified Polars DataFrame."""
        # 1. SVI variables
        svi_long_dfs = []
        for var in self.svi_variables:
            var_path = self.svi_dir / f"{var}_final_rates.csv"
            df = (
                pl.read_csv(var_path, schema_overrides={"FIPS": pl.Utf8})
                .with_columns(pl.col("FIPS").str.zfill(5))
                .unpivot(index="FIPS", variable_name="year_str", value_name=var)
                .with_columns(
                    pl.col("year_str")
                    .str.extract(r"(\d{4})")
                    .cast(pl.Int64)
                    .alias("year")
                )
                .filter(pl.col("year").is_between(self.start_year, self.end_year))
                .select(["FIPS", "year", var])
            )
            svi_long_dfs.append(df)

        svi_merged = svi_long_dfs[0]
        for df in svi_long_dfs[1:]:
            svi_merged = svi_merged.join(df, on=["FIPS", "year"], how="full")
            for col in ["FIPS_right", "year_right"]:
                if col in svi_merged.columns:
                    svi_merged = svi_merged.drop(col)

        # 2. Mortality
        # mort_long = (
        #     pl.read_csv(self.mort_path, schema_overrides={"FIPS": pl.Utf8})
        #     .with_columns(pl.col("FIPS").str.zfill(5))
        #     .unpivot(index="FIPS", variable_name="year_str", value_name="mortality_rate")
        #     .with_columns(
        #         pl.col("year_str")
        #         .str.extract(r"(\d{4})")
        #         .cast(pl.Int64)
        #         .alias("year")
        #     )
        #     .filter(pl.col("year").is_between(self.start_year, self.end_year))
        #     .select(["FIPS", "year", "mortality_rate"])
        # )
        mort_long = self._wide_to_long(
            path=self.mort_path,
            value_name="mortality_rate",
        )

        # 3. Prescription
        # rx_long = (
        #     pl.read_csv(self.rx_path, schema_overrides={"FIPS": pl.Utf8})
        #     .with_columns(pl.col("FIPS").str.zfill(5))
        #     .unpivot(index="FIPS", variable_name="year_str", value_name="rx_rate")
        #     .with_columns(
        #         pl.col("year_str")
        #         .str.extract(r"(\d{4})")
        #         .cast(pl.Int64)
        #         .alias("year")
        #     )
        #     .filter(pl.col("year").is_between(self.start_year, self.end_year))
        #     .select(["FIPS", "year", "rx_rate"])
        # )
        rx_long = self._wide_to_long(
            path=self.rx_path,
            value_name="rx_rate",
            cast_float=True,
        )
        
        # 4. Unemployment, from USDA ERS, not SVI
        # unemp_long = (
        #     pl.read_csv(self.unemp_path, schema_overrides={"FIPS": pl.Utf8})
        #     .with_columns(pl.col("FIPS").str.zfill(5))
        #     .unpivot(
        #         index="FIPS",
        #         variable_name="year_str",
        #         value_name="unemp_rate",
        #     )
        #     .with_columns(
        #         pl.col("year_str")
        #         .str.extract(r"(\d{4})")
        #         .cast(pl.Int64)
        #         .alias("year"),
        #         pl.col("unemp_rate").cast(pl.Float64),
        #     )
        #     .filter(pl.col("year").is_between(self.start_year, self.end_year))
        #     .select(["FIPS", "year", "unemp_rate"])
        # )
        unemp_long = self._wide_to_long(
            path=self.unemp_path,
            value_name="unemp_rate",
            cast_float=True,
        )

        # 5. Uninsured (SAHIE), already long format in parquet
        uninsured_long = (
            pl.read_parquet(self.uninsured_path)
            .with_columns(
                pl.col("FIPS").cast(pl.Utf8).str.zfill(5),
                pl.col("year").cast(pl.Int64),
                pl.col("uninsured_rate").cast(pl.Float64),
                #2/4/26, EB: Only pulling in the uninsured_rate for now, can uncomment below if needed.
                # pl.col("uninsured_rate_moe90").cast(pl.Float64),
                # pl.col("uninsured_rate_se").cast(pl.Float64),
            )
            # Clip to your loader year window (this will keep 2010–2013 rows absent here, as nulls after join)
            .filter(pl.col("year").is_between(self.start_year, self.end_year))
            .select(["FIPS", "year", "uninsured_rate"])#, "uninsured_rate_moe90", "uninsured_rate_se"])
        )

        # 6. Urbanicity
        urb_df = (
            pl.read_csv(self.urb_path, schema_overrides={"FIPS": pl.Utf8})
            .with_columns(
                pl.col("FIPS").str.zfill(5),
                pl.col(self.urb_code).cast(pl.Utf8).alias("urbanicity_class"),
            )
            .select(["FIPS", "urbanicity_class"])
        )

        if self.urb_mapping:
            # Build a small mapping DataFrame and left-join to apply the mapping.
            # Using an explicit join is type-checker friendly and avoids relying
            # on `Expr.map_dict`, which some static analyzers may not recognize.
            mapping_df = pl.DataFrame(
                {
                    "urbanicity_class": list(self.urb_mapping.keys()),
                    "urbanicity_class_mapped": list(self.urb_mapping.values()),
                }
            ).with_columns(pl.col("urbanicity_class").cast(pl.Utf8))

            urb_df = (
                urb_df.join(mapping_df, on="urbanicity_class", how="left")
                .with_columns(
                    pl.coalesce(["urbanicity_class_mapped", "urbanicity_class"]).alias("urbanicity_class")
                )
                .drop("urbanicity_class_mapped")
            )

        # 5. Merge all
        merged = (
            svi_merged.join(mort_long, on=["FIPS", "year"], how="inner")
            .join(rx_long, on=["FIPS", "year"], how="left")
            .join(unemp_long, on=["FIPS", "year"], how="left")
            .join(uninsured_long, on=['FIPS', 'year'], how='left')
            .join(urb_df, on="FIPS", how="left")
            .with_columns(pl.col("urbanicity_class").fill_null("Non-CONUS"))
        )
        
        # ------------------------------------------------------------------
        # Explicit temporal alignment and required-feature enforcement
        # RX and SAHIE are only available from 2014 onward, so we explicitly
        # restrict the modeling window to the common support.
        # ------------------------------------------------------------------

        effective_start_year = max(
            self.start_year,
            self.rx_start_year,  # RX availability
            self.uninsured_start_year,  #SAHIE availability
        )

        merged = merged.filter(
            pl.col("year").is_between(effective_start_year, self.end_year)
        )

        required_cols = (
            ["FIPS", "year", "mortality_rate", "rx_rate", "unemp_rate", "uninsured_rate"]
            + self.svi_variables
        )

        merged = merged.drop_nulls(subset=required_cols)
        
        # Last check to guarantee panel keys are unique
        dup = merged.group_by(["FIPS","year"]).len().filter(pl.col("len") > 1)
        assert dup.height == 0

        return merged

### 1/3026, EB: Pasting new dataloader below, which will include the uninsured rates. Commenting out for now because 
# ### I need to do more testing and visualization with the one above.
# @dataclass
# class CountyDataLoader:
#     """
#     Dataclass for loading and merging SVI, mortality, prescription,
#     and urbanicity datasets into a unified long-format Polars DataFrame.
#     """

#     # === Configuration parameters ===
#     svi_dir: Path = Path("data/Processed/SVI")
#     mort_path: Path = Path("data/Processed/Mortality/Mortality_final_rates.csv")
#     rx_path: Path = Path("data/Processed/Prescriptions/Prescription_dispensing_rates.csv")
#     urb_path: Path = Path("data/Processed/Urbanicity/RUCC_urbrur_2013_2023.csv")
#     unemp_path: Path = Path("data/Processed/Unemployment/USDA_Unemployment_rates.csv")
#     uninsured_path: Path = Path("data/Processed/Uninsured/SAHIE_Uninsured_rates.parquet")

#     urb_code: str = "RUCC_2023"
#     urb_mapping: dict | None = None
#     svi_variables: list[str] = field(default_factory=lambda: [
#         'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
#         'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
#         'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
#         'Single-Parent Household'#, 'Unemployment' #Using new unemployment data separately
#     ])
#     start_year: int = 2010
#     end_year: int = 2022
#     rx_start_year: int = 2014  # Prescription data starts in 2014
#     uninsured_start_year: int = 2014  # SAHIE uninsured data starts in 2014

#     def load(self) -> pl.DataFrame:
#         """Load, clean, and merge all datasets into a unified Polars DataFrame."""
#         # 1. SVI variables
#         svi_long_dfs = []
#         for var in self.svi_variables:
#             var_path = self.svi_dir / f"{var}_final_rates.csv"
#             df = (
#                 pl.read_csv(var_path, schema_overrides={"FIPS": pl.Utf8})
#                 .with_columns(pl.col("FIPS").str.zfill(5))
#                 .unpivot(index="FIPS", variable_name="year_str", value_name=var)
#                 .with_columns(
#                     pl.col("year_str")
#                     .str.extract(r"(\d{4})")
#                     .cast(pl.Int64)
#                     .alias("year")
#                 )
#                 .filter(pl.col("year").is_between(self.start_year, self.end_year))
#                 .select(["FIPS", "year", var])
#             )
#             svi_long_dfs.append(df)

#         svi_merged = svi_long_dfs[0]
#         for df in svi_long_dfs[1:]:
#             svi_merged = svi_merged.join(df, on=["FIPS", "year"], how="full")
#             for col in ["FIPS_right", "year_right"]:
#                 if col in svi_merged.columns:
#                     svi_merged = svi_merged.drop(col)

#         # 2. Mortality
#         mort_long = (
#             pl.read_csv(self.mort_path, schema_overrides={"FIPS": pl.Utf8})
#             .with_columns(pl.col("FIPS").str.zfill(5))
#             .unpivot(index="FIPS", variable_name="year_str", value_name="mortality_rate")
#             .with_columns(
#                 pl.col("year_str")
#                 .str.extract(r"(\d{4})")
#                 .cast(pl.Int64)
#                 .alias("year")
#             )
#             .filter(pl.col("year").is_between(self.start_year, self.end_year))
#             .select(["FIPS", "year", "mortality_rate"])
#         )

#         # 3. Prescription
#         rx_long = (
#             pl.read_csv(self.rx_path, schema_overrides={"FIPS": pl.Utf8})
#             .with_columns(pl.col("FIPS").str.zfill(5))
#             .unpivot(index="FIPS", variable_name="year_str", value_name="rx_rate")
#             .with_columns(
#                 pl.col("year_str")
#                 .str.extract(r"(\d{4})")
#                 .cast(pl.Int64)
#                 .alias("year")
#             )
#             .filter(pl.col("year").is_between(self.start_year, self.end_year))
#             .select(["FIPS", "year", "rx_rate"])
#         )
        
#         # 4. Unemployment, from USDA ERS, not SVI
#         unemp_long = (
#             pl.read_csv(self.unemp_path, schema_overrides={"FIPS": pl.Utf8})
#             .with_columns(pl.col("FIPS").str.zfill(5))
#             .unpivot(
#                 index="FIPS",
#                 variable_name="year_str",
#                 value_name="unemp_rate",
#             )
#             .with_columns(
#                 pl.col("year_str")
#                 .str.extract(r"(\d{4})")
#                 .cast(pl.Int64)
#                 .alias("year"),
#                 pl.col("unemp_rate").cast(pl.Float64),
#             )
#             .filter(pl.col("year").is_between(self.start_year, self.end_year))
#             .select(["FIPS", "year", "unemp_rate"])
#         )

#         # 4b. Uninsured (SAHIE), already long format in parquet
#         uninsured_long = (
#             pl.read_parquet(self.uninsured_path)
#             .with_columns(
#                 pl.col("FIPS").cast(pl.Utf8).str.zfill(5),
#                 pl.col("year").cast(pl.Int64),
#                 pl.col("uninsured_rate").cast(pl.Float64),
#                 # pl.col("uninsured_rate_moe90").cast(pl.Float64),
#                 # pl.col("uninsured_rate_se").cast(pl.Float64),
#             )
#             # Clip to your loader year window (this will keep 2010–2013 rows absent here, as nulls after join)
#             .filter(pl.col("year").is_between(self.start_year, self.end_year))
#             .select(["FIPS", "year", "uninsured_rate"])#, "uninsured_rate_moe90", "uninsured_rate_se"])
#         )

#         # 5. Urbanicity
#         urb_df = (
#             pl.read_csv(self.urb_path, schema_overrides={"FIPS": pl.Utf8})
#             .with_columns(
#                 pl.col("FIPS").str.zfill(5),
#                 pl.col(self.urb_code).cast(pl.Utf8).alias("urbanicity_class"),
#             )
#             .select(["FIPS", "urbanicity_class"])
#         )

#         if self.urb_mapping:
#             # Build a small mapping DataFrame and left-join to apply the mapping.
#             # Using an explicit join is type-checker friendly and avoids relying
#             # on `Expr.map_dict`, which some static analyzers may not recognize.
#             mapping_df = pl.DataFrame(
#                 {
#                     "urbanicity_class": list(self.urb_mapping.keys()),
#                     "urbanicity_class_mapped": list(self.urb_mapping.values()),
#                 }
#             ).with_columns(pl.col("urbanicity_class").cast(pl.Utf8))

#             urb_df = (
#                 urb_df.join(mapping_df, on="urbanicity_class", how="left")
#                 .with_columns(
#                     pl.coalesce(["urbanicity_class_mapped", "urbanicity_class"]).alias("urbanicity_class")
#                 )
#                 .drop("urbanicity_class_mapped")
#             )

#         # 5. Merge all
#         merged = (
#             svi_merged.join(mort_long, on=["FIPS", "year"], how="inner")
#             ### 1/29/26, EB: While adding new unemployment and uninsured data, I was confused on why
#             ### I used pl.coalesce() here. I think we should try a left join instead of full, and inspect the results.
#             .join(rx_long, on=["FIPS", "year"], how="left")
#             # .join(rx_long, on=["FIPS", "year"], how="full", suffix="_rx")
#             # .with_columns(
#             #     pl.coalesce(["FIPS", "FIPS_rx"]).alias("FIPS"),
#             #     pl.coalesce(["year", "year_rx"]).alias("year"),
#             # )
#             # .drop(["FIPS_rx", "year_rx"])
#             .join(unemp_long, on=["FIPS", "year"], how="left")
#             .join(uninsured_long, on=['FIPS', 'year'], how='left')
#             .join(urb_df, on="FIPS", how="left")
#             .with_columns(pl.col("urbanicity_class").fill_null("Non-CONUS"))
#             ### 1/29/26, EB: The following .drop_nulls() was implicitly enforcing the correct time window
#             ### which is pretty bad practice. I'm removing it and adding explicit temporal filtering below.
#             # .drop_nulls()
#         )
#         assert "Unemployment" not in merged.columns          # SVI unemployment gone
#         assert "unemp_rate" in merged.columns                # USDA unemployment present
#         unemp_nulls = merged.select(pl.col('unemp_rate').is_null().sum()).item()
#         assert unemp_nulls == 0
#         # assert merged.select(pl.col("unemp_rate").is_null().sum()).item() < merged.height
        
#         # ------------------------------------------------------------------
#         # Explicit temporal alignment and required-feature enforcement
#         # RX and SAHIE are only available from 2014 onward, so we explicitly
#         # restrict the modeling window to the common support.
#         # ------------------------------------------------------------------

#         effective_start_year = max(
#             self.start_year,
#             self.rx_start_year,  # RX availability
#             self.uninsured_start_year,  # SAHIE availability
#         )

#         merged = merged.filter(
#             pl.col("year").is_between(effective_start_year, self.end_year)
#         )

#         required_cols = (
#             ["FIPS", "year", "mortality_rate", "rx_rate", "unemp_rate", "uninsured_rate"]
#             + self.svi_variables
#         )

#         merged = merged.drop_nulls(subset=required_cols)
        
        
#         return merged



# Load YAML model config file
def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file as a dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("❌ Config file must define a dictionary of parameters.")
    return config

# Convert --model_args into a dict
def parse_model_args(arg_list):
    args_dict = {}
    for arg in arg_list:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        # try to cast to int, float, bool, tuple automatically
        if val.lower() in ["true", "false"]:
            val = val.lower() == "true"
        elif val.startswith("(") and val.endswith(")"):
            try:
                val = tuple(map(int, val.strip("()").split(",")))
            except ValueError:
                val = tuple(map(float, val.strip("()").split(",")))
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        args_dict[key] = val
    return args_dict