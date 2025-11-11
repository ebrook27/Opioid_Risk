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
    rx_path: Path = Path("data/Processed/Prescriptions/Prescription_dispensing_rates.csv")
    urb_path: Path = Path("data/Processed/Urbanicity/RUCC_urbrur_2013_2023.csv")

    urb_code: str = "RUCC_2023"
    urb_mapping: dict | None = None
    svi_variables: list[str] = field(default_factory=lambda: [
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
        'Single-Parent Household', 'Unemployment'
    ])
    start_year: int = 2010
    end_year: int = 2022

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
        mort_long = (
            pl.read_csv(self.mort_path, schema_overrides={"FIPS": pl.Utf8})
            .with_columns(pl.col("FIPS").str.zfill(5))
            .unpivot(index="FIPS", variable_name="year_str", value_name="mortality_rate")
            .with_columns(
                pl.col("year_str")
                .str.extract(r"(\d{4})")
                .cast(pl.Int64)
                .alias("year")
            )
            .filter(pl.col("year").is_between(self.start_year, self.end_year))
            .select(["FIPS", "year", "mortality_rate"])
        )

        # 3. Prescription
        rx_long = (
            pl.read_csv(self.rx_path, schema_overrides={"FIPS": pl.Utf8})
            .with_columns(pl.col("FIPS").str.zfill(5))
            .unpivot(index="FIPS", variable_name="year_str", value_name="rx_rate")
            .with_columns(
                pl.col("year_str")
                .str.extract(r"(\d{4})")
                .cast(pl.Int64)
                .alias("year")
            )
            .filter(pl.col("year").is_between(self.start_year, self.end_year))
            .select(["FIPS", "year", "rx_rate"])
        )

        # 4. Urbanicity
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
            .join(rx_long, on=["FIPS", "year"], how="full", suffix="_rx")
            .with_columns(
                pl.coalesce(["FIPS", "FIPS_rx"]).alias("FIPS"),
                pl.coalesce(["year", "year_rx"]).alias("year"),
            )
            .drop(["FIPS_rx", "year_rx"])
            .join(urb_df, on="FIPS", how="left")
            .with_columns(pl.col("urbanicity_class").fill_null("Non-CONUS"))
            .drop_nulls()
        )

        return merged

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