### 1/28/26, EB: Original script to plot the state-level opioid laws along with RX rates was very messy.
### Here is my refactored version. If it works and recreates the original plots, I'll delete the old script.

# from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------------
# 1) Utilities
# -----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


def is_yes(x) -> bool:
    """
    PDAPS-style indicators are typically 0/1, but can appear as '.', '', NaN, or strings.
    Treat only a clean '1' as True.
    """
    if pd.isna(x):
        return False
    if x == 1:
        return True
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return False
        try:
            return float(x) == 1.0
        except ValueError:
            return False
    try:
        return float(x) == 1.0
    except Exception:
        return False


def ensure_two_digit_fips(series: pd.Series) -> pd.Series:
    # Handles ints, floats like 51.0, strings like "51"
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

### A couple utility functions to make the path names:

def slugify_state(name: str) -> str:
    """
    Convert state name to a filesystem-safe slug.
    Examples:
      'NEW YORK' -> 'new_york'
      'DISTRICT OF COLUMBIA' -> 'district_of_columbia'
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)   # non-alnum -> _
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def make_rx_plot_path(
    out_dir: Path,
    state_fips: str,
    state_name_lookup: dict[str, str] | None = None,
    # mean_type: str = "state-mean",
    policy_tag: str = "days+mme",
    year_range: str = "2014-2023",
) -> Path:
    state_fips = str(state_fips).zfill(2)

    if state_name_lookup is not None:
        state_name = state_name_lookup.get(state_fips, f"STATE_{state_fips}")
    else:
        state_name = f"STATE_{state_fips}"

    state_slug = slugify_state(state_name)

    fname = (
        f"rx_trends_{state_fips}{state_slug}_"
        f"_{policy_tag}_{year_range}.png"
    )
    return out_dir / fname


# -----------------------------
# 2) Loading + scope filtering
# -----------------------------

def load_laws_with_fips(
    law_xlsx: Path,
    state_fips_txt: Path,
    *,
    state_col_in_crosswalk: str = "STATE",
    fips_col_in_crosswalk: str = "FIPS",
) -> pd.DataFrame:
    """
    Loads PDAPS law file and merges state FIPS codes.
    Keeps both Jurisdiction (from law file) and STATE (from crosswalk) for plotting labels later.
    """
    laws = pd.read_excel(law_xlsx)
    laws = standardize_columns(laws)

    state_fips = pd.read_csv(state_fips_txt, dtype={fips_col_in_crosswalk: str})
    state_fips[state_col_in_crosswalk] = state_fips[state_col_in_crosswalk].str.strip().str.upper()
    state_fips[fips_col_in_crosswalk] = state_fips[fips_col_in_crosswalk].str.strip().str.zfill(2)

    # normalize Jurisdiction to match crosswalk
    laws["Jurisdiction"] = laws["Jurisdiction"].str.strip().str.upper()

    merged = laws.merge(
        state_fips[[fips_col_in_crosswalk, state_col_in_crosswalk]],
        left_on="Jurisdiction",
        right_on=state_col_in_crosswalk,
        how="left",
        validate="many_to_one",
    )

    # Ensure FIPS is clean 2-digit str
    merged["FIPS"] = ensure_two_digit_fips(merged[fips_col_in_crosswalk])

    # Check missing merges
    if merged["FIPS"].isna().any():
        missing = merged.loc[merged["FIPS"].isna(), "Jurisdiction"].unique()
        raise ValueError(f"Missing FIPS for states: {missing}")

    # Parse dates
    # Parse dates safely
    merged["Effective_Date"] = pd.to_datetime(
        merged["Effective_Date"], errors="coerce"
    )

    if "Valid_Through_Date" in merged.columns:
        merged["Valid_Through_Date"] = pd.to_datetime(
            merged["Valid_Through_Date"], errors="coerce"
        )
    else:
        merged["Valid_Through_Date"] = pd.NaT

    return merged


def filter_laws_scope(laws: pd.DataFrame, include_minors: bool = False) -> pd.DataFrame:
    """
    Reproduces your current scope restriction:
      - must have a restriction at all (opio_ == 1)
      - drop minors-only supply limits
      - keep adult duration restrictions (opio_prescr1 == 1)
    NOTE: You mentioned excluding acute pain, setting restrictions, provider-specific, etc.,
    but your current code does NOT explicitly filter those out (beyond the above).
    This function matches the code you pasted to preserve parity.
    """
    df = laws.copy()

    # must have restriction at all
    df = df[df["opio_"] == 1]

    # drop minors-only supply limits
    # minors_only_col = "opio_maxnumber_Law_only_limits_maximum_supply_for_minors"
    # if minors_only_col in df.columns:
    #     df = df[df[minors_only_col] != 1]

    # Including minors only limits now.
    if not include_minors and "opio_maxnumber_Law_only_limits_maximum_supply_for_minors" in df.columns:
        df = df[df["opio_maxnumber_Law_only_limits_maximum_supply_for_minors"] != 1]

    # keep adult duration restrictions
    # (this is what you have: laws["opio_prescr1"] == 1)
    df = df[df["opio_prescr1"] == 1]

    # Ensure dates exist
    df = df[df["Effective_Date"].notna()].copy()

    return df


# -----------------------------
# 3) Spec-driven policy extraction
# -----------------------------

@dataclass(frozen=True)
class PolicySpec:
    """
    Spec for extracting a single numeric 'cap' (or value) per law row.
    - applies_if: function(row) -> bool, determines if the category applies
    - value_map: indicator-column -> numeric value
    - combine: how to combine multiple active values in a row ("min" or "max")
    - out_col: output column name in laws dataframe
    """
    name: str
    applies_if: Callable[[pd.Series], bool]
    value_map: Mapping[str, float]
    combine: str  # "min" or "max"
    out_col: str


def extract_policy_from_row(row: pd.Series, spec: PolicySpec) -> float:
    if not spec.applies_if(row):
        return np.nan

    active_vals = []
    for col, val in spec.value_map.items():
        if col in row.index and is_yes(row.get(col, 0)):
            active_vals.append(val)

    if not active_vals:
        return np.nan

    if spec.combine == "min":
        return float(min(active_vals))
    if spec.combine == "max":
        return float(max(active_vals))

    raise ValueError(f"Unknown combine='{spec.combine}' for spec '{spec.name}'")


def apply_policy_specs(laws: pd.DataFrame, specs: list[PolicySpec]) -> pd.DataFrame:
    """
    Adds one output column per spec to the laws dataframe.
    Keeps your current semantics but makes policy extraction fully spec-driven.
    """
    df = laws.copy()

    for spec in specs:
        df[spec.out_col] = df.apply(lambda r: extract_policy_from_row(r, spec), axis=1)  # type: ignore[call-overload]
        df[spec.out_col] = df[spec.out_col].astype("Float64")

    return df


# -----------------------------
# 4) Collapse to regime changes (same logic as you had)
# -----------------------------

def collapse_policy_regimes(
    laws: pd.DataFrame,
    policy_cols: list[str],
    group_col: str = "Jurisdiction",
) -> pd.DataFrame:
    """
    Keeps only rows where any binding policy component changes within each state.
    Preserves state name columns for plotting: Jurisdiction, STATE, FIPS, Effective_Date.
    """

    def is_regime_change(curr: pd.Series, prev: pd.Series) -> bool:
        for col in policy_cols:
            a = curr[col]
            b = prev[col]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                return True
            if a != b:
                return True
        return False

    def collapse_state(df_state: pd.DataFrame) -> pd.DataFrame:
        df_state = df_state.sort_values("Effective_Date").reset_index(drop=True)
        keep = [True]
        for i in range(1, len(df_state)):
            keep.append(is_regime_change(df_state.iloc[i], df_state.iloc[i - 1]))
        return df_state.loc[keep]

    out = (
        laws.groupby(group_col, group_keys=False)
        .apply(collapse_state)
        .reset_index(drop=True)
    )

    # convenience year column for plotting
    # out["year"] = out["Effective_Date"].dt.year.astype("Int64")
    # Fixing Pylance issue with dt.year
    out["Effective_Date"] = pd.to_datetime(out["Effective_Date"], errors="coerce")
    out["year"] = pd.DatetimeIndex(out["Effective_Date"]).year.astype("Int64")

    return out


# -----------------------------
# 5) Define the policy specs (PARITY with your current code)
# -----------------------------

DAYS_MAP = {
    "opio_maxnumber_3_days": 3,
    "opio_maxnumber_4_days": 4,
    "opio_maxnumber_5_days": 5,
    "opio_maxnumber_7_days": 7,
    "opio_maxnumber_10_days": 10,
    "opio_maxnumber_14_days": 14,
    "opio_maxnumber_30_days": 30,
    "opio_maxnumber_31_days": 31,
    "opio_maxnumber_90_days": 90,
}

DAILY_MME_MAP = {
    "opio_mme_amount_24_MME": 24,
    "opio_mme_amount_30_MME": 30,
    "opio_mme_amount_50_MME": 50,
    "opio_mme_amount_90_MME": 90,
    "opio_mme_amount_100_MME": 100,
    "opio_mme_amount_120_MME": 120,
}

TOTAL_MME_MAP = {
    "opio_restri1_72_MME": 72,
    "opio_restri1_350_MME": 350,
    "opio_restri1_1200_MME": 1200,
}

LOWEST_DAILY_COL = "opio_mme_amount_Lowest_effective_dosage"

### Extra legal specs, non-numeric:
# --- MINORS restriction columns (examples; customize) ---
MINORS_FLAG_COLS = [
    "opio_restri_All_opioid_prescriptions_for_minors",
    "opio_restri_All_initial_prescriptions_for_minors",
    "opio_restri_Initial_prescriptions_for_acute_pain_for_minors",
    "opio_restri_All_prescriptions_for_acute_pain_for_minors",
    "opio_restri_Prescriptions_for_outpatient_use_for_minors",
    # also keep this if you want:
    "opio_maxnumber_Law_only_limits_maximum_supply_for_minors",
]

# --- Scope/context restriction columns (examples; customize) ---
SCOPE_FLAG_COLS = [
    # setting
    "opio_restri_Prescriptions_for_a_specified_health_care_setting",
    "opio_applys_Emergency_department",
    "opio_applys_Urgent_care_center",
    "opio_applys_Walk-in_clinic",
    # outpatient + schedule restrictions
    "opio_restri_Prescriptions_for_outpatient_use",
    "opio_restri_Prescriptions_for_specified_DEA_Schedule",
    "opio_schedu_Schedule_II",
    "opio_schedu_Schedule_III",
    "opio_schedu_Schedule_IV",
    "opio_schedu_Schedule_V",
    # concurrency with benzos (optional; it *is* a scope condition)
    "opio_restri_Opioids_prescribed_concurrently_with_benzodiazepines",
]

# --- Exemption/exception columns (you likely have many; customize) ---
EXEMPTION_COLS = [
    "opio_exempt_condit_Palliative_care",
    "opio_exempt_condit_Cancer-related_pain",
    "opio_exempt_condit_Substance_use_disorder",
    "opio_exempt_condit_Chronic_pain",
    "opio_exempt_condit_Traumatic_injuries",
    "opio_exempt_condit_Professional_judgment",
    "opio_exempt_condit_Emergency_department_care",
    "opio_exempt_condit_Post-operative_care",
    "opio_exempt_condit_Nursing_facility",
    "opio_exempt_condit_Burns",
    "opio_exempt_condit_Inpatient_care",
    "opio_exempt_condit_Sickle_cell_anemia",
    "opio_exempt_condit_Acute_medical_condition",
]

# Choose a threshold that yields a manageable number of "high exemption" regime lines
HIGH_EXEMPTION_THRESHOLD = 3




def build_policy_specs() -> list[PolicySpec]:
    # NOTE: These gating flags match your current implementation.
    # - Days supply: no explicit applies_if gate (always try to infer; will be NaN if no indicators are 1)
    # - Daily MME: requires opio_doseli == 1
    # - Total MME: requires opio_prescr12345 == 1

    days_spec = PolicySpec(
        name="Days supply cap",
        applies_if=lambda row: True,
        value_map=DAYS_MAP,
        combine="min",  # matches your current "binding constraint"
        out_col="days_supply_cap",
    )

    daily_spec = PolicySpec(
        name="Daily MME cap",
        applies_if=lambda row: is_yes(row.get("opio_doseli", 0)),
        value_map=DAILY_MME_MAP,
        combine="min",
        out_col="daily_mme_cap",
    )

    total_spec = PolicySpec(
        name="Total MME cap",
        applies_if=lambda row: is_yes(row.get("opio_prescr12345", 0)),
        value_map=TOTAL_MME_MAP,
        combine="min",
        out_col="total_mme_cap",
    )

    return [days_spec, daily_spec, total_spec]


### This function adds the extra legal flags to the laws dataframe (minors-only, restricted scope (location and schedule), exemptions)
def add_policy_family_flags(laws: pd.DataFrame) -> pd.DataFrame:
    df = laws.copy()

    def any_yes(cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(False, index=df.index)
        # treat exactly 1 as True; if your data has strings like "1", cast first
        X = df[present].apply(pd.to_numeric, errors="coerce").fillna(0)
        return (X == 1).any(axis=1)

    def count_yes(cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(0, index=df.index)
        X = df[present].apply(pd.to_numeric, errors="coerce").fillna(0)
        return (X == 1).sum(axis=1)

    # Minors restrictions (any of the minors-specific flags)
    df["minors_restriction_flag"] = any_yes(MINORS_FLAG_COLS)

    # Restricted scope/context (any scope-limiting flags)
    df["restricted_scope_flag"] = any_yes(SCOPE_FLAG_COLS)

    # Exemptions intensity (count flags; then threshold)
    df["exemption_count"] = count_yes(EXEMPTION_COLS).astype("Int64")
    df["high_exemption_flag"] = (df["exemption_count"] >= HIGH_EXEMPTION_THRESHOLD)

    return df


def add_lowest_effective_flag(laws: pd.DataFrame) -> pd.DataFrame:
    df = laws.copy()
    if LOWEST_DAILY_COL in df.columns:
        df["daily_mme_lowest_effective_flag"] = df[LOWEST_DAILY_COL].apply(is_yes)
    else:
        df["daily_mme_lowest_effective_flag"] = False
    return df


# -----------------------------
# 6) One top-level function to run the full policy pipeline
# -----------------------------

def build_policy_events(
    laws_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - laws_scoped_with_caps: scoped laws with extracted cap columns
      - policy_events: collapsed state policy regime-change table with 'year'
    """
    laws_scoped = filter_laws_scope(laws_raw, include_minors=True)

    specs = build_policy_specs()
    laws_with_caps = apply_policy_specs(laws_scoped, specs)
    laws_with_caps = add_lowest_effective_flag(laws_with_caps)
    laws_with_caps = add_policy_family_flags(laws_with_caps) ### Adds the new minors-only, scope, exemption flags

    # Numeric policy dimensions
    policy_cols = [s.out_col for s in specs]
    
    # Boolean regime flags
    policy_flag_cols = [
        "minors_restriction_flag",
        "restricted_scope_flag",
        "high_exemption_flag",
    ]
    
    # Final regime defintion
    policy_cols.extend(policy_flag_cols)
    
    policy_events = collapse_policy_regimes(laws_with_caps, policy_cols)

    return laws_with_caps, policy_events


### Now we add in the RX rates file information

def load_rx_long(RX_CSV: Path) -> pd.DataFrame:
    rx = pd.read_csv(RX_CSV, dtype={"FIPS": str})
    rx["FIPS"] = rx["FIPS"].str.zfill(5)

    year_cols = [c for c in rx.columns if c.endswith(" DR")]

    rx[year_cols] = (
        rx[year_cols]
        .replace(-9, 0)
        .replace("", 0)
        .fillna(0)
        .astype(float)
    )

    rx_long = rx.melt(
        id_vars="FIPS",
        value_vars=year_cols,
        var_name="year",
        value_name="rx_rate",
    )

    rx_long["year"] = rx_long["year"].str.replace(" DR", "", regex=False).astype(int)
    rx_long["STATE_FIPS"] = rx_long["FIPS"].str[:2]

    return rx_long

### To get the black state mean RX rate lines, we compute them here:
def compute_state_means(rx_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_mean = (
        rx_long
        .groupby(["STATE_FIPS", "year"], as_index=False)
        .agg(state_rx_mean=("rx_rate", "mean"))
    )

    state_mean_nonzero = (
        rx_long
        .assign(rx_rate=lambda d: d["rx_rate"].replace({0: np.nan}))
        .groupby(["STATE_FIPS", "year"], as_index=False)
        .agg(state_rx_mean=("rx_rate", "mean"))
    )

    return state_mean, state_mean_nonzero

### To get the state name on the plots, we use the state FIPS code to look it up:
def build_state_name_lookup_from_crosswalk(STATE_FIPS_TXT: Path) -> dict[str, str]:
    s = pd.read_csv(STATE_FIPS_TXT, dtype={"FIPS": str})
    s["FIPS"] = s["FIPS"].str.zfill(2)
    s["STATE"] = s["STATE"].str.upper()
    return dict(zip(s["FIPS"], s["STATE"]))


### Finally, the plotting function:

def plot_state_rx_with_policy(
    state_fips: str,
    rx_long: pd.DataFrame,
    state_mean: pd.DataFrame,
    policy_events: pd.DataFrame,
    state_name_lookup: dict[str, str] | None = None,
    y_max: float = 200,
    outpath: Path | None = None,
    show: bool = False,
):
    state_fips = str(state_fips).zfill(2)

    fig, ax = plt.subplots(figsize=(10, 6))

    df_state = rx_long[rx_long["STATE_FIPS"] == state_fips]
    if df_state.empty:
        plt.close(fig)
        return

    # county lines
    for _, g in df_state.groupby("FIPS"):
        ax.plot(g["year"], g["rx_rate"], color="lightgray", linewidth=0.7, alpha=0.6)

    # state mean line
    sm = state_mean[state_mean["STATE_FIPS"] == state_fips]
    ax.plot(sm["year"], sm["state_rx_mean"], color="black", linewidth=2, label="State mean")

    # # policy lines
    # pe = policy_events.copy()
    # pe["FIPS"] = pe["FIPS"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    # if "year" not in pe.columns:
    #     pe["year"] = pd.DatetimeIndex(pd.to_datetime(pe["Effective_Date"], errors="coerce")).year.astype("Int64")

    # pe_state = pe[pe["FIPS"] == state_fips]
    
    # Moved the commented block to the main() function to avoid the redundant processing here.
    pe_state = policy_events[policy_events["FIPS"] == state_fips]

    # Define vertical offsets for each policy type to avoid visual overlap    
    X_OFFSETS = {
        "days_supply_cap":        -0.12,
        "daily_mme_cap":          -0.08,
        "total_mme_cap":          -0.02,
        "minors_restriction_flag": 0.02,
        "restricted_scope_flag":   0.08,
        "high_exemption_flag":     0.12,
    }
     
    for _, row in pe_state.iterrows():
        yr = int(row["year"])
        
        # Numeric policy caps
        if pd.notna(row.get("days_supply_cap")):
            ax.axvline(yr + X_OFFSETS["days_supply_cap"], color="tab:blue", linestyle="--", alpha=0.7)
        if pd.notna(row.get("daily_mme_cap")):
            ax.axvline(yr + X_OFFSETS["daily_mme_cap"], color="tab:red", linestyle="--", alpha=0.7)
        if pd.notna(row.get("total_mme_cap")):
            ax.axvline(yr + X_OFFSETS["total_mme_cap"], color="tab:purple", linestyle="--", alpha=0.7)

        # Boolean policy flags
        if bool(row.get("minors_restriction_flag", False)):
            ax.axvline(yr + X_OFFSETS["minors_restriction_flag"], color="tab:green", linestyle=":", alpha=0.7)
        if bool(row.get("restricted_scope_flag", False)):
            ax.axvline(yr + X_OFFSETS["restricted_scope_flag"], color="tab:cyan", linestyle=":", alpha=0.7)
        if bool(row.get("high_exemption_flag", False)):
            ax.axvline(yr + X_OFFSETS["high_exemption_flag"], color="tab:pink", linestyle=":", alpha=0.7)


    # title
    if state_name_lookup is None:
        title_state = f"State {state_fips}"
    else:
        title_state = state_name_lookup.get(state_fips, f"State {state_fips}")
    ax.set_title(f"Opioid RX Dispensing Rates – {title_state} ({state_fips})")

    ax.set_xlabel("Year")
    ax.set_ylabel("Prescriptions per 100 people")

    xmin = int(rx_long["year"].min())
    xmax = int(rx_long["year"].max())
    
    ### I realized that by jittering the vertical policy lines, I probably was pushing some of them off the left side of the plot, so they weren't visible.
    ### I'm adding just a little bit of padding to the x-axis limits to ensure all lines are visible.
    pad = max(abs(v) for v in X_OFFSETS.values())
    ax.set_xlim(xmin - (pad + 0.02), xmax + (pad + 0.02))
    
    # ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, y_max)
    ax.text(
        0.01, 0.01,
        "Note: policy lines within a year are slightly offset for visibility.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8
    )

    legend_items = [
        Line2D([0], [0], color="black", lw=2, label="State mean"),
        Line2D([0], [0], color="lightgray", lw=1, label="County rates"),
        Line2D([0], [0], color="tab:blue", lw=1.5, ls="--", label="Days supply cap"),
        Line2D([0], [0], color="tab:red", lw=1.5, ls="--", label="Daily MME cap"),
        Line2D([0], [0], color="tab:purple", lw=1.5, ls="--", label="Total MME cap"),
        Line2D([0], [0], color="tab:green", lw=1.5, ls=":", label="Minors restrictions"),
        Line2D([0], [0], color="tab:cyan", lw=1.5, ls=":", label="Restricted scope/context"),
        Line2D([0], [0], color="tab:pink", lw=1.5, ls=":", label="High exemptions"),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)

    plt.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200)
    if show:
        plt.show()

    plt.close(fig)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
PLOTS_DIR = PROJECT_ROOT / "rx_policy_plots"





def main():
    LAW_XLSX = Path('data/Raw/Prescriptions_raw/Opioid_Prescribing_Limit_Laws_for_Acute_Pain_April_2022.xlsx')
    STATE_FIPS_TXT = Path('data/Raw/STATE_FIPS_CODES.txt')
    RX_CSV = Path('data/Processed/Prescriptions/Prescription_dispensing_rates.csv')

        
    laws_raw = load_laws_with_fips(LAW_XLSX, STATE_FIPS_TXT)

    laws_scoped_with_caps, policy_events = build_policy_events(laws_raw)
    
    # Standardize policy_events once
    policy_events["FIPS"] = (
        policy_events["FIPS"].astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    # Should already have year, but just in case:
    if "year" not in policy_events.columns:
        policy_events["Effective_Date"] = pd.to_datetime(policy_events["Effective_Date"], errors="coerce")
        policy_events["year"] = pd.DatetimeIndex(policy_events["Effective_Date"]).year.astype("Int64")

    ## Some prints to troubleshoot
    # print(policy_events.columns)
    # print(policy_events[["STATE", "FIPS", "Effective_Date", "year", "days_supply_cap", "daily_mme_cap", "total_mme_cap"]].head(20))

    # print(
    #     policy_events.loc[
    #         policy_events["FIPS"] == "02",
    #         [
    #             "Effective_Date",
    #             "days_supply_cap",
    #             "daily_mme_cap",
    #             "total_mme_cap",
    #             "minors_restriction_flag",
    #             "restricted_scope_flag",
    #             "high_exemption_flag",
    #         ],
    #     ]
    # )

    rx_long = load_rx_long(RX_CSV)
    state_mean, state_mean_nonzero = compute_state_means(rx_long)

    # Prefer crosswalk for names (covers all states even if no policy events)
    state_name_lookup = build_state_name_lookup_from_crosswalk(STATE_FIPS_TXT)

    states = sorted(rx_long["STATE_FIPS"].dropna().unique())
    # print(f"{len(states)} states found.")

    for st in states:
        outpath = make_rx_plot_path(
            out_dir=PLOTS_DIR,
            state_fips=st,
            state_name_lookup=state_name_lookup,
            # mean_type="state-mean",
            policy_tag="days+mme+minors+scope+exempt",
            year_range="2014-2023",
        )
        plot_state_rx_with_policy(
            state_fips=st,
            rx_long=rx_long,
            state_mean=state_mean,          # or state_mean_nonzero if you want
            policy_events=policy_events,
            state_name_lookup=state_name_lookup,
            y_max=200,
            outpath=outpath,
            show=False,
        )


if __name__ == "__main__":
    main()