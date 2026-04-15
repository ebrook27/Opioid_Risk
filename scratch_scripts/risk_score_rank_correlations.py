"""
Compute pairwise Spearman rank correlations and Jaccard top-k overlap
between model risk scores.

Assumes three CSVs (one per model) with columns:
    FIPS, Year, AbsError_Risk, SqError_Risk, RawError_Risk,
    AbsError_EWMA_Risk, SqError_EWMA_Risk, RawError_EWMA_Risk

Outputs:
    1. Final-year rank correlation table (for main text)
    2. Year-by-year rank correlation trajectory (for figure)
    3. Jaccard top-k overlap table (for main text)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations

### Redefine compute_all_risk_scores here for standalone use
def compute_all_risk_scores(predictions_df, alpha=0.3):
    df = predictions_df.copy()
    if 'SqError' not in df.columns:
        df['SqError'] = (df['True'] - df['Predicted']) ** 2
    if 'RawError' not in df.columns:
        df['RawError'] = df['True'] - df['Predicted']
    df = df.sort_values(['FIPS', 'Year']).copy()
    errors = ['AbsError', 'SqError', 'RawError']
    for error in errors:
        df[f'{error}_Risk'] = (
            df.groupby('FIPS')[error]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
        )
        df[f'{error}_EWMA_Risk'] = (
            df.groupby('FIPS')[error]
              .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
              .reset_index(level=0, drop=True)
        )
    risk_cols = (
        ['FIPS', 'Year'] +
        [f'{err}_Risk' for err in errors] +
        [f'{err}_EWMA_Risk' for err in errors]
    )
    return df[risk_cols]


# ============================================================================
# 1. LOAD AND MERGE
# ============================================================================

# Adjust paths to your actual file locations
# xgb = pd.read_csv("model_outputs/xgbregressor/2026-03-04_15-51-31/risk_scores.csv")
# rf  = pd.read_csv("model_outputs/randomforestregressor/2026-03-05_20-29-09/risk_scores.csv")
# mlp = pd.read_csv("model_outputs/mlpregressor/2026-03-05_14-53-07/risk_scores.csv")

# print("\n=== SAMPLE RISK SCORES ===")
# print(xgb.head())
# print(rf.head())
# print(mlp.head())

### 3/23/26, EB: Rather than re-running all the models, just using the predictions to compute risk scores here with the best alpha from scratch_scripts/ewma_alpha_sweep.py
xgb_preds = pd.read_csv("model_outputs/xgbregressor/2026-03-04_15-51-31/predictions.csv")
rf_preds  = pd.read_csv("model_outputs/randomforestregressor/2026-03-05_20-29-09/predictions.csv")
mlp_preds = pd.read_csv("model_outputs/mlpregressor/2026-03-05_14-53-07/predictions.csv")

xgb = compute_all_risk_scores(xgb_preds, alpha=0.14) # 3/23/26, EB: alpha=0.14 was identified as best in scratch_scripts/ewma_alpha_sweep.py
rf  = compute_all_risk_scores(rf_preds, alpha=0.14)
mlp = compute_all_risk_scores(mlp_preds, alpha=0.14)

# print("\n=== SAMPLE RISK SCORES ===")
# print(xgb.head())
# print(rf.head())
# print(mlp.head())

# Tag each with a model name
xgb["Model"] = "XGB"
rf["Model"]  = "RF"
mlp["Model"] = "MLP"

# The risk score columns we care about
LOSS_COLS = {
    "MAE":    ("AbsError_Risk",      "AbsError_EWMA_Risk"),
    "MSE":    ("SqError_Risk",       "SqError_EWMA_Risk"),
    "Signed": ("RawError_Risk",      "RawError_EWMA_Risk"),
}

# Merge all three on FIPS + Year with suffixes
merged = (
    xgb.merge(rf, on=["FIPS", "Year"], suffixes=("_XGB", "_RF"))
       .merge(mlp.rename(columns=lambda c: c if c in ["FIPS", "Year"] 
              else f"{c}_MLP"), on=["FIPS", "Year"])
)

MODEL_PAIRS = [("XGB", "RF"), ("XGB", "MLP"), ("RF", "MLP")]
FINAL_YEAR = merged["Year"].max()

# print(merged.head())

# ============================================================================
# 2. FINAL-YEAR RANK CORRELATIONS (for the main text table)
# ============================================================================

final = merged[merged["Year"] == FINAL_YEAR].copy()

results_final = []
for loss_name, (eq_col, ewma_col) in LOSS_COLS.items():
    for m1, m2 in MODEL_PAIRS:
        # Equal-weighted
        rho_eq, _ = spearmanr(
            final[f"{eq_col}_{m1}"], 
            final[f"{eq_col}_{m2}"]
        )
        # EWMA
        rho_ewma, _ = spearmanr(
            final[f"{ewma_col}_{m1}"], 
            final[f"{ewma_col}_{m2}"]
        )
        results_final.append({
            "Model Pair": f"{m1} -- {m2}",
            "Loss": loss_name,
            "Spearman (Equal-Weighted)": round(rho_eq, 3), # type: ignore
            "Spearman (EWMA)": round(rho_ewma, 3),         # type: ignore
        })

df_final = pd.DataFrame(results_final)

# Pivot for the table format you want:
# Rows = model pairs, Columns = Loss x Aggregation
print("\n=== FINAL-YEAR RANK CORRELATIONS (Year = {}) ===".format(FINAL_YEAR))
print("\nEqual-Weighted:")
print(df_final.pivot(index="Model Pair", columns="Loss", 
                      values="Spearman (Equal-Weighted)")[["MAE", "MSE", "Signed"]])
print("\nEWMA:")
print(df_final.pivot(index="Model Pair", columns="Loss", 
                      values="Spearman (EWMA)")[["MAE", "MSE", "Signed"]])

# ============================================================================
# 3. YEAR-BY-YEAR RANK CORRELATION TRAJECTORY (for the figure)
# ============================================================================

years = sorted(merged["Year"].unique())
trajectory = []

for year in years:
    yr_data = merged[merged["Year"] == year]
    for loss_name, (eq_col, ewma_col) in LOSS_COLS.items():
        for m1, m2 in MODEL_PAIRS:
            rho_eq, _ = spearmanr(
                yr_data[f"{eq_col}_{m1}"], 
                yr_data[f"{eq_col}_{m2}"]
            )
            rho_ewma, _ = spearmanr(
                yr_data[f"{ewma_col}_{m1}"], 
                yr_data[f"{ewma_col}_{m2}"]
            )
            trajectory.append({
                "Year": year,
                "Model Pair": f"{m1} -- {m2}",
                "Loss": loss_name,
                "Spearman_EW": rho_eq,
                "Spearman_EWMA": rho_ewma,
            })

df_traj = pd.DataFrame(trajectory)

# Save for plotting
df_traj.to_csv("scratch_scripts/rank_correlation_trajectory.csv", index=False)
print("\n=== TRAJECTORY SAVED ===")
print(df_traj.groupby(["Year", "Loss"])["Spearman_EW"].mean())

# ============================================================================
# 4. JACCARD TOP-K OVERLAP (for the main text table)
# ============================================================================

def jaccard(set_a, set_b):
    """Jaccard similarity: |A ∩ B| / |A ∪ B|"""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

N_counties = len(final)
TOP_K_FRACTIONS = {"Top 5%": 0.05, "Top 10%": 0.10}

jaccard_results = []
for loss_name, (eq_col, ewma_col) in LOSS_COLS.items():
    for agg_label, col in [("Equal-Weighted", eq_col), ("EWMA", ewma_col)]:
        for frac_label, frac in TOP_K_FRACTIONS.items():
            k = int(np.ceil(N_counties * frac))
            
            # Get top-k FIPS for each model
            top_k = {}
            for model in ["XGB", "RF", "MLP"]:
                col_name = f"{col}_{model}"
                # For signed error, "high risk" = large positive values
                # For abs/sq error, "high risk" = large values
                top_k[model] = set(
                    final.nlargest(k, col_name)["FIPS"].values
                )
            
            for m1, m2 in MODEL_PAIRS:
                j = jaccard(top_k[m1], top_k[m2])
                jaccard_results.append({
                    "Model Pair": f"{m1} -- {m2}",
                    "Loss": loss_name,
                    "Aggregation": agg_label,
                    "Threshold": frac_label,
                    "Jaccard": round(j, 3),
                })

df_jaccard = pd.DataFrame(jaccard_results)

print("\n=== JACCARD TOP-K OVERLAP (Final Year = {}) ===".format(FINAL_YEAR))
for agg in ["Equal-Weighted", "EWMA"]:
    print(f"\n{agg}:")
    sub = df_jaccard[df_jaccard["Aggregation"] == agg]
    print(sub.pivot_table(
        index="Model Pair", 
        columns=["Threshold", "Loss"], 
        values="Jaccard"
    ))

# ============================================================================
# 5. PLOTTING SUGGESTION (trajectory figure)
# ============================================================================
"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, agg_col, title in zip(
    axes, 
    ["Spearman_EW", "Spearman_EWMA"], 
    ["Equal-Weighted Risk", "EWMA Risk"]
):
    # Average across loss functions for cleaner visualization
    # (or plot one loss function at a time — your choice)
    for pair in MODEL_PAIRS:
        pair_label = f"{pair[0]} -- {pair[1]}"
        sub = df_traj[
            (df_traj["Model Pair"] == pair_label) & 
            (df_traj["Loss"] == "MAE")  # pick one loss, or average
        ]
        ax.plot(sub["Year"], sub[agg_col], marker="o", label=pair_label)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.5, 1.0)  # adjust as needed
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Figures/rank_correlation_trajectory.pdf")
"""