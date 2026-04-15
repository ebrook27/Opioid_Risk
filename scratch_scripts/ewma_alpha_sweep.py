"""
EWMA Alpha Tuning via Cross-Model Concordance

Selects the alpha that maximizes the average pairwise Spearman rank
correlation across all three model pairs, evaluated at the final study year.

Inputs: predictions.csv for each model (columns: FIPS, Year, True, Predicted, Fold, AbsError)
Output: optimal alpha, full results DataFrame, and a plot of concordance vs alpha.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import matplotlib.pyplot as plt


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
# Load predictions
# ============================================================================

# Adjust paths
xgb_preds = pd.read_csv("model_outputs/xgbregressor/2026-03-04_15-51-31/predictions.csv")
rf_preds  = pd.read_csv("model_outputs/randomforestregressor/2026-03-05_20-29-09/predictions.csv")
mlp_preds = pd.read_csv("model_outputs/mlpregressor/2026-03-05_14-53-07/predictions.csv")

MODEL_DATA = {"XGB": xgb_preds, "RF": rf_preds, "MLP": mlp_preds}
MODEL_PAIRS = [("XGB", "RF"), ("XGB", "MLP"), ("RF", "MLP")]
EWMA_COLS = ["AbsError_EWMA_Risk", "SqError_EWMA_Risk", "RawError_EWMA_Risk"]


# ============================================================================
# Alpha sweep
# ============================================================================

def evaluate_alpha(alpha, model_data, model_pairs, ewma_cols):
    """
    For a given alpha, compute EWMA risk scores for all models,
    then return the average pairwise Spearman correlation at the final year,
    averaged across all EWMA loss columns and all model pairs.
    """
    # Compute risk scores for each model at this alpha
    risk_dfs = {}
    for name, preds in model_data.items():
        risk = compute_all_risk_scores(preds, alpha=alpha)
        risk_dfs[name] = risk

    # Find the final year
    final_year = max(risk_dfs["XGB"]["Year"].max(),
                     risk_dfs["RF"]["Year"].max(),
                     risk_dfs["MLP"]["Year"].max())

    # Merge at final year
    final_dfs = {}
    for name, df in risk_dfs.items():
        final_dfs[name] = df[df["Year"] == final_year][["FIPS"] + ewma_cols].copy()
        final_dfs[name].columns = ["FIPS"] + [f"{c}_{name}" for c in ewma_cols]

    merged = final_dfs["XGB"]
    for name in ["RF", "MLP"]:
        merged = merged.merge(final_dfs[name], on="FIPS")

    # Compute pairwise Spearman for each EWMA column and model pair
    correlations = []
    for col in ewma_cols:
        for m1, m2 in model_pairs:
            rho, _ = spearmanr(merged[f"{col}_{m1}"], merged[f"{col}_{m2}"])
            correlations.append(rho)

    return np.mean(correlations)


def tune_alpha(model_data, model_pairs, ewma_cols,
               alpha_min=0.05, alpha_max=0.95, step=0.05):
    """
    Sweep over candidate alpha values and return results.
    """
    alphas = np.arange(alpha_min, alpha_max + step/2, step)
    results = []

    for alpha in alphas:
        avg_rho = evaluate_alpha(alpha, model_data, model_pairs, ewma_cols)
        results.append({"alpha": round(alpha, 3), "avg_spearman": round(avg_rho, 6)})
        print(f"  alpha = {alpha:.3f}  |  avg Spearman = {avg_rho:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# Run the sweep
# ============================================================================

print("=== EWMA ALPHA TUNING ===\n")
print("Coarse sweep (step=0.05):")
coarse = tune_alpha(MODEL_DATA, MODEL_PAIRS, EWMA_COLS,
                    alpha_min=0.05, alpha_max=0.95, step=0.05)

# Find the best region from coarse sweep
best_coarse = coarse.loc[coarse["avg_spearman"].idxmax()]
print(f"\nBest coarse alpha: {best_coarse['alpha']:.3f} "
      f"(avg Spearman = {best_coarse['avg_spearman']:.4f})")

# Fine sweep around the best coarse value
fine_min = max(0.01, best_coarse["alpha"] - 0.10)
fine_max = min(0.99, best_coarse["alpha"] + 0.10)
print(f"\nFine sweep (step=0.01) in [{fine_min:.2f}, {fine_max:.2f}]:")
fine = tune_alpha(MODEL_DATA, MODEL_PAIRS, EWMA_COLS,
                  alpha_min=fine_min, alpha_max=fine_max, step=0.01) # type: ignore

best_fine = fine.loc[fine["avg_spearman"].idxmax()]
print(f"\nOptimal alpha: {best_fine['alpha']:.3f} "
      f"(avg Spearman = {best_fine['avg_spearman']:.4f})")


# ============================================================================
# Plot: concordance vs alpha
# ============================================================================

# Combine coarse and fine for a full picture
all_results = pd.concat([coarse, fine]).drop_duplicates(subset="alpha").sort_values("alpha")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(all_results["alpha"], all_results["avg_spearman"],
        marker="o", markersize=4, linewidth=1.5)
ax.axvline(best_fine["alpha"], color="red", linestyle="--", alpha=0.7, # type: ignore
           label=f"Optimal α = {best_fine['alpha']:.2f}")
ax.set_xlabel("EWMA Smoothing Parameter (α)")
ax.set_ylabel("Average Pairwise Spearman ρ")
ax.set_title("Cross-Model Concordance as a Function of α")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scratch_scripts/alpha_sensitivity.png")
plt.show()

print("\nPlot saved to scratch_scripts/alpha_sensitivity.png")


# ============================================================================
# Bonus: breakdown by loss function at optimal alpha
# ============================================================================

print(f"\n=== BREAKDOWN AT OPTIMAL ALPHA = {best_fine['alpha']:.3f} ===\n")

risk_dfs_opt = {}
for name, preds in MODEL_DATA.items():
    risk_dfs_opt[name] = compute_all_risk_scores(preds, alpha=best_fine["alpha"]) # type: ignore

final_year = max(df["Year"].max() for df in risk_dfs_opt.values())
final_opt = {}
for name, df in risk_dfs_opt.items():
    f = df[df["Year"] == final_year][["FIPS"] + EWMA_COLS].copy()
    f.columns = ["FIPS"] + [f"{c}_{name}" for c in EWMA_COLS]
    final_opt[name] = f

merged_opt = final_opt["XGB"]
for name in ["RF", "MLP"]:
    merged_opt = merged_opt.merge(final_opt[name], on="FIPS")

for col in EWMA_COLS:
    loss_name = col.replace("_EWMA_Risk", "")
    print(f"{loss_name}:")
    for m1, m2 in MODEL_PAIRS:
        rho, _ = spearmanr(merged_opt[f"{col}_{m1}"], merged_opt[f"{col}_{m2}"])
        print(f"  {m1} -- {m2}: {rho:.4f}")
    print()