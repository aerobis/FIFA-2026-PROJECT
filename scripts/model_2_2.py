"""
FIFA 2026 Project - Objective 2: Model 2.2 (Individual Team Goals Regression)
Group Coursework - Charles Darwin University

This script fits an OLS regression model to predict the goals a team scores in
one match (goals_for) using team Elo ratings, qualifying metrics, match status
and rest days. Includes VIF multicollinearity checks,
a check that accounts for two rows per match, and diagnostic plots.

Run from the OBJECTIVE-2 folder:   python scripts/model_2_2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Set clean visual theme for diagnostic charts
sns.set_theme(style="whitegrid", palette="muted")

# Folders relative to this script: OBJECTIVE-2/scripts -> OBJECTIVE-2/data and /results
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_dataset():
    # Load model_2_2.csv from the data folder
    file_path = os.path.join(DATA_DIR, "model_2_2.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset at {file_path}. Make sure build_tables.py has been run.")
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded {file_path} successfully. Total rows: {len(df)}")
    return df


def calculate_vif(X_const):
    # Calculate Variance Inflation Factor (VIF) for predictor variables
    vif_df = pd.DataFrame()
    vif_df["Predictor"] = X_const.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])
    ]
    # Exclude intercept constant for clean presentation
    return vif_df[vif_df["Predictor"] != "const"].reset_index(drop=True)


def plot_diagnostics(model, output_path):
    # Generate and save a 4-panel regression diagnostic plot grid
    residuals = model.resid
    fitted = model.fittedvalues
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]
    n = len(residuals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Model 2.2 Diagnostic Plots", fontsize=14, fontweight="bold")

    # 1. Residuals vs Fitted (Linearity & Homoscedasticity)
    sns.residplot(
        x=fitted, y=residuals, lowess=True, ax=axes[0, 0],
        scatter_kws={'alpha': 0.6, 'color': 'navy'},
        line_kws={'color': 'red', 'lw': 1.5}
    )
    axes[0, 0].axhline(0, color='grey', linestyle='--')
    axes[0, 0].set_title("1. Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Normal Q-Q Plot (Normality of Residuals)
    sm.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
    axes[0, 1].set_title("2. Normal Q-Q Plot")

    # 3. Residual Distribution Histogram
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='steelblue', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_title("3. Residual Distribution")
    axes[1, 0].set_xlabel("Residuals")

    # 4. Cook's Distance (Influential Points)
    threshold = 4 / n
    axes[1, 1].stem(np.arange(n), cooks, markerfmt=".")
    axes[1, 1].axhline(threshold, color='red', linestyle='--', label=f'Threshold (4/n = {threshold:.3f})')
    axes[1, 1].set_title("4. Cook's Distance")
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].legend()

    plt.tight_layout()

    # Make sure the results folder exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Diagnostic plots saved to: '{output_path}'")
    plt.close()


def main():
    print("=" * 60)
    print("      FIFA 2026 Project: Model 2.2 OLS Regression Analysis")
    print("=" * 60)

    # 1. Load data (column names are the ones build_tables.py writes)
    df = load_dataset()

    target = "goals_for"
    predictors = [
        "own_elo", "opp_elo", "own_qual_gf_pg", "opp_qual_ga_pg",
        "is_host", "knockout", "rest_days", "is_uefa_conmebol"
    ]

    missing = [c for c in [target] + predictors if c not in df.columns]
    if missing:
        raise KeyError(f"[ERROR] Columns missing from model_2_2.csv: {missing}. Found: {list(df.columns)}")

    # Drop missing values for clean estimation (there should be none, ideally)
    df_clean = df.dropna(subset=[target] + predictors)
    X = df_clean[predictors]
    y = df_clean[target]

    # 2. Fit OLS model with constant
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    print("\n" + "=" * 60)
    print("            MODEL 2.2: OLS REGRESSION SUMMARY")
    print("=" * 60)
    print(model.summary())

    # 3. Multicollinearity (VIF)
    print("\n" + "=" * 60)
    print("            VARIANCE INFLATION FACTOR (VIF)")
    print("=" * 60)
    print(calculate_vif(X_const).to_string(index=False))

    # 4. Breusch-Pagan test for heteroscedasticity
    print("\n" + "=" * 60)
    print("                BREUSCH-PAGAN TEST")
    print("=" * 60)
    bp_test = het_breuschpagan(model.resid, X_const)
    print(f"Lagrange Multiplier Stat : {bp_test[0]:.4f}")
    print(f"p-value                  : {bp_test[1]:.4e}")
    if bp_test[1] < 0.05:
        print("Note: Heteroscedasticity detected (p < 0.05). Consider reporting robust standard errors.")
    else:
        print("Note: Homoscedasticity assumption holds (p >= 0.05).")

    # 5. Each match appears twice (once per team), so rows are not fully independent
    #    Re-estimate the standard errors grouping the two rows of a match together
    print("\n" + "=" * 60)
    print("   P-VALUES WITH STANDARD ERRORS CLUSTERED BY MATCH")
    print("=" * 60)
    clustered = sm.OLS(y, X_const).fit(
        cov_type="cluster", cov_kwds={"groups": df_clean["match_id"]}
    )
    comparison = pd.DataFrame({
        "coef": model.params,
        "p_value_normal": model.pvalues,
        "p_value_clustered": clustered.pvalues,
    })
    print(comparison.round(4).to_string())

    # 6. Diagnostic plots
    plot_diagnostics(model, os.path.join(RESULTS_DIR, "model_2_2_diagnostics.png"))

# Calling to run the program
if __name__ == "__main__":
    main()
