"""
FIFA 2026 Project - Objective 2: Model 2.2 (Individual Team Goals Regression)
Group Coursework - Charles Darwin University

This script fits an OLS regression model to predict individual team goals scored 
using team Elo ratings, qualification metrics, match status, and rest days.
Includes VIF multicollinearity checks, Breusch-Pagan test, and diagnostic plots.
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


def load_dataset():
    """Locate and load model_2_2.csv across common project directory paths."""
    candidate_paths = [
        os.path.join('OBJECTIVE-2', 'model_2_2.csv'),
        os.path.join('data', 'model_2_2.csv'),
        'model_2_2.csv',
        os.path.join('..', 'model_2_2.csv')
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"[INFO] Successfully loaded dataset from: '{path}'")
            return pd.read_csv(path)
            
    raise FileNotFoundError("[ERROR] Could not find 'model_2_2.csv'. Please verify file location.")


def calculate_vif(X_const):
    """Calculate Variance Inflation Factor (VIF) for predictor variables."""
    vif_df = pd.DataFrame()
    vif_df["Predictor"] = X_const.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X_const.values, i) 
        for i in range(X_const.shape[1])
    ]
    # Exclude intercept constant for clean presentation
    return vif_df[vif_df["Predictor"] != "const"].reset_index(drop=True)


def plot_diagnostics(model, output_path="OBJECTIVE-2/model_2_2_diagnostics.png"):
    """Generate and save a 4-panel regression diagnostic plot grid."""
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
    
    # Ensure directory exists before saving
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Diagnostic plots saved to: '{output_path}'")
    plt.close()


def main():
    print("=" * 60)
    print("      FIFA 2026 Project: Model 2.2 OLS Regression Analysis")
    print("=" * 60)

    # 1. Load Data
    df = load_dataset()
    print(f"[INFO] Total observations loaded: {len(df)} rows")

    # 2. Standardize Column Names
    rename_dict = {
        'goals': 'goals_scored',
        'team_elo': 'own_elo',
        'opponent_elo': 'opp_elo',
        'qual_gf': 'own_qual_gf_pg',
        'qual_ga': 'opp_qual_ga_pg'
    }
    df.rename(columns=rename_dict, inplace=True)

    # Target and Predictors setup
    target = 'goals_scored'
    all_predictors = [
        'own_elo', 'opp_elo', 'own_qual_gf_pg', 'opp_qual_ga_pg',
        'is_host', 'knockout', 'rest_days', 'is_uefa_conmebol'
    ]

    # Dynamically pick columns present in CSV
    predictors = [col for col in all_predictors if col in df.columns]
    print(f"[INFO] Active predictor variables ({len(predictors)}): {predictors}")

    if target not in df.columns:
        raise KeyError(f"[ERROR] Target column '{target}' missing from dataset. Found: {list(df.columns)}")

    # Drop missing values for clean estimation
    df_clean = df.dropna(subset=[target] + predictors)
    X = df_clean[predictors]
    y = df_clean[target]

    # 3. Fit OLS Model
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # Print Summary Results