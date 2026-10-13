import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Style setup for plots
sns.set_theme(style="whitegrid")

# Locate data folder relative to this script
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_dataset():
    file_path = os.path.join(DATA_DIR, "model_2_1.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset at {file_path}. Make sure build_tables.py has been run.")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path} successfully. Total matches: {len(df)}")
    return df


def calculate_vif(X_const):
    # Calculate VIF for each feature, dropping the intercept column
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])
    ]
    # Return without the 'const' row
    return vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)


def plot_diagnostics(model, output_filename="model_2_1_diagnostics.png"):
    fitted = model.fittedvalues
    residuals = model.resid
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]
    n = len(residuals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")

    # 1. Residuals vs Fitted (linearity & homoscedasticity)
    sns.residplot(
        x=fitted, y=residuals, lowess=True, ax=axes[0, 0],
        scatter_kws={"alpha": 0.6}, line_kws={"color": "crimson", "lw": 1.5}
    )
    axes[0, 0].axhline(0, color="gray", linestyle="--")
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Normal Q-Q Plot
    sm.qqplot(residuals, line="45", fit=True, ax=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q")

    # 3. Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color="steelblue", edgecolor="black")
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].set_xlabel("Residuals")

    # 4. Cook's Distance (influential points)
    cooks_threshold = 4 / n
    axes[1, 1].stem(np.arange(n), cooks, markerfmt=".")
    axes[1, 1].axhline(cooks_threshold, color="red", linestyle="--", label=f"Threshold (4/n = {cooks_threshold:.3f})")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].legend()

    # Save to disk
    plt.savefig(output_filename, dpi=300)
    print(f"Saved diagnostic plot to: {output_filename}")
    plt.close()


def main():
    # 1. Load data
    df = load_dataset()

    target = "goal_diff"
    predictors = [
        "fifa_points_diff",
        "avg_caps_diff",
        "avg_age_diff",
        "host_diff",
        "knockout",
        "rest_days_diff",
        "gf_pg_diff",
        "ga_pg_diff"
    ]

    # Clean missing values if any
    clean_df = df.dropna(subset=[target] + predictors)
    X = clean_df[predictors]
    y = clean_df[target]

    # 2. Fit OLS model with constant
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    print("\n" + "=" * 65)
    print("               MODEL 2.1: OLS REGRESSION SUMMARY")
    print("=" * 65)
    print(model.summary())

    # 3. Check multicollinearity via VIF
    print("\n" + "=" * 65)
    print("            VARIANCE INFLATION FACTOR (VIF)")
    print("=" * 65)
    vif_results = calculate_vif(X_const)
    print(vif_results.to_string(index=False))

    # 4. Breusch-Pagan test for heteroscedasticity
    print("\n" + "=" * 65)
    print("                BREUSCH-PAGAN TEST")
    print("=" * 65)
    bp_test = het_breuschpagan(model.resid, X_const)
    print(f"Lagrange Multiplier Stat : {bp_test[0]:.4f}")
    print(f"p-value                  : {bp_test[1]:.4e}")
    if bp_test[1] < 0.05:
        print("Note: Heteroscedasticity detected (p < 0.05). Consider reporting HC3 robust SE.")
    else:
        print("Note: Homoscedasticity assumption holds (p >= 0.05).")

    # 5. Generate and save diagnostic plot
    plot_diagnostics(model, output_filename="model_2_1_diagnostics.png")


if __name__ == "__main__":
    main()