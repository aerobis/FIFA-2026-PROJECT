"""
figures.py - draws Figure 1 of the report

Fits both models on all rows (same variables as model_2_1.py and model_2_2.py)
and plots, for each model, residuals vs fitted values and a histogram of the residuals

Writes  results/report_figure1.png
Run from the OBJECTIVE-2 folder:   python scripts/figures.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")


def fit_all_rows(name):
    # OLS on all rows. Column 0 is match_id, the last column is the target
    df = pd.read_csv(os.path.join(DATA, name + ".csv"))
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return sm.OLS(y, sm.add_constant(x)).fit()


def main():
    models = [("Model 2.1", fit_all_rows("model_2_1")),
              ("Model 2.2", fit_all_rows("model_2_2"))]

    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for row, (title, fit) in enumerate(models):
        ax1, ax2 = axes[row]
        ax1.scatter(fit.fittedvalues, fit.resid, s=8)
        ax1.axhline(0, color="black", linewidth=1)
        ax1.set_title(title + ": residuals vs fitted", fontsize=9)
        ax1.set_xlabel("Fitted value", fontsize=8)
        ax1.set_ylabel("Residual", fontsize=8)
        ax2.hist(fit.resid, bins=12, edgecolor="black")
        ax2.set_title(title + ": distribution of residuals", fontsize=9)
        ax2.set_xlabel("Residual", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
    fig.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "report_figure1.png"), dpi=200)
    print("Saved results/report_figure1.png")


main()