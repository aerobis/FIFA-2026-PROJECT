"""
Reads  data/model_2_1.csv  (104 rows, goal_diff)
       data/model_2_2.csv  (208 rows, goals_for)
Writes into results/:
    model_X_summary.txt        all numbers: split sizes, metrics, OLS summary, coefficients
    model_X_metrics.csv        test-set metrics for the model and the baseline
    model_X_coefficients.csv   coefficient, p-value for each variable
    model_X_heatmap.png        correlation between the explanatory variables
    model_X_residuals.png      residuals vs predicted values, and a histogram of residuals

Method (same as the unit's multiple_linreg.py):
    80% training / 20% test split with a fixed random_state,
    linear regression fitted on the training set, evaluated on the test set,
    compared against a baseline that always predicts the training mean
    statsmodels OLS gives p-values and adjusted R-squared

Needs:  pip install pandas scikit-learn statsmodels matplotlib
Run from the OBJECTIVE-2 folder:   python scripts/run_models.py
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")

RANDOM_STATE = 0       # fixed so everyone gets the same split
TEST_SIZE = 0.2        # 80% train / 20% test


def split_data(df, name):
    # Return X_train, X_test, y_train, y_test
    # Columns: match_id first (not a variable), target last
    x = df.iloc[:, 1:-1]       # the 8 explanatory variables
    y = df.iloc[:, -1]         # the target

    if name == "model_2_1":
        # One row per match, so an ordinary random split is fine
        return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Model 2.2 has two rows per match (one per team). Split by MATCH,
    # so both rows of a match land in the same set
    match_ids = df["match_id"].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    is_train = df["match_id"].isin(train_ids)
    return x[is_train], x[~is_train], y[is_train], y[~is_train]


def get_metrics(y_true, y_pred, y_all):
    # MAE, MSE, RMSE, normalised RMSE and R-squared 
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    rmse_norm = rmse / (y_all.max() - y_all.min())
    r2 = metrics.r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "RMSE_normalised": rmse_norm, "R2": r2}


def cross_validate(df, name):
    # 5 fold cross-validated R squared: a more stable check than one 80/20 split
    # With only 104 / 208 rows, a single test set (21 / 42 rows) can be lucky or unlucky
    # Model 2.2 keeps both rows of a match in the same fold
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    if name == "model_2_1":
        folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(LinearRegression(), x, y, cv=folds, scoring="r2")
    else:
        scores = cross_val_score(LinearRegression(), x, y, cv=GroupKFold(n_splits=5),
                                 groups=df["match_id"], scoring="r2")
    return scores


def plot_heatmap(x, title, path):
    # Correlation between the explanatory variables (to spot collinearity)
    corr = x.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_pred, residuals, title, path):
    # Residuals vs predicted values, and a histogram of the residuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(y_pred, residuals, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xlabel("Predicted value")
    ax1.set_ylabel("Residual (actual - predicted)")
    ax1.set_title("Residuals vs predicted (test set)")
    ax2.hist(residuals, bins=12, edgecolor="black")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram of residuals (test set)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_model(name, target_name):
    print("=" * 60)
    print(name, "- target:", target_name)

    df = pd.read_csv(os.path.join(DATA, name + ".csv"))
    X_train, X_test, y_train, y_test = split_data(df, name)
    x_all = df.iloc[:, 1:-1]
    y_all = df.iloc[:, -1]
    print("Training rows:", len(X_train), "| Test rows:", len(X_test))

    # ---- linear regression (scikit-learn) ----
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_scores = get_metrics(y_test, y_pred, y_all)

    # ---- baseline: always predict the mean of the training target ----
    baseline_pred = [np.mean(y_train)] * len(y_test)
    baseline_scores = get_metrics(y_test, baseline_pred, y_all)

    scores = pd.DataFrame({"model": model_scores, "baseline": baseline_scores}).T
    print(scores.round(4).to_string())

    # ---- cross-validation (extra check, see cross_validate) ----
    cv_scores = cross_validate(df, name)
    print("5-fold cross-validated R2:", cv_scores.round(3), "| mean:", round(cv_scores.mean(), 3))

    # ---- statsmodels OLS (fitted on the training set): p-values, adjusted R-squared ----
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    print(ols.summary())

    coefficients = pd.DataFrame({
        "coefficient": ols.params,
        "p_value": ols.pvalues,
    })

    # ---- save everything ----
    os.makedirs(RESULTS, exist_ok=True)
    scores.to_csv(os.path.join(RESULTS, name + "_metrics.csv"))
    coefficients.to_csv(os.path.join(RESULTS, name + "_coefficients.csv"))

    with open(os.path.join(RESULTS, name + "_summary.txt"), "w", encoding="utf-8") as f:
        f.write(name + " - target: " + target_name + "\n")
        f.write("Training rows: %d | Test rows: %d | random_state=%d\n\n"
                % (len(X_train), len(X_test), RANDOM_STATE))
        f.write("Test-set performance (model vs mean baseline):\n")
        f.write(scores.round(4).to_string() + "\n\n")
        f.write("5-fold cross-validated R-squared per fold: %s\n" % cv_scores.round(3))
        f.write("5-fold cross-validated R-squared, mean: %.4f\n\n" % cv_scores.mean())
        f.write("R-squared on the training set: %.4f\n" % ols.rsquared)
        f.write("Adjusted R-squared on the training set: %.4f\n\n" % ols.rsquared_adj)
        f.write(str(ols.summary()) + "\n")

    plot_heatmap(x_all, name + ": correlation between explanatory variables",
                 os.path.join(RESULTS, name + "_heatmap.png"))
    residuals = y_test - y_pred
    plot_residuals(y_pred, residuals, name, os.path.join(RESULTS, name + "_residuals.png"))

    print("Saved results for", name, "in the results folder")


def main():
    run_model("model_2_1", "goal_diff")
    run_model("model_2_2", "goals_for")


main()