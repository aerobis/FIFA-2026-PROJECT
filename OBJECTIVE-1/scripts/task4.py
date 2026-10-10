# ============================================================
# FIFA WORLD CUP 2026 DATA ANALYSIS
# TASK 4
# ============================================================

# ============================================================
# 1. Analytic Question Formulation
# ============================================================

# Research Question:
# For teams that qualify for the Knockout Stages, do their midfielders
# average more assists (per 90 minutes) than their forwards?

#
# Variables:
# Independent Variable   ->   Player Position (MIDFIELDER or FORWARD)
# Dependent Variable     ->   Assists (per 90 minutes)
#
# Hypotheses:
# Null Hypothesis (H0): μ_midfielders = μ_forwards
# Alternate Hypothesis (H1): μ_midfielders > μ_forwards
#
# ============================================================


# ============================================================
# IMPORTING LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 2. DATA WRANGLING
# ============================================================

# Data Source:
#   - The Stats Don't Lie
#   - FBref World Cup Stats
#   - Official FIFA Website
#
# Data Description:
# Raw dataset contains players from knockout-qualified teams with:
# PLAYER | TEAM | POSITION | ASSISTS | MINUTES
#
# Data has been manually collected and stored in MsExcel.
#
# ============================================================

# LOAD RAW DATASET:
raw_data = pd.read_excel("data/raw/task4_raw.xlsx")

# PREVIEW DATA:
print("Raw Data Preview:")
print(raw_data.head())


# ============================================================
# 3. DATA PREPARATION AND CLEANING
# ============================================================

# Objectives:
#   - Ensure correct data types
#   - Standardize text fields (PLAYER, TEAM, POSITION)
#   - Validate POSITION values (MIDFIELDER / FORWARD only)
#   - Remove duplicates and invalid rows
#   - Handle missing values
#   - Apply minimum minutes threshold (e.g., >= 90)
#   - Create derived variable (ASSISTS_PER_90)
#
# ============================================================

df = raw_data.copy()

# -------------------------------------
# STANDARDIZE DATASET
# -------------------------------------
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

print("\n--- CLEANED COLUMN NAMES ---")
print(df.columns)

# CHECK REQUIRED COLUMNS
required_columns = ["PLAYER", "TEAM", "POSITION", "ASSISTS", "MINUTES"]
assert set(required_columns).issubset(df.columns), \
    "Error: Required columns are missing"

# -------------------------------------
# BASIC DATA VALIDATION
# -------------------------------------
print("\n--- DATASET VALIDATION CHECK ---")
print(f"Total player records: {len(df)}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# -------------------------------------
# CLEANING TEXT COLUMNS
# -------------------------------------
df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
df["TEAM"] = df["TEAM"].astype(str).str.strip()
df["POSITION"] = df["POSITION"].astype(str).str.strip().str.upper()

# -------------------------------------
# CLEANING NUMERICAL COLUMNS
# -------------------------------------
df["ASSISTS"] = pd.to_numeric(df["ASSISTS"], errors="coerce")
df["MINUTES"] = pd.to_numeric(df["MINUTES"], errors="coerce")

# ============================================================
# DATA VALIDATION CHECKS
# ============================================================

# CHECK VALID POSITION VALUES
assert set(df["POSITION"].unique()).issubset({"MIDFIELDER", "FORWARD"}), \
    "Error: POSITION contains invalid values"

# CHECK FOR MISSING VALUES AFTER CONVERSION
assert not df[["PLAYER", "TEAM", "POSITION", "ASSISTS", "MINUTES"]].isnull().any().any(), \
    "Error: Missing or invalid values detected"

# CHECK FOR NEGATIVE VALUES
assert (df["ASSISTS"] >= 0).all(), \
    "Error: ASSISTS cannot be negative"
assert (df["MINUTES"] > 0).all(), \
    "Error: MINUTES must be greater than zero"

# CHECK FOR DUPLICATE PLAYER RECORDS
assert df.duplicated(subset=["PLAYER", "TEAM"]).sum() == 0, \
    "Error: Duplicate player records detected"

print("\n Data validation checks passed.")

# -------------------------------------
# HANDLE MISSING VALUES
# -------------------------------------
print("\n--- MISSING VALUES (If any) ---")
print(df.isnull().sum())

# PLAYERS WITH LESS THAN 90 MINUTES OF PLAYTIME ARE NOT CONSIDERED
df = df[df["MINUTES"] >= 90]

# ============================================================
# DERIVED VARIABLE
# ============================================================
df["ASSISTS_PER_90"] = (df["ASSISTS"] / df["MINUTES"]) * 90
df["ASSISTS_PER_90"] = df["ASSISTS_PER_90"].round(4)

# CHECK DERIVED VARIABLE
assert df["ASSISTS_PER_90"].notnull().all(), \
    "Error: ASSISTS_PER_90 contains missing values"
assert (df["ASSISTS_PER_90"] >= 0).all(), \
    "Error: ASSISTS_PER_90 cannot be negative"

print("Derived variable validation passed.")

# -------------------------------
# SAVE CLEANED DATASET
# -------------------------------

print("\n--- SAVE CLEANED DATASET --")
df.to_csv("data/cleaned/task4_cleaned.csv", index=False)

print("Cleaned dataset saved as: task4_cleaned.csv")

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

# Compare ASSISTS_PER_90 between:
#   - MIDFIELDERS
#   - FORWARDS
# ============================================================

#SPLIT GROUPS
mf = df[df["POSITION"] == "MIDFIELDER"]["ASSISTS_PER_90"]
fw = df[df["POSITION"] == "FORWARD"]["ASSISTS_PER_90"]

assert len(mf) > 0, "Error: No midfielder observations available"
assert len(fw) > 0, "Error: No forward observations available"

print('\n=== DESCRIPTIVE STATISTICS ===')

print('\n--- MIDFIELDERS ---')
print(f"Mean: {mf.mean():.4f}")
print(f"Standard Deviation: {mf.std():.4f}")

print('\n--- FORWARDS ---')
print(f"Mean: {fw.mean():.4f}")
print(f"Standard Deviation: {fw.std():.4f}")

# VISUALIZATION: PLAYER MEANS AND 95% CONFIDENCE INTERVALS
mf_ci = stats.sem(mf) * stats.t.ppf(0.975, len(mf) - 1)
fw_ci = stats.sem(fw) * stats.t.ppf(0.975, len(fw) - 1)

plt.bar(
    ['MIDFIELDER', 'FORWARD'],
    [mf.mean(), fw.mean()],
    yerr=[mf_ci, fw_ci],
    capsize=5,
    color=['#2B5C8F', '#D9534F'],
    alpha=0.8
)

np.random.seed(42)
plt.scatter(np.random.normal(0, 0.05, len(mf)), mf, color='black', alpha=0.2, s=15)
plt.scatter(np.random.normal(1, 0.05, len(fw)), fw, color='black', alpha=0.2, s=15)
plt.title('Player-Level Assists per 90 (Mean with 95% CI)')
plt.xlabel('Player Position')
plt.ylabel('Assists per 90 minutes')
plt.tight_layout()
plt.show()

# ============================================================
# 5. INFERENTIAL STATISTICS: CONFIDENCE INTERVAL
# ============================================================

# 95% Confidence Interval for ASSISTS_PER_90 for:
#   - MIDFIELDERS
#   - FORWARDS
#
# Formula:
# mean ± t * (std / sqrt(n))
#
# ============================================================

print('\n === CONFIDENCE INTERVAL (95%) === ')

# MIDFIELDERS CI
n_mf = mf.count()
mean_mf = mf.mean()
sem_mf = stats.sem(mf)

ci_mf = stats.t.interval(
    0.95,
    df = n_mf - 1, 
    loc = mean_mf,
    scale = sem_mf 
)

print("\n --- MIDFIELDERS ---")
print(f"95% CI: [{ci_mf[0]:.4f}, {ci_mf[1]:.4f}]")

# FORWARDS CI
n_fw = fw.count()
mean_fw = fw.mean()
sem_fw = stats.sem(fw)

ci_fw = stats.t.interval(
    0.95, 
    df = n_fw - 1,
    loc = mean_fw,
    scale = sem_fw
)

print("\n --- FORWARDS ---")
print(f"95% CI: [{ci_fw[0]:.4f}, {ci_fw[1]:.4f}]")

# ============================================================
# 6. INFERENTIAL STATISTICS: TWO-SAMPLE T-TEST
# ============================================================

# Use Welch’s Independent Two-Sample T-Test (unequal variance is assumed)
#
# Compare:
#   MIDFIELDERS vs FORWARDS (ASSISTS_PER_90)
#
# ============================================================

t_stat, p_val = stats.ttest_ind(
    mf,
    fw,
    equal_var = False,
    alternative = "greater"
)

print("\n=== T-TEST RESULTS ===")
print(f"T-statistic (T*): {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")


# ============================================================
# CONCLUSION
# ============================================================

alpha = 0.05

print("\n === CONCLUSION ===")

if p_val < alpha:
    print("We reject the null hypothesis. There is sufficient evidence to conclude that Midfielders have a higher average assists per 90 rate than Forwards.")
else:
    print("We fail to reject the null hypothesis. There is insufficient evidence to suggest Midfielders have higher assists per 90 than Forwards.")


