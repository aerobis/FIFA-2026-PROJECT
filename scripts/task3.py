# ============================================================
# FIFA WORLD CUP 2026 DATA ANALYSIS
# TASK 3
# ============================================================

# ============================================================
# 1. Analytic Question Formulation
# ============================================================
# Research Question:
# Do teams that qualify for the Knockout Stages have a significantly higher ball possession 
# time on average than teams that are eliminated in the Group Stage?
# Variables:
# Independent Variable   ->      Qualification Status (YES/NO)
# Dependent Variable     ->      Average Ball Possession (%)
#
# Hypotheses:
# Null Hypotehsis (H0): μ_qualified = μ_eliminated
# Alternate Hypothesis (h1): μ_qualified > μ_eliminated
# ============================================================

# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 2. DATA WRANGLING
# ============================================================
# Data Source:
#   - The Stats Don't Lie
#   - FBref World Cup Stats
#   - Official Fifa Site
#
# Data Description:
# Raw dataset contains 48 teams with the following columns:
# TEAM | QUALIFIED | MATCH_1_POSSESSION (%) | MATCH_2_POSSESSION (%) | MATCH_3_POSSESSION (%)  
# 
# Ball possession rates for each team during each of their group stage matches have been recorded
# Data has been manually collected and stored in MsExcel.
# ============================================================

#LOAD RAW DATASET
raw_data = pd.read_excel("data/raw/task3_raw.xlsx")

print("Raw Data Preview: ")
print(raw_data.head())

# ============================================================
# 3. DATA PREPARATION AND CLEANING
# ============================================================
# Objectives:
#   - Ensure data has correct data types
#   - Remove any inconsistencies
#   - Validate dataset integrity and check for errors/duplicates
#   - Create derived variable
# ============================================================

df = raw_data.copy()

# STANDARDIZE COLUMN NAMES
df.columns = (
    df.columns
    .str.strip() 
    .str.upper()
    # REPLACE WHITESPACES WITH UNDERSCORES
    .str.replace(" ", "_") 
    # REMOVE PARANTHESES AND PERCENTAGE SYMBOL
    .str.replace("(" , "", regex=False)
    .str.replace(")" , "", regex=False)
    .str.replace("%" , "", regex=False)
    # FOR WHEN THERE'S MULTIPLE UNDERSCORES
    .str.replace("_+", "_", regex=False) 
    # REMOVE TRAILING UNDERSCORES
    .str.strip("_") 
)

print("Cleaned Columns: ", df.columns.tolist())

print("\n--- 1. NUMBER OF OBSERVATIONS ---")
print(f"Total rows: {len(df)}")

print("\n--- 2. CHECK FOR DUPLICATES ---")
print(f"Duplicate teams: {df['TEAM'].duplicated().sum()}")

print("\n--- 3. CHECK FOR MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- 4. CHECK DATA TYPES ---")
print(df.dtypes)

# CLEAN POSSESSION COLUMNS (REMOVE % AND CONVERT TO NUMERIC)
possession_cols = [
    'MATCH_1_POSSESSION',
    'MATCH_2_POSSESSION',
    'MATCH_3_POSSESSION'
]


# REMOVE PERCENTAGE SYMBOLS FROM ROWS (IF ANY)
# THEN CONVERT THE DATA TO NUMBERS
for col in possession_cols:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace('%', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# STANDARDIZE QUALIFIED
df["QUALIFIED"] = df["QUALIFIED"].astype(str).str.strip().str.upper()

print("\n--- 5. VALIDATE QUALIFICATION STATUS ---")
print(df["QUALIFIED"].value_counts())

# ============================================================
# DERIVED VARIABLE
# ============================================================

df["AVG_POSSESSION"] = df[possession_cols].mean(axis=1).round(4)

# -------------------------------
# SAVE CLEANED DATASET
# -------------------------------

print("\n--- SAVE CLEANED DATASET --")
df.to_csv("data/cleaned/task3_cleaned.csv", index=False)

print("Cleaned dataset saved as: task3_cleaned.csv")

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
# Compare ball possession percentage between:
#   - Qualified Teams
#   - Eliminated Teams
# ============================================================

qualified = df[df["QUALIFIED"] == "YES"]["AVG_POSSESSION"]
eliminated = df[df["QUALIFIED"] == "NO"]["AVG_POSSESSION"]

print("\n=== DESCRIPTIVE STATISTICS ===")

print("\n--- Qualified ---")
print(f"Mean: {qualified.mean():.2f}")
print(f"Std: {qualified.std():.2f}")

print("\n--- Eliminated ---")
print(f"Mean: {eliminated.mean():.2f}")
print(f"Std: {eliminated.std():.2f}")

# ============================================================
# 5. INFERENTIAL STATISTICS: CONFIDENCE INTERVAL
# 95% Confidence Interval for mean possession for both groups of data
# Formula for calculating CI:
# mean +- t * (std / sqrt(n))
# ============================================================

ci_qual = stats.t.interval(
    0.95,
    df = len(qualified - 1),
    loc = qualified.mean(),
    scale = stats.sem(qualified)
)

ci_elim = stats.t.interval(
    0.95,
    df = len(eliminated - 1),
    loc = eliminated.mean(),
    scale = stats.sem(eliminated)
)

print("\n=== CONFIDENCE INTERVALS ===")
print(f"Qualified CI: [{ci_qual[0]:.2f}, {ci_qual[1]:.2f}]")
print(f"Eliminated CI: [{ci_elim[0]:.2f}, {ci_elim[1]:.2f}]")

# ============================================================
# 6. INFERENTIAL STATISTICS: PAIRED T-TEST
# ============================================================
# Welch's T-Test used due to independent groups being observed
# ============================================================

t_stat, p_val = stats.ttest_ind(
    qualified,
    eliminated,
    equal_var = False,
    alternative = "greater"
)

print("\n=== T-TEST RESULTS ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# ============================================================
# CONCLUSION
# ============================================================

alpha = 0.05

print("\n === CONCLUSION ===")

if p_val < alpha:
    print("We reject the null hypothesis. (Qualified teams have significantly higher ball possession rates than eliminated teams)")
else:
    print("We accept the null hypothesis. (Qualified teams do not have significantly higher ball possession rates than eliminated teams)")