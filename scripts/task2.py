# ============================================================
# FIFA WORLD CUP 2026 DATA ANALYSIS
# TASK 2
# ============================================================

# ============================================================
# 1. Analytic Question Formulation
# ============================================================
# Research Question:
# Do teams that qualify for the Kncokout Stags have a significant difference in 
# the goal-per-shot efficiency between the Group Stages and Knockout stages?
#
# Variables:
# Independent Variable   ->      Match Stage (Group or Knockout)
# Dependent Variable     ->      Goal per shot efficiency (Goals / Shots)
#
# Hypotheses:
# Null Hypotehsis (H0): μ_diff = 0
# Alternate Hypothesis (h1): μ_diff != 0
# where μ_diff = (Knockout - Group) efficiency
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
# Raw dataset contains 32 teams with the following columns:
# TEAMS | STAGE | GOALS | SHOTS
# 
# Only teams that qualify for the Knockout Stages have been considered
# Data has been manually collected and stored in MsExcel.
# ============================================================

#LOAD RAW DATASET
raw_data = pd.read_excel("data/raw/task2_raw.xlsx")

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

# -------------------------------
# BASIC DATA CHECKS
# -------------------------------
print("--- 1. NUMBER OF OBSERVATIONS ---")
print(f"Total rows (should be 64): {len(df)}")

print("\n--- 2. CHECK FOR DUPLICATE TEAM + STAGE ROWS ---")
duplicates = df.duplicated(subset=["TEAM", "STAGE"]).sum()
print(f"Duplicate TEAM + STAGE rows found: {duplicates}")

print("\n--- 3. CHECK FOR MISSING VALUES ---")
print(df.isnull().sum())

# -------------------------------
# STANDARDIZING COLUMN TEXT
# -------------------------------

print("\n--- 4. CLEAN TEXT COLUMNS ---")
df["TEAM"] = df["TEAM"].astype(str).str.strip()
df["STAGE"] = df["STAGE"].astype(str).str.strip().str.upper()

# -------------------------------
# CHECKING AND FIXING DATA COLUMN DATA TYPES
# -------------------------------

print("\n--- 5. CHECK & FIX DATA TYPES ---")
print("Original data types:")
print(df.dtypes)

df["GOALS"] = pd.to_numeric(df["GOALS"], errors="coerce")
df["SHOTS"] = pd.to_numeric(df["SHOTS"], errors="coerce")

print("\nData types after conversion:")
print(df.dtypes)

# -------------------------------
# VALIDATE VALUES FOR THE 'STAGE' COLUMN
# -------------------------------

print("\n--- 6. VALIDATE STAGE VALUES ---")
unique_stages = df["STAGE"].unique()
print(f"Unique values in STAGE column: {unique_stages}")

invalid_stage = df[~df["STAGE"].isin(["GROUP", "KNOCKOUT"])]
print(f"Rows with invalid STAGE values: {len(invalid_stage)}")

# -------------------------------
# VALIDATE COLUMN DETAILS TO ENSURE DATA INTEGRITY AND CONSISTENCY
# -------------------------------

print("\n--- 7. CHECK NUMBER OF UNIQUE TEAMS ---")
unique_teams = df["TEAM"].nunique()
print(f"Unique teams (should be 32): {unique_teams}")

print("\n--- 8. CHECK EACH TEAM APPEARS EXACTLY TWICE ---")
team_counts = df["TEAM"].value_counts()
invalid_team_counts = team_counts[team_counts != 2]
print(f"Teams not appearing exactly twice: {len(invalid_team_counts)}")
if len(invalid_team_counts) > 0:
    print(invalid_team_counts)

print("\n--- 9. CHECK EACH TEAM HAS BOTH GROUP AND KNOCKOUT ---")
stage_check = df.groupby("TEAM")["STAGE"].apply(set)
invalid_stage_pairs = stage_check[
    stage_check.apply(lambda x: x != {"GROUP", "KNOCKOUT"})
]
print(f"Teams missing GROUP or KNOCKOUT row: {len(invalid_stage_pairs)}")
if len(invalid_stage_pairs) > 0:
    print(invalid_stage_pairs)

print("\n--- 10. CHECK FOR NEGATIVE GOALS OR SHOTS ---")
invalid_numbers = df[(df["GOALS"] < 0) | (df["SHOTS"] < 0)]
print(f"Rows with negative GOALS/SHOTS: {len(invalid_numbers)}")

# -------------------------------
# SANITIZE DUPLICATED THROUGH REMOVAL
# -------------------------------

print("\n--- 11. REMOVE EXACT DUPLICATES ---")
df = df.drop_duplicates(subset=["TEAM", "STAGE"], keep="first")

print("\n--- 12. FINAL MISSING VALUE CHECK ---")
print(df.isnull().sum())

# ============================================================
# DERIVED VARIABLE
# ============================================================

df["GOALS_PER_SHOT"] = df["GOALS"] / df["SHOTS"]

print("\n--- Derived Variable Check ---")
print(df[["TEAM", "STAGE", "GOALS_PER_SHOT"]].head())

# -------------------------------
# SAVE CLEANED DATASET
# -------------------------------

print("\n--- 13. SAVE CLEANED DATASET ---")
df.to_csv("data/cleaned/task2_cleaned.csv", index=False)

print("Cleaned dataset saved as: task2_cleaned.csv")
print(f"Final rows: {len(df)}")
print(f"Final unique teams: {df['TEAM'].nunique()}")

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
# Compare goal per shot efficiency for a team in:
#   - Their Group Stage Matches
#   - Their Knockout Stage Matches
# ============================================================

print("\n=== DESCRIPTIVE STATISTICS (BY STAGE) ===")

for stage in ["GROUP", "KNOCKOUT"]:
    stage_df = df[df["STAGE"] == stage]
    data = stage_df["GOALS_PER_SHOT"] #Take out goal per shot (gps) for each stage
    
    n = len(data)
    mean = data.mean()
    std = data.std(ddof=1)
    sem = stats.sem(data)
    median = data.median()
    iqr = data.quantile(0.75) - data.quantile(0.25)
    skew = data.skew()
    
    print(f"\n--- {stage} STAGE ---")
    print(f"Count (n): {n}")
    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Median: {median:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Skewness: {skew:.4f}")

# ============================================================
# 5. INFERENTIAL STATISTICS: CONFIDENCE INTERVAL
# 95% Confidence Interval for mean difference in goal-per-shot efficiency
# Formula for calculating CI:
# mean +- t * (std / sqrt(n))
# ============================================================

print("\n=== CONFIDENCE INTERVAL (PAIRED DIFFERENCE) ===")

# SORT TEAMS AND ALIGN
group_data = df[df["STAGE"]=="GROUP"].sort_values("TEAM").reset_index(drop=True)
ko_data = df[df["STAGE"] == "KNOCKOUT"].sort_values("TEAM").reset_index(drop=True)

# COMPUTE THE DIFFERENCE
diff = ko_data["GOALS_PER_SHOT"] - group_data["GOALS_PER_SHOT"]

n = len(data)
mean_diff = diff.mean()
sem_diff = stats.sem(diff)

t_crit = stats.t.ppf(0.975, df = n - 1)

ci_lower = mean_diff - t_crit * sem_diff
ci_upper = mean_diff + t_crit * sem_diff

print(f"Count (n): {n}")
print(f"Mean Difference: {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ============================================================
# 6. INFERENTIAL STATISTICS: PAIRED T-TEST
# ============================================================
# A paired t-test is used since the same teams are observed
# in both GROUP and KNOCKOUT stages
# ============================================================

print('\n=== PAIRED T-TEST RESULTS ===')

# ALIGN DATA USING PIVOT (Safer than sorting)
pivot = df.pivot(index="TEAM", columns="STAGE", values="GOALS_PER_SHOT")

# DROP ANY INCOMPLETE ROWS (JUST IN CASE)
pivot = pivot.dropna()

# PERFORM THE PAIRED T-TEST
t_stat, p_val = stats.ttest_rel(
    pivot["KNOCKOUT"],
    pivot["GROUP"]
)

print(f"T-Statistic (t*): {t_stat:.4f}")
print(f"P-Value (two-sided): {p_val:.4f}")

# ============================================================
# CONCLUSION
# ============================================================

alpha = 0.05

print("\n<=== CONCLUSION ===>")

if p_val < alpha:
    print("We reject the null hypothesis. (There is a significant difference in the goal-per-shot efficiency of teams between Group and Knockout Stages.)")
else:
    print("We accept the null hpothesis. (There is no significant difference in goal-per-shot efficiency of teams between stages.)")