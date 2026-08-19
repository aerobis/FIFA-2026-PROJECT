# ============================================================
# FIFA WORLD CUP 2026 DATA ANALYSIS
# TASK 1
# ============================================================

# ============================================================
# 1. Analytic Question Formulation
# ============================================================
# Research Question:
# Do teams that qualify for the Knockout Stage score significantly more
# goals on average than the teams that are eliminated in the Group Stage?
#
# Variables:
# Independent Variable   ->      Qualification Status (Yes or No)
# Dependent Variable     ->      Goal on Average (Goal per match) for Group Stage
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
# TEAM | GROUP_STAGE_MATCHES | GROUP_STAGE_GOALS | QUALIFIED
# 
# Data has been manually collected and stored in MsExcel.
# ============================================================

# LOAD RAW DATASET:
raw_data = pd.read_excel("../data/raw/task1_raw.xlsx");

# PREVIEW DATA:
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

# CONVERT NUMERIC COLUMNS
# Convert the columns to numeric, if not possible, return NaN for the column
df["GROUP_STAGE_MATCHES"] = pd.to_numeric(df["GROUP_STAGE_MATCHES"], errors = "coerce")
df["GROUP_STAGE_GOALS"] = pd.to_numeric(df["GROUP_STAGE_GOALS"], errors = "coerce")

# STANDARDIZE "QUALIFIED" COLUMN
# i.e. Remove any leading/trailing whitespaces and capitalize the words
df["QUALIFIED"] = df["QUALIFIED"].str.strip().str.upper()

# ============================================================
# DATA VALIDATION CHECKS
# ============================================================


# Using built-in assert statements instead of print statements
# to verify data conversion logic works as expected

# CHECK TOTAL NUMBER OF TEAMS
assert len(df) == 48, "Error: Dataset doesn't contain the required 48 teams"

# CHECK UNIQUE TEAMS
assert df["TEAMS"].nunique() == 48, "Error: Duplicate teams detected"

# CHECK GROUP MATCHES CONSISTENCY
assert (df["GROUP_STAGE_MATCHES"] == 3).all(), "Error: All teams must have 3 matches played"

# CHECK QUALIFICATION DISTRIBUTION
assert (df["QUALIFIED"] == "YES").sum() == 32, "Error: 32 teams should have qualified"
assert (df["QUALIFIED"] == "NO").sum() == 16,  "Error: 16 teams should have been eliminated"

# Used for further checking given the first test failed
# print(f"Sum of qualified: {(df["QUALIFIED"] == "YES").sum()}")

# CHECK FOR ANY MISSING VALUES
print(f"\n Missing values check: ${df.isnull().sum()}") #Count out how many, if any, data are missing

# ============================================================
# DERIVED VARIABLE
# ============================================================

# GOALS PER MATCH (KEY ANALYSIS VARIABLE)
df["GOALS_PER_MATCH"] = df["GROUP_STAGE_GOALS"] / df["GROUP_STAGE_MATCHES"]

# SAVE THE CLEANED DATASET
df.to_csv("../data/cleaned/task1_cleaned.csv", index = False) # To keep Pandas from writing row numbers into the csv file

print("\n Cleaned Data Preview: ")
print(df.head())

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
# Compare goals per match between:
#   - Qualified Teams
#   - Eliminated Teams
# ============================================================

