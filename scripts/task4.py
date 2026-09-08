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

# -------------------------------------
# BASIC DATA VALIDATION
# -------------------------------------
print("\n--- DATASET VALIDATION CHECK ---")
print(f"Total player recors: {len(df)}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# -------------------------------------
# CLEANING TEXT COLUMNS
# -------------------------------------
df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
df["TEAM"] = df["TEAM"].astype(str).str.strip()
df["POSITION"] = df["POSITION"].astype(str).str.strip()

# -------------------------------------
# CLEANING NUMERICAL COLUMNS
# -------------------------------------
df["ASSISTS"] = pd.to_numeric(df["MINUTES"], errors="coerce")
df["MINUTES"] = pd.to_numeric(df["MINUTES"], errors="coerce")

# -------------------------------------
# HANDLE MISSING VALUES
# -------------------------------------
print("\n--- MISSING VALUES (If any) ---")
print(df.isnull().sum())

df.dropna(subset = ["MINUTES", "ASSISTS", "POSITION"])

# =========================
# REMOVE INVALID DATA
# =========================
df = df[df["MINUTES"] > 0]

# PLAYERS WITH LESS THAN 90 MINUTES OF PLAYTIME ARE NOT CONSIDERED
df = df[df["MINUTES"] >= 90]

# ============================================================
# DERIVED VARIABLE
# ============================================================



# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

# Compare ASSISTS_PER_90 between:
#   - MIDFIELDERS
#   - FORWARDS
# ============================================================


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


# ============================================================
# 6. INFERENTIAL STATISTICS: TWO-SAMPLE T-TEST
# ============================================================

# Use Welch’s Two-Sample T-Test (unequal variance is assumed)
#
# Compare:
#   MIDFIELDERS vs FORWARDS (ASSISTS_PER_90)
#
# ============================================================


# ============================================================
# CONCLUSION
# ============================================================

# alpha = 0.05