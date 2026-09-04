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


# ============================================================
# DERIVED VARIABLE
# ============================================================



# -------------------------------
# SAVE CLEANED DATASET
# -------------------------------



# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
# Compare ball possession percentage between:
#   - Qualified Teams
#   - Eliminated Teams
# ============================================================



# ============================================================
# 5. INFERENTIAL STATISTICS: CONFIDENCE INTERVAL
# 95% Confidence Interval for mean possession for both groups of data
# Formula for calculating CI:
# mean +- t * (std / sqrt(n))
# ============================================================


# ============================================================
# 6. INFERENTIAL STATISTICS: PAIRED T-TEST
# ============================================================
# Welch's T-Test used due to independent groups being observed
# ============================================================



# ============================================================
# CONCLUSION
# ============================================================

