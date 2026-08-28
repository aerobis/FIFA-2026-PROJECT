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
# Alternate Hypothesis (h1): μ_diff > != 0
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
# Raw dataset contains 48 teams with the following columns:
# TEAMS | STAGE | GOALS | SHOTS
# 
# Only teams that qualify for the Knockout Stages have been considered
# Data has been manually collected and stored in MsExcel.
# ============================================================




# ============================================================
# 3. DATA PREPARATION AND CLEANING
# ============================================================
# Objectives:
#   - Ensure data has correct data types
#   - Remove any inconsistencies
#   - Validate dataset integrity and check for errors/duplicates
#   - Create derived variable
# ============================================================


# ============================================================
# DERIVED VARIABLE
# ============================================================



# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
# Compare goal per shot efficiency for a team in:
#   - Their Group Stage Matches
#   - Their Knockout Stage Matches
# ============================================================



# ============================================================
# 5. INFERENTIAL STATISTICS: CONFIDENCE INTERVAL
# 95% Confidence Interval for mean goals per match 
# Formula for calculating CI:
# mean +- t * (std / sqrt(n))
# ============================================================


# ============================================================
# 6. INFERENTIAL STATISTICS: TWO-SAMPLE T-TEST
# ============================================================
# An Unpooled Welch's T-test is used for this Task
# ============================================================
