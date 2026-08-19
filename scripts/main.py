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

