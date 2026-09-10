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
# Dependent Variable     ->      Average goals per match during the Group Stages
#
# Hypotheses:
# Null Hypothesis (H0): μ_qualified = μ_eliminated
# Alternate Hypothesis (H1): μ_qualified > μ_eliminated
# ============================================================

# IMPORTING LIBRARIES

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
#   - Official Fifa Site
#
# Data Description:
# Raw dataset contains 48 teams with the following columns:
# TEAMS | GROUP_STAGE_MATCHES | GROUP_STAGE_GOALS | QUALIFIED
# 
# Data has been manually collected and stored in MsExcel.
# ============================================================

# LOAD RAW DATASET:
raw_data = pd.read_excel("data/raw/task1_raw.xlsx");

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
df.to_csv("data/cleaned/task1_cleaned.csv", index = False) # To keep Pandas from writing row numbers into the csv file

print("\n Cleaned Data Preview: ")
print(df.head())

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
# Compare goals per match between:
#   - Overall Dataset
#   - Qualified Teams
#   - Eliminated Teams
# ============================================================

# FOR OVERALL DATASET
gpm = df['GOALS_PER_MATCH']
mean_gpm = gpm.mean()
std_gpm = gpm.std()
n_gpm = gpm.count()

# Verify the data
print(f'--- Overall Descriptive Statistics ---')
print(f'Total data count (n): {n_gpm}')
print(f'Mean Goal Per Match: {mean_gpm:.4f}')
print(f'Standard Deviation: {std_gpm:.4f}')

# GROUPED DESCRIPTIVE COMPARISON:
print('\n === Group Breakdown (On basis of Goals Per Match) === ')
grouped_stats = df.groupby('QUALIFIED')['GOALS_PER_MATCH'].agg(['count', 'mean', 'std'])
print(grouped_stats.round(4))

# VISUALIZATION: GOALS PER MATCH BY QUALIFICATION STATUS
# MAP ELIMINATED TEAM SCATTERPOINTS TO POSITION X = 0, QUALIFIED TO X = 1
qualified_status = df['QUALIFIED'].map({'NO': 0, 'YES': 1})
# INTRODUCE JITTER TO MREDUCE OVERLAPPING DATA POINTS
jitter = np.linspace(-0.1, 0.1, len(df))

plt.scatter(
    qualified_status + jitter,
    df['GOALS_PER_MATCH'],
    c=df['QUALIFIED'].map({'NO': '#D95F02', 'YES': '#2E8B57'}),
    alpha=0.75,
    edgecolors='black',
    linewidths=0.5
)
plt.xticks([0, 1], ['Eliminated', 'Qualified'])
plt.xlabel('Qualification Status')
plt.ylabel('Goals per Match')
plt.title('Goals per Match by Qualification Status')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 5. INFERENTIAL STATISTICS: CONFIDENCE INTERVAL
# 95% Confidence Interval for mean goals per match 
# Formula for calculating CI:
# mean +- t * (std / sqrt(n))
# ============================================================

# 95% CONFIDENCE INTERVAL FOR THE QUALIFYING TEAMS
qualified_gpm = df[df['QUALIFIED'] == 'YES']['GOALS_PER_MATCH']

n_qual = qualified_gpm.count()
mean_qual = qualified_gpm.mean()
std_qual = qualified_gpm.std()
sem_qual = stats.sem(qualified_gpm)
ci_qual = stats.t.interval(0.95, df=n_qual - 1, loc=mean_qual, scale=sem_qual)

# Verify the data
print('\n --- Knockout Qualified Teams (YES) ---')
print('Count (n):', n_qual)
print(f'Mean: {mean_qual:.4f}')
print(f'Standard Deviation: {std_qual:.4f}')
print(f'95% Confidence Interval: [{ci_qual[0]:.4f}, {ci_qual[1]:.4f}]')

# 95% CONFIDENCE INTERVAL FOR THE ELIMINATED TEAMS

non_qualified_gpm = df[df['QUALIFIED'] == 'NO']['GOALS_PER_MATCH']

n_non_qual = non_qualified_gpm.count()
mean_non_qual = non_qualified_gpm.mean()
std_non_qual = non_qualified_gpm.std()
sem_non_qual = stats.sem(non_qualified_gpm)
ci_non_qual = stats.t.interval(0.95, df=n_non_qual - 1, loc=mean_non_qual, scale=sem_non_qual)

# Verify the data
print('\n --- Non-qualifying Teams/Eliminated Teams (NO) ---')
print('Count (n):', n_non_qual)
print(f'Mean: {mean_non_qual:.4f}')
print(f'Standard Deviation: {std_non_qual:.4f}')
print(f'95% Confidence Interval: [{ci_non_qual[0]:.4f}, {ci_non_qual[1]:.4f}]')

# ============================================================
# 6. INFERENTIAL STATISTICS: TWO-SAMPLE T-TEST
# ============================================================
# An Unpooled Welch's T-test is used for this Task
# ============================================================

t_stats, p_val = stats.ttest_ind(
    qualified_gpm,
    non_qualified_gpm,
    equal_var = False, #Expanded below:
    alternative = 'greater' #Convert to a one-sided p-value
)

# Since Welch's T-test assumes inequal variance between the two groups

print("\n === T-Test Results ===")
print(f'\t T-statistic (t*): {t_stats}')

print("\n === P-Value Results ===")
print(f'\t P-value (one-sided): {p_val}')

# ============================================================
# CONCLUSION
# ============================================================

print("\n <===== CONCLUSION =====>")
if p_val < 0.05:
    print("We reject the null hypothesis. There is sufficient evidence to conclude that Qualified Teams score significantly more goals per match on average than teams that are Eliminated.")
else:
    print("We fail to reject the null hypothesis. There is insufficient evidence to suggest that Qualified Teams score significantly more goals per match on average than teams that are Eliminated.")
