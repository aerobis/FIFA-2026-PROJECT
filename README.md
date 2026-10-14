# FIFA World Cup 2026: Group Stage Goals Analysis ###

#
## GROUP MEMBERS:
## SHREYASH UPRETI                                           - S406921
## JUNG-CHUAN CHIANG (Herein called Raymond in this project) - S406089
## SOBIT PAUDEL                                              - S403784
## SUSMITA SUBEDI                                            - S369901
#

# OBJECTIVE 1
This project aims to perform statistical calculations on various analytical questions relating to the FIFA World Cup 2026, with four distinct analytic tasks being derived for the same purpose.

The websites used for the project's data collection (as referred to by our professor) are listed below:
- The Official FIFA Stats Website
- The Stats Don't Lie Website
- FBRef Website

The four dinstinct analytic questions have been mentioned below:

# TASK 1:
Do teams that qualify for the knockout stages score significantly more goals on average than the teams that are eliminated in the Group Stage?
(Adapted from ChatGPT, OpenAI, accessed 13/08/2026)

# TASK 2:
Do teams that qualify for the knockout stage have a significant difference in goal-per-shot efficiency between the group stages and knockout stages?
(Adapted from ChatGPT, OpenAI, accessed 13/08/2026)

# TASK 3:
Do teams that qualify for the knockout stages have a significantly higher ball possession time on average than teams that get knocked out in the group stages?

# TASK 4:
For teams that qualify for the knockout stages, do their midfielders average more assists (per 90 minutes) than their forwards?


#### DETAILS ####

# Role Division:

## 1) Data Collection:
    Collecting the raw data that is relevant to our calculations from the Site
    ### Handled by Susmita
## 2) Data Cleaning:
    Cleaning the raw data after validating it to get it ready for formal statistical calculations:
    ### Handled by Sobit
## 3) Analytics:
    Doing the relevant calculations required for the task
    # Handled by Shreyash and Raymond
## 4) Writing:
    Implementing the final file by joining and arranging code and data alike, documenting the code and the codebase, preparing the base for the Presentation
    ### Handled by Shreyash

# OBJECTIVE 2
Two multiple linear regression models using only information available before kick-off (11 June 2026).
- Model 2.1 predicts a match's goal difference (Team A goals minus Team B goals). 104 rows, one per match.
- Model 2.2 predicts the goals one team scores in one match. 208 rows, one per team per match.

# Folder structure (OBJECTIVE-2)
data/       input files, plus the model tables built from them
scripts/    all code
results/    everything the scripts write (summaries, metrics, coefficients, plots)

# Input data (data/)
- matches.csv        match_id, date, stage, team_a, team_b, goals_a, goals_b
- team_context.csv   team, confederation, is_host, fifa_points
                     FIFA ranking points: last release before 11 June 2026 
- team_squad.csv     team, avg_caps, avg_age
                     Source: Wikipedia, 2026 FIFA World Cup squads page (as of 11 June 2026)
- team_elo_qual.csv  team, elo, qual_gf_pg, qual_ga_pg
                     Source: eloratings.net. Elo is each team's rating after
                     its last match before 11 June 2026. Qualifying goals per game use World Cup
                     qualifiers from 1 Jan 2023 to 31 Dec 2025. The three hosts had no qualifying
                     campaign, so their last eight non-friendly matches before the tournament are used.
Team names follow the FIFA spelling in matches.csv. build_tables.py stops if a name differs in any file.

# Variables
Model 2.1 (Team A minus Team B): fifa_points_diff, avg_caps_diff, avg_age_diff, host_diff,
    knockout, rest_days_diff, gf_pg_diff, ga_pg_diff. Target: goal_diff.
Model 2.2: own_elo, opp_elo, own_qual_gf_pg, opp_qual_ga_pg, is_host, knockout, rest_days,
    is_uefa_conmebol. Target: goals_for.
Rules: goals per game to date exclude the current match, and a team's first match uses its
qualifying averages instead. Rest days are the days since the team's last match: 7 for a first
match, capped at 7. Knockout = 1 for every round after the group stage, including the third-place match.
Qualifying averages are not adjusted for opposition strength.

# How to run
Install: python -m pip install pandas scikit-learn statsmodels matplotlib seaborn
Run from the OBJECTIVE-2 folder, in this order:
1. python scripts/build_tables.py     builds data/model_2_1.csv, model_2_2.csv and merged_long.csv
2. python scripts/model_2_1.py        Model 2.1 on all rows: OLS summary, VIF, Breusch-Pagan, diagnostic plots
3. python scripts/model_2_2.py        Model 2.2 on all rows: same checks, plus p-values clustered by match
4. python scripts/run_models.py       80/20 train/test split (random_state=0), baseline comparison and
                                      5-fold cross-validation for both models
5. python scripts/figures.py          Figure 1 of the report (results/report_figure1.png)
(scripts/build_elo_qual.py, if included, is the downloader that produced team_elo_qual.csv.)

# Report notes
Tables 2 and 3 of the report come from steps 2-4. Model 2.2's test split is by match, so both rows of a match stay in the same set.