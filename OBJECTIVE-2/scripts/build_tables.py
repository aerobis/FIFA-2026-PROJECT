"""
build_tables.py - builds the two model tables for Objective 2

Inputs (CSV files, a folder up the tree (in a folder called 'data')):
    matches.csv         match_id, date, stage, team_a, team_b, goals_a, goals_b   (104 rows)
    team_context.csv    team, confederation, is_host, fifa_points                 (48 rows)
    team_squad.csv      team, avg_caps, avg_age                                   (48 rows)
    team_elo_qual.csv   team, elo, qual_gf_pg, qual_ga_pg                         (48 rows)

Outputs:
    model_2_1.csv   104 rows: match_id, 8 explanatory variables, goal_diff (last column)
    model_2_2.csv   208 rows: match_id, 8 explanatory variables, goals_for (last column)
    merged_long.csv one row per team per match with every intermediate value (for checking)

Run:  
         python scripts/build_tables.py

Team names must be spelled IDENTICALLY in all four files (the matches.csv
spelling is taken as standard). The script stops with a clear message if they are not
"""
import os
import sys

import pandas as pd

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def read_files():
    # Read the four input files
    matches = pd.read_csv(os.path.join(FOLDER, "matches.csv"))
    context = pd.read_csv(os.path.join(FOLDER, "team_context.csv"))
    squad = pd.read_csv(os.path.join(FOLDER, "team_squad.csv"))
    elo = pd.read_csv(os.path.join(FOLDER, "team_elo_qual.csv"))
    print("Rows read:", len(matches), len(context), len(squad), len(elo))
    return matches, context, squad, elo

