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

def check_team_names(matches, context, squad, elo):
    # Stop if a team is spelled differently in any file
    # A merge on team name silently loses teams if the spelling differs
    match_teams = set(matches["team_a"]) | set(matches["team_b"])
 
    for name, table in [("team_context", context), ("team_squad", squad), ("team_elo_qual", elo)]:
        table_teams = set(table["team"])
        missing = match_teams - table_teams
        extra = table_teams - match_teams
        if len(missing) > 0 or len(extra) > 0:
            print("NAME PROBLEM in", name)
            print("  in matches.csv but not in", name + ":", sorted(missing))
            print("  in", name, "but not in matches.csv:", sorted(extra))
            raise SystemExit("Fix the team names above and run again.")
 
    print("Team names match in all files.")

def make_team_info(context, squad, elo):
    # Put everything known about each team in one dictionary
    # Example: info["Spain"]["elo"] gives Spain's Elo rating
    teams = context.merge(squad, on="team")
    teams = teams.merge(elo, on="team")
 
    if len(teams) != 48 or teams.isna().sum().sum() > 0:
        raise SystemExit("The team table should have 48 rows and no blanks.")