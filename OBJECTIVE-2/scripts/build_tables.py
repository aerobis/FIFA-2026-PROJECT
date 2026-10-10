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

    return teams.set_index("team").to_dict("index")

def make_long_table(matches, info):
    """One row per team per match (208 rows) in total
 
    We go through the matches in date order and keep a running record of
    each team's games so far, so every row only uses information from
    BEFORE that match"""
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "match_id"])
 
    # Running record for every team
    record = {}
    for team in info:
        record[team] = {"last_date": None, "games": 0, "goals_for": 0, "goals_against": 0}
 
    rows = []
 
    for i, m in matches.iterrows():
        # 1 for knockout round, 0 for group match
        if m["stage"].startswith("Group"):
            knockout = 0
        else:
            knockout = 1
 
        # Each match gives two rows: Team A's view and Team B's view
        views = [
            ("A", m["team_a"], m["team_b"], m["goals_a"], m["goals_b"]),
            ("B", m["team_b"], m["team_a"], m["goals_b"], m["goals_a"]),
        ]
 
        for side, team, opp, goals_for, goals_against in views:
            rec = record[team]
 
            # Rest days: days since this team's previous match
            # First match = 7, and rest days are capped to 7
            if rec["last_date"] is None:
                rest_days = 7
            else:
                rest_days = (m["date"] - rec["last_date"]).days
                if rest_days > 7:
                    rest_days = 7
 
            # Goals per game so far in the tournament (this match NOT included)
            # On a team's first match there is nothing yet, so we use its
            # qualifying averages instead (never 0)
            if rec["games"] == 0:
                gf_pg = info[team]["qual_gf_pg"]
                ga_pg = info[team]["qual_ga_pg"]
            else:
                gf_pg = rec["goals_for"] / rec["games"]
                ga_pg = rec["goals_against"] / rec["games"]
 
            if info[team]["confederation"] in ["UEFA", "CONMEBOL"]:
                is_uefa_conmebol = 1
            else:
                is_uefa_conmebol = 0
 
            rows.append({
                "match_id": m["match_id"],
                "side": side,
                "team": team,
                "opp": opp,
                "knockout": knockout,
                "rest_days": rest_days,
                "gf_pg": gf_pg,
                "ga_pg": ga_pg,
                "fifa_points": info[team]["fifa_points"],
                "avg_caps": info[team]["avg_caps"],
                "avg_age": info[team]["avg_age"],
                "is_host": info[team]["is_host"],
                "elo": info[team]["elo"],
                "qual_gf_pg": info[team]["qual_gf_pg"],
                "opp_elo": info[opp]["elo"],
                "opp_qual_ga_pg": info[opp]["qual_ga_pg"],
                "is_uefa_conmebol": is_uefa_conmebol,
                "goals_for": goals_for,
                "goals_against": goals_against,
            })
 
        # now update the running records
        for side, team, opp, goals_for, goals_against in views:
            record[team]["last_date"] = m["date"]
            record[team]["games"] = record[team]["games"] + 1
            record[team]["goals_for"] = record[team]["goals_for"] + goals_for
            record[team]["goals_against"] = record[team]["goals_against"] + goals_against
 
    long = pd.DataFrame(rows)
    long = long.sort_values(["match_id", "side"]).reset_index(drop=True)
    print("Team-match rows:", len(long))
    return long

def make_model_2_1(long):
    # One row per match: Team A's value - Team B's value
    A = long[long["side"] == "A"].set_index("match_id")
    B = long[long["side"] == "B"].set_index("match_id")
 
    table = pd.DataFrame({
        "fifa_points_diff": A["fifa_points"] - B["fifa_points"],
        "avg_caps_diff": A["avg_caps"] - B["avg_caps"],
        "avg_age_diff": A["avg_age"] - B["avg_age"],
        "host_diff": A["is_host"] - B["is_host"],
        "knockout": A["knockout"],
        "rest_days_diff": A["rest_days"] - B["rest_days"],
        "gf_pg_diff": A["gf_pg"] - B["gf_pg"],
        "ga_pg_diff": A["ga_pg"] - B["ga_pg"],
        "goal_diff": A["goals_for"] - A["goals_against"],    # target becomes the last column
    })
    return table.reset_index()    # match_id becomes the first column

def make_model_2_2(long):
    # One row per team per match
    return pd.DataFrame({
        "match_id": long["match_id"],
        "own_elo": long["elo"],
        "opp_elo": long["opp_elo"],
        "own_qual_gf_pg": long["qual_gf_pg"],
        "opp_qual_ga_pg": long["opp_qual_ga_pg"],
        "is_host": long["is_host"],
        "knockout": long["knockout"],
        "rest_days": long["rest_days"],
        "is_uefa_conmebol": long["is_uefa_conmebol"],
        "goals_for": long["goals_for"],                      # target is last column
    })