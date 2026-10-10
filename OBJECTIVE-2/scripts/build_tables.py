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
         python build_tables.py ./data

Team names must be spelled IDENTICALLY in all four files (the matches.csv
spelling is taken as standard). The script stops with a clear message if they are not
"""
import os
import sys

import pandas as pd

FOLDER = (sys.argv[1] if len(sys.argv) > 1
          else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
MATCHES_FILE = "matches.csv"
CONTEXT_FILE = "team_context.csv"
SQUAD_FILE = "team_squad.csv"
ELO_FILE = "team_elo_qual.csv"

REST_CAP = 7          # rest days capped at 7; first match = 7


def read(name):
    return pd.read_csv(os.path.join(FOLDER, name))


def build_team_table(matches):
    # One row per team with everything we know about it before kick off
    ctx = read(CONTEXT_FILE)
    squad = read(SQUAD_FILE).rename(columns={"average caps": "avg_caps"})
    elo = read(ELO_FILE)

    teams = set(matches.team_a) | set(matches.team_b)
    for name, df in [("context", ctx), ("squad", squad), ("elo", elo)]:
        assert len(df) == 48 and df.team.is_unique, f"{name} file must have 48 unique teams"
        missing, extra = teams - set(df.team), set(df.team) - teams
        assert not missing and not extra, (
            f"Team names in the {name} file do not match matches.csv.\n"
            f"  missing from {name} file: {sorted(missing)}\n"
            f"  not in matches.csv:       {sorted(extra)}")

    t = (ctx.merge(squad, on="team", validate="1:1")
            .merge(elo, on="team", validate="1:1"))
    assert len(t) == 48 and not t.isna().any().any(), "team table has blanks"
    return t


def build_long(matches, teams):
    """One row per team per match (208 rows), with the pre-kickoff variables."""
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["knockout"] = (~m["stage"].str.startswith("Group")).astype(int)

    a = pd.DataFrame({"match_id": m.match_id, "date": m.date, "knockout": m.knockout,
                      "side": "A", "team": m.team_a, "opp": m.team_b,
                      "gf": m.goals_a, "ga": m.goals_b})
    b = pd.DataFrame({"match_id": m.match_id, "date": m.date, "knockout": m.knockout,
                      "side": "B", "team": m.team_b, "opp": m.team_a,
                      "gf": m.goals_b, "ga": m.goals_a})
    long = pd.concat([a, b], ignore_index=True)
    long = long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g = long.groupby("team")

    # rest days: days since this team's previous match (7 for the first, capped at 7)
    gap = (long["date"] - g["date"].shift(1)).dt.days
    long["rest_days"] = gap.fillna(REST_CAP).clip(upper=REST_CAP).astype(int)

    # tournament-to-date goals per game, EXCLUDING the current match
    long["games_before"] = g.cumcount()
    long["gf_before"] = g["gf"].cumsum() - long["gf"]
    long["ga_before"] = g["ga"].cumsum() - long["ga"]

    long = long.merge(teams, on="team", how="left", validate="m:1")
    first = long["games_before"] == 0
    # matchday 1 fallback (fixed rule): use the qualifying average, never 0
    long["gf_pg"] = (long["gf_before"] / long["games_before"].where(~first)).where(~first, long["qual_gf_pg"])
    long["ga_pg"] = (long["ga_before"] / long["games_before"].where(~first)).where(~first, long["qual_ga_pg"])

    # opponent's pre-kickoff values
    opp = teams[["team", "elo", "qual_ga_pg"]].rename(
        columns={"team": "opp", "elo": "opp_elo", "qual_ga_pg": "opp_qual_ga_pg"})
    long = long.merge(opp, on="opp", how="left", validate="m:1")
    long["is_uefa_conmebol"] = long["confederation"].isin(["UEFA", "CONMEBOL"]).astype(int)
    return long.sort_values(["match_id", "side"]).reset_index(drop=True)


def main():
    matches = read(MATCHES_FILE)
    assert len(matches) == 104 and matches.match_id.is_unique, "matches.csv must have 104 unique matches"
    assert not matches.isna().any().any(), "matches.csv has blanks"

    teams = build_team_table(matches)
    long = build_long(matches, teams)
    assert len(long) == 208

    A = long[long.side == "A"].set_index("match_id")
    B = long[long.side == "B"].set_index("match_id")

    # ---------- Model 2.1: one row per match, Team A minus Team B ----------
    m21 = pd.DataFrame({
        "fifa_points_diff": A.fifa_points - B.fifa_points,
        "avg_caps_diff": A.avg_caps - B.avg_caps,
        "avg_age_diff": A.avg_age - B.avg_age,
        "host_diff": A.is_host - B.is_host,
        "knockout": A.knockout,
        "rest_days_diff": A.rest_days - B.rest_days,
        "gf_pg_diff": A.gf_pg - B.gf_pg,
        "ga_pg_diff": A.ga_pg - B.ga_pg,
        "goal_diff": A.gf - A.ga,
    }).reset_index()                       # match_id becomes the first column

    # ---------- Model 2.2: one row per team per match ----------
    m22 = pd.DataFrame({
        "match_id": long.match_id,
        "own_elo": long.elo,
        "opp_elo": long.opp_elo,
        "own_qual_gf_pg": long.qual_gf_pg,
        "opp_qual_ga_pg": long.opp_qual_ga_pg,
        "is_host": long.is_host,
        "knockout": long.knockout,
        "rest_days": long.rest_days,
        "is_uefa_conmebol": long.is_uefa_conmebol,
        "goals_for": long.gf,
    })

    # ---------- checks ----------
    assert m21.shape == (104, 10), m21.shape      # match_id + 8 + target
    assert m22.shape == (208, 10), m22.shape
    assert not m21.isna().any().any() and not m22.isna().any().any(), "blanks in model tables"
    assert (m21.goal_diff == matches.set_index("match_id").loc[m21.match_id, "goals_a"].values
            - matches.set_index("match_id").loc[m21.match_id, "goals_b"].values).all()
    assert long.rest_days.between(1, REST_CAP).all()
    assert (long.groupby("team").size() >= 3).all()
    assert long.groupby("match_id").size().eq(2).all()
    assert m22.goals_for.sum() == matches.goals_a.sum() + matches.goals_b.sum()

    out = lambda n: os.path.join(FOLDER, n)
    m21 = m21.round(4)                     # tidy floating-point noise (e.g. 0.0500000001)
    m22 = m22.round(4)
    m21.to_csv(out("model_2_1.csv"), index=False)
    m22.to_csv(out("model_2_2.csv"), index=False)
    long.to_csv(out("merged_long.csv"), index=False)
    print("model_2_1.csv", m21.shape, "| model_2_2.csv", m22.shape)
    print("\nModel 2.1 summary:\n", m21.drop(columns="match_id").describe().round(2).T.to_string())
    print("\nModel 2.2 summary:\n", m22.drop(columns="match_id").describe().round(2).T.to_string())


if __name__ == "__main__":
    main()