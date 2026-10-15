"""
build_elo_qual.py - builds team_elo_qual.csv dataset for Objective 2

For each of the 48 World Cup teams it downloads that team's match history
from eloratings.net (one .tsv file per team) and works out:

  elo         : the team's Elo rating going INTO the tournament
                (its rating after the last match played before 11 June 2026)
  qual_gf_pg  : goals scored per game in World Cup qualifying (code is "WQ")
  qual_ga_pg  : goals conceded per game in World Cup qualifying

Hosts (Mexico, United States, Canada) have no qualifying campaign, so 
their last 8 competitive (non-friendly) matches before the
tournament are used instead.

Run:     python scripts/build_elo_qual.py
Output:  Writes into the 'data/' folder: 
            team_elo_qual.csv         (final dataset)
            team_elo_qual_checks.csv  (extra columns for checking the numbers)
            host_matches_used.txt     (the 8 matches used for each host)
"""
import csv
import os
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
BASE = "https://www.eloratings.net/"
CUTOFF = 20260611        # tournament start: only matches BEFORE this date count
QUAL_START = 20230101    # start of the 2026 qualifying cycle
QUAL_END = 20251231      # excludes the March 2026 play-offs
HOST_MATCHES = 8         

# If a team's file cannot be found, put its eloratings.net file name here
FILE_OVERRIDES = {}

# (team name exactly as in matches.csv / team_context.csv, eloratings.net team code, and is_host)
TEAMS = [
    ("Algeria", "DZ", False), ("Argentina", "AR", False),
    ("Australia", "AU", False), ("Austria", "AT", False),
    ("Belgium", "BE", False), ("Bosnia and Herzegovina", "BA", False),
    ("Brazil", "BR", False), ("Canada", "CA", True),
    ("Cabo Verde", "CV", False), ("Colombia", "CO", False),
    ("Croatia", "HR", False), ("Curaçao", "CW", False),
    ("Czechia", "CZ", False), ("Congo DR", "CD", False),
    ("Ecuador", "EC", False), ("Egypt", "EG", False),
    ("England", "EN", False), ("France", "FR", False),
    ("Germany", "DE", False), ("Ghana", "GH", False),
    ("Haiti", "HT", False), ("IR Iran", "IR", False),
    ("Iraq", "IQ", False), ("Côte d'Ivoire", "CI", False),
    ("Japan", "JP", False), ("Jordan", "JO", False),
    ("Mexico", "MX", True), ("Morocco", "MA", False),
    ("Netherlands", "NL", False), ("New Zealand", "NZ", False),
    ("Norway", "NO", False), ("Panama", "PA", False),
    ("Paraguay", "PY", False), ("Portugal", "PT", False),
    ("Qatar", "QA", False), ("Saudi Arabia", "SA", False),
    ("Scotland", "SQ", False), ("Senegal", "SN", False),
    ("South Africa", "ZA", False), ("Korea Republic", "KR", False),
    ("Spain", "ES", False), ("Sweden", "SE", False),
    ("Switzerland", "CH", False), ("Tunisia", "TN", False),
    ("Türkiye", "TR", False), ("USA", "US", True),
    ("Uruguay", "UY", False), ("Uzbekistan", "UZ", False),
]


def fetch(url):
    # Percent encode non-ASCII characters 
    # urllib refuses to send them raw. Existing %20 etc. are left alone
    url = urllib.parse.quote(url, safe=":/%")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def load_site_names():
    # eloratings.net's own name for each team code (from en.teams.tsv)
    names = {}
    for line in fetch(BASE + "en.teams.tsv").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names[parts[0]] = parts[1]
    return names


def download_team_file(code, site_name):
    # try a few likely file names for the team's .tsv; return (name, text)
    plain = unicodedata.normalize("NFKD", site_name).encode("ascii", "ignore").decode()
    candidates = [
        FILE_OVERRIDES.get(code),
        site_name.replace(" ", "_"),
        site_name.replace(" ", "%20"),
        site_name,
        plain.replace(" ", "_"),
        plain,
        code,
    ]
    tried = []
    for cand in candidates:
        if not cand or cand in tried:
            continue
        tried.append(cand)
        try:
            return cand, fetch(BASE + cand + ".tsv")
        except urllib.error.HTTPError:
            continue
    raise RuntimeError(
        f"Could not download a file for {site_name} ({code}). Tried: {tried}. "
        f"Open the team's page on eloratings.net and check the file name in "
        f"the address bar, then put it in FILE_OVERRIDES at the top of this script."
    )


def parse_matches(text, code):
    """One dict per match involving `code`, in date order

    Column layout of the eloratings.net team files (tab separated) are as follows:
      0: year, 1: month, 2: day, 3: home code, 4: away code, 5: home goals,
      6: away goals, 7: tournament code (F = friendly, WQ = World Cup
      qualifier), 8: location, 9: rating change, 10: home rating AFTER the
      match, 11: away rating AFTER the match, then finally: rank columns
    """
    matches = []
    for line in text.splitlines():
        r = line.split("\t")
        if len(r) < 12 or code not in (r[3], r[4]):
            continue
        is_home = r[3] == code
        matches.append({
            "date": int(r[0]) * 10000 + int(r[1]) * 100 + int(r[2]),
            "comp": r[7],
            "opp": r[4] if is_home else r[3],
            "gf": int(r[5] if is_home else r[6]),
            "ga": int(r[6] if is_home else r[5]),
            "elo_after": int(r[10] if is_home else r[11]),
        })
    matches.sort(key=lambda m: m["date"])   # to keep same-day order
    return matches


def team_numbers(matches, is_host):
    # Return the columns (elo, games, gf_per_game, ga_per_game, matches_used)
    before = [m for m in matches if m["date"] < CUTOFF]
    if not before:
        raise ValueError("no matches before the tournament")
    elo = before[-1]["elo_after"]

    if is_host:
        used = [m for m in before if m["comp"] != "F"][-HOST_MATCHES:]
    else:
        used = [m for m in before
                if m["comp"] == "WQ" and QUAL_START <= m["date"] <= QUAL_END]

    if not used:
        return elo, 0, None, None, used
    games = len(used)
    gf = sum(m["gf"] for m in used) / games
    ga = sum(m["ga"] for m in used) / games
    return elo, games, round(gf, 3), round(ga, 3), used


def main():
    site_names = load_site_names()
    main_rows, check_rows, host_log = [], [], []

    for name, code, is_host in TEAMS:
        try:
            file_used, text = download_team_file(code, site_names.get(code, name))
            matches = parse_matches(text, code)
            elo, games, gf, ga, used = team_numbers(matches, is_host)
        except Exception as err:
            print(f"PROBLEM  {name}: {err}")
            main_rows.append([name, "", "", ""])
            check_rows.append([name, "", "", "", "", "FAILED: " + str(err)])
            continue

        main_rows.append([name, elo, gf, ga])
        note = "host rule (last 8 non-friendly)" if is_host else "WQ matches 2023-2025"
        check_rows.append([name, elo, gf, ga, games, f"{file_used}.tsv; {note}"])
        if is_host:
            host_log.append(f"{name}:")
            host_log += [f"  {m['date']}  vs {m['opp']}  {m['gf']}-{m['ga']}  ({m['comp']})"
                         for m in used]
        if not is_host and games < 4:
            print(f"CHECK    {name}: only {games} qualifying games found")
        print(f"ok       {name}: elo {elo}, {games} games, GF/g {gf}, GA/g {ga}")
        time.sleep(0.5)   # to avoid exhausting the website

    with open(os.path.join(DATA_DIR, "team_elo_qual.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "elo", "qual_gf_pg", "qual_ga_pg"])
        w.writerows(main_rows)
    with open(os.path.join(DATA_DIR, "team_elo_qual_checks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "elo", "qual_gf_pg", "qual_ga_pg", "games", "source_note"])
        w.writerows(check_rows)
    with open(os.path.join(DATA_DIR, "host_matches_used.txt"), "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(host_log) + "\n")
    print("\nWrote team_elo_qual.csv, team_elo_qual_checks.csv, host_matches_used.txt")


if __name__ == "__main__":
    main()