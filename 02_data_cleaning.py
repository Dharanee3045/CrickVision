"""
STEP 2 — DATA CLEANING
========================
Purpose: Fix inconsistent team names, handle missing values, and
standardize columns so every later step works on clean data.
"""

import pandas as pd

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# ---- Standardize column names (handles both old & new dataset versions) ----
matches.columns = [c.strip().lower() for c in matches.columns]
deliveries.columns = [c.strip().lower() for c in deliveries.columns]

if "batter" in deliveries.columns and "batsman" not in deliveries.columns:
    deliveries = deliveries.rename(columns={"batter": "batsman"})
if "id" in matches.columns and "match_id" not in matches.columns:
    matches = matches.rename(columns={"id": "match_id"})

# ---- Fix inconsistent / renamed franchise names ----
TEAM_NAME_FIX = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

for col in ["team1", "team2", "toss_winner", "winner"]:
    if col in matches.columns:
        matches[col] = matches[col].replace(TEAM_NAME_FIX)

for col in ["batting_team", "bowling_team"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].replace(TEAM_NAME_FIX)

# ---- Drop matches with no result (abandoned / no-result games) ----
if "result" in matches.columns:
    matches = matches[matches["result"] != "no result"]
matches = matches.dropna(subset=["winner"])

# ---- Handle missing values ----
# city sometimes missing where venue name implies the city -> fill with 'Unknown'
if "city" in matches.columns:
    matches["city"] = matches["city"].fillna("Unknown")

# player_dismissed / dismissal_kind are legitimately blank on non-wicket balls -> fill blank string
for col in ["player_dismissed", "dismissal_kind"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].fillna("")

# ---- Save cleaned versions ----
matches.to_csv("matches_clean.csv", index=False)
deliveries.to_csv("deliveries_clean.csv", index=False)

print("Cleaning complete.")
print("matches_clean.csv  ->", matches.shape)
print("deliveries_clean.csv ->", deliveries.shape)
