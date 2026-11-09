"""
STEP 1 — LOAD & EXPLORE
========================
Purpose: Load the two raw CSV files and confirm the data looks correct
before doing any analysis. Always start here.

Dataset source (download both files, place in the same folder as this script):
https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
  -> matches.csv
  -> deliveries.csv
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---- Load ----
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

print("=" * 60)
print("MATCHES.CSV")
print("=" * 60)
print("Shape:", matches.shape)
print("\nColumns:", list(matches.columns))
print("\nFirst 5 rows:")
print(matches.head())
print("\nMissing values per column:")
print(matches.isnull().sum())

print("\n" + "=" * 60)
print("DELIVERIES.CSV")
print("=" * 60)
print("Shape:", deliveries.shape)
print("\nColumns:", list(deliveries.columns))
print("\nFirst 5 rows:")
print(deliveries.head())
print("\nMissing values per column:")
print(deliveries.isnull().sum())

# ---- Quick sanity checks ----
print("\nUnique seasons:", sorted(matches["season"].astype(str).unique()) if "season" in matches.columns else "column not found - check names")
print("Unique teams (team1):", matches["team1"].nunique() if "team1" in matches.columns else "check column name")
print("Total unique matches in deliveries:", deliveries["match_id"].nunique() if "match_id" in deliveries.columns else "check column name (may be 'id')")

# NOTE: Column names differ slightly between dataset versions.
# Common variants you may see:
#   batsman / batter          -> who's on strike
#   match_id / id             -> links deliveries to matches
#   winner / match_winner     -> match result
# If any KeyError appears in later scripts, print(deliveries.columns) and
# matches.columns first, then adjust the column names used below to match.
