"""
STEP 4 — TEAM STRENGTH + WIN PREDICTION MODEL (PHASE 2)
===========================================================
Purpose:
  1. For every historical match, work out which players actually played
     (derived from deliveries.csv - anyone who batted or bowled for a
     team in that match_id is treated as part of the Playing XI).
  2. Convert that XI into a single Team Strength Score using the
     player_scores.csv from Step 3.
  3. Train a classifier that predicts the winner using:
     team1_strength, team2_strength, strength_diff, toss_winner, toss_decision, venue.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

matches = pd.read_csv("matches_clean.csv")
deliveries = pd.read_csv("deliveries_clean.csv")
players = pd.read_csv("player_scores.csv")

score_lookup = dict(zip(players["player"], players["overall_impact_score"]))
DEFAULT_SCORE = players["overall_impact_score"].median()  # fallback for players with too few balls to be scored

# -----------------------------------------------------------------
# PART A — DERIVE PLAYING XI PER TEAM PER MATCH, THEN TEAM STRENGTH
# -----------------------------------------------------------------
def team_strength_for_match(match_id, team_name):
    """Average Impact Score of everyone who batted or bowled for `team_name` in this match."""
    rows = deliveries[deliveries["match_id"] == match_id]
    batters = rows[rows["batting_team"] == team_name]["batsman"].unique()
    bowlers = rows[rows["bowling_team"] == team_name]["bowler"].unique()
    xi = set(batters) | set(bowlers)
    if not xi:
        return np.nan
    scores = [score_lookup.get(p, DEFAULT_SCORE) for p in xi]
    return round(np.mean(scores), 2)

print("Calculating team strength for every historical match (this may take a minute)...")
matches["team1_strength"] = matches.apply(lambda r: team_strength_for_match(r["match_id"], r["team1"]), axis=1)
matches["team2_strength"] = matches.apply(lambda r: team_strength_for_match(r["match_id"], r["team2"]), axis=1)
matches = matches.dropna(subset=["team1_strength", "team2_strength"])
matches["strength_diff"] = matches["team1_strength"] - matches["team2_strength"]

# -----------------------------------------------------------------
# PART B — FEATURE ENGINEERING
# -----------------------------------------------------------------
matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)  # target label

le_toss_winner = LabelEncoder()
le_toss_decision = LabelEncoder()
le_venue = LabelEncoder()
le_team1 = LabelEncoder()
le_team2 = LabelEncoder()

matches["toss_winner_enc"] = le_toss_winner.fit_transform(matches["toss_winner"].astype(str))
matches["toss_decision_enc"] = le_toss_decision.fit_transform(matches["toss_decision"].astype(str))
matches["venue_enc"] = le_venue.fit_transform(matches["venue"].astype(str))
matches["team1_enc"] = le_team1.fit_transform(matches["team1"].astype(str))
matches["team2_enc"] = le_team2.fit_transform(matches["team2"].astype(str))

FEATURES = [
    "team1_strength", "team2_strength", "strength_diff",
    "toss_winner_enc", "toss_decision_enc", "venue_enc",
    "team1_enc", "team2_enc",
]
X = matches[FEATURES]
y = matches["team1_won"]

# -----------------------------------------------------------------
# PART C — TRAIN / TEST SPLIT + MODEL
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# PART D — EVALUATION
# -----------------------------------------------------------------
y_pred = model.predict(X_test)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

feat_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importance:")
print(feat_importance)

# -----------------------------------------------------------------
# PART E — SAVE EVERYTHING NEEDED FOR THE DASHBOARD
# -----------------------------------------------------------------
joblib.dump(model, "win_prediction_model.pkl")
joblib.dump(le_toss_winner, "le_toss_winner.pkl")
joblib.dump(le_toss_decision, "le_toss_decision.pkl")
joblib.dump(le_venue, "le_venue.pkl")
joblib.dump(le_team1, "le_team1.pkl")
joblib.dump(le_team2, "le_team2.pkl")

print("\nSaved model + encoders. Ready for Step 5 (Streamlit dashboard).")
