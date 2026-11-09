"""CrickVision Streamlit dashboard.

Pages: Player Leaderboard, Player Search, Team Builder, Win Prediction.
Auction valuation is a transparent performance-based prototype until historical
auction prices are supplied. Compatibility scores are heuristic recommendations,
not a scientifically validated probability of success.
"""
from pathlib import Path
import hashlib
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
st.set_page_config(page_title="CrickVision", page_icon="🏏", layout="wide")

# ---------- Load project data ----------
players = pd.read_csv(ROOT / "player_scores.csv")
required_columns = {
    "player", "role", "runs", "balls_faced", "batting_average", "strike_rate",
    "death_over_runs", "death_over_balls", "death_over_strike_rate",
    "batting_impact_score", "balls_bowled", "runs_conceded", "wickets",
    "economy", "bowling_average", "death_over_wickets",
    "death_over_runs_conceded", "death_over_legal_balls", "death_over_economy",
    "bowling_impact_score", "overall_impact_score", "data_confidence"
}
missing = required_columns - set(players.columns)
if missing:
    st.error("player_scores.csv is missing required columns: " + ", ".join(sorted(missing)) + ". Run 03_player_analysis.py first.")
    st.stop()

model = joblib.load(ROOT / "win_prediction_model.pkl")
le_toss_winner = joblib.load(ROOT / "le_toss_winner.pkl")
le_toss_decision = joblib.load(ROOT / "le_toss_decision.pkl")
le_venue = joblib.load(ROOT / "le_venue.pkl")
le_team1 = joblib.load(ROOT / "le_team1.pkl")
le_team2 = joblib.load(ROOT / "le_team2.pkl")

players["player"] = players["player"].astype(str)
players["role"] = players["role"].fillna("Unknown")
players["death_over_run_rate"] = np.where(
    pd.to_numeric(players["death_over_balls"], errors="coerce").fillna(0) > 0,
    6 * pd.to_numeric(players["death_over_runs"], errors="coerce").fillna(0) /
    pd.to_numeric(players["death_over_balls"], errors="coerce").replace(0, np.nan), 0
)
players["death_over_run_rate"] = players["death_over_run_rate"].fillna(0).round(2)
score_lookup = dict(zip(players["player"], pd.to_numeric(players["overall_impact_score"], errors="coerce").fillna(0)))
DEFAULT_SCORE = float(pd.to_numeric(players["overall_impact_score"], errors="coerce").median())
all_teams = sorted(set(le_team1.classes_) & set(le_team2.classes_))

# Keep selections in non-widget session keys. Streamlit may clean up widget keys
# when navigating to another page; these saved keys survive page changes.
def save_team_a_name():
    st.session_state["team_a_name_saved"] = st.session_state.get("team_a_widget", all_teams[0] if all_teams else "")
def save_team_b_name():
    st.session_state["team_b_name_saved"] = st.session_state.get("team_b_widget", all_teams[1] if len(all_teams) > 1 else (all_teams[0] if all_teams else ""))
def save_selected_a():
    st.session_state["selected_a_saved"] = list(st.session_state.get("selected_a_widget", []))
def save_selected_b():
    st.session_state["selected_b_saved"] = list(st.session_state.get("selected_b_widget", []))
all_venues = sorted(le_venue.classes_)
all_players = sorted(players["player"].dropna().unique())

# ---------- Styling ----------
st.markdown("""
<style>
.cv-header{background:linear-gradient(120deg,#21295C 0%,#065A82 55%,#1C7293 100%);padding:1.6rem 2rem;border-radius:16px;margin-bottom:1rem}
.cv-header h1{color:white;font-size:2.25rem;font-weight:800;margin:0}
.cv-header p{color:#EAF3F8;font-size:1rem;margin:.35rem 0 0}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:12px}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="cv-header"><h1>🏏 CrickVision</h1><p>IPL player analysis · team composition · compatibility · win prediction</p></div>', unsafe_allow_html=True)

# ---------- Helpers ----------
def get_initials(name):
    parts = [p for p in str(name).replace(".", " ").split() if p]
    if not parts:
        return "??"
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()

AVATAR_COLORS = ["#065A82", "#1C7293", "#21295C", "#E8A33D", "#C0392B", "#2E7D32", "#6A1B9A", "#00838F", "#AD1457", "#5D4037"]
def get_player_photo(name, size=200):
    photo_dir = ROOT / "player_photos"
    if photo_dir.is_dir():
        for candidate in [name, name.replace(" ", "_"), name.replace(" ", "-"), name.lower(), name.lower().replace(" ", "_")]:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                path = photo_dir / (candidate + ext)
                if path.is_file():
                    try:
                        return Image.open(path).convert("RGB").resize((size, size))
                    except Exception:
                        pass
    color = AVATAR_COLORS[int(hashlib.md5(str(name).encode()).hexdigest(), 16) % len(AVATAR_COLORS)]
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    font = None
    for f in ["arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(f, size // 3)
            break
        except Exception:
            continue
    font = font or ImageFont.load_default()
    initials = get_initials(name)
    box = draw.textbbox((0, 0), initials, font=font)
    draw.text(((size-(box[2]-box[0]))/2-box[0], (size-(box[3]-box[1]))/2-box[1]), initials, fill="white", font=font)
    return img

BIO_FIELD_CANDIDATES = {
    "player": ["player", "player_name", "name", "batsman", "batter"],
    "nationality": ["nationality", "country", "nation"],
    "batting_style": ["batting_style", "battingstyle", "batting"],
    "bowling_style": ["bowling_style", "bowlingstyle", "bowling"],
    "date_of_birth": ["date_of_birth", "dob", "born", "birth_date", "birthdate"],
    "playing_role": ["playing_role", "role", "player_role"],
}
def resolve_bio_col(columns, field):
    lookup = {str(c).lower().strip(): c for c in columns}
    return next((lookup[c] for c in BIO_FIELD_CANDIDATES[field] if c in lookup), None)
def load_bio_data():
    frames = []
    manual_path = ROOT / "player_bio.csv"
    if manual_path.exists():
        manual = pd.read_csv(manual_path)
        manual["source"] = "manual"
        frames.append(manual)
    exclude = {"matches.csv", "deliveries.csv", "matches_clean.csv", "deliveries_clean.csv", "player_scores.csv", "player_bio.csv"}
    for f in glob.glob(str(ROOT / "*.csv")):
        if Path(f).name in exclude:
            continue
        try:
            raw = pd.read_csv(f, nrows=1)
            cols = raw.columns
            hits = sum(resolve_bio_col(cols, x) is not None for x in ["nationality", "batting_style", "bowling_style", "date_of_birth"])
            if resolve_bio_col(cols, "player") and hits >= 2:
                full = pd.read_csv(f)
                ren = {resolve_bio_col(full.columns, field): field for field in BIO_FIELD_CANDIDATES if resolve_bio_col(full.columns, field)}
                full = full.rename(columns=ren)
                keep = [c for c in ["player", "nationality", "batting_style", "bowling_style", "date_of_birth", "playing_role"] if c in full.columns]
                full = full[keep].copy()
                full["source"] = Path(f).name
                frames.append(full)
                break
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "player" not in combined.columns:
        return pd.DataFrame()
    combined["player"] = combined["player"].astype(str).str.strip()
    combined = combined.drop_duplicates("player", keep="first")
    if "date_of_birth" in combined.columns:
        dob = pd.to_datetime(combined["date_of_birth"], errors="coerce")
        combined["age"] = ((pd.Timestamp.today().normalize() - dob).dt.days / 365.25).floordiv(1).astype("Int64")
    else:
        combined["age"] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
    return combined.set_index("player")
bio_data = load_bio_data()

def safe_encode(encoder, value):
    return int(encoder.transform([value])[0]) if value in encoder.classes_ else 0

def role_counts(names):
    subset = players[players["player"].isin(names)]
    return subset["role"].value_counts().to_dict()

def percentile_score(series, value, reverse=False):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 50.0
    pct = float((vals <= value).mean() * 100)
    return 100.0 - pct if reverse else pct

def player_fit_score(player_row, selected_names):
    """Heuristic team-fit score: relative to squad needs, not a win probability."""
    current = players[players["player"].isin(selected_names)]
    counts = current["role"].value_counts().to_dict()
    role = str(player_row["role"])
    # Broad role targets for a balanced XI; role categories are approximate.
    targets = {"Batsman": (4, 6), "Bowler": (3, 5), "All-Rounder": (1, 3), "Unknown": (0, 1)}
    lo, hi = targets.get(role, (1, 4))
    have = counts.get(role, 0)
    role_fit = 100 if have < lo else (78 if have < hi else 42)
    if len(selected_names) == 0:
        role_fit = 75
    impact = percentile_score(players["overall_impact_score"], float(player_row["overall_impact_score"]))
    death_bat = percentile_score(players["death_over_run_rate"], float(player_row["death_over_run_rate"]))
    death_bowl = percentile_score(players["death_over_economy"], float(player_row["death_over_economy"]), reverse=True)
    death_wickets = percentile_score(players["death_over_wickets"], float(player_row["death_over_wickets"]))
    # A tiny death-over sample should not create an artificially high fit score.
    if float(player_row.get("death_over_balls", 0) or 0) < 12:
        death_bat = 50.0
    if float(player_row.get("death_over_legal_balls", 0) or 0) < 30:
        death_bowl, death_wickets = 50.0, 50.0
    if role == "Batsman":
        phase_fit = 0.75 * death_bat + 0.25 * impact
    elif role == "Bowler":
        phase_fit = 0.65 * death_bowl + 0.35 * death_wickets
    else:
        phase_fit = 0.4 * death_bat + 0.4 * death_bowl + 0.2 * impact
    # Measure projected squad balance after adding this candidate.
    projected = dict(counts)
    projected[role] = projected.get(role, 0) + 1
    role_balance = 100.0
    for role_name, (minimum, maximum) in {"Batsman": (4, 6), "Bowler": (3, 5), "All-Rounder": (1, 3)}.items():
        count = projected.get(role_name, 0)
        if count < minimum:
            role_balance -= (minimum - count) * 7.0
        elif count > maximum:
            role_balance -= (count - maximum) * 12.0
    role_balance = float(np.clip(role_balance, 0, 100))
    return round(float(np.clip(0.30 * role_fit + 0.25 * role_balance + 0.25 * impact + 0.20 * phase_fit, 0, 100)), 1)

def fit_explanation(row, selected_names):
    role = row["role"]
    counts = role_counts(selected_names)
    if role == "Batsman":
        phase = "death-over batting scoring rate"
    elif role == "Bowler":
        phase = "death-over bowling economy and wickets"
    elif role == "All-Rounder":
        phase = "batting and bowling flexibility"
    else:
        phase = "available performance metrics"
    return f"Adds {role.lower()} depth; assessed using {phase} and Overall Impact Score. Current squad roles: {counts or 'none selected yet'}."

def auction_estimates(row, base_cr):
    """Performance-based scenario estimates for every player, not an auction-trained model.

    Expected is a conservative performance estimate; high-demand models bidding competition.
    The ceiling is capped at INR 33 Cr as a scenario boundary, not a claim that a player will sell for that amount.
    Replace/validate this heuristic with real auction-history training data before calling it an ML prediction.
    """
    def num(key, default=0.0):
        value = pd.to_numeric(row.get(key, default), errors="coerce")
        return float(value) if pd.notna(value) else default
    impact = float(np.clip(num("overall_impact_score"), 0, 10))
    premium = 13.0 * (impact / 10.0) ** 2
    role = str(row.get("role", "Unknown"))
    if role == "All-Rounder": premium += 0.5
    death_rate = num("death_over_run_rate")
    premium += min(max(death_rate - 7.0, 0), 6.0) * 0.12
    death_econ = num("death_over_economy")
    if num("death_over_legal_balls") >= 60:
        premium += max(0, 9.0 - death_econ) * 0.15
    confidence = str(row.get("data_confidence", "Limited Data"))
    if confidence != "Reliable": premium *= 0.65
    expected = round(max(float(base_cr), float(base_cr) + premium), 2)
    # Demand uplift increases with impact and all-rounder scarcity; limited data reduces confidence.
    demand_multiplier = 1.35 + 0.45 * (impact / 10.0) + (0.12 if role == "All-Rounder" else 0.0)
    if confidence != "Reliable": demand_multiplier *= 0.85
    high_demand = round(min(33.0, max(expected, expected * demand_multiplier)), 2)
    return expected, high_demand

def auction_estimate(row, base_cr):
    """Backward-compatible expected estimate used by older page logic."""
    return auction_estimates(row, base_cr)[0]

# ---------- Navigation ----------
page = st.sidebar.radio("Navigate", ["Player Leaderboard", "Player Search", "Player Compatibility", "Team Builder", "Win Prediction"], index=0)
st.sidebar.caption("Player-fit and auction-price figures are prototype estimates unless verified auction/role data is provided.")

# ---------- Player Leaderboard ----------
if page == "Player Leaderboard":
    st.title("🏆 Player Leaderboard")
    st.write("Compare IPL players by their historical performance-based impact scores.")
    with st.expander("💰 All-player auction valuation estimates (prototype)", expanded=False):
        st.caption("Estimates for every player in player_scores.csv. Set a base price below. Values are heuristic scenarios, not actual prices or a trained auction model.")
        all_base_options = {"₹20 Lakh": 0.20, "₹30 Lakh": 0.30, "₹50 Lakh": 0.50, "₹75 Lakh": 0.75, "₹1 Crore": 1.0, "₹1.5 Crore": 1.5, "₹2 Crore": 2.0}
        all_base_label = st.selectbox("Base price for this table", list(all_base_options.keys()), index=6, key="all_player_base")
        valuation_rows = []
        for _, player_row in players.iterrows():
            expected_val, high_val = auction_estimates(player_row, all_base_options[all_base_label])
            valuation_rows.append({"Player": player_row["player"], "Role": player_row["role"], "Overall Impact": round(float(player_row["overall_impact_score"]), 2), "Data Confidence": player_row["data_confidence"], "Expected Estimate (₹ Cr)": expected_val, "High-Demand Scenario (₹ Cr)": high_val})
        valuation_table = pd.DataFrame(valuation_rows).sort_values("Expected Estimate (₹ Cr)", ascending=False)
        st.dataframe(valuation_table, use_container_width=True, hide_index=True)
        st.download_button("Download all-player estimates CSV", valuation_table.to_csv(index=False).encode("utf-8"), file_name="crickvision_all_player_auction_estimates.csv", mime="text/csv")
    c1, c2 = st.columns(2)
    with c1:
        role_filter = st.selectbox("Filter by role", ["All", "Batsman", "Bowler", "All-Rounder", "Unknown"])
    with c2:
        hide_limited = st.checkbox("Hide players with limited data", value=False)
    view = players.copy()
    if role_filter != "All":
        view = view[view["role"] == role_filter]
    if hide_limited:
        view = view[view["data_confidence"] == "Reliable"]
    view = view.sort_values("overall_impact_score", ascending=False)
    st.caption(f"Showing {len(view)} players; {(view['data_confidence'] == 'Limited Data').sum()} have limited data.")
    show_cols = [c for c in ["player", "role", "runs", "wickets", "batting_impact_score", "bowling_impact_score", "overall_impact_score", "data_confidence"] if c in view.columns]
    st.dataframe(view[show_cols].reset_index(drop=True).rename_axis("Rank"), use_container_width=True, hide_index=True)
    st.subheader("Top 10 by Overall Impact")
    st.bar_chart(view.head(10).set_index("player")["overall_impact_score"])

# ---------- Player Search ----------
elif page == "Player Search":
    st.title("🔍 Player Search & Auction Profile")
    search_name = st.selectbox("Search a player", all_players, index=None, placeholder="Start typing a player name...")
    if search_name:
        p = players.loc[players["player"] == search_name].iloc[0]
        bio = bio_data.loc[search_name] if not bio_data.empty and search_name in bio_data.index else None
        col_photo, col_info = st.columns([1, 3])
        with col_photo:
            st.image(get_player_photo(search_name), width=170)
        with col_info:
            st.subheader(search_name)
            st.markdown(f"**Role:** {p['role']}  ·  **Data confidence:** {p['data_confidence']}")
            if p["data_confidence"] != "Reliable":
                st.warning("Limited sample: interpret this player's scores cautiously.")
        st.divider()
        st.subheader("Personal Information")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Age", f"{int(bio['age'])} yrs" if bio is not None and pd.notna(bio.get("age", pd.NA)) else "Not available")
        i2.metric("Nationality", str(bio.get("nationality", "Not available")) if bio is not None and pd.notna(bio.get("nationality", np.nan)) else "Not available")
        i3.metric("Batting Style", str(bio.get("batting_style", "Not available")) if bio is not None and pd.notna(bio.get("batting_style", np.nan)) else "Not available")
        i4.metric("Bowling Style", str(bio.get("bowling_style", "Not available")) if bio is not None and pd.notna(bio.get("bowling_style", np.nan)) else "Not available")
        st.subheader("Batting Stats")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Runs", int(p["runs"]))
        b2.metric("Batting Average", f"{float(p['batting_average']):.1f}")
        b3.metric("Strike Rate", f"{float(p['strike_rate']):.1f}")
        b4.metric("Batting Impact Score", f"{float(p['batting_impact_score']):.2f}")
        st.subheader("Bowling Stats")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Wickets", int(p["wickets"]))
        w2.metric("Economy", f"{float(p['economy']):.2f}")
        w3.metric("Bowling Average", f"{float(p['bowling_average']):.1f}")
        w4.metric("Bowling Impact Score", f"{float(p['bowling_impact_score']):.2f}")
        st.subheader("Pressure Situation Performance — Death Overs (16–20)")
        st.markdown("**Death-over batting**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Runs Scored", int(p["death_over_runs"]))
        d2.metric("Strike Rate", f"{float(p['death_over_strike_rate']):.1f}")
        d3.metric("Run Rate (runs per 6 balls faced)", f"{float(p['death_over_run_rate']):.2f}")
        st.markdown("**Death-over bowling**")
        d4, d5, d6 = st.columns(3)
        d4.metric("Wickets Taken", int(p["death_over_wickets"]))
        d5.metric("Runs Conceded", int(p["death_over_runs_conceded"]))
        d6.metric("Economy", f"{float(p['death_over_economy']):.2f}")
        st.caption("Bowler runs conceded exclude byes and leg-byes. Death-over batting run rate uses the batter's recorded balls faced.")
        st.divider()
        st.subheader("💰 Base Price → Predicted Sold Value")
        st.caption("Choose the player's assumed/known base price. Because no actual auction-history dataset was included, this is a transparent performance-based prototype, not an auction-trained ML prediction.")
        base_options = {"₹20 Lakh (₹0.20 Cr)": 0.20, "₹30 Lakh (₹0.30 Cr)": 0.30, "₹50 Lakh (₹0.50 Cr)": 0.50, "₹75 Lakh (₹0.75 Cr)": 0.75, "₹1 Crore": 1.0, "₹1.5 Crore": 1.5, "₹2 Crore": 2.0}
        chosen_base = st.selectbox("Base price", list(base_options.keys()), index=0, key=f"base_{search_name}")
        base_cr = base_options[chosen_base]
        sold, high_demand = auction_estimates(p, base_cr)
        v1, v2 = st.columns(2)
        v1.metric("Base Price", f"₹{base_cr:.2f} Cr")
        v2.metric("Expected Estimate", f"₹{sold:.2f} Cr")
    else:
        st.info("Select a player to view their statistics, death-over run rate, and valuation prototype.")

# ---------- Player Compatibility ----------
elif page == "Player Compatibility":
    st.title("🧩 Player Compatibility & Team Fit")
    st.write("Evaluate how well a player fits a squad's role balance and performance needs. Scores are heuristic recommendations, not win probabilities.")

    # Franchise-based comparison: rank 4–5 candidates against selected team needs.
    st.subheader("🧩 Player Compatibility Analysis")
    st.caption("Choose a franchise and the roles it needs. CrickVision ranks five available player options using performance statistics and role fit.")
    comp_c1, comp_c2 = st.columns([1, 2])
    with comp_c1:
        comp_team = st.selectbox("Select IPL team", all_teams, key="compat_franchise_team")
    need_options = ["Death-over bowler", "Lower-order finisher", "Top-order batter", "All-rounder", "Wicketkeeper-batter", "Powerplay bowler", "Economical bowler"]
    with comp_c2:
        comp_needs = st.multiselect("Current team needs", need_options, default=["Death-over bowler", "Lower-order finisher"], key="compat_franchise_needs")

    def compatibility_for_need(row, needs):
        role = str(row.get("role", "Unknown"))
        impact = percentile_score(players["overall_impact_score"], float(row.get("overall_impact_score", 0) or 0))
        death_bat = percentile_score(players["death_over_run_rate"], float(row.get("death_over_run_rate", 0) or 0))
        death_econ = percentile_score(players["death_over_economy"], float(row.get("death_over_economy", 0) or 0), reverse=True)
        death_wickets = percentile_score(players["death_over_wickets"], float(row.get("death_over_wickets", 0) or 0))
        bat_score = percentile_score(players["strike_rate"], float(row.get("strike_rate", 0) or 0))
        wicket_score = percentile_score(players["wickets"], float(row.get("wickets", 0) or 0))
        parts=[]
        for need in needs:
            if need == "Death-over bowler": parts.append(0.55*death_econ + 0.30*death_wickets + 0.15*impact)
            elif need == "Lower-order finisher": parts.append(0.45*death_bat + 0.30*bat_score + 0.25*impact)
            elif need == "Top-order batter": parts.append(0.45*bat_score + 0.30*impact + 0.25*death_bat)
            elif need == "All-rounder": parts.append((75 if role == "All-Rounder" else 35) * 0.55 + impact * 0.45)
            elif need == "Wicketkeeper-batter": parts.append((80 if "keeper" in role.lower() else (65 if role == "Batsman" else 40))*0.45 + bat_score*0.30 + impact*0.25)
            elif need == "Powerplay bowler": parts.append(0.45*wicket_score + 0.35*death_wickets + 0.20*impact if role in ["Bowler", "All-Rounder"] else 0.45*wicket_score + 0.2*impact)
            elif need == "Economical bowler": parts.append(0.55*death_econ + 0.25*wicket_score + 0.20*impact if role in ["Bowler", "All-Rounder"] else 0.4*death_econ + 0.2*impact)
        if not parts:
            parts=[0.55*impact + 0.25*death_bat + 0.20*death_econ]
        return round(float(np.clip(np.mean(parts), 0, 100)), 1)

    if comp_needs:
        comp_candidates = players.copy()
        # Avoid candidates already selected in either current XI, when those lists exist.
        already_selected = set(st.session_state.get("selected_a_saved", [])) | set(st.session_state.get("selected_b_saved", []))
        if already_selected:
            comp_candidates = comp_candidates[~comp_candidates["player"].isin(already_selected)]
        comp_candidates["Compatibility"] = comp_candidates.apply(lambda r: compatibility_for_need(r, comp_needs), axis=1)
        comp_candidates = comp_candidates.sort_values(["Compatibility", "overall_impact_score"], ascending=False).head(5)
        st.markdown(f"**{comp_team} — best player options for your selected needs**")
        for rank, (_, candidate) in enumerate(comp_candidates.iterrows(), start=1):
            role = str(candidate.get("role", "Player"))
            if "Death-over bowler" in comp_needs and role in ["Bowler", "All-Rounder"]:
                reason = "Death-over bowling metrics and wickets are weighted for this need."
            elif "Lower-order finisher" in comp_needs and role in ["Batsman", "All-Rounder"]:
                reason = "Death-over scoring rate, strike rate and overall impact are weighted for finishing."
            elif "All-rounder" in comp_needs and role == "All-Rounder":
                reason = "Provides both batting and bowling flexibility, with overall impact included."
            else:
                reason = f"Ranked using the selected needs ({', '.join(comp_needs)}), role and available performance statistics."
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{rank}. {candidate['player']}** · {role}")
                    st.write(reason)
                with right:
                    st.metric("Fit score", f"{candidate['Compatibility']:.1f}%")
                if rank == 1:
                    st.success(f"Best fit for {comp_team} from the available dataset based on the needs you selected.")
        st.caption("Fit scores are calculated heuristic compatibility scores, not probabilities of success or official franchise decisions. The selected franchise is a comparison context; franchise-specific needs are not inferred from private scouting data.")
    else:
        st.info("Select at least one current team need to compare the best five player options.")


    saved_a = st.session_state.get("selected_a_saved", [])
    saved_b = st.session_state.get("selected_b_saved", [])
    source_options = ["Build a custom squad"]
    if saved_a:
        source_options.insert(0, f"Use Team A XI ({len(saved_a)} selected)")
    if saved_b:
        source_options.insert(0, f"Use Team B XI ({len(saved_b)} selected)")
    source = st.radio("Squad to evaluate", source_options, horizontal=True, key="compat_squad_source")
    if source.startswith("Use Team A"):
        squad = list(saved_a)
        squad_label = st.session_state.get("team_a_name_saved", "Team A")
    elif source.startswith("Use Team B"):
        squad = list(saved_b)
        squad_label = st.session_state.get("team_b_name_saved", "Team B")
    else:
        squad = st.multiselect("Select current squad players (up to 11)", all_players, max_selections=11, key="compat_custom_squad")
        squad_label = "Custom Squad"

    squad_df = players[players["player"].isin(squad)].copy()
    role_order = ["Batsman", "Bowler", "All-Rounder", "Unknown"]
    counts = squad_df["role"].value_counts().to_dict()
    st.subheader(f"{squad_label} — Squad balance")
    mcols = st.columns(4)
    for col, role_name in zip(mcols, role_order):
        col.metric(role_name, int(counts.get(role_name, 0)))
    if not squad:
        st.info("Select a squad or build Team A/Team B in Team Builder to see compatibility recommendations.")
    else:
        needs = []
        targets = {"Batsman": (4, 6), "Bowler": (3, 5), "All-Rounder": (1, 3)}
        for role_name, (minimum, maximum) in targets.items():
            have = int(counts.get(role_name, 0))
            if have < minimum:
                needs.append(f"Needs more {role_name.lower()}s (currently {have}, target at least {minimum})")
            elif have > maximum:
                needs.append(f"High number of {role_name.lower()}s ({have}, suggested range {minimum}–{maximum})")
        if needs:
            for need in needs:
                st.warning(need)
        else:
            st.success("The squad's broad role counts are within the suggested balance ranges.")

        candidates = players[~players["player"].isin(squad)].copy()
        candidates["Team Fit Score"] = candidates.apply(lambda r: player_fit_score(r, squad), axis=1)
        candidates["Fit explanation"] = candidates.apply(lambda r: fit_explanation(r, squad), axis=1)
        role_filter = st.selectbox("Filter recommended players by role", ["All", "Batsman", "Bowler", "All-Rounder", "Unknown"], key="compat_role_filter")
        if role_filter != "All":
            candidates = candidates[candidates["role"] == role_filter]
        candidates = candidates.sort_values(["Team Fit Score", "overall_impact_score"], ascending=False)
        show = candidates.head(20).copy()
        show["Overall Impact"] = pd.to_numeric(show["overall_impact_score"], errors="coerce").round(2)
        show["Death-over Batting RR"] = pd.to_numeric(show["death_over_run_rate"], errors="coerce").round(2)
        show["Death-over Economy"] = pd.to_numeric(show["death_over_economy"], errors="coerce").round(2)
        st.subheader("Best-fit player recommendations")
        st.caption("Team Fit Score combines role balance, relative impact, and relevant death-over performance. It is a heuristic score out of 100, not a probability.")
        st.dataframe(show[["player", "role", "Team Fit Score", "Overall Impact", "Death-over Batting RR", "Death-over Economy", "Fit explanation"]].rename(columns={"player":"Player", "role":"Role"}), use_container_width=True, hide_index=True)
        st.download_button("Download compatibility recommendations CSV", show[["player", "role", "Team Fit Score", "Overall Impact", "Death-over Batting RR", "Death-over Economy", "Fit explanation"]].to_csv(index=False).encode("utf-8"), file_name="crickvision_player_compatibility.csv", mime="text/csv")

# ---------- Team Builder ----------
elif page == "Team Builder":
    st.title("👥 Team Builder")
    st.write("Build two Playing XIs and compare how well each player fits the team's current role balance.")

    if len(all_teams) < 2:
        st.error("Team encoders do not contain enough team names. Retrain the model from 04_train_win_model.py.")
        st.stop()
    c1, c2 = st.columns(2)
    # Restore widget values from persistent keys whenever returning to this page.
    st.session_state.setdefault("team_a_name_saved", all_teams[0])
    st.session_state.setdefault("team_b_name_saved", all_teams[1])
    st.session_state.setdefault("selected_a_saved", [])
    st.session_state.setdefault("selected_b_saved", [])
    if "team_a_widget" not in st.session_state:
        st.session_state["team_a_widget"] = st.session_state["team_a_name_saved"]
    if "team_b_widget" not in st.session_state:
        st.session_state["team_b_widget"] = st.session_state["team_b_name_saved"]
    if "selected_a_widget" not in st.session_state:
        st.session_state["selected_a_widget"] = list(st.session_state["selected_a_saved"])
    if "selected_b_widget" not in st.session_state:
        st.session_state["selected_b_widget"] = list(st.session_state["selected_b_saved"])
    with c1:
        default_a = st.session_state.get("team_a_name_saved", all_teams[0])
        team_a_name = st.selectbox("Team A", all_teams, index=all_teams.index(default_a) if default_a in all_teams else 0, key="team_a_widget", on_change=save_team_a_name)
        st.session_state["team_a_name_saved"] = team_a_name
        search_a = st.text_input("Search Team A players", key="search_a", placeholder="e.g. Bumrah")
        role_a = st.selectbox("Role filter — Team A", ["All", "Batsman", "Bowler", "All-Rounder"], key="role_a")
        pool_a = players.copy()
        if role_a != "All": pool_a = pool_a[pool_a.role == role_a]
        if search_a: pool_a = pool_a[pool_a.player.str.contains(search_a, case=False, na=False)]
        options_a = sorted(set(pool_a.player.tolist()) | set(st.session_state.get("selected_a_saved", [])) | set(st.session_state.get("selected_a_widget", [])))
        selected_a = st.multiselect("Select up to 11 players for Team A", options_a, max_selections=11, key="selected_a_widget", on_change=save_selected_a)
        st.session_state["selected_a_saved"] = list(selected_a)
        if selected_a:
            summary_a = players[players.player.isin(selected_a)][["player", "role", "overall_impact_score"]].sort_values("overall_impact_score", ascending=False).copy()
            summary_a.insert(0, "No.", range(1, len(summary_a)+1))
            st.dataframe(summary_a, hide_index=True, use_container_width=True)
        st.progress(len(selected_a)/11, text=f"{len(selected_a)}/11 selected")
    with c2:
        default_b = st.session_state.get("team_b_name_saved", all_teams[1])
        team_b_name = st.selectbox("Team B", all_teams, index=all_teams.index(default_b) if default_b in all_teams else 1, key="team_b_widget", on_change=save_team_b_name)
        st.session_state["team_b_name_saved"] = team_b_name
        search_b = st.text_input("Search Team B players", key="search_b", placeholder="e.g. Kohli")
        role_b = st.selectbox("Role filter — Team B", ["All", "Batsman", "Bowler", "All-Rounder"], key="role_b")
        pool_b = players.copy()
        if role_b != "All": pool_b = pool_b[pool_b.role == role_b]
        if search_b: pool_b = pool_b[pool_b.player.str.contains(search_b, case=False, na=False)]
        options_b = sorted(set(pool_b.player.tolist()) | set(st.session_state.get("selected_b_saved", [])) | set(st.session_state.get("selected_b_widget", [])))
        selected_b = st.multiselect("Select up to 11 players for Team B", options_b, max_selections=11, key="selected_b_widget", on_change=save_selected_b)
        st.session_state["selected_b_saved"] = list(selected_b)
        if selected_b:
            summary_b = players[players.player.isin(selected_b)][["player", "role", "overall_impact_score"]].sort_values("overall_impact_score", ascending=False).copy()
            summary_b.insert(0, "No.", range(1, len(summary_b)+1))
            st.dataframe(summary_b, hide_index=True, use_container_width=True)
        st.progress(len(selected_b)/11, text=f"{len(selected_b)}/11 selected")
    st.divider()
    # Team Builder is kept focused on selecting the two Playing XIs.
    # Compatibility analysis lives on the separate "Player Compatibility & Team Fit" page.
    if len(selected_a) == 11 and len(selected_b) == 11:
        a_strength = float(players[players.player.isin(selected_a)]["overall_impact_score"].mean())
        b_strength = float(players[players.player.isin(selected_b)]["overall_impact_score"].mean())
        x, y = st.columns(2)
        x.metric(f"{team_a_name} average impact", f"{a_strength:.2f}/10")
        y.metric(f"{team_b_name} average impact", f"{b_strength:.2f}/10")
        st.info("Your two XIs are saved in this Streamlit session. Open Win Prediction in the sidebar to predict the match outcome.")

# ---------- Win Prediction ----------
elif page == "Win Prediction":
    st.title("📊 IPL Win Prediction")
    st.write("Predict a match outcome separately from team selection, using the saved Playing XIs and match conditions.")
    selected_a = st.session_state.get("selected_a_saved", st.session_state.get("selected_a", []))
    selected_b = st.session_state.get("selected_b_saved", st.session_state.get("selected_b", []))
    team_a_name = st.session_state.get("team_a_name_saved", st.session_state.get("team_a_name", all_teams[0]))
    team_b_name = st.session_state.get("team_b_name_saved", st.session_state.get("team_b_name", all_teams[1]))
    overlapping_players = sorted(set(selected_a) & set(selected_b))
    if len(selected_a) != 11 or len(selected_b) != 11:
        st.warning("First open Team Builder, select exactly 11 players for each team, then return here.")
        st.write(f"Current selections: {team_a_name} {len(selected_a)}/11 · {team_b_name} {len(selected_b)}/11")
    elif team_a_name == team_b_name:
        st.warning("Choose two different team names in Team Builder before predicting.")
    elif overlapping_players:
        st.error("The same player cannot be in both Playing XIs. Return to Team Builder and replace: " + ", ".join(overlapping_players))
    else:
        st.subheader("Selected Playing XIs")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{team_a_name}**")
            st.write(", ".join(selected_a))
        with c2:
            st.markdown(f"**{team_b_name}**")
            st.write(", ".join(selected_b))
        a_strength = round(float(players[players.player.isin(selected_a)]["overall_impact_score"].mean()), 2)
        b_strength = round(float(players[players.player.isin(selected_b)]["overall_impact_score"].mean()), 2)
        m1, m2 = st.columns(2)
        m1.metric(f"{team_a_name} Team Strength", f"{a_strength:.2f}")
        m2.metric(f"{team_b_name} Team Strength", f"{b_strength:.2f}")
        venue = st.selectbox("Venue", all_venues, key="prediction_venue")
        toss_winner = st.selectbox("Toss winner", [team_a_name, team_b_name], key="prediction_toss_winner")
        toss_decision = st.selectbox("Toss decision", ["bat", "field"], key="prediction_toss_decision")
        if st.button("Predict Match Winner", type="primary"):
            X_live = pd.DataFrame([{
                "team1_strength": a_strength,
                "team2_strength": b_strength,
                "strength_diff": a_strength - b_strength,
                "toss_winner_enc": safe_encode(le_toss_winner, toss_winner),
                "toss_decision_enc": safe_encode(le_toss_decision, toss_decision),
                "venue_enc": safe_encode(le_venue, venue),
                "team1_enc": safe_encode(le_team1, team_a_name),
                "team2_enc": safe_encode(le_team2, team_b_name),
            }])
            proba = model.predict_proba(X_live)[0]
            classes = list(model.classes_)
            p_a = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
            p_b = float(proba[classes.index(0)]) if 0 in classes else 1.0-p_a
            p_a_pct, p_b_pct = round(p_a*100, 1), round(p_b*100, 1)
            st.subheader("Predicted Win Probability")
            st.bar_chart(pd.DataFrame({"Win Probability (%)": [p_a_pct, p_b_pct]}, index=[team_a_name, team_b_name]))
            winner = team_a_name if p_a_pct >= p_b_pct else team_b_name
            st.success(f"Model prediction: **{winner}** ({max(p_a_pct, p_b_pct):.1f}% estimated probability)")
            st.caption("This is a model estimate, not a guarantee. Its quality depends on training data and validation performance.")
