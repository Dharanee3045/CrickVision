# CrickVision
## IPL Player Analysis, Team Composition, Compatibility Assessment and Match Win Prediction

**Project type:** BCA Capstone Project  
**Student:** P. Dharaneedhar  
**Enrollment Number:** 24BCAHH010045  
**Programme:** BCA – 3rd Year  

---

## 1. Project Overview

CrickVision is a Python-based IPL analytics application that analyses player performance, supports team composition, provides player-fit recommendations and predicts a match outcome using a machine-learning model.

The project processes historical IPL match and ball-by-ball delivery data. It calculates batting and bowling statistics, death-over performance and player impact scores. These outputs are presented through an interactive Streamlit dashboard.

### Main objectives

- Analyse individual IPL player performance using batting and bowling statistics.
- Compare players using calculated impact scores.
- Search for a player and view their performance profile.
- Build two Playing XIs and compare their average player impact.
- Recommend possible player additions based on team balance and available performance indicators.
- Estimate the probability of Team 1 winning using a trained Random Forest classifier.

---

## 2. Technologies Used

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading, cleaning and analysing tabular data |
| NumPy | Numerical calculations and data processing |
| Scikit-learn | Random Forest model, label encoding, train/test split and evaluation |
| Joblib | Saving and loading the trained model and encoders |
| Streamlit | Interactive web dashboard |
| CSV | Storage format for match, delivery and player data |

---

## 3. Application Modules

The Streamlit dashboard contains the following modules. Select a module from the navigation menu in the sidebar.

### 3.1 Player Leaderboard

- Displays player rankings based on their Overall Impact Score.
- Supports role-based player analysis.
- Provides a quick comparison of player performance.
- Includes prototype auction-value estimates. These are heuristic estimates, not predictions from a model trained on historical auction prices.

### 3.2 Player Search

- Search for an individual player.
- View available player information and role.
- View batting statistics such as runs, batting average and strike rate.
- View bowling statistics such as wickets, economy and bowling average.
- Review death-over batting and bowling performance.
- View a base-price-based auction value scenario estimate. This feature is a prototype heuristic, not a trained auction-price prediction model.

### 3.3 Player Compatibility

- Analyse the balance of a selected squad.
- Identify possible player additions for a squad.
- Calculate a Team Fit Score using role balance, relative Overall Impact Score and relevant death-over performance indicators.
- Display a short explanation of the suggested fit.

**Important:** Compatibility scores are heuristic recommendations. They are not win probabilities, official franchise decisions or proof of actual chemistry between two players.

### 3.4 Team Builder

- Select two IPL teams.
- Search and filter available players by role.
- Select up to 11 players for each team.
- Compare the average Overall Impact Score of the selected Playing XIs.
- Save the selected XIs in the current Streamlit session for use in the Win Prediction module.

### 3.5 Win Prediction

- Uses the two Playing XIs selected in Team Builder.
- Takes team strength, venue, toss winner and toss decision into account.
- Uses the saved Random Forest classification model to estimate the probability of Team 1 winning.
- Displays the prediction and probability output in the dashboard.

The prediction is a machine-learning estimate based on the historical dataset and selected inputs. It should not be treated as a guarantee of an actual match result.

---

## 4. Project Folder Structure

Keep the Python scripts, CSV files and model files together in the same project folder.

```text
CrickVision/
│
├── .streamlit/
│   └── config.toml
│
├── 01_load_and_explore.py
├── 02_data_cleaning.py
├── 03_player_analysis.py
├── 04_train_win_model.py
├── 05_app.py
├── complete_player_names.py
│
├── matches.csv
├── deliveries.csv
├── matches_clean.csv
├── deliveries_clean.csv
├── player_scores.csv
├── player_bio.csv
├── player_name_map.csv
├── auction_history_template.csv
│
├── win_prediction_model.pkl
├── le_team1.pkl
├── le_team2.pkl
├── le_toss_decision.pkl
├── le_toss_winner.pkl
├── le_venue.pkl
│
├── requirements.txt
└── README.md
```

Some generated CSV and model files are created or refreshed when the preparation and training scripts are run.

### Important files

| File | Description |
|---|---|
| `01_load_and_explore.py` | Loads the raw match and delivery CSV files and prints basic information for inspection. |
| `02_data_cleaning.py` | Standardises columns and team names, handles selected missing values and creates cleaned CSV files. |
| `03_player_analysis.py` | Calculates player batting, bowling, death-over statistics, roles and impact scores. |
| `04_train_win_model.py` | Builds historical team-strength features, trains and evaluates the Random Forest model, and saves the model and encoders. |
| `05_app.py` | Runs the Streamlit dashboard and its application modules. |
| `complete_player_names.py` | Helper script for player-name completion/mapping. |
| `matches.csv` | Raw IPL match-level data. |
| `deliveries.csv` | Raw ball-by-ball delivery data. |
| `matches_clean.csv` | Cleaned match-level data. |
| `deliveries_clean.csv` | Cleaned delivery-level data. |
| `player_scores.csv` | Generated player statistics, roles and impact scores. |
| `player_bio.csv` | Available player biography information. |
| `player_name_map.csv` | Curated player display-name mapping. |
| `win_prediction_model.pkl` | Saved Random Forest win prediction model. |
| `le_*.pkl` | Saved label encoders used for categorical model inputs. |
| `requirements.txt` | Python package dependencies. |

---

## 5. System Requirements

- Windows, macOS or Linux computer.
- Python installed (Python 3.10 or later is recommended).
- Internet connection for installing Python packages.
- Sufficient storage for the project files and datasets.
- Command Prompt or terminal.

Check whether Python is installed:

```bash
python --version
```

On Windows, the Python launcher can also be checked with:

```cmd
py --version
```

---

## 6. Installation Steps

### Step 1: Download the project

Download and extract the CrickVision source-code ZIP file.

### Step 2: Open the project folder

Open the extracted `CrickVision` folder. The files `requirements.txt`, `05_app.py` and the CSV datasets should be available in this folder.

### Step 3: Open Command Prompt in the project folder

In File Explorer, open the project folder, click the address bar, type `cmd`, and press Enter. This opens Command Prompt in that folder.

### Step 4: (Recommended) Create a virtual environment

Windows:

```cmd
py -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 5: Install dependencies

Windows:

```cmd
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file lists the following packages:

- pandas
- numpy
- scikit-learn
- streamlit
- joblib
- Pillow

---

## 7. Execution Steps

Run the commands from inside the `CrickVision` project folder.

### Step 1: Inspect the raw datasets (optional)

```cmd
py 01_load_and_explore.py
```

This script loads `matches.csv` and `deliveries.csv` and displays their shapes, columns, sample records and missing-value counts.

### Step 2: Clean the datasets

```cmd
py 02_data_cleaning.py
```

This script creates:

- `matches_clean.csv`
- `deliveries_clean.csv`

Run this step when rebuilding the project from the raw CSV files or after changing those files.

### Step 3: Generate player statistics

```cmd
py 03_player_analysis.py
```

This script processes the cleaned delivery data and generates `player_scores.csv`, including player statistics, death-over metrics, role classifications and impact scores.

### Step 4: Train the match win prediction model

```cmd
py 04_train_win_model.py
```

This script trains the Random Forest classifier and saves:

- `win_prediction_model.pkl`
- `le_toss_winner.pkl`
- `le_toss_decision.pkl`
- `le_venue.pkl`
- `le_team1.pkl`
- `le_team2.pkl`

It also prints evaluation metrics such as accuracy, precision, recall, F1-score and a confusion matrix for the test split.

### Step 5: Start the Streamlit application

```cmd
py -m streamlit run 05_app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that address in a web browser to use CrickVision.

To stop the application, return to the terminal and press `Ctrl + C`.

### Quick start

If the cleaned data, player scores, model and encoder files are already present and up to date, the dashboard can be started directly:

```cmd
py -m streamlit run 05_app.py
```

If you have changed the raw datasets, follow the full preparation and training sequence before starting the dashboard.

---

## 8. Data Processing and Calculation Notes

### 8.1 Batting statistics

The player analysis script calculates batting runs, balls faced, fours, sixes, matches played, batting average and strike rate.

Common formulas:

**Batting Average**

```text
Batting Average = Total Runs / Times Dismissed
```

**Strike Rate**

```text
Strike Rate = (Total Runs / Balls Faced) × 100
```

**Death-over Run Rate**

```text
Death-over Run Rate = (6 × Death-over Runs) / Death-over Balls Faced
```

### 8.2 Bowling statistics

The script calculates balls bowled, runs conceded, wickets, overs bowled, economy and bowling average.

**Economy Rate**

```text
Economy Rate = (Runs Conceded × 6) / Legal Balls Bowled
```

**Bowling Average**

```text
Bowling Average = Runs Conceded / Wickets
```

The script excludes selected dismissal types that are not credited to the bowler when calculating bowler wickets. It also handles extras when calculating legal balls and runs conceded.

### 8.3 Player impact scores

The player analysis script normalises performance indicators to a 0–10 scale and combines them using weighted formulas.

**Batting Impact Score**

```text
Batting Impact =
    0.40 × Normalised Batting Average
  + 0.35 × Normalised Strike Rate
  + 0.25 × Normalised Death-over Runs
```

**Bowling Impact Score**

```text
Bowling Impact =
    0.45 × Normalised Wickets
  + 0.30 × Normalised Inverse Economy
  + 0.25 × Normalised Inverse Bowling Average
```

**Overall Impact Score for an All-Rounder**

```text
Overall Impact =
    0.55 × Batting Impact
  + 0.45 × Bowling Impact
```

For players classified as Batsmen or Bowlers, the corresponding batting or bowling impact score is used as their Overall Impact Score.

The role classification is heuristic and is based on participation and performance thresholds. A verified `player_roles.csv` file can optionally override roles for exact matching player names. Its required columns are `player` and `role`.

### 8.4 Team strength

The dashboard calculates team strength as the average Overall Impact Score of the selected players:

```text
Team Strength = Sum of Selected Players' Overall Impact Scores
                / Number of Selected Players
```

The historical model-training script derives the players who batted or bowled for each team in a match and calculates the corresponding average impact score.

### 8.5 Win prediction model

The project uses a Random Forest Classifier with:

- Number of trees (`n_estimators`): 300
- Maximum depth (`max_depth`): 8
- Random state: 42
- Train/test split: 80% training and 20% testing
- Stratified split to preserve class proportions

The model features are:

- Team 1 strength
- Team 2 strength
- Difference between team strengths
- Toss winner
- Toss decision
- Venue
- Team 1 identity
- Team 2 identity

The target label indicates whether Team 1 won the historical match.

---

## 9. Dataset Information

The project uses two main historical IPL datasets:

- `matches.csv` — match-level information, including teams, venue, toss information and match winner.
- `deliveries.csv` — ball-by-ball information, including batting team, bowling team, batter, bowler, runs, extras and dismissals.

The initial data preparation script documents the Kaggle IPL Complete Dataset (2008–2020) as the source for the raw match and delivery files. Keep the dataset files in the project folder and retain any applicable dataset attribution or licence information when redistributing them.

The project also uses player biography and player-name mapping CSV files where available.

---

## 10. Known Limitations

1. **Auction value:** The supplied project does not include a complete historical auction dataset containing player, season, base price, actual sold price and unsold status. Therefore, the displayed auction-value estimates are heuristic scenario estimates, not outputs of a trained or validated auction-price model.
2. **Player compatibility:** Compatibility and Team Fit Scores are heuristic recommendations based on role balance, relative impact and available death-over indicators. They are not probabilities of success or verified player-pair chemistry.
3. **Player roles:** Roles are inferred using performance and participation rules unless verified role overrides are supplied.
4. **Data coverage:** Statistics and predictions depend on the available historical dataset and its completeness.
5. **Win prediction:** A predicted probability is a model estimate, not a guarantee of the actual match result. Model performance should be assessed using the evaluation metrics printed during training.

---

## 11. Troubleshooting

### Problem: `py` or `python` is not recognised

Install Python and ensure the Python launcher or Python executable is added to the system PATH. Reopen Command Prompt and check the version again.

### Problem: `ModuleNotFoundError`

Activate the virtual environment, then install the dependencies again:

```cmd
py -m pip install -r requirements.txt
```

### Problem: `FileNotFoundError` for a CSV or model file

Check that Command Prompt is open in the `CrickVision` project folder and that the required files are present. Run the data preparation and model training scripts in the order described above.

### Problem: `player_scores.csv` is missing

Run:

```cmd
py 03_player_analysis.py
```

Make sure `deliveries_clean.csv` exists first.

### Problem: Model or encoder file is missing

Run:

```cmd
py 04_train_win_model.py
```

Make sure the cleaned datasets and `player_scores.csv` are available first.

### Problem: Streamlit command is not recognised

Run Streamlit through Python:

```cmd
py -m streamlit run 05_app.py
```

---

## 12. Future Enhancements

Possible future improvements include:

- Add a verified and regularly updated player-role dataset.
- Add complete historical IPL auction data and train and validate a dedicated auction-price model.
- Validate player-fit recommendations against historical team composition and match outcomes.
- Expand the dataset with newer IPL seasons.
- Evaluate model performance across different seasons and venues.
- Improve the dashboard with additional visualisations and comparison features.

---

## 13. Author

**P. Dharaneedhar**  
**Enrollment Number:** 24BCAHH010045  
**Programme:** BCA – 3rd Year  

**Project:** CrickVision – IPL Player Analysis, Team Composition, Compatibility Assessment and Match Win Prediction

---

## 14. Usage Note

This project is an academic capstone prototype developed for IPL data analysis and demonstration. Predictions and heuristic scores should be interpreted in the context of the available dataset and the limitations described in this document.
