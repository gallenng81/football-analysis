"""
Generates a synthetic season of match results so the model can be tested
end-to-end before you plug in real historical data from an API.

Replace this with real historical results (e.g. from API-Football or
football-data.co.uk) once you're ready to go live.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

teams = [
    "Arsenal", "Man City", "Liverpool", "Chelsea", "Man United",
    "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
]

# Ground-truth attack/defense strengths (hidden from the model — it has to
# recover something like these from goals data alone).
true_attack = {t: np.random.normal(1.3, 0.25) for t in teams}
true_defense = {t: np.random.normal(1.0, 0.2) for t in teams}
HOME_ADV = 1.35

rows = []
for round_num in range(1, 29):  # 28 rounds, each team plays each other ~twice
    shuffled = list(np.random.permutation(teams))
    for i in range(0, len(shuffled), 2):
        home, away = shuffled[i], shuffled[i + 1]
        lam_home = true_attack[home] / true_defense[away] * HOME_ADV
        lam_away = true_attack[away] / true_defense[home]
        home_goals = np.random.poisson(lam_home)
        away_goals = np.random.poisson(lam_away)
        rows.append({
            "date": f"2025-{(round_num % 12) + 1:02d}-{(round_num % 27) + 1:02d}",
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
        })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/football_predictor/data/sample_matches.csv", index=False)
print(f"Generated {len(df)} matches across {len(teams)} teams")
print(df.head())
