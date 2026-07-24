import pandas as pd
from dixon_coles import DixonColes

matches = pd.read_csv("data/sample_matches.csv")

model = DixonColes()
model.fit(matches)

print("=== Team ratings (higher attack = more goals scored, higher defense = fewer conceded) ===")
print(model.team_ratings().to_string(index=False))

home, away = "Arsenal", "Man City"
print(f"\n=== Prediction: {home} vs {away} ===")
outcome = model.predict_outcome_probs(home, away)
print(f"Home win: {outcome['home_win']:.1%}  Draw: {outcome['draw']:.1%}  Away win: {outcome['away_win']:.1%}")

print("\nMost likely scorelines:")
for score, prob in model.most_likely_scores(home, away, top_n=5):
    print(f"  {score}: {prob:.1%}")
