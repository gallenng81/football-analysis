"""
Complete pipeline for predicting real upcoming matches using ONLY
football-data.co.uk - no API key, no signup, no rate limits to worry
about. Trade-off vs predict_upcoming.py (the API-Football/Odds-API
version): fewer leagues (~30 top divisions rather than 1200+), and odds
are a once-per-round snapshot rather than continuously live.

Usage:
    python predict_upcoming_free.py --league E0 --seasons 2022 2023 2024

League codes: E0=Premier League, E1=Championship, SP1=La Liga,
D1=Bundesliga, I1=Serie A, F1=Ligue 1, N1=Eredivisie, P1=Primeira Liga.
Full list: https://www.football-data.co.uk/notes.txt

What it does:
1. Downloads several recent seasons of real results for the league
2. Fits the Dixon-Coles model on them
3. Downloads this week's upcoming fixtures (with pre-match odds bundled in)
4. Predicts scorelines, 1X2, BTTS, over/under for each fixture
5. Reports notable divergences using the bundled odds, for informational purposes
6. Saves everything to predictions_free.csv
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd

from dixon_coles import DixonColes
from markets import all_markets
from odds_comparison import find_probability_divergences
from team_matching import match_fixture_to_model_teams
from football_data_uk import fetch_multi_season_results, fetch_upcoming_fixtures
from odds_analysis import generate_match_report, log_odds_snapshot


def run(league_code: str, seasons: list[int]):
    print(f"Downloading {league_code} results for seasons {seasons}...")
    results = fetch_multi_season_results(league_code, seasons)
    if len(results) < 30:
        print(f"Only got {len(results)} matches - check the league code is valid "
              f"and that football-data.co.uk has data for these seasons.", file=sys.stderr)
        return None

    print(f"Fitting model on {len(results)} matches...")
    model = DixonColes().fit(results)

    print("Downloading this week's fixtures...")
    fixtures = fetch_upcoming_fixtures(league_code)
    if fixtures.empty:
        print("No upcoming fixtures found - football-data.co.uk updates fixtures.csv "
              "on Friday and Tuesday afternoons (UK time), so check back then.", file=sys.stderr)
        return None

    has_odds = "home_odds" in fixtures.columns
    rows = []
    for _, fx in fixtures.iterrows():
        home_raw, away_raw = fx["home_team"], fx["away_team"]
        home, away = match_fixture_to_model_teams(home_raw, away_raw, model.teams)

        if home is None or away is None:
            print(f"Skipping {home_raw} vs {away_raw} - team not found in fitted history.")
            continue

        matrix = model.predict_score_matrix(home, away, max_goals=6)
        outcome = model.predict_outcome_probs(home, away)
        top_scores = model.most_likely_scores(home, away, top_n=3)
        markets = all_markets(matrix)

        row = {
            "date": fx["date"],
            "home_team": home_raw,
            "away_team": away_raw,
            "home_win_prob": round(outcome["home_win"], 4),
            "draw_prob": round(outcome["draw"], 4),
            "away_win_prob": round(outcome["away_win"], 4),
            "top_scoreline": top_scores[0][0],
            "top_scoreline_prob": round(top_scores[0][1], 4),
            "btts_yes_prob": round(markets["btts"]["yes"], 4),
            "over_2.5_prob": round(markets["over_under_2.5"]["over"], 4),
        }

        if has_odds and pd.notna(fx.get("home_odds")):
            bookmaker_odds = {
                "home_win": fx["home_odds"],
                "draw": fx["draw_odds"],
                "away_win": fx["away_odds"],
            }
            divergences = find_probability_divergences(outcome, bookmaker_odds, divergence_threshold=0.03)
            row["home_odds"] = bookmaker_odds["home_win"]
            row["draw_odds"] = bookmaker_odds["draw"]
            row["away_odds"] = bookmaker_odds["away_win"]
            row["notable_divergence"] = divergences[0]["outcome"] if divergences else None
            row["divergence_size"] = divergences[0]["divergence"] if divergences else None

            log_odds_snapshot(home_raw, away_raw, bookmaker_odds)
            print("\n" + generate_match_report(
                home_raw, away_raw, outcome, bookmaker_odds, top_scores, markets
            ) + "\n")

        rows.append(row)

    predictions = pd.DataFrame(rows)
    predictions.to_csv("predictions_free.csv", index=False)
    print(f"\nSaved {len(predictions)} predictions to predictions_free.csv")
    print(predictions.to_string(index=False))
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real upcoming matches - no API key needed")
    parser.add_argument("--league", type=str, required=True, help="football-data.co.uk league code, e.g. E0")
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                         help="Season start years to train on, e.g. --seasons 2022 2023 2024")
    args = parser.parse_args()

    run(args.league, args.seasons)
