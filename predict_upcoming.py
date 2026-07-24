"""
The complete pipeline for predicting REAL upcoming matches. Run this on
your own machine (this sandbox can't reach the live APIs) once you have:

    export FOOTBALL_API_KEY=your_key   # from api-football.com
    export ODDS_API_KEY=your_key       # from the-odds-api.com

Usage:
    python predict_upcoming.py --league 39 --season 2025 --odds-key soccer_epl

League IDs (API-Football):  39=Premier League, 140=La Liga, 78=Bundesliga,
135=Serie A, 61=Ligue 1, 2=Champions League
Odds sport keys (The Odds API): soccer_epl, soccer_spain_la_liga,
soccer_germany_bundesliga, soccer_italy_serie_a, soccer_france_ligue_one

What it does:
1. Pulls recent real results for the league and fits the Dixon-Coles model
2. Pulls the next N scheduled fixtures
3. For each fixture, predicts scorelines, 1X2, BTTS, over/under
4. Pulls live bookmaker odds and fuzzy-matches team names across providers
5. Reports notable divergences between the model and the market, for informational purposes
6. Saves everything to predictions.csv
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd

from dixon_coles import DixonColes
from markets import all_markets
from odds_comparison import find_probability_divergences
from team_matching import match_fixture_to_model_teams
from fetch_live_data import fetch_recent_results, fetch_upcoming_fixtures, fetch_live_odds
from odds_analysis import generate_match_report, log_odds_snapshot


def run(league_id: int, season: int, odds_sport_key: str | None, next_n: int = 20,
        min_history: int = 100):
    print(f"Fetching recent results for league {league_id}, season {season}...")
    results = fetch_recent_results(league_id, season, last_n=max(min_history, 150))
    if len(results) < 30:
        print(f"Only got {len(results)} historical matches - check your API key, "
              f"league_id, and season. The model needs a reasonable sample to fit on.",
              file=sys.stderr)
        return None

    print(f"Fitting model on {len(results)} matches...")
    model = DixonColes().fit(results)

    print(f"Fetching next {next_n} upcoming fixtures...")
    fixtures = fetch_upcoming_fixtures(league_id, season, next_n=next_n)
    if fixtures.empty:
        print("No upcoming fixtures found for this league/season.", file=sys.stderr)
        return None

    odds_df = pd.DataFrame()
    if odds_sport_key:
        print(f"Fetching live odds ({odds_sport_key})...")
        try:
            odds_df = fetch_live_odds(odds_sport_key)
        except Exception as e:
            print(f"Could not fetch live odds ({e}) - continuing with predictions only.", file=sys.stderr)

    rows = []
    for _, fx in fixtures.iterrows():
        home_raw, away_raw = fx["home_team"], fx["away_team"]
        home, away = match_fixture_to_model_teams(home_raw, away_raw, model.teams)

        if home is None or away is None:
            print(f"Skipping {home_raw} vs {away_raw} - team not found in fitted history "
                  f"(newly promoted team, or name mismatch).")
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

        # Match against live odds if we have them
        if not odds_df.empty:
            fixture_odds = odds_df[(odds_df["home_team"] == home_raw) & (odds_df["away_team"] == away_raw)]
            if not fixture_odds.empty:
                best = fixture_odds.iloc[0]  # first bookmaker found; extend to best-price shopping if desired
                bookmaker_odds = {
                    "home_win": best["home_odds"],
                    "draw": best["draw_odds"],
                    "away_win": best["away_odds"],
                }
                if all(pd.notna(v) for v in bookmaker_odds.values()):
                    divergences = find_probability_divergences(outcome, bookmaker_odds, divergence_threshold=0.03)
                    row["bookmaker"] = best["bookmaker"]
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
    predictions.to_csv("predictions.csv", index=False)
    print(f"\nSaved {len(predictions)} predictions to predictions.csv")
    print(predictions.to_string(index=False))
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real upcoming football matches")
    parser.add_argument("--league", type=int, required=True, help="API-Football league ID, e.g. 39 for Premier League")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--odds-key", type=str, default=None, help="The Odds API sport key, e.g. soccer_epl")
    parser.add_argument("--next", type=int, default=20, help="Number of upcoming fixtures to predict")
    args = parser.parse_args()

    run(args.league, args.season, args.odds_key, next_n=args.next)
