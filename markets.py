"""
Fetches this week's fixture odds from football-data.co.uk (free, no API
key) across whichever leagues you care about, and appends a timestamped
snapshot for each match to data/market_odds_history.csv.

Designed to be run automatically on a schedule (see
.github/workflows/track_odds.yml) so odds history builds up over time
without anyone needing to manually log snapshots. Safe to run repeatedly -
it just appends; market_tracker.py handles deduplication logic when
computing movement (it compares first vs latest, so extra snapshots in
between only add more resolution, not noise).

Usage:
    python fetch_market_odds.py --leagues E0 SP1 D1 I1 F1
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from football_data_uk import fetch_upcoming_fixtures

HISTORY_PATH = "data/market_odds_history.csv"


def track_odds(leagues: list[str], history_path: str = HISTORY_PATH) -> int:
    """Fetches current fixtures+odds for each league and appends new
    snapshot rows. Returns the number of match-odds rows appended."""
    timestamp = datetime.now(timezone.utc).isoformat()
    all_rows = []

    for league in leagues:
        try:
            fixtures = fetch_upcoming_fixtures(league)
        except Exception as e:
            print(f"Could not fetch fixtures for {league}: {e}")
            continue

        if fixtures.empty or "home_odds" not in fixtures.columns:
            continue

        for _, fx in fixtures.iterrows():
            if pd.isna(fx.get("home_odds")):
                continue
            all_rows.append({
                "timestamp": timestamp,
                "league": fx["league"],
                "home_team": fx["home_team"],
                "away_team": fx["away_team"],
                "date": fx["date"],
                "home_odds": fx["home_odds"],
                "draw_odds": fx["draw_odds"],
                "away_odds": fx["away_odds"],
            })

    if not all_rows:
        print("No fixture odds found for the requested leagues right now.")
        return 0

    new_rows = pd.DataFrame(all_rows)
    file_exists = os.path.exists(history_path)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    new_rows.to_csv(history_path, mode="a", header=not file_exists, index=False)
    print(f"Appended {len(new_rows)} snapshot rows to {history_path}")
    return len(new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot current market odds for tracked leagues")
    parser.add_argument("--leagues", nargs="+", default=["E0"],
                        help="football-data.co.uk league codes, e.g. E0 SP1 D1 I1 F1")
    args = parser.parse_args()

    track_odds(args.leagues)
