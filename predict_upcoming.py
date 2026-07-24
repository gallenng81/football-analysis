"""
Tracks odds movement across MANY matches at once, not just a single
match you look up manually. Reads a history file that's built up over
time (see fetch_market_odds.py for how it gets populated automatically),
and reports which matches have moved the most since first being tracked.
"""
from __future__ import annotations
import os
import pandas as pd


MATCH_KEY_COLS = ["league", "home_team", "away_team", "date"]


def load_market_history(log_path: str = "data/market_odds_history.csv") -> pd.DataFrame:
    if not os.path.exists(log_path):
        return pd.DataFrame()
    return pd.read_csv(log_path)


def market_movements(log_path: str = "data/market_odds_history.csv",
                      min_snapshots: int = 2) -> pd.DataFrame:
    """
    For every match with at least `min_snapshots` logged odds snapshots,
    compares the first snapshot against the latest and reports the
    movement on each outcome. Sorted by total movement (largest first),
    so the biggest market shifts surface at the top.
    """
    df = load_market_history(log_path)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for key, group in df.groupby(MATCH_KEY_COLS):
        if len(group) < min_snapshots:
            continue
        group = group.sort_values("timestamp")
        first, last = group.iloc[0], group.iloc[-1]
        league, home, away, match_date = key

        home_change = round(last["home_odds"] - first["home_odds"], 3)
        draw_change = round(last["draw_odds"] - first["draw_odds"], 3)
        away_change = round(last["away_odds"] - first["away_odds"], 3)

        rows.append({
            "league": league,
            "home_team": home,
            "away_team": away,
            "match_date": match_date,
            "snapshots": len(group),
            "first_seen": first["timestamp"],
            "last_seen": last["timestamp"],
            "home_odds_now": last["home_odds"],
            "home_change": home_change,
            "draw_odds_now": last["draw_odds"],
            "draw_change": draw_change,
            "away_odds_now": last["away_odds"],
            "away_change": away_change,
            "total_movement": round(abs(home_change) + abs(draw_change) + abs(away_change), 3),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("total_movement", ascending=False).reset_index(drop=True)


def biggest_movers(log_path: str = "data/market_odds_history.csv", top_n: int = 10) -> pd.DataFrame:
    """Convenience wrapper - just the top N matches by movement size."""
    movements = market_movements(log_path)
    return movements.head(top_n)


def steam_moves(log_path: str = "data/market_odds_history.csv",
                threshold: float = 0.15) -> pd.DataFrame:
    """
    Flags matches where an outcome's odds have shortened by more than
    `threshold` (in decimal odds) - a common informal signal that the
    market has grown notably more confident in that outcome. Purely
    descriptive of market data, not a betting recommendation.
    """
    movements = market_movements(log_path)
    if movements.empty:
        return movements

    def outcome_steaming(row):
        candidates = []
        if row["home_change"] <= -threshold:
            candidates.append(("home", row["home_team"], row["home_change"]))
        if row["draw_change"] <= -threshold:
            candidates.append(("draw", "Draw", row["draw_change"]))
        if row["away_change"] <= -threshold:
            candidates.append(("away", row["away_team"], row["away_change"]))
        return candidates

    flagged_rows = []
    for _, row in movements.iterrows():
        for outcome, label, change in outcome_steaming(row):
            flagged_rows.append({
                "league": row["league"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "match_date": row["match_date"],
                "outcome": label,
                "odds_change": change,
                "snapshots": row["snapshots"],
            })

    return pd.DataFrame(flagged_rows).sort_values("odds_change").reset_index(drop=True)
