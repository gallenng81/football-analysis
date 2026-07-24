"""
Derives common betting markets from a Dixon-Coles scoreline probability
matrix — the same matrix `predict_score_matrix()` already produces.
Every market here is just a different way of summing cells in that matrix,
so they're all automatically consistent with each other (no separate model
needed per market, unlike sites that fit BTTS/O-U/1X2 independently).
"""
from __future__ import annotations
import pandas as pd


def btts_probability(matrix: pd.DataFrame) -> dict[str, float]:
    """Both teams to score - yes/no."""
    yes = 0.0
    for h in matrix.index:
        for a in matrix.columns:
            if h >= 1 and a >= 1:
                yes += matrix.loc[h, a]
    return {"yes": float(yes), "no": float(1 - yes)}


def over_under_probability(matrix: pd.DataFrame, line: float = 2.5) -> dict[str, float]:
    """Total goals over/under a line (e.g. 2.5, 3.5)."""
    over = 0.0
    for h in matrix.index:
        for a in matrix.columns:
            if h + a > line:
                over += matrix.loc[h, a]
    return {"over": float(over), "under": float(1 - over)}


def asian_handicap_probability(matrix: pd.DataFrame, home_handicap: float) -> dict[str, float]:
    """
    home_handicap e.g. -1.0 means home team must win by 2+ to cover.
    Whole/half lines only (no quarter-line splitting for simplicity).
    Adjusted home margin = (home_goals - away_goals) + home_handicap.
    """
    home_covers = 0.0
    push = 0.0
    for h in matrix.index:
        for a in matrix.columns:
            adj_margin = (h - a) + home_handicap
            if adj_margin > 0:
                home_covers += matrix.loc[h, a]
            elif adj_margin == 0:
                push += matrix.loc[h, a]
    away_covers = 1 - home_covers - push
    return {"home_covers": float(home_covers), "push": float(push), "away_covers": float(away_covers)}


def correct_score_probabilities(matrix: pd.DataFrame, top_n: int = 10) -> list[dict]:
    flat = matrix.stack().sort_values(ascending=False).head(top_n)
    return [{"score": f"{h}-{a}", "probability": float(p)} for (h, a), p in flat.items()]


def team_goals_over_under(matrix: pd.DataFrame, team: str, line: float = 1.5) -> dict[str, float]:
    """team: 'home' or 'away'. Probability that specific team scores over/under a line."""
    axis = matrix.index if team == "home" else matrix.columns
    over = 0.0
    for g in axis:
        row_or_col = matrix.loc[g, :] if team == "home" else matrix.loc[:, g]
        if g > line:
            over += row_or_col.sum()
    return {"over": float(over), "under": float(1 - over)}


def all_markets(matrix: pd.DataFrame) -> dict:
    """Convenience bundle of every market for a dashboard view."""
    return {
        "btts": btts_probability(matrix),
        "over_under_2.5": over_under_probability(matrix, 2.5),
        "over_under_1.5": over_under_probability(matrix, 1.5),
        "over_under_3.5": over_under_probability(matrix, 3.5),
        "asian_handicap_0": asian_handicap_probability(matrix, 0.0),
        "asian_handicap_-1": asian_handicap_probability(matrix, -1.0),
        "correct_scores": correct_score_probabilities(matrix, top_n=5),
    }
