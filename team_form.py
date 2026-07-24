"""
Recent form analysis for teams: points per game, win/draw/loss record,
and scoring trends over their last N matches. This is standard sports
analytics content - the same kind of thing you'd see on a league table
website - and is purely descriptive of past results.
"""
from __future__ import annotations
import pandas as pd


def _match_outcome_for_team(row: pd.Series, team: str) -> tuple[str, int, int]:
    """Returns (result, goals_for, goals_against) from `team`'s perspective."""
    if row["home_team"] == team:
        gf, ga = row["home_goals"], row["away_goals"]
    else:
        gf, ga = row["away_goals"], row["home_goals"]

    if gf > ga:
        result = "W"
    elif gf < ga:
        result = "L"
    else:
        result = "D"
    return result, int(gf), int(ga)


def team_form(matches: pd.DataFrame, team: str, n: int = 5, date_col: str = "date") -> dict:
    """
    Computes recent form for one team over their last n matches (as
    either home or away). Returns points per game, W/D/L counts, average
    goals for/against, and a form string like 'WWDLW' (oldest to most
    recent, matching how form guides are usually displayed).
    """
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values(date_col)

    recent = team_matches.tail(n)
    if recent.empty:
        return {"team": team, "matches_played": 0}

    results, gf_list, ga_list = [], [], []
    for _, row in recent.iterrows():
        result, gf, ga = _match_outcome_for_team(row, team)
        results.append(result)
        gf_list.append(gf)
        ga_list.append(ga)

    points = sum(3 if r == "W" else 1 if r == "D" else 0 for r in results)
    wins = results.count("W")
    draws = results.count("D")
    losses = results.count("L")

    return {
        "team": team,
        "matches_played": len(recent),
        "form_string": "".join(results),
        "points": points,
        "points_per_game": round(points / len(recent), 2),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for_avg": round(sum(gf_list) / len(recent), 2),
        "goals_against_avg": round(sum(ga_list) / len(recent), 2),
        "goal_difference_avg": round((sum(gf_list) - sum(ga_list)) / len(recent), 2),
    }


def all_teams_form(matches: pd.DataFrame, n: int = 5, date_col: str = "date") -> pd.DataFrame:
    """Form table for every team in the dataset, sorted by points per
    game descending - the same shape as a 'last 5 games' form guide."""
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    rows = [team_form(matches, t, n=n, date_col=date_col) for t in teams]
    df = pd.DataFrame(rows)
    return df.sort_values("points_per_game", ascending=False).reset_index(drop=True)


def form_trend(matches: pd.DataFrame, team: str, recent_n: int = 5, prior_n: int = 5,
               date_col: str = "date") -> dict:
    """
    Compares a team's most recent n matches against the n matches before
    that, to describe whether form is improving, declining, or stable.
    Purely descriptive of historical results.
    """
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values(date_col)

    total_needed = recent_n + prior_n
    if len(team_matches) < total_needed:
        return {"team": team, "sufficient_data": False}

    prior_window = team_matches.iloc[-total_needed:-recent_n]
    recent_window = team_matches.iloc[-recent_n:]

    def ppg(window: pd.DataFrame) -> float:
        pts = 0
        for _, row in window.iterrows():
            result, _, _ = _match_outcome_for_team(row, team)
            pts += 3 if result == "W" else 1 if result == "D" else 0
        return pts / len(window)

    recent_ppg = ppg(recent_window)
    prior_ppg = ppg(prior_window)
    change = round(recent_ppg - prior_ppg, 2)

    if change > 0.3:
        trend = "improving"
    elif change < -0.3:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "team": team,
        "sufficient_data": True,
        "recent_points_per_game": round(recent_ppg, 2),
        "prior_points_per_game": round(prior_ppg, 2),
        "change": change,
        "trend": trend,
    }
