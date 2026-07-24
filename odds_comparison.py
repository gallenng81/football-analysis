"""
Estimates how a key player's absence (injury, suspension) might shift a
team's attack/defense strength for prediction purposes.

This project has no player-level statistical data (only team-level goals
history), so this is a transparent heuristic: the user specifies which
player is missing and how important that player is to the team (as a
percentage of team output), and this scales the team's attack or defense
rating down accordingly before generating a match prediction. It is not a
learned model - it is a clearly-labeled adjustment tool, since real
injury-impact modeling would require player-level data this project
doesn't have.

For real injury data feeds, API-Football's /injuries endpoint (see
fetch_live_data.py) can supply who's out; this module still needs an
importance estimate per player, since "who is injured" and "how much
does losing them hurt" are different problems.
"""
from __future__ import annotations
from dixon_coles import DixonColes


def adjusted_team_strength(model: DixonColes, team: str, missing_player_impact_pct: float) -> dict:
    """
    missing_player_impact_pct: the missing player's estimated share of the
    team's attacking output, as a percentage (e.g. 25 for a striker who
    scores/creates roughly a quarter of the team's goal threat). This
    reduces the team's attack rating proportionally.

    This is a simple linear heuristic, not a fitted statistical
    relationship - treat the resulting prediction shift as illustrative,
    not precise.
    """
    if team not in model.attack:
        raise ValueError(f"Unknown team: {team}")

    original_attack = model.attack[team]
    impact_fraction = missing_player_impact_pct / 100
    adjusted_attack = original_attack - impact_fraction * abs(original_attack) if original_attack != 0 else original_attack * (1 - impact_fraction)

    return {
        "team": team,
        "original_attack_rating": round(original_attack, 3),
        "adjusted_attack_rating": round(adjusted_attack, 3),
        "impact_applied_pct": missing_player_impact_pct,
    }


def predict_with_injury_adjustment(model: DixonColes, home_team: str, away_team: str,
                                    home_missing_impact_pct: float = 0,
                                    away_missing_impact_pct: float = 0,
                                    max_goals: int = 6) -> dict:
    """
    Temporarily adjusts attack ratings for missing-player impact, predicts
    the match, then restores the original ratings. Returns both the
    baseline (unadjusted) and adjusted outcome probabilities so the user
    can see the size of the shift.
    """
    baseline = model.predict_outcome_probs(home_team, away_team, max_goals)

    original_home_attack = model.attack[home_team]
    original_away_attack = model.attack[away_team]

    try:
        if home_missing_impact_pct:
            model.attack[home_team] = adjusted_team_strength(
                model, home_team, home_missing_impact_pct)["adjusted_attack_rating"]
        if away_missing_impact_pct:
            model.attack[away_team] = adjusted_team_strength(
                model, away_team, away_missing_impact_pct)["adjusted_attack_rating"]

        adjusted = model.predict_outcome_probs(home_team, away_team, max_goals)
    finally:
        model.attack[home_team] = original_home_attack
        model.attack[away_team] = original_away_attack

    return {
        "baseline": baseline,
        "adjusted": adjusted,
        "shift": {k: round(adjusted[k] - baseline[k], 4) for k in baseline},
    }
