"""
A single interpretable "confidence score" (0-100) for a prediction,
combining two things:
1. How decisive the model's own probability distribution is (a 45/30/25
   split is far less confident than a 70/20/10 split, even before
   considering data quality).
2. How much historical data the model was fitted on for these two teams
   specifically (a rating built on 3 matches is far shakier than one
   built on 30).

This is a heuristic communication tool, not a statistically calibrated
metric - it exists to give users an at-a-glance read on how much weight
to put on a given prediction.
"""
from __future__ import annotations
import math
import pandas as pd


def _outcome_sharpness(probs: dict[str, float]) -> float:
    """
    Normalized entropy-based sharpness: 1.0 = fully certain (one outcome
    at 100%), 0.0 = maximally uncertain (all outcomes equally likely).
    """
    values = [p for p in probs.values() if p > 0]
    max_entropy = math.log(len(probs))
    entropy = -sum(p * math.log(p) for p in values)
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


def _sample_size_factor(matches: pd.DataFrame, home_team: str, away_team: str,
                        target_count: int = 20) -> float:
    """
    0-1 factor for how much history each team has in the fitted dataset,
    relative to a target count where we'd consider the fit well-supported.
    Uses the team with LESS data as the bottleneck.
    """
    home_count = len(matches[(matches["home_team"] == home_team) | (matches["away_team"] == home_team)])
    away_count = len(matches[(matches["home_team"] == away_team) | (matches["away_team"] == away_team)])
    min_count = min(home_count, away_count)
    return min(min_count / target_count, 1.0)


def confidence_score(model_probs: dict[str, float], matches: pd.DataFrame,
                      home_team: str, away_team: str) -> dict:
    """
    Returns a 0-100 confidence score plus its components, and a short
    plain-language label.
    """
    sharpness = _outcome_sharpness(model_probs)
    data_factor = _sample_size_factor(matches, home_team, away_team)

    # Weighted blend - data sufficiency matters more than raw sharpness,
    # since a sharp prediction from a thin sample is often overconfident.
    score = round((0.4 * sharpness + 0.6 * data_factor) * 100, 1)

    if score >= 75:
        label = "high confidence"
    elif score >= 50:
        label = "moderate confidence"
    elif score >= 25:
        label = "low confidence"
    else:
        label = "very low confidence - limited data"

    return {
        "score": score,
        "label": label,
        "outcome_sharpness": round(sharpness * 100, 1),
        "data_sufficiency": round(data_factor * 100, 1),
    }
