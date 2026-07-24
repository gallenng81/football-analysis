"""
Turns raw model probabilities + bookmaker odds into a plain-language
analysis a user can actually read and act on - not just numbers, but an
explanation of what the odds say, where the model disagrees, and how
confident that disagreement is.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone

import pandas as pd

from odds_comparison import implied_probability, devig_market_probabilities


def market_overround(bookmaker_odds: dict[str, float]) -> float:
    """The bookmaker's built-in margin, e.g. 0.06 means the odds imply
    106% total probability - that extra 6% is the house edge."""
    raw_implied = sum(implied_probability(o) for o in bookmaker_odds.values())
    return raw_implied - 1.0


def compare_model_market(model_probs: dict[str, float], bookmaker_odds: dict[str, float]) -> pd.DataFrame:
    """Side-by-side table: model probability, market's fair (de-vigged)
    probability, the odds themselves, and the gap between model and market."""
    raw_implied = {k: implied_probability(v) for k, v in bookmaker_odds.items()}
    fair_implied = devig_market_probabilities(raw_implied)

    rows = []
    for outcome in model_probs:
        model_p = model_probs[outcome]
        fair_p = fair_implied[outcome]
        rows.append({
            "outcome": outcome,
            "model_probability": round(model_p, 4),
            "market_fair_probability": round(fair_p, 4),
            "bookmaker_odds": bookmaker_odds[outcome],
            "gap": round(model_p - fair_p, 4),
        })
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def confidence_label(gap: float) -> str:
    """Translates a raw probability gap into a plain-English confidence read."""
    abs_gap = abs(gap)
    if abs_gap < 0.02:
        return "negligible - model and market essentially agree"
    elif abs_gap < 0.05:
        return "slight disagreement - within normal model noise"
    elif abs_gap < 0.10:
        return "moderate disagreement - worth a second look"
    else:
        return "large disagreement - unusual, double-check team news/injuries before trusting this"


def generate_match_report(home_team: str, away_team: str,
                           model_probs: dict[str, float],
                           bookmaker_odds: dict[str, float],
                           top_scorelines: list[tuple[str, float]] | None = None,
                           extra_markets: dict | None = None) -> str:
    """Produces a readable text report for one match, combining the
    model's view, the market's view, and where/how much they disagree."""
    comparison = compare_model_market(model_probs, bookmaker_odds)
    overround = market_overround(bookmaker_odds)

    model_favorite = max(model_probs, key=model_probs.get)
    market_fair = devig_market_probabilities({k: implied_probability(v) for k, v in bookmaker_odds.items()})
    market_favorite = max(market_fair, key=market_fair.get)

    label = {"home_win": home_team, "draw": "a draw", "away_win": away_team}

    lines = []
    lines.append(f"{home_team} vs {away_team}")
    lines.append("=" * len(f"{home_team} vs {away_team}"))
    lines.append("")
    lines.append(f"Model's favorite: {label[model_favorite]} ({model_probs[model_favorite]:.1%})")
    lines.append(f"Market's favorite: {label[market_favorite]} ({market_fair[market_favorite]:.1%}, after removing the bookmaker's margin)")
    lines.append(f"Bookmaker margin on this match: {overround:.1%} (the house edge built into these odds)")
    lines.append("")

    if model_favorite == market_favorite:
        lines.append("Model and market agree on the favorite. Any edge here is about magnitude, not direction.")
    else:
        lines.append(f"Model and market DISAGREE on the favorite - the model favors "
                      f"{label[model_favorite]} while the market favors {label[market_favorite]}. "
                      f"This is the more interesting (and riskier) kind of signal.")
    lines.append("")

    lines.append("Outcome-by-outcome breakdown:")
    for _, row in comparison.iterrows():
        outcome_label = label[row["outcome"]]
        lines.append(
            f"  {outcome_label}: model {row['model_probability']:.1%} vs market {row['market_fair_probability']:.1%} "
            f"(gap {row['gap']:+.1%}, odds {row['bookmaker_odds']}) - {confidence_label(row['gap'])}"
        )
    lines.append("")

    if top_scorelines:
        lines.append("Most likely scorelines (model):")
        for score, prob in top_scorelines:
            lines.append(f"  {score}: {prob:.1%}")
        lines.append("")

    if extra_markets:
        lines.append("Other markets (model view):")
        if "btts" in extra_markets:
            lines.append(f"  Both teams to score: yes {extra_markets['btts']['yes']:.1%} / no {extra_markets['btts']['no']:.1%}")
        if "over_under_2.5" in extra_markets:
            ou = extra_markets["over_under_2.5"]
            lines.append(f"  Over/under 2.5 goals: over {ou['over']:.1%} / under {ou['under']:.1%}")
        lines.append("")

    best_gap_row = comparison.iloc[0]
    if best_gap_row["gap"] > 0.03:
        lines.append(
            f"Bottom line: the model's strongest disagreement with the market is on "
            f"{label[best_gap_row['outcome']]} ({best_gap_row['gap']:+.1%}) - "
            f"{confidence_label(best_gap_row['gap'])}."
        )
    else:
        lines.append("Bottom line: no strong disagreements here - the market's pricing looks efficient for this match.")

    return "\n".join(lines)


# --- Odds movement tracking (works even with periodic, non-live polling) ---

def log_odds_snapshot(home_team: str, away_team: str, bookmaker_odds: dict[str, float],
                       log_path: str = "data/odds_history.csv") -> None:
    """Appends a timestamped odds snapshot. Run this each time you check a
    match and you build up a movement history even without a live feed."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": bookmaker_odds.get("home_win"),
        "draw_odds": bookmaker_odds.get("draw"),
        "away_odds": bookmaker_odds.get("away_win"),
    }
    file_exists = os.path.exists(log_path)
    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", header=not file_exists, index=False)


def odds_movement(home_team: str, away_team: str,
                   log_path: str = "data/odds_history.csv") -> dict | None:
    """Compares the current snapshot against the earliest one logged for
    this match, to show which way the market has moved (a 'steam move'
    toward a team usually means informed money is backing them)."""
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    match_history = df[(df["home_team"] == home_team) & (df["away_team"] == away_team)]
    if len(match_history) < 2:
        return None

    match_history = match_history.sort_values("timestamp")
    first, last = match_history.iloc[0], match_history.iloc[-1]

    return {
        "snapshots_logged": len(match_history),
        "first_seen": first["timestamp"],
        "last_seen": last["timestamp"],
        "home_odds_change": round(last["home_odds"] - first["home_odds"], 3),
        "draw_odds_change": round(last["draw_odds"] - first["draw_odds"], 3),
        "away_odds_change": round(last["away_odds"] - first["away_odds"], 3),
    }


def describe_movement(movement: dict, home_team: str, away_team: str) -> str:
    """Plain-language read of odds movement. Shortening odds (going down)
    means the market is growing more confident in that outcome."""
    if movement is None:
        return "No odds history yet for this match - movement analysis needs at least two snapshots over time."

    def direction(change: float, label: str) -> str:
        if abs(change) < 0.02:
            return f"{label} odds barely moved ({change:+.2f})"
        elif change < 0:
            return f"{label} odds shortened ({change:+.2f}) - market growing more confident"
        else:
            return f"{label} odds drifted out ({change:+.2f}) - market growing less confident"

    lines = [
        f"Tracked across {movement['snapshots_logged']} snapshots, "
        f"{movement['first_seen']} to {movement['last_seen']}:",
        f"  {direction(movement['home_odds_change'], home_team)}",
        f"  {direction(movement['draw_odds_change'], 'Draw')}",
        f"  {direction(movement['away_odds_change'], away_team)}",
    ]
    return "\n".join(lines)
