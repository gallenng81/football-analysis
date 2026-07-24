"""
Compares model-derived probabilities against bookmaker market prices, for
informational and educational purposes. This module does not place bets,
recommend wagers, or link to any bookmaker - it simply shows where a
statistical model's view of a match differs from the market's, which is
useful for understanding how odds are formed and how model-based
forecasts compare to market consensus.
"""
from __future__ import annotations


def implied_probability(decimal_odds: float) -> float:
    """Converts decimal odds (e.g. 2.50) to implied probability (0.40)."""
    return 1 / decimal_odds


def devig_market_probabilities(implied_probs: dict[str, float]) -> dict[str, float]:
    """
    Bookmaker odds always overround (sum of implied probabilities exceeds
    100%) - that excess is the bookmaker's built-in margin. This rescales
    so probabilities sum to 1, giving a fairer, margin-free market
    estimate to compare against the model.
    """
    total = sum(implied_probs.values())
    return {k: v / total for k, v in implied_probs.items()}


def find_probability_divergences(model_probs: dict[str, float], market_odds: dict[str, float],
                                  divergence_threshold: float = 0.03) -> list[dict]:
    """
    Identifies outcomes where the model's probability estimate differs
    meaningfully from the market's margin-free implied probability.
    This is presented purely as a statistical comparison - it is not a
    betting recommendation.

    model_probs: e.g. {'home_win': 0.45, 'draw': 0.25, 'away_win': 0.30}
    market_odds: decimal odds for the same keys, e.g. {'home_win': 2.10, 'draw': 3.40, 'away_win': 3.80}
    divergence_threshold: minimum (model_prob - market_fair_prob) to report

    Returns a list of divergences, sorted by size (largest first).
    """
    raw_implied = {k: implied_probability(v) for k, v in market_odds.items()}
    fair_implied = devig_market_probabilities(raw_implied)

    divergences = []
    for outcome, model_p in model_probs.items():
        fair_p = fair_implied[outcome]
        divergence = model_p - fair_p
        if divergence >= divergence_threshold:
            divergences.append({
                "outcome": outcome,
                "model_probability": round(model_p, 4),
                "market_odds": market_odds[outcome],
                "market_fair_probability": round(fair_p, 4),
                "divergence": round(divergence, 4),
                "model_implied_odds": round(1 / model_p, 2) if model_p > 0 else None,
            })
    return sorted(divergences, key=lambda x: x["divergence"], reverse=True)


if __name__ == "__main__":
    # Example: the model's view differs from the market's on the home side
    model_probs = {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}
    market_odds = {"home_win": 2.50, "draw": 3.40, "away_win": 3.20}

    divergences = find_probability_divergences(model_probs, market_odds)
    for d in divergences:
        print(f"{d['outcome']}: model {d['model_probability']:.1%} vs "
              f"market {d['market_fair_probability']:.1%} "
              f"(divergence {d['divergence']:+.1%}, market odds {d['market_odds']})")
