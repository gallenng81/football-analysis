"""
Defines Free vs Premium feature tiers and gates access accordingly.

IMPORTANT - what this module does and doesn't do:
This module defines WHICH features belong to which tier and provides a
clean `is_premium(user)` check to gate them in the UI. It does NOT
process any actual payment or verify a real subscription - there is no
payment processor wired in here.

To make this real, you'd connect a payment provider (Stripe is the
standard choice for subscription billing) roughly like this:
1. Stripe Checkout or Payment Links to collect the monthly charge
2. A webhook endpoint (needs a real backend - Streamlit alone can't
   receive webhooks) that listens for `checkout.session.completed` and
   `customer.subscription.deleted` events
3. Store each user's subscription status (e.g. in a database keyed by
   user ID or email) and update it from the webhook events
4. Replace the `is_premium()` function below with a lookup against that
   stored status

None of that billing infrastructure exists in this project. The toggle
in the dashboard is a placeholder so you can see how the tiers behave -
swap it out for a real subscription check once billing is wired up.
"""
from __future__ import annotations

PRICING = {
    "free": {"price_sgd": 0, "label": "Free"},
    "premium": {"price_sgd_min": 9.90, "price_sgd_max": 29.90, "label": "Premium"},
}

FREE_FEATURES = [
    "match_predictions",
    "team_form",
    "team_ratings",
]

PREMIUM_FEATURES = [
    "confidence_score",
    "odds_movement_analysis",
    "injury_impact",
    "ai_explanation",
    "notable_divergence_alerts",
    "historical_performance_dashboard",
]

ALL_FEATURES = FREE_FEATURES + PREMIUM_FEATURES

FEATURE_LABELS = {
    "match_predictions": "AI match predictions",
    "team_form": "Team form",
    "team_ratings": "Team ratings / league table",
    "confidence_score": "Confidence score",
    "odds_movement_analysis": "Odds movement analysis",
    "injury_impact": "Player injury impact",
    "ai_explanation": "AI explanation",
    "notable_divergence_alerts": "Notable match alerts",
    "historical_performance_dashboard": "Historical performance dashboard",
}


def is_premium(user_tier: str) -> bool:
    """user_tier is expected to be 'free' or 'premium'. Replace the
    caller of this function with a real subscription lookup once payment
    processing is wired up - see module docstring."""
    return user_tier == "premium"


def has_access(user_tier: str, feature: str) -> bool:
    if feature in FREE_FEATURES:
        return True
    if feature in PREMIUM_FEATURES:
        return is_premium(user_tier)
    raise ValueError(f"Unknown feature: {feature}")


def feature_list_for_tier(user_tier: str) -> dict[str, bool]:
    return {f: has_access(user_tier, f) for f in ALL_FEATURES}
