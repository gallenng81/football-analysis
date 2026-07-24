"""
Walk-forward backtest: refits the model periodically using only data
available up to that point in time, predicts each upcoming match BEFORE
seeing its result, then logs predicted probabilities against what actually
happened. This is the honest way to check whether the model has any real
skill - never evaluate on matches the model was fitted on.

Produces a ledger (like the public accuracy trackers on sites such as
MyGameOdds) plus summary metrics: Brier score, log loss, and accuracy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dixon_coles import DixonColes


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    elif home_goals < away_goals:
        return "away_win"
    return "draw"


def brier_score(predicted_probs: dict[str, float], actual: str) -> float:
    """Multi-class Brier score: mean squared error between predicted probs
    and the one-hot actual outcome. 0 = perfect, ~0.67 = random guessing
    on 3 outcomes, 2.0 = worst possible."""
    outcomes = ["home_win", "draw", "away_win"]
    return sum((predicted_probs[o] - (1.0 if o == actual else 0.0)) ** 2 for o in outcomes)


def log_loss(predicted_probs: dict[str, float], actual: str, eps: float = 1e-10) -> float:
    p = max(min(predicted_probs[actual], 1 - eps), eps)
    return -np.log(p)


def rolling_backtest(
    matches: pd.DataFrame,
    min_train_matches: int = 60,
    refit_every: int = 10,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    matches: full history, sorted or not (will be sorted by date here).
    min_train_matches: how much history before the model starts making
        predictions (early matches are just used to bootstrap fitting).
    refit_every: how many matches to predict before refitting the model
        on the newly available data (refitting every single match is
        accurate but slow; refitting periodically is the practical
        approach real prediction services use too).

    Returns a DataFrame log: one row per predicted match with model
    probabilities, the actual result, and per-match Brier/log-loss scores.
    """
    df = matches.sort_values(date_col).reset_index(drop=True)
    log_rows = []
    model = None
    matches_since_fit = refit_every  # force a fit on the first eligible match

    for i in range(min_train_matches, len(df)):
        train = df.iloc[:i]
        test_row = df.iloc[i]

        if model is None or matches_since_fit >= refit_every:
            model = DixonColes().fit(train, date_col=date_col)
            matches_since_fit = 0
        matches_since_fit += 1

        home, away = test_row["home_team"], test_row["away_team"]
        if home not in model.attack or away not in model.attack:
            continue  # new team with no history yet - skip

        pred = model.predict_outcome_probs(home, away)
        actual = _actual_outcome(test_row["home_goals"], test_row["away_goals"])

        log_rows.append({
            "date": test_row[date_col],
            "home_team": home,
            "away_team": away,
            "home_goals": test_row["home_goals"],
            "away_goals": test_row["away_goals"],
            "pred_home_win": pred["home_win"],
            "pred_draw": pred["draw"],
            "pred_away_win": pred["away_win"],
            "predicted_favorite": max(pred, key=pred.get),
            "actual": actual,
            "correct_favorite": max(pred, key=pred.get) == actual,
            "brier_score": brier_score(pred, actual),
            "log_loss": log_loss(pred, actual),
        })

    return pd.DataFrame(log_rows)


def summarize_backtest(log: pd.DataFrame) -> dict:
    """Headline metrics + a naive baseline (always predict home win, since
    home advantage alone beats random guessing) so you have something to
    judge the model against."""
    if log.empty:
        return {"error": "No predictions logged - check min_train_matches vs data size."}

    n = len(log)
    naive_baseline = (log["actual"] == "home_win").mean()  # accuracy of "always pick home"

    return {
        "matches_evaluated": n,
        "accuracy": float(log["correct_favorite"].mean()),
        "naive_always_home_accuracy": float(naive_baseline),
        "mean_brier_score": float(log["brier_score"].mean()),
        "mean_log_loss": float(log["log_loss"].mean()),
        "outcome_distribution": log["actual"].value_counts(normalize=True).to_dict(),
    }


if __name__ == "__main__":
    matches = pd.read_csv("data/sample_matches.csv")
    log = rolling_backtest(matches, min_train_matches=60, refit_every=10)
    log.to_csv("data/backtest_log.csv", index=False)

    summary = summarize_backtest(log)
    print("=== Backtest summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
