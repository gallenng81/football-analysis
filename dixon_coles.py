"""
Generates a plain-language explanation of a prediction: which factors
(attack/defense ratings, home advantage, recent form) are driving the
model's view, in terms a non-technical user can follow.

This is a template-driven explainer built from the model's own numbers -
it does not call an external language model, so it works without any
extra API key. If richer, more varied prose is wanted later, this is the
natural place to swap in a call to an LLM API, passing these same
structured numbers as context so the explanation stays grounded in the
actual model output rather than inventing reasons.
"""
from __future__ import annotations
from dixon_coles import DixonColes
from team_form import team_form
import pandas as pd


def explain_prediction(model: DixonColes, matches: pd.DataFrame,
                        home_team: str, away_team: str, n_form: int = 5) -> str:
    outcome = model.predict_outcome_probs(home_team, away_team)
    favorite = max(outcome, key=outcome.get)

    home_attack, home_defense = model.attack[home_team], model.defense[home_team]
    away_attack, away_defense = model.attack[away_team], model.defense[away_team]

    home_form = team_form(matches, home_team, n=n_form)
    away_form = team_form(matches, away_team, n=n_form)

    lines = []

    # Overall verdict
    label = {"home_win": f"{home_team} winning", "draw": "a draw", "away_win": f"{away_team} winning"}
    lines.append(f"The model leans toward {label[favorite]} ({outcome[favorite]:.1%}), driven mainly by:")
    lines.append("")

    # Attack/defense comparison
    if home_attack > away_attack:
        lines.append(f"- Attack: {home_team}'s attack rating ({home_attack:.2f}) is stronger than "
                      f"{away_team}'s ({away_attack:.2f}), suggesting more scoring threat.")
    elif away_attack > home_attack:
        lines.append(f"- Attack: {away_team}'s attack rating ({away_attack:.2f}) is stronger than "
                      f"{home_team}'s ({home_attack:.2f}), suggesting more scoring threat despite playing away.")
    else:
        lines.append(f"- Attack: both teams have similar attack ratings ({home_attack:.2f} vs {away_attack:.2f}).")

    if home_defense > away_defense:
        lines.append(f"- Defense: {home_team}'s defense rating ({home_defense:.2f}) is stronger, "
                      f"suggesting they concede fewer chances than {away_team} ({away_defense:.2f}).")
    elif away_defense > home_defense:
        lines.append(f"- Defense: {away_team}'s defense rating ({away_defense:.2f}) is stronger, "
                      f"suggesting they concede fewer chances than {home_team} ({home_defense:.2f}).")
    else:
        lines.append(f"- Defense: both teams have similar defense ratings ({home_defense:.2f} vs {away_defense:.2f}).")

    # Home advantage
    lines.append(f"- Home advantage: the model applies a standard home-boost "
                  f"(fitted value {model.home_adv:.2f}) that favors {home_team} to some degree, "
                  f"as is typical in football at all levels.")

    # Recent form
    if home_form.get("matches_played") and away_form.get("matches_played"):
        lines.append("")
        lines.append(f"Recent form (last {n_form} matches):")
        lines.append(f"- {home_team}: {home_form['form_string']} "
                      f"({home_form['points_per_game']} points/game, "
                      f"{home_form['goals_for_avg']} scored / {home_form['goals_against_avg']} conceded per game)")
        lines.append(f"- {away_team}: {away_form['form_string']} "
                      f"({away_form['points_per_game']} points/game, "
                      f"{away_form['goals_for_avg']} scored / {away_form['goals_against_avg']} conceded per game)")

        if home_form["points_per_game"] > away_form["points_per_game"] + 0.5:
            lines.append(f"- {home_team}'s recent form is notably stronger, reinforcing the attack/defense ratings above.")
        elif away_form["points_per_game"] > home_form["points_per_game"] + 0.5:
            lines.append(f"- {away_team}'s recent form is notably stronger, which may offset some of the home advantage.")
        else:
            lines.append("- Recent form is fairly similar between the two teams.")

    lines.append("")
    lines.append("Note: this explanation describes what the statistical model is doing, not a "
                 "guarantee of the outcome - football results carry inherent randomness that no "
                 "model fully captures.")

    return "\n".join(lines)
