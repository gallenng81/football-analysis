"""
Dixon-Coles model for predicting football match scorelines.

This is the standard statistical approach used widely in football
analytics: each team gets an attack strength and a defense strength,
fitted from historical goals data via maximum likelihood. Combined with
a home-advantage term and a low-score correlation correction (the "Dixon-
Coles adjustment"), it produces a full probability matrix over scorelines
for any upcoming match.

Usage:
    model = DixonColes()
    model.fit(matches_df)
    matrix = model.predict_score_matrix("Arsenal", "Chelsea")
    outcome_probs = model.predict_outcome_probs("Arsenal", "Chelsea")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _dc_adjustment(home_goals, away_goals, lam_home, lam_away, rho):
    """Correction factor for low-scoring games (0-0, 1-0, 0-1, 1-1),
    where goals aren't quite independent Poisson processes."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lam_home * lam_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lam_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lam_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class DixonColes:
    def __init__(self, xi: float = 0.0018):
        """
        xi: time-decay rate. Higher values weight recent matches more
        heavily when fitting. 0 = no decay (all matches weighted equally).
        """
        self.xi = xi
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self._fitted = False

    def fit(self, matches: pd.DataFrame, date_col: str = "date"):
        """
        matches must have columns: home_team, away_team, home_goals, away_goals.
        date_col is optional but enables time-weighting recent form more heavily.
        """
        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        if date_col in matches.columns:
            dates = pd.to_datetime(matches[date_col])
            days_ago = (dates.max() - dates).dt.days.values
            weights = np.exp(-self.xi * days_ago)
        else:
            weights = np.ones(len(matches))

        home_idx = matches["home_team"].map(idx).values
        away_idx = matches["away_team"].map(idx).values
        home_goals = matches["home_goals"].values
        away_goals = matches["away_goals"].values

        def unpack(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(params):
            attack, defense, home_adv, rho = unpack(params)
            lam_home = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
            lam_away = np.exp(attack[away_idx] - defense[home_idx])

            ll = (
                poisson.logpmf(home_goals, lam_home)
                + poisson.logpmf(away_goals, lam_away)
            )
            # low-score correlation adjustment, applied multiplicatively
            adj = np.array([
                _dc_adjustment(hg, ag, lh, la, rho)
                for hg, ag, lh, la in zip(home_goals, away_goals, lam_home, lam_away)
            ])
            adj = np.clip(adj, 1e-6, None)  # avoid log(negative/0)
            ll = ll + np.log(adj)
            return -np.sum(ll * weights)

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [0.0]])
        # constrain average attack strength to 0 for identifiability
        constraints = [{
            "type": "eq",
            "fun": lambda p: np.mean(p[:n]),
        }]
        result = minimize(
            neg_log_likelihood, x0, method="SLSQP", constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        attack, defense, home_adv, rho = unpack(result.x)

        self.attack = dict(zip(self.teams, attack))
        self.defense = dict(zip(self.teams, defense))
        self.home_adv = float(home_adv)
        self.rho = float(np.clip(rho, -0.3, 0.3))
        self._fitted = True
        return self

    def _expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Call .fit() before predicting.")
        for t in (home_team, away_team):
            if t not in self.attack:
                raise ValueError(f"Unknown team: {t}")
        lam_home = np.exp(self.attack[home_team] - self.defense[away_team] + self.home_adv)
        lam_away = np.exp(self.attack[away_team] - self.defense[home_team])
        return float(lam_home), float(lam_away)

    def predict_score_matrix(self, home_team: str, away_team: str, max_goals: int = 8) -> pd.DataFrame:
        """Returns a (max_goals+1) x (max_goals+1) DataFrame of scoreline probabilities."""
        lam_home, lam_away = self._expected_goals(home_team, away_team)
        home_range = np.arange(max_goals + 1)
        away_range = np.arange(max_goals + 1)
        matrix = np.outer(
            poisson.pmf(home_range, lam_home),
            poisson.pmf(away_range, lam_away),
        )
        for h in range(2):
            for a in range(2):
                matrix[h, a] *= _dc_adjustment(h, a, lam_home, lam_away, self.rho)
        matrix = matrix / matrix.sum()  # renormalize after adjustment
        return pd.DataFrame(matrix, index=home_range, columns=away_range)

    def predict_outcome_probs(self, home_team: str, away_team: str, max_goals: int = 8) -> dict[str, float]:
        """Returns {'home_win':, 'draw':, 'away_win':} probabilities."""
        matrix = self.predict_score_matrix(home_team, away_team, max_goals).values
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        return {"home_win": float(home_win), "draw": float(draw), "away_win": float(away_win)}

    def most_likely_scores(self, home_team: str, away_team: str, top_n: int = 5, max_goals: int = 8):
        matrix = self.predict_score_matrix(home_team, away_team, max_goals)
        flat = matrix.stack().sort_values(ascending=False).head(top_n)
        return [(f"{h}-{a}", float(p)) for (h, a), p in flat.items()]

    def team_ratings(self) -> pd.DataFrame:
        """Attack/defense strength table, useful for a dashboard leaderboard."""
        return pd.DataFrame({
            "team": self.teams,
            "attack": [self.attack[t] for t in self.teams],
            "defense": [self.defense[t] for t in self.teams],
        }).sort_values("attack", ascending=False).reset_index(drop=True)
