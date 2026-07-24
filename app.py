"""
Run with: streamlit run app.py

A sports analytics platform for football match analysis, with Free and
Premium feature tiers (see subscription.py for the tier definitions and
an important note on what is and isn't actually implemented re: payment
processing - short version: there is NO real billing here, the tier
toggle in the sidebar is a placeholder for demonstration).

Free tier: basic AI match predictions, team form, team ratings/league table.
Premium tier (S$9.90-29.90/month): confidence score, odds movement
analysis, player injury impact, AI explanation, notable match alerts,
historical performance dashboard.

This tool does not accept wagers, place bets, or link to any bookmaker.
All odds comparisons are presented for informational and educational
purposes only.
"""
import pandas as pd
import streamlit as st

from dixon_coles import DixonColes
from markets import all_markets
from backtest import rolling_backtest, summarize_backtest
from team_form import team_form, all_teams_form, form_trend
from odds_analysis import (
    generate_match_report, log_odds_snapshot, odds_movement, describe_movement
)
from odds_comparison import find_probability_divergences
from confidence_score import confidence_score
from injury_impact import predict_with_injury_adjustment
from ai_explanation import explain_prediction
from subscription import is_premium, PRICING, PREMIUM_FEATURES, FEATURE_LABELS

st.set_page_config(page_title="Football Analytics Platform", layout="wide")
st.title("Football Analytics Platform")
st.caption("AI match predictions, team form, and odds analysis for informational and "
           "educational purposes. This tool does not accept wagers or link to bookmakers.")

# ---------------------------------------------------------------- Tier selector (placeholder)
st.sidebar.header("Subscription")
st.sidebar.caption("Demo toggle only - no real billing is connected. See subscription.py "
                   "for what a production payment integration would need.")
tier_choice = st.sidebar.radio("Viewing as", ["Free", "Premium"], index=0)
user_tier = tier_choice.lower()

with st.sidebar.expander("Pricing"):
    st.write(f"**Free** — S${PRICING['free']['price_sgd']}/month")
    st.write("Basic AI prediction, team form, league table")
    st.write(f"**Premium** — S${PRICING['premium']['price_sgd_min']}–"
             f"{PRICING['premium']['price_sgd_max']}/month")
    for f in PREMIUM_FEATURES:
        st.write(f"- {FEATURE_LABELS[f]}")

st.sidebar.divider()
st.sidebar.header("Data")
data_file = st.sidebar.file_uploader("Upload historical results CSV", type="csv")
st.sidebar.caption("Columns needed: date, home_team, away_team, home_goals, away_goals")

if data_file is not None:
    matches = pd.read_csv(data_file)
else:
    matches = pd.read_csv("data/sample_matches.csv")
    st.sidebar.info("Using bundled sample data — upload your own for real predictions.")


@st.cache_resource
def fit_model(df: pd.DataFrame):
    return DixonColes().fit(df)


model = fit_model(matches)
teams = model.teams


def premium_lock(feature_key: str):
    """Renders a lock message in place of a premium feature when the
    user isn't on the premium tier."""
    st.info(f"🔒 **{FEATURE_LABELS[feature_key]}** is a Premium feature "
            f"(S${PRICING['premium']['price_sgd_min']}–{PRICING['premium']['price_sgd_max']}/month). "
            f"Switch to Premium in the sidebar to preview it.")


tab_predict, tab_form, tab_markets, tab_odds, tab_history, tab_ratings = st.tabs(
    ["Match predictions", "Team form", "Statistical markets", "Odds movement analysis",
     "Historical performance", "Team ratings"]
)

# ---------------------------------------------------------------- Match predictions
with tab_predict:
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("Home team", teams, index=0, key="home_predict")
    away_team = col2.selectbox("Away team", teams, index=1, key="away_predict")

    if home_team == away_team:
        st.warning("Pick two different teams.")
        st.stop()

    outcome = model.predict_outcome_probs(home_team, away_team)
    scores = model.most_likely_scores(home_team, away_team, top_n=6)

    st.subheader(f"{home_team} vs {away_team}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} win", f"{outcome['home_win']:.1%}")
    c2.metric("Draw", f"{outcome['draw']:.1%}")
    c3.metric(f"{away_team} win", f"{outcome['away_win']:.1%}")

    st.markdown("**Most likely scorelines**")
    score_df = pd.DataFrame(scores, columns=["Scoreline", "Probability"])
    score_df["Probability"] = score_df["Probability"].apply(lambda p: f"{p:.1%}")
    st.table(score_df)

    st.markdown("**Full scoreline probability matrix**")
    display_matrix = model.predict_score_matrix(home_team, away_team, max_goals=5)
    display_matrix.index.name = f"{home_team} goals"
    display_matrix.columns.name = f"{away_team} goals"
    st.dataframe(display_matrix.style.format("{:.1%}").background_gradient(cmap="Greens"))

    st.divider()
    st.subheader("Premium insights")

    # --- Confidence score (premium) ---
    if is_premium(user_tier):
        conf = confidence_score(outcome, matches, home_team, away_team)
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence score", f"{conf['score']}/100", help=conf["label"])
        c2.metric("Outcome sharpness", f"{conf['outcome_sharpness']}%",
                  help="How decisive the model's own probability split is")
        c3.metric("Data sufficiency", f"{conf['data_sufficiency']}%",
                  help="How much match history backs this prediction")
    else:
        premium_lock("confidence_score")

    # --- AI explanation (premium) ---
    if is_premium(user_tier):
        st.markdown("**AI explanation**")
        st.text(explain_prediction(model, matches, home_team, away_team))
    else:
        premium_lock("ai_explanation")

    # --- Injury impact (premium) ---
    if is_premium(user_tier):
        st.markdown("**Player injury impact**")
        st.caption("Estimate how a missing key player might shift the prediction. "
                   "This is a heuristic adjustment, not a learned effect - see injury_impact.py.")
        i1, i2 = st.columns(2)
        home_impact = i1.slider(f"{home_team} - missing player's estimated share of attack (%)", 0, 60, 0)
        away_impact = i2.slider(f"{away_team} - missing player's estimated share of attack (%)", 0, 60, 0)

        if home_impact or away_impact:
            result = predict_with_injury_adjustment(model, home_team, away_team, home_impact, away_impact)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{home_team} win (adjusted)", f"{result['adjusted']['home_win']:.1%}",
                      delta=f"{result['shift']['home_win']:+.1%}")
            c2.metric("Draw (adjusted)", f"{result['adjusted']['draw']:.1%}",
                      delta=f"{result['shift']['draw']:+.1%}")
            c3.metric(f"{away_team} win (adjusted)", f"{result['adjusted']['away_win']:.1%}",
                      delta=f"{result['shift']['away_win']:+.1%}")
        else:
            st.caption("Set a missing-player impact above to see the adjusted prediction.")
    else:
        premium_lock("injury_impact")

# ---------------------------------------------------------------- Team form
with tab_form:
    st.subheader("Recent form")
    st.caption("Points per game, results, and scoring trends over each team's most recent matches — "
               "purely descriptive of past results.")

    n_matches = st.slider("Number of recent matches to analyze", 3, 15, 5)
    form_table = all_teams_form(matches, n=n_matches)
    st.dataframe(form_table, use_container_width=True)

    st.divider()
    st.subheader("Single team form trend")
    selected_team = st.selectbox("Team", teams, key="form_team")
    trend = form_trend(matches, selected_team, recent_n=n_matches, prior_n=n_matches)

    if not trend.get("sufficient_data"):
        st.info("Not enough match history for this team to compute a trend with the selected window size.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Recent points per game", trend["recent_points_per_game"])
        c2.metric("Prior points per game", trend["prior_points_per_game"])
        c3.metric("Trend", trend["trend"].capitalize(), delta=trend["change"])

# ---------------------------------------------------------------- Statistical markets
with tab_markets:
    st.caption("Additional statistical breakdowns derived from the same model, for informational purposes.")
    col1, col2 = st.columns(2)
    home_m = col1.selectbox("Home team", teams, index=0, key="home_markets")
    away_m = col2.selectbox("Away team", teams, index=1, key="away_markets")

    if home_m == away_m:
        st.warning("Pick two different teams.")
    else:
        m_matrix = model.predict_score_matrix(home_m, away_m, max_goals=6)
        markets = all_markets(m_matrix)

        st.subheader(f"{home_m} vs {away_m}")

        st.markdown("**Both teams to score**")
        st.write(f"Yes: {markets['btts']['yes']:.1%}  |  No: {markets['btts']['no']:.1%}")

        st.markdown("**Total goals**")
        for line_key in ["over_under_1.5", "over_under_2.5", "over_under_3.5"]:
            line = line_key.split("_")[-1]
            ou = markets[line_key]
            st.write(f"Over/Under {line}: Over {ou['over']:.1%}  |  Under {ou['under']:.1%}")

        st.markdown("**Asian handicap**")
        ah0 = markets["asian_handicap_0"]
        st.write(f"Level (0): {home_m} covers {ah0['home_covers']:.1%}  |  push {ah0['push']:.1%}  |  {away_m} covers {ah0['away_covers']:.1%}")
        ah1 = markets["asian_handicap_-1"]
        st.write(f"{home_m} -1: covers {ah1['home_covers']:.1%}  |  push {ah1['push']:.1%}  |  {away_m} covers {ah1['away_covers']:.1%}")

        st.markdown("**Top correct scores**")
        cs_df = pd.DataFrame(markets["correct_scores"])
        cs_df["probability"] = cs_df["probability"].apply(lambda p: f"{p:.1%}")
        st.table(cs_df)

# ---------------------------------------------------------------- Odds movement analysis (Premium)
with tab_odds:
    if not is_premium(user_tier):
        premium_lock("odds_movement_analysis")
        premium_lock("notable_divergence_alerts")
    else:
        st.subheader("Understand what the odds are telling you")
        st.caption("Enter current market odds for a match to get a plain-language, informational "
                   "breakdown of where the model's probabilities agree or diverge from the market's, "
                   "plus odds movement tracking if you check back on the same match later. "
                   "This is presented for informational and educational purposes only — not a betting recommendation.")

        col1, col2 = st.columns(2)
        home_o = col1.selectbox("Home team", teams, index=0, key="home_odds")
        away_o = col2.selectbox("Away team", teams, index=1, key="away_odds")

        if home_o != away_o:
            o1, o2, o3 = st.columns(3)
            home_odds_val = o1.number_input(f"{home_o} win — market odds", min_value=1.01, value=2.50, step=0.01, key="odds_h")
            draw_odds_val = o2.number_input("Draw — market odds", min_value=1.01, value=3.40, step=0.01, key="odds_d")
            away_odds_val = o3.number_input(f"{away_o} win — market odds", min_value=1.01, value=3.20, step=0.01, key="odds_a")

            market_odds = {"home_win": home_odds_val, "draw": draw_odds_val, "away_win": away_odds_val}
            outcome_o = model.predict_outcome_probs(home_o, away_o)
            matrix_o = model.predict_score_matrix(home_o, away_o, max_goals=6)
            top_scores_o = model.most_likely_scores(home_o, away_o, top_n=3)
            markets_o = all_markets(matrix_o)

            report = generate_match_report(home_o, away_o, outcome_o, market_odds, top_scores_o, markets_o)
            st.text(report)

            st.divider()
            st.subheader("Notable match alerts")
            st.caption("Flags outcomes where the model's probability notably diverges from the "
                       "market's - informational only, not a betting recommendation.")
            divergences = find_probability_divergences(outcome_o, market_odds, divergence_threshold=0.03)
            if divergences:
                for d in divergences:
                    st.warning(
                        f"**{d['outcome'].replace('_', ' ').title()}**: model {d['model_probability']:.1%} "
                        f"vs market {d['market_fair_probability']:.1%} (divergence {d['divergence']:+.1%})"
                    )
            else:
                st.success("No notable divergences at these odds - model and market are aligned.")

            st.divider()
            if st.button("Log this odds snapshot", help="Saves a timestamped snapshot so you can track movement if you check back later"):
                log_odds_snapshot(home_o, away_o, market_odds)
                st.success("Snapshot logged.")

            movement = odds_movement(home_o, away_o)
            if movement:
                st.markdown("**Odds movement since first snapshot**")
                st.text(describe_movement(movement, home_o, away_o))
            else:
                st.info("No movement history yet for this matchup - log a snapshot now, then check back "
                        "after the odds change to see which way the market has moved.")

# ---------------------------------------------------------------- Historical performance (Premium)
with tab_history:
    if not is_premium(user_tier):
        premium_lock("historical_performance_dashboard")
    else:
        st.subheader("Walk-forward accuracy tracker")
        st.caption("Predicts each match using only data available BEFORE it happened, then checks "
                   "against the real result. This is the honest way to know if the model has any skill.")

        min_train = st.slider("Minimum training matches before evaluating", 20, 100, 60, 10)
        refit_every = st.slider("Refit model every N matches", 1, 20, 10, 1)

        if st.button("Run backtest"):
            with st.spinner("Running walk-forward backtest..."):
                log = rolling_backtest(matches, min_train_matches=min_train, refit_every=refit_every)
            summary = summarize_backtest(log)

            if "error" in summary:
                st.error(summary["error"])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Matches evaluated", summary["matches_evaluated"])
                c2.metric("Favorite pick accuracy", f"{summary['accuracy']:.1%}")
                c3.metric("Naive 'always home' accuracy", f"{summary['naive_always_home_accuracy']:.1%}")

                c4, c5 = st.columns(2)
                c4.metric("Mean Brier score", f"{summary['mean_brier_score']:.3f}", help="Lower is better. 0 = perfect, ~0.67 = random guessing.")
                c5.metric("Mean log loss", f"{summary['mean_log_loss']:.3f}", help="Lower is better.")

                if summary["accuracy"] <= summary["naive_always_home_accuracy"]:
                    st.warning("The model isn't beating the naive 'always pick home team' baseline here. "
                               "On a small synthetic sample that's expected — rerun this against real "
                               "historical data to get a meaningful read.")

                st.markdown("**Rolling accuracy over time**")
                log["correct_favorite_numeric"] = log["correct_favorite"].astype(int)
                log["rolling_accuracy"] = log["correct_favorite_numeric"].expanding().mean()
                st.line_chart(log.set_index("date")["rolling_accuracy"])

                st.markdown("**Prediction ledger**")
                st.dataframe(log, use_container_width=True)
                st.download_button("Download ledger as CSV", log.to_csv(index=False), "backtest_log.csv")

# ---------------------------------------------------------------- Team ratings
with tab_ratings:
    st.subheader("Team attack/defense ratings")
    st.caption("Derived from the fitted model - higher attack means more goals scored on average, "
               "higher defense means fewer goals conceded on average.")
    st.dataframe(model.team_ratings(), use_container_width=True)

st.divider()
st.caption("This platform provides AI-generated statistical predictions and analysis for "
           "informational and educational purposes only. It does not accept wagers, place bets, "
           "or link to any bookmaker or betting service.")
