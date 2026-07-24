# Football Analytics Platform

An AI-powered sports analytics tool for football (soccer) match analysis, built around a
Dixon-Coles statistical model.

**This is an informational and educational analytics tool. It does not accept wagers, place
bets, or link to any bookmaker or betting service.**

## Features

- **AI match predictions** — full scoreline probability distributions for any matchup
- **Win/draw/loss probabilities** — 1X2 outcome probabilities derived from the model
- **Team form analysis** — recent points per game, results, and scoring trends per team
- **Additional statistical markets** — both teams to score, over/under goals, Asian handicap,
  most likely correct scores
- **Odds movement analysis** — informational comparison between the model's probabilities and
  market odds, plus tracking of how odds shift over time
- **Historical back-testing** — walk-forward accuracy tracking so you can see honestly whether
  the model has any predictive skill, rather than just producing plausible-looking numbers

No betting links. No bookmaker affiliate links. No wagers accepted anywhere in this tool.

## Subscription tiers

The dashboard (`app.py`) includes a Free/Premium tier structure:

**Free** — S$0/month
- Basic AI match predictions
- Team form
- Team ratings / league table

**Premium** — S$9.90–S$29.90/month
- Confidence score
- Odds movement analysis
- Player injury impact
- AI explanation
- Notable match alerts
- Historical performance dashboard

**Important - no real billing is implemented.** The sidebar tier toggle
in `app.py` is a placeholder for demonstrating how the tiers behave -
there is no payment processor, no account system, and no subscription
verification. `subscription.py` documents exactly what a real
integration would need (Stripe Checkout, a webhook listener, a database
of subscription status) - none of that infrastructure exists here. Wiring
up actual billing requires a real backend (Streamlit alone can't receive
webhooks) and a Stripe (or similar) account, which you'd need to set up
and connect yourself.

## Feature files (new in this version)

- `subscription.py` — tier definitions (Free/Premium) and feature-gating
  logic. Read the module docstring for what's needed to make billing real.
- `confidence_score.py` — a 0-100 confidence score combining how decisive
  the model's probability split is with how much match history backs the
  prediction. A communication aid, not a statistically calibrated metric.
- `injury_impact.py` — heuristic adjustment showing how a missing key
  player might shift a prediction, based on a user-supplied estimate of
  that player's share of the team's attacking output. There's no
  player-level data in this project, so this is a transparent, clearly
  labeled adjustment tool rather than a learned effect.
- `ai_explanation.py` — plain-language explanation of what's driving a
  prediction (attack/defense ratings, home advantage, recent form).
  Template-driven from the model's own numbers, so it works without any
  additional API key; the natural extension point if you want richer
  prose is to pass these same numbers to an LLM call instead.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (with bundled sample data)

```bash
python run_demo.py          # command-line demo
streamlit run app.py        # interactive dashboard
```

## Files

- `dixon_coles.py` — the core model. Fits attack/defense strength per team
  from historical goals data, predicts full scoreline probability matrices.
- `team_form.py` — recent form analysis: points per game, W/D/L record, and
  scoring trends over a team's last N matches, plus a simple trend read
  (improving/declining/stable). Purely descriptive of past results.
- `markets.py` — derives BTTS, over/under, Asian handicap, and correct
  score markets from the same scoreline matrix, so every market is
  automatically consistent with the others.
- `odds_comparison.py` — compares model probabilities against market odds
  for informational purposes: converts odds to implied probabilities,
  removes the bookmaker's built-in margin, and reports where the model's
  view diverges from the market's. No betting recommendations.
- `odds_analysis.py` — turns model + market odds into a plain-language
  report: which side each favors, how big the divergence is and what
  that means, plus odds-movement tracking across snapshots over time.
- `backtest.py` — walk-forward backtesting: refits the model periodically
  using only data available at the time, predicts each match before
  seeing the result, and logs accuracy, Brier score, and log loss against
  a naive "always pick home team" baseline.
- `fetch_live_data.py` — templates for pulling live odds (The Odds API),
  recent results, and upcoming fixtures (API-Football). Requires signup
  and API keys: `ODDS_API_KEY`, `FOOTBALL_API_KEY`.
- `football_data_uk.py` — an alternative that needs no signup or API key
  at all. Downloads free CSV data from football-data.co.uk: historical
  results, market odds, and upcoming fixtures with pre-match odds bundled
  in. Covers ~30 countries' top divisions.
- `team_matching.py` — fuzzy-matches team names across providers (e.g.
  "Man City" from one feed vs "Manchester City" from another).
- `predict_upcoming.py` — the real-match analysis pipeline using
  API-Football + The Odds API. Fetches recent results, fits the model,
  fetches upcoming fixtures, compares them against market odds, and saves
  predictions plus notable divergences to `predictions.csv`.
- `predict_upcoming_free.py` — the same pipeline using football-data.co.uk
  instead — no API keys needed. Saves to `predictions_free.csv`.
- `app.py` — the Streamlit dashboard with six tabs and Free/Premium tier
  gating: match predictions, team form, statistical markets, odds
  movement analysis (Premium), historical performance (Premium), and
  team ratings.
- `data/generate_sample_data.py` — generates synthetic match data so you
  can test everything before connecting real data.

## Deploying as a website

This project is now ready to deploy as-is - `Dockerfile`, `render.yaml`,
`Procfile`, and `.streamlit/config.toml` are all included and have been
tested to boot cleanly.

### Option A: Streamlit Community Cloud (free, easiest)

1. Push this project to a GitHub repo.
2. Go to https://share.streamlit.io, sign in, and click "New app."
3. Point it at your repo, branch, and `app.py` as the entry file.
4. Deploy. You'll get a URL like `your-app.streamlit.app`.

Good for a quick public demo. Limitations: the app sleeps after a period
of inactivity (a visitor's first load will be slow while it wakes up),
limited compute/memory, and no custom domain on the free tier.

### Option B: Render (custom domain, no sleep, room to grow)

1. Push this project to a GitHub repo (needs `Dockerfile` and
   `render.yaml`, both included).
2. Go to https://render.com, sign in, and click "New +" → "Blueprint."
3. Point it at your repo - Render will read `render.yaml` and configure
   the service automatically.
4. Deploy. Render gives you a `.onrender.com` URL immediately, and you
   can attach a custom domain in the service settings once you have one.

The free tier on Render also spins down after inactivity; paid tiers
($7/month and up) keep it always-on. This is also the option to pick
once you're ready to add real Stripe billing (see `subscription.py`),
since Render lets you run a second small service alongside this one to
receive Stripe's webhook - something Streamlit alone can't do.

### Option C: Railway

Same idea as Render - push to GitHub, connect the repo at
https://railway.app, and it will detect the included `Procfile`
automatically. Pricing and always-on behavior are broadly similar to
Render.

### Local testing before deploying

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

Or with Docker, once you have it installed locally:
```bash
docker build -t football-analytics .
docker run -p 8501:8501 football-analytics
```



```bash
python predict_upcoming_free.py --league E0 --seasons 2022 2023 2024
```

This downloads three seasons of real Premier League results from
football-data.co.uk (no signup required), fits the model, downloads this
week's fixtures with market odds bundled in, and saves predictions plus
notable divergences to `predictions_free.csv`.

Common league codes: `E0` Premier League, `E1` Championship, `SP1` La
Liga, `D1` Bundesliga, `I1` Serie A, `F1` Ligue 1, `N1` Eredivisie, `P1`
Primeira Liga. Full list at football-data.co.uk/notes.txt.

Two things worth knowing:
- Fixtures are only available for the **current week** — the site
  refreshes `fixtures.csv` Friday afternoons (weekend games) and Tuesday
  afternoons (midweek games), UK time.
- Odds are a one-time snapshot collected before each round, not
  continuously live.

## Predicting real upcoming matches — with live odds (API keys required)

1. Sign up for API keys:
   - https://the-odds-api.com (live odds)
   - https://www.api-football.com (fixtures, results, injuries)
2. Set them as environment variables:
   ```bash
   export FOOTBALL_API_KEY=your_key
   export ODDS_API_KEY=your_key
   ```
3. Run the pipeline for a league:
   ```bash
   python predict_upcoming.py --league 39 --season 2025 --odds-key soccer_epl --next 20
   ```
   League IDs: 39=Premier League, 140=La Liga, 78=Bundesliga, 135=Serie A,
   61=Ligue 1, 2=Champions League. Odds sport keys: soccer_epl,
   soccer_spain_la_liga, soccer_germany_bundesliga, soccer_italy_serie_a,
   soccer_france_ligue_one.
4. Fixtures for newly promoted teams or unmatched team names get skipped
   with a printed reason rather than silently producing a wrong prediction.

## Odds movement analysis

Since continuously live odds require an API key, `odds_analysis.py` also
supports periodic snapshot logging: every time you check a match, it logs
a timestamped odds snapshot. Check back later and it reports whether the
market has shortened or drifted on each outcome — useful signal even from
occasional manual checks, without needing a live feed.

## Tracking market-wide odds movement automatically

Beyond checking one match at a time, `market_tracker.py` and
`fetch_market_odds.py` track odds across many matches automatically, with
no manual clicking required:

- `fetch_market_odds.py` pulls this week's fixture odds from
  football-data.co.uk across whichever leagues you list, and appends a
  timestamped snapshot for every match to `data/market_odds_history.csv`.
- `.github/workflows/track_odds.yml` runs that script on a schedule
  (twice daily by default) via GitHub Actions, and commits the updated
  history file back to your repo — so the data persists even though
  Streamlit Cloud's filesystem doesn't.
- `market_tracker.py` reads that history and reports the biggest movers
  and any "steam moves" (odds shortening notably since first tracked)
  across every match with at least two snapshots logged.
- The dashboard's "Odds movement analysis" tab (Premium) shows this as a
  market-wide table, above the existing single-match manual lookup.

**To turn this on once your repo is on GitHub:**
1. The workflow file is already included at
   `.github/workflows/track_odds.yml` — no extra setup needed for it to
   exist, but GitHub Actions needs permission to push commits back to
   your repo. Go to your repo's Settings → Actions → General → Workflow
   permissions, and select "Read and write permissions."
2. That's it - the workflow runs automatically on its schedule. You can
   also trigger it manually anytime from the "Actions" tab on GitHub
   ("Track market odds" → "Run workflow").
3. Edit the `--leagues` list in `track_odds.yml` to whichever
   football-data.co.uk league codes you want tracked.

Since football-data.co.uk itself only refreshes fixture odds on Friday
and Tuesday afternoons (UK time), most scheduled runs between those
refreshes will just re-log the same numbers — that's expected, and just
means there's nothing new to report until the next real market update.

## How the model works

Each team gets an **attack strength** and **defense strength**, estimated
via maximum likelihood from historical goals scored/conceded (with a
home-advantage term and a small correction for low-scoring games — the
"Dixon-Coles adjustment"). Combined, these give expected goals for both
teams in any matchup, which produces a full Poisson probability matrix
over scorelines.

**Caveat**: predicting football matches is genuinely hard. Market odds
are set by professionals with extensive data and resources, and models
frequently show no consistent edge over the market. Run the **Backtest
tracker** tab before drawing conclusions from any divergence the tool
reports — on the bundled synthetic data it's normal (and honest) for the
model to not beat a naive "always pick home team" baseline, since that's
a small, noisy sample. The point of the tracker is to give you an honest
read against real historical data.

## Scope and limitations

This is a statistical analytics tool for informational and educational
purposes. It:
- Does not accept wagers or process any form of payment
- Does not place bets on the user's behalf
- Does not link to bookmakers or betting services, and contains no
  affiliate links
- Does not provide financial or betting advice — probability comparisons
  are presented as statistical information, not recommendations
