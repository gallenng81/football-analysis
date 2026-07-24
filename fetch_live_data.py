"""
Templates for pulling live data. Fill in your API keys and run this on
your own machine — this sandbox can't reach these external APIs directly.

Recommended providers:
- Odds:    The Odds API (https://the-odds-api.com) - simple REST, free tier available
- Stats:   API-Football (https://www.api-football.com) - fixtures, results, injuries, xG-adjacent stats
"""
import os
import requests
import pandas as pd

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_ODDS_API_KEY")
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "YOUR_API_FOOTBALL_KEY")


def fetch_live_odds(sport_key: str = "soccer_epl", regions: str = "uk", markets: str = "h2h") -> pd.DataFrame:
    """
    Pulls current odds for upcoming matches. sport_key examples:
    'soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga'.
    See https://the-odds-api.com/sports-odds-data/sports-apis.html for the full list.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    events = resp.json()

    rows = []
    for event in events:
        home, away = event["home_team"], event["away_team"]
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                odds = {o["name"]: o["price"] for o in market["outcomes"]}
                rows.append({
                    "commence_time": event["commence_time"],
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": bookmaker["title"],
                    "home_odds": odds.get(home),
                    "draw_odds": odds.get("Draw"),
                    "away_odds": odds.get(away),
                })
    return pd.DataFrame(rows)


def fetch_recent_results(league_id: int, season: int, last_n: int = 100) -> pd.DataFrame:
    """
    Pulls recent match results for model fitting. league_id examples (API-Football):
    39 = Premier League, 140 = La Liga, 78 = Bundesliga, 135 = Serie A.
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    params = {"league": league_id, "season": season, "status": "FT", "last": last_n}
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    fixtures = resp.json().get("response", [])

    rows = []
    for fx in fixtures:
        rows.append({
            "date": fx["fixture"]["date"][:10],
            "home_team": fx["teams"]["home"]["name"],
            "away_team": fx["teams"]["away"]["name"],
            "home_goals": fx["goals"]["home"],
            "away_goals": fx["goals"]["away"],
        })
    return pd.DataFrame(rows)


def fetch_upcoming_fixtures(league_id: int, season: int, next_n: int = 20) -> pd.DataFrame:
    """
    Pulls the next N scheduled (not yet played) fixtures for a league.
    Same league_id codes as fetch_recent_results.
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    params = {"league": league_id, "season": season, "status": "NS", "next": next_n}
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    fixtures = resp.json().get("response", [])

    rows = []
    for fx in fixtures:
        rows.append({
            "date": fx["fixture"]["date"][:10],
            "kickoff": fx["fixture"]["date"],
            "home_team": fx["teams"]["home"]["name"],
            "away_team": fx["teams"]["away"]["name"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Set ODDS_API_KEY and FOOTBALL_API_KEY as environment variables, then:")
    print("  odds = fetch_live_odds('soccer_epl')")
    print("  results = fetch_recent_results(league_id=39, season=2025)")
    print("  fixtures = fetch_upcoming_fixtures(league_id=39, season=2025)")
