"""
Alternative to fetch_live_data.py that needs NO API key and NO signup:
football-data.co.uk publishes free CSV downloads of historical results,
match odds, and upcoming fixtures with pre-match odds, updated twice
weekly (results) and Friday/Tuesday afternoons (fixtures).

Trade-off vs the API-Football/Odds-API route: this only covers ~30
countries' top divisions (not 1200+ leagues), and "upcoming odds" means
a snapshot collected once before each round rather than truly live odds
that update by the minute. For most personal-use score prediction this
is a perfectly good trade for not needing any account at all.

League codes (used in both functions below):
    E0 = England Premier League      E1 = England Championship
    SP1 = Spain La Liga               SP2 = Spain Segunda
    D1 = Germany Bundesliga           D2 = Germany Bundesliga 2
    I1 = Italy Serie A                I2 = Italy Serie B
    F1 = France Ligue 1               F2 = France Ligue 2
    N1 = Netherlands Eredivisie       B1 = Belgium First Division
    P1 = Portugal Primeira Liga       T1 = Turkey Super Lig
    G1 = Greece Super League
Full list and column meanings: https://www.football-data.co.uk/notes.txt
"""
from __future__ import annotations
import requests
import pandas as pd
from io import StringIO

BASE_URL = "https://www.football-data.co.uk"

# Priority order for odds columns - not every season/league has every
# bookmaker, so fall back down this list until one is found.
ODDS_COLUMN_SETS = [
    ("PSH", "PSD", "PSA"),      # Pinnacle - sharpest line, when available
    ("B365H", "B365D", "B365A"),  # Bet365 - most consistently available
    ("AvgH", "AvgD", "AvgA"),    # market average across bookmakers
    ("WHH", "WHD", "WHA"),      # William Hill
]


def _parse_dates(date_series: pd.Series) -> pd.Series:
    """football-data.co.uk uses dd/mm/yy in recent seasons and dd/mm/yyyy
    in some older ones - try both explicitly rather than let pandas guess
    per-row (slow and warns)."""
    parsed = pd.to_datetime(date_series, format="%d/%m/%y", errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        parsed_alt = pd.to_datetime(date_series[still_missing], format="%d/%m/%Y", errors="coerce")
        parsed.loc[still_missing] = parsed_alt
    return parsed.dt.strftime("%Y-%m-%d")


def _season_code(season_start_year: int) -> str:
    """2025 -> '2526' (season 2025/26), matching football-data.co.uk's URL format."""
    yy1 = season_start_year % 100
    yy2 = (season_start_year + 1) % 100
    return f"{yy1:02d}{yy2:02d}"


def _pick_odds_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    for h, d, a in ODDS_COLUMN_SETS:
        if h in df.columns and d in df.columns and a in df.columns:
            return h, d, a
    return None


def fetch_results(league_code: str, season_start_year: int) -> pd.DataFrame:
    """
    Downloads full-time results (+ odds if available) for one league/season.
    Returns columns: date, home_team, away_team, home_goals, away_goals,
    plus home_odds/draw_odds/away_odds if an odds column set was found.

    Example: fetch_results('E0', 2024) -> Premier League 2024/25 season.
    """
    url = f"{BASE_URL}/mmz4281/{_season_code(season_start_year)}/{league_code}.csv"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))

    out = pd.DataFrame({
        "date": _parse_dates(df["Date"]),
        "home_team": df["HomeTeam"],
        "away_team": df["AwayTeam"],
        "home_goals": df["FTHG"],
        "away_goals": df["FTAG"],
    })

    odds_cols = _pick_odds_columns(df)
    if odds_cols:
        h, d, a = odds_cols
        out["home_odds"] = df[h]
        out["draw_odds"] = df[d]
        out["away_odds"] = df[a]

    return out.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)


def fetch_multi_season_results(league_code: str, start_years: list[int]) -> pd.DataFrame:
    """Convenience wrapper to pull several seasons and concatenate them -
    gives the model more history to fit on. E.g. start_years=[2022,2023,2024]."""
    frames = []
    for year in start_years:
        try:
            frames.append(fetch_results(league_code, year))
        except Exception as e:
            print(f"Could not fetch {league_code} season {year}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_upcoming_fixtures(league_code: str | None = None) -> pd.DataFrame:
    """
    Downloads the current week's upcoming fixtures with pre-match odds
    across all main leagues. Pass league_code to filter to one league
    (e.g. 'E0'), or None to get everything.

    Odds here are a snapshot collected Friday afternoons (weekend
    fixtures) or Tuesday afternoons (midweek fixtures) - not live, but
    free and good enough for pre-match value comparison.
    """
    url = f"{BASE_URL}/fixtures.csv"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))

    if league_code:
        df = df[df["Div"] == league_code]

    out = pd.DataFrame({
        "date": _parse_dates(df["Date"]),
        "league": df["Div"],
        "home_team": df["HomeTeam"],
        "away_team": df["AwayTeam"],
    })

    odds_cols = _pick_odds_columns(df)
    if odds_cols:
        h, d, a = odds_cols
        out["home_odds"] = df[h]
        out["draw_odds"] = df[d]
        out["away_odds"] = df[a]

    return out.reset_index(drop=True)


if __name__ == "__main__":
    print("No API key needed. Example usage:")
    print("  results = fetch_results('E0', 2024)              # one season")
    print("  history = fetch_multi_season_results('E0', [2022, 2023, 2024])")
    print("  fixtures = fetch_upcoming_fixtures('E0')          # this week's fixtures + odds")
