#!/usr/bin/env python3
"""
MLB Data Fetcher - Fully Automated
Fetches all pitcher/team stats from pybaseball (FanGraphs) and statsapi.
Run daily before first game (~11am ET).

Completely separate from NBA data pipeline.
"""
import json
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional

import pybaseball as pyb
import statsapi

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache file paths
PITCHER_CACHE = "mlb_pitcher_stats_cache.json"
TEAM_CACHE = "mlb_team_batting_cache.json"
PARK_CACHE = "mlb_park_factors_cache.json"

# Enable pybaseball caching for faster subsequent runs
pyb.cache.enable()


def parse_pct(val) -> Optional[float]:
    """Parse percentage value from FanGraphs (handles '25.3%' or 0.253 formats)."""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            return float(val.strip('%'))
        # If it's a decimal (0.253), convert to percentage
        return float(val) * 100 if abs(val) < 1 else float(val)
    except (ValueError, TypeError):
        return None


def fetch_pitcher_stats(year: int = 2026, min_ip: int = 1) -> Dict[str, Any]:
    """
    Fetch ALL pitcher stats from FanGraphs via pybaseball.

    Returns dict keyed by lowercase pitcher name.
    NO MANUAL INPUT REQUIRED.
    """
    logger.info(f"[MLB] Fetching pitcher stats from FanGraphs for {year}...")

    actual_year = year  # Track what year we actually loaded

    try:
        # This single call gets K%, SwStr%, CSW%, and 50+ other metrics
        df = pyb.fg_pitching_data(year, qual=min_ip)

        if df is None or df.empty:
            logger.warning(f"No pitcher data found for {year}, trying {year-1}")
            df = pyb.fg_pitching_data(year - 1, qual=min_ip)
            actual_year = year - 1
    except Exception as e:
        logger.error(f"Error fetching {year} data: {e}, trying {year-1}")
        df = pyb.fg_pitching_data(year - 1, qual=min_ip)
        actual_year = year - 1

    if df is None or df.empty:
        logger.error("No pitcher data available")
        return {}

    pitchers = {}
    for _, row in df.iterrows():
        name = row['Name']

        # Determine handedness
        throws = str(row.get('Throws', 'R'))
        hand = 'L' if 'L' in throws else 'R'

        pitchers[name.lower()] = {
            'name': name,
            'team': row.get('Team', ''),
            'hand': hand,
            'ip': float(row.get('IP', 0)),
            'games': int(row.get('G', 0)),
            'games_started': int(row.get('GS', 0)),

            # KEY METRICS for projection
            'k_pct': parse_pct(row.get('K%')),
            'swstr_pct': parse_pct(row.get('SwStr%')),
            'csw_pct': parse_pct(row.get('CSW%')),

            # Secondary metrics
            'bb_pct': parse_pct(row.get('BB%')),
            'k_per_9': float(row.get('K/9', 0)),
            'bb_per_9': float(row.get('BB/9', 0)),
            'o_swing_pct': parse_pct(row.get('O-Swing%')),
            'z_contact_pct': parse_pct(row.get('Z-Contact%')),
            'contact_pct': parse_pct(row.get('Contact%')),

            # Quality metrics
            'whip': float(row.get('WHIP', 0)),
            'era': float(row.get('ERA', 0)),
            'fip': float(row.get('FIP', 0)),
            'xfip': float(row.get('xFIP', 0)),
            'babip': float(row.get('BABIP', 0)),
            'war': float(row.get('WAR', 0)),
        }

    # Save cache with actual year loaded
    cache = {
        'updated': datetime.now().isoformat(),
        'season': actual_year,
        'requested_season': year,
        'count': len(pitchers),
        'pitchers': pitchers
    }

    with open(PITCHER_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    logger.info(f"[MLB] Cached {len(pitchers)} pitchers from {actual_year} season")
    return pitchers


def fetch_team_batting(year: int = 2026) -> Dict[str, Any]:
    """
    Fetch team batting stats including K% and P/PA.

    Returns dict keyed by team abbreviation.
    NO MANUAL INPUT REQUIRED.
    """
    logger.info(f"[MLB] Fetching team batting stats from FanGraphs for {year}...")

    actual_year = year  # Track what year we actually loaded

    try:
        df = pyb.fg_team_batting_data(year)
        if df is None or df.empty:
            logger.warning(f"No team data for {year}, trying {year-1}")
            df = pyb.fg_team_batting_data(year - 1)
            actual_year = year - 1
    except Exception as e:
        logger.error(f"Error fetching {year} team data: {e}, trying {year-1}")
        df = pyb.fg_team_batting_data(year - 1)
        actual_year = year - 1

    if df is None or df.empty:
        logger.error("No team batting data available")
        return {}

    teams = {}
    for _, row in df.iterrows():
        team = row['Team']

        pa = int(row.get('PA', 1))
        pitches = row.get('Pitches')
        if pitches is not None:
            p_per_pa = round(float(pitches) / pa, 2) if pa > 0 else 3.9
        else:
            p_per_pa = 3.9  # League average default

        teams[team] = {
            'team': team,
            'pa': pa,
            'k_pct': parse_pct(row.get('K%')),
            'bb_pct': parse_pct(row.get('BB%')),
            'p_per_pa': p_per_pa,
            'o_swing_pct': parse_pct(row.get('O-Swing%')),
            'z_contact_pct': parse_pct(row.get('Z-Contact%')),
            'contact_pct': parse_pct(row.get('Contact%')),
            'woba': float(row.get('wOBA', 0)),
            'wrc_plus': float(row.get('wRC+', 100)),
        }

    # Try to fetch L/R splits
    teams = _add_lr_splits(teams, actual_year)

    # Save cache with actual year loaded
    cache = {
        'updated': datetime.now().isoformat(),
        'season': actual_year,
        'requested_season': year,
        'count': len(teams),
        'teams': teams
    }

    with open(TEAM_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    logger.info(f"[MLB] Cached {len(teams)} teams with K%, P/PA")
    return teams


def _add_lr_splits(teams: Dict, year: int) -> Dict:
    """Add K% vs LHP/RHP splits to team data."""
    try:
        # Fetch splits vs left-handed pitchers
        df_vs_l = pyb.fg_team_batting_data(year, split_seasons=False)
        # Note: pybaseball may not support split parameter directly
        # If not available, we'll use overall K% for both
        logger.info("[MLB] L/R splits not directly available, using overall K%")

        for team in teams:
            # Default to overall K% for both
            teams[team]['k_pct_vs_lhp'] = teams[team]['k_pct']
            teams[team]['k_pct_vs_rhp'] = teams[team]['k_pct']

    except Exception as e:
        logger.warning(f"[MLB] Could not fetch L/R splits: {e}")
        for team in teams:
            teams[team]['k_pct_vs_lhp'] = teams[team]['k_pct']
            teams[team]['k_pct_vs_rhp'] = teams[team]['k_pct']

    return teams


def fetch_todays_games() -> list:
    """
    Fetch today's games with probable pitchers from MLB Stats API.

    NO MANUAL INPUT REQUIRED.
    """
    logger.info("[MLB] Fetching today's schedule from MLB Stats API...")

    today = date.today().strftime("%m/%d/%Y")

    try:
        schedule = statsapi.schedule(start_date=today, end_date=today)
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        return []

    games = []
    for game in schedule:
        games.append({
            'game_pk': game.get('game_id'),
            'game_date': game.get('game_date'),
            'game_time': game.get('game_datetime'),
            'status': game.get('status'),
            'home_team': game.get('home_name'),
            'away_team': game.get('away_name'),
            'home_pitcher': game.get('home_probable_pitcher'),
            'away_pitcher': game.get('away_probable_pitcher'),
            'venue': game.get('venue_name'),
        })

    logger.info(f"[MLB] Found {len(games)} games today")
    for g in games:
        hp = g['home_pitcher'] or 'TBD'
        ap = g['away_pitcher'] or 'TBD'
        logger.info(f"  {g['away_team']} @ {g['home_team']}: {ap} vs {hp}")

    return games


def fetch_park_factors() -> Dict[str, float]:
    """
    Get park K factors.

    These are relatively stable year-to-year.
    Source: Baseball Savant park factors.
    """
    logger.info("[MLB] Loading park K factors...")

    # Park K factors (>1 = more Ks, <1 = fewer Ks)
    # Based on historical Baseball Savant data
    park_factors = {
        # Pitcher-friendly (more Ks)
        'Tropicana Field': 1.08,
        'T-Mobile Park': 1.06,
        'Oracle Park': 1.05,
        'Oakland Coliseum': 1.04,
        'Petco Park': 1.03,
        'Dodger Stadium': 1.02,
        'Kauffman Stadium': 1.02,
        'Globe Life Field': 1.02,
        'Minute Maid Park': 1.01,
        'Progressive Field': 1.01,

        # Neutral
        'Yankee Stadium': 1.00,
        'Wrigley Field': 1.00,
        'Busch Stadium': 1.00,
        'Citizens Bank Park': 1.00,
        'Nationals Park': 1.00,
        'Truist Park': 1.00,
        'Target Field': 1.00,
        'American Family Field': 1.00,
        'Comerica Park': 1.00,
        'Angel Stadium': 1.00,
        'Rogers Centre': 1.00,
        'loanDepot park': 1.00,
        'PNC Park': 1.00,
        'Great American Ball Park': 0.99,
        'Camden Yards': 0.99,
        'Guaranteed Rate Field': 0.99,

        # Hitter-friendly (fewer Ks)
        'Fenway Park': 0.97,
        'Chase Field': 0.95,
        'Coors Field': 0.85,  # Altitude significantly reduces K rate

        # Default for unknown venues
        'default': 1.00
    }

    # Save cache
    cache = {
        'updated': datetime.now().isoformat(),
        'note': 'K factor: >1 = more Ks, <1 = fewer Ks',
        'parks': park_factors
    }

    with open(PARK_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)

    logger.info(f"[MLB] Cached {len(park_factors)} park factors")
    return park_factors


def get_pitcher_stats(pitcher_name: str) -> Optional[Dict]:
    """Get cached stats for a specific pitcher."""
    try:
        with open(PITCHER_CACHE, 'r') as f:
            cache = json.load(f)
        return cache['pitchers'].get(pitcher_name.lower())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_team_stats(team_name: str) -> Optional[Dict]:
    """Get cached stats for a specific team."""
    try:
        with open(TEAM_CACHE, 'r') as f:
            cache = json.load(f)

        # Try exact match first
        teams = cache['teams']
        if team_name in teams:
            return teams[team_name]

        # Try partial match
        team_lower = team_name.lower()
        for key, val in teams.items():
            if team_lower in key.lower():
                return val

        return None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_park_factor(venue: str) -> float:
    """Get K factor for a venue."""
    try:
        with open(PARK_CACHE, 'r') as f:
            cache = json.load(f)
        return cache['parks'].get(venue, cache['parks'].get('default', 1.0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 1.0


def refresh_all(year: int = 2026):
    """Refresh all MLB data caches."""
    print("=" * 60)
    print("MLB DAILY DATA REFRESH")
    print("=" * 60)

    fetch_pitcher_stats(year)
    fetch_team_batting(year)
    fetch_park_factors()
    games = fetch_todays_games()

    print("\n" + "=" * 60)
    print(f"ALL DATA REFRESHED - {len(games)} games today")
    print("=" * 60)

    return games


if __name__ == "__main__":
    refresh_all(2026)
