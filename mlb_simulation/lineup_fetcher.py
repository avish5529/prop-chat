#!/usr/bin/env python3
"""
MLB Lineup Fetcher for Strikeout Simulation

Fetches confirmed starting lineups from MLB Stats API.
Used for PA-level Monte Carlo simulation.

Phase 1 of Tier 3 MLB Simulation.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

LINEUP_CACHE_FILE = CACHE_DIR / "mlb_lineups.json"
LINEUP_CACHE_TTL_MINUTES = 15  # Lineups can change close to game time

# Team abbreviation mappings
TEAM_ABBREV = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
}

# Reverse mapping
ABBREV_TO_TEAM = {v: k for k, v in TEAM_ABBREV.items()}

# City name to abbreviation mapping (for FanGraphs data that uses city names)
CITY_TO_ABBREV = {
    'arizona': 'ARI', 'atlanta': 'ATL', 'baltimore': 'BAL', 'boston': 'BOS',
    'chicago': 'CHC',  # Default to Cubs, will need context for White Sox
    'cincinnati': 'CIN', 'cleveland': 'CLE', 'colorado': 'COL', 'detroit': 'DET',
    'houston': 'HOU', 'kansas city': 'KC', 'los angeles': 'LAD',  # Default to Dodgers
    'miami': 'MIA', 'milwaukee': 'MIL', 'minnesota': 'MIN', 'new york': 'NYY',  # Default to Yankees
    'oakland': 'OAK', 'philadelphia': 'PHI', 'pittsburgh': 'PIT', 'san diego': 'SD',
    'san francisco': 'SF', 'seattle': 'SEA', 'st. louis': 'STL', 'st louis': 'STL',
    'tampa bay': 'TB', 'texas': 'TEX', 'toronto': 'TOR', 'washington': 'WSH',
    # Team nicknames
    'd-backs': 'ARI', 'diamondbacks': 'ARI', 'braves': 'ATL', 'orioles': 'BAL',
    'red sox': 'BOS', 'cubs': 'CHC', 'white sox': 'CHW', 'reds': 'CIN',
    'guardians': 'CLE', 'rockies': 'COL', 'tigers': 'DET', 'astros': 'HOU',
    'royals': 'KC', 'angels': 'LAA', 'dodgers': 'LAD', 'marlins': 'MIA',
    'brewers': 'MIL', 'twins': 'MIN', 'mets': 'NYM', 'yankees': 'NYY',
    'athletics': 'OAK', "a's": 'OAK', 'phillies': 'PHI', 'pirates': 'PIT',
    'padres': 'SD', 'giants': 'SF', 'mariners': 'SEA', 'cardinals': 'STL',
    'rays': 'TB', 'rangers': 'TEX', 'blue jays': 'TOR', 'nationals': 'WSH',
}


def load_lineup_cache() -> Optional[Dict]:
    """Load lineup cache if exists and not stale."""
    if not LINEUP_CACHE_FILE.exists():
        return None

    try:
        with open(LINEUP_CACHE_FILE, 'r') as f:
            data = json.load(f)

        # Check staleness
        metadata = data.get('_metadata', {})
        updated_str = metadata.get('updated')
        if updated_str:
            updated = datetime.fromisoformat(updated_str)
            age_minutes = (datetime.now() - updated).total_seconds() / 60
            if age_minutes > LINEUP_CACHE_TTL_MINUTES:
                logger.info(f"[Lineups] Cache is stale ({age_minutes:.1f}min old)")
                return None

        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"[Lineups] Failed to load cache: {e}")
        return None


def save_lineup_cache(data: Dict) -> bool:
    """Save lineup data to cache."""
    data['_metadata'] = {
        'updated': datetime.now().isoformat(),
        'cache_name': 'lineups',
    }

    try:
        with open(LINEUP_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[Lineups] Saved {len(data.get('games', {}))} game lineups to cache")
        return True
    except IOError as e:
        logger.error(f"[Lineups] Failed to save cache: {e}")
        return False


def get_team_abbrev(team_name: str) -> str:
    """Convert full team name, city name, or nickname to abbreviation."""
    if not team_name:
        return team_name

    # Already an abbreviation?
    team_upper = team_name.upper()
    if team_upper in ABBREV_TO_TEAM:
        return team_upper

    # Exact match in full names
    if team_name in TEAM_ABBREV:
        return TEAM_ABBREV[team_name]

    # Check city/nickname mapping
    team_lower = team_name.lower().strip()
    if team_lower in CITY_TO_ABBREV:
        return CITY_TO_ABBREV[team_lower]

    # Try partial match on full team names
    for full_name, abbrev in TEAM_ABBREV.items():
        if team_lower in full_name.lower() or full_name.lower() in team_lower:
            return abbrev

    # Try partial match on city/nickname
    for city, abbrev in CITY_TO_ABBREV.items():
        if team_lower in city or city in team_lower:
            return abbrev

    return team_name


def fetch_game_lineup(game_id: int) -> Optional[Dict]:
    """
    Fetch lineup for a specific game.

    Args:
        game_id: MLB game ID

    Returns:
        Dict with home and away lineups, or None if not available
    """
    if not STATSAPI_AVAILABLE:
        logger.error("[Lineups] statsapi not available")
        return None

    try:
        # Get boxscore which contains lineup info
        boxscore = statsapi.boxscore_data(game_id)

        if not boxscore:
            return None

        home_team = boxscore.get('teamInfo', {}).get('home', {}).get('teamName', '')
        away_team = boxscore.get('teamInfo', {}).get('away', {}).get('teamName', '')

        # Extract batting order
        home_lineup = []
        away_lineup = []

        # Home batters
        home_batters = boxscore.get('homeBatters', [])
        for batter in home_batters:
            if batter.get('battingOrder'):
                order = int(str(batter['battingOrder'])[:1])  # First digit is batting order
                if 1 <= order <= 9:
                    home_lineup.append({
                        'order': order,
                        'name': batter.get('name', ''),
                        'id': batter.get('personId'),
                        'position': batter.get('position', ''),
                    })

        # Away batters
        away_batters = boxscore.get('awayBatters', [])
        for batter in away_batters:
            if batter.get('battingOrder'):
                order = int(str(batter['battingOrder'])[:1])
                if 1 <= order <= 9:
                    away_lineup.append({
                        'order': order,
                        'name': batter.get('name', ''),
                        'id': batter.get('personId'),
                        'position': batter.get('position', ''),
                    })

        # Sort by batting order
        home_lineup.sort(key=lambda x: x['order'])
        away_lineup.sort(key=lambda x: x['order'])

        # Get probable pitchers
        home_pitcher = None
        away_pitcher = None

        home_pitchers = boxscore.get('homePitchers', [])
        if home_pitchers:
            home_pitcher = {
                'name': home_pitchers[0].get('name', ''),
                'id': home_pitchers[0].get('personId'),
            }

        away_pitchers = boxscore.get('awayPitchers', [])
        if away_pitchers:
            away_pitcher = {
                'name': away_pitchers[0].get('name', ''),
                'id': away_pitchers[0].get('personId'),
            }

        return {
            'game_id': game_id,
            'home_team': home_team,
            'home_abbrev': get_team_abbrev(home_team),
            'away_team': away_team,
            'away_abbrev': get_team_abbrev(away_team),
            'home_lineup': home_lineup,
            'away_lineup': away_lineup,
            'home_pitcher': home_pitcher,
            'away_pitcher': away_pitcher,
            'lineup_confirmed': len(home_lineup) >= 9 and len(away_lineup) >= 9,
        }

    except Exception as e:
        logger.warning(f"[Lineups] Failed to fetch lineup for game {game_id}: {e}")
        return None


def fetch_todays_lineups() -> Dict[str, Any]:
    """
    Fetch lineups for all of today's games.

    Returns:
        Dict with game lineups keyed by game_id
    """
    if not STATSAPI_AVAILABLE:
        logger.error("[Lineups] statsapi not available")
        return {}

    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"[Lineups] Fetching lineups for {today}...")

    try:
        schedule = statsapi.schedule(date=today)
    except Exception as e:
        logger.error(f"[Lineups] Failed to fetch schedule: {e}")
        return {}

    games = {}
    confirmed_count = 0

    for game in schedule:
        game_id = game.get('game_id')
        if not game_id:
            continue

        lineup = fetch_game_lineup(game_id)
        if lineup:
            games[str(game_id)] = lineup
            if lineup.get('lineup_confirmed'):
                confirmed_count += 1

        # Be nice to the API
        import time
        time.sleep(0.5)

    result = {
        'date': today,
        'total_games': len(games),
        'confirmed_lineups': confirmed_count,
        'games': games,
    }

    save_lineup_cache(result)
    logger.info(f"[Lineups] Fetched {len(games)} games, {confirmed_count} with confirmed lineups")

    return result


def get_lineup(team_abbrev: str, game_date: str = None) -> Optional[List[Dict]]:
    """
    Get batting lineup for a team.

    Args:
        team_abbrev: Team abbreviation (e.g., 'NYY', 'LAD')
        game_date: Date string (defaults to today)

    Returns:
        List of batter dicts in batting order, or None if not found
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    # Check cache first
    cache = load_lineup_cache()

    # If cache is for a different date, refresh
    if cache and cache.get('date') != game_date:
        cache = None

    if not cache:
        cache = fetch_todays_lineups()

    if not cache:
        return None

    # Normalize team name to abbreviation
    team_normalized = get_team_abbrev(team_abbrev)

    # Search through games
    for game_id, game_data in cache.get('games', {}).items():
        if game_data.get('home_abbrev') == team_normalized:
            return game_data.get('home_lineup', [])
        elif game_data.get('away_abbrev') == team_normalized:
            return game_data.get('away_lineup', [])

    return None


def get_opponent_lineup(pitcher_team: str, game_date: str = None) -> Optional[Dict]:
    """
    Get the lineup that will face a pitcher.

    Args:
        pitcher_team: Pitcher's team abbreviation
        game_date: Date string (defaults to today)

    Returns:
        Dict with lineup and pitcher info, or None if not found
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    cache = load_lineup_cache()

    if cache and cache.get('date') != game_date:
        cache = None

    if not cache:
        cache = fetch_todays_lineups()

    if not cache:
        return None

    # Normalize team name to abbreviation (handles city names like "Milwaukee" -> "MIL")
    team_normalized = get_team_abbrev(pitcher_team)

    for game_id, game_data in cache.get('games', {}).items():
        if game_data.get('home_abbrev') == team_normalized:
            # Pitcher is home team, opponent is away
            opponent_lineup = game_data.get('away_lineup', [])
            return {
                'lineup': opponent_lineup,
                'team': game_data.get('away_abbrev'),
                'team_name': game_data.get('away_team'),
                'pitcher': game_data.get('home_pitcher'),
                'game_id': game_id,
                # Check if OPPONENT's lineup specifically has 9 batters
                'confirmed': len(opponent_lineup) >= 9,
            }
        elif game_data.get('away_abbrev') == team_normalized:
            # Pitcher is away team, opponent is home
            opponent_lineup = game_data.get('home_lineup', [])
            return {
                'lineup': opponent_lineup,
                'team': game_data.get('home_abbrev'),
                'team_name': game_data.get('home_team'),
                'pitcher': game_data.get('away_pitcher'),
                'game_id': game_id,
                # Check if OPPONENT's lineup specifically has 9 batters
                'confirmed': len(opponent_lineup) >= 9,
            }

    return None


def is_lineup_confirmed(team_abbrev: str, game_date: str = None) -> bool:
    """
    Check if a team's lineup is confirmed for today.

    Args:
        team_abbrev: Team abbreviation
        game_date: Date string (defaults to today)

    Returns:
        True if lineup has 9 batters confirmed
    """
    lineup = get_lineup(team_abbrev, game_date)
    return lineup is not None and len(lineup) >= 9


def refresh_lineups(force: bool = False) -> Dict:
    """
    Refresh lineup cache.

    Args:
        force: Force refresh even if cache is fresh

    Returns:
        Lineup data dict
    """
    if not force:
        cache = load_lineup_cache()
        if cache:
            # Check if it's for today
            today = date.today().strftime("%Y-%m-%d")
            if cache.get('date') == today:
                logger.info("[Lineups] Cache is fresh, skipping refresh")
                return cache

    return fetch_todays_lineups()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("Force refreshing lineup cache...")
        data = refresh_lineups(force=True)
    else:
        print("Loading/refreshing lineup cache...")
        data = refresh_lineups()

    if data:
        print(f"\nDate: {data.get('date')}")
        print(f"Total games: {data.get('total_games', 0)}")
        print(f"Confirmed lineups: {data.get('confirmed_lineups', 0)}")

        # Show sample lineups
        print("\nSample lineups:")
        for game_id, game_data in list(data.get('games', {}).items())[:3]:
            home = game_data.get('home_abbrev', '?')
            away = game_data.get('away_abbrev', '?')
            confirmed = "✓" if game_data.get('lineup_confirmed') else "?"

            print(f"\n  {away} @ {home} [{confirmed}]")

            home_lineup = game_data.get('home_lineup', [])
            if home_lineup:
                print(f"    {home} lineup:")
                for batter in home_lineup[:3]:
                    print(f"      {batter['order']}. {batter['name']} ({batter['position']})")
                if len(home_lineup) > 3:
                    print(f"      ... and {len(home_lineup) - 3} more")
    else:
        print("Failed to load lineup data")
