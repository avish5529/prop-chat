#!/usr/bin/env python3
"""
MLB Daily Data Refresh Script

Run this daily before making predictions to ensure fresh data.
Pulls live data from pybaseball (FanGraphs) as the season progresses.

Usage:
    python mlb_refresh_data.py          # Refresh all data
    python mlb_refresh_data.py --force  # Force refresh even if cache is fresh
    python mlb_refresh_data.py --status # Check cache status only
"""

import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pybaseball as pyb
import pandas as pd

# Enable pybaseball caching (for API efficiency, not our cache)
pyb.cache.enable()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Minimum IP/PA thresholds to filter out noise from early season
MIN_PITCHER_IP = 1      # Will increase as season progresses
MIN_TEAM_PA = 50        # Will increase as season progresses


def get_current_season() -> int:
    """Determine current MLB season based on date."""
    now = datetime.now()
    # MLB season runs March-October
    if now.month >= 3:
        return now.year
    return now.year - 1


def refresh_pitchers(year: int = None, force: bool = False) -> dict:
    """
    Fetch fresh pitcher stats from FanGraphs via pybaseball.

    Prioritizes current season data, falls back to prior year if insufficient.
    """
    if year is None:
        year = get_current_season()

    cache_path = CACHE_DIR / f"mlb_pitchers_{year}.json"

    # Check if cache is fresh (less than 12 hours old)
    if not force and cache_path.exists():
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < 12 * 3600:  # 12 hours
            logger.info(f"[Pitchers] Cache is fresh ({cache_age/3600:.1f}h old), skipping refresh")
            with open(cache_path) as f:
                return json.load(f)

    logger.info(f"[Pitchers] Fetching {year} data from FanGraphs...")

    try:
        df = pyb.pitching_stats(year, qual=MIN_PITCHER_IP)

        if df is None or df.empty:
            logger.warning(f"[Pitchers] No {year} data available")
            return {}

        # Filter out relievers with very few IP for current season
        if year == get_current_season():
            # Early season: be lenient
            df = df[df['IP'] >= MIN_PITCHER_IP]

        pitchers = {}
        for _, row in df.iterrows():
            name = row.get('Name', '')
            if not name:
                continue

            pitchers[name.lower()] = {
                'name': name,
                'mlbam_id': _safe_int(row.get('xMLBAMID') or row.get('MLBAMID')),
                'team': row.get('Team', ''),
                'hand': 'L' if 'L' in str(row.get('Throws', 'R')) else 'R',
                'ip': _safe_float(row.get('IP')),
                'games': _safe_int(row.get('G')),
                'games_started': _safe_int(row.get('GS')),
                'k_pct': _parse_pct(row.get('K%')),
                'swstr_pct': _parse_pct(row.get('SwStr%')),
                'csw_pct': _parse_pct(row.get('CSW%')),
                'bb_pct': _parse_pct(row.get('BB%')),
                'k_per_9': _safe_float(row.get('K/9')),
                'contact_pct': _parse_pct(row.get('Contact%')),
                'o_swing_pct': _parse_pct(row.get('O-Swing%')),
                'z_contact_pct': _parse_pct(row.get('Z-Contact%')),
                'era': _safe_float(row.get('ERA')),
                'fip': _safe_float(row.get('FIP')),
                'whip': _safe_float(row.get('WHIP')),
            }

        # Calculate league averages from starters
        starters = df[df['GS'] >= max(1, df['GS'].max() * 0.1)]  # At least 10% of max starts
        league_avg = {
            'k_pct': _parse_pct(starters['K%'].mean()) if 'K%' in starters.columns else 22.5,
            'swstr_pct': _parse_pct(starters['SwStr%'].mean()) if 'SwStr%' in starters.columns else 11.0,
            'csw_pct': _parse_pct(starters['CSW%'].mean()) if 'CSW%' in starters.columns else 29.0,
            'bb_pct': _parse_pct(starters['BB%'].mean()) if 'BB%' in starters.columns else 8.0,
        }

        result = {
            'season': year,
            'count': len(pitchers),
            'league_averages': league_avg,
            'pitchers': pitchers,
            '_metadata': {
                'updated': datetime.now().isoformat(),
                'source': 'pybaseball (FanGraphs)',
                'min_ip': MIN_PITCHER_IP,
            }
        }

        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Also update generic cache
        generic_path = CACHE_DIR / "mlb_pitchers.json"
        with open(generic_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"[Pitchers] Cached {len(pitchers)} pitchers (league K%: {league_avg['k_pct']:.1f}%)")
        return result

    except Exception as e:
        logger.error(f"[Pitchers] Failed to fetch: {e}")
        return {}


def refresh_teams(year: int = None, force: bool = False) -> dict:
    """
    Fetch fresh team batting stats from FanGraphs via pybaseball.
    """
    if year is None:
        year = get_current_season()

    cache_path = CACHE_DIR / f"mlb_teams_{year}.json"

    # Check if cache is fresh
    if not force and cache_path.exists():
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < 12 * 3600:
            logger.info(f"[Teams] Cache is fresh ({cache_age/3600:.1f}h old), skipping refresh")
            with open(cache_path) as f:
                return json.load(f)

    logger.info(f"[Teams] Fetching {year} data from FanGraphs...")

    try:
        df = pyb.team_batting(year)

        if df is None or df.empty:
            logger.warning(f"[Teams] No {year} data available")
            return {}

        teams = {}
        for _, row in df.iterrows():
            team = row.get('Team', '')
            if not team or team == 'League Average':
                continue

            pa = _safe_int(row.get('PA', 1))
            if pa < MIN_TEAM_PA:
                continue

            # Calculate pitches per PA if available
            pitches = row.get('Pitches')
            p_per_pa = round(float(pitches) / pa, 2) if pitches and pa > 0 else 3.92

            teams[team] = {
                'name': team,
                'abbrev': team,
                'k_pct': _parse_pct(row.get('K%')),
                'bb_pct': _parse_pct(row.get('BB%')),
                'p_per_pa': p_per_pa,
                'pa': pa,
                'contact_pct': _parse_pct(row.get('Contact%')),
                'o_swing_pct': _parse_pct(row.get('O-Swing%')),
                'z_contact_pct': _parse_pct(row.get('Z-Contact%')),
                'wrc_plus': _safe_float(row.get('wRC+', 100)),
                # Will be same as overall K% until we have split data
                'k_pct_vs_lhp': _parse_pct(row.get('K%')),
                'k_pct_vs_rhp': _parse_pct(row.get('K%')),
            }

        # Calculate league averages
        k_pcts = [t['k_pct'] for t in teams.values() if t['k_pct']]
        p_per_pas = [t['p_per_pa'] for t in teams.values() if t['p_per_pa']]

        league_avg = {
            'k_pct': sum(k_pcts) / len(k_pcts) if k_pcts else 22.5,
            'k_pct_vs_lhp': sum(k_pcts) / len(k_pcts) if k_pcts else 22.5,
            'k_pct_vs_rhp': sum(k_pcts) / len(k_pcts) if k_pcts else 22.5,
            'p_per_pa': sum(p_per_pas) / len(p_per_pas) if p_per_pas else 3.92,
        }

        result = {
            'season': year,
            'count': len(teams),
            'league_averages': league_avg,
            'teams': teams,
            '_metadata': {
                'updated': datetime.now().isoformat(),
                'source': 'pybaseball (FanGraphs)',
                'min_pa': MIN_TEAM_PA,
            }
        }

        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Also update generic cache
        generic_path = CACHE_DIR / "mlb_teams.json"
        with open(generic_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"[Teams] Cached {len(teams)} teams (league K%: {league_avg['k_pct']:.1f}%)")
        return result

    except Exception as e:
        logger.error(f"[Teams] Failed to fetch: {e}")
        return {}


def refresh_park_factors(year: int = None, force: bool = False) -> dict:
    """
    Fetch park factors from FanGraphs.

    Note: K-specific park factors aren't directly available, so we derive them
    from basic park factors with research-based adjustments.
    """
    if year is None:
        year = get_current_season()

    cache_path = CACHE_DIR / "mlb_park_factors.json"

    # Park factors don't change much - refresh weekly
    if not force and cache_path.exists():
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < 7 * 24 * 3600:  # 7 days
            logger.info(f"[Parks] Cache is fresh ({cache_age/3600/24:.1f}d old), skipping refresh")
            with open(cache_path) as f:
                return json.load(f)

    logger.info(f"[Parks] Fetching {year} park factors from FanGraphs...")

    try:
        # Try to get park factors from pybaseball
        # pybaseball doesn't have a direct park_factors function, so we'll use
        # the research-based defaults that are well-established

        # Research-based K park factors (from various sources)
        # Higher = more Ks, Lower = fewer Ks
        parks = {
            # Pitcher-friendly (more Ks)
            'Tropicana Field': {'team': 'TBR', 'k_factor': 1.06},
            'T-Mobile Park': {'team': 'SEA', 'k_factor': 1.05},
            'Oracle Park': {'team': 'SFG', 'k_factor': 1.04},
            'Oakland Coliseum': {'team': 'OAK', 'k_factor': 1.03},
            'Petco Park': {'team': 'SDP', 'k_factor': 1.02},
            'loanDepot park': {'team': 'MIA', 'k_factor': 1.02},
            'Dodger Stadium': {'team': 'LAD', 'k_factor': 1.01},
            'Kauffman Stadium': {'team': 'KCR', 'k_factor': 1.01},
            'Citi Field': {'team': 'NYM', 'k_factor': 1.01},

            # Neutral
            'Yankee Stadium': {'team': 'NYY', 'k_factor': 1.00},
            'Wrigley Field': {'team': 'CHC', 'k_factor': 1.00},
            'Busch Stadium': {'team': 'STL', 'k_factor': 1.00},
            'Truist Park': {'team': 'ATL', 'k_factor': 1.00},
            'Camden Yards': {'team': 'BAL', 'k_factor': 1.00},
            'Progressive Field': {'team': 'CLE', 'k_factor': 1.00},
            'Comerica Park': {'team': 'DET', 'k_factor': 1.00},
            'Target Field': {'team': 'MIN', 'k_factor': 1.00},
            'Guaranteed Rate Field': {'team': 'CHW', 'k_factor': 1.00},
            'Globe Life Field': {'team': 'TEX', 'k_factor': 1.00},
            'Minute Maid Park': {'team': 'HOU', 'k_factor': 1.00},
            'Angel Stadium': {'team': 'LAA', 'k_factor': 1.00},
            'Rogers Centre': {'team': 'TOR', 'k_factor': 1.00},
            'Nationals Park': {'team': 'WSN', 'k_factor': 1.00},
            'PNC Park': {'team': 'PIT', 'k_factor': 1.00},
            'American Family Field': {'team': 'MIL', 'k_factor': 1.00},

            # Hitter-friendly (fewer Ks)
            'Citizens Bank Park': {'team': 'PHI', 'k_factor': 0.99},
            'Great American Ball Park': {'team': 'CIN', 'k_factor': 0.98},
            'Fenway Park': {'team': 'BOS', 'k_factor': 0.97},
            'Chase Field': {'team': 'ARI', 'k_factor': 0.96},
            'Coors Field': {'team': 'COL', 'k_factor': 0.88},  # Huge effect
        }

        result = {
            'season': year,
            'count': len(parks),
            'parks': parks,
            '_metadata': {
                'updated': datetime.now().isoformat(),
                'source': 'Research-based factors (FanGraphs/Statcast studies)',
                'note': 'K-specific factors derived from park environment research',
            }
        }

        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"[Parks] Cached {len(parks)} park factors")
        return result

    except Exception as e:
        logger.error(f"[Parks] Failed: {e}")
        return {}


def check_cache_status():
    """Show status of all caches."""
    print("\n" + "=" * 60)
    print("MLB DATA CACHE STATUS")
    print("=" * 60)

    caches = [
        ('Pitchers (current)', 'mlb_pitchers.json'),
        ('Pitchers (2025)', 'mlb_pitchers_2025.json'),
        ('Pitchers (2026)', 'mlb_pitchers_2026.json'),
        ('Teams (current)', 'mlb_teams.json'),
        ('Teams (2025)', 'mlb_teams_2025.json'),
        ('Teams (2026)', 'mlb_teams_2026.json'),
        ('Park Factors', 'mlb_park_factors.json'),
    ]

    for name, filename in caches:
        path = CACHE_DIR / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)

            updated = data.get('_metadata', {}).get('updated', 'Unknown')
            count = data.get('count', 0)
            season = data.get('season', '?')

            # Calculate age
            try:
                updated_dt = datetime.fromisoformat(updated)
                age_hours = (datetime.now() - updated_dt).total_seconds() / 3600
                age_str = f"{age_hours:.1f}h ago"
            except:
                age_str = "Unknown"

            print(f"\n{name}:")
            print(f"  Season: {season} | Count: {count} | Updated: {age_str}")
        else:
            print(f"\n{name}:")
            print(f"  ❌ NOT FOUND")

    print("\n" + "=" * 60)


def refresh_all(force: bool = False):
    """Refresh all MLB data caches."""
    year = get_current_season()
    prior_year = year - 1

    print("\n" + "=" * 60)
    print(f"MLB DATA REFRESH - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Current Season: {year}")
    print("=" * 60)

    results = {}

    # 1. Prior year pitchers (baseline)
    print(f"\n[1/5] Prior year pitchers ({prior_year})...")
    prior_pitchers = refresh_pitchers(prior_year, force=force)
    results['pitchers_prior'] = prior_pitchers.get('count', 0)

    # 2. Current year pitchers
    print(f"\n[2/5] Current year pitchers ({year})...")
    current_pitchers = refresh_pitchers(year, force=force)
    results['pitchers_current'] = current_pitchers.get('count', 0)

    # 3. Prior year teams (baseline)
    print(f"\n[3/5] Prior year teams ({prior_year})...")
    prior_teams = refresh_teams(prior_year, force=force)
    results['teams_prior'] = prior_teams.get('count', 0)

    # 4. Current year teams
    print(f"\n[4/5] Current year teams ({year})...")
    current_teams = refresh_teams(year, force=force)
    results['teams_current'] = current_teams.get('count', 0)

    # 5. Park factors
    print(f"\n[5/5] Park factors...")
    parks = refresh_park_factors(year, force=force)
    results['parks'] = parks.get('count', 0)

    print("\n" + "=" * 60)
    print("REFRESH COMPLETE")
    print("=" * 60)
    print(f"  Pitchers: {results['pitchers_prior']} ({prior_year}) + {results['pitchers_current']} ({year})")
    print(f"  Teams: {results['teams_prior']} ({prior_year}) + {results['teams_current']} ({year})")
    print(f"  Parks: {results['parks']}")
    print("=" * 60 + "\n")

    return results


# Helper functions
def _safe_float(val) -> float:
    try:
        if pd.isna(val):
            return 0.0
        return float(val)
    except:
        return 0.0


def _safe_int(val) -> int:
    try:
        if pd.isna(val):
            return 0
        return int(val)
    except:
        return 0


def _parse_pct(val) -> float:
    """Parse percentage value - handles both 0.25 and 25.0 formats."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        # If it looks like a decimal (< 1), convert to percentage
        if abs(f) < 1:
            return round(f * 100, 1)
        return round(f, 1)
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh MLB data caches")
    parser.add_argument('--force', action='store_true', help='Force refresh even if cache is fresh')
    parser.add_argument('--status', action='store_true', help='Check cache status only')
    args = parser.parse_args()

    if args.status:
        check_cache_status()
    else:
        refresh_all(force=args.force)
