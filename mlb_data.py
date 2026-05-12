#!/usr/bin/env python3
"""
MLB Data Module - Fully Automated Data Fetching & Caching

All data is fetched from free sources:
- FanGraphs (via pybaseball): Pitcher stats, team batting
- Baseball Reference (scrape): Team splits vs LHP/RHP
- Baseball Savant (via pybaseball): Pitcher game logs
- MLB Stats API (via statsapi): Schedule, lineups, umpires
- FanGraphs Guts (scrape): Park factors

No hardcoded data. Run refresh_all() daily before first game.
"""

import json
import logging
import os
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

import pandas as pd
import requests
import pybaseball as pyb
import statsapi

# Enable pybaseball caching
pyb.cache.enable()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILES = {
    'pitchers': CACHE_DIR / 'mlb_pitchers.json',
    'pitchers_2025': CACHE_DIR / 'mlb_pitchers_2025.json',  # Prior year baseline
    'pitchers_2026': CACHE_DIR / 'mlb_pitchers_2026.json',  # Current year
    'pitcher_logs': CACHE_DIR / 'mlb_pitcher_logs.json',
    'teams': CACHE_DIR / 'mlb_teams.json',
    'teams_2025': CACHE_DIR / 'mlb_teams_2025.json',        # Prior year baseline
    'teams_2026': CACHE_DIR / 'mlb_teams_2026.json',        # Current year
    'park_factors': CACHE_DIR / 'mlb_park_factors.json',
    'umpires': CACHE_DIR / 'mlb_umpires.json',
    'schedule': CACHE_DIR / 'mlb_schedule.json',
}

# TTL in hours
CACHE_TTL = {
    'pitchers': 24,
    'pitchers_2025': 8760,  # 1 year (baseline doesn't change)
    'pitchers_2026': 24,
    'pitcher_logs': 24,
    'teams': 24,
    'teams_2025': 8760,     # 1 year (baseline doesn't change)
    'teams_2026': 24,
    'park_factors': 168,    # 7 days
    'umpires': 168,         # 7 days
    'schedule': 0.25,       # 15 minutes
}

# ============================================================================
# MARCEL BLENDING CONFIGURATION
# ============================================================================

# Marcel weights: most recent year = 3, prior year = 2 (for pitchers)
MARCEL_WEIGHTS = {
    'current_year': 3,
    'prior_year': 2,
}

# Regression constant in outs (~44 IP for pitchers)
MARCEL_REGRESSION_OUTS = 134

# Stabilization points (batters faced) for different stats
STABILIZATION_BF = {
    'k_pct': 70,
    'bb_pct': 170,
    'swstr_pct': 100,
    'csw_pct': 100,
}

# ============================================================================
# CACHE MANAGER
# ============================================================================

def load_cache(cache_name: str) -> Optional[Dict]:
    """Load cache file if it exists and is not stale."""
    filepath = CACHE_FILES.get(cache_name)
    if not filepath or not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load cache {cache_name}: {e}")
        return None


def save_cache(cache_name: str, data: Dict) -> bool:
    """Save data to cache file with metadata."""
    filepath = CACHE_FILES.get(cache_name)
    if not filepath:
        return False

    # Add metadata
    data['_metadata'] = {
        'updated': datetime.now().isoformat(),
        'cache_name': cache_name,
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[Cache] Saved {cache_name} to {filepath}")
        return True
    except IOError as e:
        logger.error(f"Failed to save cache {cache_name}: {e}")
        return False


def is_cache_stale(cache_name: str) -> bool:
    """Check if cache needs refresh based on TTL."""
    cache = load_cache(cache_name)
    if not cache:
        return True

    metadata = cache.get('_metadata', {})
    updated_str = metadata.get('updated')
    if not updated_str:
        return True

    try:
        updated = datetime.fromisoformat(updated_str)
        ttl_hours = CACHE_TTL.get(cache_name, 24)
        age_hours = (datetime.now() - updated).total_seconds() / 3600
        return age_hours > ttl_hours
    except (ValueError, TypeError):
        return True


def get_cache_status() -> Dict[str, Any]:
    """Get status of all caches for health check."""
    status = {}
    for name, filepath in CACHE_FILES.items():
        if filepath.exists():
            cache = load_cache(name)
            metadata = cache.get('_metadata', {}) if cache else {}
            status[name] = {
                'exists': True,
                'updated': metadata.get('updated'),
                'stale': is_cache_stale(name),
                'ttl_hours': CACHE_TTL.get(name),
            }
        else:
            status[name] = {'exists': False, 'stale': True}
    return status


# ============================================================================
# MARCEL BLENDING FUNCTIONS
# ============================================================================

def blend_pitcher_stats(pitcher_2025: Dict, pitcher_2026: Dict, league_avg: Dict) -> Dict:
    """
    Blend two seasons of pitcher stats using Marcel methodology.

    Marcel for pitchers: weights 3/2/1 (most recent = 3)
    Since we only have 2 years, we use 3/2 weighting.

    Formula:
    1. Calculate weighted average of stats
    2. Regress toward league mean based on total sample size
    3. Apply reliability = total_outs / (total_outs + 134)

    Returns blended pitcher dict with combined stats.
    """
    if not pitcher_2025 and not pitcher_2026:
        return {}

    # If only one year exists, use it with regression
    if not pitcher_2026:
        return _regress_single_season(pitcher_2025, league_avg, is_prior=True)

    if not pitcher_2025:
        return _regress_single_season(pitcher_2026, league_avg, is_prior=False)

    # Both years exist - blend them
    ip_2025 = pitcher_2025.get('ip', 0) or 0
    ip_2026 = pitcher_2026.get('ip', 0) or 0

    # Convert IP to outs (IP * 3)
    outs_2025 = ip_2025 * 3
    outs_2026 = ip_2026 * 3

    # Marcel weights: current year = 3, prior year = 2
    weight_2026 = MARCEL_WEIGHTS['current_year']
    weight_2025 = MARCEL_WEIGHTS['prior_year']

    # Weighted outs (for determining reliability)
    weighted_outs = (outs_2026 * weight_2026) + (outs_2025 * weight_2025)
    total_weight = (weight_2026 * ip_2026) + (weight_2025 * ip_2025)

    # Stats to blend
    stats_to_blend = ['k_pct', 'swstr_pct', 'csw_pct', 'bb_pct', 'k_per_9', 'contact_pct']

    blended = pitcher_2026.copy()  # Start with current year data
    blended['prior_year_ip'] = ip_2025
    blended['current_year_ip'] = ip_2026
    blended['blend_method'] = 'marcel'

    for stat in stats_to_blend:
        val_2025 = pitcher_2025.get(stat)
        val_2026 = pitcher_2026.get(stat)
        league_val = league_avg.get(stat)

        # Ensure we have a league value
        if league_val is None:
            league_val = _get_default_league_avg(stat)

        if val_2026 is not None and val_2025 is not None and total_weight > 0:
            # Weighted average
            weighted_avg = ((val_2026 * weight_2026 * ip_2026) +
                           (val_2025 * weight_2025 * ip_2025)) / total_weight

            # Reliability: how much to trust observed vs league avg
            # reliability = outs / (outs + regression_constant)
            reliability = weighted_outs / (weighted_outs + MARCEL_REGRESSION_OUTS)

            # Regress toward league mean
            blended[stat] = (weighted_avg * reliability) + (league_val * (1 - reliability))

        elif val_2026 is not None:
            # Only current year - regress more heavily
            reliability = outs_2026 / (outs_2026 + MARCEL_REGRESSION_OUTS) if outs_2026 > 0 else 0
            blended[stat] = (val_2026 * reliability) + (league_val * (1 - reliability))

        elif val_2025 is not None:
            # Only prior year - use it but regress
            reliability = outs_2025 / (outs_2025 + MARCEL_REGRESSION_OUTS) if outs_2025 > 0 else 0
            # Additional penalty for being prior year data
            reliability *= 0.8
            blended[stat] = (val_2025 * reliability) + (league_val * (1 - reliability))

        else:
            # No data for either year - use league average
            blended[stat] = league_val

    # Calculate blend weights for transparency
    if ip_2026 + ip_2025 > 0:
        blended['weight_2026'] = round((weight_2026 * ip_2026) / total_weight, 3) if total_weight > 0 else 0
        blended['weight_2025'] = round((weight_2025 * ip_2025) / total_weight, 3) if total_weight > 0 else 0
        blended['reliability'] = round(weighted_outs / (weighted_outs + MARCEL_REGRESSION_OUTS), 3)

    # Combined IP for reference
    blended['ip'] = ip_2026  # Current year IP for projection purposes
    blended['ip_combined'] = ip_2025 + ip_2026

    logger.debug(f"[Blend] {blended.get('name')}: 2025={ip_2025:.0f}IP, 2026={ip_2026:.0f}IP, "
                f"weights={blended.get('weight_2025', 0):.0%}/{blended.get('weight_2026', 0):.0%}, "
                f"reliability={blended.get('reliability', 0):.0%}")

    return blended


def _regress_single_season(pitcher: Dict, league_avg: Dict, is_prior: bool = False) -> Dict:
    """Regress a single season toward league average."""
    if not pitcher:
        return {}

    result = pitcher.copy()
    ip = pitcher.get('ip', 0) or 0
    outs = ip * 3

    # If prior year only, apply additional penalty
    if is_prior:
        outs *= 0.67  # Treat as if 2/3 the sample size

    reliability = outs / (outs + MARCEL_REGRESSION_OUTS) if outs > 0 else 0
    result['reliability'] = round(reliability, 3)
    result['blend_method'] = 'prior_only' if is_prior else 'current_only'

    stats_to_regress = ['k_pct', 'swstr_pct', 'csw_pct', 'bb_pct', 'k_per_9', 'contact_pct']

    for stat in stats_to_regress:
        val = pitcher.get(stat)
        league_val = league_avg.get(stat)

        # Handle None values - fall back to defaults
        if league_val is None:
            league_val = _get_default_league_avg(stat)

        if val is not None and league_val is not None:
            result[stat] = (val * reliability) + (league_val * (1 - reliability))
        elif league_val is not None:
            # No observed value, use league average
            result[stat] = league_val

    return result


def _get_default_league_avg(stat: str) -> float:
    """Default league averages if not available."""
    defaults = {
        'k_pct': 22.5,
        'swstr_pct': 11.0,
        'csw_pct': 29.0,
        'bb_pct': 8.0,
        'k_per_9': 9.0,
        'contact_pct': 78.0,
    }
    return defaults.get(stat, 0)


def get_blended_pitcher(name: str) -> Optional[Dict]:
    """
    Get pitcher stats blended across 2025 and 2026 using Marcel methodology.

    This is the primary function to call for projections.
    """
    # Load both years
    cache_2025 = load_cache('pitchers_2025')
    cache_2026 = load_cache('pitchers_2026')

    # Fall back to generic cache if year-specific not available
    if not cache_2025:
        cache_2025 = load_cache('pitchers')
    if not cache_2026:
        cache_2026 = load_cache('pitchers')

    if not cache_2025 and not cache_2026:
        return None

    name_lower = name.lower().strip()

    # Find pitcher in each cache
    pitcher_2025 = _find_pitcher_in_cache(name_lower, cache_2025)
    pitcher_2026 = _find_pitcher_in_cache(name_lower, cache_2026)

    # Get league averages (prefer current year)
    league_avg = {}
    if cache_2026 and 'league_averages' in cache_2026:
        league_avg = cache_2026['league_averages']
    elif cache_2025 and 'league_averages' in cache_2025:
        league_avg = cache_2025['league_averages']

    # Blend the stats
    blended = blend_pitcher_stats(pitcher_2025, pitcher_2026, league_avg)

    if not blended:
        return None

    # Add game logs from current season if available
    if blended.get('mlbam_id'):
        logs = get_pitcher_logs(blended['mlbam_id'])
        if logs:
            blended.update(logs)

    return blended


def _find_pitcher_in_cache(name_lower: str, cache: Dict) -> Optional[Dict]:
    """Find a pitcher in a cache by name."""
    if not cache:
        return None

    pitchers = cache.get('pitchers', {})

    # Exact match
    if name_lower in pitchers:
        return pitchers[name_lower]

    # Partial match
    for key, val in pitchers.items():
        if name_lower in key:
            return val

    # Last name match
    for key, val in pitchers.items():
        if name_lower in key.split()[-1]:
            return val

    return None


def get_blend_info(pitcher_name: str) -> Dict:
    """
    Get detailed blending information for a pitcher (for debugging/transparency).

    Returns dict with:
    - 2025 stats
    - 2026 stats
    - blended stats
    - weights and reliability
    """
    cache_2025 = load_cache('pitchers_2025') or load_cache('pitchers')
    cache_2026 = load_cache('pitchers_2026') or load_cache('pitchers')

    name_lower = pitcher_name.lower().strip()

    pitcher_2025 = _find_pitcher_in_cache(name_lower, cache_2025)
    pitcher_2026 = _find_pitcher_in_cache(name_lower, cache_2026)

    league_avg = {}
    if cache_2026 and 'league_averages' in cache_2026:
        league_avg = cache_2026['league_averages']
    elif cache_2025 and 'league_averages' in cache_2025:
        league_avg = cache_2025['league_averages']

    blended = blend_pitcher_stats(pitcher_2025, pitcher_2026, league_avg)

    return {
        'name': pitcher_name,
        'stats_2025': {
            'ip': pitcher_2025.get('ip') if pitcher_2025 else None,
            'k_pct': pitcher_2025.get('k_pct') if pitcher_2025 else None,
            'swstr_pct': pitcher_2025.get('swstr_pct') if pitcher_2025 else None,
            'csw_pct': pitcher_2025.get('csw_pct') if pitcher_2025 else None,
        },
        'stats_2026': {
            'ip': pitcher_2026.get('ip') if pitcher_2026 else None,
            'k_pct': pitcher_2026.get('k_pct') if pitcher_2026 else None,
            'swstr_pct': pitcher_2026.get('swstr_pct') if pitcher_2026 else None,
            'csw_pct': pitcher_2026.get('csw_pct') if pitcher_2026 else None,
        },
        'blended': {
            'k_pct': blended.get('k_pct') if blended else None,
            'swstr_pct': blended.get('swstr_pct') if blended else None,
            'csw_pct': blended.get('csw_pct') if blended else None,
            'weight_2025': blended.get('weight_2025') if blended else None,
            'weight_2026': blended.get('weight_2026') if blended else None,
            'reliability': blended.get('reliability') if blended else None,
            'blend_method': blended.get('blend_method') if blended else None,
        },
        'league_averages': league_avg,
    }


# ============================================================================
# PITCHER STATS FETCHER (FanGraphs via pybaseball)
# ============================================================================

def fetch_pitchers(year: int = None, min_ip: int = 1, cache_key: str = None) -> Dict[str, Any]:
    """
    Fetch all pitcher stats from FanGraphs.

    Args:
        year: Season year to fetch
        min_ip: Minimum IP to include
        cache_key: Cache key to save to (e.g., 'pitchers_2025', 'pitchers_2026')

    Returns dict with pitchers keyed by lowercase name and league averages.
    """
    if year is None:
        year = datetime.now().year

    # Determine cache key based on year if not specified
    if cache_key is None:
        cache_key = f'pitchers_{year}' if f'pitchers_{year}' in CACHE_FILES else 'pitchers'

    logger.info(f"[Pitchers] Fetching from FanGraphs for {year} (cache: {cache_key})...")

    actual_year = year
    df = None

    try:
        df = pyb.pitching_stats(year, qual=min_ip)
        if df is None or df.empty:
            raise ValueError("No data returned")
    except Exception as e:
        logger.warning(f"[Pitchers] {year} failed: {e}, trying {year-1}")
        try:
            df = pyb.pitching_stats(year - 1, qual=min_ip)
            actual_year = year - 1
        except Exception as e2:
            logger.error(f"[Pitchers] Failed to fetch data: {e2}")
            return {}

    if df is None or df.empty:
        logger.error("[Pitchers] No data available")
        return {}

    # Parse pitchers
    pitchers = {}
    for _, row in df.iterrows():
        name = row.get('Name', '')
        if not name:
            continue

        # Get MLBAM ID if available
        mlbam_id = row.get('xMLBAMID') or row.get('MLBAMID') or row.get('playerid')

        pitchers[name.lower()] = {
            'name': name,
            'mlbam_id': int(mlbam_id) if pd.notna(mlbam_id) else None,
            'team': row.get('Team', ''),
            'hand': 'L' if 'L' in str(row.get('Throws', 'R')) else 'R',
            'ip': float(row.get('IP', 0)),
            'games': int(row.get('G', 0)),
            'games_started': int(row.get('GS', 0)),

            # Key metrics
            'k_pct': _parse_pct(row.get('K%')),
            'swstr_pct': _parse_pct(row.get('SwStr%')),
            'csw_pct': _parse_pct(row.get('CSW%')),

            # Secondary metrics
            'bb_pct': _parse_pct(row.get('BB%')),
            'k_per_9': float(row.get('K/9', 0)),
            'contact_pct': _parse_pct(row.get('Contact%')),
            'o_swing_pct': _parse_pct(row.get('O-Swing%')),
            'z_contact_pct': _parse_pct(row.get('Z-Contact%')),

            # Quality metrics
            'era': float(row.get('ERA', 0)) if pd.notna(row.get('ERA')) else None,
            'fip': float(row.get('FIP', 0)) if pd.notna(row.get('FIP')) else None,
            'whip': float(row.get('WHIP', 0)) if pd.notna(row.get('WHIP')) else None,
        }

    # Calculate league averages from the data
    starters = df[df['GS'] >= 3]  # At least 3 starts
    league_averages = {
        'k_pct': _parse_pct(starters['K%'].mean()) if 'K%' in starters.columns else 22.5,
        'swstr_pct': _parse_pct(starters['SwStr%'].mean()) if 'SwStr%' in starters.columns else 11.0,
        'csw_pct': _parse_pct(starters['CSW%'].mean()) if 'CSW%' in starters.columns else 29.0,
        'bb_pct': _parse_pct(starters['BB%'].mean()) if 'BB%' in starters.columns else 8.0,
    }

    result = {
        'season': actual_year,
        'count': len(pitchers),
        'league_averages': league_averages,
        'pitchers': pitchers,
    }

    save_cache(cache_key, result)
    logger.info(f"[Pitchers] Cached {len(pitchers)} pitchers to {cache_key}, league K%: {league_averages['k_pct']:.1f}%")

    return result


def _parse_pct(val) -> Optional[float]:
    """Parse percentage value from FanGraphs."""
    if val is None or pd.isna(val):
        return None
    try:
        if isinstance(val, str):
            return float(val.strip('%').strip())
        # If decimal (0.253), convert to percentage
        if isinstance(val, (int, float)):
            return float(val) * 100 if abs(val) < 1 else float(val)
    except (ValueError, TypeError):
        return None
    return None


# ============================================================================
# PITCHER GAME LOGS FETCHER (Statcast via pybaseball)
# ============================================================================

def fetch_pitcher_logs(days_back: int = 45) -> Dict[str, Any]:
    """
    Fetch recent game logs for all starting pitchers.

    Uses Statcast pitch-level data aggregated to game level.
    """
    logger.info(f"[PitcherLogs] Fetching last {days_back} days from Statcast...")

    # Load pitcher cache to get MLBAM IDs
    pitcher_cache = load_cache('pitchers')
    if not pitcher_cache:
        logger.error("[PitcherLogs] Pitcher cache not found. Run fetch_pitchers() first.")
        return {}

    # Get starters only (at least 1 start)
    starters = {k: v for k, v in pitcher_cache.get('pitchers', {}).items()
                if v.get('games_started', 0) >= 1 and v.get('mlbam_id')}

    logger.info(f"[PitcherLogs] Processing {len(starters)} starting pitchers...")

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    pitcher_logs = {}
    processed = 0

    for name, pitcher in starters.items():
        mlbam_id = pitcher.get('mlbam_id')
        if not mlbam_id:
            continue

        try:
            # Fetch pitch-level data for this pitcher
            data = pyb.statcast_pitcher(
                start_dt=start_date.strftime('%Y-%m-%d'),
                end_dt=end_date.strftime('%Y-%m-%d'),
                player_id=mlbam_id
            )

            if data is None or data.empty:
                continue

            # Aggregate to game level
            game_logs = _aggregate_pitcher_games(data)

            if not game_logs:
                continue

            # Calculate averages from last 5 starts
            last_5 = game_logs[:5]

            pitcher_logs[str(mlbam_id)] = {
                'name': pitcher['name'],
                'mlbam_id': mlbam_id,
                'last_starts': game_logs[:10],  # Keep last 10
                'last_5_avg_ks': sum(g['strikeouts'] for g in last_5) / len(last_5) if last_5 else None,
                'last_5_avg_bf': sum(g['batters_faced'] for g in last_5) / len(last_5) if last_5 else None,
                'last_5_avg_pitches': sum(g['pitches'] for g in last_5) / len(last_5) if last_5 else None,
                'last_5_k_rate': sum(g['strikeouts'] for g in last_5) / sum(g['batters_faced'] for g in last_5) if last_5 and sum(g['batters_faced'] for g in last_5) > 0 else None,
                'last_10_k_std': _calculate_std([g['strikeouts'] for g in game_logs[:10]]) if len(game_logs) >= 3 else None,
                'last_10_k_avg': sum(g['strikeouts'] for g in game_logs[:10]) / len(game_logs[:10]) if game_logs else None,
            }

            processed += 1
            if processed % 25 == 0:
                logger.info(f"[PitcherLogs] Processed {processed}/{len(starters)} pitchers...")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            logger.warning(f"[PitcherLogs] Failed for {pitcher['name']}: {e}")
            continue

    result = {
        'days_back': days_back,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'count': len(pitcher_logs),
        'pitchers': pitcher_logs,
    }

    save_cache('pitcher_logs', result)
    logger.info(f"[PitcherLogs] Cached logs for {len(pitcher_logs)} pitchers")

    return result


def _aggregate_pitcher_games(df: pd.DataFrame) -> List[Dict]:
    """Aggregate pitch-level Statcast data to game-level stats."""
    if df.empty:
        return []

    games = []

    # Group by game date
    for game_date, game_data in df.groupby('game_date'):
        # Count strikeouts (events column contains 'strikeout')
        strikeouts = len(game_data[game_data['events'] == 'strikeout'])

        # Count unique batters faced
        batters_faced = game_data['batter'].nunique()

        # Total pitches
        pitches = len(game_data)

        # Get opponent
        opponent = game_data['home_team'].iloc[0] if game_data['inning_topbot'].iloc[0] == 'Top' else game_data['away_team'].iloc[0]

        # Calculate innings (approximate from outs)
        # Each strikeout/groundout/flyout = 1 out, 3 outs = 1 inning
        outs = len(game_data[game_data['events'].isin(['strikeout', 'field_out', 'grounded_into_double_play', 'force_out', 'sac_fly', 'sac_bunt', 'fielders_choice_out', 'double_play', 'triple_play'])])
        innings = round(outs / 3, 1)

        games.append({
            'date': str(game_date)[:10],
            'opponent': opponent,
            'strikeouts': strikeouts,
            'batters_faced': batters_faced,
            'pitches': pitches,
            'innings': innings,
        })

    # Sort by date descending (most recent first)
    games.sort(key=lambda x: x['date'], reverse=True)

    return games


def _calculate_std(values: List[float]) -> Optional[float]:
    """Calculate standard deviation."""
    if not values or len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


# ============================================================================
# TEAM BATTING FETCHER (FanGraphs + Baseball Reference splits)
# ============================================================================

def fetch_teams(year: int = None) -> Dict[str, Any]:
    """
    Fetch team batting stats from FanGraphs and splits from Baseball Reference.
    """
    if year is None:
        year = datetime.now().year

    logger.info(f"[Teams] Fetching team batting from FanGraphs for {year}...")

    actual_year = year
    df = None

    try:
        df = pyb.team_batting(year)
        if df is None or df.empty:
            raise ValueError("No data returned")
    except Exception as e:
        logger.warning(f"[Teams] {year} failed: {e}, trying {year-1}")
        try:
            df = pyb.team_batting(year - 1)
            actual_year = year - 1
        except Exception as e2:
            logger.error(f"[Teams] Failed to fetch data: {e2}")
            return {}

    if df is None or df.empty:
        logger.error("[Teams] No data available")
        return {}

    teams = {}
    for _, row in df.iterrows():
        team = row.get('Team', '')
        if not team or team == 'League Average':
            continue

        # Calculate P/PA if we have the data
        pa = int(row.get('PA', 1))
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
            'wrc_plus': float(row.get('wRC+', 100)) if pd.notna(row.get('wRC+')) else 100,
            # Splits will be added below
            'k_pct_vs_lhp': None,
            'k_pct_vs_rhp': None,
        }

    # Fetch splits from Baseball Reference
    logger.info("[Teams] Fetching splits from Baseball Reference...")
    splits = _fetch_team_splits(actual_year)

    # Merge splits into teams
    for team, data in teams.items():
        if team in splits:
            data['k_pct_vs_lhp'] = splits[team].get('k_pct_vs_lhp')
            data['k_pct_vs_rhp'] = splits[team].get('k_pct_vs_rhp')
        else:
            # Fallback to overall K%
            data['k_pct_vs_lhp'] = data['k_pct']
            data['k_pct_vs_rhp'] = data['k_pct']

    # Calculate league averages
    k_pcts = [t['k_pct'] for t in teams.values() if t['k_pct']]
    k_vs_lhp = [t['k_pct_vs_lhp'] for t in teams.values() if t['k_pct_vs_lhp']]
    k_vs_rhp = [t['k_pct_vs_rhp'] for t in teams.values() if t['k_pct_vs_rhp']]
    p_per_pas = [t['p_per_pa'] for t in teams.values() if t['p_per_pa']]

    league_averages = {
        'k_pct': sum(k_pcts) / len(k_pcts) if k_pcts else 22.5,
        'k_pct_vs_lhp': sum(k_vs_lhp) / len(k_vs_lhp) if k_vs_lhp else 22.5,
        'k_pct_vs_rhp': sum(k_vs_rhp) / len(k_vs_rhp) if k_vs_rhp else 22.5,
        'p_per_pa': sum(p_per_pas) / len(p_per_pas) if p_per_pas else 3.92,
    }

    result = {
        'season': actual_year,
        'count': len(teams),
        'league_averages': league_averages,
        'teams': teams,
    }

    save_cache('teams', result)
    logger.info(f"[Teams] Cached {len(teams)} teams with splits")

    return result


def _fetch_team_splits(year: int) -> Dict[str, Dict]:
    """Scrape team batting splits vs LHP/RHP from Baseball Reference."""
    splits = {}

    # Team abbreviation mapping for BBRef
    bbref_teams = {
        'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS', 'CHC': 'CHC',
        'CHW': 'CHW', 'CIN': 'CIN', 'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET',
        'HOU': 'HOU', 'KCR': 'KCR', 'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA',
        'MIL': 'MIL', 'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'OAK': 'OAK',
        'PHI': 'PHI', 'PIT': 'PIT', 'SDP': 'SDP', 'SFG': 'SFG', 'SEA': 'SEA',
        'STL': 'STL', 'TBR': 'TBR', 'TEX': 'TEX', 'TOR': 'TOR', 'WSN': 'WSN',
    }

    try:
        # vs LHP
        url_lhp = f"https://www.baseball-reference.com/leagues/majors/{year}-batting-splits.shtml"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        resp = requests.get(url_lhp, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[Splits] BBRef returned {resp.status_code}")
            return splits

        # Parse tables
        tables = pd.read_html(resp.text)

        # Find the platoon split tables
        for table in tables:
            if 'Split' in table.columns or 'Tm' in table.columns:
                # Look for vs LHP and vs RHP rows
                for _, row in table.iterrows():
                    split_type = str(row.get('Split', row.get('Tm', '')))
                    if 'vs LHP' in split_type or 'vs RHP' in split_type:
                        # This is league-wide data, we need team-specific
                        pass

        # Alternative: Fetch team-by-team (slower but more reliable)
        logger.info("[Splits] Fetching individual team splits...")

        for fg_abbrev, bbref_abbrev in bbref_teams.items():
            try:
                team_url = f"https://www.baseball-reference.com/teams/{bbref_abbrev}/{year}-batting.shtml"
                resp = requests.get(team_url, headers=headers, timeout=10)

                if resp.status_code != 200:
                    continue

                # Look for splits in the page
                if 'vs LHP' in resp.text and 'vs RHP' in resp.text:
                    tables = pd.read_html(resp.text)

                    for table in tables:
                        cols = [str(c).lower() for c in table.columns]
                        if 'split' in cols or any('so' in c for c in cols):
                            for _, row in table.iterrows():
                                split_name = str(row.iloc[0]) if len(row) > 0 else ''

                                if 'vs LHP' in split_name:
                                    so = row.get('SO', row.get('K', 0))
                                    pa = row.get('PA', row.get('AB', 1))
                                    if pa and int(pa) > 0:
                                        k_pct = round(float(so) / float(pa) * 100, 1)
                                        if fg_abbrev not in splits:
                                            splits[fg_abbrev] = {}
                                        splits[fg_abbrev]['k_pct_vs_lhp'] = k_pct

                                elif 'vs RHP' in split_name:
                                    so = row.get('SO', row.get('K', 0))
                                    pa = row.get('PA', row.get('AB', 1))
                                    if pa and int(pa) > 0:
                                        k_pct = round(float(so) / float(pa) * 100, 1)
                                        if fg_abbrev not in splits:
                                            splits[fg_abbrev] = {}
                                        splits[fg_abbrev]['k_pct_vs_rhp'] = k_pct

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.debug(f"[Splits] Failed for {fg_abbrev}: {e}")
                continue

    except Exception as e:
        logger.warning(f"[Splits] Failed to fetch splits: {e}")

    logger.info(f"[Splits] Got splits for {len(splits)} teams")
    return splits


# ============================================================================
# PARK FACTORS FETCHER (Baseball Savant Statcast)
# ============================================================================

def fetch_park_factors(year: int = None) -> Dict[str, Any]:
    """
    Fetch strikeout-specific park factors from Baseball Savant.
    Uses actual SO park factors, not derived from run environment.

    Note: Baseball Savant shows rolling 3-year data (e.g., 2023-2025).
    Early in the season, use prior year data until current year has enough games.
    """
    if year is None:
        year = datetime.now().year

    logger.info(f"[Parks] Fetching SO park factors from Baseball Savant...")

    try:
        from playwright.sync_api import sync_playwright
        import time as time_module

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Baseball Savant park factors page
            url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            time_module.sleep(3)  # Wait for table to load

            # Find the table
            table = page.query_selector("table")
            if not table:
                logger.warning("[Parks] No table found on Baseball Savant, using defaults")
                browser.close()
                return _get_default_park_factors()

            # Get headers to find SO column index
            headers = page.query_selector_all("table thead th")
            header_text = [h.inner_text().strip() for h in headers]

            # Find column indices
            so_index = None
            team_index = None
            venue_index = None
            for i, h in enumerate(header_text):
                if h == "SO":
                    so_index = i
                if h == "Team":
                    team_index = i
                if h == "Venue":
                    venue_index = i

            if so_index is None:
                logger.warning("[Parks] Could not find SO column, using defaults")
                browser.close()
                return _get_default_park_factors()

            # Extract data from rows
            rows = page.query_selector_all("table tbody tr")
            parks = {}

            for row in rows:
                cells = row.query_selector_all("td")
                if len(cells) > so_index:
                    team = cells[team_index].inner_text().strip() if team_index else ""
                    venue = cells[venue_index].inner_text().strip() if venue_index else ""
                    so_factor = cells[so_index].inner_text().strip()

                    try:
                        so_value = int(so_factor)
                        k_factor = so_value / 100.0  # Convert to multiplier (100 = 1.0)
                        parks[venue] = {
                            "team": team,
                            "so_factor": so_value,
                            "k_factor": round(k_factor, 3)
                        }
                    except ValueError:
                        pass  # Skip invalid values

            browser.close()

            if not parks:
                logger.warning("[Parks] No parks extracted, using defaults")
                return _get_default_park_factors()

            result = {
                'source': 'Baseball Savant Statcast',
                'season': '2023-2025',  # Rolling 3-year
                'fetched_at': datetime.now().isoformat(),
                'count': len(parks),
                'parks': parks,
            }

            save_cache('park_factors', result)
            logger.info(f"[Parks] Cached {len(parks)} SO park factors from Baseball Savant")

            return result

    except ImportError:
        logger.warning("[Parks] Playwright not available, using defaults")
        return _get_default_park_factors()
    except Exception as e:
        logger.warning(f"[Parks] Failed to fetch from Baseball Savant: {e}, using defaults")
        return _get_default_park_factors()


def _team_to_park_name(team: str) -> str:
    """Map team name/abbrev to park name."""
    park_map = {
        'ARI': 'Chase Field', 'Arizona Diamondbacks': 'Chase Field',
        'ATL': 'Truist Park', 'Atlanta Braves': 'Truist Park',
        'BAL': 'Camden Yards', 'Baltimore Orioles': 'Camden Yards',
        'BOS': 'Fenway Park', 'Boston Red Sox': 'Fenway Park',
        'CHC': 'Wrigley Field', 'Chicago Cubs': 'Wrigley Field',
        'CHW': 'Guaranteed Rate Field', 'Chicago White Sox': 'Guaranteed Rate Field',
        'CIN': 'Great American Ball Park', 'Cincinnati Reds': 'Great American Ball Park',
        'CLE': 'Progressive Field', 'Cleveland Guardians': 'Progressive Field',
        'COL': 'Coors Field', 'Colorado Rockies': 'Coors Field',
        'DET': 'Comerica Park', 'Detroit Tigers': 'Comerica Park',
        'HOU': 'Minute Maid Park', 'Houston Astros': 'Minute Maid Park',
        'KCR': 'Kauffman Stadium', 'Kansas City Royals': 'Kauffman Stadium',
        'LAA': 'Angel Stadium', 'Los Angeles Angels': 'Angel Stadium',
        'LAD': 'Dodger Stadium', 'Los Angeles Dodgers': 'Dodger Stadium',
        'MIA': 'loanDepot park', 'Miami Marlins': 'loanDepot park',
        'MIL': 'American Family Field', 'Milwaukee Brewers': 'American Family Field',
        'MIN': 'Target Field', 'Minnesota Twins': 'Target Field',
        'NYM': 'Citi Field', 'New York Mets': 'Citi Field',
        'NYY': 'Yankee Stadium', 'New York Yankees': 'Yankee Stadium',
        'OAK': 'Oakland Coliseum', 'Oakland Athletics': 'Oakland Coliseum',
        'PHI': 'Citizens Bank Park', 'Philadelphia Phillies': 'Citizens Bank Park',
        'PIT': 'PNC Park', 'Pittsburgh Pirates': 'PNC Park',
        'SDP': 'Petco Park', 'San Diego Padres': 'Petco Park',
        'SFG': 'Oracle Park', 'San Francisco Giants': 'Oracle Park',
        'SEA': 'T-Mobile Park', 'Seattle Mariners': 'T-Mobile Park',
        'STL': 'Busch Stadium', 'St. Louis Cardinals': 'Busch Stadium',
        'TBR': 'Tropicana Field', 'Tampa Bay Rays': 'Tropicana Field',
        'TEX': 'Globe Life Field', 'Texas Rangers': 'Globe Life Field',
        'TOR': 'Rogers Centre', 'Toronto Blue Jays': 'Rogers Centre',
        'WSN': 'Nationals Park', 'Washington Nationals': 'Nationals Park',
    }
    return park_map.get(team, team)


def _get_default_park_factors() -> Dict[str, Any]:
    """Return default park factors if scraping fails."""
    # Research-based K factors
    parks = {
        'Tropicana Field': {'team': 'TBR', 'k_factor': 1.06, 'basic_factor': 0.93},
        'T-Mobile Park': {'team': 'SEA', 'k_factor': 1.05, 'basic_factor': 0.94},
        'Oracle Park': {'team': 'SFG', 'k_factor': 1.04, 'basic_factor': 0.95},
        'Oakland Coliseum': {'team': 'OAK', 'k_factor': 1.03, 'basic_factor': 0.96},
        'Petco Park': {'team': 'SDP', 'k_factor': 1.02, 'basic_factor': 0.97},
        'Dodger Stadium': {'team': 'LAD', 'k_factor': 1.01, 'basic_factor': 0.98},
        'Kauffman Stadium': {'team': 'KCR', 'k_factor': 1.01, 'basic_factor': 0.98},
        'Citi Field': {'team': 'NYM', 'k_factor': 1.01, 'basic_factor': 0.98},
        'Yankee Stadium': {'team': 'NYY', 'k_factor': 1.00, 'basic_factor': 1.00},
        'Wrigley Field': {'team': 'CHC', 'k_factor': 1.00, 'basic_factor': 1.00},
        'Busch Stadium': {'team': 'STL', 'k_factor': 1.00, 'basic_factor': 1.00},
        'Citizens Bank Park': {'team': 'PHI', 'k_factor': 0.99, 'basic_factor': 1.02},
        'Fenway Park': {'team': 'BOS', 'k_factor': 0.97, 'basic_factor': 1.04},
        'Chase Field': {'team': 'ARI', 'k_factor': 0.96, 'basic_factor': 1.06},
        'Coors Field': {'team': 'COL', 'k_factor': 0.88, 'basic_factor': 1.15},
    }

    # Fill in remaining parks with neutral factors
    all_teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
                 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
                 'PHI', 'PIT', 'SDP', 'SFG', 'SEA', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']

    for team in all_teams:
        park_name = _team_to_park_name(team)
        if park_name not in parks:
            parks[park_name] = {'team': team, 'k_factor': 1.00, 'basic_factor': 1.00}

    team_to_park = {_team_to_park_name(t): _team_to_park_name(t) for t in all_teams}

    return {
        'season': datetime.now().year,
        'count': len(parks),
        'parks': parks,
        'team_to_park': team_to_park,
        'note': 'Using default factors (scrape failed)',
    }


# ============================================================================
# UMPIRE DATA FETCHER
# ============================================================================

def fetch_umpires() -> Dict[str, Any]:
    """
    Fetch umpire K tendencies.

    Note: UmpScorecards doesn't have a public API, so we use historical averages.
    In production, you could scrape or use Kaggle dataset.
    """
    logger.info("[Umpires] Loading umpire K tendencies...")

    # Historical umpire data (based on UmpScorecards research)
    # k_index > 1.0 means more Ks than average
    umpires = {
        'Pat Hoberg': {'k_index': 1.04, 'games': 900, 'accuracy': 95.2},
        'John Tumpane': {'k_index': 1.03, 'games': 850, 'accuracy': 94.8},
        'Nic Lentz': {'k_index': 1.02, 'games': 600, 'accuracy': 94.5},
        'Tripp Gibson': {'k_index': 1.02, 'games': 500, 'accuracy': 94.3},
        'David Rackley': {'k_index': 1.01, 'games': 400, 'accuracy': 94.1},
        'Mark Carlson': {'k_index': 1.00, 'games': 1200, 'accuracy': 93.8},
        'Jim Wolf': {'k_index': 1.00, 'games': 1100, 'accuracy': 93.5},
        'Lance Barksdale': {'k_index': 0.99, 'games': 1000, 'accuracy': 93.2},
        'Angel Hernandez': {'k_index': 0.97, 'games': 1800, 'accuracy': 91.5},
        'CB Bucknor': {'k_index': 0.96, 'games': 1500, 'accuracy': 91.8},
        'Joe West': {'k_index': 0.98, 'games': 2000, 'accuracy': 92.5},
        'Doug Eddings': {'k_index': 0.98, 'games': 1300, 'accuracy': 92.8},
    }

    result = {
        'count': len(umpires),
        'league_avg_k_index': 1.0,
        'umpires': umpires,
        'note': 'Historical averages - consider scraping UmpScorecards for live data',
    }

    save_cache('umpires', result)
    logger.info(f"[Umpires] Cached {len(umpires)} umpire profiles")

    return result


# ============================================================================
# SCHEDULE FETCHER (MLB Stats API)
# ============================================================================

def fetch_schedule(target_date: date = None) -> Dict[str, Any]:
    """
    Fetch today's games with probable pitchers from MLB Stats API.
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%m/%d/%Y")
    logger.info(f"[Schedule] Fetching games for {date_str}...")

    try:
        schedule = statsapi.schedule(start_date=date_str, end_date=date_str)
    except Exception as e:
        logger.error(f"[Schedule] Failed to fetch: {e}")
        return {'games': [], 'date': target_date.isoformat()}

    games = []
    for game in schedule:
        games.append({
            'game_pk': game.get('game_id'),
            'game_date': game.get('game_date'),
            'game_time': game.get('game_datetime'),
            'status': game.get('status'),
            'home_team': game.get('home_name'),
            'away_team': game.get('away_name'),
            'home_team_abbrev': _get_team_abbrev(game.get('home_name', '')),
            'away_team_abbrev': _get_team_abbrev(game.get('away_name', '')),
            'home_pitcher': game.get('home_probable_pitcher'),
            'away_pitcher': game.get('away_probable_pitcher'),
            'venue': game.get('venue_name'),
        })

    result = {
        'date': target_date.isoformat(),
        'count': len(games),
        'games': games,
    }

    save_cache('schedule', result)
    logger.info(f"[Schedule] Found {len(games)} games")

    return result


def _get_team_abbrev(team_name: str) -> str:
    """Convert full team name to abbreviation."""
    abbrev_map = {
        'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
        'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
        'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
        'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
        'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
        'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
        'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
        'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
        'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
        'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    }
    return abbrev_map.get(team_name, team_name[:3].upper())


# ============================================================================
# GETTER FUNCTIONS (for projection engine)
# ============================================================================

def get_pitcher(name: str, use_blending: bool = True) -> Optional[Dict]:
    """
    Get pitcher stats by name (case-insensitive, partial match).

    Args:
        name: Pitcher name to search for
        use_blending: If True, use Marcel blending across 2025/2026 seasons

    Returns pitcher dict with stats, or None if not found.
    """
    # Use blended stats by default (Marcel methodology)
    if use_blending:
        blended = get_blended_pitcher(name)
        if blended:
            return blended

    # Fall back to single cache lookup
    cache = load_cache('pitchers')
    if not cache:
        # Try year-specific caches
        cache = load_cache('pitchers_2026') or load_cache('pitchers_2025')

    if not cache:
        return None

    pitchers = cache.get('pitchers', {})
    name_lower = name.lower().strip()

    # Exact match
    if name_lower in pitchers:
        pitcher = pitchers[name_lower].copy()
        pitcher['blend_method'] = 'single_cache'
        # Add game logs if available
        logs = get_pitcher_logs(pitcher.get('mlbam_id'))
        if logs:
            pitcher.update(logs)
        return pitcher

    # Partial match
    for key, val in pitchers.items():
        if name_lower in key:
            pitcher = val.copy()
            pitcher['blend_method'] = 'single_cache'
            logs = get_pitcher_logs(pitcher.get('mlbam_id'))
            if logs:
                pitcher.update(logs)
            return pitcher

    # Match by last name + first initial (handles "Cam Schlittler" -> "Cameron Schlittler")
    query_parts = name_lower.split()
    if len(query_parts) >= 2:
        query_last = query_parts[-1]
        query_first = query_parts[0]
        for key, val in pitchers.items():
            key_parts = key.split()
            if len(key_parts) >= 2:
                key_last = key_parts[-1]
                key_first = key_parts[0]
                # Match if last names match and first name starts with query first
                if key_last == query_last and key_first.startswith(query_first):
                    pitcher = val.copy()
                    pitcher['blend_method'] = 'single_cache'
                    logs = get_pitcher_logs(pitcher.get('mlbam_id'))
                    if logs:
                        pitcher.update(logs)
                    return pitcher

    # FALLBACK: Check legacy cache file (mlb_pitcher_stats_cache.json)
    # This has additional pitchers not in the FanGraphs pybaseball data
    # Check this BEFORE "last name only" match to get exact/partial matches first
    legacy_pitcher = _check_legacy_cache(name_lower)
    if legacy_pitcher:
        return legacy_pitcher

    # Last name only match (last resort - can be ambiguous)
    if query_parts:
        query_last = query_parts[-1]
        for key, val in pitchers.items():
            if query_last == key.split()[-1]:
                pitcher = val.copy()
                pitcher['blend_method'] = 'single_cache'
                logs = get_pitcher_logs(pitcher.get('mlbam_id'))
                if logs:
                    pitcher.update(logs)
                return pitcher

    return None


def _check_legacy_cache(name_lower: str) -> Optional[Dict]:
    """Fallback to legacy mlb_pitcher_stats_cache.json for pitchers not in main cache."""
    legacy_path = Path(__file__).parent / "mlb_pitcher_stats_cache.json"
    if not legacy_path.exists():
        return None

    try:
        with open(legacy_path, 'r') as f:
            data = json.load(f)
        pitchers = data.get('pitchers', {})

        # Exact match
        if name_lower in pitchers:
            pitcher = pitchers[name_lower].copy()
            pitcher['blend_method'] = 'legacy_cache'
            return pitcher

        # Partial match
        for key, val in pitchers.items():
            if name_lower in key:
                pitcher = val.copy()
                pitcher['blend_method'] = 'legacy_cache'
                return pitcher

        # Last name match
        query_parts = name_lower.split()
        if query_parts:
            query_last = query_parts[-1]
            for key, val in pitchers.items():
                if query_last == key.split()[-1]:
                    pitcher = val.copy()
                    pitcher['blend_method'] = 'legacy_cache'
                    return pitcher

    except (json.JSONDecodeError, IOError):
        pass

    return None


def get_pitcher_logs(mlbam_id: int) -> Optional[Dict]:
    """Get pitcher game logs by MLBAM ID."""
    if not mlbam_id:
        return None

    cache = load_cache('pitcher_logs')
    if not cache:
        return None

    return cache.get('pitchers', {}).get(str(mlbam_id))


def get_team(name: str) -> Optional[Dict]:
    """Get team stats by name or abbreviation."""
    cache = load_cache('teams')
    if not cache:
        return None

    teams = cache.get('teams', {})
    name_upper = name.upper().strip()
    name_lower = name.lower().strip()

    # Direct match (abbreviation)
    if name_upper in teams:
        return teams[name_upper]

    # Try common variations
    name_map = {
        'yankees': 'NYY', 'new york yankees': 'NYY',
        'red sox': 'BOS', 'boston red sox': 'BOS',
        'blue jays': 'TOR', 'toronto blue jays': 'TOR',
        'orioles': 'BAL', 'baltimore orioles': 'BAL',
        'rays': 'TBR', 'tampa bay rays': 'TBR',
        'guardians': 'CLE', 'cleveland guardians': 'CLE',
        'tigers': 'DET', 'detroit tigers': 'DET',
        'twins': 'MIN', 'minnesota twins': 'MIN',
        'white sox': 'CHW', 'chicago white sox': 'CHW',
        'royals': 'KCR', 'kansas city royals': 'KCR',
        'astros': 'HOU', 'houston astros': 'HOU',
        'mariners': 'SEA', 'seattle mariners': 'SEA',
        'rangers': 'TEX', 'texas rangers': 'TEX',
        'angels': 'LAA', 'los angeles angels': 'LAA',
        'athletics': 'OAK', 'oakland athletics': 'OAK',
        'mets': 'NYM', 'new york mets': 'NYM',
        'braves': 'ATL', 'atlanta braves': 'ATL',
        'phillies': 'PHI', 'philadelphia phillies': 'PHI',
        'marlins': 'MIA', 'miami marlins': 'MIA',
        'nationals': 'WSN', 'washington nationals': 'WSN',
        'cubs': 'CHC', 'chicago cubs': 'CHC',
        'cardinals': 'STL', 'st. louis cardinals': 'STL',
        'brewers': 'MIL', 'milwaukee brewers': 'MIL',
        'reds': 'CIN', 'cincinnati reds': 'CIN',
        'pirates': 'PIT', 'pittsburgh pirates': 'PIT',
        'dodgers': 'LAD', 'los angeles dodgers': 'LAD',
        'giants': 'SFG', 'san francisco giants': 'SFG',
        'padres': 'SDP', 'san diego padres': 'SDP',
        'diamondbacks': 'ARI', 'arizona diamondbacks': 'ARI',
        'rockies': 'COL', 'colorado rockies': 'COL',
    }

    if name_lower in name_map:
        abbrev = name_map[name_lower]
        return teams.get(abbrev)

    # Partial match
    for key in teams:
        if name_lower in key.lower():
            return teams[key]

    return None


def get_park_factor(venue: str) -> float:
    """Get K factor for a venue."""
    cache = load_cache('park_factors')
    if not cache:
        return 1.0

    parks = cache.get('parks', {})

    # Direct match
    if venue in parks:
        return parks[venue].get('k_factor', 1.0)

    # Partial match
    venue_lower = venue.lower()
    for park_name, data in parks.items():
        if venue_lower in park_name.lower():
            return data.get('k_factor', 1.0)

    return 1.0


def get_umpire(name: str) -> Optional[Dict]:
    """Get umpire K tendency by name."""
    cache = load_cache('umpires')
    if not cache:
        return None

    umpires = cache.get('umpires', {})

    # Direct match
    if name in umpires:
        return umpires[name]

    # Partial match
    name_lower = name.lower()
    for ump_name, data in umpires.items():
        if name_lower in ump_name.lower():
            return data

    return None


def get_league_averages() -> Dict[str, float]:
    """Get current league averages for pitchers and teams."""
    pitcher_cache = load_cache('pitchers')
    team_cache = load_cache('teams')

    averages = {
        'k_pct': 22.5,
        'swstr_pct': 11.0,
        'csw_pct': 29.0,
        'team_k_pct': 22.5,
        'p_per_pa': 3.92,
    }

    if pitcher_cache and 'league_averages' in pitcher_cache:
        averages.update({
            'k_pct': pitcher_cache['league_averages'].get('k_pct', 22.5),
            'swstr_pct': pitcher_cache['league_averages'].get('swstr_pct', 11.0),
            'csw_pct': pitcher_cache['league_averages'].get('csw_pct', 29.0),
        })

    if team_cache and 'league_averages' in team_cache:
        averages.update({
            'team_k_pct': team_cache['league_averages'].get('k_pct', 22.5),
            'team_k_pct_vs_lhp': team_cache['league_averages'].get('k_pct_vs_lhp', 22.5),
            'team_k_pct_vs_rhp': team_cache['league_averages'].get('k_pct_vs_rhp', 22.5),
            'p_per_pa': team_cache['league_averages'].get('p_per_pa', 3.92),
        })

    return averages


def get_todays_games() -> List[Dict]:
    """Get today's games, refreshing if stale."""
    if is_cache_stale('schedule'):
        fetch_schedule()

    cache = load_cache('schedule')
    return cache.get('games', []) if cache else []


# ============================================================================
# REFRESH ALL DATA
# ============================================================================

def refresh_all(year: int = None, force: bool = False) -> Dict[str, Any]:
    """
    Refresh all MLB data caches.

    Run daily before first game (~10-11am ET).

    For Marcel blending:
    - 2025 data is fetched once and cached as baseline
    - 2026 data is refreshed daily during the season
    """
    if year is None:
        year = datetime.now().year

    prior_year = year - 1

    logger.info("=" * 60)
    logger.info(f"MLB DATA REFRESH (Marcel blending: {prior_year}/{year})")
    logger.info("=" * 60)

    results = {}

    # 1a. Prior year pitcher stats (2025 baseline - only fetch if missing)
    cache_key_prior = f'pitchers_{prior_year}'
    if cache_key_prior in CACHE_FILES:
        if force or not load_cache(cache_key_prior):
            logger.info(f"[Pitchers] Fetching {prior_year} baseline...")
            pitcher_data_prior = fetch_pitchers(prior_year, cache_key=cache_key_prior)
            results['pitchers_prior'] = len(pitcher_data_prior.get('pitchers', {}))
        else:
            cache = load_cache(cache_key_prior)
            results['pitchers_prior'] = len(cache.get('pitchers', {})) if cache else 0
            logger.info(f"[Pitchers] {prior_year} baseline exists, skipping")

    # 1b. Current year pitcher stats (2026 - refresh daily)
    cache_key_current = f'pitchers_{year}'
    if cache_key_current in CACHE_FILES:
        if force or is_cache_stale(cache_key_current):
            logger.info(f"[Pitchers] Fetching {year} current season...")
            pitcher_data_current = fetch_pitchers(year, cache_key=cache_key_current)
            results['pitchers_current'] = len(pitcher_data_current.get('pitchers', {}))
        else:
            cache = load_cache(cache_key_current)
            results['pitchers_current'] = len(cache.get('pitchers', {})) if cache else 0
            logger.info(f"[Pitchers] {year} cache still fresh, skipping")

    # 1c. Also update generic 'pitchers' cache with current year for backward compatibility
    if force or is_cache_stale('pitchers'):
        pitcher_data = fetch_pitchers(year, cache_key='pitchers')
        results['pitchers'] = len(pitcher_data.get('pitchers', {}))
    else:
        cache = load_cache('pitchers')
        results['pitchers'] = len(cache.get('pitchers', {})) if cache else 0
        logger.info("[Pitchers] Generic cache still fresh, skipping")

    # 2. Pitcher game logs (always refresh daily)
    if force or is_cache_stale('pitcher_logs'):
        logs_data = fetch_pitcher_logs()
        results['pitcher_logs'] = len(logs_data.get('pitchers', {}))
    else:
        cache = load_cache('pitcher_logs')
        results['pitcher_logs'] = len(cache.get('pitchers', {})) if cache else 0
        logger.info("[PitcherLogs] Cache still fresh, skipping")

    # 3. Team batting + splits (always refresh daily)
    if force or is_cache_stale('teams'):
        team_data = fetch_teams(year)
        results['teams'] = len(team_data.get('teams', {}))
    else:
        cache = load_cache('teams')
        results['teams'] = len(cache.get('teams', {})) if cache else 0
        logger.info("[Teams] Cache still fresh, skipping")

    # 4. Park factors (weekly refresh)
    if force or is_cache_stale('park_factors'):
        park_data = fetch_park_factors(year)
        results['parks'] = len(park_data.get('parks', {}))
    else:
        cache = load_cache('park_factors')
        results['parks'] = len(cache.get('parks', {})) if cache else 0
        logger.info("[Parks] Cache still fresh, skipping")

    # 5. Umpires (weekly refresh)
    if force or is_cache_stale('umpires'):
        ump_data = fetch_umpires()
        results['umpires'] = len(ump_data.get('umpires', {}))
    else:
        cache = load_cache('umpires')
        results['umpires'] = len(cache.get('umpires', {})) if cache else 0
        logger.info("[Umpires] Cache still fresh, skipping")

    # 6. Today's schedule
    schedule_data = fetch_schedule()
    results['games'] = len(schedule_data.get('games', []))

    logger.info("=" * 60)
    logger.info(f"REFRESH COMPLETE: {results}")
    logger.info("=" * 60)

    return results


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing MLB Data Module\n")

    # Refresh all data
    results = refresh_all(force=True)
    print(f"\nRefresh results: {results}")

    # Test getters
    print("\n--- Testing Getters ---")

    pitcher = get_pitcher("Logan Webb")
    if pitcher:
        print(f"\nPitcher: {pitcher['name']}")
        print(f"  Team: {pitcher['team']}, Hand: {pitcher['hand']}")
        print(f"  K%: {pitcher.get('k_pct')}%, SwStr%: {pitcher.get('swstr_pct')}%")
        print(f"  Last 5 avg Ks: {pitcher.get('last_5_avg_ks')}")

    team = get_team("Yankees")
    if team:
        print(f"\nTeam: {team['name']}")
        print(f"  K%: {team.get('k_pct')}%")
        print(f"  K% vs LHP: {team.get('k_pct_vs_lhp')}%, K% vs RHP: {team.get('k_pct_vs_rhp')}%")

    park_factor = get_park_factor("Oracle Park")
    print(f"\nOracle Park K factor: {park_factor}")

    league_avg = get_league_averages()
    print(f"\nLeague averages: {league_avg}")

    games = get_todays_games()
    print(f"\nToday's games: {len(games)}")
    for g in games[:3]:
        print(f"  {g['away_team']} @ {g['home_team']}: {g.get('away_pitcher', 'TBD')} vs {g.get('home_pitcher', 'TBD')}")
