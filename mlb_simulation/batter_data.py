#!/usr/bin/env python3
"""
MLB Batter Data Module for Strikeout Simulation

Fetches batter K%/BB% from Baseball Reference via pybaseball.
Used for PA-level Monte Carlo simulation.

Phase 1 of Tier 3 MLB Simulation.

Note: Using Baseball Reference instead of FanGraphs due to 403 errors.
Platoon splits are estimated based on league averages since BR doesn't
provide splits in this dataset.
"""

import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def normalize_name(name: str) -> str:
    """Normalize name by removing accents and converting to lowercase.

    Examples:
        José Ramírez → jose ramirez
        Ángel Martínez → angel martinez

    Also handles:
    - Corrupted UTF-8 bytes in strings (e.g., 'jos\xc3\xa9' → 'jose')
    - Escaped hex sequences (e.g., 'jos\\xc3\\xa9' → 'jose')
    """
    import codecs

    # Handle escaped hex sequences like 'jos\\xc3\\xa9 ram\\xc3\\xadrez'
    if '\\x' in name:
        try:
            # Decode the escaped sequences
            decoded = codecs.decode(name, 'unicode_escape')
            # Then decode as UTF-8 bytes
            name = decoded.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
            pass

    # Try to fix corrupted UTF-8 bytes in the string
    try:
        # If the string contains raw UTF-8 bytes as characters, encode then decode
        fixed = name.encode('latin-1').decode('utf-8')
        name = fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass  # Name is fine, continue with original

    # Decompose unicode characters, then remove combining marks (accents)
    normalized = unicodedata.normalize('NFKD', name)
    ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_name.lower().strip()

try:
    import pybaseball as pyb
    pyb.cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

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

BATTER_CACHE_FILE = CACHE_DIR / "mlb_batters.json"
BATTER_CACHE_FILE_2025 = CACHE_DIR / "mlb_batters_2025.json"
BATTER_CACHE_TTL_HOURS = 24
BATTER_CACHE_TTL_HOURS_2025 = 720  # 30 days for prior year

# League average K% for regression (2023-2025 average)
LEAGUE_K_PCT = 22.7
LEAGUE_BB_PCT = 8.5


def load_batter_cache() -> Optional[Dict]:
    """Load batter cache if exists and not stale."""
    if not BATTER_CACHE_FILE.exists():
        return None

    try:
        with open(BATTER_CACHE_FILE, 'r') as f:
            data = json.load(f)

        # Check staleness
        metadata = data.get('_metadata', {})
        updated_str = metadata.get('updated')
        if updated_str:
            updated = datetime.fromisoformat(updated_str)
            age_hours = (datetime.now() - updated).total_seconds() / 3600
            if age_hours > BATTER_CACHE_TTL_HOURS:
                logger.info(f"[Batters] Cache is stale ({age_hours:.1f}h old)")
                return None

        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"[Batters] Failed to load cache: {e}")
        return None


def save_batter_cache(data: Dict) -> bool:
    """Save batter data to cache."""
    data['_metadata'] = {
        'updated': datetime.now().isoformat(),
        'cache_name': 'batters',
    }

    try:
        with open(BATTER_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"[Batters] Saved {len(data.get('batters', {}))} batters to cache")
        return True
    except IOError as e:
        logger.error(f"[Batters] Failed to save cache: {e}")
        return False


def fetch_batters(year: int = None, min_pa: int = 30) -> Dict[str, Any]:
    """
    Fetch all batter stats from Baseball Reference.

    Args:
        year: Season year (defaults to current)
        min_pa: Minimum PA to include

    Returns:
        Dict with batters keyed by lowercase name
    """
    if not PYBASEBALL_AVAILABLE:
        logger.error("[Batters] pybaseball not available")
        return {}

    if year is None:
        year = datetime.now().year

    logger.info(f"[Batters] Fetching from Baseball Reference for {year}...")

    try:
        # Use Baseball Reference (more reliable than FanGraphs which returns 403)
        df = pyb.batting_stats_bref(year)
        if df is None or df.empty:
            raise ValueError("No data returned")
    except Exception as e:
        logger.warning(f"[Batters] {year} failed: {e}, trying {year-1}")
        try:
            df = pyb.batting_stats_bref(year - 1)
            year = year - 1
        except Exception as e2:
            logger.error(f"[Batters] Failed to fetch data: {e2}")
            return {}

    if df is None or df.empty:
        logger.error("[Batters] No data available")
        return {}

    # Filter by minimum PA
    df = df[df['PA'] >= min_pa]
    logger.info(f"[Batters] Got {len(df)} batters with {min_pa}+ PA")

    # Parse batters
    batters = {}
    for _, row in df.iterrows():
        name = row.get('Name', '')
        if not name:
            continue

        # Get MLB ID
        mlbam_id = row.get('mlbID')

        # Calculate K% and BB% from raw stats
        pa = int(row.get('PA', 0))
        so = int(row.get('SO', 0))
        bb = int(row.get('BB', 0))

        k_pct = (so / pa * 100) if pa > 0 else LEAGUE_K_PCT
        bb_pct = (bb / pa * 100) if pa > 0 else LEAGUE_BB_PCT

        # Default handedness (Baseball Reference doesn't provide this)
        # Will use league-average platoon splits
        bats = 'R'

        # Platoon split estimates (based on MLB averages)
        # Without knowing handedness, use average split (+2% vs same hand)
        k_pct_vs_lhp = k_pct * 1.02
        k_pct_vs_rhp = k_pct * 1.02
        bb_pct_vs_lhp = bb_pct
        bb_pct_vs_rhp = bb_pct

        # Get team
        team = row.get('Tm', '')

        batters[name.lower()] = {
            'name': name,
            'mlbam_id': int(mlbam_id) if pd.notna(mlbam_id) else None,
            'team': team,
            'bats': bats,
            'pa': pa,

            # Overall rates (calculated from SO/BB/PA)
            'k_pct': round(k_pct, 1),
            'bb_pct': round(bb_pct, 1),

            # Platoon splits (estimated - no handedness data)
            'k_pct_vs_lhp': round(k_pct_vs_lhp, 1),
            'k_pct_vs_rhp': round(k_pct_vs_rhp, 1),
            'bb_pct_vs_lhp': round(bb_pct_vs_lhp, 1),
            'bb_pct_vs_rhp': round(bb_pct_vs_rhp, 1),

            # PA splits (estimated 72/28 vs RHP/LHP based on league averages)
            'pa_vs_lhp': int(pa * 0.28),
            'pa_vs_rhp': int(pa * 0.72),

            # Raw stats for reference
            'so': so,
            'bb': bb,
        }

    # Calculate league averages from the data
    total_pa = df['PA'].sum()
    total_so = df['SO'].sum()
    total_bb = df['BB'].sum()

    league_averages = {
        'k_pct': round((total_so / total_pa * 100), 1) if total_pa > 0 else LEAGUE_K_PCT,
        'bb_pct': round((total_bb / total_pa * 100), 1) if total_pa > 0 else LEAGUE_BB_PCT,
    }

    result = {
        'season': year,
        'count': len(batters),
        'league_averages': league_averages,
        'batters': batters,
    }

    save_batter_cache(result)
    logger.info(f"[Batters] Cached {len(batters)} batters, league K%: {league_averages['k_pct']:.1f}%")

    return result


def _load_2025_cache() -> Optional[Dict]:
    """Load 2025 batter cache as fallback."""
    if not BATTER_CACHE_FILE_2025.exists():
        return None

    try:
        with open(BATTER_CACHE_FILE_2025, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_2025_cache(data: Dict) -> bool:
    """Save 2025 batter data to cache."""
    data['_metadata'] = {
        'updated': datetime.now().isoformat(),
        'cache_name': 'batters_2025',
    }

    try:
        with open(BATTER_CACHE_FILE_2025, 'w') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        return True
    except IOError:
        return False


def fetch_batters_2025() -> Dict[str, Any]:
    """Fetch 2025 batter stats as baseline/fallback."""
    cache = _load_2025_cache()
    if cache:
        return cache

    logger.info("[Batters] Fetching 2025 baseline data...")
    data = fetch_batters(year=2025, min_pa=100)
    if data:
        _save_2025_cache(data)
    return data


def _find_in_batters(name_lower: str, batters: Dict) -> Optional[Dict]:
    """Find a batter in a batters dict, handling accented names."""
    # Normalize the search term (remove accents)
    name_normalized = normalize_name(name_lower)

    # Exact match (original)
    if name_lower in batters:
        return batters[name_lower]

    # Try normalized match against all keys
    for key, val in batters.items():
        key_normalized = normalize_name(key)
        if name_normalized == key_normalized:
            return val

    # Partial match (normalized)
    for key, val in batters.items():
        key_normalized = normalize_name(key)
        if name_normalized in key_normalized or key_normalized in name_normalized:
            return val

    # Last name match (normalized)
    name_parts = name_normalized.split()
    if name_parts:
        last_name = name_parts[-1]
        for key, val in batters.items():
            key_normalized = normalize_name(key)
            key_parts = key_normalized.split()
            if key_parts and last_name == key_parts[-1]:
                return val

    return None


def get_batter(name: str) -> Optional[Dict]:
    """
    Get batter stats by name.

    Checks 2026 data first, falls back to 2025 baseline.

    Args:
        name: Batter name (case insensitive)

    Returns:
        Batter dict or None if not found
    """
    name_lower = name.lower().strip()

    # Try 2026 data first
    cache = load_batter_cache()
    if not cache:
        cache = fetch_batters()

    if cache:
        batters = cache.get('batters', {})
        batter = _find_in_batters(name_lower, batters)
        if batter:
            return batter

    # Fall back to 2025 data
    cache_2025 = _load_2025_cache()
    if not cache_2025:
        cache_2025 = fetch_batters_2025()

    if cache_2025:
        batters_2025 = cache_2025.get('batters', {})
        return _find_in_batters(name_lower, batters_2025)

    return None


def get_batter_k_rate(name: str, pitcher_hand: str) -> float:
    """
    Get batter's K% vs pitcher handedness.

    Args:
        name: Batter name
        pitcher_hand: 'L' or 'R'

    Returns:
        K% (0-100 scale) or league average if not found
    """
    batter = get_batter(name)
    if not batter:
        return LEAGUE_K_PCT

    if pitcher_hand.upper() == 'L':
        return batter.get('k_pct_vs_lhp') or batter.get('k_pct') or LEAGUE_K_PCT
    else:
        return batter.get('k_pct_vs_rhp') or batter.get('k_pct') or LEAGUE_K_PCT


def get_batter_bb_rate(name: str, pitcher_hand: str) -> float:
    """
    Get batter's BB% vs pitcher handedness.

    Args:
        name: Batter name
        pitcher_hand: 'L' or 'R'

    Returns:
        BB% (0-100 scale) or league average if not found
    """
    batter = get_batter(name)
    if not batter:
        return LEAGUE_BB_PCT

    if pitcher_hand.upper() == 'L':
        return batter.get('bb_pct_vs_lhp') or batter.get('bb_pct') or LEAGUE_BB_PCT
    else:
        return batter.get('bb_pct_vs_rhp') or batter.get('bb_pct') or LEAGUE_BB_PCT


def get_league_k_rate() -> float:
    """Get current league average K%."""
    cache = load_batter_cache()
    if cache and 'league_averages' in cache:
        return cache['league_averages'].get('k_pct', LEAGUE_K_PCT)
    return LEAGUE_K_PCT


def refresh_batters(force: bool = False) -> Dict:
    """
    Refresh batter cache.

    Args:
        force: Force refresh even if cache is fresh

    Returns:
        Batter data dict
    """
    if not force:
        cache = load_batter_cache()
        if cache:
            logger.info("[Batters] Cache is fresh, skipping refresh")
            return cache

    return fetch_batters()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("Force refreshing batter cache...")
        data = refresh_batters(force=True)
    else:
        print("Loading/refreshing batter cache...")
        data = refresh_batters(force=True)  # Force on first run

    if data:
        print(f"\nLoaded {data.get('count', 0)} batters from {data.get('season', 'unknown')} season")
        print(f"League K%: {data.get('league_averages', {}).get('k_pct', 'N/A'):.1f}%")

        # Test a few batters
        test_batters = ['aaron judge', 'mookie betts', 'shohei ohtani']
        print("\nSample batters:")
        for name in test_batters:
            batter = get_batter(name)
            if batter:
                print(f"  {batter['name']}: K%={batter.get('k_pct')}%, vs LHP={batter.get('k_pct_vs_lhp')}%, vs RHP={batter.get('k_pct_vs_rhp')}%")
            else:
                print(f"  {name}: NOT FOUND")
    else:
        print("Failed to load batter data")
