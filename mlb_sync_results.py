#!/usr/bin/env python3
"""
MLB Results Sync - Fully Automated
Syncs actual strikeout results from MLB Stats API.
Run the morning after games.

Completely separate from NBA sync.
"""
import sqlite3
import logging
import unicodedata
from datetime import datetime, timedelta
from typing import Optional

import statsapi


def normalize_name(name: str) -> str:
    """Normalize name by removing accents and converting to lowercase."""
    # NFD decomposition separates base characters from combining marks (accents)
    # encode('ascii', 'ignore') drops the non-ASCII combining marks
    return unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('ascii').lower()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "predictions.db"

# MLB team abbreviation to full name mapping (includes alternate abbreviations)
TEAM_ABBREV_MAP = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
    # Alternate abbreviations (3-letter variants)
    "KCR": "Kansas City Royals", "TBR": "Tampa Bay Rays", "SDP": "San Diego Padres",
    "SFG": "San Francisco Giants", "AZ": "Arizona Diamondbacks", "WAS": "Washington Nationals",
    # Athletics moved from Oakland - API returns "Athletics" not "Oakland Athletics"
    "Athletics": "Athletics", "OAK": "Athletics",
}


def get_game_boxscore(game_pk: int) -> Optional[dict]:
    """Get boxscore data for a game."""
    try:
        return statsapi.boxscore_data(game_pk)
    except Exception as e:
        logger.error(f"Error fetching boxscore for game {game_pk}: {e}")
        return None


def find_pitcher_stats(boxscore: dict, pitcher_name: str) -> Optional[dict]:
    """Find a pitcher's stats in the boxscore."""
    pitcher_normalized = normalize_name(pitcher_name)
    pitcher_last = pitcher_normalized.split()[-1] if pitcher_normalized else ''

    for side in ['away', 'home']:
        pitchers = boxscore.get(side + 'Pitchers', [])

        for pitcher in pitchers:
            name = pitcher.get('name', '')
            name_normalized = normalize_name(name)

            # Try various matching strategies
            if pitcher_normalized in name_normalized or name_normalized in pitcher_normalized:
                return {
                    'strikeouts': pitcher.get('so', pitcher.get('k', 0)),
                    'innings_pitched': pitcher.get('ip', 0),
                    'pitches': pitcher.get('p', pitcher.get('pitches', 0)),
                    'batters_faced': pitcher.get('bf', 0),
                }

            # Try matching last name only
            name_last = name_normalized.split()[-1] if name_normalized else ''
            if pitcher_last and name_last and pitcher_last == name_last:
                return {
                    'strikeouts': pitcher.get('so', pitcher.get('k', 0)),
                    'innings_pitched': pitcher.get('ip', 0),
                    'pitches': pitcher.get('p', pitcher.get('pitches', 0)),
                    'batters_faced': pitcher.get('bf', 0),
                }

    return None


def get_pitcher_game_stats(game_pk: int, pitcher_name: str) -> Optional[dict]:
    """Get a specific pitcher's game stats using statsapi."""
    try:
        # Get game data
        game = statsapi.get('game', {'gamePk': game_pk})

        if not game or 'liveData' not in game:
            return None

        boxscore = game.get('liveData', {}).get('boxscore', {})
        teams = boxscore.get('teams', {})

        # Normalize the search name (removes accents)
        pitcher_normalized = normalize_name(pitcher_name)
        pitcher_last = pitcher_normalized.split()[-1] if pitcher_normalized else ''

        for side in ['away', 'home']:
            pitchers = teams.get(side, {}).get('pitchers', [])
            players = teams.get(side, {}).get('players', {})

            for pitcher_id in pitchers:
                player_key = f'ID{pitcher_id}'
                player_data = players.get(player_key, {})
                full_name = player_data.get('person', {}).get('fullName', '')
                # Normalize the player name from API (removes accents like Vásquez → Vasquez)
                player_normalized = normalize_name(full_name)

                # Try full name match
                if pitcher_normalized in player_normalized or player_normalized in pitcher_normalized:
                    stats = player_data.get('stats', {}).get('pitching', {})
                    return {
                        'strikeouts': int(stats.get('strikeOuts', 0)),
                        'innings_pitched': float(stats.get('inningsPitched', 0)),
                        'pitches': int(stats.get('numberOfPitches', 0)),
                        'batters_faced': int(stats.get('battersFaced', 0)),
                    }

                # Try last name match
                player_last = player_normalized.split()[-1] if player_normalized else ''
                if pitcher_last and player_last and pitcher_last == player_last:
                    stats = player_data.get('stats', {}).get('pitching', {})
                    return {
                        'strikeouts': int(stats.get('strikeOuts', 0)),
                        'innings_pitched': float(stats.get('inningsPitched', 0)),
                        'pitches': int(stats.get('numberOfPitches', 0)),
                        'batters_faced': int(stats.get('battersFaced', 0)),
                    }

        return None

    except Exception as e:
        logger.error(f"Error getting pitcher stats: {e}")
        return None


def sync_mlb_results(target_date: str = None):
    """
    Sync MLB results for a specific date.

    Args:
        target_date: Date in YYYY-MM-DD format (defaults to yesterday)
    """
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"[MLB] Syncing results for {target_date}")
    print(f"\n{'='*60}")
    print(f"MLB RESULTS SYNC - {target_date}")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get pending predictions for this date
    cursor.execute("""
        SELECT id, pitcher_name, opponent_team, line, recommended_side, game_pk
        FROM mlb_predictions
        WHERE game_date = ? AND status = 'pending'
        ORDER BY id
    """, (target_date,))

    pending = [dict(row) for row in cursor.fetchall()]
    logger.info(f"[MLB] Found {len(pending)} pending predictions")

    if not pending:
        print("No pending predictions for this date.")
        conn.close()
        return

    # Get schedule for this date to find game PKs
    date_formatted = datetime.strptime(target_date, "%Y-%m-%d").strftime("%m/%d/%Y")
    try:
        schedule = statsapi.schedule(start_date=date_formatted, end_date=date_formatted)
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        schedule = []

    # Build game lookup by team
    game_lookup = {}
    for game in schedule:
        game_lookup[game['home_name'].lower()] = game
        game_lookup[game['away_name'].lower()] = game
        # Also add abbreviation-style lookups
        for abbrev in [game.get('home_name', ''), game.get('away_name', '')]:
            game_lookup[abbrev.lower()] = game

    stats = {'hits': 0, 'misses': 0, 'voided': 0, 'errors': 0}

    for pred in pending:
        pred_id = pred['id']
        pitcher_name = pred['pitcher_name']
        opponent = pred['opponent_team']
        line = pred['line']
        rec_side = pred['recommended_side']
        game_pk = pred.get('game_pk')

        print(f"[{pred_id}] {pitcher_name} vs {opponent} | Line: {line} | Pick: {rec_side}")

        # Try to find the game
        if not game_pk:
            # Try to find game by opponent (try abbreviation first, then full name)
            opponent_full = TEAM_ABBREV_MAP.get(opponent.upper(), opponent)
            game = game_lookup.get(opponent_full.lower()) or game_lookup.get(opponent.lower())
            if game:
                game_pk = game['game_id']

        if not game_pk:
            logger.warning(f"  Could not find game for {pitcher_name} vs {opponent}")
            stats['errors'] += 1
            continue

        # Get pitcher stats
        pitcher_stats = get_pitcher_game_stats(game_pk, pitcher_name)

        if not pitcher_stats:
            # Pitcher didn't play - void
            cursor.execute("""
                UPDATE mlb_predictions SET status = 'voided' WHERE id = ?
            """, (pred_id,))
            print(f"  → DNP - VOIDED")
            stats['voided'] += 1
            continue

        actual_ks = pitcher_stats['strikeouts']
        actual_ip = pitcher_stats['innings_pitched']
        actual_pitches = pitcher_stats['pitches']
        actual_bf = pitcher_stats['batters_faced']

        # Determine hit/miss
        if rec_side == 'over':
            hit = 1 if actual_ks > line else 0
        else:
            hit = 1 if actual_ks < line else 0

        # Update prediction
        cursor.execute("""
            UPDATE mlb_predictions SET
                actual_ks = ?,
                actual_ip = ?,
                actual_pitches = ?,
                actual_bf = ?,
                hit = ?,
                status = 'resolved',
                resolved_at = ?
            WHERE id = ?
        """, (
            actual_ks, actual_ip, actual_pitches, actual_bf,
            hit, datetime.now().isoformat(), pred_id
        ))

        result_str = "HIT ✓" if hit else "MISS ✗"
        print(f"  → Actual: {actual_ks} Ks ({actual_ip} IP, {actual_pitches} pitches) | {result_str}")

        if hit:
            stats['hits'] += 1
        else:
            stats['misses'] += 1

    conn.commit()
    conn.close()

    # Print summary
    total = stats['hits'] + stats['misses']
    hit_rate = stats['hits'] / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Resolved: {total}")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit Rate: {hit_rate:.1f}%")
    print(f"Voided: {stats['voided']}")
    print(f"Errors: {stats['errors']}")

    return stats


def get_mlb_overall_stats():
    """Get overall MLB prediction statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'voided' THEN 1 ELSE 0 END) as voided,
            SUM(hit) as hits
        FROM mlb_predictions
    """)

    row = cursor.fetchone()
    conn.close()

    total, resolved, pending, voided, hits = row
    hit_rate = hits / resolved * 100 if resolved > 0 else 0

    return {
        'total': total or 0,
        'resolved': resolved or 0,
        'pending': pending or 0,
        'voided': voided or 0,
        'hits': hits or 0,
        'hit_rate': round(hit_rate, 1)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = None  # Yesterday

    sync_mlb_results(target_date)

    print("\n" + "="*60)
    print("OVERALL MLB STATS")
    print("="*60)
    stats = get_mlb_overall_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
