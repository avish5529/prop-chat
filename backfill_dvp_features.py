"""
Backfill DvP features for all live predictions.

This script:
1. Loads player positions from cache
2. Loads DvP data from cache
3. For each prediction, adds player_position, dvp_rank, dvp_allowed

Run with: python backfill_dvp_features.py
"""

import sqlite3
import json

DB_PATH = "predictions.db"
POSITIONS_CACHE = "player_positions_cache.json"
DVP_CACHE = "dvp_cache.json"

# Prop type to DvP stat mapping
PROP_TO_STAT = {
    'points': 'pts',
    'rebounds': 'reb',
    'assists': 'ast',
    'pra': ['pts', 'reb', 'ast'],  # Sum of all three
    'pr': ['pts', 'reb'],
    'pa': ['pts', 'ast'],
    'ra': ['reb', 'ast'],
}


def load_positions():
    """Load player positions from cache."""
    try:
        with open(POSITIONS_CACHE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading positions cache: {e}")
        return {}


def load_dvp():
    """Load DvP data from cache."""
    try:
        with open(DVP_CACHE, 'r') as f:
            data = json.load(f)
        return data.get('teams', {})
    except Exception as e:
        print(f"Error loading DvP cache: {e}")
        return {}


def normalize_team_abbrev(team_str):
    """Normalize team string to match DvP cache keys."""
    # DvP cache uses 3-letter abbreviations
    TEAM_MAP = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
        'LA Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }

    if not team_str:
        return None

    # Already an abbreviation
    if len(team_str) == 3:
        return team_str.upper()

    # Full team name
    if team_str in TEAM_MAP:
        return TEAM_MAP[team_str]

    # Try partial match
    team_lower = team_str.lower()
    for name, abbrev in TEAM_MAP.items():
        if team_lower in name.lower():
            return abbrev

    return team_str.upper()[:3]


def get_dvp_data(dvp_cache, opponent, position, prop_type):
    """
    Get DvP rank and allowed stat for a specific matchup.

    Returns: (dvp_rank, dvp_allowed) or (None, None) if not found
    """
    opponent_abbrev = normalize_team_abbrev(opponent)

    if not opponent_abbrev or opponent_abbrev not in dvp_cache:
        return None, None

    team_dvp = dvp_cache[opponent_abbrev]

    if position not in team_dvp:
        return None, None

    pos_dvp = team_dvp[position]

    # Get the stat(s) for this prop type
    stat_keys = PROP_TO_STAT.get(prop_type)
    if not stat_keys:
        return None, None

    # Handle combo props (sum multiple stats)
    if isinstance(stat_keys, list):
        total_allowed = 0
        avg_rank = 0
        for stat in stat_keys:
            allowed_key = f'{stat}_allowed'
            rank_key = f'{stat}_rank'
            if allowed_key in pos_dvp:
                total_allowed += pos_dvp[allowed_key]
            if rank_key in pos_dvp:
                avg_rank += pos_dvp[rank_key]
        avg_rank = avg_rank // len(stat_keys)  # Average rank for combo
        return avg_rank, round(total_allowed, 1)
    else:
        # Single stat
        allowed_key = f'{stat_keys}_allowed'
        rank_key = f'{stat_keys}_rank'
        allowed = pos_dvp.get(allowed_key)
        rank = pos_dvp.get(rank_key)
        return rank, round(allowed, 1) if allowed else None


def get_predictions_to_backfill():
    """Get all resolved predictions missing DvP data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, player_name, opponent_team, prop_type
        FROM predictions
        WHERE status = 'resolved'
        AND (player_position IS NULL OR dvp_rank IS NULL OR dvp_allowed IS NULL)
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return rows


def update_prediction(pred_id, position, dvp_rank, dvp_allowed):
    """Update a single prediction with DvP features."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE predictions
        SET player_position = ?, dvp_rank = ?, dvp_allowed = ?
        WHERE id = ?
    """, (position, dvp_rank, dvp_allowed, pred_id))
    conn.commit()
    conn.close()


def backfill_all():
    """Main backfill function."""
    print("=" * 60)
    print("BACKFILLING DvP FEATURES")
    print("=" * 60)

    # Load caches
    positions = load_positions()
    dvp_cache = load_dvp()

    print(f"\nLoaded {len(positions)} player positions")
    print(f"Loaded DvP data for {len(dvp_cache)} teams")

    # Get predictions to backfill
    predictions = get_predictions_to_backfill()
    print(f"\nPredictions to backfill: {len(predictions)}")

    if not predictions:
        print("Nothing to backfill!")
        return

    # Track stats
    updated = 0
    missing_position = 0
    missing_dvp = 0

    for i, pred in enumerate(predictions):
        player_name = pred['player_name']
        opponent = pred['opponent_team']
        prop_type = pred['prop_type']

        # Get player position
        position = positions.get(player_name)

        if not position:
            missing_position += 1
            position = 'Unknown'

        # Get DvP data
        dvp_rank, dvp_allowed = get_dvp_data(dvp_cache, opponent, position, prop_type)

        if dvp_rank is None:
            missing_dvp += 1

        # Update prediction
        update_prediction(pred['id'], position, dvp_rank, dvp_allowed)
        updated += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(predictions)}...")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Updated: {updated}")
    print(f"Missing position (set to 'Unknown'): {missing_position}")
    print(f"Missing DvP data (NULL): {missing_dvp}")


if __name__ == "__main__":
    backfill_all()
