"""
Backfill avg_vs_opponent for all live predictions.

This script:
1. Fetches all resolved predictions missing avg_vs_opponent
2. For each player, gets their full game log
3. Computes historical average vs each opponent (games BEFORE prediction date)
4. Updates the database

Run with: python backfill_avg_vs_opponent.py
"""

import sqlite3
import time
from datetime import datetime
from collections import defaultdict
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd

DB_PATH = "predictions.db"

# NBA API headers to avoid rate limiting
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}

# Team name to abbreviation mapping
TEAM_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# Reverse mapping
ABBREV_TO_TEAM = {v: k for k, v in TEAM_ABBREV.items()}


def get_predictions_to_backfill():
    """Get all resolved predictions missing avg_vs_opponent."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, player_name, player_id, opponent_team, prop_type, game_date
        FROM predictions
        WHERE status = 'resolved'
        AND avg_vs_opponent IS NULL
        ORDER BY player_name, game_date
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return rows


def find_player_id(player_name):
    """Find NBA player ID from name."""
    player_list = players.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]['id']

    # Try partial match
    all_players = players.get_players()
    for p in all_players:
        if player_name.lower() in p['full_name'].lower():
            return p['id']

    return None


def get_player_game_log(player_id, season="2025-26"):
    """Fetch player's full game log for the season."""
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            headers=NBA_HEADERS,
            timeout=30
        )
        df = log.get_data_frames()[0]

        # Parse dates
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')

        # Extract opponent from MATCHUP (e.g., "LAL vs. BOS" or "LAL @ BOS")
        def extract_opponent(matchup):
            if ' vs. ' in matchup:
                return matchup.split(' vs. ')[1]
            elif ' @ ' in matchup:
                return matchup.split(' @ ')[1]
            return None

        df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)

        return df
    except Exception as e:
        print(f"  Error fetching game log: {e}")
        return None


def compute_avg_vs_opponent(game_log, opponent_abbrev, prop_type, before_date):
    """
    Compute player's average vs opponent for games BEFORE the prediction date.

    Args:
        game_log: DataFrame with player's games
        opponent_abbrev: 3-letter team abbreviation (e.g., "BOS")
        prop_type: points, rebounds, assists, pra, pr, pa, ra
        before_date: Only include games before this date

    Returns:
        float: Average stat value, or None if no historical games
    """
    if game_log is None or len(game_log) == 0:
        return None

    # Filter for opponent and games before prediction date
    before_dt = pd.to_datetime(before_date)
    mask = (game_log['OPPONENT'] == opponent_abbrev) & (game_log['GAME_DATE'] < before_dt)
    opponent_games = game_log[mask]

    if len(opponent_games) == 0:
        return None

    # Compute the stat based on prop type
    stat_map = {
        'points': opponent_games['PTS'].mean(),
        'rebounds': opponent_games['REB'].mean(),
        'assists': opponent_games['AST'].mean(),
        'pra': (opponent_games['PTS'] + opponent_games['REB'] + opponent_games['AST']).mean(),
        'pr': (opponent_games['PTS'] + opponent_games['REB']).mean(),
        'pa': (opponent_games['PTS'] + opponent_games['AST']).mean(),
        'ra': (opponent_games['REB'] + opponent_games['AST']).mean(),
    }

    return stat_map.get(prop_type)


def normalize_opponent(opponent_str):
    """Normalize opponent string to 3-letter abbreviation."""
    if not opponent_str:
        return None

    # Already an abbreviation
    if len(opponent_str) == 3 and opponent_str.upper() in ABBREV_TO_TEAM:
        return opponent_str.upper()

    # Full team name
    if opponent_str in TEAM_ABBREV:
        return TEAM_ABBREV[opponent_str]

    # Try matching part of the name
    opponent_lower = opponent_str.lower()
    for team_name, abbrev in TEAM_ABBREV.items():
        if opponent_lower in team_name.lower() or team_name.lower() in opponent_lower:
            return abbrev

    # Check if it contains a city name
    for team_name, abbrev in TEAM_ABBREV.items():
        city = team_name.split()[0].lower()
        if city in opponent_lower:
            return abbrev

    return opponent_str.upper()[:3] if opponent_str else None


def update_prediction(pred_id, avg_vs_opp):
    """Update a single prediction with avg_vs_opponent."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE predictions SET avg_vs_opponent = ? WHERE id = ?",
        (avg_vs_opp, pred_id)
    )
    conn.commit()
    conn.close()


def backfill_all():
    """Main backfill function."""
    print("=" * 60)
    print("BACKFILLING avg_vs_opponent")
    print("=" * 60)

    predictions = get_predictions_to_backfill()
    print(f"\nPredictions to backfill: {len(predictions)}")

    if not predictions:
        print("Nothing to backfill!")
        return

    # Group by player to minimize API calls
    by_player = defaultdict(list)
    for pred in predictions:
        by_player[pred['player_name']].append(pred)

    print(f"Unique players: {len(by_player)}")

    # Cache for game logs
    game_log_cache = {}

    total_updated = 0
    total_no_history = 0
    total_errors = 0

    for i, (player_name, player_preds) in enumerate(by_player.items()):
        print(f"\n[{i+1}/{len(by_player)}] {player_name} ({len(player_preds)} predictions)")

        # Find player ID
        player_id = player_preds[0].get('player_id')
        if not player_id:
            player_id = find_player_id(player_name)

        if not player_id:
            print(f"  Could not find player ID - skipping")
            total_errors += len(player_preds)
            continue

        # Fetch game log (with caching)
        if player_id not in game_log_cache:
            print(f"  Fetching game log...")
            game_log_cache[player_id] = get_player_game_log(player_id)
            time.sleep(0.6)  # Rate limit

        game_log = game_log_cache[player_id]

        if game_log is None:
            print(f"  No game log available - skipping")
            total_errors += len(player_preds)
            continue

        # Process each prediction for this player
        for pred in player_preds:
            opponent = normalize_opponent(pred['opponent_team'])
            prop_type = pred['prop_type']
            game_date = pred['game_date']

            avg = compute_avg_vs_opponent(game_log, opponent, prop_type, game_date)

            if avg is not None:
                update_prediction(pred['id'], round(avg, 2))
                total_updated += 1
                print(f"  [{pred['id']}] vs {opponent} {prop_type}: {avg:.1f}")
            else:
                # No historical games vs this opponent - use season average as fallback
                total_no_history += 1
                print(f"  [{pred['id']}] vs {opponent} {prop_type}: No history")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Updated: {total_updated}")
    print(f"No history (NULL): {total_no_history}")
    print(f"Errors: {total_errors}")


if __name__ == "__main__":
    backfill_all()
