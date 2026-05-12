"""
Backfill avg_vs_opponent for backtest data.

This script:
1. Loads backtest_data.csv (15,453 rows, 50 players)
2. Fetches full game logs for each player via nba_api
3. Computes avg_vs_opponent for each row (using only games BEFORE that date)
4. Saves enriched data to backtest_data_enriched.csv

Usage: python backfill_backtest_avg_vs_opponent.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# Rate limiting
REQUEST_DELAY = 0.8  # seconds between API calls

# NBA API headers
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}

# Team abbreviation mapping (standardize to NBA API format)
TEAM_ABBREV_MAP = {
    'PHX': 'PHO',  # Phoenix
    'CHA': 'CHO',  # Charlotte (sometimes)
    'BKN': 'BRK',  # Brooklyn
    'NYK': 'NYK',
    'NOP': 'NOP',
    'SAS': 'SAS',
    'GSW': 'GSW',
    'LAL': 'LAL',
    'LAC': 'LAC',
}

def normalize_team(team):
    """Normalize team abbreviation."""
    team = team.upper().strip()
    return TEAM_ABBREV_MAP.get(team, team)


def find_player_id(player_name):
    """Find NBA player ID by name."""
    # Try exact match first
    player_list = players.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]['id']

    # Try partial match
    all_players = players.get_players()
    name_lower = player_name.lower()

    for p in all_players:
        if name_lower in p['full_name'].lower():
            return p['id']

    # Handle special characters
    name_normalized = player_name.replace('ć', 'c').replace('č', 'c').replace('ö', 'o').replace('ü', 'u')
    for p in all_players:
        if name_normalized.lower() in p['full_name'].lower():
            return p['id']

    return None


def fetch_player_game_logs(player_id, seasons=['2025-26', '2024-25']):
    """Fetch all game logs for a player across specified seasons."""
    all_games = []

    for season in seasons:
        try:
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                headers=NBA_HEADERS,
                timeout=30
            )
            df = log.get_data_frames()[0]

            if not df.empty:
                # Parse dates
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
                # Extract opponent from MATCHUP (e.g., "HOU vs. LAC" -> "LAC" or "HOU @ LAC" -> "LAC")
                df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
                all_games.append(df)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"    Warning: Could not fetch {season} for player {player_id}: {e}")

    if all_games:
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()


def compute_avg_vs_opponent(game_logs, opponent, game_date, prop_type):
    """
    Compute average vs opponent using only games BEFORE the given date.

    Args:
        game_logs: DataFrame with player's game history
        opponent: Team abbreviation (e.g., 'LAC')
        game_date: Date of the game we're predicting
        prop_type: Type of prop (points, rebounds, assists, pra, pr, pa, ra)

    Returns:
        Average stat vs opponent, or None if no history
    """
    if game_logs.empty:
        return None

    # Filter to games before this date against this opponent
    opponent_normalized = normalize_team(opponent)

    mask = (game_logs['GAME_DATE'] < game_date) & (game_logs['OPPONENT'] == opponent_normalized)
    prior_games = game_logs[mask]

    # Also try without normalization
    if prior_games.empty:
        mask = (game_logs['GAME_DATE'] < game_date) & (game_logs['OPPONENT'] == opponent)
        prior_games = game_logs[mask]

    if prior_games.empty:
        return None

    # Compute the relevant stat
    if prop_type == 'points':
        return prior_games['PTS'].mean()
    elif prop_type == 'rebounds':
        return prior_games['REB'].mean()
    elif prop_type == 'assists':
        return prior_games['AST'].mean()
    elif prop_type == 'pra':
        return (prior_games['PTS'] + prior_games['REB'] + prior_games['AST']).mean()
    elif prop_type == 'pr':
        return (prior_games['PTS'] + prior_games['REB']).mean()
    elif prop_type == 'pa':
        return (prior_games['PTS'] + prior_games['AST']).mean()
    elif prop_type == 'ra':
        return (prior_games['REB'] + prior_games['AST']).mean()

    return None


def main():
    print("=" * 60)
    print("BACKFILLING avg_vs_opponent FOR BACKTEST DATA")
    print("=" * 60)

    # Load backtest data
    print("\n1. Loading backtest_data.csv...")
    df = pd.read_csv('backtest_data.csv')
    print(f"   Loaded {len(df)} rows")

    # Get unique players
    unique_players = df['player_name'].unique()
    print(f"   Found {len(unique_players)} unique players")

    # Parse dates
    df['game_date_parsed'] = pd.to_datetime(df['game_date'], format='%m/%d/%y')

    # Fetch game logs for all players
    print("\n2. Fetching game logs from NBA API...")
    player_game_logs = {}

    for i, player_name in enumerate(unique_players):
        print(f"   [{i+1}/{len(unique_players)}] {player_name}...", end=" ")

        player_id = find_player_id(player_name)
        if player_id is None:
            print("NOT FOUND")
            continue

        game_logs = fetch_player_game_logs(player_id)

        if not game_logs.empty:
            player_game_logs[player_name] = game_logs
            print(f"{len(game_logs)} games")
        else:
            print("no games")

    print(f"\n   Fetched logs for {len(player_game_logs)} players")

    # Compute avg_vs_opponent for each row
    print("\n3. Computing avg_vs_opponent for each row...")

    avg_vs_opponent_values = []
    matches_found = 0

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"   Processing row {idx}/{len(df)}...")

        player_name = row['player_name']
        opponent = row['opponent_team']
        game_date = row['game_date_parsed']
        prop_type = row['prop_type']

        if player_name not in player_game_logs:
            avg_vs_opponent_values.append(None)
            continue

        avg = compute_avg_vs_opponent(
            player_game_logs[player_name],
            opponent,
            game_date,
            prop_type
        )

        if avg is not None:
            matches_found += 1

        avg_vs_opponent_values.append(avg)

    df['avg_vs_opponent'] = avg_vs_opponent_values

    # Fill missing with season_avg (fallback)
    missing_count = df['avg_vs_opponent'].isna().sum()
    df['avg_vs_opponent'] = df['avg_vs_opponent'].fillna(df['season_avg'])

    print(f"\n   Found historical matchup data for {matches_found} rows ({100*matches_found/len(df):.1f}%)")
    print(f"   Filled {missing_count} missing values with season_avg")

    # Drop helper column
    df = df.drop(columns=['game_date_parsed'])

    # Save enriched data
    output_path = 'backtest_data_enriched.csv'
    print(f"\n4. Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"   Saved {len(df)} rows")

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Rows with matchup history: {matches_found} ({100*matches_found/len(df):.1f}%)")
    print(f"Rows using season_avg fallback: {missing_count} ({100*missing_count/len(df):.1f}%)")
    print(f"\navg_vs_opponent stats:")
    print(f"  Mean: {df['avg_vs_opponent'].mean():.2f}")
    print(f"  Std:  {df['avg_vs_opponent'].std():.2f}")
    print(f"  Min:  {df['avg_vs_opponent'].min():.2f}")
    print(f"  Max:  {df['avg_vs_opponent'].max():.2f}")


if __name__ == "__main__":
    main()
