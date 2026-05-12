"""
Backfill dvp_rank, dvp_allowed, and player_position for backtest data.

Uses dvp.xlsx (position-specific DvP data from FantasyPros) and NBA API for player positions.

Usage: python backfill_backtest_dvp.py
"""

import pandas as pd
import numpy as np
import time

from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo

# Rate limiting for NBA API
REQUEST_DELAY = 0.6

NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}

# Team name to abbreviation mapping
TEAM_NAME_TO_ABBREV = {
    'atlanta hawks': 'ATL',
    'boston celtics': 'BOS',
    'brooklyn nets': 'BKN',
    'charlotte hornets': 'CHA',
    'chicago bulls': 'CHI',
    'cleveland cavaliers': 'CLE',
    'dallas mavericks': 'DAL',
    'denver nuggets': 'DEN',
    'detroit pistons': 'DET',
    'golden state warriors': 'GSW',
    'houston rockets': 'HOU',
    'indiana pacers': 'IND',
    'la clippers': 'LAC',
    'los angeles clippers': 'LAC',
    'los angeles lakers': 'LAL',
    'la lakers': 'LAL',
    'memphis grizzlies': 'MEM',
    'miami heat': 'MIA',
    'milwaukee bucks': 'MIL',
    'minnesota timberwolves': 'MIN',
    'new orleans pelicans': 'NOP',
    'new york knicks': 'NYK',
    'oklahoma city thunder': 'OKC',
    'orlando magic': 'ORL',
    'philadelphia 76ers': 'PHI',
    'phoenix suns': 'PHX',
    'portland trail blazers': 'POR',
    'sacramento kings': 'SAC',
    'san antonio spurs': 'SAS',
    'toronto raptors': 'TOR',
    'utah jazz': 'UTA',
    'washington wizards': 'WAS',
}

# Abbreviation normalization
ABBREV_MAP = {
    'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'BRK': 'BKN',
    'CHA': 'CHA', 'CHO': 'CHA', 'CHI': 'CHI', 'CLE': 'CLE',
    'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GSW': 'GSW', 'GS': 'GSW',
    'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL',
    'MEM': 'MEM', 'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN',
    'NOP': 'NOP', 'NOR': 'NOP', 'NO': 'NOP', 'NOH': 'NOP',
    'NYK': 'NYK', 'NY': 'NYK',
    'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHX', 'PHO': 'PHX',
    'POR': 'POR', 'SAC': 'SAC', 'SAS': 'SAS', 'SA': 'SAS',
    'TOR': 'TOR', 'UTA': 'UTA', 'UTH': 'UTA',
    'WAS': 'WAS', 'WSH': 'WAS',
}


def parse_team_name(raw_name):
    """Parse team name from DvP data to get abbreviation."""
    raw_name = str(raw_name).strip()

    # Check if it starts with a known abbreviation (e.g., "DALDallas Mavericks")
    for prefix, abbrev in ABBREV_MAP.items():
        if raw_name.upper().startswith(prefix):
            return abbrev

    # Try full name match
    name_lower = raw_name.lower()
    for team_name, abbrev in TEAM_NAME_TO_ABBREV.items():
        if team_name in name_lower:
            return abbrev

    print(f"  Warning: Could not parse team: {raw_name}")
    return None


def normalize_opponent(opponent):
    """Normalize opponent abbreviation."""
    if not opponent:
        return None
    opp = str(opponent).strip().upper()
    return ABBREV_MAP.get(opp, opp)


def load_dvp_data(xlsx_path='dvp.xlsx'):
    """Load DvP data from Excel file and create lookup tables."""
    print("Loading DvP data from Excel...")

    xlsx = pd.ExcelFile(xlsx_path)

    # Dictionary: (team_abbrev, position) -> {rank, pts, reb, ast}
    dvp_lookup = {}

    for position in ['PG', 'SG', 'SF', 'PF', 'C']:
        df = pd.read_excel(xlsx, sheet_name=position, header=None)

        for rank, row in df.iterrows():
            team_name = row[0]
            team_abbrev = parse_team_name(team_name)

            if team_abbrev:
                dvp_lookup[(team_abbrev, position)] = {
                    'rank': rank + 1,  # 1-indexed rank (1 = worst defense)
                    'pts': float(row[2]) if pd.notna(row[2]) else 20.0,
                    'reb': float(row[3]) if pd.notna(row[3]) else 8.0,
                    'ast': float(row[4]) if pd.notna(row[4]) else 5.0,
                }

        print(f"  {position}: {len([k for k in dvp_lookup if k[1] == position])} teams")

    return dvp_lookup


def find_player_id(player_name):
    """Find NBA player ID by name."""
    player_list = players.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]['id']

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


def get_player_position(player_id):
    """Get player's primary position from NBA API."""
    try:
        info = commonplayerinfo.CommonPlayerInfo(
            player_id=player_id,
            headers=NBA_HEADERS,
            timeout=30
        )
        df = info.get_data_frames()[0]

        if not df.empty:
            position = df['POSITION'].iloc[0]
            if position:
                pos = position.split('-')[0].strip()
                # Map to DvP positions
                pos_map = {
                    'Guard': 'PG',
                    'Point Guard': 'PG',
                    'Shooting Guard': 'SG',
                    'Forward': 'SF',
                    'Small Forward': 'SF',
                    'Power Forward': 'PF',
                    'Center': 'C',
                    'G': 'PG',
                    'F': 'SF',
                    'C': 'C',
                }
                return pos_map.get(pos, pos)

        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def fetch_player_positions(player_names):
    """Fetch positions for all players."""
    print("\nFetching player positions from NBA API...")

    positions = {}

    for i, name in enumerate(player_names):
        print(f"  [{i+1}/{len(player_names)}] {name}...", end=" ")

        player_id = find_player_id(name)
        if player_id is None:
            print("NOT FOUND - defaulting to SF")
            positions[name] = 'SF'
            continue

        position = get_player_position(player_id)
        if position:
            positions[name] = position
            print(position)
        else:
            positions[name] = 'SF'
            print("DEFAULT (SF)")

        time.sleep(REQUEST_DELAY)

    return positions


def get_dvp_for_prop(dvp_lookup, opponent, position, prop_type):
    """
    Get DvP rank and allowed stat for a specific prop type.

    Returns: (dvp_rank, dvp_allowed)
    """
    opponent_norm = normalize_opponent(opponent)
    key = (opponent_norm, position)

    if key not in dvp_lookup:
        return None, None

    data = dvp_lookup[key]
    rank = data['rank']

    # Get the relevant stat based on prop_type
    if prop_type == 'points':
        allowed = data['pts']
    elif prop_type == 'rebounds':
        allowed = data['reb']
    elif prop_type == 'assists':
        allowed = data['ast']
    elif prop_type == 'pra':
        allowed = data['pts'] + data['reb'] + data['ast']
    elif prop_type == 'pr':
        allowed = data['pts'] + data['reb']
    elif prop_type == 'pa':
        allowed = data['pts'] + data['ast']
    elif prop_type == 'ra':
        allowed = data['reb'] + data['ast']
    else:
        allowed = data['pts']

    return rank, allowed


def main():
    print("=" * 60)
    print("BACKFILLING DvP FEATURES FOR BACKTEST DATA")
    print("=" * 60)

    # Load DvP data from Excel
    dvp_lookup = load_dvp_data()
    print(f"\nTotal DvP entries: {len(dvp_lookup)}")

    # Load enriched backtest data
    print("\nLoading backtest_data_enriched.csv...")
    df = pd.read_csv('backtest_data_enriched.csv')
    print(f"Loaded {len(df)} rows")

    # Get unique players
    unique_players = df['player_name'].unique()
    print(f"Found {len(unique_players)} unique players")

    # Fetch player positions
    player_positions = fetch_player_positions(unique_players)

    # Add position column
    df['player_position'] = df['player_name'].map(player_positions)

    # Compute DvP features for each row
    print("\nComputing DvP features for each row...")

    dvp_ranks = []
    dvp_alloweds = []
    matches_found = 0

    for idx, row in df.iterrows():
        if idx % 2000 == 0:
            print(f"  Processing row {idx}/{len(df)}...")

        opponent = row['opponent_team']
        position = row['player_position']
        prop_type = row['prop_type']

        rank, allowed = get_dvp_for_prop(dvp_lookup, opponent, position, prop_type)

        if rank is not None:
            matches_found += 1

        dvp_ranks.append(rank)
        dvp_alloweds.append(allowed)

    df['dvp_rank'] = dvp_ranks
    df['dvp_allowed'] = dvp_alloweds

    # Fill missing values
    missing_rank = df['dvp_rank'].isna().sum()
    df['dvp_rank'] = df['dvp_rank'].fillna(15)  # Middle rank
    df['dvp_allowed'] = df['dvp_allowed'].fillna(df['season_avg'])

    print(f"\n  Found DvP data for {matches_found} rows ({100*matches_found/len(df):.1f}%)")
    print(f"  Filled {missing_rank} missing values with defaults")

    # Save
    output_path = 'backtest_data_enriched.csv'
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows with new columns: player_position, dvp_rank, dvp_allowed")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Rows with DvP data: {matches_found} ({100*matches_found/len(df):.1f}%)")

    print("\nPosition distribution:")
    print(df['player_position'].value_counts().to_string())

    print("\ndvp_rank stats:")
    print(f"  Mean: {df['dvp_rank'].mean():.1f}")
    print(f"  Min:  {df['dvp_rank'].min():.0f}")
    print(f"  Max:  {df['dvp_rank'].max():.0f}")

    print("\ndvp_allowed stats:")
    print(f"  Mean: {df['dvp_allowed'].mean():.2f}")
    print(f"  Std:  {df['dvp_allowed'].std():.2f}")


if __name__ == "__main__":
    main()
