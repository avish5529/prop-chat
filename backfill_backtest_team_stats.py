"""
Backfill opp_def_rating and opp_pace for backtest data.

Uses NBA API to fetch team defensive rating and pace.

Usage: python backfill_backtest_team_stats.py
"""

import pandas as pd
import numpy as np
import time

from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams

# Rate limiting
REQUEST_DELAY = 1.0

NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}

# Team abbreviation normalization
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


def normalize_abbrev(abbrev):
    """Normalize team abbreviation."""
    if not abbrev:
        return None
    return ABBREV_MAP.get(abbrev.upper(), abbrev.upper())


# Team name to abbreviation mapping
TEAM_NAME_TO_ABBREV = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}


def fetch_team_stats(season='2025-26'):
    """Fetch team defensive rating and pace from NBA API."""
    print(f"  Fetching team stats for {season}...")

    try:
        # Fetch advanced team stats
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            headers=NBA_HEADERS,
            timeout=60
        )

        df = stats.get_data_frames()[0]

        if df.empty:
            print(f"    No data for {season}")
            return {}

        # Create lookup by team abbreviation (map from TEAM_NAME)
        team_stats = {}
        for _, row in df.iterrows():
            team_name = row['TEAM_NAME']
            abbrev = TEAM_NAME_TO_ABBREV.get(team_name)

            if not abbrev:
                print(f"    Warning: Unknown team name: {team_name}")
                continue

            team_stats[abbrev] = {
                'def_rating': float(row['DEF_RATING']) if pd.notna(row.get('DEF_RATING')) else 110.0,
                'pace': float(row['PACE']) if pd.notna(row.get('PACE')) else 100.0,
            }

        print(f"    Got stats for {len(team_stats)} teams")
        return team_stats

    except Exception as e:
        print(f"    Error fetching {season}: {e}")
        return {}


def fetch_all_team_stats():
    """Fetch team stats for relevant seasons."""
    print("\nFetching team stats from NBA API...")

    all_stats = {}

    # Fetch for both seasons (backtest data spans Oct 2025 - Feb 2026)
    for season in ['2025-26', '2024-25']:
        stats = fetch_team_stats(season)

        # Merge into all_stats (2025-26 takes priority)
        for team, data in stats.items():
            if team not in all_stats:
                all_stats[team] = data

        time.sleep(REQUEST_DELAY)

    print(f"\nTotal teams with stats: {len(all_stats)}")
    return all_stats


def main():
    print("=" * 60)
    print("BACKFILLING TEAM STATS FOR BACKTEST DATA")
    print("(opp_def_rating and opp_pace)")
    print("=" * 60)

    # Fetch team stats
    team_stats = fetch_all_team_stats()

    if not team_stats:
        print("ERROR: Could not fetch team stats!")
        return

    # Show sample
    print("\nSample team stats:")
    for team in list(team_stats.keys())[:5]:
        print(f"  {team}: DEF_RTG={team_stats[team]['def_rating']:.1f}, PACE={team_stats[team]['pace']:.1f}")

    # Load enriched backtest data
    print("\nLoading backtest_data_enriched.csv...")
    df = pd.read_csv('backtest_data_enriched.csv')
    print(f"Loaded {len(df)} rows")

    # Get unique opponents
    unique_opponents = df['opponent_team'].unique()
    print(f"Found {len(unique_opponents)} unique opponents")

    # Map opponents to stats
    print("\nMapping opponent stats...")

    def_ratings = []
    paces = []
    matches_found = 0
    missing_teams = set()

    for idx, row in df.iterrows():
        opponent = normalize_abbrev(row['opponent_team'])

        if opponent in team_stats:
            def_ratings.append(team_stats[opponent]['def_rating'])
            paces.append(team_stats[opponent]['pace'])
            matches_found += 1
        else:
            # Try without normalization
            raw_opp = row['opponent_team']
            if raw_opp in team_stats:
                def_ratings.append(team_stats[raw_opp]['def_rating'])
                paces.append(team_stats[raw_opp]['pace'])
                matches_found += 1
            else:
                def_ratings.append(None)
                paces.append(None)
                missing_teams.add(opponent)

    df['opp_def_rating'] = def_ratings
    df['opp_pace'] = paces

    # Report missing teams
    if missing_teams:
        print(f"\n  Missing teams: {missing_teams}")

    # Fill missing with league averages
    missing_count = df['opp_def_rating'].isna().sum()
    df['opp_def_rating'] = df['opp_def_rating'].fillna(110.0)  # League avg ~110
    df['opp_pace'] = df['opp_pace'].fillna(100.0)  # League avg ~100

    print(f"\n  Found team stats for {matches_found} rows ({100*matches_found/len(df):.1f}%)")
    print(f"  Filled {missing_count} missing values with league averages")

    # Save
    output_path = 'backtest_data_enriched.csv'
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Rows with team stats: {matches_found} ({100*matches_found/len(df):.1f}%)")

    print("\nopp_def_rating stats:")
    print(f"  Mean: {df['opp_def_rating'].mean():.1f}")
    print(f"  Min:  {df['opp_def_rating'].min():.1f}")
    print(f"  Max:  {df['opp_def_rating'].max():.1f}")

    print("\nopp_pace stats:")
    print(f"  Mean: {df['opp_pace'].mean():.1f}")
    print(f"  Min:  {df['opp_pace'].min():.1f}")
    print(f"  Max:  {df['opp_pace'].max():.1f}")

    # Final column check
    print("\nFinal columns in enriched data:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
