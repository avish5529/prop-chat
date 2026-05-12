"""
Prepare Training Data V2 for CatBoost Model
============================================
Adds new features focused on:
1. Line Quality (mispriced lines)
2. Matchup Edges (player vs opponent patterns)

New features:
- line_vs_last_5: closing_line - last_5_avg
- line_difficulty: (closing_line - season_avg) / std_dev (z-score)
- consistency: inverse of coefficient of variation (season_avg / std_dev)
- avg_vs_opponent: player's historical average vs this specific opponent
- opp_def_rating: opponent team's defensive rating
- opp_pace: opponent team's pace
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

# Try to import nba_api for team stats
try:
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("Warning: nba_api not available. Using cached team stats if available.")

# Headers to avoid NBA API blocks
NBA_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
}

def load_backtest_data(filepath='backtest_data.csv'):
    """Load raw backtest data."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} rows")

    # Filter to rows with closing_line
    df = df[df['closing_line'].notna()].copy()
    print(f"  After filtering for closing_line: {len(df):,} rows")

    # Sort by date for temporal processing
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['player_name', 'game_date']).reset_index(drop=True)

    return df

def compute_line_quality_features(df):
    """
    Compute line quality features that indicate mispriced lines.

    Features:
    - line_vs_last_5: How far is line from recent form?
    - line_difficulty: Z-score of line vs season average
    - consistency: How consistent is the player? (higher = more predictable)
    """
    print("\nComputing line quality features...")

    # line_vs_last_5: difference between line and last 5 game avg
    # Positive = line above recent form (favors under)
    # Negative = line below recent form (favors over)
    df['line_vs_last_5'] = df['closing_line'] - df['last_5_avg']

    # line_difficulty: z-score of line
    # Positive = line above season average (harder to hit over)
    # Negative = line below season average (easier to hit over)
    df['line_difficulty'] = np.where(
        df['std_dev'] > 0,
        (df['closing_line'] - df['season_avg']) / df['std_dev'],
        0
    )

    # consistency: coefficient of variation (lower = more consistent)
    # We'll use inverse so higher = more predictable
    # CV = std_dev / mean, so consistency = mean / std_dev
    df['consistency'] = np.where(
        df['std_dev'] > 0,
        df['season_avg'] / df['std_dev'],
        df['season_avg']  # If no std_dev, use season_avg as proxy
    )

    # Cap extreme values
    df['line_difficulty'] = df['line_difficulty'].clip(-5, 5)
    df['consistency'] = df['consistency'].clip(0, 20)

    print(f"  line_vs_last_5: mean={df['line_vs_last_5'].mean():.2f}, std={df['line_vs_last_5'].std():.2f}")
    print(f"  line_difficulty: mean={df['line_difficulty'].mean():.2f}, std={df['line_difficulty'].std():.2f}")
    print(f"  consistency: mean={df['consistency'].mean():.2f}, std={df['consistency'].std():.2f}")

    return df

def compute_avg_vs_opponent(df):
    """
    Compute player's historical average vs each opponent.

    For each row, looks at all PRIOR games of that player vs that opponent
    to compute rolling average. This avoids data leakage.
    """
    print("\nComputing avg_vs_opponent (player historical average vs opponent)...")

    # Initialize column
    df['avg_vs_opponent'] = np.nan
    df['games_vs_opponent'] = 0

    # Group by player-opponent pairs
    unique_pairs = df.groupby(['player_name', 'opponent_team']).size().reset_index(name='count')
    print(f"  Processing {len(unique_pairs):,} unique player-opponent pairs...")

    # For each player
    for player in df['player_name'].unique():
        player_df = df[df['player_name'] == player].copy()
        player_idx = player_df.index

        for opponent in player_df['opponent_team'].unique():
            # Get all games vs this opponent for this player
            mask = (df['player_name'] == player) & (df['opponent_team'] == opponent)
            opp_games = df.loc[mask].sort_values('game_date')

            if len(opp_games) == 0:
                continue

            # Calculate rolling average of prior games (excluding current)
            for i, (idx, row) in enumerate(opp_games.iterrows()):
                prior_games = opp_games.iloc[:i]

                if len(prior_games) > 0:
                    avg = prior_games['actual_result'].mean()
                    df.loc[idx, 'avg_vs_opponent'] = avg
                    df.loc[idx, 'games_vs_opponent'] = len(prior_games)

    # Fill NaN with season_avg (first matchup against opponent)
    df['avg_vs_opponent'] = df['avg_vs_opponent'].fillna(df['season_avg'])

    # Stats
    has_history = df['games_vs_opponent'] > 0
    print(f"  Rows with prior matchup history: {has_history.sum():,} ({100*has_history.mean():.1f}%)")
    print(f"  avg_vs_opponent: mean={df['avg_vs_opponent'].mean():.2f}, std={df['avg_vs_opponent'].std():.2f}")

    return df

def fetch_team_stats_from_api():
    """Fetch team defensive rating and pace from NBA API."""
    if not NBA_API_AVAILABLE:
        return None

    print("\nFetching team stats from NBA API...")

    try:
        # Get team stats for current season
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season='2025-26',
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            headers=NBA_API_HEADERS,
            timeout=60
        )
        time.sleep(1)  # Rate limiting

        stats_df = team_stats.get_data_frames()[0]

        # Create mapping dict
        team_data = {}
        for _, row in stats_df.iterrows():
            team_abbrev = row['TEAM_ABBREVIATION']
            team_data[team_abbrev] = {
                'def_rating': row.get('DEF_RATING', row.get('DEF_RTG', 110)),  # Default 110 if missing
                'pace': row.get('PACE', 100)  # Default 100 if missing
            }

        print(f"  Fetched stats for {len(team_data)} teams")
        return team_data

    except Exception as e:
        print(f"  Error fetching from API: {e}")
        return None

def get_team_stats():
    """Get team stats from API or cached file."""
    cache_file = 'team_stats_cache.json'

    # Try to load from cache first
    try:
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            print(f"Loaded team stats from cache ({len(cached)} teams)")
            return cached
    except FileNotFoundError:
        pass

    # Fetch from API
    stats = fetch_team_stats_from_api()

    if stats is None:
        # Fallback to hardcoded values (2025-26 approximate)
        print("Using hardcoded team stats fallback...")
        stats = {
            'ATL': {'def_rating': 115.0, 'pace': 100.5},
            'BOS': {'def_rating': 108.5, 'pace': 99.2},
            'BKN': {'def_rating': 117.0, 'pace': 98.8},
            'CHA': {'def_rating': 116.5, 'pace': 101.2},
            'CHI': {'def_rating': 114.0, 'pace': 99.5},
            'CLE': {'def_rating': 107.5, 'pace': 98.5},
            'DAL': {'def_rating': 112.0, 'pace': 100.0},
            'DEN': {'def_rating': 111.5, 'pace': 99.8},
            'DET': {'def_rating': 115.5, 'pace': 100.8},
            'GSW': {'def_rating': 110.0, 'pace': 101.5},
            'HOU': {'def_rating': 110.5, 'pace': 99.0},
            'IND': {'def_rating': 116.0, 'pace': 102.5},
            'LAC': {'def_rating': 112.5, 'pace': 99.2},
            'LAL': {'def_rating': 111.0, 'pace': 100.2},
            'MEM': {'def_rating': 111.5, 'pace': 100.5},
            'MIA': {'def_rating': 112.0, 'pace': 98.5},
            'MIL': {'def_rating': 113.0, 'pace': 99.5},
            'MIN': {'def_rating': 109.5, 'pace': 98.0},
            'NOP': {'def_rating': 114.5, 'pace': 100.0},
            'NYK': {'def_rating': 109.0, 'pace': 99.0},
            'OKC': {'def_rating': 106.5, 'pace': 99.5},
            'ORL': {'def_rating': 107.0, 'pace': 97.5},
            'PHI': {'def_rating': 110.5, 'pace': 99.8},
            'PHX': {'def_rating': 113.5, 'pace': 100.5},
            'POR': {'def_rating': 118.0, 'pace': 99.0},
            'SAC': {'def_rating': 114.0, 'pace': 101.0},
            'SAS': {'def_rating': 117.5, 'pace': 99.5},
            'TOR': {'def_rating': 115.0, 'pace': 99.2},
            'UTA': {'def_rating': 116.5, 'pace': 98.5},
            'WAS': {'def_rating': 119.0, 'pace': 100.0},
        }

    # Cache for future use
    with open(cache_file, 'w') as f:
        json.dump(stats, f, indent=2)
        print(f"Saved team stats to {cache_file}")

    return stats

def add_team_stats(df, team_stats):
    """Add opponent defensive rating and pace to dataframe."""
    print("\nAdding team stats (opp_def_rating, opp_pace)...")

    # Map abbreviations (some teams have alternate names)
    abbrev_map = {
        'LA Clippers': 'LAC',
        'LA Lakers': 'LAL',
        'L.A. Clippers': 'LAC',
        'L.A. Lakers': 'LAL',
        'New York': 'NYK',
        'Golden State': 'GSW',
        'San Antonio': 'SAS',
        'Oklahoma City': 'OKC',
        'New Orleans': 'NOP',
    }

    def get_team_abbrev(team_name):
        """Convert team name to abbreviation."""
        if team_name in abbrev_map:
            return abbrev_map[team_name]
        # Try first 3 letters uppercase
        if team_name.upper()[:3] in team_stats:
            return team_name.upper()[:3]
        return team_name.upper()[:3]

    df['opp_def_rating'] = df['opponent_team'].apply(
        lambda x: team_stats.get(get_team_abbrev(x), {}).get('def_rating', 112.0)
    )

    df['opp_pace'] = df['opponent_team'].apply(
        lambda x: team_stats.get(get_team_abbrev(x), {}).get('pace', 100.0)
    )

    print(f"  opp_def_rating: mean={df['opp_def_rating'].mean():.1f}, std={df['opp_def_rating'].std():.2f}")
    print(f"  opp_pace: mean={df['opp_pace'].mean():.1f}, std={df['opp_pace'].std():.2f}")

    return df

def compute_target(df):
    """Compute binary target: 1 if actual > line (over hits), 0 otherwise."""
    print("\nComputing target variable (hit)...")
    df['hit'] = (df['actual_result'] > df['closing_line']).astype(int)

    over_rate = df['hit'].mean()
    print(f"  Over hit rate: {100*over_rate:.1f}%")
    print(f"  Under hit rate: {100*(1-over_rate):.1f}%")

    return df

def temporal_split(df, train_end='2026-01-20', val_end='2026-01-31'):
    """
    Split data temporally (no random splitting for time series).

    Training:   Oct 22, 2025 - Jan 20, 2026 (~75%)
    Validation: Jan 21, 2026 - Jan 31, 2026 (~10%)
    Test:       Feb 1, 2026 - Feb 9, 2026 (~15%)
    """
    print("\nPerforming temporal train/val/test split...")

    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    train_mask = df['game_date'] <= train_end_dt
    val_mask = (df['game_date'] > train_end_dt) & (df['game_date'] <= val_end_dt)
    test_mask = df['game_date'] > val_end_dt

    print(f"  Training: {train_mask.sum():,} rows ({100*train_mask.mean():.1f}%)")
    print(f"  Validation: {val_mask.sum():,} rows ({100*val_mask.mean():.1f}%)")
    print(f"  Test: {test_mask.sum():,} rows ({100*test_mask.mean():.1f}%)")

    df['split'] = 'train'
    df.loc[val_mask, 'split'] = 'val'
    df.loc[test_mask, 'split'] = 'test'

    return df

def save_training_data(df, output_file='training_data_v2.csv'):
    """Save processed training data."""
    print(f"\nSaving to {output_file}...")

    # Select columns for training
    feature_cols = [
        # IDs and metadata
        'id', 'player_name', 'game_date', 'opponent_team', 'prop_type', 'split',

        # Target
        'hit', 'actual_result', 'closing_line',

        # Existing features
        'is_home', 'season_avg', 'last_10_avg', 'last_5_avg', 'std_dev',
        'games_played', 'minutes_avg', 'days_rest', 'is_b2b',

        # Line quality features (NEW)
        'line_vs_season',  # from original training_data.py
        'line_vs_recent',  # from original training_data.py
        'line_vs_last_5',  # NEW
        'line_difficulty', # NEW
        'consistency',     # NEW

        # Matchup features (NEW)
        'avg_vs_opponent',    # NEW
        'games_vs_opponent',  # NEW
        'opp_def_rating',     # NEW
        'opp_pace',           # NEW
    ]

    # Add existing line_vs features if not present
    if 'line_vs_season' not in df.columns:
        df['line_vs_season'] = df['closing_line'] - df['season_avg']
    if 'line_vs_recent' not in df.columns:
        df['line_vs_recent'] = df['closing_line'] - df['last_10_avg']

    # Ensure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"  Warning: Column {col} not found, will be NaN")
            df[col] = np.nan

    # Save
    df_out = df[feature_cols].copy()
    df_out.to_csv(output_file, index=False)
    print(f"  Saved {len(df_out):,} rows with {len(feature_cols)} columns")

    # Also save feature column config
    feature_config = {
        'categorical_features': ['player_name', 'opponent_team', 'prop_type'],
        'numeric_features': [
            'closing_line', 'season_avg', 'last_10_avg', 'last_5_avg', 'std_dev',
            'minutes_avg', 'days_rest', 'games_played',
            'line_vs_season', 'line_vs_recent', 'line_vs_last_5',
            'line_difficulty', 'consistency',
            'avg_vs_opponent', 'games_vs_opponent',
            'opp_def_rating', 'opp_pace'
        ],
        'binary_features': ['is_home', 'is_b2b'],
        'target': 'hit',
        'version': 2,
        'created_at': datetime.now().isoformat()
    }

    with open('feature_columns_v2.json', 'w') as f:
        json.dump(feature_config, f, indent=2)
    print("  Saved feature configuration to feature_columns_v2.json")

    return df_out

def analyze_features(df):
    """Print feature analysis and correlations with target."""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)

    # Feature correlations with target
    numeric_cols = [
        'line_vs_season', 'line_vs_recent', 'line_vs_last_5',
        'line_difficulty', 'consistency',
        'avg_vs_opponent', 'opp_def_rating', 'opp_pace',
        'season_avg', 'last_10_avg', 'last_5_avg', 'std_dev'
    ]

    print("\nFeature correlations with 'hit' (over):")
    print("-" * 40)

    correlations = []
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df['hit'])
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, corr in correlations:
        direction = "↑ favors OVER" if corr > 0 else "↓ favors UNDER"
        print(f"  {col:20s}: {corr:+.4f}  {direction}")

    # Prop type breakdown
    print("\nOver hit rate by prop_type:")
    print("-" * 40)
    prop_stats = df.groupby('prop_type')['hit'].agg(['mean', 'count'])
    for prop_type, row in prop_stats.iterrows():
        print(f"  {prop_type:10s}: {100*row['mean']:.1f}% over ({int(row['count']):,} rows)")

    print("\n")

def main():
    """Main pipeline."""
    print("="*60)
    print("PREPARE TRAINING DATA V2")
    print("="*60)

    # Load data
    df = load_backtest_data()

    # Compute features
    df = compute_line_quality_features(df)
    df = compute_avg_vs_opponent(df)

    # Add team stats
    team_stats = get_team_stats()
    df = add_team_stats(df, team_stats)

    # Compute target
    df = compute_target(df)

    # Temporal split
    df = temporal_split(df)

    # Analyze features
    analyze_features(df)

    # Save
    df_out = save_training_data(df)

    print("="*60)
    print("DONE")
    print("="*60)

    return df_out

if __name__ == '__main__':
    main()
