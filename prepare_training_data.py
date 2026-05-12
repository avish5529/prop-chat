"""
Phase 1: Data Preparation & Feature Engineering for CatBoost
Loads backtest_data.csv and creates training_data.csv with derived features.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_and_prepare_data(input_path: str = "backtest_data.csv") -> pd.DataFrame:
    """Load backtest data and filter to rows with closing lines."""
    print(f"[Data Prep] Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"[Data Prep] Total rows: {len(df)}")

    # Filter to rows with closing_line (not NaN)
    df_with_lines = df[df["closing_line"].notna()].copy()
    print(f"[Data Prep] Rows with closing_line: {len(df_with_lines)}")

    return df_with_lines


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target variable: hit = 1 if actual_result > closing_line else 0"""
    df["hit"] = (df["actual_result"] > df["closing_line"]).astype(int)

    hit_rate = df["hit"].mean()
    print(f"[Data Prep] Target distribution: {hit_rate:.1%} overs hit")

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for CatBoost model."""

    # line_vs_season: How does the line compare to season average?
    # Positive = line is above season avg (harder over)
    df["line_vs_season"] = df["closing_line"] - df["season_avg"]

    # line_vs_recent: How does line compare to last 10 avg?
    # Positive = line is above recent form (harder over)
    df["line_vs_recent"] = df["closing_line"] - df["last_10_avg"]

    # form_trend: Is player trending up or down?
    # Positive = recent 5 games better than last 10 (hot streak)
    df["form_trend"] = df["last_5_avg"] - df["last_10_avg"]

    # line_difficulty: How many std devs is line from season avg?
    # Higher = harder to hit the over
    # Handle missing std_dev by using a reasonable default
    df["line_difficulty"] = np.where(
        df["std_dev"].notna() & (df["std_dev"] > 0),
        (df["closing_line"] - df["season_avg"]) / df["std_dev"],
        np.nan  # CatBoost handles NaN natively
    )

    # Additional useful features
    # games_experience: bucket into experience levels
    df["games_experience"] = pd.cut(
        df["games_played"],
        bins=[0, 10, 25, 50, 82],
        labels=["few", "moderate", "experienced", "veteran"]
    ).astype(str)

    # minutes_bucket: categorize expected minutes
    df["minutes_bucket"] = pd.cut(
        df["minutes_avg"],
        bins=[0, 20, 28, 34, 50],
        labels=["low", "medium", "high", "starter"]
    ).astype(str)

    print(f"[Data Prep] Added derived features:")
    print(f"  - line_vs_season: mean={df['line_vs_season'].mean():.2f}")
    print(f"  - line_vs_recent: mean={df['line_vs_recent'].mean():.2f}")
    print(f"  - form_trend: mean={df['form_trend'].mean():.2f}")
    print(f"  - line_difficulty: mean={df['line_difficulty'].dropna().mean():.2f}")

    return df


def create_temporal_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add split column for temporal train/val/test split."""

    # Parse game_date (format: "2/11/26" -> datetime)
    def parse_date(date_str):
        try:
            # Handle format like "2/11/26"
            parts = date_str.split("/")
            month, day = int(parts[0]), int(parts[1])
            year = int(parts[2])
            # Assume 2026 for now (adjust if needed)
            full_year = 2000 + year if year < 50 else 1900 + year
            return datetime(full_year, month, day)
        except:
            return None

    df["parsed_date"] = df["game_date"].apply(parse_date)

    # Define split boundaries
    # Training:   Oct 22, 2025 - Jan 20, 2026 (~75%)
    # Validation: Jan 21, 2026 - Jan 31, 2026 (~10%)
    # Test:       Feb 1, 2026 - Feb 9, 2026 (~15%)

    train_end = datetime(2026, 1, 20)
    val_end = datetime(2026, 1, 31)

    def assign_split(date):
        if date is None:
            return "train"  # Default to train if date parsing fails
        if date <= train_end:
            return "train"
        elif date <= val_end:
            return "val"
        else:
            return "test"

    df["split"] = df["parsed_date"].apply(assign_split)

    # Print split distribution
    split_counts = df["split"].value_counts()
    print(f"\n[Data Prep] Temporal split distribution:")
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        pct = count / len(df) * 100
        print(f"  - {split}: {count} rows ({pct:.1f}%)")

    return df


def save_training_data(df: pd.DataFrame, output_path: str = "training_data.csv"):
    """Save processed data to CSV."""

    # Select columns for training
    columns_to_keep = [
        # Identifiers
        "id", "player_name", "game_date", "opponent_team", "prop_type",
        # Target
        "hit",
        # Categorical features
        "is_home", "is_b2b", "games_experience", "minutes_bucket",
        # Numeric features
        "closing_line", "season_avg", "last_10_avg", "last_5_avg",
        "std_dev", "games_played", "minutes_avg", "days_rest",
        # Derived features
        "line_vs_season", "line_vs_recent", "form_trend", "line_difficulty",
        # Actual results (for validation)
        "actual_result",
        # Split assignment
        "split", "parsed_date"
    ]

    # Only keep columns that exist
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    df_out = df[existing_cols].copy()

    df_out.to_csv(output_path, index=False)
    print(f"\n[Data Prep] Saved {len(df_out)} rows to {output_path}")
    print(f"[Data Prep] Columns: {', '.join(existing_cols)}")

    return df_out


def print_feature_stats(df: pd.DataFrame):
    """Print summary statistics for key features."""
    print("\n" + "="*60)
    print("FEATURE SUMMARY STATISTICS")
    print("="*60)

    numeric_features = [
        "closing_line", "season_avg", "last_10_avg", "last_5_avg",
        "std_dev", "games_played", "minutes_avg", "days_rest",
        "line_vs_season", "line_vs_recent", "form_trend", "line_difficulty"
    ]

    for feat in numeric_features:
        if feat in df.columns:
            print(f"\n{feat}:")
            print(f"  min={df[feat].min():.2f}, max={df[feat].max():.2f}, "
                  f"mean={df[feat].mean():.2f}, std={df[feat].std():.2f}, "
                  f"missing={df[feat].isna().sum()}")

    categorical_features = ["player_name", "opponent_team", "prop_type", "is_home", "is_b2b"]
    print("\n" + "-"*60)
    print("CATEGORICAL FEATURE CARDINALITY")
    print("-"*60)

    for feat in categorical_features:
        if feat in df.columns:
            n_unique = df[feat].nunique()
            print(f"  {feat}: {n_unique} unique values")


def main():
    """Main execution pipeline."""
    print("="*60)
    print("PHASE 1: DATA PREPARATION & FEATURE ENGINEERING")
    print("="*60 + "\n")

    # Load data
    df = load_and_prepare_data("backtest_data.csv")

    # Compute target
    df = compute_target(df)

    # Add derived features
    df = compute_derived_features(df)

    # Create temporal splits
    df = create_temporal_splits(df)

    # Print feature stats
    print_feature_stats(df)

    # Save processed data
    df_out = save_training_data(df, "training_data.csv")

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)

    # Verification
    print(f"\nVerification:")
    print(f"  - Total rows with closing_line: {len(df_out)}")
    print(f"  - Target (hit) distribution: {df_out['hit'].value_counts().to_dict()}")
    print(f"  - Temporal splits: {df_out['split'].value_counts().to_dict()}")

    return df_out


if __name__ == "__main__":
    main()
