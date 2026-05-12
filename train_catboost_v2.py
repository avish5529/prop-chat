"""
Train CatBoost Model V2
========================
New model focused on:
1. Line Quality features (is the line mispriced?)
2. Matchup features (how does player perform vs this opponent?)

Key changes from v1:
- REMOVED player_name as categorical (was 12% importance, causing overfitting)
- REDUCED prop_type importance by keeping as numeric indicator
- ADDED new features: line_vs_last_5, line_difficulty, consistency, avg_vs_opponent
- ADDED matchup features: opp_def_rating, opp_pace
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='training_data_v2.csv'):
    """Load prepared training data."""
    print("Loading training data...")
    df = pd.read_csv(filepath)
    print(f"  Total rows: {len(df):,}")

    # Check splits
    print(f"  Train: {(df['split']=='train').sum():,}")
    print(f"  Val: {(df['split']=='val').sum():,}")
    print(f"  Test: {(df['split']=='test').sum():,}")

    return df

def prepare_features(df):
    """
    Prepare features for CatBoost.

    Key design decisions:
    1. NO player_name as categorical - was causing overfitting to specific players
    2. prop_type kept but as a category (model will learn but won't dominate)
    3. Focus on LINE QUALITY features - these tell us if line is mispriced
    4. Focus on MATCHUP features - historical performance vs opponent

    The goal: predict whether the LINE is beatable, not whether the player is "good"
    """

    # Feature columns
    categorical_features = [
        'opponent_team',  # Keep opponent for matchup learning
        'prop_type',      # Keep but reduce importance via early stopping
    ]

    numeric_features = [
        # Line quality (PRIMARY)
        'line_vs_season',    # Line - season avg
        'line_vs_recent',    # Line - last 10 avg
        'line_vs_last_5',    # Line - last 5 avg (NEW)
        'line_difficulty',   # Z-score of line (NEW)
        'consistency',       # Player consistency (NEW)

        # Matchup (PRIMARY)
        'avg_vs_opponent',   # Historical avg vs this opponent (NEW)
        'opp_def_rating',    # Opponent defense rating (NEW)
        'opp_pace',          # Opponent pace (NEW)

        # Context
        'closing_line',      # The actual line
        'season_avg',        # Player's season average
        'last_10_avg',       # Recent form
        'last_5_avg',        # Very recent form

        # Situational
        'is_home',           # Home/away
        'is_b2b',            # Back-to-back
        'days_rest',         # Days since last game
        'minutes_avg',       # Expected minutes
        'games_played',      # Games played this season
    ]

    # Verify all features exist
    all_features = categorical_features + numeric_features
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"  Warning: Missing features: {missing}")
        for f in missing:
            df[f] = 0

    return categorical_features, numeric_features

def train_model(train_df, val_df, cat_features, num_features):
    """Train CatBoost classifier."""
    print("\nTraining CatBoost model...")

    all_features = cat_features + num_features

    X_train = train_df[all_features]
    y_train = train_df['hit']

    X_val = val_df[all_features]
    y_val = val_df['hit']

    # Convert categorical columns to string
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    # Create pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    # Model config - more regularization to prevent prop_type dominance
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,                    # Reduced from 6 to prevent overfitting
        l2_leaf_reg=5,              # L2 regularization
        min_data_in_leaf=20,        # Minimum samples in leaf
        cat_features=cat_features,
        nan_mode='Min',
        eval_metric='Logloss',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
        use_best_model=True,
    )

    # Train
    model.fit(train_pool, eval_set=val_pool)

    return model, all_features

def evaluate_model(model, df, features, cat_features, split_name):
    """Evaluate model on a dataset."""
    X = df[features].copy()
    y = df['hit']

    # Convert categorical columns to string
    for col in cat_features:
        X[col] = X[col].astype(str)

    pool = Pool(X, y, cat_features=cat_features)

    # Predictions
    y_pred_proba = model.predict_proba(pool)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    brier = brier_score_loss(y, y_pred_proba)
    logloss = log_loss(y, y_pred_proba)

    print(f"\n{split_name} Results:")
    print(f"  Accuracy: {100*accuracy:.1f}%")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {logloss:.4f}")

    return {
        'accuracy': accuracy,
        'brier_score': brier,
        'log_loss': logloss,
        'samples': len(y)
    }

def analyze_feature_importance(model, features):
    """Analyze and display feature importance."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)

    importance = model.get_feature_importance()
    feat_imp = list(zip(features, importance))
    feat_imp.sort(key=lambda x: x[1], reverse=True)

    total_imp = sum(importance)
    for feat, imp in feat_imp:
        pct = 100 * imp / total_imp
        bar = "█" * int(pct / 2)
        print(f"  {feat:20s}: {pct:5.1f}% {bar}")

    # Check if we've reduced prop_type dominance
    prop_type_imp = next((imp for feat, imp in feat_imp if feat == 'prop_type'), 0)
    print(f"\n  prop_type importance: {100*prop_type_imp/total_imp:.1f}% (target: <30%)")

    return feat_imp

def analyze_by_prop_type(model, df, features, cat_features):
    """Analyze accuracy by prop type."""
    print("\n" + "="*60)
    print("ACCURACY BY PROP TYPE")
    print("="*60)

    X = df[features].copy()
    for col in cat_features:
        X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=cat_features)
    y_pred_proba = model.predict_proba(pool)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    df_eval = df.copy()
    df_eval['pred_proba'] = y_pred_proba
    df_eval['pred'] = y_pred
    df_eval['correct'] = (df_eval['pred'] == df_eval['hit']).astype(int)

    # By prop type
    for prop_type in df_eval['prop_type'].unique():
        subset = df_eval[df_eval['prop_type'] == prop_type]
        acc = subset['correct'].mean()
        over_pred = (subset['pred'] == 1).mean()
        over_actual = subset['hit'].mean()
        print(f"  {prop_type:10s}: {100*acc:.1f}% accuracy | Pred over: {100*over_pred:.1f}% | Actual over: {100*over_actual:.1f}%")

def simulate_betting_roi(model, df, features, cat_features, threshold=0.55):
    """Simulate betting ROI at confidence threshold."""
    print(f"\n{'='*60}")
    print(f"BETTING SIMULATION (threshold={threshold})")
    print("="*60)

    X = df[features].copy()
    for col in cat_features:
        X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=cat_features)
    y_pred_proba = model.predict_proba(pool)[:, 1]

    # Betting simulation
    # Standard -110 odds: win $100 to risk $110 (10% vig)
    WIN_PAYOUT = 100
    LOSS_COST = 110

    total_bets = 0
    total_profit = 0
    wins = 0

    for i, proba in enumerate(y_pred_proba):
        actual = df.iloc[i]['hit']

        # Bet OVER if proba >= threshold
        if proba >= threshold:
            total_bets += 1
            if actual == 1:  # Over hit
                total_profit += WIN_PAYOUT
                wins += 1
            else:
                total_profit -= LOSS_COST

        # Bet UNDER if proba <= (1 - threshold)
        elif proba <= (1 - threshold):
            total_bets += 1
            if actual == 0:  # Under hit
                total_profit += WIN_PAYOUT
                wins += 1
            else:
                total_profit -= LOSS_COST

    if total_bets > 0:
        win_rate = wins / total_bets
        roi = total_profit / (total_bets * LOSS_COST)
        print(f"  Total bets: {total_bets}")
        print(f"  Wins: {wins} ({100*win_rate:.1f}%)")
        print(f"  Net profit: ${total_profit:+.0f}")
        print(f"  ROI: {100*roi:+.1f}%")
        print(f"  Breakeven needed: 52.4%")
    else:
        print("  No bets placed at this threshold")
        roi = 0
        win_rate = 0

    return {
        'threshold': threshold,
        'total_bets': total_bets,
        'wins': wins,
        'win_rate': win_rate,
        'roi': roi
    }

def save_model(model, features, cat_features, metrics):
    """Save model and metadata."""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    # Save model
    model.save_model('catboost_model_v2.cbm')
    print("  Saved model to catboost_model_v2.cbm")

    # Save feature config
    config = {
        'version': 2,
        'all_features': features,
        'categorical_features': cat_features,
        'numeric_features': [f for f in features if f not in cat_features],
        'created_at': datetime.now().isoformat(),
        'metrics': metrics
    }

    with open('feature_columns_v2.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("  Saved feature config to feature_columns_v2.json")

    # Save training metrics
    with open('training_metrics_v2.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  Saved metrics to training_metrics_v2.json")

def main():
    """Main training pipeline."""
    print("="*60)
    print("CATBOOST V2 TRAINING")
    print("="*60)

    # Load data
    df = load_data()

    # Prepare features
    cat_features, num_features = prepare_features(df)
    all_features = cat_features + num_features

    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    # Train model
    model, features = train_model(train_df, val_df, cat_features, num_features)

    # Evaluate
    metrics = {}
    metrics['train'] = evaluate_model(model, train_df, features, cat_features, "Training")
    metrics['val'] = evaluate_model(model, val_df, features, cat_features, "Validation")
    metrics['test'] = evaluate_model(model, test_df, features, cat_features, "Test")

    # Feature importance
    feat_imp = analyze_feature_importance(model, features)
    metrics['feature_importance'] = {f: float(i) for f, i in feat_imp}

    # Accuracy by prop type
    analyze_by_prop_type(model, test_df, features, cat_features)

    # ROI simulation
    roi_55 = simulate_betting_roi(model, test_df, features, cat_features, threshold=0.55)
    roi_60 = simulate_betting_roi(model, test_df, features, cat_features, threshold=0.60)
    metrics['roi_simulation'] = {
        'threshold_55': roi_55,
        'threshold_60': roi_60
    }

    # Save
    save_model(model, features, cat_features, metrics)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTest Accuracy: {100*metrics['test']['accuracy']:.1f}%")
    print(f"Test Brier Score: {metrics['test']['brier_score']:.4f}")

    return model, metrics

if __name__ == '__main__':
    main()
