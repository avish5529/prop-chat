"""
CatBoost V2.1 Training Script
=============================
Trains on LIVE DATA ONLY (no backtest) with V2 features + DvP.

Key differences from V3:
- Uses only live predictions (clean data, no poisoning)
- Preserves avg_vs_opponent (42% importance in V2)
- Adds DvP features (real values, not defaults)
- Uses 5-fold CV for robust evaluation on small dataset

Run with: python train_catboost_v2_1.py
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "predictions.db"
MODEL_PATH = "catboost_model_v2_1.cbm"
METRICS_PATH = "training_metrics_v2_1.json"
FEATURE_COLS_PATH = "feature_columns_v2_1.json"

# Feature configuration
CATEGORICAL_FEATURES = ['opponent_team', 'prop_type', 'player_position']

NUMERIC_FEATURES = [
    # Core stats (from V2)
    'closing_line', 'season_avg', 'last_10_avg',
    'std_dev', 'minutes_avg',
    'is_home', 'is_b2b', 'days_rest',
    'opp_def_rating', 'opp_pace',
    # Key matchup feature (42% importance in V2)
    'avg_vs_opponent',
    # Derived features
    'line_vs_season', 'line_vs_last_5', 'line_difficulty', 'consistency',
    # DvP features (NEW)
    'dvp_rank', 'dvp_allowed',
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# CatBoost hyperparameters (conservative for small dataset)
CATBOOST_PARAMS = {
    'iterations': 300,
    'learning_rate': 0.03,
    'depth': 4,
    'l2_leaf_reg': 10,
    'min_data_in_leaf': 20,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'cat_features': CATEGORICAL_FEATURES,
}

N_FOLDS = 5

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load resolved predictions from database."""
    print("Loading live prediction data...")

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            id,
            player_name,
            opponent_team,
            prop_type,
            player_position,
            closing_line,
            season_avg,
            last_10_avg,
            std_dev,
            minutes_avg,
            is_home,
            is_b2b,
            days_rest,
            opp_def_rating,
            opp_pace,
            avg_vs_opponent,
            dvp_rank,
            dvp_allowed,
            catboost_hit as hit
        FROM predictions
        WHERE status = 'resolved'
        AND catboost_hit IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"  Loaded {len(df)} predictions")
    return df


def compute_derived_features(df):
    """Compute derived features from base stats."""
    print("\nComputing derived features...")

    # Use closing_line if available, otherwise use season_avg as proxy
    line = df['closing_line'].fillna(df['season_avg'])

    # last_5_avg - approximate from last_10 if missing
    df['last_5_avg'] = df['last_10_avg']  # We don't have last_5 in DB, use last_10

    # Line vs season average
    df['line_vs_season'] = line - df['season_avg']

    # Line vs last 5 (using last_10 as proxy)
    df['line_vs_last_5'] = line - df['last_5_avg']

    # Line difficulty (z-score)
    df['line_difficulty'] = np.where(
        df['std_dev'] > 0,
        (line - df['season_avg']) / df['std_dev'],
        0
    )
    df['line_difficulty'] = df['line_difficulty'].clip(-5, 5)

    # Consistency (inverse CV)
    df['consistency'] = np.where(
        df['season_avg'] > 0,
        1 - (df['std_dev'] / df['season_avg']).clip(0, 1),
        0.5
    )

    print(f"  line_vs_season: mean={df['line_vs_season'].mean():.2f}")
    print(f"  line_difficulty: mean={df['line_difficulty'].mean():.2f}")
    print(f"  consistency: mean={df['consistency'].mean():.2f}")

    return df


def prepare_features(df):
    """Prepare features for training."""
    print("\nPreparing features...")

    # Handle missing values
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            if col == 'avg_vs_opponent':
                # Fill with season_avg if no matchup history
                df[col] = df[col].fillna(df['season_avg'])
            elif col == 'dvp_rank':
                df[col] = df[col].fillna(15)  # Middle rank
            elif col == 'dvp_allowed':
                df[col] = df[col].fillna(df['season_avg'])  # Use player's avg
            elif col == 'opp_def_rating':
                df[col] = df[col].fillna(110.0)  # League average
            elif col == 'opp_pace':
                df[col] = df[col].fillna(100.0)  # League average
            elif col == 'days_rest':
                df[col] = df[col].fillna(1)
            else:
                df[col] = df[col].fillna(0)

    # Handle categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    # Check for missing columns
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: Missing features: {missing}")
        for col in missing:
            df[col] = 0 if col in NUMERIC_FEATURES else 'Unknown'

    # Final check
    for col in ALL_FEATURES:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  Warning: {col} has {null_count} nulls")

    print(f"  Features ready: {len(ALL_FEATURES)}")
    return df


# =============================================================================
# TRAINING WITH CROSS-VALIDATION
# =============================================================================

def train_with_cv(df):
    """Train model with 5-fold stratified cross-validation."""
    print("\n" + "=" * 60)
    print(f"TRAINING WITH {N_FOLDS}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    X = df[ALL_FEATURES]
    y = df['hit'].values

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Track metrics across folds
    fold_metrics = []
    all_predictions = np.zeros(len(y))
    all_probas = np.zeros(len(y))

    feature_importance_sum = np.zeros(len(ALL_FEATURES))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Create pools
        train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
        val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_FEATURES)

        # Train model
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        # Predict
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Store predictions
        all_predictions[val_idx] = y_pred
        all_probas[val_idx] = y_pred_proba

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        brier = brier_score_loss(y_val, y_pred_proba)

        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': acc,
            'brier_score': brier,
            'n_samples': len(val_idx)
        })

        print(f"  Accuracy: {100*acc:.1f}%")
        print(f"  Brier Score: {brier:.4f}")

        # Accumulate feature importance
        feature_importance_sum += model.get_feature_importance()

    # Average feature importance
    avg_importance = feature_importance_sum / N_FOLDS

    # Overall CV metrics
    overall_accuracy = accuracy_score(y, all_predictions)
    overall_brier = brier_score_loss(y, all_probas)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Overall CV Accuracy: {100*overall_accuracy:.1f}%")
    print(f"Overall CV Brier Score: {overall_brier:.4f}")

    # Per-fold summary
    accs = [m['accuracy'] for m in fold_metrics]
    print(f"Fold Accuracies: {[f'{100*a:.1f}%' for a in accs]}")
    print(f"Mean ± Std: {100*np.mean(accs):.1f}% ± {100*np.std(accs):.1f}%")

    return fold_metrics, avg_importance, all_probas, overall_accuracy, overall_brier


def train_final_model(df):
    """Train final model on all data."""
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 60)

    X = df[ALL_FEATURES]
    y = df['hit'].values

    train_pool = Pool(X, y, cat_features=CATEGORICAL_FEATURES)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(train_pool, verbose=100)

    # Save model
    model.save_model(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_feature_importance(importance, feature_names):
    """Analyze and display feature importance."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (averaged across folds)")
    print("=" * 60)

    sorted_idx = np.argsort(importance)[::-1]

    importance_dict = {}
    for i, idx in enumerate(sorted_idx):
        feat = feature_names[idx]
        imp = importance[idx]
        importance_dict[feat] = float(imp)
        print(f"  {i+1:2d}. {feat}: {imp:.1f}%")

    return importance_dict


def analyze_by_prop_type(df, probas):
    """Analyze accuracy by prop type."""
    print("\n" + "=" * 60)
    print("ACCURACY BY PROP TYPE")
    print("=" * 60)

    df_analysis = df.copy()
    df_analysis['pred'] = (probas >= 0.5).astype(int)
    df_analysis['correct'] = (df_analysis['hit'] == df_analysis['pred']).astype(int)

    prop_stats = {}
    for prop_type in sorted(df_analysis['prop_type'].unique()):
        mask = df_analysis['prop_type'] == prop_type
        total = mask.sum()
        correct = df_analysis.loc[mask, 'correct'].sum()
        acc = correct / total if total > 0 else 0
        prop_stats[prop_type] = {'total': int(total), 'accuracy': float(acc)}
        print(f"  {prop_type}: {100*acc:.1f}% ({correct}/{total})")

    return prop_stats


def analyze_roi(y_true, probas):
    """Analyze ROI at different confidence thresholds."""
    print("\n" + "=" * 60)
    print("ROI SIMULATION (assuming -110 odds)")
    print("=" * 60)

    roi_stats = {}
    for threshold in [0.52, 0.55, 0.58, 0.60]:
        mask = probas >= threshold
        if mask.sum() > 0:
            hits = y_true[mask].sum()
            total = mask.sum()
            hit_rate = hits / total
            # ROI at -110 odds: win 100, lose 110
            roi = (hit_rate * 100 - (1 - hit_rate) * 110) / 110
            roi_stats[f'threshold_{int(threshold*100)}'] = {
                'threshold': threshold,
                'total_bets': int(total),
                'wins': int(hits),
                'hit_rate': float(hit_rate),
                'roi': float(roi)
            }
            print(f"  {int(threshold*100)}%+ conf: {total} bets, {100*hit_rate:.1f}% hit, {100*roi:+.1f}% ROI")

    return roi_stats


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_metrics(fold_metrics, importance_dict, prop_stats, roi_stats,
                 overall_accuracy, overall_brier):
    """Save training metrics to JSON."""
    print(f"\nSaving metrics to {METRICS_PATH}...")

    metrics = {
        'cv_results': {
            'overall_accuracy': overall_accuracy,
            'overall_brier_score': overall_brier,
            'n_folds': N_FOLDS,
            'fold_details': fold_metrics
        },
        'feature_importance': importance_dict,
        'prop_type_accuracy': prop_stats,
        'roi_simulation': roi_stats,
        'model_params': {k: v for k, v in CATBOOST_PARAMS.items() if k != 'cat_features'},
        'features': ALL_FEATURES,
        'training_samples': sum(m['n_samples'] for m in fold_metrics) // (N_FOLDS - 1)
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("  Metrics saved.")


def save_feature_columns():
    """Save feature column configuration."""
    print(f"\nSaving feature columns to {FEATURE_COLS_PATH}...")

    with open(FEATURE_COLS_PATH, 'w') as f:
        json.dump({
            'categorical': CATEGORICAL_FEATURES,
            'numeric': NUMERIC_FEATURES,
            'all': ALL_FEATURES
        }, f, indent=2)

    print("  Feature columns saved.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CATBOOST V2.1 TRAINING")
    print("Live Data Only + V2 Features + DvP")
    print("=" * 60)

    # Load and prepare data
    df = load_data()
    df = compute_derived_features(df)
    df = prepare_features(df)

    # Cross-validation
    fold_metrics, avg_importance, all_probas, overall_accuracy, overall_brier = train_with_cv(df)

    # Train final model
    model = train_final_model(df)

    # Analysis
    importance_dict = analyze_feature_importance(avg_importance, ALL_FEATURES)
    prop_stats = analyze_by_prop_type(df, all_probas)
    roi_stats = analyze_roi(df['hit'].values, all_probas)

    # Save outputs
    save_metrics(fold_metrics, importance_dict, prop_stats, roi_stats,
                 overall_accuracy, overall_brier)
    save_feature_columns()

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"CV Accuracy: {100*overall_accuracy:.1f}%")
    print(f"CV Brier Score: {overall_brier:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Top feature: {list(importance_dict.keys())[0]} ({list(importance_dict.values())[0]:.1f}%)")


if __name__ == "__main__":
    main()
