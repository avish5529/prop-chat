"""
CatBoost V2.2 Training Script
=============================
Trains on ENRICHED BACKTEST DATA (15,453 rows) with ALL V2.1 features.

Key improvements over V2.1:
- 30x more training data (15,453 vs 512)
- Real backfilled features (avg_vs_opponent, dvp_rank, dvp_allowed, opp_def_rating, opp_pace)
- Should eliminate prop-type biases from small sample sizes

Run with: python train_catboost_v2_2.py
"""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "backtest_data_enriched.csv"
MODEL_PATH = "catboost_model_v2_2.cbm"
METRICS_PATH = "training_metrics_v2_2.json"
FEATURE_COLS_PATH = "feature_columns_v2_2.json"

# Feature configuration (same as V2.1)
CATEGORICAL_FEATURES = ['opponent_team', 'prop_type', 'player_position']

NUMERIC_FEATURES = [
    # Core stats
    'closing_line', 'season_avg', 'last_10_avg',
    'std_dev', 'minutes_avg',
    'is_home', 'is_b2b', 'days_rest',
    # Team stats
    'opp_def_rating', 'opp_pace',
    # Matchup feature
    'avg_vs_opponent',
    # Derived features (computed below)
    'line_vs_season', 'line_vs_last_5', 'line_difficulty', 'consistency',
    # DvP features
    'dvp_rank', 'dvp_allowed',
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# CatBoost hyperparameters (tuned for larger dataset)
CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 5,
    'l2_leaf_reg': 5,
    'min_data_in_leaf': 50,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'cat_features': CATEGORICAL_FEATURES,
}

N_FOLDS = 5

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_data():
    """Load enriched backtest data."""
    print("Loading enriched backtest data...")

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows")

    # Create target variable: did the over hit?
    df['hit'] = (df['actual_result'] > df['closing_line']).astype(int)

    # Handle rows where closing_line is missing (use season_avg as proxy)
    mask = df['closing_line'].isna()
    df.loc[mask, 'hit'] = (df.loc[mask, 'actual_result'] > df.loc[mask, 'season_avg']).astype(int)

    print(f"  Target distribution: {df['hit'].mean()*100:.1f}% overs hit")

    return df


def compute_derived_features(df):
    """Compute derived features from base stats."""
    print("\nComputing derived features...")

    # Fill missing closing_line with season_avg
    df['closing_line'] = df['closing_line'].fillna(df['season_avg'])

    # line_vs_season
    df['line_vs_season'] = df['closing_line'] - df['season_avg']

    # line_vs_last_5 (use last_5_avg if available, else last_10_avg)
    last_5 = df['last_5_avg'].fillna(df['last_10_avg'])
    df['line_vs_last_5'] = df['closing_line'] - last_5

    # line_difficulty (z-score) - handle missing std_dev
    std_dev = df['std_dev'].fillna(df['std_dev'].median())
    std_dev = std_dev.replace(0, df['std_dev'].median())  # Avoid division by zero
    df['line_difficulty'] = (df['closing_line'] - df['season_avg']) / std_dev
    df['line_difficulty'] = df['line_difficulty'].clip(-5, 5)

    # consistency (inverse CV)
    df['consistency'] = np.where(
        std_dev > 0,
        df['season_avg'] / std_dev,
        10.0  # Default for missing
    )
    df['consistency'] = df['consistency'].clip(0, 20)

    print(f"  line_vs_season: mean={df['line_vs_season'].mean():.2f}, std={df['line_vs_season'].std():.2f}")
    print(f"  line_difficulty: mean={df['line_difficulty'].mean():.2f}, std={df['line_difficulty'].std():.2f}")
    print(f"  consistency: mean={df['consistency'].mean():.2f}, std={df['consistency'].std():.2f}")

    return df


def prepare_features(df):
    """Prepare features for training."""
    print("\nPreparing features...")

    # Fill missing numeric values
    numeric_defaults = {
        'closing_line': df['season_avg'],
        'season_avg': df['season_avg'].median(),
        'last_10_avg': df['season_avg'],
        'std_dev': df['std_dev'].median(),
        'minutes_avg': 32.0,
        'days_rest': 1,
        'opp_def_rating': 110.0,
        'opp_pace': 100.0,
        'avg_vs_opponent': df['season_avg'],
        'dvp_rank': 15,
        'dvp_allowed': df['season_avg'],
        'line_vs_season': 0.0,
        'line_vs_last_5': 0.0,
        'line_difficulty': 0.0,
        'consistency': 5.0,
    }

    for col, default in numeric_defaults.items():
        if col in df.columns:
            if isinstance(default, pd.Series):
                df[col] = df[col].fillna(default)
            else:
                df[col] = df[col].fillna(default)

    # Fill missing categorical values
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Unknown').astype(str)

    # Report any remaining nulls
    for col in ALL_FEATURES:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"  Warning: {col} has {null_count} nulls remaining")

    print(f"  Features ready: {len(ALL_FEATURES)}")
    return df


# =============================================================================
# TRAINING
# =============================================================================

def train_with_cv(df):
    """Train model with 5-fold stratified cross-validation."""
    print("\n" + "=" * 60)
    print(f"TRAINING WITH {N_FOLDS}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    X = df[ALL_FEATURES].copy()
    y = df['hit'].values

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Track metrics
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

    accs = [m['accuracy'] for m in fold_metrics]
    print(f"Fold Accuracies: {[f'{100*a:.1f}%' for a in accs]}")
    print(f"Mean ± Std: {100*np.mean(accs):.1f}% ± {100*np.std(accs):.1f}%")

    return fold_metrics, avg_importance, all_probas, overall_accuracy, overall_brier


def train_final_model(df):
    """Train final model on all data."""
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 60)

    X = df[ALL_FEATURES].copy()
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
    print("FEATURE IMPORTANCE")
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

        # Also check prediction distribution (over vs under picks)
        over_picks = (df_analysis.loc[mask, 'pred'] == 1).sum()
        over_pct = 100 * over_picks / total if total > 0 else 0

        prop_stats[prop_type] = {
            'total': int(total),
            'accuracy': float(acc),
            'over_pick_pct': float(over_pct)
        }
        print(f"  {prop_type}: {100*acc:.1f}% acc ({correct}/{total}) | {over_pct:.0f}% over picks")

    return prop_stats


def analyze_prediction_distribution(df, probas):
    """Check if model has over/under bias."""
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION (Bias Check)")
    print("=" * 60)

    preds = (probas >= 0.5).astype(int)
    over_picks = preds.sum()
    under_picks = len(preds) - over_picks

    actual_overs = df['hit'].sum()
    actual_unders = len(df) - actual_overs

    print(f"  Model picks:  {over_picks} over ({100*over_picks/len(preds):.1f}%) | {under_picks} under ({100*under_picks/len(preds):.1f}%)")
    print(f"  Actual results: {actual_overs} over ({100*actual_overs/len(df):.1f}%) | {actual_unders} under ({100*actual_unders/len(df):.1f}%)")

    bias = (over_picks/len(preds)) - (actual_overs/len(df))
    print(f"  Over-pick bias: {100*bias:+.1f}%")

    return {
        'over_picks': int(over_picks),
        'under_picks': int(under_picks),
        'over_pick_pct': float(100*over_picks/len(preds)),
        'actual_over_pct': float(100*actual_overs/len(df)),
        'bias': float(100*bias)
    }


def analyze_roi(y_true, probas):
    """Analyze ROI at different confidence thresholds."""
    print("\n" + "=" * 60)
    print("ROI SIMULATION (assuming -110 odds)")
    print("=" * 60)

    roi_stats = {}
    for threshold in [0.52, 0.55, 0.58, 0.60]:
        # Bet when model is confident (either direction)
        confident_over = probas >= threshold
        confident_under = probas <= (1 - threshold)

        # For overs
        if confident_over.sum() > 0:
            over_hits = y_true[confident_over].sum()
            over_total = confident_over.sum()
            over_hit_rate = over_hits / over_total
        else:
            over_hit_rate = 0
            over_total = 0

        # For unders
        if confident_under.sum() > 0:
            under_hits = (1 - y_true[confident_under]).sum()
            under_total = confident_under.sum()
            under_hit_rate = under_hits / under_total
        else:
            under_hit_rate = 0
            under_total = 0

        # Combined
        total_bets = over_total + under_total
        if total_bets > 0:
            total_hits = over_hits + under_hits if over_total > 0 else under_hits
            hit_rate = (over_hits + under_hits) / total_bets
            roi = (hit_rate * 100 - (1 - hit_rate) * 110) / 110
        else:
            hit_rate = 0
            roi = 0

        roi_stats[f'threshold_{int(threshold*100)}'] = {
            'threshold': threshold,
            'total_bets': int(total_bets),
            'hit_rate': float(hit_rate),
            'roi': float(roi)
        }
        print(f"  {int(threshold*100)}%+ conf: {total_bets} bets, {100*hit_rate:.1f}% hit, {100*roi:+.1f}% ROI")

    return roi_stats


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_metrics(fold_metrics, importance_dict, prop_stats, roi_stats, bias_stats,
                 overall_accuracy, overall_brier):
    """Save training metrics to JSON."""
    print(f"\nSaving metrics to {METRICS_PATH}...")

    metrics = {
        'model_version': 'V2.2',
        'training_data': DATA_PATH,
        'training_samples': sum(m['n_samples'] for m in fold_metrics),
        'cv_results': {
            'overall_accuracy': overall_accuracy,
            'overall_brier_score': overall_brier,
            'n_folds': N_FOLDS,
            'fold_details': fold_metrics
        },
        'feature_importance': importance_dict,
        'prop_type_accuracy': prop_stats,
        'prediction_distribution': bias_stats,
        'roi_simulation': roi_stats,
        'model_params': {k: v for k, v in CATBOOST_PARAMS.items() if k != 'cat_features'},
        'features': ALL_FEATURES,
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
    print("CATBOOST V2.2 TRAINING")
    print("Enriched Backtest Data (15,453 rows)")
    print("Full V2.1 Feature Set")
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
    bias_stats = analyze_prediction_distribution(df, all_probas)
    roi_stats = analyze_roi(df['hit'].values, all_probas)

    # Save outputs
    save_metrics(fold_metrics, importance_dict, prop_stats, roi_stats, bias_stats,
                 overall_accuracy, overall_brier)
    save_feature_columns()

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"CV Accuracy: {100*overall_accuracy:.1f}%")
    print(f"CV Brier Score: {overall_brier:.4f}")
    print(f"Over-pick bias: {bias_stats['bias']:+.1f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Top feature: {list(importance_dict.keys())[0]} ({list(importance_dict.values())[0]:.1f}%)")


if __name__ == "__main__":
    main()
