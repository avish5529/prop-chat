"""
Phase 3: CatBoost Model Training
Train CatBoost binary classifier with categorical features.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import json
import os


# Feature configuration
CATEGORICAL_FEATURES = [
    "player_name",
    "opponent_team",
    "prop_type",
]

NUMERIC_FEATURES = [
    "closing_line",
    "season_avg",
    "last_10_avg",
    "last_5_avg",
    "minutes_avg",
    "days_rest",
    "is_home",
    "is_b2b",
    "line_vs_season",
    "line_vs_recent",
    "form_trend",
    "games_played",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_training_data(path: str = "training_data.csv") -> tuple:
    """Load and split training data."""
    df = pd.read_csv(path)

    # Parse dates for proper sorting
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    print(f"[CatBoost] Data splits:")
    print(f"  - Train: {len(train)} rows")
    print(f"  - Val: {len(val)} rows")
    print(f"  - Test: {len(test)} rows")

    return train, val, test


def prepare_data(df: pd.DataFrame, feature_cols: list, cat_features: list) -> tuple:
    """Prepare feature matrix and target."""
    X = df[feature_cols].copy()
    y = df["hit"].values

    # Convert categorical columns to string (CatBoost requirement)
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("missing").astype(str)

    return X, y


def train_catboost(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   feature_cols: list, cat_features: list) -> CatBoostClassifier:
    """Train CatBoost classifier."""

    # Prepare data
    X_train, y_train = prepare_data(train_df, feature_cols, cat_features)
    X_val, y_val = prepare_data(val_df, feature_cols, cat_features)

    # Get categorical feature indices
    cat_indices = [feature_cols.index(f) for f in cat_features if f in feature_cols]

    # Create CatBoost pools
    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_indices
    )
    val_pool = Pool(
        X_val,
        y_val,
        cat_features=cat_indices
    )

    # Model configuration
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_indices,
        nan_mode="Min",
        eval_metric="Logloss",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
        use_best_model=True,
    )

    print(f"\n[CatBoost] Starting training...")
    print(f"[CatBoost] Features: {len(feature_cols)} ({len(cat_features)} categorical)")

    # Train
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=100
    )

    print(f"\n[CatBoost] Best iteration: {model.get_best_iteration()}")

    return model


def evaluate_model(model: CatBoostClassifier, df: pd.DataFrame,
                   feature_cols: list, cat_features: list, split_name: str) -> dict:
    """Evaluate model on a dataset."""

    X, y = prepare_data(df, feature_cols, cat_features)

    # Get predictions
    y_probs = model.predict_proba(X)[:, 1]
    y_preds = model.predict(X)

    # Calculate metrics
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(y, y_preds)),
        "log_loss": float(log_loss(y, y_probs)),
        "brier_score": float(brier_score_loss(y, y_probs)),
        "precision_over": float(precision_score(y, y_preds, pos_label=1)),
        "recall_over": float(recall_score(y, y_preds, pos_label=1)),
        "f1_over": float(f1_score(y, y_preds, pos_label=1)),
        "precision_under": float(precision_score(y, y_preds, pos_label=0)),
        "recall_under": float(recall_score(y, y_preds, pos_label=0)),
        "f1_under": float(f1_score(y, y_preds, pos_label=0)),
    }

    print(f"\n{'='*60}")
    print(f"{split_name.upper()} SET METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Log Loss:     {metrics['log_loss']:.4f}")
    print(f"Brier Score:  {metrics['brier_score']:.4f}")
    print(f"\nOver predictions:")
    print(f"  Precision: {metrics['precision_over']:.4f}")
    print(f"  Recall:    {metrics['recall_over']:.4f}")
    print(f"  F1:        {metrics['f1_over']:.4f}")
    print(f"\nUnder predictions:")
    print(f"  Precision: {metrics['precision_under']:.4f}")
    print(f"  Recall:    {metrics['recall_under']:.4f}")
    print(f"  F1:        {metrics['f1_under']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    return metrics


def get_feature_importance(model: CatBoostClassifier, feature_cols: list) -> dict:
    """Get feature importance from trained model."""
    importance = model.get_feature_importance()
    importance_dict = {
        feature_cols[i]: float(importance[i])
        for i in range(len(feature_cols))
    }

    # Sort by importance
    sorted_importance = dict(sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*60}")

    for feat, imp in sorted_importance.items():
        bar = "#" * int(imp / 2)
        print(f"  {feat:20s}: {imp:6.2f} {bar}")

    return sorted_importance


def simulate_roi(model: CatBoostClassifier, df: pd.DataFrame,
                 feature_cols: list, cat_features: list,
                 thresholds: list = [0.50, 0.55, 0.60, 0.65, 0.70]) -> dict:
    """Simulate ROI at various confidence thresholds."""
    X, y = prepare_data(df, feature_cols, cat_features)
    y_probs = model.predict_proba(X)[:, 1]

    results = {}
    payout_ratio = 90.91 / 100  # -110 odds

    print(f"\n{'='*60}")
    print("ROI SIMULATION (assuming -110 odds)")
    print(f"{'='*60}")

    for thresh in thresholds:
        over_mask = y_probs >= thresh
        under_mask = y_probs <= (1 - thresh)

        df_subset = df.copy()
        df_subset["pred_prob"] = y_probs
        df_subset["bet_side"] = None
        df_subset.loc[over_mask, "bet_side"] = "over"
        df_subset.loc[under_mask, "bet_side"] = "under"

        bets = df_subset[df_subset["bet_side"].notna()].copy()

        if len(bets) == 0:
            results[thresh] = {"n_bets": 0, "roi": 0.0, "win_rate": 0.0}
            continue

        over_bets = bets[bets["bet_side"] == "over"]
        under_bets = bets[bets["bet_side"] == "under"]

        over_wins = (over_bets["hit"] == 1).sum()
        under_wins = (under_bets["hit"] == 0).sum()

        total_bets = len(bets)
        total_wins = over_wins + under_wins
        win_rate = total_wins / total_bets if total_bets > 0 else 0

        profit = (total_wins * payout_ratio) - (total_bets - total_wins)
        roi = (profit / total_bets) * 100 if total_bets > 0 else 0

        results[thresh] = {
            "n_bets": int(total_bets),
            "n_over_bets": int(len(over_bets)),
            "n_under_bets": int(len(under_bets)),
            "wins": int(total_wins),
            "win_rate": float(win_rate),
            "profit": float(profit),
            "roi": float(roi)
        }

        print(f"\nThreshold >= {thresh:.0%}:")
        print(f"  Bets: {total_bets} ({len(over_bets)} overs, {len(under_bets)} unders)")
        print(f"  Wins: {total_wins} ({win_rate:.1%})")
        print(f"  ROI:  {roi:+.1f}%")

    return results


def save_model_artifacts(model: CatBoostClassifier, metrics: dict,
                         feature_cols: list, cat_features: list):
    """Save model and associated files."""

    # Save model
    model_path = "catboost_model.cbm"
    model.save_model(model_path)
    print(f"\n[CatBoost] Saved model to {model_path}")

    # Save feature configuration
    feature_config = {
        "all_features": feature_cols,
        "categorical_features": cat_features,
        "numeric_features": [f for f in feature_cols if f not in cat_features],
    }

    with open("feature_columns.json", "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"[CatBoost] Saved feature config to feature_columns.json")

    # Save metrics
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[CatBoost] Saved metrics to training_metrics.json")


def compare_with_baseline():
    """Load and compare with baseline metrics."""
    baseline_path = "baseline_metrics.json"
    if not os.path.exists(baseline_path):
        print("\n[CatBoost] No baseline metrics found for comparison")
        return

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    with open("training_metrics.json", "r") as f:
        catboost_metrics = json.load(f)

    print(f"\n{'='*60}")
    print("COMPARISON: CATBOOST VS BASELINE")
    print(f"{'='*60}")

    for split in ["val", "test"]:
        print(f"\n{split.upper()} SET:")
        baseline_acc = baseline[split]["accuracy"]
        catboost_acc = catboost_metrics[split]["accuracy"]
        diff_acc = (catboost_acc - baseline_acc) * 100

        baseline_brier = baseline[split]["brier_score"]
        catboost_brier = catboost_metrics[split]["brier_score"]
        diff_brier = catboost_brier - baseline_brier

        print(f"  Accuracy:    Baseline={baseline_acc:.1%}, CatBoost={catboost_acc:.1%} ({diff_acc:+.1f}pp)")
        print(f"  Brier Score: Baseline={baseline_brier:.4f}, CatBoost={catboost_brier:.4f} ({diff_brier:+.4f})")


def main():
    """Main execution pipeline."""
    print("="*60)
    print("PHASE 3: CATBOOST MODEL TRAINING")
    print("="*60 + "\n")

    # Load data
    train_df, val_df, test_df = load_training_data("training_data.csv")

    # Train model
    model = train_catboost(
        train_df, val_df,
        ALL_FEATURES, CATEGORICAL_FEATURES
    )

    # Evaluate on all splits
    train_metrics = evaluate_model(model, train_df, ALL_FEATURES, CATEGORICAL_FEATURES, "Training")
    val_metrics = evaluate_model(model, val_df, ALL_FEATURES, CATEGORICAL_FEATURES, "Validation")
    test_metrics = evaluate_model(model, test_df, ALL_FEATURES, CATEGORICAL_FEATURES, "Test")

    # Feature importance
    importance = get_feature_importance(model, ALL_FEATURES)

    # ROI simulation
    val_roi = simulate_roi(model, val_df, ALL_FEATURES, CATEGORICAL_FEATURES)
    test_roi = simulate_roi(model, test_df, ALL_FEATURES, CATEGORICAL_FEATURES)

    # Compile all metrics
    all_metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "roi_validation": {str(k): v for k, v in val_roi.items()},
        "roi_test": {str(k): v for k, v in test_roi.items()},
        "feature_importance": importance,
        "best_iteration": model.get_best_iteration(),
    }

    # Save artifacts
    save_model_artifacts(model, all_metrics, ALL_FEATURES, CATEGORICAL_FEATURES)

    # Compare with baseline
    compare_with_baseline()

    print("\n" + "="*60)
    print("CATBOOST TRAINING COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  - Validation Accuracy: {val_metrics['accuracy']:.1%}")
    print(f"  - Validation Brier Score: {val_metrics['brier_score']:.4f}")
    print(f"  - Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"  - Test Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"  - Best Iteration: {model.get_best_iteration()}")

    return model, all_metrics


if __name__ == "__main__":
    main()
