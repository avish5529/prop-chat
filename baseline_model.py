"""
Phase 2: Baseline Model (Logistic Regression)
Establishes performance floor that CatBoost must beat.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import json


def load_training_data(path: str = "training_data.csv") -> tuple:
    """Load and split training data."""
    df = pd.read_csv(path)

    # Parse dates for proper sorting
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    print(f"[Baseline] Data splits:")
    print(f"  - Train: {len(train)} rows")
    print(f"  - Val: {len(val)} rows")
    print(f"  - Test: {len(test)} rows")

    return train, val, test


def prepare_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Prepare feature matrix and target vector."""
    X = df[feature_cols].copy()
    y = df["hit"].values

    # Fill NaN with median for numeric features
    for col in feature_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X.values, y


def train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   feature_cols: list) -> tuple:
    """Train logistic regression baseline."""

    # Prepare data
    X_train, y_train = prepare_features(train_df, feature_cols)
    X_val, y_val = prepare_features(val_df, feature_cols)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    train_preds = (train_probs >= 0.5).astype(int)
    val_preds = (val_probs >= 0.5).astype(int)

    return model, scaler, train_probs, val_probs, train_preds, val_preds


def evaluate_model(y_true, y_probs, y_preds, split_name: str) -> dict:
    """Compute evaluation metrics."""

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_preds),
        "log_loss": log_loss(y_true, y_probs),
        "brier_score": brier_score_loss(y_true, y_probs),
        "precision_over": precision_score(y_true, y_preds, pos_label=1),
        "recall_over": recall_score(y_true, y_preds, pos_label=1),
        "f1_over": f1_score(y_true, y_preds, pos_label=1),
        "precision_under": precision_score(y_true, y_preds, pos_label=0),
        "recall_under": recall_score(y_true, y_preds, pos_label=0),
        "f1_under": f1_score(y_true, y_preds, pos_label=0),
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
    cm = confusion_matrix(y_true, y_preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    return metrics


def plot_calibration(y_true, y_probs, split_name: str, save_path: str = None):
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, "s-", label=f"Logistic Regression")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (actual)")
    plt.title(f"Calibration Curve - {split_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Baseline] Saved calibration plot to {save_path}")

    plt.close()


def simulate_roi(df: pd.DataFrame, y_probs: np.ndarray,
                 thresholds: list = [0.50, 0.55, 0.60, 0.65, 0.70]) -> dict:
    """Simulate ROI at various confidence thresholds."""
    results = {}

    # Assumed odds: -110 (standard juice)
    # Payout: $100 bet wins $90.91 profit
    payout_ratio = 90.91 / 100

    print(f"\n{'='*60}")
    print("ROI SIMULATION (assuming -110 odds)")
    print(f"{'='*60}")

    for thresh in thresholds:
        # For overs (prob >= thresh) and unders (prob <= 1-thresh)
        over_mask = y_probs >= thresh
        under_mask = y_probs <= (1 - thresh)

        # Combine: bet on overs where confident, unders where confident
        df_subset = df.copy()
        df_subset["pred_prob"] = y_probs
        df_subset["bet_side"] = None
        df_subset.loc[over_mask, "bet_side"] = "over"
        df_subset.loc[under_mask, "bet_side"] = "under"

        bets = df_subset[df_subset["bet_side"].notna()].copy()

        if len(bets) == 0:
            results[thresh] = {"n_bets": 0, "roi": 0, "win_rate": 0}
            continue

        # Calculate wins
        over_bets = bets[bets["bet_side"] == "over"]
        under_bets = bets[bets["bet_side"] == "under"]

        over_wins = (over_bets["hit"] == 1).sum()
        under_wins = (under_bets["hit"] == 0).sum()  # Under wins when hit=0

        total_bets = len(bets)
        total_wins = over_wins + under_wins
        win_rate = total_wins / total_bets if total_bets > 0 else 0

        # ROI calculation
        profit = (total_wins * payout_ratio) - (total_bets - total_wins)
        roi = (profit / total_bets) * 100 if total_bets > 0 else 0

        results[thresh] = {
            "n_bets": total_bets,
            "n_over_bets": len(over_bets),
            "n_under_bets": len(under_bets),
            "wins": total_wins,
            "win_rate": win_rate,
            "profit": profit,
            "roi": roi
        }

        print(f"\nThreshold >= {thresh:.0%}:")
        print(f"  Bets: {total_bets} ({len(over_bets)} overs, {len(under_bets)} unders)")
        print(f"  Wins: {total_wins} ({win_rate:.1%})")
        print(f"  ROI:  {roi:+.1f}%")

    return results


def print_feature_importance(model, feature_cols: list):
    """Print feature importance from logistic regression coefficients."""
    coeffs = model.coef_[0]
    importance = list(zip(feature_cols, coeffs))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print(f"{'='*60}")

    for feat, coef in importance:
        direction = "+" if coef > 0 else "-"
        print(f"  {feat:20s}: {direction}{abs(coef):.4f}")


def main():
    """Main execution pipeline."""
    print("="*60)
    print("PHASE 2: BASELINE MODEL (LOGISTIC REGRESSION)")
    print("="*60 + "\n")

    # Load data
    train_df, val_df, test_df = load_training_data("training_data.csv")

    # Define features for baseline
    feature_cols = [
        "line_vs_season",   # How line compares to season avg
        "line_vs_recent",   # How line compares to last 10 avg
        "is_home",          # Home game indicator
        "is_b2b",           # Back-to-back indicator
        "days_rest",        # Days of rest
    ]

    print(f"\n[Baseline] Features: {feature_cols}")

    # Train model
    model, scaler, train_probs, val_probs, train_preds, val_preds = train_baseline(
        train_df, val_df, feature_cols
    )

    # Get target values
    _, y_train = prepare_features(train_df, feature_cols)
    _, y_val = prepare_features(val_df, feature_cols)

    # Evaluate on training set
    train_metrics = evaluate_model(y_train, train_probs, train_preds, "Training")

    # Evaluate on validation set
    val_metrics = evaluate_model(y_val, val_probs, val_preds, "Validation")

    # Plot calibration curves
    plot_calibration(y_train, train_probs, "Training", "baseline_calibration_train.png")
    plot_calibration(y_val, val_probs, "Validation", "baseline_calibration_val.png")

    # ROI simulation on validation set
    val_df_copy = val_df.copy()
    roi_results = simulate_roi(val_df_copy, val_probs)

    # Feature importance
    print_feature_importance(model, feature_cols)

    # Evaluate on test set
    X_test, y_test = prepare_features(test_df, feature_cols)
    X_test_scaled = scaler.transform(X_test)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_metrics = evaluate_model(y_test, test_probs, test_preds, "Test")
    plot_calibration(y_test, test_probs, "Test", "baseline_calibration_test.png")

    # ROI on test set
    test_df_copy = test_df.copy()
    test_roi = simulate_roi(test_df_copy, test_probs)

    # Save metrics - convert numpy types to native Python
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_metrics = {
        "train": convert_to_serializable(train_metrics),
        "val": convert_to_serializable(val_metrics),
        "test": convert_to_serializable(test_metrics),
        "roi_validation": {str(k): convert_to_serializable(v) for k, v in roi_results.items()},
        "roi_test": {str(k): convert_to_serializable(v) for k, v in test_roi.items()},
        "features": feature_cols
    }

    with open("baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Baseline] Saved metrics to baseline_metrics.json")

    print("\n" + "="*60)
    print("BASELINE MODEL COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  - Validation Accuracy: {val_metrics['accuracy']:.1%}")
    print(f"  - Validation Brier Score: {val_metrics['brier_score']:.4f}")
    print(f"  - Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"  - Test Brier Score: {test_metrics['brier_score']:.4f}")

    return model, scaler, all_metrics


if __name__ == "__main__":
    main()
