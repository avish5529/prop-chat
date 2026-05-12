"""
Phase 4: Model Evaluation
Comprehensive evaluation of CatBoost model with calibration analysis.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import calibration_curve, IsotonicRegression
import matplotlib.pyplot as plt
import json


def load_model_and_data():
    """Load trained model and data."""
    # Load model
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    print("[Evaluate] Loaded model from catboost_model.cbm")

    # Load feature config
    with open("feature_columns.json", "r") as f:
        feature_config = json.load(f)

    # Load data
    df = pd.read_csv("training_data.csv")
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    return model, feature_config, train, val, test


def prepare_data(df: pd.DataFrame, feature_cols: list, cat_features: list) -> tuple:
    """Prepare features and target."""
    X = df[feature_cols].copy()
    y = df["hit"].values

    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("missing").astype(str)

    return X, y


def plot_calibration_comparison(y_true, y_probs_raw, y_probs_calibrated,
                                 split_name: str, save_path: str = None):
    """Plot calibration curves comparing raw and calibrated probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    ax1 = axes[0]
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_probs_raw, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_probs_calibrated, n_bins=10)

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(prob_pred_raw, prob_true_raw, "s-", color="blue", label="Raw CatBoost")
    ax1.plot(prob_pred_cal, prob_true_cal, "o-", color="green", label="Isotonic Calibrated")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives (actual)")
    ax1.set_title(f"Calibration Curve - {split_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_probs_raw, bins=20, alpha=0.5, label="Raw", color="blue")
    ax2.hist(y_probs_calibrated, bins=20, alpha=0.5, label="Calibrated", color="green")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Probability Distribution - {split_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] Saved calibration plot to {save_path}")

    plt.close()


def calibrate_with_isotonic(y_train, train_probs, val_probs, test_probs):
    """Apply isotonic regression calibration."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(train_probs, y_train)

    train_calibrated = calibrator.predict(train_probs)
    val_calibrated = calibrator.predict(val_probs)
    test_calibrated = calibrator.predict(test_probs)

    return calibrator, train_calibrated, val_calibrated, test_calibrated


def calculate_expected_calibration_error(y_true, y_probs, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def analyze_by_prop_type(df: pd.DataFrame, y_probs: np.ndarray):
    """Analyze performance by prop type."""
    df = df.copy()
    df["pred_prob"] = y_probs
    df["correct"] = ((df["pred_prob"] >= 0.5) == df["hit"]).astype(int)

    print(f"\n{'='*60}")
    print("PERFORMANCE BY PROP TYPE")
    print(f"{'='*60}")

    results = {}
    for prop_type in df["prop_type"].unique():
        subset = df[df["prop_type"] == prop_type]
        accuracy = subset["correct"].mean()
        brier = brier_score_loss(subset["hit"], subset["pred_prob"])
        n_samples = len(subset)

        results[prop_type] = {
            "n_samples": int(n_samples),
            "accuracy": float(accuracy),
            "brier_score": float(brier)
        }

        print(f"  {prop_type:10s}: n={n_samples:4d}, accuracy={accuracy:.1%}, brier={brier:.4f}")

    return results


def analyze_by_player(df: pd.DataFrame, y_probs: np.ndarray, top_n: int = 15):
    """Analyze performance by player."""
    df = df.copy()
    df["pred_prob"] = y_probs
    df["correct"] = ((df["pred_prob"] >= 0.5) == df["hit"]).astype(int)

    print(f"\n{'='*60}")
    print(f"TOP {top_n} PLAYERS BY SAMPLE SIZE")
    print(f"{'='*60}")

    player_stats = df.groupby("player_name").agg({
        "correct": ["sum", "count", "mean"],
        "pred_prob": "mean"
    }).round(4)
    player_stats.columns = ["wins", "total", "accuracy", "avg_prob"]
    player_stats = player_stats.sort_values("total", ascending=False).head(top_n)

    for player, row in player_stats.iterrows():
        print(f"  {player:22s}: {int(row['wins']):3d}/{int(row['total']):3d} ({row['accuracy']:.1%})")

    return player_stats


def simulate_bankroll_growth(df: pd.DataFrame, y_probs: np.ndarray,
                              initial_bankroll: float = 1000,
                              flat_bet_size: float = 10,
                              threshold: float = 0.55) -> dict:
    """Simulate bankroll growth using flat betting."""
    df = df.copy()
    df["pred_prob"] = y_probs
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    df = df.sort_values("parsed_date")

    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_made = 0
    wins = 0

    # Assumed odds: -110
    payout = 9.09  # Win $9.09 on $10 bet at -110

    for _, row in df.iterrows():
        prob = row["pred_prob"]
        actual_hit = row["hit"]

        # Determine if we bet
        if prob >= threshold:
            bet_on_over = True
        elif prob <= (1 - threshold):
            bet_on_over = False
        else:
            continue  # No bet

        # Flat bet sizing
        bet_size = flat_bet_size

        # Resolve bet
        won = (bet_on_over and actual_hit == 1) or (not bet_on_over and actual_hit == 0)

        if won:
            bankroll += payout
            wins += 1
        else:
            bankroll -= bet_size

        bankroll_history.append(bankroll)
        bets_made += 1

    roi = ((bankroll - initial_bankroll) / (bets_made * flat_bet_size)) * 100 if bets_made > 0 else 0
    total_wagered = bets_made * flat_bet_size

    profit = bankroll - initial_bankroll

    print(f"\n{'='*60}")
    print("BANKROLL SIMULATION (Flat $10 bets, -110 odds)")
    print(f"{'='*60}")
    print(f"  Initial bankroll: ${initial_bankroll:,.2f}")
    print(f"  Final bankroll:   ${bankroll:,.2f}")
    print(f"  Total bets:       {bets_made}")
    print(f"  Total wagered:    ${total_wagered:,.2f}")
    print(f"  Wins:             {wins} ({wins/bets_made*100:.1f}%)")
    print(f"  Profit:           ${profit:+,.2f}")
    print(f"  ROI:              {roi:+.1f}%")

    return {
        "initial": initial_bankroll,
        "final": float(bankroll),
        "bets_made": bets_made,
        "wins": wins,
        "win_rate": wins / bets_made if bets_made > 0 else 0,
        "roi": float(roi),
        "profit": float(profit),
        "history": bankroll_history
    }


def plot_bankroll_growth(simulation_result: dict, save_path: str = None):
    """Plot bankroll growth over time."""
    history = simulation_result["history"]

    plt.figure(figsize=(10, 5))
    plt.plot(history, linewidth=2)
    plt.axhline(y=simulation_result["initial"], color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll ($)")
    plt.title(f"Bankroll Growth (Final: ${simulation_result['final']:,.2f}, ROI: {simulation_result['roi']:+.1f}%)")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] Saved bankroll plot to {save_path}")

    plt.close()


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("PHASE 4: MODEL EVALUATION")
    print("="*60 + "\n")

    # Load model and data
    model, feature_config, train_df, val_df, test_df = load_model_and_data()

    all_features = feature_config["all_features"]
    cat_features = feature_config["categorical_features"]

    # Get predictions
    X_train, y_train = prepare_data(train_df, all_features, cat_features)
    X_val, y_val = prepare_data(val_df, all_features, cat_features)
    X_test, y_test = prepare_data(test_df, all_features, cat_features)

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Calibration metrics (raw)
    print("\n" + "="*60)
    print("CALIBRATION METRICS (RAW)")
    print("="*60)

    for name, y, probs in [("Train", y_train, train_probs),
                            ("Val", y_val, val_probs),
                            ("Test", y_test, test_probs)]:
        brier = brier_score_loss(y, probs)
        ece = calculate_expected_calibration_error(y, probs)
        print(f"  {name:6s}: Brier={brier:.4f}, ECE={ece:.4f}")

    # Apply isotonic calibration
    print("\n[Evaluate] Applying isotonic regression calibration...")
    calibrator, train_cal, val_cal, test_cal = calibrate_with_isotonic(
        y_train, train_probs, val_probs, test_probs
    )

    # Calibration metrics (calibrated)
    print("\n" + "="*60)
    print("CALIBRATION METRICS (ISOTONIC CALIBRATED)")
    print("="*60)

    for name, y, probs in [("Train", y_train, train_cal),
                            ("Val", y_val, val_cal),
                            ("Test", y_test, test_cal)]:
        brier = brier_score_loss(y, probs)
        ece = calculate_expected_calibration_error(y, probs)
        print(f"  {name:6s}: Brier={brier:.4f}, ECE={ece:.4f}")

    # Plot calibration
    plot_calibration_comparison(y_val, val_probs, val_cal, "Validation", "catboost_calibration_val.png")
    plot_calibration_comparison(y_test, test_probs, test_cal, "Test", "catboost_calibration_test.png")

    # Analysis by prop type
    prop_results = analyze_by_prop_type(test_df, test_probs)

    # Analysis by player
    player_results = analyze_by_player(test_df, test_probs)

    # Bankroll simulation
    simulation = simulate_bankroll_growth(test_df, test_probs, threshold=0.55)
    plot_bankroll_growth(simulation, "bankroll_simulation.png")

    # Save calibrator
    import pickle
    with open("isotonic_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    print(f"\n[Evaluate] Saved calibrator to isotonic_calibrator.pkl")

    # Compile evaluation results
    evaluation_results = {
        "raw_metrics": {
            "val_brier": float(brier_score_loss(y_val, val_probs)),
            "val_ece": float(calculate_expected_calibration_error(y_val, val_probs)),
            "test_brier": float(brier_score_loss(y_test, test_probs)),
            "test_ece": float(calculate_expected_calibration_error(y_test, test_probs)),
        },
        "calibrated_metrics": {
            "val_brier": float(brier_score_loss(y_val, val_cal)),
            "val_ece": float(calculate_expected_calibration_error(y_val, val_cal)),
            "test_brier": float(brier_score_loss(y_test, test_cal)),
            "test_ece": float(calculate_expected_calibration_error(y_test, test_cal)),
        },
        "by_prop_type": prop_results,
        "bankroll_simulation": {
            "final_bankroll": simulation["final"],
            "roi": simulation["roi"],
            "win_rate": simulation["win_rate"]
        }
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"[Evaluate] Saved evaluation results to evaluation_results.json")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    # Summary
    print("\nSUMMARY:")
    print(f"  Test Accuracy: {accuracy_score(y_test, (test_probs >= 0.5).astype(int)):.1%}")
    print(f"  Test Brier (Raw): {brier_score_loss(y_test, test_probs):.4f}")
    print(f"  Test Brier (Calibrated): {brier_score_loss(y_test, test_cal):.4f}")
    print(f"  Bankroll ROI (55% threshold): {simulation['roi']:+.1f}%")

    return evaluation_results


if __name__ == "__main__":
    main()
