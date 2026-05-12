"""
CatBoost Inference Module V2.2
Production-ready predictor class for making predictions on live data.

V2.2 Changes (Apr 15, 2026):
- Uses catboost_model_v2_2.cbm (59.7% CV accuracy)
- Trained on enriched backtest data (15,453 rows) - 30x more than V2.1
- FIXED prop-type biases: all prop types now 51-55% over picks (vs 0-96% in V2.1)
- Balanced feature importance - no single feature dominates
- ROI at 55%+ confidence: +27.2%

V2.1 Changes (Apr 8, 2026):
- Trained on live data only (512 predictions) - HAD PROP-TYPE BIASES
- Added player_position, dvp_rank, dvp_allowed features

V2 Changes:
- New features: line_vs_last_5, line_difficulty, consistency
- New matchup features: avg_vs_opponent, opp_def_rating, opp_pace
"""

import json
import os
from typing import Optional
import pandas as pd
import numpy as np

# Check for CatBoost availability
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostPredictor:
    """
    Production predictor for NBA player props.
    Loads trained CatBoost model and feature configuration.
    """

    def __init__(self, model_path: str = None, feature_config_path: str = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved CatBoost model (.cbm file)
            feature_config_path: Path to feature configuration JSON
        """
        if not CATBOOST_AVAILABLE:
            self.model = None
            self.feature_config = None
            self.is_loaded = False
            print("[CatBoostPredictor] CatBoost not available - predictions disabled")
            return

        # Set default paths relative to this file - use v2.2 model by default
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(base_dir, "catboost_model_v2_2.cbm")
        feature_config_path = feature_config_path or os.path.join(base_dir, "feature_columns_v2_2.json")

        self.model = None
        self.feature_config = None
        self.is_loaded = False

        # Load model and config
        self._load_model(model_path)
        self._load_feature_config(feature_config_path)

    def _load_model(self, model_path: str):
        """Load the CatBoost model."""
        if not os.path.exists(model_path):
            print(f"[CatBoostPredictor] Model not found at {model_path}")
            return

        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            print(f"[CatBoostPredictor] Loaded model from {model_path}")
        except Exception as e:
            print(f"[CatBoostPredictor] Error loading model: {e}")
            self.model = None

    def _load_feature_config(self, config_path: str):
        """Load feature configuration."""
        if not os.path.exists(config_path):
            print(f"[CatBoostPredictor] Feature config not found at {config_path}")
            return

        try:
            with open(config_path, "r") as f:
                self.feature_config = json.load(f)
            self.is_loaded = self.model is not None
            print(f"[CatBoostPredictor] Loaded feature config from {config_path}")
        except Exception as e:
            print(f"[CatBoostPredictor] Error loading feature config: {e}")
            self.feature_config = None

    def _compute_derived_features(self, features: dict) -> dict:
        """Compute derived features from raw inputs."""
        result = features.copy()

        closing_line = features.get("closing_line")
        season_avg = features.get("season_avg")
        last_10_avg = features.get("last_10_avg")
        last_5_avg = features.get("last_5_avg")
        std_dev = features.get("std_dev", 0)

        # line_vs_season
        if closing_line is not None and season_avg is not None:
            result["line_vs_season"] = closing_line - season_avg
        else:
            result["line_vs_season"] = 0.0

        # line_vs_recent (vs last 10)
        if closing_line is not None and last_10_avg is not None:
            result["line_vs_recent"] = closing_line - last_10_avg
        else:
            result["line_vs_recent"] = 0.0

        # line_vs_last_5 (NEW in v2)
        if closing_line is not None and last_5_avg is not None:
            result["line_vs_last_5"] = closing_line - last_5_avg
        else:
            result["line_vs_last_5"] = 0.0

        # line_difficulty - z-score of line (NEW in v2)
        if closing_line is not None and season_avg is not None and std_dev and std_dev > 0:
            result["line_difficulty"] = (closing_line - season_avg) / std_dev
            # Clip extreme values
            result["line_difficulty"] = max(-5, min(5, result["line_difficulty"]))
        else:
            result["line_difficulty"] = 0.0

        # consistency - inverse CV (NEW in v2)
        if season_avg is not None and std_dev and std_dev > 0:
            result["consistency"] = season_avg / std_dev
            # Clip extreme values
            result["consistency"] = max(0, min(20, result["consistency"]))
        elif season_avg is not None:
            result["consistency"] = season_avg
        else:
            result["consistency"] = 10.0  # Default middle value

        # form_trend (kept for compatibility)
        if last_5_avg is not None and last_10_avg is not None:
            result["form_trend"] = last_5_avg - last_10_avg
        else:
            result["form_trend"] = 0.0

        return result

    def predict(self, features: dict) -> Optional[dict]:
        """
        Make a prediction for a single prop.

        Args:
            features: Dictionary with feature values. Required keys:
                - opponent_team: str
                - prop_type: str (points, rebounds, assists, pra, pr, pa, ra)
                - player_position: str (PG, SG, SF, PF, C) - V2.1
                - closing_line: float (the line being bet)
                - season_avg: float
                - last_10_avg: float
                - std_dev: float (for line_difficulty, consistency)
                - minutes_avg: float
                - days_rest: int
                - is_home: int (0 or 1)
                - is_b2b: int (0 or 1)

                V2 features (optional, defaults provided):
                - avg_vs_opponent: float (historical avg vs this opponent)
                - opp_def_rating: float (opponent defensive rating)
                - opp_pace: float (opponent pace)

                V2.1 new features (optional, defaults provided):
                - dvp_rank: int (1-30, opponent's DvP rank for this position)
                - dvp_allowed: float (points/rebounds/assists allowed by opponent)

        Returns:
            Dictionary with prediction results:
                - prob_over: float (probability of hitting over)
                - prob_under: float (probability of hitting under)
                - recommended_side: str ("over" or "under")
                - confidence: float (how confident the model is)
                - should_bet: bool (whether confidence exceeds threshold)
        """
        if not self.is_loaded:
            return None

        try:
            # Compute derived features
            features = self._compute_derived_features(features)

            # Build feature vector in correct order
            # V2.1 uses "all" and "categorical", V2 uses "all_features" and "categorical_features"
            all_features = self.feature_config.get("all") or self.feature_config.get("all_features")
            cat_features = self.feature_config.get("categorical") or self.feature_config.get("categorical_features")

            feature_values = []
            for feat in all_features:
                value = features.get(feat)

                # Handle categorical features
                if feat in cat_features:
                    value = str(value) if value is not None else "missing"
                else:
                    # Handle numeric features - use 0 as default for missing
                    if value is None:
                        value = 0.0
                    value = float(value)

                feature_values.append(value)

            # Create DataFrame with correct column names
            X = pd.DataFrame([feature_values], columns=all_features)

            # Convert categorical columns to string
            for col in cat_features:
                X[col] = X[col].astype(str)

            # Get prediction
            probs = self.model.predict_proba(X)[0]
            prob_under = probs[0]  # Class 0 = under
            prob_over = probs[1]   # Class 1 = over

            # Determine recommendation
            if prob_over > prob_under:
                recommended_side = "over"
                confidence = prob_over
            else:
                recommended_side = "under"
                confidence = prob_under

            # Should we bet? (55% threshold)
            should_bet = bool(confidence >= 0.55)

            return {
                "prob_over": float(round(prob_over, 4)),
                "prob_under": float(round(prob_under, 4)),
                "recommended_side": recommended_side,
                "confidence": float(round(confidence, 4)),
                "should_bet": should_bet
            }

        except Exception as e:
            print(f"[CatBoostPredictor] Prediction error: {e}")
            return None

    def predict_batch(self, features_list: list) -> list:
        """
        Make predictions for multiple props.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(f) for f in features_list]


# Singleton instance for use in main.py
_predictor_instance = None


def get_predictor() -> CatBoostPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CatBoostPredictor()
    return _predictor_instance


if __name__ == "__main__":
    # Test the predictor
    print("Testing CatBoostPredictor V2.2...")

    predictor = get_predictor()

    if predictor.is_loaded:
        # Test prediction with sample data (including v2.1 features)
        test_features = {
            "opponent_team": "BOS",
            "prop_type": "points",
            "player_position": "SF",  # V2.1
            "closing_line": 25.5,
            "season_avg": 25.0,
            "last_10_avg": 26.5,
            "std_dev": 5.0,
            "minutes_avg": 35.0,
            "days_rest": 2,
            "is_home": 1,
            "is_b2b": 0,
            # V2 features
            "avg_vs_opponent": 28.0,  # Historical avg vs BOS
            "opp_def_rating": 108.5,  # BOS defensive rating
            "opp_pace": 99.2,         # BOS pace
            # V2.1 DvP features
            "dvp_rank": 5,            # BOS rank vs SF
            "dvp_allowed": 22.5,      # Points allowed to SF
        }

        result = predictor.predict(test_features)

        if result:
            print("\nTest Prediction (V2.2 Model):")
            print(f"  Opponent: {test_features['opponent_team']}")
            print(f"  Position: {test_features['player_position']}")
            print(f"  Prop: {test_features['prop_type']} {test_features['closing_line']}")
            print(f"  Prob Over: {result['prob_over']:.1%}")
            print(f"  Prob Under: {result['prob_under']:.1%}")
            print(f"  Recommendation: {result['recommended_side'].upper()}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Should Bet: {result['should_bet']}")
        else:
            print("Prediction failed")
    else:
        print("Predictor not loaded - check model files")
