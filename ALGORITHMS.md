# Prop.chat Prediction Algorithms

This document explains how the NBA and MLB prediction models work.

---

## NBA Player Props: CatBoost ML Model (V2.2)

### Overview
The NBA system uses a **CatBoost gradient boosting classifier** trained on 15,453 historical predictions. It predicts the probability of a player prop hitting OVER or UNDER.

**Current Performance:** 59.7% cross-validation accuracy, +27.2% ROI at 55%+ confidence threshold.

### Input Features (20 total)

#### Categorical Features (3)
| Feature | Description |
|---------|-------------|
| `opponent_team` | Team abbreviation (e.g., "BOS", "LAL") |
| `prop_type` | Type of prop: points, rebounds, assists, pra, pr, pa, ra |
| `player_position` | Player position: PG, SG, SF, PF, C |

#### Numeric Features (17)
| Feature | Description |
|---------|-------------|
| `closing_line` | The betting line (e.g., 25.5 points) |
| `season_avg` | Player's season average for this stat |
| `last_10_avg` | Player's last 10 games average |
| `std_dev` | Standard deviation of player's performance |
| `minutes_avg` | Player's average minutes played |
| `is_home` | 1 if home game, 0 if away |
| `is_b2b` | 1 if back-to-back game |
| `days_rest` | Days since last game |
| `opp_def_rating` | Opponent's defensive rating |
| `opp_pace` | Opponent's pace (possessions per game) |
| `avg_vs_opponent` | Player's historical average vs this specific opponent |
| `line_vs_season` | Line minus season average |
| `line_vs_last_5` | Line minus last 5 games average |
| `line_difficulty` | Z-score: (line - season_avg) / std_dev |
| `consistency` | Inverse coefficient of variation: season_avg / std_dev |
| `dvp_rank` | Defense vs Position rank (1-30) for opponent |
| `dvp_allowed` | Points/rebounds/assists allowed by opponent to this position |

### Derived Features (computed at prediction time)

```
line_vs_season = closing_line - season_avg
line_vs_last_5 = closing_line - last_5_avg
line_difficulty = (closing_line - season_avg) / std_dev  # capped at ±5
consistency = season_avg / std_dev  # capped at 0-20
```

### Feature Importance (V2.2)
1. `minutes_avg` - 18.2%
2. `opponent_team` - 8.5%
3. `opp_def_rating` - 7.9%
4. `dvp_rank` - 7.7%
5. `line_vs_last_5` - 7.6%
6. `days_rest` - 7.2%
7. `line_difficulty` - 5.8%
8. `opp_pace` - 5.7%
9. `line_vs_season` - 5.0%
10. `avg_vs_opponent` - 4.8%

### Prediction Process

1. **Feature Extraction**: Gather all 20 features from player stats, game context, and opponent data
2. **Feature Transformation**: Compute derived features (line_difficulty, consistency, etc.)
3. **Model Inference**: CatBoost outputs `prob_over` and `prob_under`
4. **Recommendation**: Pick side with higher probability
5. **Confidence**: The winning probability becomes the confidence score
6. **Betting Threshold**: Only bet when confidence >= 55%

### Output
```python
{
    "prob_over": 0.62,       # Probability of over hitting
    "prob_under": 0.38,      # Probability of under hitting
    "recommended_side": "over",
    "confidence": 0.62,      # Higher of the two probabilities
    "should_bet": True       # True if confidence >= 55%
}
```

---

## MLB Pitcher Strikeouts: Dual Model System

The MLB system uses **two models** with automatic fallback:

| Model | When Used | Method |
|-------|-----------|--------|
| **Monte Carlo Simulation** | Lineups confirmed (primary) | 5,000 PA-level simulations |
| **Beta-Binomial** | Lineups not posted (fallback) | Statistical distribution |

---

### Model 1: Monte Carlo Simulation (Primary)

#### Overview
Simulates 5,000 complete games, plate appearance by plate appearance, using the confirmed batting lineup. Each simulation tracks:
- Individual batter K% vs pitcher handedness
- Pitcher fatigue as pitch count rises
- Times Through Order (TTO) penalty
- Daily "stuff" variance
- Pull probability based on game state

#### Core Algorithm: Log5 Matchup Probability

For each plate appearance, the K probability is calculated using the **Log5 formula**:

```
P(K) = (Batter_K% × Pitcher_K% / League_K%) /
       (Batter_K% × Pitcher_K% / League_K% + (1-Batter_K%) × (1-Pitcher_K%) / (1-League_K%))
```

With **regression to mean** applied to extreme rates:
```python
def regress_to_mean(rate, league_rate):
    if abs(rate - league_rate) < 0.05:
        return rate  # Don't regress near-average rates
    return rate + (league_rate - rate) * 0.20  # Pull 20% toward league average
```

#### Fatigue Model

Pitcher K% decreases as pitch count rises:

| Pitch Count | K% Multiplier |
|-------------|---------------|
| 0-70 | 1.00 (fresh) |
| 70-80 | 0.96 (-4%) |
| 80-90 | 0.92 (-8%) |
| 90-100 | 0.87 (-13%) |
| 100+ | 0.82 (-18%) |

#### Times Through Order (TTO) Penalty

Batters improve each time they face the pitcher:

| Time Through | Batters | K% Multiplier |
|--------------|---------|---------------|
| 1st | 1-9 | 1.00 (baseline) |
| 2nd | 10-18 | 0.93 (-7%) |
| 3rd | 19-27 | 0.82 (-18%) |
| 4th+ | 28+ | 0.78 (-22%) |

#### Per-Game "Stuff" Variance

Each simulation applies a random "stuff" modifier to capture good/bad days:
- Gaussian distribution centered at 1.0
- Standard deviation: 12%
- Capped at 0.70-1.30 (±30%)

#### Pull Probability (Sigmoid Curve)

```python
P(pull) = 1 / (1 + exp(-(pitch_count - 82) / 6))
```

Additional factors:
- +25% per run allowed in current inning
- +40% if down 5+ runs (blowout)
- +15% at end of each inning after 5th
- Ace pitchers (K% >= 27%) get +8 pitch bonus

#### Start Type Variance

| Type | Probability | Effect |
|------|-------------|--------|
| Normal | 70% | Standard simulation |
| Bad Start | 18% | Early hook (40-70 pitches) |
| Hot Start | 12% | +18% K rate, +10 pitch leash |

#### Vegas Total Integration

High-scoring games affect simulations:

| Vegas Total | Effect |
|-------------|--------|
| < 8.5 | +3% K rate (close game) |
| 8.5-9.5 | Neutral |
| 9.5-10.5 | +50% bad start prob, -4 pitch target |
| > 10.5 | +100% bad start prob, -8 pitch target, -15% K rate in blowouts |

#### Simulation Output

```python
{
    "mean_k": 6.8,           # Average strikeouts across 5,000 sims
    "median_k": 7,
    "std_k": 2.1,
    "prob_over_5.5": 0.72,   # P(K > 5.5)
    "prob_over_6.5": 0.58,
    "avg_ip": 5.2,           # Average innings pitched
    "avg_pitches": 84,
    "cumulative_probs": {    # P(N+ Ks)
        1: 0.997,
        5: 0.802,
        7: 0.507,
        10: 0.139
    }
}
```

---

### Model 2: Beta-Binomial Distribution (Fallback)

Used when lineups aren't confirmed (typically 2-3 hours before game time).

#### Step 1: Expected Batters Faced

```
Base BF = Pitcher's last 5 starts average BF (or pitch count / opponent P/PA)
Vegas Adjustment = 1 + (vegas_total - 9.0) × 0.012  # ±6% max
Favorite Adjustment = ±2.5% based on spread

Expected BF = Base BF × Vegas Adj × Favorite Adj
```

Clamped to 18-27 range.

#### Step 2: True Talent K%

```
xK% = (SwStr% × 1.4) + (CSW% × 0.35) - 6.0   # If CSW% available
xK% = SwStr% × 1.9                            # Fallback

IP Weight = min(0.7, IP / 80)
Base K% = (xK% × (1 - IP_weight)) + (Actual_K% × IP_weight)

Form Adjustment = (Last 5 K rate - Season K rate) × 15%
Talent K% = Base K% + Form Adjustment
```

#### Step 3: Matchup Adjustments

```
Opponent Adj = (Opponent K% vs Hand / League K%) - 1
Park Adj = Park K Factor - 1.0
Umpire Adj = (Umpire K Index - 1.0) × 0.4

Matchup Multiplier = 1.0 + (Opponent × 0.35) + (Park × 0.50) + Umpire
Adjusted K% = Talent K% × Matchup Multiplier
```

#### Step 4: Beta-Binomial Distribution

```python
# Dynamic concentration based on pitcher consistency
CV = last_10_k_std / last_10_k_avg
concentration = 1 / (CV ** 2)  # Clamped 6-25

alpha = K_probability × concentration
beta = (1 - K_probability) × concentration

distribution = BetaBinomial(n=Expected_BF, alpha=alpha, beta=beta)
projected_ks = distribution.mean()
prob_over = 1 - distribution.cdf(line)
```

#### Output Format

Same as simulation, but with `model_type: "beta_binomial"`.

---

## Shared: Edge & EV Calculation

Both MLB models use the same betting math:

```python
# Convert American odds to decimal
def odds_to_decimal(odds):
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))

# Calculate Expected Value
EV = (probability × (decimal_odds - 1)) - (1 - probability)

# Calculate Edge (vs implied probability)
implied_prob = 1 / decimal_odds
edge = our_probability - implied_prob

# Kelly Criterion
kelly = (probability × (decimal_odds - 1) - (1 - probability)) / (decimal_odds - 1)
kelly = max(0, min(0.25, kelly))  # Cap at 25%
```

---

## Confidence Grades

### NBA
Based on CatBoost confidence (probability of winning side):
- **55%+**: Should bet
- **< 55%**: Pass

### MLB
Based on edge magnitude:
| Edge | Grade |
|------|-------|
| >= 8% | A |
| >= 5% | B |
| >= 2% | C |
| < 2% | D |

---

## Data Sources

### NBA
- **Player Stats**: NBA API (game logs, season averages)
- **Team Stats**: NBA API (defensive rating, pace)
- **DvP Data**: FantasyPros (defense vs position)
- **Odds**: The Odds API (live lines from 5+ sportsbooks)

### MLB
- **Pitcher Stats**: FanGraphs via pybaseball (K%, SwStr%, CSW%, BB%)
- **Team Batting**: FanGraphs via pybaseball (team K% vs LHP/RHP)
- **Lineups**: MLB Stats API (confirmed lineups)
- **Batter Stats**: Baseball Reference (individual K%/BB%)
- **Park Factors**: Baseball Savant (Statcast SO factors)
- **Odds**: The Odds API

---

## Model Selection Flow

```
NBA Request
    │
    └──► CatBoost V2.2 ──► Prediction

MLB Request
    │
    ├── Lineup Confirmed? ──► YES ──► Monte Carlo Simulation (5,000 sims)
    │
    └── NO ──► Beta-Binomial Distribution
```

---

## Performance Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| NBA CatBoost V2.2 | 59.7% CV | +27.2% ROI at 55%+ confidence |
| MLB Combined | 51.5% | Simulation primary, Beta-Binomial fallback |

---

*Last Updated: May 2026*
