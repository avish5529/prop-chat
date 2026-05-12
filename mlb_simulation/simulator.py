#!/usr/bin/env python3
"""
MLB Strikeout Simulation Engine

Monte Carlo simulation for pitcher strikeout props.
Simulates each plate appearance using Log5 matchup probabilities.

Phase 2 of Tier 3 MLB Simulation.
"""

import random
import logging
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter
import statistics

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# League averages (2023-2025)
LEAGUE_K_PCT = 22.7  # Will be updated from actual data
LEAGUE_BB_PCT = 8.5
LEAGUE_HIT_PCT = 23.5  # Approximate BA + errors

# Pitches per PA (MLB average ~3.9-4.0)
PITCHES_PER_PA_BASE = 3.95
PITCHES_PER_K = 4.5      # Ks take more pitches
PITCHES_PER_BB = 5.5     # BBs take more pitches
PITCHES_PER_HIT = 3.2    # Hits often early in count
PITCHES_PER_OUT = 3.8    # Outs average

# Team abbreviation normalization (FanGraphs -> MLB Stats API)
TEAM_ABBREV_MAP = {
    'SFG': 'SF', 'SDP': 'SD', 'KCR': 'KC', 'TBR': 'TB',
    'WSN': 'WSH', 'CHW': 'CWS', 'ANA': 'LAA',
}

def normalize_team_abbrev(abbrev: str) -> str:
    """Convert FanGraphs team abbreviation to MLB Stats API format."""
    return TEAM_ABBREV_MAP.get(abbrev.upper(), abbrev.upper())

# Fatigue model (K% multiplier based on pitch count)
# Research: K% drops significantly in high-pitch counts
# Source: Baseball Savant pitch-by-pitch data, FanGraphs pitcher fatigue studies
FATIGUE_THRESHOLDS = [
    (70, 1.00),   # 0-70 pitches: no fatigue (fresh)
    (80, 0.96),   # 70-80: -4% (getting warm)
    (90, 0.92),   # 80-90: -8% (tired)
    (100, 0.87),  # 90-100: -13% (gassed)
    (999, 0.82),  # 100+: -18% (cooked)
]

# Pull probability factors (tuned for ~5.4 IP, ~85 pitch average - 2024 MLB avg)
# Uses sigmoid curve for smoother, more realistic pull probability
MIN_PITCHES_TO_PULL = 40        # Don't pull before 40 pitches unless disaster

# Sigmoid curve parameters for pull probability
# Formula: P(pull) = 1 / (1 + exp(-(pitches - midpoint) / steepness))
# Calibrated for ~5.4 IP, ~85 pitches average (2024 MLB)
PULL_SIGMOID_MIDPOINT = 82      # 50% pull probability at ~82 pitches (effective)
PULL_SIGMOID_STEEPNESS = 6      # Steeper curve (lower = sharper transition)
PULL_PROB_RUNS_IN_INNING = 0.25 # +25% per run allowed in current inning
PULL_PROB_BLOWOUT = 0.40        # +40% if down 5+ runs
PULL_PROB_END_OF_INNING = 0.15  # +15% chance to pull at end of each inning after 5th

# Times Through Order (TTO) penalty - batters improve each time they face pitcher
# Research: OPS+ jumps from 91→117 (28% increase) by third time through
# ERA increases from 4.08→4.57 (12% increase) third time through
# Source: Baseball Prospectus, MLB.com TTO research
TTO_MULTIPLIERS = {
    1: 1.00,   # First time through lineup: baseline K%
    2: 0.93,   # Second time: -7% K rate (was -5%)
    3: 0.82,   # Third time: -18% K rate (was -12%)
    4: 0.78,   # Fourth time (rare): -22% K rate (was -15%)
}

# Bad start variance - some percentage of starts are disasters
# Research: Only 36% of starts are quality starts (6+ IP), many end early
# Early exits like Jacob Lopez (1.0 IP), Jesus Luzardo (4.2 IP) need to be captured
BAD_START_PROBABILITY = 0.18    # 18% of starts are bad (was 10%)
BAD_START_MAX_PITCHES = 70      # Bad starts end by 40-70 pitches (was 65)

# Hot start variance - some percentage of starts are exceptional
# Captures games like Bailey Ober (10 Ks), Gavin Williams (11 Ks)
# When pitcher's "stuff is ON", they dominate and go deep
HOT_START_PROBABILITY = 0.12    # 12% of starts are exceptional
HOT_START_K_BOOST = 1.18        # +18% K rate when stuff is working
HOT_START_PULL_DELAY = 10       # Managers let hot pitchers go +10 pitches longer

# Ace pitcher longer leash - star pitchers get to go deeper
# Based on pitcher K% (higher K% = more trust from manager)
ACE_K_THRESHOLD = 27.0          # K% above this = ace treatment
ACE_PULL_DELAY = 8              # Aces get 8 extra pitches before pull consideration
ACE_BAD_START_REDUCTION = 0.6   # Aces have 60% of bad start probability (was 50%)
ACE_HOT_START_BOOST = 1.15      # Aces get additional +15% boost on hot starts (multiplicative)

# Workload distribution - target pitch count varies per game
# Some games managers let pitchers go deeper, others they're quick to pull
# Models: bullpen availability, game situation, manager tendency variance
WORKLOAD_MEAN = 0               # Mean adjustment to pitch targets (centered at 0)
WORKLOAD_STD = 8                # Std dev in pitch count variance (~±8 pitches)
WORKLOAD_MIN = -15              # Maximum early pull (-15 pitches from normal)
WORKLOAD_MAX = 15               # Maximum extended leash (+15 pitches from normal)

# Per-game "stuff" variance - pitchers have good/bad days
# Research shows significant game-to-game K rate variance beyond what matchups explain
# This captures "stuff is working" vs "doesn't have it today" variance
STUFF_VARIANCE_STD = 0.12       # ~12% standard deviation in daily K rate
STUFF_MODIFIER_MIN = 0.70       # Floor: worst stuff day = -30% K rate
STUFF_MODIFIER_MAX = 1.30       # Ceiling: best stuff day = +30% K rate

# Log5 regression to mean - reduces bias for extreme rates
# Research: Standard Log5 overestimates K% for high-K pitchers vs high-K batters
# Regressing rates toward league average reduces this asymmetric bias
# Source: SABR, Tom Tango's Log5 corrections
REGRESSION_FACTOR = 0.20        # Regress rates 20% toward league average
REGRESSION_THRESHOLD = 0.05    # Only regress if rate is >5% from league avg (~17-28% range unaffected)

# Simulation defaults
# Industry standard: 5,000 sims (FullCountProps, BallparkPal use similar counts)
# Higher count reduces variance in probability estimates
DEFAULT_NUM_SIMS = 5000

# Vegas total / game script adjustments (NEW - addresses high-scoring game failures)
# Research: High-total games have more early pulls, more variance, lower K rates
# Games like SEA 11 @ STL 9 (20 runs) had pitchers pulled at 3 IP
VEGAS_TOTAL_NEUTRAL = 8.5          # MLB average game total
VEGAS_TOTAL_HIGH = 9.5             # High-scoring game threshold
VEGAS_TOTAL_SHOOTOUT = 10.5        # Shootout territory

# Early pull adjustment based on expected runs (higher total = more runs = earlier pulls)
HIGH_TOTAL_BAD_START_BOOST = 1.5   # +50% bad start probability in high-total games
SHOOTOUT_BAD_START_BOOST = 2.0     # +100% bad start probability in shootout games

# IP reduction for high-total games (more runs = shorter outings)
HIGH_TOTAL_IP_PENALTY = -0.3       # -0.3 IP expected in high-total games
SHOOTOUT_IP_PENALTY = -0.6         # -0.6 IP expected in shootout games

# Game script K% adjustments (when team is losing badly)
# Research: Pitchers in blowouts "cruise" - fewer strikeouts, pitch to contact
BLOWOUT_DEFICIT = 4                # Down 4+ runs = blowout
BLOWOUT_K_PENALTY = 0.85           # -15% K rate when cruising in blowout
CLOSE_GAME_K_BOOST = 1.03          # +3% K rate in close games (pitcher engaged)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BatterStats:
    """Batter statistics for simulation."""
    name: str
    k_pct: float      # 0-100 scale
    bb_pct: float     # 0-100 scale

    @property
    def k_rate(self) -> float:
        """K% as decimal (0-1)."""
        return self.k_pct / 100

    @property
    def bb_rate(self) -> float:
        """BB% as decimal (0-1)."""
        return self.bb_pct / 100


def to_baseball_ip(decimal_ip: float) -> float:
    """
    Convert decimal innings to baseball notation.

    Baseball uses .0, .1, .2 for outs (not .33, .67):
    - 5.0 = 5 complete innings
    - 5.1 = 5 innings + 1 out
    - 5.2 = 5 innings + 2 outs
    - 6.0 = 6 complete innings

    Example: 5.67 (decimal) → 5.2 (baseball)
    """
    full_innings = int(decimal_ip)
    partial = decimal_ip - full_innings
    outs = round(partial * 3)  # 0, 1, or 2 outs
    return full_innings + (outs / 10)


@dataclass
class PitcherStats:
    """Pitcher statistics for simulation."""
    name: str
    k_pct: float      # 0-100 scale
    bb_pct: float     # 0-100 scale
    hand: str         # 'L' or 'R'

    @property
    def k_rate(self) -> float:
        """K% as decimal (0-1)."""
        return self.k_pct / 100

    @property
    def bb_rate(self) -> float:
        """BB% as decimal (0-1)."""
        return self.bb_pct / 100


@dataclass
class GameState:
    """Current state of a simulated game."""
    inning: int = 1
    outs: int = 0
    runs_this_inning: int = 0
    total_runs: int = 0
    pitch_count: int = 0
    strikeouts: int = 0
    batter_idx: int = 0  # 0-8 for lineup position
    pitcher_pulled: bool = False


@dataclass
class SimulationResult:
    """Result of a single simulation."""
    strikeouts: int
    innings_pitched: float
    pitch_count: int
    batters_faced: int


# ============================================================================
# LOG5 PROBABILITY CALCULATOR
# ============================================================================

def regress_to_mean(rate: float, league_rate: float) -> float:
    """
    Regress an extreme rate toward the league average.

    This reduces the bias in Log5 for asymmetric probabilities. High-K pitchers
    and high-K batters have their rates pulled slightly toward league average,
    which produces more realistic matchup probabilities.

    Args:
        rate: Individual rate (0-1 scale)
        league_rate: League average rate (0-1 scale)

    Returns:
        Regressed rate (0-1 scale)
    """
    deviation = abs(rate - league_rate)

    # Only regress if rate is significantly different from league average
    if deviation < REGRESSION_THRESHOLD:
        return rate

    # Apply regression toward mean
    regressed = rate + (league_rate - rate) * REGRESSION_FACTOR
    return regressed


def log5_probability(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """
    Calculate matchup probability using Log5 formula with regression to mean.

    Log5 is the standard method for combining two rates when both
    parties contribute to the outcome (batter vs pitcher).

    This implementation includes regression to mean to reduce known Log5 bias
    that overestimates K rates when both parties have extreme rates.

    Args:
        batter_rate: Batter's rate (0-1 scale, e.g., K%)
        pitcher_rate: Pitcher's rate (0-1 scale)
        league_rate: League average rate (0-1 scale)

    Returns:
        Combined probability (0-1 scale)
    """
    if league_rate <= 0 or league_rate >= 1:
        league_rate = 0.227  # Default

    # Avoid division by zero
    batter_rate = max(0.01, min(0.99, batter_rate))
    pitcher_rate = max(0.01, min(0.99, pitcher_rate))

    # Apply regression to mean for extreme rates
    batter_rate_adj = regress_to_mean(batter_rate, league_rate)
    pitcher_rate_adj = regress_to_mean(pitcher_rate, league_rate)

    # Log5 formula with regressed rates
    numerator = batter_rate_adj * pitcher_rate_adj / league_rate
    denominator = (
        numerator +
        (1 - batter_rate_adj) * (1 - pitcher_rate_adj) / (1 - league_rate)
    )

    return numerator / denominator if denominator > 0 else league_rate


# ============================================================================
# FATIGUE MODEL
# ============================================================================

def get_fatigue_multiplier(pitch_count: int) -> float:
    """
    Get K% multiplier based on pitcher fatigue.

    Research shows K% drops significantly as pitch count increases:
    - Fresh (0-70): Full effectiveness
    - Warm (70-80): -4% drop
    - Tired (80-90): -8% drop
    - Gassed (90-100): -13% drop
    - Cooked (100+): -18% drop

    Source: Baseball Savant pitch-by-pitch data shows K rate drops
    accelerate after 70 pitches, with dramatic decline past 90.

    Args:
        pitch_count: Current pitch count

    Returns:
        Multiplier for pitcher's K rate (0.82-1.00)
    """
    for threshold, multiplier in FATIGUE_THRESHOLDS:
        if pitch_count <= threshold:
            return multiplier
    return FATIGUE_THRESHOLDS[-1][1]


# ============================================================================
# TIMES THROUGH ORDER (TTO) MODEL
# ============================================================================

def get_tto_multiplier(batters_faced: int) -> float:
    """
    Get K% multiplier based on times through order.

    Research shows batters improve each time they face a pitcher:
    - 1st time (batters 1-9): Baseline performance
    - 2nd time (batters 10-18): -7% K rate for pitcher
    - 3rd time (batters 19-27): -18% K rate for pitcher
    - 4th time (batters 28+): -22% K rate for pitcher

    Research basis: OPS+ increases from 91→117 (28%) by third time through.
    ERA increases from 4.08→4.57 (12%) by third time through.
    This is one of the most robust findings in sabermetrics.

    Args:
        batters_faced: Total batters faced so far (0-indexed, so add 1)

    Returns:
        Multiplier for pitcher's K rate (0.78-1.00)
    """
    # Determine which time through the order
    batter_num = batters_faced + 1  # Convert to 1-indexed
    tto = min(4, (batter_num - 1) // 9 + 1)  # 1, 2, 3, or 4
    return TTO_MULTIPLIERS.get(tto, 0.85)


# ============================================================================
# PER-GAME STUFF VARIANCE MODEL
# ============================================================================

def get_stuff_modifier() -> float:
    """
    Generate a per-game "stuff" modifier for pitcher K rate.

    This captures the reality that pitchers have good and bad days
    independent of matchups, fatigue, or TTO effects. Some days the
    fastball is popping, the slider is biting - other days it's flat.

    Research shows significant game-to-game variance in K rates that
    isn't explained by matchup factors alone. This modifier adds that
    realistic variance to the simulation.

    Returns:
        Multiplier for pitcher's K rate (0.70-1.30, centered at 1.0)
        - 1.0 = normal stuff
        - >1.0 = stuff is working (more Ks)
        - <1.0 = doesn't have it today (fewer Ks)
    """
    modifier = random.gauss(1.0, STUFF_VARIANCE_STD)
    return max(STUFF_MODIFIER_MIN, min(STUFF_MODIFIER_MAX, modifier))


def get_workload_modifier() -> int:
    """
    Generate a per-game workload modifier for pitch count targets.

    This captures the variance in how deep pitchers are allowed to go:
    - Bullpen availability (tired pen = pitcher goes deeper)
    - Manager tendencies (some pull earlier than others)
    - Game situation (blowouts vs close games)
    - Day-to-day variance

    Returns:
        Integer adjustment to pitch count thresholds (-15 to +15)
        - Positive = extended leash (go deeper)
        - Negative = short leash (pulled earlier)
    """
    modifier = int(random.gauss(WORKLOAD_MEAN, WORKLOAD_STD))
    return max(WORKLOAD_MIN, min(WORKLOAD_MAX, modifier))


# ============================================================================
# PULL PROBABILITY MODEL
# ============================================================================

def sigmoid_pull_probability(pitch_count: int, midpoint: float = PULL_SIGMOID_MIDPOINT,
                              steepness: float = PULL_SIGMOID_STEEPNESS) -> float:
    """
    Calculate base pull probability using sigmoid curve.

    The sigmoid function creates a smooth S-curve that matches
    realistic manager behavior - low at first, accelerating
    through the middle, then leveling off near certainty.

    Args:
        pitch_count: Effective pitch count
        midpoint: Pitch count where probability = 50%
        steepness: Controls curve steepness (lower = steeper)

    Returns:
        Probability between 0 and 1
    """
    return 1.0 / (1.0 + math.exp(-(pitch_count - midpoint) / steepness))


def should_pull_pitcher(state: GameState, is_blowout: bool = False, ace_pitch_bonus: int = 0) -> bool:
    """
    Determine if pitcher should be pulled using sigmoid probability curve.

    Factors:
    - Pitch count (main driver via sigmoid curve)
    - Inning (managers prefer clean innings)
    - Runs allowed in current inning
    - Blowout situations
    - Minimum pitch threshold
    - Ace/workload bonus (shifts sigmoid midpoint)

    Args:
        state: Current game state
        is_blowout: Whether team is down 5+ runs
        ace_pitch_bonus: Extra pitches before pull consideration (includes ace + hot start + workload)

    Returns:
        True if pitcher should be pulled
    """
    # Apply bonus - effectively shifts sigmoid curve to the right
    effective_pitch_count = state.pitch_count - ace_pitch_bonus

    if effective_pitch_count < MIN_PITCHES_TO_PULL:
        return False

    # Base probability from sigmoid curve
    base_prob = sigmoid_pull_probability(effective_pitch_count)

    # Add probability for runs in inning (disaster inning)
    base_prob += state.runs_this_inning * PULL_PROB_RUNS_IN_INNING

    # Add probability for blowout
    if is_blowout:
        base_prob += PULL_PROB_BLOWOUT

    # End of inning considerations (managers love clean breaks)
    if state.outs == 3:
        # After 5th inning, increasing chance to pull at end of inning
        if state.inning >= 5:
            innings_past_5 = state.inning - 5
            base_prob += PULL_PROB_END_OF_INNING * (innings_past_5 + 1)

        # End of inning bonus scales with pitch count (managers like clean breaks)
        if effective_pitch_count >= 80:
            base_prob += 0.25  # Strong end-of-inning pull bias at high pitch count
        elif effective_pitch_count >= 70:
            base_prob += 0.12  # Moderate bias

    return random.random() < min(0.98, base_prob)


# ============================================================================
# PLATE APPEARANCE SIMULATOR
# ============================================================================

def simulate_pa(
    batter: BatterStats,
    pitcher: PitcherStats,
    fatigue: float,
    batters_faced: int = 0,
    stuff_modifier: float = 1.0,
    league_k: float = LEAGUE_K_PCT / 100,
    league_bb: float = LEAGUE_BB_PCT / 100
) -> Tuple[str, int]:
    """
    Simulate a single plate appearance.

    Args:
        batter: Batter statistics
        pitcher: Pitcher statistics
        fatigue: Fatigue multiplier for pitcher (0.88-1.00)
        batters_faced: Total batters faced so far (for TTO penalty)
        stuff_modifier: Per-game "stuff" modifier (0.70-1.30, default 1.0)
        league_k: League K rate
        league_bb: League BB rate

    Returns:
        Tuple of (outcome, pitches)
        outcome: 'K', 'BB', 'HIT', or 'OUT'
    """
    # Calculate matchup probabilities with fatigue, TTO penalty, AND stuff modifier
    tto_mult = get_tto_multiplier(batters_faced)
    k_prob = log5_probability(batter.k_rate, pitcher.k_rate * fatigue * tto_mult * stuff_modifier, league_k)
    bb_prob = log5_probability(batter.bb_rate, pitcher.bb_rate, league_bb)

    # Remaining probability split between hits and outs
    remaining = 1.0 - k_prob - bb_prob
    hit_prob = remaining * (LEAGUE_HIT_PCT / 100) / (1 - league_k - league_bb)
    hit_prob = min(remaining * 0.7, hit_prob)  # Cap hit probability
    out_prob = remaining - hit_prob

    # Roll for outcome
    roll = random.random()

    if roll < k_prob:
        outcome = 'K'
        pitches = int(random.gauss(PITCHES_PER_K, 1.2))
    elif roll < k_prob + bb_prob:
        outcome = 'BB'
        pitches = int(random.gauss(PITCHES_PER_BB, 1.0))
    elif roll < k_prob + bb_prob + hit_prob:
        outcome = 'HIT'
        pitches = int(random.gauss(PITCHES_PER_HIT, 1.0))
    else:
        outcome = 'OUT'
        pitches = int(random.gauss(PITCHES_PER_OUT, 1.0))

    # Ensure valid pitch count
    pitches = max(1, min(12, pitches))

    return outcome, pitches


# ============================================================================
# GAME SIMULATOR
# ============================================================================

def simulate_game(
    pitcher: PitcherStats,
    lineup: List[BatterStats],
    league_k: float = LEAGUE_K_PCT / 100,
    league_bb: float = LEAGUE_BB_PCT / 100,
    is_bad_start: bool = False,
    is_hot_start: bool = False,
    is_ace: bool = False,
    stuff_modifier: float = 1.0,
    workload_modifier: int = 0,
    vegas_total: float = None,
    opp_runs_modifier: float = 1.0
) -> SimulationResult:
    """
    Simulate a single game for a starting pitcher.

    Args:
        pitcher: Pitcher statistics
        lineup: List of 9 batters in order
        league_k: League K rate
        league_bb: League BB rate
        is_bad_start: If True, pitcher gets pulled early (disaster outing)
        is_hot_start: If True, pitcher gets extended leash (stuff is working)
        is_ace: If True, pitcher gets longer leash before being pulled
        stuff_modifier: Per-game "stuff" modifier (0.70-1.30, default 1.0)
        workload_modifier: Per-game workload adjustment (-15 to +15 pitches)
        vegas_total: Expected game total from Vegas (affects game script)
        opp_runs_modifier: Opponent run-scoring modifier for this sim (0.7-1.3)

    Returns:
        SimulationResult with strikeouts, IP, pitches, BF
    """
    state = GameState()

    # Track simulated opponent runs for game script adjustments
    sim_opp_runs = 0
    is_blowout = False

    # Calculate total pitch bonus from ace status, hot start, and workload variance
    ace_pitch_bonus = ACE_PULL_DELAY if is_ace else 0
    hot_start_bonus = HOT_START_PULL_DELAY if is_hot_start else 0

    # Adjust workload for high-total games (pitchers go shorter in shootouts)
    vegas_workload_adj = 0
    if vegas_total and vegas_total >= VEGAS_TOTAL_SHOOTOUT:
        vegas_workload_adj = -8  # Pull ~8 pitches earlier in shootouts
    elif vegas_total and vegas_total >= VEGAS_TOTAL_HIGH:
        vegas_workload_adj = -4  # Pull ~4 pitches earlier in high-total games

    total_pitch_bonus = ace_pitch_bonus + hot_start_bonus + workload_modifier + vegas_workload_adj

    # Bad start: pitcher gets pulled early (40-70 pitches)
    bad_start_max_pitches = random.randint(40, BAD_START_MAX_PITCHES) if is_bad_start else 999

    while not state.pitcher_pulled and state.inning <= 9:
        # Get current batter
        batter = lineup[state.batter_idx % 9]

        # Calculate fatigue
        fatigue = get_fatigue_multiplier(state.pitch_count)

        # Simulate opponent runs for game script (at start of each inning)
        # Higher vegas_total = more runs expected = more blowout scenarios
        if state.outs == 0 and state.inning > 1 and vegas_total:
            # Expected opponent runs per inning based on vegas total
            # Vegas total is for BOTH teams, so divide by 2 for one team's share
            # Then divide by 9 for per-inning rate, with variance
            opp_runs_per_inning = (vegas_total / 2) / 9 * opp_runs_modifier
            # Simulate runs scored this inning using Poisson-like distribution
            inning_runs = 0
            for _ in range(3):  # 3 chances for runs (simplified)
                if random.random() < opp_runs_per_inning:
                    inning_runs += 1
            sim_opp_runs += inning_runs

            # Check if we're in a blowout (down 4+ runs)
            is_blowout = sim_opp_runs - state.total_runs >= BLOWOUT_DEFICIT

        # Apply game script K% adjustment
        game_script_modifier = 1.0
        if is_blowout:
            # Pitcher "cruising" in blowout - fewer strikeouts, pitching to contact
            game_script_modifier = BLOWOUT_K_PENALTY
        elif vegas_total and vegas_total <= VEGAS_TOTAL_NEUTRAL - 1.0:
            # Low-scoring game, close throughout - pitcher stays engaged
            game_script_modifier = CLOSE_GAME_K_BOOST

        # Combine stuff modifier with game script modifier
        effective_stuff = stuff_modifier * game_script_modifier

        # Simulate plate appearance with fatigue, TTO penalty, and stuff modifier
        outcome, pitches = simulate_pa(
            batter, pitcher, fatigue,
            batters_faced=state.batter_idx,
            stuff_modifier=effective_stuff,
            league_k=league_k,
            league_bb=league_bb
        )

        # Bad start: force pull at low pitch count
        if is_bad_start and state.pitch_count + pitches >= bad_start_max_pitches:
            state.pitch_count += pitches
            state.batter_idx += 1
            if outcome == 'K':
                state.strikeouts += 1
            state.pitcher_pulled = True
            break

        # Update state
        state.pitch_count += pitches
        state.batter_idx += 1

        if outcome == 'K':
            state.strikeouts += 1
            state.outs += 1
        elif outcome == 'OUT':
            state.outs += 1
        elif outcome == 'BB':
            pass  # Runner on base (simplified - no baserunning sim)
        elif outcome == 'HIT':
            # Simplified run scoring: 20% chance of run on a hit
            if random.random() < 0.20:
                state.runs_this_inning += 1
                state.total_runs += 1

        # Check for end of inning
        if state.outs >= 3:
            state.inning += 1
            state.outs = 0
            state.runs_this_inning = 0

            # Check for pull at end of inning (aces and hot starters get longer leash)
            if should_pull_pitcher(state, ace_pitch_bonus=total_pitch_bonus):
                state.pitcher_pulled = True
        else:
            # Check for mid-inning pull (disaster inning)
            if state.runs_this_inning >= 3 and state.pitch_count >= 60:
                if should_pull_pitcher(state, ace_pitch_bonus=total_pitch_bonus):
                    state.pitcher_pulled = True

    # Calculate innings pitched
    full_innings = state.inning - 1
    partial_outs = state.outs
    innings_pitched = full_innings + (partial_outs / 3)

    return SimulationResult(
        strikeouts=state.strikeouts,
        innings_pitched=round(innings_pitched, 1),
        pitch_count=state.pitch_count,
        batters_faced=state.batter_idx
    )


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_simulation(
    pitcher: PitcherStats,
    lineup: List[BatterStats],
    n_sims: int = DEFAULT_NUM_SIMS,
    league_k: float = LEAGUE_K_PCT / 100,
    league_bb: float = LEAGUE_BB_PCT / 100,
    vegas_total: float = None
) -> Dict:
    """
    Run Monte Carlo simulation for strikeout projection.

    Args:
        pitcher: Pitcher statistics
        lineup: List of 9 batters
        n_sims: Number of simulations to run
        league_k: League K rate
        league_bb: League BB rate
        vegas_total: Expected game total from Vegas (affects game script, pulls)

    Returns:
        Dict with simulation results and probability distribution
    """
    results = []
    stuff_modifiers = []    # Track for analysis
    workload_modifiers = [] # Track for analysis
    hot_start_count = 0     # Track for analysis

    # Determine if pitcher is an "ace" (gets longer leash)
    is_ace = pitcher.k_pct >= ACE_K_THRESHOLD

    # Ace pitchers have lower bad start probability (managers trust them more)
    bad_start_prob = BAD_START_PROBABILITY * ACE_BAD_START_REDUCTION if is_ace else BAD_START_PROBABILITY

    # High-total games have more bad starts (more runs = more early pulls)
    if vegas_total:
        if vegas_total >= VEGAS_TOTAL_SHOOTOUT:
            bad_start_prob *= SHOOTOUT_BAD_START_BOOST  # +100% bad start chance
        elif vegas_total >= VEGAS_TOTAL_HIGH:
            bad_start_prob *= HIGH_TOTAL_BAD_START_BOOST  # +50% bad start chance

    for _ in range(n_sims):
        # Generate per-game "stuff" modifier - captures good/bad days
        stuff_modifier = get_stuff_modifier()

        # Generate per-game workload modifier - captures variance in how deep pitchers go
        workload_modifier = get_workload_modifier()
        workload_modifiers.append(workload_modifier)

        # Determine start type (mutually exclusive: bad, hot, or normal)
        roll = random.random()
        is_bad_start = False
        is_hot_start = False

        if roll < bad_start_prob:
            # Bad start - pitcher gets pulled early
            is_bad_start = True
        elif roll < bad_start_prob + HOT_START_PROBABILITY:
            # Hot start - stuff is ON, pitcher dominates
            is_hot_start = True
            hot_start_count += 1
            # Apply hot start K boost to stuff modifier
            stuff_modifier *= HOT_START_K_BOOST
            # Aces get additional boost on hot starts (multiplicative)
            if is_ace:
                stuff_modifier *= ACE_HOT_START_BOOST
            # Cap at max stuff modifier
            stuff_modifier = min(stuff_modifier, STUFF_MODIFIER_MAX * 1.2)

        stuff_modifiers.append(stuff_modifier)

        # Generate opponent run modifier for this simulation
        # Variance in how many runs opponent scores (0.7 to 1.3 of expected)
        opp_runs_modifier = random.gauss(1.0, 0.15)
        opp_runs_modifier = max(0.7, min(1.3, opp_runs_modifier))

        result = simulate_game(
            pitcher, lineup, league_k, league_bb,
            is_bad_start, is_hot_start, is_ace, stuff_modifier, workload_modifier,
            vegas_total, opp_runs_modifier
        )
        results.append(result)

    # Extract strikeout counts
    k_counts = [r.strikeouts for r in results]

    # Calculate statistics
    mean_k = statistics.mean(k_counts)
    median_k = statistics.median(k_counts)
    std_k = statistics.stdev(k_counts) if len(k_counts) > 1 else 0

    # Calculate probability distribution
    k_distribution = Counter(k_counts)
    total = len(k_counts)

    # Calculate over/under probabilities for common lines
    over_probs = {}
    for line in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]:
        over_count = sum(1 for k in k_counts if k > line)
        over_probs[line] = over_count / total

    # Calculate cumulative probabilities P(N+ Ks) for 1 through 12
    cumulative_probs = {}
    for n in range(1, 13):
        count = sum(1 for k in k_counts if k >= n)
        cumulative_probs[n] = round(count / total, 4)

    # Additional stats
    avg_ip = statistics.mean([r.innings_pitched for r in results])
    avg_pitches = statistics.mean([r.pitch_count for r in results])
    avg_bf = statistics.mean([r.batters_faced for r in results])

    # Stuff modifier stats (for analysis/debugging)
    avg_stuff = statistics.mean(stuff_modifiers)
    std_stuff = statistics.stdev(stuff_modifiers) if len(stuff_modifiers) > 1 else 0

    # Workload modifier stats (for analysis/debugging)
    avg_workload = statistics.mean(workload_modifiers)
    std_workload = statistics.stdev(workload_modifiers) if len(workload_modifiers) > 1 else 0

    return {
        'pitcher': pitcher.name,
        'n_sims': n_sims,
        'mean_k': round(mean_k, 2),
        'median_k': median_k,
        'std_k': round(std_k, 2),
        'min_k': min(k_counts),
        'max_k': max(k_counts),
        'over_probs': {k: round(v, 4) for k, v in over_probs.items()},
        'cumulative_probs': cumulative_probs,  # P(1+ K), P(2+ K), ..., P(12+ K)
        'distribution': dict(sorted(k_distribution.items())),
        'avg_ip': to_baseball_ip(avg_ip),  # Baseball notation: 5.2 = 5 innings + 2 outs
        'avg_pitches': round(avg_pitches, 1),
        'avg_bf': round(avg_bf, 1),
        'stuff_variance': {
            'avg_modifier': round(avg_stuff, 3),
            'std_modifier': round(std_stuff, 3),
            'min_modifier': round(min(stuff_modifiers), 3),
            'max_modifier': round(max(stuff_modifiers), 3),
            'hot_start_pct': round(hot_start_count / n_sims * 100, 1),
        },
        'workload_variance': {
            'avg_modifier': round(avg_workload, 1),
            'std_modifier': round(std_workload, 1),
            'min_modifier': min(workload_modifiers),
            'max_modifier': max(workload_modifiers),
        },
        # Vegas total / game script tracking
        'game_script': {
            'vegas_total': vegas_total,
            'is_high_total': vegas_total >= VEGAS_TOTAL_HIGH if vegas_total else False,
            'is_shootout': vegas_total >= VEGAS_TOTAL_SHOOTOUT if vegas_total else False,
            'bad_start_prob_adj': (
                SHOOTOUT_BAD_START_BOOST if vegas_total and vegas_total >= VEGAS_TOTAL_SHOOTOUT
                else HIGH_TOTAL_BAD_START_BOOST if vegas_total and vegas_total >= VEGAS_TOTAL_HIGH
                else 1.0
            ),
        },
        # Calibration tracking - for comparing predicted vs actual
        'calibration': {
            # Percentiles for distribution analysis
            'p5': sorted(k_counts)[int(len(k_counts) * 0.05)],
            'p10': sorted(k_counts)[int(len(k_counts) * 0.10)],
            'p25': sorted(k_counts)[int(len(k_counts) * 0.25)],
            'p50': sorted(k_counts)[int(len(k_counts) * 0.50)],  # Median
            'p75': sorted(k_counts)[int(len(k_counts) * 0.75)],
            'p90': sorted(k_counts)[int(len(k_counts) * 0.90)],
            'p95': sorted(k_counts)[int(len(k_counts) * 0.95)],
            # Range probabilities for calibration comparison
            'prob_0_3': round(sum(1 for k in k_counts if k <= 3) / total, 4),
            'prob_4_6': round(sum(1 for k in k_counts if 4 <= k <= 6) / total, 4),
            'prob_7_9': round(sum(1 for k in k_counts if 7 <= k <= 9) / total, 4),
            'prob_10_plus': round(sum(1 for k in k_counts if k >= 10) / total, 4),
        },
    }


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def simulate_strikeouts(
    pitcher_name: str,
    opponent_team: str,
    n_sims: int = DEFAULT_NUM_SIMS
) -> Optional[Dict]:
    """
    Run strikeout simulation for a pitcher vs opponent.

    This is the main entry point that integrates with existing data modules.

    Args:
        pitcher_name: Pitcher's name
        opponent_team: Opponent team abbreviation (e.g., 'NYY')
        n_sims: Number of simulations

    Returns:
        Simulation results dict or None if data unavailable
    """
    # Import data modules
    try:
        from mlb_simulation.batter_data import get_batter, get_batter_k_rate, get_batter_bb_rate, get_league_k_rate
        from mlb_simulation.lineup_fetcher import get_opponent_lineup
        import sys
        sys.path.insert(0, str(__file__).replace('/mlb_simulation/simulator.py', ''))
        from mlb_data import get_blended_pitcher
    except ImportError as e:
        logger.error(f"Failed to import data modules: {e}")
        return None

    # Get pitcher data
    pitcher_data = get_blended_pitcher(pitcher_name)
    if not pitcher_data:
        logger.error(f"Pitcher not found: {pitcher_name}")
        return None

    pitcher = PitcherStats(
        name=pitcher_data.get('name', pitcher_name),
        k_pct=pitcher_data.get('k_pct', LEAGUE_K_PCT),
        bb_pct=pitcher_data.get('bb_pct', LEAGUE_BB_PCT),
        hand=pitcher_data.get('hand', 'R')
    )

    # Get opponent lineup
    opponent_data = get_opponent_lineup(opponent_team)
    if not opponent_data or not opponent_data.get('lineup'):
        logger.warning(f"Lineup not found for opponent of {opponent_team}, using league average")
        # Create default lineup with league average stats
        lineup = [
            BatterStats(name=f"Batter {i+1}", k_pct=LEAGUE_K_PCT, bb_pct=LEAGUE_BB_PCT)
            for i in range(9)
        ]
    else:
        lineup = []
        for batter in opponent_data['lineup'][:9]:
            name = batter.get('name', 'Unknown')
            k_pct = get_batter_k_rate(name, pitcher.hand)
            bb_pct = get_batter_bb_rate(name, pitcher.hand)
            lineup.append(BatterStats(name=name, k_pct=k_pct, bb_pct=bb_pct))

        # Pad to 9 if needed
        while len(lineup) < 9:
            lineup.append(BatterStats(name="Unknown", k_pct=LEAGUE_K_PCT, bb_pct=LEAGUE_BB_PCT))

    # Get league K rate
    league_k = get_league_k_rate() / 100

    # Run simulation
    logger.info(f"Running {n_sims} simulations: {pitcher.name} vs {opponent_team}")
    results = run_simulation(pitcher, lineup, n_sims, league_k)

    # Add lineup info to results
    results['opponent'] = opponent_team
    results['lineup'] = [{'name': b.name, 'k_pct': b.k_pct} for b in lineup]

    return results


# ============================================================================
# FULL ANALYSIS API (Compatible with Beta-Binomial interface)
# ============================================================================

def analyze_prop_simulation(
    pitcher_name: str,
    opponent_abbrev: str,
    line: float,
    over_odds: int = -110,
    under_odds: int = -110,
    venue: str = None,
    n_sims: int = DEFAULT_NUM_SIMS,
    vegas_total: float = None
) -> Optional[Dict]:
    """
    Full prop analysis using Monte Carlo simulation.

    Returns results in same format as Beta-Binomial analyze_prop() for compatibility.

    Args:
        pitcher_name: Pitcher's name
        opponent_abbrev: Opponent team abbreviation
        line: Strikeout line (e.g., 5.5)
        over_odds: Over odds (e.g., -110)
        under_odds: Under odds (e.g., -105)
        venue: Venue name (optional, for park factor)
        n_sims: Number of simulations
        vegas_total: Expected game total from Vegas (affects game script)

    Returns:
        Analysis dict compatible with API response, or None if failed
    """
    try:
        from mlb_simulation.batter_data import get_batter, get_batter_k_rate, get_batter_bb_rate, get_league_k_rate
        from mlb_simulation.lineup_fetcher import get_opponent_lineup, get_lineup
        import sys
        sys.path.insert(0, str(__file__).replace('/mlb_simulation/simulator.py', ''))
        from mlb_data import get_blended_pitcher, get_park_factor
    except ImportError as e:
        logger.error(f"Failed to import data modules: {e}")
        return None

    # Get pitcher data
    pitcher_data = get_blended_pitcher(pitcher_name)
    if not pitcher_data:
        logger.error(f"Pitcher not found: {pitcher_name}")
        return None

    pitcher = PitcherStats(
        name=pitcher_data.get('name', pitcher_name),
        k_pct=pitcher_data.get('k_pct', LEAGUE_K_PCT),
        bb_pct=pitcher_data.get('bb_pct', LEAGUE_BB_PCT),
        hand=pitcher_data.get('hand', 'R')
    )

    # Normalize opponent abbreviation (FanGraphs uses SFG, KC uses KCR, etc.)
    opponent_normalized = normalize_team_abbrev(opponent_abbrev)

    # Get opponent lineup directly using the opponent_abbrev parameter
    # This fixes the bug where traded pitchers had stale team data in cache
    opponent_lineup = get_lineup(opponent_normalized)
    lineup_confirmed = opponent_lineup and len(opponent_lineup) >= 9

    if not lineup_confirmed:
        # No confirmed lineup - return None to signal fallback to Beta-Binomial
        logger.info(f"[Simulation] No confirmed lineup for {opponent_normalized}, signaling fallback")
        return None

    # Wrap in dict for compatibility with rest of function
    opponent_data = {'lineup': opponent_lineup, 'team': opponent_normalized}

    # Build lineup with batter-specific K rates
    lineup = []
    lineup_details = []
    for batter in opponent_data['lineup'][:9]:
        name = batter.get('name', 'Unknown')
        k_pct_vs_hand = get_batter_k_rate(name, pitcher.hand)
        bb_pct = get_batter_bb_rate(name, pitcher.hand)
        lineup.append(BatterStats(name=name, k_pct=k_pct_vs_hand, bb_pct=bb_pct))

        # Get full batter data for additional details
        batter_data = get_batter(name)
        overall_k_pct = batter_data.get('k_pct', LEAGUE_K_PCT) if batter_data else LEAGUE_K_PCT
        bats = batter_data.get('bats', 'R') if batter_data else 'R'

        lineup_details.append({
            'name': name,
            'bats': bats,
            'k_pct': round(overall_k_pct, 1),           # Overall K%
            'k_pct_vs_hand': round(k_pct_vs_hand, 1),   # K% vs pitcher's hand
            'position': batter.get('position', '')
        })

    # Pad to 9 if needed
    while len(lineup) < 9:
        lineup.append(BatterStats(name="Unknown", k_pct=LEAGUE_K_PCT, bb_pct=LEAGUE_BB_PCT))
        lineup_details.append({
            'name': 'Unknown',
            'bats': 'R',
            'k_pct': LEAGUE_K_PCT,
            'k_pct_vs_hand': LEAGUE_K_PCT,
            'position': ''
        })

    # Get league K rate
    league_k = get_league_k_rate() / 100

    # Get park factor if venue provided
    park_factor = 1.0
    if venue:
        pf = get_park_factor(venue)
        if pf:
            # get_park_factor returns float directly (already normalized to 1.0 = neutral)
            if isinstance(pf, (int, float)):
                park_factor = pf
            else:
                park_factor = pf.get('so_factor', 100) / 100

    # Run simulation
    vegas_info = f", vegas total: {vegas_total}" if vegas_total else ""
    logger.info(f"[Simulation] Running {n_sims} sims: {pitcher.name} vs {opponent_abbrev} (lineup confirmed{vegas_info})")
    sim_results = run_simulation(pitcher, lineup, n_sims, league_k, vegas_total=vegas_total)

    # Calculate probability for the specific line
    k_counts = []
    for k_val, count in sim_results['distribution'].items():
        k_counts.extend([k_val] * count)

    over_count = sum(1 for k in k_counts if k > line)
    under_count = sum(1 for k in k_counts if k < line)
    push_count = sum(1 for k in k_counts if k == line)
    total = len(k_counts)

    prob_over = over_count / total if total > 0 else 0.5
    prob_under = under_count / total if total > 0 else 0.5

    # Normalize (exclude pushes)
    if prob_over + prob_under > 0:
        prob_over_norm = prob_over / (prob_over + prob_under)
        prob_under_norm = prob_under / (prob_over + prob_under)
    else:
        prob_over_norm = 0.5
        prob_under_norm = 0.5

    # Calculate EV
    def odds_to_decimal(odds: int) -> float:
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))

    over_decimal = odds_to_decimal(over_odds)
    under_decimal = odds_to_decimal(under_odds)

    ev_over = (prob_over_norm * (over_decimal - 1)) - (1 - prob_over_norm)
    ev_under = (prob_under_norm * (under_decimal - 1)) - (1 - prob_under_norm)

    # Calculate edge (vs implied probability)
    implied_over = 1 / over_decimal
    implied_under = 1 / under_decimal

    edge_over = prob_over_norm - implied_over
    edge_under = prob_under_norm - implied_under

    # Determine recommendation
    if ev_over > ev_under and ev_over > 0.02:
        recommended_side = 'over'
        edge = edge_over
        ev = ev_over
    elif ev_under > ev_over and ev_under > 0.02:
        recommended_side = 'under'
        edge = edge_under
        ev = ev_under
    else:
        recommended_side = 'pass'
        edge = 0
        ev = 0

    # Kelly criterion
    def kelly(prob: float, decimal_odds: float) -> float:
        if decimal_odds <= 1:
            return 0
        q = 1 - prob
        b = decimal_odds - 1
        kelly_pct = (prob * b - q) / b
        return max(0, min(0.25, kelly_pct))  # Cap at 25%

    kelly_over = kelly(prob_over_norm, over_decimal)
    kelly_under = kelly(prob_under_norm, under_decimal)

    # Confidence grade based on edge
    if abs(edge) >= 0.08:
        confidence_grade = 'A'
    elif abs(edge) >= 0.05:
        confidence_grade = 'B'
    elif abs(edge) >= 0.02:
        confidence_grade = 'C'
    else:
        confidence_grade = 'D'

    # Confidence score
    confidence = abs(prob_over_norm - 0.5) * 2  # 0 to 1 scale

    # Build result matching Beta-Binomial format
    return {
        'pitcher': pitcher.name,
        'opponent': opponent_abbrev,
        'line': line,
        'projected_ks': sim_results['mean_k'],
        'prob_over': round(prob_over_norm, 4),
        'prob_under': round(prob_under_norm, 4),
        'ev_over': round(ev_over, 4),
        'ev_under': round(ev_under, 4),
        'edge_over': round(edge_over, 4),
        'edge_under': round(edge_under, 4),
        'recommended_side': recommended_side,
        'edge': round(edge, 4),
        'ev': round(ev, 4),
        'confidence': round(confidence, 4),
        'confidence_grade': confidence_grade,
        'kelly_over': round(kelly_over, 4),
        'kelly_under': round(kelly_under, 4),

        # Simulation-specific data
        'model_type': 'simulation',
        'n_sims': n_sims,
        'lineup_confirmed': True,
        'sim_stats': {
            'mean_k': sim_results['mean_k'],
            'median_k': sim_results['median_k'],
            'std_k': sim_results['std_k'],
            'min_k': sim_results['min_k'],
            'max_k': sim_results['max_k'],
            'avg_ip': sim_results['avg_ip'],
            'avg_pitches': sim_results['avg_pitches'],
            'avg_bf': sim_results['avg_bf'],
            'cumulative_probs': sim_results.get('cumulative_probs', {}),  # P(1+ K), P(2+ K), ..., P(12+ K)
        },
        'over_probs': sim_results['over_probs'],
        'lineup': lineup_details,

        # Pitcher stats for compatibility
        'pitcher_stats': {
            'name': pitcher.name,
            'team': pitcher_data.get('team', ''),
            'hand': pitcher.hand,
            'k_pct': round(pitcher.k_pct, 1),
            'bb_pct': round(pitcher.bb_pct, 1),
            'swstr_pct': round(pitcher_data.get('swstr_pct', 0), 1) if pitcher_data.get('swstr_pct') else None,
            'csw_pct': round(pitcher_data.get('csw_pct', 0), 1) if pitcher_data.get('csw_pct') else None,
            'ip': pitcher_data.get('ip', 0),
            'avg_pitches': round(sim_results['avg_pitches'], 0),  # From simulation
            'season_k_per_start': None,  # Not tracked in simulation
        },

        # Matchup factors (compatible with Beta-Binomial format)
        'matchup_factors': {
            'opp_k_pct': round(sum(b.k_pct for b in lineup) / 9, 1),  # Lineup avg K%
            'opp_k_vs_hand': round(sum(b.k_pct for b in lineup) / 9, 1),  # Same (already vs hand)
            'platoon_adj': 0,  # Not used - individual batter rates already account for this
            'park_adj': round(park_factor - 1.0, 3),  # Convert factor to adjustment
            'park_factor': park_factor,
            'lineup_avg_k': round(sum(b.k_pct for b in lineup) / 9, 1),
        },

        # Summary
        'summary': f"Monte Carlo simulation ({n_sims} sims) projects {sim_results['mean_k']} Ks. "
                   f"P(Over {line}) = {prob_over_norm:.1%}. "
                   f"Lineup avg K%: {sum(b.k_pct for b in lineup) / 9:.1f}%.",

        # Breakdown for display
        'breakdown': {
            'method': 'Monte Carlo Simulation',
            'simulations': n_sims,
            'projected_ks': sim_results['mean_k'],
            'std_dev': sim_results['std_k'],
            'prob_over': f"{prob_over_norm:.1%}",
            'prob_under': f"{prob_under_norm:.1%}",
        },

        # Placeholder for compatibility
        'last_5_starts': [],
    }


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    import sys

    # Test with sample data
    print("=== MLB Strikeout Simulation Engine Test ===\n")

    # Create sample pitcher
    pitcher = PitcherStats(
        name="Test Pitcher",
        k_pct=25.0,
        bb_pct=6.0,
        hand='R'
    )

    # Create sample lineup (varying K rates)
    lineup = [
        BatterStats("Leadoff", k_pct=18.0, bb_pct=10.0),
        BatterStats("2-hole", k_pct=15.0, bb_pct=12.0),
        BatterStats("3-hole", k_pct=22.0, bb_pct=8.0),
        BatterStats("Cleanup", k_pct=28.0, bb_pct=9.0),
        BatterStats("5-hole", k_pct=25.0, bb_pct=7.0),
        BatterStats("6-hole", k_pct=24.0, bb_pct=6.0),
        BatterStats("7-hole", k_pct=26.0, bb_pct=5.0),
        BatterStats("8-hole", k_pct=30.0, bb_pct=5.0),
        BatterStats("9-hole", k_pct=22.0, bb_pct=6.0),
    ]

    print(f"Pitcher: {pitcher.name} (K%: {pitcher.k_pct}%, BB%: {pitcher.bb_pct}%)")
    print(f"Lineup avg K%: {sum(b.k_pct for b in lineup)/9:.1f}%")
    print()

    # Run simulation
    results = run_simulation(pitcher, lineup, n_sims=5000)

    print(f"Simulations: {results['n_sims']}")
    print(f"Mean Ks: {results['mean_k']}")
    print(f"Median Ks: {results['median_k']}")
    print(f"Std Dev: {results['std_k']}")
    print(f"Range: {results['min_k']} - {results['max_k']}")
    print()

    print("Over probabilities:")
    for line, prob in results['over_probs'].items():
        print(f"  Over {line}: {prob:.1%}")
    print()

    print(f"Avg IP: {results['avg_ip']}")
    print(f"Avg Pitches: {results['avg_pitches']}")
    print(f"Avg BF: {results['avg_bf']}")
    print()

    print("Distribution (top 10):")
    dist = sorted(results['distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for k, count in dist:
        pct = count / results['n_sims'] * 100
        print(f"  {k} Ks: {pct:.1f}%")

    # Test with real data if available
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        print("\n" + "="*50)
        print("=== Live Data Test ===\n")

        result = simulate_strikeouts("Logan Webb", "SF", n_sims=3000)
        if result:
            print(f"Pitcher: {result['pitcher']}")
            print(f"Opponent: {result['opponent']}")
            print(f"Mean Ks: {result['mean_k']}")
            print(f"Over 5.5: {result['over_probs'].get(5.5, 0):.1%}")
            print(f"Over 6.5: {result['over_probs'].get(6.5, 0):.1%}")
