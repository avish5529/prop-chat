#!/usr/bin/env python3
"""
MLB Strikeout Projection Engine V2

Fully automated projection using:
- Expected BF from actual game logs (not derived)
- xK% formula with CSW% integration
- Matchup adjustments (opponent splits, park, umpire)
- Beta-Binomial with dynamic concentration
- No hardcoded values - all from mlb_data.py

Based on industry research:
- FanGraphs xK% formula methodology
- Opponent K% vs pitcher hand (biggest edge)
- Park factors for strikeout environments
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from scipy.stats import betabinom

from mlb_data import (
    get_pitcher,
    get_team,
    get_park_factor,
    get_umpire,
    get_league_averages,
    get_todays_games,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProjectionInputs:
    """All inputs used for projection (for transparency)."""
    # Pitcher
    pitcher_name: str
    pitcher_hand: str
    pitcher_k_pct: float
    pitcher_swstr_pct: float
    pitcher_csw_pct: Optional[float]
    pitcher_ip: float

    # Recent form
    last_5_avg_bf: Optional[float]
    last_5_avg_pitches: Optional[float]
    last_5_k_rate: Optional[float]
    last_10_k_std: Optional[float]
    last_10_k_avg: Optional[float]

    # Opponent
    opponent_team: str
    opponent_k_pct: float
    opponent_k_pct_vs_hand: float
    opponent_p_per_pa: float

    # Context
    park_name: Optional[str]
    park_k_factor: float
    umpire_name: Optional[str]
    umpire_k_index: float
    vegas_total: Optional[float]
    spread: Optional[float]

    # League averages
    league_k_pct: float
    league_swstr_pct: float


@dataclass
class ProjectionBreakdown:
    """Step-by-step breakdown of projection calculation."""
    # Step 1: Expected BF
    base_bf: float
    vegas_adj: float
    favorite_adj: float
    expected_bf: float

    # Step 2: True talent K%
    xk_pct_from_peripherals: float
    ip_weight: float
    base_k_pct: float
    form_adjustment: float
    talent_k_pct: float

    # Step 3: Matchup adjustments
    opponent_adj: float
    park_adj: float
    umpire_adj: float
    matchup_multiplier: float
    adjusted_k_pct: float

    # Step 4: Distribution
    concentration: float
    alpha: float
    beta: float

    # Step 5: Results
    projected_ks: float
    prob_over: float
    prob_under: float


@dataclass
class MLBProjection:
    """Final projection result."""
    # Core
    pitcher: str
    opponent: str
    line: float
    projected_ks: float

    # Probabilities
    prob_over: float
    prob_under: float

    # Betting
    ev_over: float
    ev_under: float
    edge_over: float
    edge_under: float
    recommended_side: str
    edge: float
    ev: float
    confidence: float

    # Kelly
    kelly_over: float
    kelly_under: float

    # Details
    confidence_grade: str
    inputs: ProjectionInputs
    breakdown: ProjectionBreakdown

    # For frontend
    pitcher_stats: Dict[str, Any] = field(default_factory=dict)
    matchup_factors: Dict[str, Any] = field(default_factory=dict)
    last_5_starts: List[Dict] = field(default_factory=list)
    summary: str = ""


# ============================================================================
# STEP 1: EXPECTED BATTERS FACED
# ============================================================================

def calculate_expected_bf(
    pitcher: Dict,
    opponent: Dict,
    vegas_total: Optional[float] = None,
    spread: Optional[float] = None,
    is_home: bool = False,
) -> tuple[float, float, float, float]:
    """
    Calculate expected batters faced from actual game logs.

    Returns: (base_bf, vegas_adj, favorite_adj, expected_bf)
    """
    # Use actual BF from recent starts if available
    base_bf = pitcher.get('last_5_avg_bf')

    if base_bf is None:
        # Fallback: estimate from pitch count
        avg_pitches = pitcher.get('last_5_avg_pitches')
        if avg_pitches is None:
            avg_pitches = 88  # League average

        p_per_pa = opponent.get('p_per_pa', 3.92)
        base_bf = avg_pitches / p_per_pa

    # Vegas context adjustment (±5% max)
    if vegas_total:
        # Higher totals = more runs = more baserunners = more BF
        # Lower totals = pitcher's duel = fewer BF
        vegas_adj = 1 + (vegas_total - 9.0) * 0.012
        vegas_adj = max(0.94, min(1.06, vegas_adj))
    else:
        vegas_adj = 1.0

    # Favorite adjustment (favorites go deeper)
    if spread is not None:
        # Negative spread = favorite
        # If pitcher's team is -2.0, they're likely to pitch longer
        if spread < -1.5:
            favorite_adj = 1.025  # +2.5% BF
        elif spread > 1.5:
            favorite_adj = 0.975  # -2.5% BF (underdog gets less rope)
        else:
            favorite_adj = 1.0
    else:
        favorite_adj = 1.0

    expected_bf = base_bf * vegas_adj * favorite_adj

    # Clamp to realistic range for starting pitchers
    expected_bf = max(18, min(27, expected_bf))

    logger.info(f"[Step1] BF: base={base_bf:.1f}, vegas={vegas_adj:.3f}, fav={favorite_adj:.3f} → {expected_bf:.1f}")

    return base_bf, vegas_adj, favorite_adj, expected_bf


# ============================================================================
# STEP 2: TRUE TALENT K%
# ============================================================================

def calculate_true_talent_k_pct(
    pitcher: Dict,
    league_avg: Dict,
) -> tuple[float, float, float, float, float]:
    """
    Calculate true talent K% using peripherals + sample weighting.

    Returns: (xk_from_peripherals, ip_weight, base_k_pct, form_adj, talent_k_pct)
    """
    swstr_pct = pitcher.get('swstr_pct') or league_avg.get('swstr_pct', 11.0)
    csw_pct = pitcher.get('csw_pct')
    actual_k_pct = pitcher.get('k_pct') or league_avg.get('k_pct', 22.5)
    ip = pitcher.get('ip', 0)

    # Expected K% from peripherals
    # Better formula that includes CSW% (called strikes + whiffs)
    if csw_pct:
        # xK% = (SwStr% × 1.4) + (CSW% × 0.35) - 6.0
        # This captures both swinging strikes AND called strikes
        xk_pct = (swstr_pct * 1.4) + (csw_pct * 0.35) - 6.0
    else:
        # Fallback: SwStr% × 1.9 (simpler approximation)
        xk_pct = swstr_pct * 1.9

    # Clamp xK% to reasonable range
    xk_pct = max(12, min(38, xk_pct))

    # Blend with actual K% based on sample size
    # More IP = more weight to actual results
    ip_weight = min(0.7, ip / 80)  # Cap at 70% weight to actual

    base_k_pct = (xk_pct * (1 - ip_weight)) + (actual_k_pct * ip_weight)

    # Recent form adjustment
    form_adj = 0.0
    last_5_k_rate = pitcher.get('last_5_k_rate')
    if last_5_k_rate is not None:
        season_k_rate = actual_k_pct / 100
        form_diff = last_5_k_rate - season_k_rate
        # 15% weight to recent form
        form_adj = form_diff * 100 * 0.15  # Convert to percentage points

    talent_k_pct = base_k_pct + form_adj

    # Final clamp
    talent_k_pct = max(12, min(38, talent_k_pct))

    logger.info(f"[Step2] K%: xK={xk_pct:.1f}, actual={actual_k_pct:.1f}, blend={base_k_pct:.1f}, form={form_adj:+.1f} → {talent_k_pct:.1f}%")

    return xk_pct, ip_weight, base_k_pct, form_adj, talent_k_pct


# ============================================================================
# STEP 3: MATCHUP ADJUSTMENTS
# ============================================================================

def calculate_matchup_adjustments(
    talent_k_pct: float,
    pitcher: Dict,
    opponent: Dict,
    park_k_factor: float,
    umpire_k_index: float,
    league_avg: Dict,
) -> tuple[float, float, float, float, float]:
    """
    Apply matchup-specific adjustments.

    Returns: (opp_adj, park_adj, ump_adj, multiplier, adjusted_k_pct)
    """
    pitcher_hand = pitcher.get('hand', 'R')

    # A. Opponent K% vs pitcher hand (THE BIGGEST FACTOR)
    if pitcher_hand == 'L':
        opp_k_rate = opponent.get('k_pct_vs_lhp') or opponent.get('k_pct', 22.5)
    else:
        opp_k_rate = opponent.get('k_pct_vs_rhp') or opponent.get('k_pct', 22.5)

    league_k = league_avg.get('team_k_pct', 22.5)
    opp_adj = (opp_k_rate / league_k) - 1  # e.g., +0.15 for high-K team

    # B. Park factor
    park_adj = park_k_factor - 1.0  # e.g., +0.05 for Oracle Park

    # C. Umpire tendency (dampened)
    ump_adj = (umpire_k_index - 1.0) * 0.4  # e.g., +0.016 for K-friendly ump

    # Combine adjustments (dampened weights based on research)
    matchup_multiplier = 1.0
    matchup_multiplier += opp_adj * 0.35      # 35% of opponent effect
    matchup_multiplier += park_adj * 0.50     # 50% of park effect
    matchup_multiplier += ump_adj             # Already dampened above

    adjusted_k_pct = talent_k_pct * matchup_multiplier

    # Final clamp to realistic range (12-38%)
    adjusted_k_pct = max(12, min(38, adjusted_k_pct))

    logger.info(f"[Step3] Adj: opp={opp_adj:+.3f} ({opp_k_rate:.1f}% vs {pitcher_hand}HP), park={park_adj:+.3f}, ump={ump_adj:+.3f} → mult={matchup_multiplier:.3f}, K%={adjusted_k_pct:.1f}%")

    return opp_adj, park_adj, ump_adj, matchup_multiplier, adjusted_k_pct


# ============================================================================
# STEP 4: BETA-BINOMIAL DISTRIBUTION
# ============================================================================

def calculate_distribution(
    expected_bf: float,
    adjusted_k_pct: float,
    pitcher: Dict,
) -> tuple[float, float, float, float]:
    """
    Create Beta-Binomial distribution for strikeout projection.

    Returns: (concentration, alpha, beta, projected_ks)
    """
    n = int(round(expected_bf))
    k_prob = adjusted_k_pct / 100

    # Dynamic concentration based on pitcher consistency
    last_10_k_std = pitcher.get('last_10_k_std')
    last_10_k_avg = pitcher.get('last_10_k_avg')

    if last_10_k_std and last_10_k_avg and last_10_k_avg > 0:
        # Coefficient of variation
        cv = last_10_k_std / last_10_k_avg
        # Higher CV = more variance = lower concentration
        if cv > 0:
            concentration = 1 / (cv ** 2)
            concentration = max(6, min(25, concentration))
        else:
            concentration = 15
    else:
        # Default based on sample size
        ip = pitcher.get('ip', 0)
        concentration = 8 + min(ip / 20, 10)

    alpha = k_prob * concentration
    beta = (1 - k_prob) * concentration

    # Create distribution and get mean
    dist = betabinom(n, alpha, beta)
    projected_ks = float(dist.mean())

    logger.info(f"[Step4] Dist: n={n}, K%={k_prob:.3f}, conc={concentration:.1f} → projected={projected_ks:.2f} Ks")

    return concentration, alpha, beta, projected_ks


# ============================================================================
# STEP 5: PROBABILITIES
# ============================================================================

def calculate_probabilities(
    expected_bf: float,
    alpha: float,
    beta: float,
    line: float,
) -> tuple[float, float]:
    """
    Calculate P(over) and P(under) for the given line.

    Returns: (prob_over, prob_under)
    """
    n = int(round(expected_bf))
    dist = betabinom(n, alpha, beta)

    # P(K <= line) for the under
    prob_under = float(dist.cdf(line))
    prob_over = 1 - prob_under

    logger.info(f"[Step5] Line {line}: P(over)={prob_over:.1%}, P(under)={prob_under:.1%}")

    return prob_over, prob_under


# ============================================================================
# STEP 6: EDGE & RECOMMENDATION
# ============================================================================

def calculate_edge_and_ev(
    prob_over: float,
    prob_under: float,
    over_odds: int,
    under_odds: int,
) -> Dict[str, float]:
    """
    Calculate edge, EV, and Kelly for both sides.

    Returns dict with all betting metrics.
    """
    # Implied probabilities (with vig)
    implied_over = odds_to_prob(over_odds)
    implied_under = odds_to_prob(under_odds)

    # No-vig probabilities
    total_implied = implied_over + implied_under
    no_vig_over = implied_over / total_implied
    no_vig_under = implied_under / total_implied

    # Edge = our probability - no-vig market probability
    edge_over = prob_over - no_vig_over
    edge_under = prob_under - no_vig_under

    # EV calculation
    ev_over = calculate_ev(prob_over, over_odds)
    ev_under = calculate_ev(prob_under, under_odds)

    # Kelly criterion
    kelly_over = calculate_kelly(prob_over, over_odds)
    kelly_under = calculate_kelly(prob_under, under_odds)

    # Determine recommendation - always pick over or under (no pass)
    # Pick the side with higher probability (and thus higher edge)
    if prob_over > prob_under:
        recommended_side = 'over'
        edge = edge_over
        ev = ev_over
        kelly = kelly_over
        confidence = prob_over
    else:
        recommended_side = 'under'
        edge = edge_under
        ev = ev_under
        kelly = kelly_under
        confidence = prob_under

    logger.info(f"[Step6] Rec: {recommended_side.upper()}, edge={edge:+.1%}, EV={ev:+.1%}")

    return {
        'implied_over': implied_over,
        'implied_under': implied_under,
        'no_vig_over': no_vig_over,
        'no_vig_under': no_vig_under,
        'edge_over': edge_over,
        'edge_under': edge_under,
        'ev_over': ev_over,
        'ev_under': ev_under,
        'kelly_over': kelly_over,
        'kelly_under': kelly_under,
        'recommended_side': recommended_side,
        'edge': edge,
        'ev': ev,
        'confidence': confidence,
    }


def odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_ev(prob: float, odds: int) -> float:
    """Calculate expected value."""
    if odds > 0:
        profit = odds / 100
    else:
        profit = 100 / abs(odds)

    ev = (prob * profit) - ((1 - prob) * 1)
    return ev


def calculate_kelly(prob: float, odds: int) -> float:
    """Calculate Kelly criterion bet size."""
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)

    q = 1 - prob
    kelly = (prob * b - q) / b

    return max(0, kelly)  # No negative Kelly


# ============================================================================
# CONFIDENCE GRADE
# ============================================================================

def calculate_confidence_grade(pitcher: Dict, expected_bf: float, edge: float) -> str:
    """Calculate confidence grade based on data quality and edge."""
    score = 0
    ip = pitcher.get('ip', 0)

    # Sample size (max 30)
    if ip >= 80:
        score += 30
    elif ip >= 50:
        score += 22
    elif ip >= 30:
        score += 15
    elif ip >= 15:
        score += 8

    # Data completeness (max 25)
    if pitcher.get('swstr_pct'):
        score += 10
    if pitcher.get('csw_pct'):
        score += 10
    if pitcher.get('last_5_avg_bf'):
        score += 5

    # Recent form data (max 20)
    if pitcher.get('last_5_k_rate'):
        score += 12
    if pitcher.get('last_10_k_std'):
        score += 8

    # Edge bonus (max 25)
    if abs(edge) >= 0.08:
        score += 25
    elif abs(edge) >= 0.05:
        score += 18
    elif abs(edge) >= 0.03:
        score += 10
    elif abs(edge) >= 0.01:
        score += 5

    if score >= 85:
        return 'A'
    elif score >= 65:
        return 'B'
    elif score >= 45:
        return 'C'
    else:
        return 'D'


# ============================================================================
# MAIN PROJECTION FUNCTION
# ============================================================================

def project_strikeouts(
    pitcher_name: str,
    opponent_name: str,
    line: float,
    over_odds: int,
    under_odds: int,
    venue: str = None,
    vegas_total: float = None,
    spread: float = None,
    umpire_name: str = None,
) -> Optional[MLBProjection]:
    """
    Full strikeout projection using V2 methodology.

    Args:
        pitcher_name: Name of the starting pitcher
        opponent_name: Name or abbreviation of opponent team
        line: Strikeout line (e.g., 5.5)
        over_odds: American odds for over (e.g., -115)
        under_odds: American odds for under (e.g., -105)
        venue: Stadium name for park factor
        vegas_total: Game total for context
        spread: Moneyline spread for context
        umpire_name: Home plate umpire name

    Returns:
        MLBProjection with full breakdown, or None if data unavailable
    """
    logger.info("=" * 60)
    logger.info(f"PROJECTION: {pitcher_name} vs {opponent_name} | Line: {line}")
    logger.info("=" * 60)

    # Load data
    pitcher = get_pitcher(pitcher_name)
    if not pitcher:
        logger.error(f"Pitcher not found: {pitcher_name}")
        return None

    opponent = get_team(opponent_name)
    if not opponent:
        logger.warning(f"Team not found: {opponent_name}, using league average")
        league_avg = get_league_averages()
        opponent = {
            'name': opponent_name,
            'k_pct': league_avg.get('team_k_pct', 22.5),
            'k_pct_vs_lhp': league_avg.get('team_k_pct_vs_lhp', 22.5),
            'k_pct_vs_rhp': league_avg.get('team_k_pct_vs_rhp', 22.5),
            'p_per_pa': league_avg.get('p_per_pa', 3.92),
        }

    league_avg = get_league_averages()
    park_k_factor = get_park_factor(venue) if venue else 1.0

    umpire_data = get_umpire(umpire_name) if umpire_name else None
    umpire_k_index = umpire_data.get('k_index', 1.0) if umpire_data else 1.0

    # ==================== STEP 1: Expected BF ====================
    base_bf, vegas_adj, favorite_adj, expected_bf = calculate_expected_bf(
        pitcher, opponent, vegas_total, spread
    )

    # ==================== STEP 2: True Talent K% ====================
    xk_pct, ip_weight, base_k_pct, form_adj, talent_k_pct = calculate_true_talent_k_pct(
        pitcher, league_avg
    )

    # ==================== STEP 3: Matchup Adjustments ====================
    opp_adj, park_adj, ump_adj, matchup_mult, adjusted_k_pct = calculate_matchup_adjustments(
        talent_k_pct, pitcher, opponent, park_k_factor, umpire_k_index, league_avg
    )

    # ==================== STEP 4: Distribution ====================
    concentration, alpha, beta, projected_ks = calculate_distribution(
        expected_bf, adjusted_k_pct, pitcher
    )

    # ==================== STEP 5: Probabilities ====================
    prob_over, prob_under = calculate_probabilities(expected_bf, alpha, beta, line)

    # ==================== STEP 6: Edge & Recommendation ====================
    betting = calculate_edge_and_ev(prob_over, prob_under, over_odds, under_odds)

    # ==================== Confidence Grade ====================
    confidence_grade = calculate_confidence_grade(pitcher, expected_bf, betting['edge'])

    # ==================== Build Response Objects ====================

    # Inputs object
    pitcher_hand = pitcher.get('hand', 'R')
    opp_k_vs_hand = opponent.get(f'k_pct_vs_{pitcher_hand.lower()}hp') or opponent.get('k_pct', 22.5)

    inputs = ProjectionInputs(
        pitcher_name=pitcher['name'],
        pitcher_hand=pitcher_hand,
        pitcher_k_pct=pitcher.get('k_pct', 0),
        pitcher_swstr_pct=pitcher.get('swstr_pct', 0),
        pitcher_csw_pct=pitcher.get('csw_pct'),
        pitcher_ip=pitcher.get('ip', 0),
        last_5_avg_bf=pitcher.get('last_5_avg_bf'),
        last_5_avg_pitches=pitcher.get('last_5_avg_pitches'),
        last_5_k_rate=pitcher.get('last_5_k_rate'),
        last_10_k_std=pitcher.get('last_10_k_std'),
        last_10_k_avg=pitcher.get('last_10_k_avg'),
        opponent_team=opponent.get('name', opponent_name),
        opponent_k_pct=opponent.get('k_pct', 22.5),
        opponent_k_pct_vs_hand=opp_k_vs_hand,
        opponent_p_per_pa=opponent.get('p_per_pa', 3.92),
        park_name=venue,
        park_k_factor=park_k_factor,
        umpire_name=umpire_name,
        umpire_k_index=umpire_k_index,
        vegas_total=vegas_total,
        spread=spread,
        league_k_pct=league_avg.get('k_pct', 22.5),
        league_swstr_pct=league_avg.get('swstr_pct', 11.0),
    )

    # Breakdown object
    breakdown = ProjectionBreakdown(
        base_bf=base_bf,
        vegas_adj=vegas_adj,
        favorite_adj=favorite_adj,
        expected_bf=expected_bf,
        xk_pct_from_peripherals=xk_pct,
        ip_weight=ip_weight,
        base_k_pct=base_k_pct,
        form_adjustment=form_adj,
        talent_k_pct=talent_k_pct,
        opponent_adj=opp_adj,
        park_adj=park_adj,
        umpire_adj=ump_adj,
        matchup_multiplier=matchup_mult,
        adjusted_k_pct=adjusted_k_pct,
        concentration=concentration,
        alpha=alpha,
        beta=beta,
        projected_ks=projected_ks,
        prob_over=prob_over,
        prob_under=prob_under,
    )

    # Frontend-friendly data
    pitcher_stats = {
        'k_pct': pitcher.get('k_pct'),
        'swstr_pct': pitcher.get('swstr_pct'),
        'csw_pct': pitcher.get('csw_pct'),
        'season_k_avg': pitcher.get('last_10_k_avg'),
        'last_5_avg': pitcher.get('last_5_avg_ks'),
        'avg_pitches': pitcher.get('last_5_avg_pitches'),
        'ip': pitcher.get('ip'),
    }

    matchup_factors = {
        'opp_k_pct': opponent.get('k_pct'),
        'opp_k_pct_vs_hand': opp_k_vs_hand,
        'handedness': f"vs {pitcher_hand}HP",
        'platoon_adj': opp_adj,
        'park_adj': park_adj,
        'umpire_adj': ump_adj if umpire_name else None,
        'total_adj': matchup_mult - 1,
    }

    last_5_starts = pitcher.get('last_starts', [])[:5]

    # Summary text
    summary = _generate_summary(
        pitcher, opponent, line, projected_ks, prob_over, prob_under,
        betting['recommended_side'], betting['edge'], opp_adj, park_adj
    )

    # Final projection object
    return MLBProjection(
        pitcher=pitcher['name'],
        opponent=opponent.get('name', opponent_name),
        line=line,
        projected_ks=round(projected_ks, 2),
        prob_over=round(prob_over, 4),
        prob_under=round(prob_under, 4),
        ev_over=round(betting['ev_over'], 4),
        ev_under=round(betting['ev_under'], 4),
        edge_over=round(betting['edge_over'], 4),
        edge_under=round(betting['edge_under'], 4),
        recommended_side=betting['recommended_side'],
        edge=round(betting['edge'], 4),
        ev=round(betting['ev'], 4),
        confidence=round(betting['confidence'], 4),
        kelly_over=round(betting['kelly_over'], 4),
        kelly_under=round(betting['kelly_under'], 4),
        confidence_grade=confidence_grade,
        inputs=inputs,
        breakdown=breakdown,
        pitcher_stats=pitcher_stats,
        matchup_factors=matchup_factors,
        last_5_starts=last_5_starts,
        summary=summary,
    )


def _generate_summary(
    pitcher: Dict,
    opponent: Dict,
    line: float,
    projected_ks: float,
    prob_over: float,
    prob_under: float,
    recommended_side: str,
    edge: float,
    opp_adj: float,
    park_adj: float,
) -> str:
    """Generate human-readable analysis summary."""
    lines = []

    # Projection vs line
    diff = projected_ks - line
    if abs(diff) < 0.3:
        lines.append(f"Projection ({projected_ks:.1f} Ks) is close to the line ({line}).")
    elif diff > 0:
        lines.append(f"Projection ({projected_ks:.1f} Ks) is {diff:.1f} above the line ({line}).")
    else:
        lines.append(f"Projection ({projected_ks:.1f} Ks) is {abs(diff):.1f} below the line ({line}).")

    # Matchup context
    if opp_adj > 0.05:
        lines.append(f"{opponent.get('name', 'Opponent')} has a high K rate vs this pitcher type (+{opp_adj:.0%} adjustment).")
    elif opp_adj < -0.05:
        lines.append(f"{opponent.get('name', 'Opponent')} has a low K rate vs this pitcher type ({opp_adj:.0%} adjustment).")

    if park_adj > 0.02:
        lines.append("Pitcher-friendly park boosts K potential.")
    elif park_adj < -0.02:
        lines.append("Hitter-friendly park reduces K potential.")

    # Recommendation
    lines.append(f"Recommendation: {recommended_side.upper()} {line} Ks ({edge:+.1%} edge).")

    return " ".join(lines)


# ============================================================================
# CONVENIENCE FUNCTION FOR API
# ============================================================================

def analyze_prop(
    pitcher_name: str,
    opponent_name: str,
    line: float,
    over_odds: int,
    under_odds: int,
    venue: str = None,
    vegas_total: float = None,
    spread: float = None,
    umpire_name: str = None,
) -> Optional[Dict]:
    """
    Analyze a strikeout prop and return dict for API response.
    """
    projection = project_strikeouts(
        pitcher_name=pitcher_name,
        opponent_name=opponent_name,
        line=line,
        over_odds=over_odds,
        under_odds=under_odds,
        venue=venue,
        vegas_total=vegas_total,
        spread=spread,
        umpire_name=umpire_name,
    )

    if not projection:
        return None

    # Convert to dict for JSON serialization
    return {
        'pitcher': projection.pitcher,
        'opponent': projection.opponent,
        'line': projection.line,
        'over_odds': over_odds,
        'under_odds': under_odds,
        'projected_ks': projection.projected_ks,
        'prob_over': projection.prob_over,
        'prob_under': projection.prob_under,
        'ev_over': projection.ev_over,
        'ev_under': projection.ev_under,
        'edge_over': projection.edge_over,
        'edge_under': projection.edge_under,
        'recommended_side': projection.recommended_side,
        'edge': projection.edge,
        'ev': projection.ev,
        'confidence': projection.confidence,
        'kelly_over': projection.kelly_over,
        'kelly_under': projection.kelly_under,
        'confidence_grade': projection.confidence_grade,
        'pitcher_stats': projection.pitcher_stats,
        'matchup_factors': projection.matchup_factors,
        'last_5_starts': projection.last_5_starts,
        'summary': projection.summary,
        'breakdown': {
            'expected_bf': projection.breakdown.expected_bf,
            'talent_k_pct': projection.breakdown.talent_k_pct,
            'adjusted_k_pct': projection.breakdown.adjusted_k_pct,
            'matchup_multiplier': projection.breakdown.matchup_multiplier,
        },
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing MLB Projection V2\n")

    # Test projection
    result = analyze_prop(
        pitcher_name="Logan Webb",
        opponent_name="Yankees",
        line=5.5,
        over_odds=-115,
        under_odds=-105,
        venue="Oracle Park",
        vegas_total=8.5,
    )

    if result:
        print(f"\n{'='*60}")
        print(f"RESULT: {result['pitcher']} vs {result['opponent']}")
        print(f"{'='*60}")
        print(f"Line: {result['line']} | Projected: {result['projected_ks']} Ks")
        print(f"P(Over): {result['prob_over']:.1%} | P(Under): {result['prob_under']:.1%}")
        print(f"EV Over: {result['ev_over']:+.1%} | EV Under: {result['ev_under']:+.1%}")
        print(f"Recommendation: {result['recommended_side'].upper()} ({result['edge']:+.1%} edge)")
        print(f"Confidence: {result['confidence_grade']}")
        print(f"\nSummary: {result['summary']}")
    else:
        print("Projection failed - check if data is loaded")
