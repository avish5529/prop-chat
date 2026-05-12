#!/usr/bin/env python3
"""
MLB Strikeout Projection Engine
Industry-aligned model using Beta-Binomial distribution.

Completely separate from NBA projection logic.
"""
import json
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import betabinom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache file paths (from mlb_fetch_data.py)
PITCHER_CACHE = "mlb_pitcher_stats_cache.json"
TEAM_CACHE = "mlb_team_batting_cache.json"
PARK_CACHE = "mlb_park_factors_cache.json"

# League averages (2024-2025 baselines)
LEAGUE_AVG_K_PCT = 22.5
LEAGUE_AVG_SWSTR = 11.5
LEAGUE_AVG_P_PER_PA = 3.9
LEAGUE_AVG_PITCH_COUNT = 90


@dataclass
class MLBProjection:
    """Result of MLB strikeout projection."""
    pitcher_name: str
    opponent_team: str
    projected_ks: float
    expected_bf: float
    k_probability: float
    prob_over: float
    prob_under: float
    confidence_grade: str
    inputs: Dict[str, Any]
    adjustments: Dict[str, float]


def load_pitcher_cache() -> Dict:
    """Load pitcher stats cache."""
    try:
        with open(PITCHER_CACHE, 'r') as f:
            return json.load(f)['pitchers']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        logger.error("Pitcher cache not found. Run mlb_fetch_data.py first.")
        return {}


def load_team_cache() -> Dict:
    """Load team batting cache."""
    try:
        with open(TEAM_CACHE, 'r') as f:
            return json.load(f)['teams']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        logger.error("Team cache not found. Run mlb_fetch_data.py first.")
        return {}


def load_park_cache() -> Dict:
    """Load park factors cache."""
    try:
        with open(PARK_CACHE, 'r') as f:
            return json.load(f)['parks']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {'default': 1.0}


def get_pitcher(name: str) -> Optional[Dict]:
    """Get pitcher stats by name (case-insensitive, flexible matching)."""
    pitchers = load_pitcher_cache()
    name_lower = name.lower().strip()

    # Try exact match first
    if name_lower in pitchers:
        return pitchers[name_lower]

    # Try partial match (query in cache key)
    for key, val in pitchers.items():
        if name_lower in key:
            return val

    # Try matching by last name (handles "Cam Schlittler" -> "Cameron Schlittler")
    query_parts = name_lower.split()
    if len(query_parts) >= 2:
        query_last = query_parts[-1]
        query_first = query_parts[0]
        for key, val in pitchers.items():
            key_parts = key.split()
            if len(key_parts) >= 2:
                key_last = key_parts[-1]
                key_first = key_parts[0]
                # Match if last names match and first name starts with query first
                if key_last == query_last and key_first.startswith(query_first):
                    return val

    # Try matching just last name
    if query_parts:
        query_last = query_parts[-1]
        for key, val in pitchers.items():
            if query_last == key.split()[-1]:
                return val

    return None


def get_team(name: str) -> Optional[Dict]:
    """Get team stats by name (handles full names and abbreviations)."""
    teams = load_team_cache()

    # Team name mappings
    team_mappings = {
        'yankees': 'NYY', 'new york yankees': 'NYY',
        'red sox': 'BOS', 'boston red sox': 'BOS',
        'blue jays': 'TOR', 'toronto blue jays': 'TOR',
        'orioles': 'BAL', 'baltimore orioles': 'BAL',
        'rays': 'TBR', 'tampa bay rays': 'TBR',
        'guardians': 'CLE', 'cleveland guardians': 'CLE',
        'tigers': 'DET', 'detroit tigers': 'DET',
        'twins': 'MIN', 'minnesota twins': 'MIN',
        'white sox': 'CHW', 'chicago white sox': 'CHW',
        'royals': 'KCR', 'kansas city royals': 'KCR',
        'astros': 'HOU', 'houston astros': 'HOU',
        'mariners': 'SEA', 'seattle mariners': 'SEA',
        'rangers': 'TEX', 'texas rangers': 'TEX',
        'angels': 'LAA', 'los angeles angels': 'LAA',
        'athletics': 'OAK', 'oakland athletics': 'OAK',
        'mets': 'NYM', 'new york mets': 'NYM',
        'braves': 'ATL', 'atlanta braves': 'ATL',
        'phillies': 'PHI', 'philadelphia phillies': 'PHI',
        'marlins': 'MIA', 'miami marlins': 'MIA',
        'nationals': 'WSN', 'washington nationals': 'WSN',
        'cubs': 'CHC', 'chicago cubs': 'CHC',
        'cardinals': 'STL', 'st. louis cardinals': 'STL',
        'brewers': 'MIL', 'milwaukee brewers': 'MIL',
        'reds': 'CIN', 'cincinnati reds': 'CIN',
        'pirates': 'PIT', 'pittsburgh pirates': 'PIT',
        'dodgers': 'LAD', 'los angeles dodgers': 'LAD',
        'giants': 'SFG', 'san francisco giants': 'SFG',
        'padres': 'SDP', 'san diego padres': 'SDP',
        'diamondbacks': 'ARI', 'arizona diamondbacks': 'ARI',
        'rockies': 'COL', 'colorado rockies': 'COL',
    }

    name_lower = name.lower()

    # Check mappings
    if name_lower in team_mappings:
        abbrev = team_mappings[name_lower]
        if abbrev in teams:
            return teams[abbrev]

    # Try direct match
    if name in teams:
        return teams[name]

    # Try uppercase
    if name.upper() in teams:
        return teams[name.upper()]

    # Try partial match
    for key, val in teams.items():
        if name_lower in key.lower():
            return val

    return None


def get_park_factor(venue: str) -> float:
    """Get K factor for a venue."""
    parks = load_park_cache()
    return parks.get(venue, parks.get('default', 1.0))


def project_strikeouts(
    pitcher_name: str,
    opponent_team: str,
    venue: str = None,
    expected_pitch_count: float = None,
    line: float = None
) -> Optional[MLBProjection]:
    """
    Project strikeouts using Beta-Binomial distribution.

    Args:
        pitcher_name: Name of the pitcher
        opponent_team: Name or abbreviation of opponent
        venue: Stadium name (for park factor)
        expected_pitch_count: Override pitch count estimate
        line: Strikeout line for probability calculation

    Returns:
        MLBProjection with all projection data
    """
    # Get pitcher data
    pitcher = get_pitcher(pitcher_name)
    if not pitcher:
        logger.error(f"Pitcher not found: {pitcher_name}")
        return None

    # Get opponent data
    opponent = get_team(opponent_team)
    if not opponent:
        logger.warning(f"Team not found: {opponent_team}, using league average")
        opponent = {
            'k_pct': LEAGUE_AVG_K_PCT,
            'p_per_pa': LEAGUE_AVG_P_PER_PA,
            'k_pct_vs_lhp': LEAGUE_AVG_K_PCT,
            'k_pct_vs_rhp': LEAGUE_AVG_K_PCT,
        }

    # Get park factor
    park_factor = get_park_factor(venue) if venue else 1.0

    # ========== STEP 1: Estimate Batters Faced ==========
    if expected_pitch_count is None:
        # Use league average for now
        # TODO: Build historical pitch count data per pitcher
        expected_pitch_count = LEAGUE_AVG_PITCH_COUNT

    opp_p_per_pa = opponent.get('p_per_pa', LEAGUE_AVG_P_PER_PA)
    expected_bf = expected_pitch_count / opp_p_per_pa

    # ========== STEP 2: Calculate K Probability ==========
    # Get pitcher metrics
    swstr = pitcher.get('swstr_pct') or LEAGUE_AVG_SWSTR
    k_pct = pitcher.get('k_pct')
    ip = pitcher.get('ip', 0)

    # SwStr% × 2 ≈ K% (research: 0.87 correlation)
    k_pct_from_swstr = swstr * 2 / 100

    if k_pct and ip > 20:
        # Blend: weight toward actual K% as sample grows
        weight = min(0.7, ip / 100)  # Max 70% weight to actual
        base_k_prob = (k_pct_from_swstr * (1 - weight)) + (k_pct / 100 * weight)
    else:
        base_k_prob = k_pct_from_swstr

    # ========== STEP 3: Adjustments ==========
    # Opponent K% vs pitcher hand
    pitcher_hand = pitcher.get('hand', 'R')
    if pitcher_hand == 'L':
        opp_k_pct = opponent.get('k_pct_vs_lhp') or opponent.get('k_pct', LEAGUE_AVG_K_PCT)
    else:
        opp_k_pct = opponent.get('k_pct_vs_rhp') or opponent.get('k_pct', LEAGUE_AVG_K_PCT)

    opp_adjustment = (opp_k_pct / LEAGUE_AVG_K_PCT) - 1

    # Apply adjustments (multiplicative, dampened)
    adjusted_k_prob = base_k_prob * (1 + opp_adjustment * 0.3) * park_factor

    # Clamp to reasonable range (10% to 40% K rate)
    adjusted_k_prob = max(0.10, min(0.40, adjusted_k_prob))

    # ========== STEP 4: Beta-Binomial Distribution ==========
    n = int(round(expected_bf))

    # Beta parameters
    # Higher concentration = more certainty in K probability
    concentration = 8 + (ip / 15)  # More IP = more confident
    alpha = adjusted_k_prob * concentration
    beta = (1 - adjusted_k_prob) * concentration

    # Create distribution
    dist = betabinom(n, alpha, beta)
    projected_ks = float(dist.mean())

    # ========== STEP 5: Calculate Probabilities ==========
    if line is not None:
        # P(K > line) for half lines like 5.5
        prob_under = float(dist.cdf(line))
        prob_over = 1 - prob_under
    else:
        # Default to 5.5 line
        prob_under = float(dist.cdf(5.5))
        prob_over = 1 - prob_under

    # ========== STEP 6: Confidence Grade ==========
    confidence_grade = _calculate_confidence_grade(pitcher, expected_bf)

    return MLBProjection(
        pitcher_name=pitcher['name'],
        opponent_team=opponent_team,
        projected_ks=round(projected_ks, 2),
        expected_bf=round(expected_bf, 1),
        k_probability=round(adjusted_k_prob, 4),
        prob_over=round(prob_over, 4),
        prob_under=round(prob_under, 4),
        confidence_grade=confidence_grade,
        inputs={
            'swstr_pct': swstr,
            'k_pct': k_pct,
            'csw_pct': pitcher.get('csw_pct'),
            'ip': ip,
            'pitcher_hand': pitcher_hand,
            'opp_k_pct': opp_k_pct,
            'opp_p_per_pa': opp_p_per_pa,
            'expected_pitch_count': expected_pitch_count,
        },
        adjustments={
            'opponent': round(opp_adjustment, 3),
            'park': park_factor,
        }
    )


def _calculate_confidence_grade(pitcher: Dict, expected_bf: float) -> str:
    """Calculate confidence grade for projection."""
    score = 0
    ip = pitcher.get('ip', 0)

    # Sample size (max 35)
    if ip >= 100:
        score += 35
    elif ip >= 50:
        score += 25
    elif ip >= 30:
        score += 18
    elif ip >= 15:
        score += 10

    # Data completeness (max 30)
    if pitcher.get('swstr_pct'):
        score += 15
    if pitcher.get('csw_pct'):
        score += 10
    if pitcher.get('k_pct'):
        score += 5

    # Workload confidence (max 20)
    gs = pitcher.get('games_started', 0)
    if gs >= 20:
        score += 20
    elif gs >= 10:
        score += 15
    elif gs >= 5:
        score += 10

    # Expected workload (max 15)
    if expected_bf >= 25:
        score += 15
    elif expected_bf >= 22:
        score += 10
    elif expected_bf >= 18:
        score += 5

    if score >= 80:
        return 'A'
    elif score >= 60:
        return 'B'
    elif score >= 40:
        return 'C'
    else:
        return 'D'


def calculate_ev(prob: float, odds: int) -> float:
    """Calculate expected value given probability and American odds."""
    if odds > 0:
        # Underdog: profit = odds, risk = 100
        ev = (prob * odds) - ((1 - prob) * 100)
    else:
        # Favorite: profit = 100, risk = abs(odds)
        ev = (prob * 100) - ((1 - prob) * abs(odds))

    return round(ev / 100, 4)  # Return as decimal


def analyze_prop(
    pitcher_name: str,
    opponent_team: str,
    line: float,
    over_odds: int,
    under_odds: int,
    venue: str = None
) -> Optional[Dict]:
    """
    Full analysis of a strikeout prop.

    Returns recommendation with edge and EV.
    """
    projection = project_strikeouts(
        pitcher_name=pitcher_name,
        opponent_team=opponent_team,
        venue=venue,
        line=line
    )

    if not projection:
        return None

    # Calculate EV for both sides
    ev_over = calculate_ev(projection.prob_over, over_odds)
    ev_under = calculate_ev(projection.prob_under, under_odds)

    # Determine recommendation
    if ev_over > ev_under and ev_over > 0:
        recommended_side = 'over'
        edge = ev_over
    elif ev_under > ev_over and ev_under > 0:
        recommended_side = 'under'
        edge = ev_under
    else:
        # No positive EV
        recommended_side = 'pass'
        edge = max(ev_over, ev_under)

    return {
        'pitcher': projection.pitcher_name,
        'opponent': projection.opponent_team,
        'projected_ks': projection.projected_ks,
        'line': line,
        'prob_over': projection.prob_over,
        'prob_under': projection.prob_under,
        'ev_over': ev_over,
        'ev_under': ev_under,
        'recommended_side': recommended_side,
        'edge': edge,
        'confidence_grade': projection.confidence_grade,
        'inputs': projection.inputs,
        'adjustments': projection.adjustments,
    }


if __name__ == "__main__":
    # Test projection
    print("=" * 60)
    print("MLB PROJECTION TEST")
    print("=" * 60)

    # Test with a known pitcher
    result = project_strikeouts(
        pitcher_name="Tarik Skubal",
        opponent_team="NYY",
        venue="Comerica Park"
    )

    if result:
        print(f"\n{result.pitcher_name} vs {result.opponent_team}")
        print(f"  Projected Ks: {result.projected_ks}")
        print(f"  Expected BF: {result.expected_bf}")
        print(f"  K Probability: {result.k_probability:.1%}")
        print(f"  P(Over 5.5): {result.prob_over:.1%}")
        print(f"  P(Under 5.5): {result.prob_under:.1%}")
        print(f"  Confidence: {result.confidence_grade}")
        print(f"\n  Inputs:")
        for k, v in result.inputs.items():
            print(f"    {k}: {v}")

    # Test full analysis with line
    print("\n" + "=" * 60)
    print("PROP ANALYSIS TEST")
    print("=" * 60)

    analysis = analyze_prop(
        pitcher_name="Logan Webb",
        opponent_team="Yankees",
        line=5.5,
        over_odds=-115,
        under_odds=-105,
        venue="Oracle Park"
    )

    if analysis:
        print(f"\n{analysis['pitcher']} vs {analysis['opponent']}")
        print(f"  Line: {analysis['line']}")
        print(f"  Projected: {analysis['projected_ks']}")
        print(f"  P(Over): {analysis['prob_over']:.1%} | EV: {analysis['ev_over']:.2%}")
        print(f"  P(Under): {analysis['prob_under']:.1%} | EV: {analysis['ev_under']:.2%}")
        print(f"  Recommendation: {analysis['recommended_side'].upper()}")
        print(f"  Confidence: {analysis['confidence_grade']}")
