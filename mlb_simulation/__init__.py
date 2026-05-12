"""
MLB Strikeout Simulation Package

Tier 3 Monte Carlo simulation for MLB strikeout props.
Simulates each plate appearance using confirmed lineups and batter-specific K rates.
"""

from .batter_data import (
    get_batter,
    get_batter_k_rate,
    get_batter_bb_rate,
    get_league_k_rate,
    fetch_batters,
    refresh_batters,
)

from .lineup_fetcher import (
    get_lineup,
    get_opponent_lineup,
    is_lineup_confirmed,
    fetch_todays_lineups,
    refresh_lineups,
)

from .simulator import (
    simulate_strikeouts,
    run_simulation,
    log5_probability,
    analyze_prop_simulation,
    PitcherStats,
    BatterStats,
    SimulationResult,
)

__all__ = [
    # Batter data
    'get_batter',
    'get_batter_k_rate',
    'get_batter_bb_rate',
    'get_league_k_rate',
    'fetch_batters',
    'refresh_batters',
    # Lineup data
    'get_lineup',
    'get_opponent_lineup',
    'is_lineup_confirmed',
    'fetch_todays_lineups',
    'refresh_lineups',
    # Simulation
    'simulate_strikeouts',
    'run_simulation',
    'log5_probability',
    'analyze_prop_simulation',
    'PitcherStats',
    'BatterStats',
    'SimulationResult',
]
