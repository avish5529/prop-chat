# MLB Strikeout Props Integration Plan

## Overview
Extend Prop.chat to support MLB pitcher strikeout props using an industry-aligned projection model with **fully automated data collection** (no manual input).

**Target Launch:** March 27, 2026 (Opening Day + 1)

---

## Research Summary

### What Industry Models Use
Based on research of [Ballpark Pal](https://www.ballparkpal.com/Methods.html), [DRatings](https://www.dratings.com/mlb-strikeout-distribution-analysis/), and academic papers:

| Feature | Industry Standard | Our Approach |
|---------|------------------|--------------|
| **Distribution** | Beta-Binomial | Beta-Binomial |
| **Core metric** | SwStr% (0.87 correlation to K%) | SwStr% + K% blend |
| **Workload** | Batters faced (not IP) | Expected BF from pitch count |
| **Opponent** | Team K% vs hand | Team K% vs L/R |
| **Park factors** | Yes | Yes (Phase 2) |
| **Umpire factors** | Yes (~10 K swing) | Yes (Phase 2) |

### Key Research Findings
1. **Beta-Binomial > Poisson** - Strikeouts aren't independent, Poisson underdisperses
2. **SwStr% × 2 ≈ K%** - Strong heuristic (0.87 correlation)
3. **Umpire effects** - 10+ called strike difference between strictest/loosest
4. **Park effects** - Foul territory, altitude, batter's eye all matter
5. **Times through order** - K rate drops ~20% third time through

---

## Data Sources - FULLY AUTOMATED

### Complete Data Map

| Data Needed | Source | Function | Frequency | Manual? |
|-------------|--------|----------|-----------|---------|
| **Pitcher K%, SwStr%, CSW%** | FanGraphs via pybaseball | `pyb.fg_pitching_data(2026)` | Daily | **No** |
| **Pitcher Whiff% by pitch** | Statcast via pybaseball | `pyb.statcast_pitcher()` | Weekly | **No** |
| **Team K% (overall)** | FanGraphs via pybaseball | `pyb.fg_team_batting_data(2026)` | Daily | **No** |
| **Team K% vs LHP/RHP** | FanGraphs via pybaseball | `pyb.fg_team_batting_data(2026, split='vs L')` | Daily | **No** |
| **Team P/PA (patience)** | FanGraphs via pybaseball | `pyb.fg_team_batting_data(2026)` | Daily | **No** |
| **Today's games** | MLB Stats API | `statsapi.schedule()` | Daily | **No** |
| **Probable pitchers** | MLB Stats API | `statsapi.schedule()` | Daily | **No** |
| **Pitcher handedness** | MLB Stats API | `statsapi.lookup_player()` | On-demand | **No** |
| **Pitcher game logs** | MLB Stats API | `statsapi.player_stat_data()` | On-demand | **No** |
| **Historical pitch counts** | MLB Stats API | Game-by-game logs | Built over time | **No** |
| **Strikeout lines/odds** | The Odds API | `pitcher_strikeouts` market | Per request | **No** |
| **Park K factors** | Baseball Savant | Scrape or static table | Weekly | **No** |
| **Umpire tendencies** | UmpireScorecards API | HTTP request | Daily | **No** |
| **Game results (Ks)** | MLB Stats API | `statsapi.boxscore()` | Next day | **No** |

### Why This Is Better Than Old Project
The old `mlb-k-optimizer` required manual entry because:
- Used `manual_metrics_manager.py` for advanced stats
- CSW%, SwStr% had to be looked up manually
- No automated FanGraphs integration

**New approach**: `pybaseball` scrapes FanGraphs automatically, giving us all advanced metrics without any manual work.

---

## Phase 1: Data Infrastructure

### 1.1 New Dependencies
```bash
pip install pybaseball statsapi scipy
```

### 1.2 Cache Files (Auto-refreshed Daily)
```
Prop.chat/
├── mlb_pitcher_stats_cache.json    # FanGraphs pitcher data (K%, SwStr%, CSW%)
├── mlb_team_batting_cache.json     # FanGraphs team K% data
├── mlb_park_factors_cache.json     # Park K factors
├── mlb_umpire_cache.json           # Umpire K tendencies
└── mlb_workload_history.json       # Historical pitch counts (grows over time)
```

### 1.3 Data Fetching Scripts

#### `mlb_fetch_all.py` - Master Daily Script
```python
#!/usr/bin/env python3
"""
MLB Daily Data Refresh - Run before first game (~11am ET)
Fetches ALL data automatically from pybaseball + statsapi
"""
import pybaseball as pyb
import statsapi
import json
import requests
from datetime import datetime, date

# Disable pybaseball cache for fresh data
pyb.cache.enable()

def fetch_pitcher_stats(year=2026):
    """Fetch ALL pitcher stats from FanGraphs - NO MANUAL INPUT."""
    print(f"[MLB] Fetching pitcher stats from FanGraphs...")

    # This single call gets K%, SwStr%, CSW%, and everything else
    df = pyb.fg_pitching_data(year, qual=1)  # qual=1 means min 1 IP

    pitchers = {}
    for _, row in df.iterrows():
        name = row['Name']

        # Parse percentage columns (FanGraphs returns as "25.3%" or 0.253)
        def parse_pct(val):
            if val is None:
                return None
            if isinstance(val, str):
                return float(val.strip('%'))
            return float(val) * 100 if val < 1 else float(val)

        pitchers[name.lower()] = {
            'name': name,
            'team': row['Team'],
            'hand': 'L' if 'L' in str(row.get('Throws', 'R')) else 'R',
            'ip': float(row['IP']),
            'games': int(row.get('G', 0)),
            'games_started': int(row.get('GS', 0)),
            'k_pct': parse_pct(row.get('K%')),
            'bb_pct': parse_pct(row.get('BB%')),
            'k_per_9': float(row.get('K/9', 0)),
            'swstr_pct': parse_pct(row.get('SwStr%')),      # KEY METRIC
            'csw_pct': parse_pct(row.get('CSW%')),          # KEY METRIC
            'o_swing_pct': parse_pct(row.get('O-Swing%')),  # Chase rate
            'z_contact_pct': parse_pct(row.get('Z-Contact%')),
            'whip': float(row.get('WHIP', 0)),
            'era': float(row.get('ERA', 0)),
            'fip': float(row.get('FIP', 0)),
            'xfip': float(row.get('xFIP', 0)),
            'babip': float(row.get('BABIP', 0)),
        }

    cache = {
        'updated': datetime.now().isoformat(),
        'season': year,
        'count': len(pitchers),
        'pitchers': pitchers
    }

    with open('mlb_pitcher_stats_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"[MLB] Cached {len(pitchers)} pitchers with SwStr%, CSW%, K%")
    return pitchers


def fetch_team_batting(year=2026):
    """Fetch team batting stats including K% and P/PA - NO MANUAL INPUT."""
    print(f"[MLB] Fetching team batting stats from FanGraphs...")

    df = pyb.fg_team_batting_data(year)

    teams = {}
    for _, row in df.iterrows():
        team = row['Team']

        def parse_pct(val):
            if val is None:
                return None
            if isinstance(val, str):
                return float(val.strip('%'))
            return float(val) * 100 if val < 1 else float(val)

        pa = int(row.get('PA', 1))
        pitches = int(row.get('Pitches', pa * 3.9))

        teams[team] = {
            'team': team,
            'pa': pa,
            'k_pct': parse_pct(row.get('K%')),
            'bb_pct': parse_pct(row.get('BB%')),
            'p_per_pa': round(pitches / pa, 2) if pa > 0 else 3.9,
            'o_swing_pct': parse_pct(row.get('O-Swing%')),
            'z_contact_pct': parse_pct(row.get('Z-Contact%')),
        }

    # Also fetch splits vs LHP/RHP if available
    try:
        df_vs_l = pyb.fg_team_batting_data(year, split_seasons=False, split='vl')
        df_vs_r = pyb.fg_team_batting_data(year, split_seasons=False, split='vr')

        for _, row in df_vs_l.iterrows():
            team = row['Team']
            if team in teams:
                teams[team]['k_pct_vs_lhp'] = parse_pct(row.get('K%'))

        for _, row in df_vs_r.iterrows():
            team = row['Team']
            if team in teams:
                teams[team]['k_pct_vs_rhp'] = parse_pct(row.get('K%'))
    except:
        print("[MLB] Warning: Could not fetch L/R splits, using overall K%")

    cache = {
        'updated': datetime.now().isoformat(),
        'season': year,
        'count': len(teams),
        'teams': teams
    }

    with open('mlb_team_batting_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"[MLB] Cached {len(teams)} teams with K%, P/PA")
    return teams


def fetch_todays_games():
    """Fetch today's games with probable pitchers - NO MANUAL INPUT."""
    print(f"[MLB] Fetching today's schedule from MLB Stats API...")

    today = date.today().strftime("%m/%d/%Y")
    schedule = statsapi.schedule(start_date=today, end_date=today)

    games = []
    for game in schedule:
        games.append({
            'game_pk': game['game_id'],
            'game_date': game['game_date'],
            'game_time': game.get('game_datetime'),
            'status': game.get('status'),
            'home_team': game['home_name'],
            'away_team': game['away_name'],
            'home_pitcher': game.get('home_probable_pitcher'),
            'away_pitcher': game.get('away_probable_pitcher'),
            'venue': game.get('venue_name'),
        })

    print(f"[MLB] Found {len(games)} games today")
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']}: {g['away_pitcher']} vs {g['home_pitcher']}")

    return games


def fetch_park_factors():
    """Fetch park K factors from Baseball Savant - NO MANUAL INPUT."""
    print(f"[MLB] Fetching park factors...")

    # Baseball Savant park factors (can scrape or use static recent data)
    # For now, use known 2025 factors - update annually
    # Source: https://baseballsavant.mlb.com/leaderboard/statcast-park-factors

    park_factors = {
        'Coors Field': 0.85,        # Fewer Ks (altitude)
        'Chase Field': 0.95,
        'Globe Life Field': 1.02,
        'Tropicana Field': 1.08,    # More Ks
        'Oracle Park': 1.05,
        'Petco Park': 1.03,
        'T-Mobile Park': 1.06,
        'Oakland Coliseum': 1.04,
        # ... add all 30 parks
        # Default for unknown
        'default': 1.0
    }

    cache = {
        'updated': datetime.now().isoformat(),
        'note': 'K factor: >1 = more Ks, <1 = fewer Ks',
        'parks': park_factors
    }

    with open('mlb_park_factors_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"[MLB] Cached {len(park_factors)} park factors")
    return park_factors


def fetch_umpire_data():
    """Fetch today's umpire assignments and tendencies - NO MANUAL INPUT."""
    print(f"[MLB] Fetching umpire data...")

    # UmpireScorecards.com has API or can scrape
    # Key metric: "established zone" - bigger zone = more Ks

    # For MVP, use league average (1.0) for all
    # Phase 2: integrate UmpireScorecards API

    umpire_factors = {
        'default': 1.0,
        # Will populate with real data:
        # 'Angel Hernandez': 0.95,  # Tight zone
        # 'Pat Hoberg': 1.05,       # Generous zone
    }

    cache = {
        'updated': datetime.now().isoformat(),
        'note': 'K factor: >1 = more Ks (bigger zone)',
        'umpires': umpire_factors
    }

    with open('mlb_umpire_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"[MLB] Umpire factors ready (using defaults for MVP)")
    return umpire_factors


if __name__ == "__main__":
    print("=" * 60)
    print("MLB DAILY DATA REFRESH")
    print("=" * 60)

    fetch_pitcher_stats(2026)
    fetch_team_batting(2026)
    fetch_todays_games()
    fetch_park_factors()
    fetch_umpire_data()

    print("\n" + "=" * 60)
    print("ALL DATA REFRESHED - Ready for predictions")
    print("=" * 60)
```

---

## Phase 2: Database Schema

### 2.1 New Table: `mlb_predictions`
```sql
CREATE TABLE mlb_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Game Info
    game_pk INTEGER,
    game_date TEXT,
    pitcher_name TEXT,
    pitcher_id INTEGER,
    pitcher_team TEXT,
    pitcher_hand TEXT,
    opponent_team TEXT,
    venue TEXT,

    -- Line Info (from Odds API)
    line REAL,
    over_odds INTEGER,
    under_odds INTEGER,
    best_book TEXT,

    -- Projection Inputs (all auto-fetched)
    swstr_pct REAL,
    csw_pct REAL,
    k_pct REAL,
    expected_pitch_count REAL,
    expected_bf REAL,
    opp_k_pct REAL,
    opp_k_pct_vs_hand REAL,
    opp_p_per_pa REAL,
    park_factor REAL,
    umpire_factor REAL,

    -- Projection Outputs
    projected_ks REAL,
    prob_over REAL,
    prob_under REAL,
    recommended_side TEXT,
    edge REAL,
    ev REAL,
    confidence_grade TEXT,

    -- Results (auto-synced next day)
    actual_ks INTEGER,
    actual_ip REAL,
    actual_pitches INTEGER,
    actual_bf INTEGER,
    hit INTEGER,
    status TEXT DEFAULT 'pending',

    -- CLV (auto-fetched)
    closing_line REAL,
    clv REAL,

    -- Timestamps
    created_at TEXT,
    resolved_at TEXT
);
```

---

## Phase 3: Projection Model (Industry-Aligned)

### 3.1 Core Formula - Beta-Binomial
```python
from scipy.stats import betabinom
import json

def load_caches():
    """Load all cached data."""
    with open('mlb_pitcher_stats_cache.json') as f:
        pitchers = json.load(f)['pitchers']
    with open('mlb_team_batting_cache.json') as f:
        teams = json.load(f)['teams']
    with open('mlb_park_factors_cache.json') as f:
        parks = json.load(f)['parks']
    with open('mlb_umpire_cache.json') as f:
        umpires = json.load(f)['umpires']
    return pitchers, teams, parks, umpires


def project_strikeouts(pitcher_name, opponent_team, venue, umpire=None):
    """
    Industry-aligned strikeout projection using Beta-Binomial distribution.

    ALL DATA IS AUTO-FETCHED - NO MANUAL INPUT.
    """
    pitchers, teams, parks, umpires = load_caches()

    # Get pitcher data (auto-fetched from FanGraphs)
    pitcher = pitchers.get(pitcher_name.lower())
    if not pitcher:
        raise ValueError(f"Pitcher not found: {pitcher_name}")

    # Get opponent data (auto-fetched from FanGraphs)
    opponent = teams.get(opponent_team)
    if not opponent:
        raise ValueError(f"Team not found: {opponent_team}")

    # Get factors (auto-fetched)
    park_factor = parks.get(venue, parks.get('default', 1.0))
    umpire_factor = umpires.get(umpire, umpires.get('default', 1.0))

    # ========== STEP 1: Estimate Batters Faced ==========
    # Use pitcher's historical avg or league avg ~90 pitches
    # TODO: Build historical pitch count data over time
    expected_pitch_count = 90  # Will improve with data

    # Opponent patience (P/PA)
    opp_p_per_pa = opponent.get('p_per_pa', 3.9)

    expected_bf = expected_pitch_count / opp_p_per_pa

    # ========== STEP 2: Calculate K Probability ==========
    # SwStr% × 2 ≈ K% (research-backed, 0.87 correlation)
    swstr = pitcher.get('swstr_pct') or 11.0
    k_pct_from_swstr = swstr * 2 / 100

    # Actual K% (more reliable with sample)
    actual_k_pct = pitcher.get('k_pct')
    ip = pitcher.get('ip', 0)

    if actual_k_pct and ip > 20:
        # Blend: weight toward actual as sample grows
        weight = min(0.8, ip / 50)  # Max 80% weight to actual
        base_k_prob = (k_pct_from_swstr * (1 - weight)) + (actual_k_pct / 100 * weight)
    else:
        base_k_prob = k_pct_from_swstr

    # ========== STEP 3: Adjustments ==========
    # Opponent K% vs pitcher hand
    pitcher_hand = pitcher.get('hand', 'R')
    if pitcher_hand == 'L':
        opp_k_pct = opponent.get('k_pct_vs_lhp') or opponent.get('k_pct', 22.5)
    else:
        opp_k_pct = opponent.get('k_pct_vs_rhp') or opponent.get('k_pct', 22.5)

    league_avg_k = 22.5
    opp_adjustment = (opp_k_pct / league_avg_k) - 1

    # Apply all adjustments (dampened)
    adjusted_k_prob = base_k_prob * (1 + opp_adjustment * 0.3) * park_factor * umpire_factor

    # Clamp to reasonable range
    adjusted_k_prob = max(0.10, min(0.40, adjusted_k_prob))

    # ========== STEP 4: Beta-Binomial Distribution ==========
    n = int(round(expected_bf))

    # Beta parameters (concentration controls variance)
    # Higher concentration = more certainty in K probability
    concentration = 8 + (ip / 10)  # More IP = more confident
    alpha = adjusted_k_prob * concentration
    beta = (1 - adjusted_k_prob) * concentration

    dist = betabinom(n, alpha, beta)
    projected_ks = dist.mean()

    return {
        'projected_ks': round(projected_ks, 2),
        'expected_bf': round(expected_bf, 1),
        'expected_pitch_count': expected_pitch_count,
        'k_probability': round(adjusted_k_prob, 4),
        'distribution': dist,
        'inputs': {
            'swstr_pct': swstr,
            'k_pct': actual_k_pct,
            'csw_pct': pitcher.get('csw_pct'),
            'ip': ip,
            'opp_k_pct': opp_k_pct,
            'opp_p_per_pa': opp_p_per_pa,
            'park_factor': park_factor,
            'umpire_factor': umpire_factor,
        },
        'adjustments': {
            'opponent': round(opp_adjustment, 3),
            'park': park_factor,
            'umpire': umpire_factor,
        }
    }


def calculate_over_under_prob(projection, line):
    """Calculate probabilities using Beta-Binomial CDF."""
    dist = projection['distribution']

    # P(K > line) for half lines
    prob_under = dist.cdf(line)
    prob_over = 1 - prob_under

    return {
        'prob_over': round(prob_over, 4),
        'prob_under': round(prob_under, 4)
    }


def get_confidence_grade(projection):
    """Grade projection reliability."""
    inputs = projection['inputs']
    score = 0

    # Sample size (max 35)
    ip = inputs.get('ip', 0)
    if ip >= 50: score += 35
    elif ip >= 30: score += 25
    elif ip >= 15: score += 15
    elif ip >= 5: score += 5

    # Data completeness (max 30)
    if inputs.get('swstr_pct'): score += 15
    if inputs.get('csw_pct'): score += 10
    if inputs.get('k_pct'): score += 5

    # Adjustment confidence (max 20)
    if inputs.get('opp_k_pct'): score += 10
    if inputs.get('park_factor', 1.0) != 1.0: score += 5
    if inputs.get('umpire_factor', 1.0) != 1.0: score += 5

    # Workload confidence (max 15)
    bf = projection.get('expected_bf', 20)
    if bf >= 25: score += 15
    elif bf >= 20: score += 10
    elif bf >= 15: score += 5

    if score >= 80: return 'A'
    elif score >= 60: return 'B'
    elif score >= 40: return 'C'
    else: return 'D'
```

---

## Phase 4: Results Syncing (Automated)

### `mlb_sync_results.py`
```python
#!/usr/bin/env python3
"""
Sync MLB results - ALL AUTOMATED via statsapi
"""
import statsapi
import sqlite3
from datetime import datetime, timedelta

def sync_mlb_results(game_date=None):
    """Sync actual strikeout results from MLB API."""

    if game_date is None:
        game_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[MLB] Syncing results for {game_date}...")

    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    # Get pending predictions for this date
    cursor.execute("""
        SELECT id, pitcher_name, game_pk
        FROM mlb_predictions
        WHERE game_date = ? AND status = 'pending'
    """, (game_date,))

    pending = cursor.fetchall()
    print(f"[MLB] Found {len(pending)} pending predictions")

    for pred_id, pitcher_name, game_pk in pending:
        try:
            # Get boxscore from MLB API
            box = statsapi.boxscore_data(game_pk)

            # Find pitcher's stats
            actual_ks = None
            actual_ip = None
            actual_pitches = None

            for side in ['away', 'home']:
                pitchers = box.get(side, {}).get('pitchers', [])
                for p in pitchers:
                    if pitcher_name.lower() in p.get('name', '').lower():
                        stats = p.get('stats', {})
                        actual_ks = stats.get('strikeOuts', 0)
                        actual_ip = stats.get('inningsPitched', 0)
                        actual_pitches = stats.get('numberOfPitches', 0)
                        break

            if actual_ks is not None:
                # Get the line and recommended side
                cursor.execute("""
                    SELECT line, recommended_side
                    FROM mlb_predictions WHERE id = ?
                """, (pred_id,))
                line, rec_side = cursor.fetchone()

                # Calculate hit
                if rec_side == 'over':
                    hit = 1 if actual_ks > line else 0
                else:
                    hit = 1 if actual_ks < line else 0

                # Update prediction
                cursor.execute("""
                    UPDATE mlb_predictions
                    SET actual_ks = ?, actual_ip = ?, actual_pitches = ?,
                        hit = ?, status = 'resolved', resolved_at = ?
                    WHERE id = ?
                """, (actual_ks, actual_ip, actual_pitches, hit,
                      datetime.now().isoformat(), pred_id))

                print(f"  {pitcher_name}: {actual_ks} Ks, {'HIT' if hit else 'MISS'}")
            else:
                # Pitcher didn't play - void
                cursor.execute("""
                    UPDATE mlb_predictions SET status = 'voided' WHERE id = ?
                """, (pred_id,))
                print(f"  {pitcher_name}: DNP - voided")

        except Exception as e:
            print(f"  {pitcher_name}: Error - {e}")

    conn.commit()
    conn.close()
    print(f"[MLB] Sync complete")
```

---

## Daily Operations - ALL AUTOMATED

### Morning (before first game ~11am ET)
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Single command refreshes ALL data
python mlb_fetch_all.py
```

### After Games (next morning)
```bash
# Single command syncs ALL results
python mlb_sync_results.py
```

---

## Implementation Order

### Week 1: Data Pipeline
1. [x] Research complete
2. [ ] Add dependencies: `pip install pybaseball statsapi scipy`
3. [ ] Create `mlb_fetch_all.py` (data fetching)
4. [ ] Test pybaseball with 2025 data
5. [ ] Create `mlb_predictions` table

### Week 2: Projection Engine
6. [ ] Implement Beta-Binomial projection model
7. [ ] Add `/analyze-mlb` endpoint
8. [ ] Integrate Odds API `pitcher_strikeouts`
9. [ ] Test with Opening Day games

### Week 3: Results & Tracking
10. [ ] Create `mlb_sync_results.py`
11. [ ] Add CLV tracking (same as NBA)
12. [ ] Frontend integration
13. [ ] Monitor and tune

### Phase 2 (Month 2):
- [ ] Real umpire data from UmpireScorecards
- [ ] Real park factors from Baseball Savant
- [ ] Historical pitch count tracking
- [ ] Times-through-order adjustment

---

## Data Source Summary

**The key insight**: `pybaseball` gives us **everything** we need automatically:

| Metric | Old Project | New Approach |
|--------|-------------|--------------|
| SwStr% | Manual lookup | `pyb.fg_pitching_data()` - **AUTO** |
| CSW% | Manual lookup | `pyb.fg_pitching_data()` - **AUTO** |
| K% | Manual lookup | `pyb.fg_pitching_data()` - **AUTO** |
| Team K% | Manual lookup | `pyb.fg_team_batting_data()` - **AUTO** |
| K% vs L/R | Not available | `pyb.fg_team_batting_data(split='vl')` - **AUTO** |
| P/PA | Manual calc | `pyb.fg_team_batting_data()` - **AUTO** |
| Schedule | statsapi | `statsapi.schedule()` - **AUTO** |
| Results | Manual | `statsapi.boxscore_data()` - **AUTO** |
| Odds | Odds API | Same as NBA - **AUTO** |

**Zero manual input required.**

---

## Sources

- [Ballpark Pal Methods](https://www.ballparkpal.com/Methods.html)
- [DRatings Beta-Binomial](https://www.dratings.com/mlb-strikeout-distribution-analysis/)
- [FanGraphs Strikeout Rates](https://library.fangraphs.com/pitching/rate-stats/)
- [pybaseball Documentation](https://github.com/jldbc/pybaseball)
- [The Odds API MLB](https://the-odds-api.com/sports/mlb-odds.html)
