# Prop.chat - Project Context

## Overview
Sports props chatbot covering **NBA player props** and **MLB pitcher strikeouts**. Analyzes betting lines using live odds data and historical player statistics. Provides EV-based recommendations with confidence grades.

## Tech Stack
- **Backend**: FastAPI (Python) with Uvicorn server
- **Frontend**: React 18 + Tailwind CSS (embedded in single HTML file)
- **Database**: SQLite (`predictions.db`)
- **ML Model**: CatBoost binary classifier V2.2 (59.7% CV accuracy, +27.2% ROI @ 55% conf) - NBA only
- **External APIs**:
  - The Odds API - live sportsbook lines (NBA + MLB)
  - nba_api - NBA player stats, game logs, team data
  - pybaseball - MLB pitcher stats from FanGraphs (automated)
  - statsapi (MLB) - schedules, boxscores, results

## File Structure
```
Prop.chat/
├── main.py                  # FastAPI backend - all core logic + CatBoost integration
├── index.html               # React frontend - chat UI with ML predictions
├── predictions.db           # SQLite database
├── requirements.txt         # Python dependencies (includes catboost, scikit-learn)
├── .env                     # API keys (ODDS_API_KEY, BALLDONTLIE_API_KEY)
├── venv/                    # Python virtual environment
│
├── # ML Model Files (V2.2 - Active)
├── catboost_model_v2_2.cbm  # CatBoost V2.2 model (59.7% CV accuracy) - CURRENT
├── catboost_predictor.py    # Production inference module (uses V2.2)
├── feature_columns_v2_2.json # V2.2 feature configuration
├── training_metrics_v2_2.json # V2.2 performance metrics
├── train_catboost_v2_2.py   # V2.2 training script
├── backtest_data_enriched.csv # Enriched backtest data (15,453 rows with all features)
│
├── # Backfill Scripts (for enriching backtest data)
├── backfill_backtest_avg_vs_opponent.py # Backfill avg_vs_opponent
├── backfill_backtest_dvp.py   # Backfill DvP features + player_position
├── backfill_backtest_team_stats.py # Backfill opp_def_rating, opp_pace
├── dvp.xlsx                 # FantasyPros DvP data by position
│
├── # ML Model Files (V2.1 - Archived)
├── catboost_model_v2_1.cbm  # V2.1 model (56.1% accuracy, had prop-type biases)
├── feature_columns_v2_1.json # V2.1 feature configuration
├── training_metrics_v2_1.json # V2.1 performance metrics
│
├── # ML Model Files (V2 - Archived)
├── catboost_model_v2.cbm    # CatBoost V2 model (60.7% accuracy)
├── feature_columns_v2.json  # V2 feature configuration
├── training_metrics_v2.json # V2 performance metrics
├── training_data_v2.csv     # V2 training data with new features
├── prepare_training_data_v2.py # V2 data prep with matchup features
├── train_catboost_v2.py     # V2 training script
│
├── # ML Model Files (V1 - Archived)
├── catboost_model.cbm       # CatBoost V1 model (57.1% accuracy)
├── feature_columns.json     # V1 feature configuration
├── training_metrics.json    # V1 performance metrics
│
├── # Training Pipeline
├── prepare_training_data.py # V1 data prep & feature engineering
├── training_data.csv        # V1 processed training data (15,253 rows)
├── baseline_model.py        # Logistic regression baseline
├── baseline_metrics.json    # Baseline performance metrics
├── train_catboost.py        # V1 CatBoost training script
├── evaluate_model.py        # Model evaluation & calibration
├── evaluation_results.json  # Evaluation metrics
│
├── # Historical Data & CLV
├── backtest_data.csv        # Raw backtest data with closing lines (15,453 rows)
│
├── # MLB Files (Separate from NBA)
├── mlb_data.py              # Core MLB data module - pitcher/team stats with cache fallback
├── mlb_projection_v2.py     # Beta-Binomial strikeout projection engine (V2)
├── mlb_sync_results.py      # Syncs results from MLB Stats API
├── mlb_fetch_clv.py         # CLV fetcher for MLB props (Odds API historical)
├── mlb_refresh_data.py      # Daily refresh script - pulls LIVE data from pybaseball
├── mlb_simulation/          # Tier 3 Monte Carlo simulation (Apr 2026)
│   ├── __init__.py          # Package exports
│   ├── batter_data.py       # Batter K%/BB% from Baseball Reference
│   ├── lineup_fetcher.py    # Confirmed lineups from MLB Stats API
│   └── simulator.py         # Monte Carlo engine (Log5, fatigue, pull model)
├── cache/                   # MLB cache directory
│   ├── mlb_pitchers.json    # FanGraphs pitcher stats (auto-refreshed)
│   ├── mlb_teams.json       # FanGraphs team batting stats (auto-refreshed)
│   ├── mlb_park_factors.json # Baseball Savant Statcast SO park factors (live)
│   ├── mlb_batters.json     # Batter K%/BB% (2026 season)
│   ├── mlb_batters_2025.json # Batter K%/BB% (2025 fallback)
│   └── mlb_lineups.json     # Today's confirmed lineups
├── mlb_pitcher_stats_cache.json   # Legacy pitcher cache (fallback)
├── MLB_INTEGRATION_PLAN.md        # Detailed implementation plan
├── scrape_closing_lines.py  # BettingPros scraper for closing lines (legacy)
├── scrape_cache.json        # Cache of scraped closing lines
├── daily_sync.py            # Daily sync script - results only (run after games)
├── fetch_clv.py             # CLV fetcher using Odds API historical (Mar 15, 2026)
│
├── # Defense vs Position (Added Mar 21, 2026)
├── scrape_dvp.py            # FantasyPros DvP scraper
├── dvp_cache.json           # Cached DvP data (refresh daily)
│
├── # Minutes Projection (Deprecated Razzball - Mar 10, 2026)
├── scrape_razzball.py       # Razzball scraper (DEPRECATED - unreliable)
├── razzball_minutes_cache.json  # Cache file (DEPRECATED)
└── nba-props-analyzer-v2.jsx  # Legacy file (unused)
```

## Running the Project
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate
uvicorn main:app --reload --port 8000
# Open index.html in browser (or serve it)
```

## Daily Operations (Sync, CLV, DvP)

### 1. Sync Results (Run morning after games)
Fetches actual box score results from NBA API and updates hit/miss for both models.
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Sync yesterday's results (default)
python3 << 'EOF'
import sqlite3
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

DB_PATH = "predictions.db"
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}

target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
# Or specify: target_date = "2026-03-23"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT * FROM predictions WHERE game_date = ? AND status = 'pending' ORDER BY id", (target_date,))
preds = [dict(row) for row in cursor.fetchall()]
conn.close()

print(f"Syncing {len(preds)} predictions for {target_date}\n")

for pred in preds:
    player_list = players.find_players_by_full_name(pred['player_name'])
    if not player_list:
        all_p = players.get_players()
        for p in all_p:
            if pred['player_name'].lower() in p['full_name'].lower():
                player_id = p['id']
                break
        else:
            print(f"[{pred['id']}] {pred['player_name']} - NOT FOUND, voiding")
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE predictions SET status = 'voided' WHERE id = ?", (pred['id'],))
            conn.commit()
            conn.close()
            continue
    else:
        player_id = player_list[0]['id']

    game_dt = datetime.strptime(pred['game_date'], "%Y-%m-%d")
    season = f"{game_dt.year - 1}-{str(game_dt.year)[2:]}" if game_dt.month < 10 else f"{game_dt.year}-{str(game_dt.year + 1)[2:]}"

    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season, headers=NBA_HEADERS, timeout=30)
        df = log.get_data_frames()[0]
        df['GAME_DATE'] = df['GAME_DATE'].apply(lambda x: datetime.strptime(x, "%b %d, %Y").strftime("%Y-%m-%d"))
        game_row = df[df['GAME_DATE'] == pred['game_date']]

        if game_row.empty:
            print(f"[{pred['id']}] {pred['player_name']} - DNP, voiding")
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE predictions SET status = 'voided' WHERE id = ?", (pred['id'],))
            conn.commit()
            conn.close()
            continue

        row = game_row.iloc[0]
        pts, reb, ast = float(row['PTS']), float(row['REB']), float(row['AST'])
        stat_map = {"points": pts, "rebounds": reb, "assists": ast, "pra": pts+reb+ast, "pr": pts+reb, "pa": pts+ast, "ra": reb+ast}
        actual = stat_map.get(pred['prop_type'])
        minutes = float(row['MIN']) if row['MIN'] else None

        hit = 1 if (pred['recommended_side'] == 'over' and actual > pred['line']) or (pred['recommended_side'] == 'under' and actual < pred['line']) else 0
        cb_hit = None
        if pred.get('catboost_pick'):
            cb_hit = 1 if (pred['catboost_pick'] == 'over' and actual > pred['line']) or (pred['catboost_pick'] == 'under' and actual < pred['line']) else 0

        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE predictions SET actual_result=?, actual_minutes=?, hit=?, catboost_hit=?, status='resolved', resolved_at=? WHERE id=?",
                     (actual, minutes, hit, cb_hit, datetime.now().isoformat(), pred['id']))
        conn.commit()
        conn.close()

        print(f"[{pred['id']}] {pred['player_name']} {pred['prop_type']} {pred['line']} → {actual} | Rule: {'HIT' if hit else 'MISS'} | CB: {'HIT' if cb_hit else 'MISS' if cb_hit is not None else 'N/A'}")
        time.sleep(0.6)
    except Exception as e:
        print(f"[{pred['id']}] {pred['player_name']} - Error: {e}")
EOF
```

**Note:** The old `daily_sync.py` script tries to scrape BettingPros (often blocked). Use the inline script above for results only, then `fetch_clv.py` for closing lines.

### 2. Fetch CLV (After syncing results)
Uses Odds API historical endpoint to get closing lines. Costs ~11 credits per prediction.
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Fetch CLV for specific date
python fetch_clv.py 2026-03-23

# Or fetch for all predictions missing CLV
python fetch_clv.py
```

### 3. Refresh DvP Cache (Daily before making predictions)
Scrapes FantasyPros for position-specific defensive rankings.
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Scrape and save
python scrape_dvp.py

# View current rankings
python scrape_dvp.py --show
```

### 4. Quick Stats Check
```bash
cd /Users/avish/Documents/Prop.chat
sqlite3 predictions.db "SELECT COUNT(*) as resolved, SUM(hit) as rule_hits, ROUND(100.0*SUM(hit)/COUNT(*),1) as rule_pct, SUM(catboost_hit) as cb_hits, COUNT(catboost_hit) as cb_total, ROUND(100.0*SUM(catboost_hit)/COUNT(catboost_hit),1) as cb_pct FROM predictions WHERE status='resolved'"
```

---

## MLB Daily Operations

### 1. Refresh Data (Morning before games ~11am ET)
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Refresh all MLB data (pitcher stats, team batting, park factors)
python mlb_refresh_data.py

# Force refresh even if cache is fresh
python mlb_refresh_data.py --force

# Check cache status (when last refreshed)
python mlb_refresh_data.py --status
```

**What gets refreshed:**
- **Pitchers** (12hr cache): K%, SwStr%, CSW%, K/9, IP, Games from FanGraphs via pybaseball
- **Teams** (12hr cache): Team K% vs LHP/RHP from FanGraphs via pybaseball
- **Parks** (7-day cache): Baseball Savant Statcast SO factors (2023-2025 rolling, switch to 2026 mid-April)

**Known limitations:**
- Some traded pitchers (Dylan Cease, Sonny Gray, Miles Mikolas, Michael Lorenzen) may be missing from Odds API
- Use manual mode with Fanatics lines for these pitchers

### 2. Analyze a Strikeout Prop
```bash
# Via API
curl -X POST http://localhost:8000/mlb/analyze \
  -H "Content-Type: application/json" \
  -d '{"pitcher": "Logan Webb", "opponent": "Yankees", "line": 5.5, "over_odds": -115, "under_odds": -105}'

# Via Python
python -c "
from mlb_projection import analyze_prop
result = analyze_prop('Logan Webb', 'Yankees', 5.5, -115, -105, 'Oracle Park')
print(f\"{result['pitcher']}: {result['projected_ks']} Ks | {result['recommended_side'].upper()} ({result['edge']:.1%} edge)\")
"
```

### 3. Sync Results (Morning after games)
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Sync yesterday's results
python mlb_sync_results.py

# Sync specific date
python mlb_sync_results.py 2026-03-25
```

### 4. Fetch CLV (After syncing results)
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Fetch CLV for all predictions missing closing_line
python mlb_fetch_clv.py

# Fetch CLV for specific date
python mlb_fetch_clv.py 2026-03-27
```

**API Credit Cost:** ~11 credits per prediction (same as NBA)

### 5. MLB Quick Stats Check
```bash
sqlite3 predictions.db "SELECT COUNT(*) as total, SUM(hit) as hits, ROUND(100.0*SUM(hit)/NULLIF(SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END),0),1) as hit_rate FROM mlb_predictions"
```

### Full MLB Daily Workflow
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# Morning BEFORE games (~11am ET):
python mlb_refresh_data.py          # Refresh pitcher/team stats

# Morning AFTER games (next day):
python mlb_sync_results.py          # Sync actual K results
python mlb_fetch_clv.py             # Fetch closing lines + CLV
curl http://localhost:8000/mlb/stats  # Check hit rate & CLV breakdown
```

### MLB API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/mlb/games` | GET | Today's games with probable pitchers |
| `/mlb/analyze` | POST | Analyze strikeout prop |
| `/mlb/predictions` | GET | View predictions |
| `/mlb/stats` | GET | Hit rate statistics |
| `/mlb/refresh-data` | POST | Refresh cached data |

### MLB Projection Formula
Uses **Beta-Binomial distribution** (industry standard):
```
1. Expected BF = Pitch Count (~90) ÷ Opponent P/PA
2. K Probability = blend(SwStr% × 2, actual K%)
3. Adjustments: opponent K% vs hand, park factor
4. Projected Ks = BetaBinomial(BF, K_prob).mean()
```

Key metrics (auto-fetched from FanGraphs via pybaseball):
- **SwStr%** - Swinging strike rate (0.87 correlation to K%)
- **CSW%** - Called strike + whiff rate
- **K%** - Strikeout percentage
- **Team K%** - Opponent strikeout rate

---

## API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze` | POST | Main analysis - takes `{"query": "LeBron points"}` |
| `/events` | GET | Today's NBA games |
| `/predictions` | GET | View saved predictions (?status=all/pending/resolved) |
| `/sync-results` | POST | Resolve predictions with actual box scores |
| `/stats` | GET | Accuracy statistics |
| `/health` | GET | Health check |
| `/backfill-features` | POST | Backfill ML features for historical predictions |

## Database Schema (predictions table)
```sql
-- Core fields
id, player_name, player_id, game_id, game_date, prop_type, line, projection,
recommended_side, confidence_grade, ev, edge, best_odds, best_book,
actual_result, hit, created_at, resolved_at

-- ML Feature fields (added Feb 8, 2026)
opponent_team, is_home, vegas_total, spread, season_avg, last_10_avg,
std_dev, minutes_avg, opp_def_rating, opp_pace, prob_over, no_vig_prob,
days_rest, is_b2b, usage_rate, home_avg, away_avg, model_projection,
actual_minutes

-- CatBoost Model fields (added Feb 21, 2026)
catboost_prob_over,   -- ML model probability of over hitting
catboost_pick,        -- ML model recommendation (over/under)
catboost_confidence,  -- ML model confidence (0-1)
catboost_hit          -- Whether ML pick was correct (resolved after game)

-- Status field values (added Mar 15, 2026)
status                -- 'pending' (awaiting result), 'resolved' (has result), 'voided' (DNP)
```

### Prediction Status Values
| Status | Description |
|--------|-------------|
| `pending` | Game not yet played, awaiting result |
| `resolved` | Game completed, has actual_result and hit values |
| `voided` | Player did not play (DNP) - excluded from hit rate calculations |

## Core Functionality

### Projection Model (main.py) - Minutes-First Approach
**Implemented Mar 2, 2026** - Research-based projection system
**Updated Mar 10, 2026** - Changed to 75/25 formula (research baseline)

1. **Project Minutes First** (75/25 Formula):
   - `Projected Minutes = (Season Avg × 0.75) + (Last 5 Avg × 0.25)`
   - More stable than reactive formulas, industry standard baseline
   - Future: Add injury adjustments, B2B penalty, blowout logic

2. **Calculate Stats Per Minute** (filtered):
   - Filter out games < 20 minutes (injuries, garbage time, blowouts)
   - Weight: 50% last 5 games + 30% next 5 + 20% rest of season

3. **Base Projection**:
   - `Projected Minutes × Stats Per Minute`

4. **Matchup Adjustments** (multiplicative):
   - Defense rating: ±15% max (based on opponent def rating vs league avg)
   - Pace: ±10% max (faster pace = more opportunities)
   - Venue: ±2% (home court advantage)
   - Trend: Hot/cold streak detection (±adjustment based on last 5 vs previous 5)

**Example**: Giannis points vs Boston
- 29.2 min × 0.86 PPM = 25.1 base
- Boston #6 defense: -1.5%, slow pace: -2.7%, home: +2%
- Final: 23.2 projected (vs 20.5 line → OVER)

### Supported Prop Types
- Individual: `points`, `rebounds`, `assists`
- Combos: `pra` (pts+reb+ast), `pr`, `pa`, `ra`

### Betting Math
- EV calculation with no-vig probabilities
- Kelly Criterion bet sizing

### Confidence Grades (A-D)
**Definition**: How reliable is our projection? (NOT edge or probability)

| Factor | Max Points | Scoring |
|--------|------------|---------|
| **Consistency** (CV = std/avg) | 40 | CV<0.2: 40, <0.3: 30, <0.4: 20, <0.5: 10 |
| **Sample Size** (games played) | 30 | >=50: 30, >=30: 20, >=15: 10 |
| **Minutes Volume** | 20 | >=32: 20, >=25: 15, >=18: 10, >=10: 5 |
| **Data Completeness** | 10 | Has opponent data: 10 |

**Grade Thresholds**: A >= 80, B >= 60, C >= 40, D < 40

**Interpretation**:
- Grade A + low edge = "Trustworthy projection, but line is fair"
- Grade D + high edge = "Big edge but uncertain projection"
- Grade A + high edge = "Best spot - reliable AND mispriced"

### Key Functions in main.py
- `parse_query()` - extracts player name + prop type from natural language
- `parse_query_with_manual_line()` - extracts player, prop, line, and odds for manual mode
- `find_nba_player()` - three-pass matching algorithm
- `get_player_stats()` - fetches game logs with combo calculations
- `extract_player_lines()` - parses odds, filters alternate lines
- `calculate_adjusted_projection()` - multi-factor projection
- `estimate_probability()` - normal distribution CDF
- `calculate_ev()` - expected value calculation
- `get_confidence_grade()` - A-D rating based on projection reliability (not edge)
- `save_prediction()` - persists to SQLite

### Manual Mode (Odds API Fallback)
When Odds API quota is exhausted, users can provide their own line and odds:

**Query formats:**
```
"LeBron James points 25.5 -110"    # line + odds
"LeBron James points 25.5"         # line only (assumes -110)
"LeBron points o25.5 -130"         # with over/under prefix
```

**How it works:**
1. `/health` endpoint returns `odds_api_available: true/false`
2. Frontend shows yellow "Manual Mode" indicator when unavailable
3. User adds line + odds to query
4. System skips Odds API, uses provided line as "Fanatics (Manual)"
5. All other features work: projections, ML model, database tracking

**What's preserved in manual mode:**
- Player stats and projections
- CatBoost ML predictions
- EV/edge calculations (using provided odds)
- Database persistence and result syncing
- Confidence grades

**What's lost:**
- Multi-book odds comparison
- Best book recommendation
- Line shopping value

## Current State (as of Feb 2026)

### What's Working
- Live odds from 5+ sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, PointsBet)
- Real-time player stats via nba_api
- Chat UI with detailed analysis cards
- Database persistence and result tracking
- Last 10 games visualization
- Adjustment factors transparency

### Database Stats (Updated Apr 15, 2026)
- 608 resolved predictions, ~25 voided (DNP)
- **CatBoost ML (V2.2): 59.7%** CV accuracy - trained on enriched backtest (15,453 rows)
- **CLV tracking:** Active via Odds API historical (572 predictions with CLV)
- **DvP integration:** Position-specific defense adjustments live
- **V2.2 complete:** Fixed prop-type biases from V2.1, all features backfilled
- **V2.3 roadmap:** Will incorporate CLV when 1,000+ live predictions collected

### Known Model Issues
1. **Unders outperform overs** - Unders hit 59.5% vs overs at 43.8%
2. **Grade D most profitable** - 77.8% hit rate (small sample)
3. **Grade B above breakeven** - 54.1% hit rate
4. **CatBoost high confidence ≠ accuracy** - ML shows 78%+ confidence on PRA props but real results are mixed
5. **Models disagree sometimes** - When they disagree, rule-based winning 2/3 so far (small sample)

### Known Gaps / TODO
- [x] ML Model implementation (CatBoost - 57.1% accuracy)
- [x] Manual mode fallback when Odds API unavailable
- [x] CLV tracking (closing_line, clv columns + daily_sync.py)
- [x] Line movement tracking (opening line = `line`, closing = `closing_line`)
- [ ] Injury-adjusted minutes (Razzball removed Mar 10 - unreliable; using 75/25 formula)
- [ ] Manual mode: calculate opposite side odds from vig (currently both sides get same odds)
- [ ] Automated result syncing (daily_sync.py exists, need cron job)
- [x] Back-to-back game detection (added `is_b2b` column)
- [ ] BallDontLie API integration (key exists but unused)
- [ ] Unit tests
- [ ] Model retraining pipeline (weekly/monthly)
- [ ] **Retrain at 500 predictions** - Add DvP features, evaluate CLV as training signal, combine backtest + live data
- [ ] **MLB Park Factors: Switch to 2026 data** - Currently using 2023-2025 rolling average from Baseball Savant. After 2-3 weeks of 2026 season (~mid-April), switch to 2026-only data when all parks have sufficient sample size

## Environment Variables (.env)
```
ODDS_API_KEY=<key>
BALLDONTLIE_API_KEY=<key>
```

## Notes
- Odds API has daily limit (~500 requests) - check remaining via response headers
- Frontend connects to `http://localhost:8000` (hardcoded in index.html)
- nba_api can be slow on first calls (no rate limits though)

## Backtest Data (Added Feb 21, 2026)

### Overview
Historical player prop data with closing lines scraped from BettingPros for backtesting the projection model.

### Data Stats
- **Total rows**: 15,453
- **With closing lines**: 15,156 (98.1%)
- **NL (No Line)**: 297 (1.9%) - games where sportsbooks didn't post a line
- **Date range**: Oct 22, 2025 - Feb 23, 2026
- **Players**: 50
- **Prop types**: points, rebounds, assists, pra, pr, pa, ra (all ~50% over rate)

### CSV Schema (backtest_data.csv)
```
id, player_name, game_date, opponent_team, is_home, prop_type, actual_result,
closing_line, season_avg, last_10_avg, last_5_avg, std_dev, games_played,
minutes_avg, days_rest, is_b2b, actual_minutes, actual_pts, actual_reb, actual_ast
```

### BettingPros Scraper Notes
The scraper (`scrape_closing_lines.py`) handles several edge cases:
1. **Lazy loading**: Must toggle season dropdown (2025→2024→2025) to trigger game log load
2. **Date formats**: Oct-Dec games are /25, Jan+ games are /26
3. **Position suffixes**: Some players have position in URL (e.g., `alperen-sengun-c`, `jalen-johnson-f-f`)
4. **NL values**: "No Line" games captured as NaN (CatBoost handles natively)
5. **PRA URL order**: BettingPros uses `/points-assists-rebounds/` (PAR), NOT `/points-rebounds-assists/` (PRA). The wrong order redirects to `/points/` page!

### Usage
```bash
cd /Users/avish/Documents/prop.chat
source venv/bin/activate
python scrape_closing_lines.py  # Re-scrape if needed
```

## ML Model (CatBoost V2) - IMPLEMENTED

### Status: LIVE (Feb 25, 2026)

CatBoost V2 binary classifier is now integrated into production. V2 focuses on:
1. **Line Quality** - Is the line mispriced relative to player's stats?
2. **Matchup Edges** - How does player perform vs this specific opponent?

### Model Performance (V1 → V2)
| Metric | V1 | V2 (final) |
|--------|-----|------------|
| Test Accuracy | 57.1% | **60.7%** |
| Test Brier Score | 0.2292 | **0.2300** |
| ROI @ 55% threshold | +24.4% | **+28.0%** |
| ROI @ 60% threshold | - | **+43.5%** |
| Training rows | 15,253 | **15,156** |

### V2 Feature Importance
1. `avg_vs_opponent` - **42.0%** (player's historical avg vs opponent)
2. `minutes_avg` - 11.6%
3. `closing_line` - 6.7%
4. `prop_type` - 5.9% (reduced from 63% in V1!)
5. `games_played` - 5.1%
6. `season_avg` - 4.9%
7. `last_10_avg` / `last_5_avg` - 4.4% / 3.9%
8. `opponent_team` - 3.7%

### V2 New Features
| Feature | Description | Source |
|---------|-------------|--------|
| `avg_vs_opponent` | Player's historical average vs this opponent | Game log |
| `opp_def_rating` | Opponent team defensive rating | NBA API |
| `opp_pace` | Opponent team pace | NBA API |
| `line_vs_last_5` | Line - last 5 game average | Computed |
| `line_difficulty` | Z-score of line vs season avg | Computed |
| `consistency` | Inverse coefficient of variation | Computed |

### Architecture
```
Live Prediction Request
         ↓
    Feature Extraction
    (from /analyze endpoint)
         ↓
    Compute avg_vs_opponent
    (from full game log)
         ↓
┌────────┴────────┐
↓                 ↓
Rule-Based       CatBoost V2
Model            Classifier
(existing)       (catboost_model_v2.cbm)
↓                 ↓
Both predictions saved to DB
         ↓
Frontend displays both
with agreement indicator
```

### Key Files
| File | Purpose |
|------|---------|
| `catboost_predictor.py` | Production inference (uses V2) |
| `catboost_model_v2.cbm` | V2 trained model |
| `feature_columns_v2.json` | V2 feature order |
| `train_catboost_v2.py` | V2 training script |
| `prepare_training_data_v2.py` | V2 data preparation |

### Training Pipeline (V2)
```bash
cd /Users/avish/Documents/Prop.chat
source venv/bin/activate

# 1. Prepare data with V2 features
python prepare_training_data_v2.py

# 2. Train CatBoost V2
python train_catboost_v2.py

# Model will be saved to catboost_model_v2.cbm
```

### Implementation Phases (All Complete)

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Data collection (15,453 backtest rows) |
| 2 | ✅ | Logistic regression baseline (55.6% accuracy) |
| 3 | ✅ | CatBoost training (57.1% accuracy) |
| 4 | ✅ | Calibration analysis (raw probs well-calibrated) |
| 5 | ✅ | Production integration (main.py + frontend) |
| 6 | ✅ | Stats tracking (`/stats` shows both models) |

### Notes
- Raw CatBoost probabilities are well-calibrated (ECE=0.02), no Platt scaling needed
- Model recommends betting when confidence >= 55%
- PRA (points+rebounds+assists) props have highest accuracy (88.9%)
- Frontend shows model agreement/disagreement indicator

### Rule-Based vs CatBoost ML Comparison

| Aspect | Rule-Based Model | CatBoost ML |
|--------|------------------|-------------|
| **Logic** | Explicit math formulas | Learned from 15,253 historical outcomes |
| **Projection** | Calculates expected stat value | Doesn't project - predicts over/under directly |
| **Odds usage** | Uses odds for EV calculation | **Ignores odds entirely** |
| **Features** | ~10 adjustment factors | 15 features (includes player/team patterns) |
| **Decision** | Pick = higher EV side | Pick = higher probability side |

**Rule-based formula:**
```
1. projection = 60% * last_10_avg + 40% * season_avg + adjustments
2. prob_over = 1 - CDF((line - projection) / std_dev)
3. EV = (prob * payout) - (1 - prob)
4. Pick = side with higher EV
```

**CatBoost approach:**
```
1. Input 15 features (player, opponent, prop_type, line, stats...)
2. Model outputs prob_over, prob_under directly
3. Pick = side with higher probability
```

**Key insight:** CatBoost learned patterns like "PRA props hit over frequently" regardless of the actual line value, which explains high confidence on PRA but mixed real-world results.

### Features to Add Before ML Implementation
High priority:
- [ ] Opening line (for line movement tracking)
- [x] Closing line - scraped from BettingPros (15,253 rows)
- [x] Minutes played (actual) - `actual_minutes` column
- [x] Days rest - `days_rest` column
- [x] Back-to-back flag - `is_b2b` column

Medium priority:
- [ ] Player injury status
- [x] Usage rate - `usage_rate` column (now tracked)
- [ ] Historical performance vs opponent
- [ ] Defense vs position (PPG allowed to PG/SG/SF/PF/C)

**Already Tracking (as of Feb 8, 2026):**
- opponent_team, is_home, vegas_total, spread
- season_avg, last_10_avg, std_dev, minutes_avg
- opp_def_rating, opp_pace
- prob_over, no_vig_prob
- home_avg, away_avg, model_projection

### Models Considered but Not Chosen

| Model | Reason Not Primary |
|-------|-------------------|
| XGBoost | Requires manual categorical encoding |
| LightGBM | Slightly worse with categoricals |
| Neural Networks | Not enough data (<10,000) |
| Random Forest | 2-3% less accurate than boosting |

### Key Metrics to Track (Not Accuracy)
- **Calibration**: Predicted probability vs actual hit rate
- **Brier Score**: Probability accuracy
- **ROI by confidence bucket**: Are high-confidence plays profitable?
- **CLV (Closing Line Value)**: Did we beat the closing line?

## Future Feature: Teammate Injury Impact

### The Problem
When a star player is injured, teammates' stats change:
- LeBron OUT → Austin Reaves gets more usage, assists, points
- Need to capture this "opportunity boost" for better projections

### Solution: PBPStats WOWY API (Free)

**Source:** [PBPStats API](https://api.pbpstats.com/docs) | [Code Examples](https://github.com/dblackrun/pbpstats-api-code-examples)

**Endpoints:**
```
https://api.pbpstats.com/get-wowy-stats/nba
https://api.pbpstats.com/get-wowy-combination-stats/nba
```

**What it provides:**
- Player stats WITH specific teammate on floor
- Player stats WITHOUT specific teammate (teammate injured/benched)
- All on/off permutations for player groups
- No API key required, free access

**Example Usage:**
```python
import requests

# How does Reaves perform when LeBron is OUT?
params = {
    "0Exactly1OnFloor": "1630559",    # Austin Reaves
    "0Exactly1OffFloor": "2544",       # LeBron James OFF
    "TeamId": "1610612747",            # Lakers
    "Season": "2025-26",
    "SeasonType": "Regular Season",
    "Type": "Player"
}
response = requests.get("https://api.pbpstats.com/get-wowy-stats/nba", params=params)
# Returns: Reaves' PPG, APG, RPG when LeBron is not on floor
```

**Data to Store:**
```sql
CREATE TABLE teammate_boosts (
    player_id INTEGER,
    teammate_id INTEGER,
    team_id INTEGER,
    season TEXT,
    pts_boost REAL,      -- PPG change when teammate OUT
    ast_boost REAL,
    reb_boost REAL,
    minutes_boost REAL,
    sample_minutes REAL,  -- Minutes of data for reliability
    updated_at TEXT,
    PRIMARY KEY (player_id, teammate_id, season)
);
```

### Implementation Phases for Injury Impact

| Phase | Data Size | Action |
|-------|-----------|--------|
| **Now** | 64 | Document only - don't implement yet |
| **Phase 2** | 500 | Pre-calculate boost factors for top 50 player-teammate pairs |
| **Phase 3** | 1000 | Add `teammate_boost_pts/ast/reb` features to predictions |
| **Phase 4** | 2000 | Integrate with injury data (nbainjuries package) |

### Injury Data Source (For Later)

**Package:** `nbainjuries` ([GitHub](https://github.com/mxufc29/nbainjuries) | [PyPI](https://pypi.org/project/nbainjuries/))
- Pulls from official NBA injury reports
- Requires Java 8+ (uses tabula-py)
- Returns: player, team, status (Out/Questionable/Available), injury reason

```python
from nbainjuries import injury
from datetime import datetime

df = injury.get_reportdata(datetime.now(), return_df=True)
# Check if LeBron is OUT, then apply Reaves boost
```

### Current Observations (17 predictions, Mar 2, 2026)
- **Rule-based: 52.9%** (9/17) - above breakeven
- **CatBoost V2: 41.2%** (7/17) - underperforming on small sample
- **CLV tracking live**: Positive CLV 100% (1/1), Negative CLV 0% (0/3), Zero CLV 61.5% (8/13)
- **Average CLV: -0.12** - slightly negative, need to find more value
- **Test mode added**: API supports `{"query": "...", "test": true}` to skip DB saves

## Defense vs Position (DvP) - IMPLEMENTED

### Overview (Added Mar 21, 2026)
Position-specific defensive adjustments using FantasyPros DvP data. Replaces generic `opp_def_rating` with position-specific matchup adjustments.

**Why it matters:**
- Utah allows 27.1 PPG to PGs (worst) but only 20.9 PPG to Centers
- OKC allows 22.6 PPG to PGs (best) but 19.5 PPG to Centers (also best)
- A PG vs Utah gets +9% boost; same PG vs OKC gets -10% penalty

### Data Source
**FantasyPros**: https://www.fantasypros.com/daily-fantasy/nba/fanduel-defense-vs-position.php

Scrapes points, rebounds, assists allowed by each team to each position (PG, SG, SF, PF, C).

### Usage
```bash
python scrape_dvp.py           # Scrape and save to dvp_cache.json
python scrape_dvp.py --show    # View current cache with rankings
```

**Recommend refreshing daily before making predictions.**

### Adjustment Scale

| DvP Rank | Description | Adjustment |
|----------|-------------|------------|
| 1-5 | Worst defense (target) | +5% to +9% |
| 6-10 | Below average | +2% to +5% |
| 11-20 | Average | -2% to +2% |
| 21-25 | Above average | -5% to -2% |
| 26-30 | Best defense (avoid) | -10% to -5% |

### Integration in main.py

```python
from scrape_dvp import get_dvp_adjustment

# In calculate_minutes_based_projection():
dvp_multiplier = get_dvp_adjustment(opponent_abbr, player_position, prop_type)
matchup_multiplier *= dvp_multiplier
```

**Console output:**
```
[Analyze] DvP: PG vs UTA → +9.0% adj
```

### Current Rankings (Mar 21, 2026)

**Worst Defenses (target):**
| Position | Top 3 Teams |
|----------|-------------|
| PG | Utah (27.1), Orlando (26.9), Sacramento (26.8) |
| SG | New Orleans (24.1), Utah (23.5), Philadelphia (23.2) |
| SF | Utah (26.2), Lakers (25.9), Atlanta (25.0) |
| PF | Chicago (25.5), Washington (25.2), Indiana (24.3) |
| C | Washington (26.9), Dallas (24.6), Portland (24.4) |

**Best Defenses (avoid):**
| Position | Top 3 Teams |
|----------|-------------|
| PG | OKC (22.6), Boston (23.0), Brooklyn (23.4) |
| SG | San Antonio (20.1), Houston (20.2), Boston (20.2) |
| SF | Golden State (20.5), OKC (20.7), Boston (21.1) |
| PF | Clippers (20.4), Houston (20.5), Orlando (20.7) |
| C | OKC (19.5), Detroit (20.0), New York (20.2) |

### Files
| File | Purpose |
|------|---------|
| `scrape_dvp.py` | Playwright scraper for FantasyPros |
| `dvp_cache.json` | Cached DvP data with rankings |

### Position Normalization
NBA API returns positions like "Guard", "Forward-Center". Normalized to DvP format:

| NBA API | DvP |
|---------|-----|
| Guard, Point Guard | PG or SG |
| Forward, Small Forward | SF |
| Center | C |
| Guard-Forward | SF |
| Forward-Center | PF |

## CLV Tracking - IMPLEMENTED

### What is CLV?
**Closing Line Value** = Did you get a better line than the market close?
- Positive CLV: Line moved in your favor after you bet (good)
- Negative CLV: Line moved against you (bad)
- Zero CLV: Line didn't move

### Database Fields
```sql
closing_line  -- Line ~5 min before game (from Odds API historical)
clv           -- opening_line - closing_line (for unders) or closing - opening (for overs)
```

### CLV Fetcher Script (Added Mar 15, 2026)
```bash
python fetch_clv.py                # Fetch CLV for all resolved predictions missing closing_line
python fetch_clv.py 2026-03-14     # Fetch CLV for specific date
python fetch_clv.py --test         # Test with one prediction
```

**How it works:**
1. Uses The Odds API historical endpoint (requires $30/month plan)
2. Queries snapshots from ~5 min before game start
3. Finds player's closing line from bookmaker data
4. Calculates CLV based on recommended_side
5. Updates database with closing_line and clv

**API Credit Cost:** ~11 credits per prediction (1 events + 10 player props)

**Prop Type Mapping:**
| Our Type | Odds API Market |
|----------|-----------------|
| points | player_points |
| rebounds | player_rebounds |
| assists | player_assists |
| pra | player_points_rebounds_assists |
| pr | player_points_rebounds |
| pa | player_points_assists |
| ra | player_rebounds_assists |

### Daily Sync Script (Results Only)
```bash
python daily_sync.py              # Sync yesterday's results (no CLV)
python daily_sync.py 2026-02-28   # Sync specific date
```
Note: BettingPros scraping deprecated due to CAPTCHA blocking. Use fetch_clv.py for closing lines.

### CLV Results (228 predictions, Mar 18, 2026)
| CLV | Count | Hit Rate |
|-----|-------|----------|
| Positive | 38 | **60.5%** (23/38) |
| Zero | 148 | 54.1% (80/148) |
| Negative | 41 | **31.7%** (13/41) |

**Key Finding:** Positive CLV bets hit **1.9x more often** than negative CLV. Getting value matters.

## Minutes Projection Formula

### Current: 75/25 Formula (Mar 10, 2026)

```
Projected Minutes = (Season Avg × 0.75) + (Last 5 Avg × 0.25)
```

**Why 75/25?** Research baseline - more stable than reactive formulas, less susceptible to small sample noise.

### History
- **Mar 2, 2026**: Implemented 65/35 formula (65% season + 35% last 5)
- **Mar 9, 2026**: Added Razzball scraper for injury-adjusted minutes
- **Mar 10, 2026**: Removed Razzball (unreliable updates, player matching bugs), reverted to formula-based approach with 75/25 weights

### Future Enhancements (Not Yet Implemented)
| Feature | Description |
|---------|-------------|
| **Injury Adjustment** | When teammate OUT, redistribute their minutes to remaining players |
| **B2B Fatigue** | -10% penalty for veterans (30+ min avg) on back-to-backs |
| **Blowout Logic** | If spread >10, shift 3-5 min from starters to bench |
| **Zero-Sum Model** | Full 240-minute redistribution based on depth charts |

### Console Output
```
[Analyze] Projection: 26.3 → 26.5 (75/25 formula)
[Analyze] SPM: 0.833, Proj Min: 31.5, Trend: stable
[CatBoost] Using projected minutes: 31.5
```

### Files (Razzball - Deprecated)
The following files exist but are no longer used:
- `scrape_razzball.py` - Razzball scraper (deprecated)
- `razzball_minutes_cache.json` - Cache file (deprecated)
```bash
python prepare_training_data_v2.py  # Will use new minutes_avg values
python train_catboost_v2.py
```

## V3 Post-Mortem (Apr 8, 2026)

### What Happened
V3 training attempted to combine backtest data (14,824 rows) with live data (494 rows) and add new features (vegas_total, spread, usage_rate, player_position, dvp_prop_diff). The result was **catastrophic**: test accuracy dropped from 60.7% (V2) to 53.2% (V3).

### Root Causes

1. **Data Poisoning (96.8% of training data)**
   - Backtest data was missing V3 features, so they were filled with constants:
     - `vegas_total`: All 230.5 (median from live data)
     - `spread`: All 0.0
     - `dvp_prop_diff`: All 0.0
     - `usage_rate`: All ~0.20
   - Model learned that `opponent_team` perfectly correlates with these constants → 45% feature importance (vs 3.7% in V2)

2. **Removed Key Feature**
   - V2's `avg_vs_opponent` had **42% feature importance** - the strongest signal
   - V3 removed it, expecting `player_position` to substitute
   - Position is far less predictive than matchup history

3. **New Features Got 0% Importance**
   - `dvp_prop_diff`: 0.0% (constant for 96.8% of data)
   - `line_difficulty`: 0.0%
   - `usage_rate`: 0.2%

4. **Tiny Test/Val Sets**
   - Test: 94 samples (0.6%) - statistically unreliable
   - Val: 229 samples (1.5%) - couldn't detect overfitting

### V3 Files (Do Not Use)
- `catboost_model_v3.cbm` - broken model
- `training_data_v3.csv` - poisoned data
- `prepare_training_data_v3.py` - flawed pipeline
- `train_catboost_v3.py` - training script
- `catboost_predictor_v3.py` - has bugs (player_name mismatch)

### Lesson Learned
Never mix real data with fake/default-filled data. Better to train on 500 real samples than 15,000 poisoned samples.

---

## V2.1 Training Results (Apr 8, 2026)

### Status: LIVE

Successfully retrained CatBoost on **live data only** (512 predictions) with 5-fold stratified CV.

### Performance
| Metric | V2 (backtest) | V2.1 (live) |
|--------|--------------|-------------|
| Test/CV Accuracy | 60.7% | **56.1%** |
| CV Brier Score | - | 0.249 |
| Fold Accuracy Std | - | ±1.2% |
| Training Samples | 15,156 | **512** |
| Training Method | 80/10/10 split | **5-fold CV** |

### V2.1 Feature Importance
1. `closing_line` - **17.1%**
2. `avg_vs_opponent` - **17.1%**
3. `last_10_avg` - 8.2%
4. `player_position` - 7.7%
5. `opp_def_rating` - 7.5%
6. `dvp_rank` - 7.2%
7. `season_avg` - 5.5%
8. `line_difficulty` - 5.2%
9. `dvp_allowed` - 4.7%
10. `line_vs_last_5` - 4.1%

### V2 vs V2.1 Comparison

| Feature | V2 (backtest) | V2.1 (live) | Change |
|---------|--------------|-------------|--------|
| `avg_vs_opponent` | **42.0%** | 17.1% | -24.9% (still top-2) |
| `minutes_avg` | 11.6% | 0.8% | -10.8% |
| `closing_line` | 6.7% | **17.1%** | +10.4% |
| `opponent_team` | 3.7% | **0.3%** | -3.4% (no overfitting!) |
| `player_position` | - | 7.7% | NEW |
| `dvp_rank` | - | 7.2% | NEW |
| `dvp_allowed` | - | 4.7% | NEW |
| **DvP total** | - | **~12%** | NEW category |

### Key Insights

1. **No overfitting to opponent_team**: Dropped from 3.7% to 0.3% - model isn't memorizing team-specific patterns that don't generalize

2. **Closing line gained importance**: Now #1 feature (17.1%) - when the line is set higher than stats suggest, the over tends to miss (and vice versa)

3. **DvP features contributing**: Combined ~12% importance - position-specific matchups matter

4. **avg_vs_opponent still critical**: While it dropped from 42% to 17%, it's still tied for #1 - matchup history remains predictive

5. **More balanced feature distribution**: No single feature dominates - healthier for generalization

### Prop Type Accuracy (CV)
| Prop Type | Samples | Accuracy |
|-----------|---------|----------|
| ra | 67 | **61.2%** |
| rebounds | 41 | 61.0% |
| pa | 32 | 62.5% |
| pr | 49 | 59.2% |
| points | 181 | 55.8% |
| assists | 44 | 52.3% |
| pra | 98 | 49.0% |

### ROI Simulation (52%+ confidence)
- **17 qualifying bets** at 52%+ threshold
- **58.8% hit rate** (10/17)
- **+12.3% ROI** (at -110 odds)

### Files
| File | Purpose |
|------|---------|
| `catboost_model_v2_1.cbm` | V2.1 trained model |
| `feature_columns_v2_1.json` | V2.1 feature configuration |
| `training_metrics_v2_1.json` | Training metrics & fold details |
| `train_catboost_v2_1.py` | V2.1 training script |
| `backfill_avg_vs_opponent.py` | Backfill script for matchup data |
| `backfill_dvp_features.py` | Backfill script for DvP data |

### V2.1 Feature Set
```
Categorical (3): opponent_team, prop_type, player_position

Numeric (17):
- Core: closing_line, season_avg, last_10_avg, std_dev, minutes_avg
- Context: is_home, is_b2b, days_rest
- Matchup: opp_def_rating, opp_pace, avg_vs_opponent
- Derived: line_vs_season, line_vs_last_5, line_difficulty, consistency
- DvP: dvp_rank, dvp_allowed
```

---

## V2.2 Training Results (Apr 15, 2026)

### Status: LIVE

Trained CatBoost on **enriched backtest data** (15,453 rows) with all V2.1 features backfilled. This fixes the massive prop-type biases in V2.1 (which was trained on only 512 live predictions).

### The Problem V2.2 Solved

V2.1 had severe prop-type biases due to small sample size:
| Prop Type | V2.1 Over Pick % | V2.2 Over Pick % | Actual Over % |
|-----------|------------------|------------------|---------------|
| pra | **96%** | 52% | 50% |
| rebounds | **0%** | 55% | 50% |
| points | 57% | 51% | 50% |
| assists | 60% | 54% | 50% |

With only 512 training rows, each prop type had 32-98 samples, leading to spurious correlations. V2.2 uses 30x more data with proper feature coverage.

### Performance
| Metric | V2.1 (live) | V2.2 (enriched backtest) |
|--------|-------------|--------------------------|
| CV Accuracy | 56.1% | **59.7%** |
| CV Brier Score | 0.249 | **0.237** |
| Training Samples | 512 | **15,453** |
| Over-pick Bias | 0-96% by prop | **51-55% all props** |
| ROI @ 55% conf | +12.3% | **+27.2%** |
| ROI @ 60% conf | - | **+41.9%** |

### V2.2 Feature Importance (Balanced)
1. `minutes_avg` - **18.2%** (playing time drives stats)
2. `opponent_team` - 8.5%
3. `opp_def_rating` - 7.9%
4. `dvp_rank` - 7.7%
5. `line_vs_last_5` - 7.6%
6. `days_rest` - 7.2%
7. `line_difficulty` - 5.8%
8. `opp_pace` - 5.7%
9. `line_vs_season` - 5.0%
10. `avg_vs_opponent` - 4.8%
11. `player_position` - 4.4%
12. `is_home` - 4.3%
13. `consistency` - 1.9%
14. `closing_line` - 1.9%
15. `last_10_avg` - 1.9%
16. `season_avg` - 1.8%
17. `dvp_allowed` - 1.6%
18. `std_dev` - 1.4%
19. `prop_type` - **1.3%** (was 63% in V1!)
20. `is_b2b` - 1.3%

**Key insight:** No single feature dominates (V1 had prop_type at 63%, V2 had avg_vs_opponent at 42%). Balanced importance = better generalization.

### Prop Type Accuracy (All Balanced)
| Prop Type | Samples | Accuracy | Over Pick % |
|-----------|---------|----------|-------------|
| pra | 2,208 | **63.9%** | 52.0% |
| pa | 2,208 | **63.2%** | 54.8% |
| pr | 2,208 | **62.3%** | 54.8% |
| points | 2,208 | 60.1% | 51.0% |
| ra | 2,205 | 59.0% | 53.7% |
| rebounds | 2,208 | 55.3% | 54.8% |
| assists | 2,208 | 54.0% | 53.8% |

### ROI Simulation by Confidence Threshold
| Threshold | Bets | Hit Rate | ROI |
|-----------|------|----------|-----|
| 52% | 11,957 | 62.3% | +19.0% |
| 55% | 7,349 | 66.6% | +27.2% |
| 58% | 3,888 | 70.4% | +34.4% |
| 60% | 2,417 | 74.3% | +41.9% |

### Backfill Scripts Used
| Script | Purpose | Coverage |
|--------|---------|----------|
| `backfill_backtest_avg_vs_opponent.py` | Fetch game logs, compute historical avg vs opponent | 92.7% (14,326 rows) |
| `backfill_backtest_dvp.py` | Position + DvP from dvp.xlsx | 100% |
| `backfill_backtest_team_stats.py` | opp_def_rating, opp_pace from NBA API | 100% |

### Files
| File | Purpose |
|------|---------|
| `catboost_model_v2_2.cbm` | V2.2 trained model (CURRENT) |
| `feature_columns_v2_2.json` | V2.2 feature configuration |
| `training_metrics_v2_2.json` | Full metrics + fold details |
| `train_catboost_v2_2.py` | V2.2 training script |
| `backtest_data_enriched.csv` | Enriched training data (15,453 rows) |

---

## V2.3 Roadmap: CLV Integration + Lineup Awareness

V2.3 has two major components:
1. **CLV Integration** - Use closing line value as a feature (when 1,000+ samples)
2. **Lineup Awareness** - Adjust projections when key players are injured/out

---

### Component 1: CLV Integration

#### Plan
When we reach **1,000+ live predictions with CLV data**, train V2.3 with CLV as a feature.

#### Why CLV Matters
From Mar 2026 analysis (228 predictions):
| CLV | Hit Rate |
|-----|----------|
| Positive | **60.5%** |
| Zero | 54.1% |
| Negative | **31.7%** |

Positive CLV bets hit **1.9x more often** than negative CLV. This is a strong signal that the current model isn't capturing.

#### Why Not Now?
- Backtest data lacks `opening_line` - can't calculate CLV retroactively
- Live predictions have CLV but only ~600 samples
- Need 1,000+ for reliable cross-validation

#### CLV Features to Add
- `clv` - Closing line value (opening_line - closing_line, adjusted for side)
- `line_movement` - Absolute line movement (|opening - closing|)
- `sharp_action` - Boolean if line moved >0.5 points

#### Timeline
- **Current:** ~680 live predictions with CLV
- **Target:** 1,000+ predictions
- **Estimated:** ~4-6 weeks at current pace

---

### Component 2: Lineup Awareness (Injury Adjustments)

#### Industry Standard Approach (Apr 2026 Research)
The industry standard for injury adjustments uses **projection adjustment**, not line adjustment:

```
Injury → Boost projection → Compare to line    ✓ Industry standard
Injury → Adjust line down → Keep projection    ✗ Not standard
```

#### Why Projection Adjustment?
- Matches how DFS/fantasy projection systems work
- More intuitive to calibrate ("player scores MORE" vs "line should be LOWER")
- Clean tracking: compare projected vs actual
- Per-minute rates stay stable; boost comes from MORE minutes/usage

#### The Formula (Industry Standard)
```
projected_stat = (per_minute_rate) × (base_minutes + injury_minutes_boost) × (1 + usage_boost)
```

Where:
- `per_minute_rate` = historical stat/minute (from rolling average)
- `base_minutes` = normal projected minutes
- `injury_minutes_boost` = additional minutes from teammate being out
- `usage_boost` = % increase in touches/opportunities

#### Architecture
```
┌─────────────────────────────────────────────────┐
│  BASE MODEL (V2.2 CatBoost)                     │
│  Outputs: base_projection for each stat         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  INJURY ADJUSTMENT LAYER                        │
│  1. Check injury report                         │
│  2. Look up WOWY minutes boost                  │
│  3. Look up WOWY usage boost                    │
│  4. adjusted = base × (1 + combined_boost)      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  COMPARE TO LINE                                │
│  If adjusted_projection > line → OVER           │
│  If adjusted_projection < line → UNDER          │
└─────────────────────────────────────────────────┘
```

#### Boost Application Per Prop Type
| Prop Type | Boost Applied |
|-----------|---------------|
| Points | `pts_boost` |
| Rebounds | `reb_boost` |
| Assists | `ast_boost` |
| PTS+REB | `pts_boost + reb_boost` |
| PTS+AST | `pts_boost + ast_boost` |
| REB+AST | `reb_boost + ast_boost` |
| PTS+REB+AST | `pts_boost + reb_boost + ast_boost` |

#### Data Sources Required
1. **Injury Reports** - NBA official injury reports (via nba_api or rotowire)
2. **WOWY Data** - PBPStats API (free, no API key needed, already documented below)
3. **Historical Lineups** - Track who played in each game

#### Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Data Infrastructure: Injury table, WOWY pipeline, lineup tracking | Not started |
| **2** | Feature Engineering: Pre-compute boost factors per player-teammate pair | Not started |
| **3** | Model Integration: Add adjustment layer to prediction pipeline | Not started |
| **4** | Calibration: Track injury-adjusted predictions, tune boost multiplier | Not started |

#### Phase 1: Data Infrastructure
- [ ] Add `injuries` table: `player_id`, `player_name`, `team`, `status`, `injury_type`, `report_date`
- [ ] Add `wowy_splits` table: `player_id`, `teammate_id`, `stat_type`, `with_value`, `without_value`, `sample_size`
- [ ] Add `daily_lineups` table: `game_date`, `team`, `player_id`, `is_starter`, `is_out`
- [ ] Integrate PBPStats WOWY API (see API documentation below)

#### Phase 2: Feature Engineering
- [ ] For each player, calculate boost when top 3 teammates are out
- [ ] Create `teammate_boost_lookup` table: `player_id`, `missing_player_id`, `pts_boost`, `reb_boost`, `ast_boost`
- [ ] Set minimum sample size threshold (>15 games without teammate)
- [ ] Cap boosts at reasonable levels (+/- 5 pts max per stat)

#### Phase 3: Model Integration
- [ ] Before making predictions, check injury report
- [ ] If key player out → fetch WOWY boost → apply to base projection
- [ ] Log which injuries affected each prediction

#### Phase 4: Calibration Loop
Add tracking columns to predictions table:
```sql
injury_boost_applied    -- The boost applied (e.g., +3.5 pts)
missing_player_id       -- Who was out
wowy_sample_size        -- Sample size for the boost
```

Calibration query (run after 100+ injury-adjusted predictions):
```sql
SELECT
    CASE
        WHEN injury_boost_applied > 3 THEN 'high_boost'
        WHEN injury_boost_applied > 1 THEN 'med_boost'
        ELSE 'low_boost'
    END as boost_tier,
    COUNT(*) as n,
    AVG(hit) as hit_rate
FROM predictions
WHERE injury_boost_applied IS NOT NULL
GROUP BY boost_tier
```

Adjust `BOOST_MULTIPLIER` based on results:
- If high_boost predictions hit <52% → reduce multiplier (e.g., 0.7)
- If high_boost predictions hit >58% → increase multiplier (e.g., 1.2)

#### Guardrails
- Only apply boosts when WOWY sample size > 15 games
- Cap individual stat boosts at +/- 5
- Lower confidence on injury-adjusted predictions (market still adjusting)
- Skip adjustment if injury news < 2 hours old

#### Why No Backtesting?
- No historical injury data collected
- Can't validate retroactively
- **Strategy:** Deploy with conservative guardrails, calibrate prospectively
- Industry standard WOWY math is directionally correct even if calibration isn't perfect

---

## MLB Strikeout Simulation (Tier 3) - IN PROGRESS

### Overview (Apr 8, 2026)
Building a Monte Carlo simulation for MLB strikeout props. Instead of team-level projections, simulate each plate appearance individually using confirmed lineups and batter-specific K rates.

**Current system:** Beta-Binomial projection with team K% (54.4% accuracy)
**Target system:** PA-level Monte Carlo simulation (57-60% accuracy)

### Why Simulation > Current Approach
| Current (Team-Level) | Simulation (Batter-Level) |
|---------------------|---------------------------|
| "Yankees K at 22% as a team" | "Judge Ks at 31%, Soto at 19%" |
| Assumes average lineup | Uses **actual confirmed lineup** |
| One expected value | **Full probability distribution** |
| Fixed batters faced | **Dynamic BF based on game flow** |

### Data Inventory

**Already Have:**
- Pitcher K%, SwStr%, CSW%, BB% (FanGraphs/pybaseball)
- Team K% vs LHP/RHP (FanGraphs/pybaseball)
- Park K factors (Baseball Savant)
- Game schedules (MLB Stats API)

**Need to Add:**
- ~~Confirmed lineups (MLB Stats API)~~ ✅ Done
- ~~Batter K% vs LHP/RHP (pybaseball)~~ ✅ Done (via Baseball Reference)
- ~~Batter BB% vs LHP/RHP (pybaseball)~~ ✅ Done
- Pitcher fatigue model
- Pitcher pull probability model

### Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Batter data + lineup fetcher | ✅ Complete |
| **2** | Core simulation engine (Log5, PA simulator, fatigue, pull model) | ✅ Complete |
| **3** | API integration + soft replacement | ✅ Complete |
| **4** | Accuracy tracking + comparison vs Beta-Binomial | ✅ Complete |

### Phase 1: Batter Data + Lineup Fetcher ✅

**Files created (Apr 8, 2026):**
```
mlb_simulation/
├── __init__.py         # Package exports
├── batter_data.py      # Batter K%/BB% from Baseball Reference (FanGraphs 403'd)
└── lineup_fetcher.py   # Confirmed lineups from MLB Stats API
```

**Features:**
- Batter cache: 460 batters (2025) + 41 batters (2026 early season)
- 2025 fallback for players without enough 2026 PA
- League K%: 21.9% (2025), 21.7% (2026)
- Lineup fetcher: Gets confirmed lineups for all games
- Platoon splits estimated from overall K% (no handedness in BR data)

**Usage:**
```python
from mlb_simulation import get_batter, get_opponent_lineup, get_batter_k_rate

# Get batter K rate vs pitcher hand
k_rate = get_batter_k_rate("Aaron Judge", "R")  # 23.7%

# Get lineup facing a pitcher's team
opponent = get_opponent_lineup("NYY")
# Returns: lineup, team, pitcher, confirmed status
```

**Batter cache structure:**
```json
{
  "aaron judge": {
    "name": "Aaron Judge", "team": "NYY", "bats": "R",
    "k_pct": 23.2, "bb_pct": 15.4,
    "k_pct_vs_lhp": 23.7, "k_pct_vs_rhp": 23.7,
    "pa": 710, "so": 165, "bb": 109
  }
}
```

### Phase 2: Simulation Engine ✅

**File created:** `mlb_simulation/simulator.py` (350 lines)

**Components implemented:**
- `log5_probability()` - Matchup probability combining batter + pitcher rates
- `get_fatigue_multiplier()` - K% drops 3-12% as pitch count rises
- `should_pull_pitcher()` - Pull probability based on pitch count, inning, runs
- `simulate_pa()` - Single plate appearance → K/BB/HIT/OUT
- `simulate_game()` - Full game simulation with game state tracking
- `run_simulation()` - Monte Carlo loop (3000 sims) → probability distribution
- `simulate_strikeouts()` - High-level API integrating all data sources

**Tuned parameters (matching 2024 MLB averages):**
- Avg IP: 5.9-6.1 (MLB: 5.4)
- Avg Pitches: 86-88 (MLB: 85-88)
- Pull threshold: 70 pitches base, aggressive after 85

**Sample output:**
```python
from mlb_simulation import simulate_strikeouts

result = simulate_strikeouts('Tarik Skubal', 'DET', n_sims=3000)
# Mean Ks: 7.4, Avg IP: 6.1
# Over 5.5: 80.5%, Over 6.5: 64.9%, Over 7.5: 45.9%
```

**Log5 formula:**
```
P(K) = (Batter_K% × Pitcher_K% / League_K%) /
       (Batter_K% × Pitcher_K% / League_K% + (1-Batter_K%) × (1-Pitcher_K%) / (1-League_K%))
```

### Phase 3: API Integration ✅

**Soft replacement implemented (Apr 11, 2026):**
- Simulation is primary model when lineups are confirmed
- Beta-Binomial fallback when lineups not posted (~2-3 hours before games)
- Frontend shows model type badge: "Monte Carlo" (purple) or "Beta-Binomial" (yellow)
- Both predictions logged for accuracy comparison

**Response includes:**
```json
{
  "model_type": "simulation",  // or "beta_binomial"
  "lineup_confirmed": true,
  "sim_stats": {"mean_k": 7.4, "std_k": 2.3, "avg_ip": 5.9},
  "lineup": [{"name": "Judge", "k_pct": 29.5}, ...]
}
```

### Phase 4: Accuracy Tracking ✅

**Database columns added (Apr 11, 2026):**
```sql
ALTER TABLE mlb_predictions ADD COLUMN model_type TEXT DEFAULT 'beta_binomial';
ALTER TABLE mlb_predictions ADD COLUMN lineup_confirmed INTEGER DEFAULT 0;
ALTER TABLE mlb_predictions ADD COLUMN sim_mean_k REAL;
ALTER TABLE mlb_predictions ADD COLUMN sim_std_k REAL;
ALTER TABLE mlb_predictions ADD COLUMN sim_avg_ip REAL;
ALTER TABLE mlb_predictions ADD COLUMN sim_avg_pitches REAL;
ALTER TABLE mlb_predictions ADD COLUMN lineup_avg_k_pct REAL;
```

**Tracking fields:**
| Field | Description |
|-------|-------------|
| `model_type` | "simulation" or "beta_binomial" |
| `lineup_confirmed` | 1 if lineup was confirmed at prediction time |
| `sim_mean_k` | Mean Ks from Monte Carlo simulation |
| `sim_std_k` | Std dev of Ks from simulation |
| `sim_avg_ip` | Average IP from simulations |
| `sim_avg_pitches` | Average pitch count from simulations |
| `lineup_avg_k_pct` | Average K% of opposing lineup |

**Query for model comparison:**
```sql
-- Compare hit rates by model type
SELECT
    model_type,
    COUNT(*) as total,
    SUM(hit) as hits,
    ROUND(100.0 * SUM(hit) / COUNT(*), 1) as hit_pct
FROM mlb_predictions
WHERE status = 'resolved'
GROUP BY model_type;
```

### Fatigue & Pull Models (Recalibrated Apr 12, 2026)

**Fatigue multiplier:**
| Pitch Count | K% Multiplier |
|-------------|---------------|
| 0-75 | 1.00 (no fatigue) |
| 75-90 | 0.97 (-3%) |
| 90-100 | 0.93 (-7%) |
| 100+ | 0.88 (-12%) |

**Pull probability factors (calibrated for 5.4 IP, 21-22 BF, ~80 pitches):**
- Base threshold: 65 pitches (start considering pull)
- +8% per pitch over 65
- +28% per run allowed in current inning
- +45% if down 5+ runs (blowout)
- +18% at end of each inning after 5th
- Hard pull at 90+ pitches (65%+ probability)

**Times Through Order (TTO) penalty:**
| Time Through | Batters | K% Multiplier |
|--------------|---------|---------------|
| 1st | 1-9 | 1.00 (baseline) |
| 2nd | 10-18 | 0.95 (-5%) |
| 3rd | 19-27 | 0.88 (-12%) |
| 4th+ | 28+ | 0.85 (-15%) |

**Bad start variance:**
- 10% of simulations model disaster outings (40-65 pitch early hook)
- Adds realistic variance for short starts

**Ace pitcher longer leash:**
- Pitchers with K% ≥ 27% get "ace" treatment
- +8 pitch bonus before pull consideration
- 50% lower bad start probability
- Results in ~0.7 more IP, ~2 more BF for aces

### Calibration Results (Apr 12, 2026)

| Metric | Before | After | Target | Status |
|--------|--------|-------|--------|--------|
| BF/Start | 25.4 | 22.8 | 21.5 | ✓ +6% |
| IP/Start | 5.91 | 5.25 | 5.4 | ✓ -3% |
| K/Start | 6.1 | 5.2 | 5.1 | ✓ +2% |
| Pitches | 87 | 78 | 85 | △ -8% |

**Net effect:** Reduced over-prediction by ~1 full strikeout on average.

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Overall accuracy | 54.4% | 57-60% |
| Over accuracy | 52.6% | 55%+ |
| Under accuracy | 54.8% | 58%+ |

### Future Optimization Roadmap

**Phase 1: Validation (50-100 predictions)** ← CURRENT
- Compare Simulation vs Beta-Binomial accuracy
- Calibration check - does 70% probability actually hit 70%?
- Identify systematic biases (over/under projecting certain pitcher types)
- Track bias (projected - actual) and MAE

**Phase 2: Parameter Tuning (200+ predictions)**
| Parameter | Current | Optimize |
|-----------|---------|----------|
| Fatigue thresholds | 75/90/100 pitches | Tune based on actual K% drop-off data |
| Pull probability | 65 pitch base | Adjust based on actual pull rates by team |
| TTO penalty | -5%/-12% | Validate against real TTO data |
| Ace threshold | 27% K | May need adjustment based on results |

**Phase 3: Feature Additions (300+ predictions)**
- Umpire K% tendencies (some umps have wider zones)
- Weather/temperature effects
- Day/night splits
- Rest days impact
- Batter vs Pitcher H2H history (if sample exists)
- 2-strike put-away rate (Statcast data available)

**Phase 4: ML Enhancement (500+ predictions)**
- Use simulation output as features for classifier
- Ensemble model: Simulation + Beta-Binomial + ML
- Learn when simulation over/under projects

**Phase 5: Advanced (1000+ predictions)**
- Real platoon splits (currently estimated)
- Pitch mix modeling (different pitch types have different K rates)
- Pitch-level simulation (whiff rates by pitch type)

**Key tracking query:**
```sql
SELECT
    model_type,
    COUNT(*) as n,
    ROUND(100.0 * SUM(hit) / COUNT(*), 1) as hit_rate,
    ROUND(AVG(ABS(projected_ks - actual_ks)), 2) as mae
FROM mlb_predictions
WHERE status = 'resolved'
GROUP BY model_type;
```

### Research Sources
- [FullCountProps Methodology](https://www.fullcountprops.com/methodology) - LightGBM + 5,000 sims
- [KSplit Analytics](https://ksplitanalytics.com/) - PA-level modeling
- [Ballpark Pal](https://www.ballparkpal.com/FAQ.php) - 3,000 sims, 100+ features
- [SABR Log5](https://sabr.org/journal/article/matchup-probabilities-in-major-league-baseball/) - Matchup probability formula

### Simulation V2 Upgrade (Apr 23, 2026)

Based on industry research, implementing 9-step upgrade to bring simulation to industry standard.

**Problem identified:** Systematic bias in past week (Apr 15-22):
- OVERS: Projected 5.32 Ks, Actual 4.40 → Over-projecting by 0.92 Ks
- UNDERS: Projected 4.80 Ks, Actual 5.70 → Under-projecting by 0.91 Ks

**9-Step Upgrade Plan:**

| Step | Change | Status |
|------|--------|--------|
| 1 | Per-game "stuff" variance (±12%) | ✅ Complete |
| 2 | Strengthen TTO penalties (3rd → 0.82) | ✅ Complete |
| 3 | Increase bad start probability (10% → 18%) | ✅ Complete |
| 4 | Add "hot start" modeling (+18% K boost) | ✅ Complete |
| 5 | Strengthen fatigue penalties | ✅ Complete |
| 6 | Add Log5 regression to mean | ✅ Complete |
| 7 | Model workload as distribution | ✅ Complete |
| 8 | Refine pull probability curve | ✅ Complete |
| 9 | Increase sims + add calibration tracking | ✅ Complete |

#### Step 1: Per-Game Stuff Variance ✅

**Implemented (Apr 23, 2026):**

Added per-simulation "stuff" modifier that captures good/bad days:
- Gaussian distribution centered at 1.0
- Standard deviation: 12%
- Capped at 0.70-1.30 (±30%)
- Applied to pitcher K rate for entire game

**New constants:**
```python
STUFF_VARIANCE_STD = 0.12       # ~12% standard deviation
STUFF_MODIFIER_MIN = 0.70       # Floor: -30% K rate
STUFF_MODIFIER_MAX = 1.30       # Ceiling: +30% K rate
```

**New function:**
```python
def get_stuff_modifier() -> float:
    """Generate per-game stuff modifier (0.70-1.30)."""
    modifier = random.gauss(1.0, STUFF_VARIANCE_STD)
    return max(STUFF_MODIFIER_MIN, min(STUFF_MODIFIER_MAX, modifier))
```

**Impact:**
- Captures "stuff working" days (pitchers like Bailey Ober getting 10+ Ks)
- Captures "off days" (pitchers like Logan Gilbert getting 3 Ks despite 7+ projection)
- Widens K distribution to match real-world variance
- Results returned include `stuff_variance` stats for tracking

#### Step 2: Strengthen TTO Penalties ✅

**Implemented (Apr 23, 2026):**

Updated Times Through Order multipliers to match research (OPS+ 91→117, ERA 4.08→4.57 by 3rd time):

| Time Through | Old | New | Change |
|--------------|-----|-----|--------|
| 1st (batters 1-9) | 1.00 | 1.00 | - |
| 2nd (batters 10-18) | 0.95 | **0.93** | -2% more penalty |
| 3rd (batters 19-27) | 0.88 | **0.82** | -6% more penalty |
| 4th (batters 28+) | 0.85 | **0.78** | -7% more penalty |

**Research basis:**
- OPS+ increases from 91→117 (28%) by third time through
- ERA increases from 4.08→4.57 (12%) by third time through
- Sources: Baseball Prospectus, MLB.com TTO research

**Impact:**
- Reduces K over-projection in late innings
- Especially affects pitchers who go deep (6+ IP, 24+ BF)
- More realistic K rate decline as batters see pitcher multiple times

#### Step 3: Increase Bad Start Probability ✅

**Implemented (Apr 23, 2026):**

Updated disaster outing parameters to capture early exits:

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `BAD_START_PROBABILITY` | 10% | **18%** | Only 36% of starts are quality starts |
| `BAD_START_MAX_PITCHES` | 65 | **70** | Range now 40-70 pitches |
| `ACE_BAD_START_REDUCTION` | 50% | **60%** | Aces still get pulled sometimes |

**Research basis:**
- Only 36% of starts result in quality starts (6+ IP)
- Early exits like Jacob Lopez (1.0 IP), Jesus Luzardo (4.2 IP) were under-represented
- Regular pitchers: 18% bad start chance
- Ace pitchers (K% ≥27%): 10.8% bad start chance (18% × 60%)

**Test results:**
```
Distribution of 0-3 K games:
  0 Ks: Regular 0.6% | Ace 0.2%
  1 Ks: Regular 3.2% | Ace 1.2%
  2 Ks: Regular 9.5% | Ace 3.7%
  3 Ks: Regular 14.9% | Ace 6.9%
```

**Impact:**
- More simulations capture early exits (1-4 IP)
- Reduces over-projection on OVER picks
- Regular pitcher avg IP: 5.07, Ace avg IP: 5.88

#### Step 4: Hot Start Modeling ✅

**Implemented (Apr 23, 2026):**

Added "hot start" modeling to capture exceptional performances:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `HOT_START_PROBABILITY` | 12% | Chance of exceptional performance |
| `HOT_START_K_BOOST` | 1.18 | +18% K rate when stuff is ON |
| `HOT_START_PULL_DELAY` | 10 | +10 pitches before pull consideration |
| `ACE_HOT_START_BOOST` | 1.15 | Additional +15% for aces (multiplicative) |

**Research basis:**
- Captures games like Bailey Ober (10 Ks), Gavin Williams (11 Ks)
- Balances bad start variance with upside variance
- Aces on hot starts can reach 1.36x K rate (1.18 × 1.15)

**Impact:**
- Better distribution of 8+ K games
- Captures exceptional performances that were under-represented
- Hot starts extend pitch count by ~10 pitches

#### Step 5: Strengthen Fatigue Penalties ✅

**Implemented (Apr 23, 2026):**

Updated fatigue thresholds to match research showing steeper K% decline with pitch count:

| Pitch Count | Old | New | Change |
|-------------|-----|-----|--------|
| 0-70 | 1.00 (was 75) | 1.00 | Earlier threshold |
| 70-80 | 1.00 | **0.96** | -4% (new tier) |
| 80-90 | 0.97 | **0.92** | -5% more penalty |
| 90-100 | 0.93 | **0.87** | -6% more penalty |
| 100+ | 0.88 | **0.82** | -6% more penalty |

**Research basis:**
- Baseball Savant pitch-by-pitch data shows K rate drops accelerate after 70 pitches
- Dramatic decline past 90 pitches as pitcher effectiveness drops
- Total decline at 100+ pitches: -18% (was -12%)

**Impact:**
- Reduces K over-projection in high pitch count scenarios
- More realistic late-inning K rate decline
- Combined with TTO penalty, captures compounding fatigue effect

#### Step 6: Add Log5 Regression to Mean ✅

**Implemented (Apr 23, 2026):**

Added regression to mean for extreme K rates before applying Log5 formula:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `REGRESSION_FACTOR` | 0.20 | Pull 20% toward league average |
| `REGRESSION_THRESHOLD` | 0.05 | Only regress rates >5% from league avg |

**How it works:**
```python
def regress_to_mean(rate, league_rate):
    if abs(rate - league_rate) < 0.05:
        return rate  # Don't regress near-average rates
    return rate + (league_rate - rate) * 0.20
```

**Example regressions (league avg = 22.7%):**
- 15% K batter → 16.5% (+1.5%)
- 28% K pitcher → 26.9% (-1.1%)
- 30% K pitcher → 28.5% (-1.5%)
- 35% K batter → 32.5% (-2.5%)

**Research basis:**
- Standard Log5 has known biases for asymmetric probabilities
- High-K pitcher vs high-K batter matchups were over-projected
- SABR and Tom Tango research on Log5 corrections

**Impact:**
- 30% pitcher vs 30% batter: 38.5% → 35.2% K prob (-3.3%)
- Regular pitcher: 5.4 → 5.3 mean Ks (-0.1)
- Ace pitcher: 7.2 → 6.8 mean Ks (-0.4)
- Larger reduction for high-K pitchers where bias was greatest

#### Step 7: Model Workload as Distribution ✅

**Implemented (Apr 23, 2026):**

Added per-simulation workload variance to capture day-to-day variance in how deep pitchers go:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `WORKLOAD_MEAN` | 0 | Centered at baseline |
| `WORKLOAD_STD` | 8 | ±8 pitches standard deviation |
| `WORKLOAD_MIN` | -15 | Maximum early pull |
| `WORKLOAD_MAX` | +15 | Maximum extended leash |

**How it works:**
```python
def get_workload_modifier() -> int:
    """Generate per-game workload adjustment (-15 to +15 pitches)."""
    modifier = int(random.gauss(0, 8))
    return max(-15, min(15, modifier))
```

**What it captures:**
- Bullpen availability (tired pen = pitcher goes deeper)
- Manager tendencies (some pull earlier than others)
- Game situation variance (blowouts vs close games)
- Day-to-day randomness in team decisions

**Impact:**
- Adds realistic variance to IP/pitches outcomes
- Some games: early hook (-15 pitches from normal)
- Some games: extended outing (+15 pitches from normal)
- Results now include `workload_variance` stats for tracking

#### Step 8: Refine Pull Probability Curve ✅

**Implemented (Apr 23, 2026):**

Replaced linear pull probability with smooth sigmoid curve:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `PULL_SIGMOID_MIDPOINT` | 82 | 50% pull probability at 82 effective pitches |
| `PULL_SIGMOID_STEEPNESS` | 6 | Steeper curve (faster transition) |

**Formula:**
```python
def sigmoid_pull_probability(pitch_count, midpoint=82, steepness=6):
    return 1.0 / (1.0 + math.exp(-(pitch_count - midpoint) / steepness))
```

**Pull probability at various pitch counts:**
| Pitches | Probability |
|---------|-------------|
| 60 | 2.5% |
| 70 | 11.9% |
| 75 | 23.7% |
| 80 | 41.7% |
| 82 | 50.0% (midpoint) |
| 85 | 62.2% |
| 90 | 79.1% |
| 95 | 89.7% |
| 100 | 95.3% |

**Why sigmoid over linear:**
- Smooth S-curve matches realistic manager behavior
- No hard thresholds or conditional branches
- Ace bonus naturally shifts curve right (delaying pull)
- Better matches actual MLB pull patterns

**Calibration results:**
- Regular pitcher: 5.7 IP, 84 pitches (target: 5.4 IP, 85)
- Ace pitcher: 6.5 IP, 93 pitches (realistic for aces)

#### Step 9: Increase Sims + Add Calibration Tracking ✅

**Implemented (Apr 23, 2026):**

| Change | Old | New |
|--------|-----|-----|
| `DEFAULT_NUM_SIMS` | 3000 | **5000** |
| Calibration data | None | Percentiles + range probs |

**Increased simulation count:**
- Industry standard is 5,000 sims (FullCountProps, BallparkPal)
- Higher count reduces variance in probability estimates
- More stable over/under probabilities

**Calibration tracking added:**
```python
'calibration': {
    # Percentiles for distribution analysis
    'p5': ..., 'p10': ..., 'p25': ..., 'p50': ..., 'p75': ..., 'p90': ..., 'p95': ...,
    # Range probabilities for calibration comparison
    'prob_0_3': ...,      # Very low K games
    'prob_4_6': ...,      # Average K games
    'prob_7_9': ...,      # Good K games
    'prob_10_plus': ...,  # Dominant performances
}
```

**Sample calibration output:**
| Pitcher Type | P5 | P25 | P50 | P75 | P95 | 0-3 | 4-6 | 7-9 | 10+ |
|--------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Regular (24% K) | 2 | 4 | 6 | 7 | 10 | 19.0% | 46.2% | 28.7% | 6.2% |
| Ace (29% K) | 3 | 5 | 7 | 9 | 12 | 6.6% | 33.1% | 39.8% | 20.5% |

**Calibration use:**
- Compare predicted percentiles vs actual outcomes
- Track P(0-3 Ks) vs actual bad game rate
- Adjust model if calibration drifts

---

### Simulation V2 Complete

All 9 steps implemented + Step 10 (Vegas Total). Final results vs pre-upgrade:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Regular Mean Ks | Over-projected | 5.7 | ~5.5 |
| Ace Mean Ks | Over-projected | 7.3 | ~7.0 |
| Avg IP | 6.0+ | 5.7 | 5.4 |
| Avg Pitches | 90+ | 84 | 85 |
| TTO Penalty | -5%/-12% | -7%/-18% | Industry |
| Fatigue Penalty | -3%/-7%/-12% | -4%/-8%/-13%/-18% | Industry |
| Pull Model | Linear | Sigmoid | Industry |

**Key improvements:**
- Reduced over-projection on OVERS
- Better variance with stuff/workload distributions
- More realistic late-inning K rate decline
- Calibration tracking for ongoing validation
- **Vegas total integration** (Step 10, Apr 26): High-scoring games properly reduce K projections

#### Step 10: Vegas Total Integration ✅

**Implemented (Apr 26, 2026):**

Addresses failures in high-scoring games (e.g., SEA 11 @ STL 9 with 3 IP starts).

| Parameter | Value | Effect |
|-----------|-------|--------|
| `VEGAS_TOTAL_NEUTRAL` | 8.5 | MLB average game total |
| `VEGAS_TOTAL_HIGH` | 9.5 | High-scoring threshold |
| `VEGAS_TOTAL_SHOOTOUT` | 10.5 | Shootout territory |
| `HIGH_TOTAL_BAD_START_BOOST` | 1.5 | +50% bad start prob |
| `SHOOTOUT_BAD_START_BOOST` | 2.0 | +100% bad start prob |
| `BLOWOUT_K_PENALTY` | 0.85 | -15% K rate when cruising |
| `CLOSE_GAME_K_BOOST` | 1.03 | +3% K rate in tight games |

**How it works:**
1. Simulates opponent runs each inning based on vegas_total / 2 / 9
2. Tracks blowout status (down 4+ runs)
3. Applies K% penalty when cruising in blowout
4. Applies K% boost in close/low-scoring games
5. Increases bad start probability in shootout games
6. Reduces workload target (-8 pitches in shootouts)

**Test results:**
| Vegas Total | Mean Ks | Avg IP | Effect |
|-------------|---------|--------|--------|
| None | 7.25 | 6.1 | Baseline |
| 11.0 (Shootout) | 6.14 | 5.2 | **-1.11 Ks** |
| 7.0 (Low-scoring) | 6.98 | 6.1 | Neutral |

**Impact:** Shootout games now correctly reduce K projections by ~1 full strikeout, matching real-world early pulls in high-scoring games.

---

## Session History
- **Apr 26, 2026**: **Implemented Vegas Total Game Script Modeling**. Completed the vegas_total integration to address MLB simulation failures in high-scoring games (like SEA 11 @ STL 9 where pitchers were pulled at 3 IP). **Changes:** (1) Added opponent run simulation inside `simulate_game()` based on vegas_total - tracks blowout status when down 4+ runs, (2) Added game script K% adjustments - `BLOWOUT_K_PENALTY` (0.85) when cruising, `CLOSE_GAME_K_BOOST` (1.03) in tight games, (3) Updated `run_simulation()` to accept and pass vegas_total with per-sim opponent run variance (0.7-1.3x), (4) Updated bad start probability to scale with vegas_total: +50% for high-total games (9.5+), +100% for shootouts (10.5+), (5) Updated `analyze_prop_simulation()` to accept vegas_total parameter, (6) Updated `main.py` to pass vegas_total from API to simulation (also increased sims to 5,000), (7) Added `game_script` tracking to results (vegas_total, is_shootout, bad_start_prob_adj). **Test results:** Shootout (11.0 total) reduces K projection by ~1.1 Ks and IP by ~0.9 innings compared to neutral. This addresses the exact issue where high-scoring games had much shorter pitcher outings than simulated.
- **Apr 25, 2026**: **Added Probability Distribution Visualization + Frontend Updates**. Added cumulative probability distribution (P(1+ K) through P(12+ K)) to MLB simulation output and frontend. Updated `simulator.py` to calculate and return `cumulative_probs` dict. Updated `index_mlb.html` with new probability distribution table showing 12 cells with green color gradient (dark green for high probability, fading to gray for low). Table appears below the simulation stats when using Monte Carlo simulation. Also verified 5,000 sims are active (from Step 9). Frontend text updated from "3,000 sims" to "5,000 sims". Example output for 28% K pitcher: 1+ K: 99.7%, 5+ K: 80.2%, 7+ K: 50.7%, 10+ K: 13.9%, 12+ K: 5.6%.
- **Apr 17, 2026**: **Fixed MLB Simulation for Traded Pitchers + Synced Results**. Synced NBA Play-In results (Apr 14-15): 14/22 (63.6%) - required using `season_type_all_star="Playoffs"` and boxscore endpoints since Play-In games have different game IDs (0052500xxx). Synced MLB results: Apr 8 (19/29, 65.5%), Apr 9 (3/9, 33.3%), Apr 14 (7/10, 70.0%), Apr 15 (2/8, 25.0%). Overall MLB: 54.8% (161/294). **Fixed MLB Simulation Bug:** Edward Cabrera (traded MIA→CHC) was falling back to Beta-Binomial because simulation used cached `pitcher_data.get('team')` (MIA) instead of the passed `opponent_abbrev` parameter. Fix in `simulator.py`: Changed `analyze_prop_simulation()` to use `get_lineup(opponent_normalized)` directly instead of looking up opponent via stale cached team. Now traded pitchers correctly use Monte Carlo simulation with confirmed lineups. Updated stats: NBA 630 resolved (52.2% CatBoost), MLB 294 resolved (54.8%).
- **Apr 15, 2026**: **V2.2 Training Complete - Fixed Prop-Type Biases**. Investigated why V2.1 was recommending too many overs. Found massive prop-type biases: PRA had 96% over picks, rebounds had 0% over picks (vs 50% actual). Root cause: V2.1 trained on only 512 live predictions - each prop type had only 32-98 samples, creating spurious correlations. **Solution:** Backfilled all V2.1 features into the 15,453-row backtest dataset. Created 3 backfill scripts: (1) `backfill_backtest_avg_vs_opponent.py` - fetched game logs for 50 players, computed historical avg vs opponent for each row (92.7% coverage), (2) `backfill_backtest_dvp.py` - read position-specific DvP data from dvp.xlsx sheets (PG/SG/SF/PF/C), fetched player positions from NBA API (100% coverage), (3) `backfill_backtest_team_stats.py` - fetched opp_def_rating and opp_pace from NBA API (100% coverage). **V2.2 Results:** 59.7% CV accuracy (up from V2.1's 56.1%), +27.2% ROI @ 55% confidence. All prop types now have balanced 51-55% over picks. Feature importance well-distributed (no single feature >20%). Top features: minutes_avg (18.2%), opponent_team (8.5%), opp_def_rating (7.9%), dvp_rank (7.7%). Updated `catboost_predictor.py` to use V2.2 model by default. **V2.3 Roadmap:** Will incorporate CLV when 1,000+ live predictions collected (currently ~600). CLV showed 1.9x hit rate difference (positive 60.5% vs negative 31.7%). Updated CLAUDE.md with V2.2 training results and V2.3 roadmap.
- **Apr 12, 2026**: **MLB Simulation Calibration + Over-Prediction Fixes**. Identified simulation was over-predicting by ~1 K/game due to: (1) BF too high (25.4 vs 21.5 target), (2) No TTO penalty, (3) No bad start variance, (4) No ace differentiation. **Implemented 4 fixes:** (1) **Tightened pull model** - base threshold 65 pitches (was 70), +8% per pitch (was 6%), more aggressive end-of-inning hooks. (2) **Times Through Order penalty** - 1st time: 100%, 2nd time: 95% (-5%), 3rd time: 88% (-12%), 4th+: 85% (-15%). Research-backed effect where batters improve each time they face a pitcher. (3) **Bad start variance** - 10% of sims model disaster outings (40-65 pitch early hook). (4) **Ace pitcher longer leash** - pitchers with K% ≥27% get +8 pitch bonus before pull consideration and 50% lower bad start probability. Managers trust aces more (validated by FanGraphs research). **Calibration results:** BF 25.4→22.8 (target 21.5), IP 5.91→5.25 (target 5.4), Ks 6.1→5.2 (target 5.1). Net effect: ~1 fewer K projected on average. Also fixed lineup confirmation bug - now checks opponent's specific lineup (not both teams) to determine if simulation can run. Fixed opponent auto-detection for manual line mode. **Research:** Explored Statcast pitch-level data via pybaseball - whiff rates by pitch type, 2-strike put-away rates, chase rates. Cole 2024: 45.6% put-away rate, slider 29.3% whiff vs fastball 19.2%. Documented for future Phase 3 enhancements. Pausing development for 2-3 weeks to collect prediction data and validate calibration.
- **Apr 11, 2026**: **MLB Simulation Phases 3 & 4 Complete + Bug Fixes**. **Phase 3 (Soft Replacement):** Integrated Monte Carlo simulation into `/mlb/analyze` as primary model. Falls back to Beta-Binomial when lineups not confirmed. Added `model_type` field to API response ("simulation" or "beta_binomial"). Updated `index_mlb.html` frontend to show model badge (purple=Monte Carlo, yellow=Beta-Binomial). Added simulation details panel showing mean/std/IP/pitches and lineup K rates. **Phase 4 (Accuracy Tracking):** Added 7 new columns to mlb_predictions table for model comparison. Updated `_save_mlb_prediction_v2()` to persist these fields. **Bug Fixes:** (1) Fixed `get_opponent_lineup()` call - was passing opponent abbrev instead of pitcher's team, (2) Added team abbreviation normalization (SFG→SF, KCR→KC) since FanGraphs uses different codes than MLB Stats API, (3) Fixed `get_park_factor()` handling - returns float not dict, (4) Fixed lineup K% display (was multiplying by 100 twice), (5) Added complete matchup_factors and pitcher_stats to simulation response for frontend compatibility. **Documented optimization roadmap** in CLAUDE.md (5 phases from validation to ML enhancement).
- **Apr 8-9, 2026**: **MLB Simulation Phases 1 & 2 Complete**. Phase 1: Created `mlb_simulation/` package with `batter_data.py` (Baseball Reference, 236 batters from 2026 at 30+ PA, 460 from 2025 fallback), `lineup_fetcher.py` (MLB Stats API). FanGraphs 403'd so used Baseball Reference. Phase 2: Built `simulator.py` (350 lines) with Log5 probability, fatigue model, pull probability, PA simulator, Monte Carlo loop. Tuned parameters to match 2024 MLB averages (5.9 IP, 87 pitches). Sample: Tarik Skubal (29.6% K) → 7.4 mean Ks, 80.5% over 5.5.
- **Apr 8, 2026**: **V3 Post-Mortem + V2.1 Training Complete + MLB Simulation Planning**. NBA: Investigated V3 failure, trained V2.1 on live data (56.1% CV accuracy). MLB: Researched simulation approaches, decided on Tier 3 (full Monte Carlo with lineup-aware, fatigue-adjusted, dynamic BF). Key sources: FullCountProps (LightGBM + 5K sims), KSplit (PA-level), Ballpark Pal (3K sims). Starting Phase 1: batter data + lineup fetcher. Investigated why V3 model failed (53.2% test vs V2's 60.7%). Found critical issues: (1) Data poisoning - 96.8% of training data (backtest) had fake/constant values for new features, (2) Removed `avg_vs_opponent` which was 42% of V2's feature importance, (3) New features got 0% importance because they were constant. **Solution:** Trained V2.1 on live data only (512 predictions) with 5-fold CV. Created `backfill_avg_vs_opponent.py` (467 updated) and `backfill_dvp_features.py` (592 updated). **V2.1 Results:** 56.1% CV accuracy, top features: closing_line (17.1%), avg_vs_opponent (17.1%), last_10_avg (8.2%), player_position (7.7%), opp_def_rating (7.5%), dvp_rank (7.2%). Key improvement: opponent_team dropped to 0.3% (no overfitting). DvP features contributing ~12% combined. Updated predictor to use V2.1 model.
- **Apr 1, 2026**: **Synced Mar 31 + Apr 1 Results + Fixed MLB CLV Fetcher**. Mar 31: NBA 11/16 (68.8%), MLB 10/23 (43.5%). Apr 1: NBA 12/19 (63.2%) with 4 voided (Embiid x2, Brunson, Ingram DNP), MLB 16/30 (53.3%). **Fixed `mlb_fetch_clv.py`** - was missing early games because it started querying at 22:00 UTC (6pm ET). Now starts at 16:00 UTC (12pm ET) and tries 6 timestamps in order from earliest to latest: 16:00, 18:00, 20:00, 22:00, 00:00+1, 02:00+1. Also improved logic to keep searching timestamps until opponent's game is found, rather than stopping at first batch of events. **Research: MLB Simulation Models** - Investigated Monte Carlo simulation approaches for strikeout props. Found [FullCountProps](https://www.fullcountprops.com/methodology) runs 5,000 PA-level sims with LightGBM (33 features, 1M+ training PAs). Open source options (baseballforecaster, BayesBall, etc.) are either fantasy-draft focused or incomplete for K props. Conclusion: build custom lightweight simulator (~200 lines) that gets lineups, calculates batter-specific K probs, and runs 5,000 Monte Carlo sims for true P(over/under). **Updated stats:** NBA 369 resolved (~57%), MLB 145 resolved (52.4%). Credits: 17,457.
- **Mar 30, 2026**: **Retired Rule-Based Model for NBA + Baseball Savant Park Factors**. CatBoost ML (57% accuracy) is now the sole recommendation model for NBA props. Rule-based (47% accuracy) has been completely removed from the UI and stats. Changes: (1) Updated `main.py` to use CatBoost pick as primary `recommended_side`, (2) Removed rule-based section from `index.html`, (3) Updated `/stats` endpoint to show CatBoost as primary with rule-based as "legacy" reference only. Also synced MLB results for Mar 29 (54.5% hit rate, 12/22) and NBA results (57.1%, 8/14). **MLB overall: 67 resolved, 58.2% hit rate**. Fixed accent matching in `mlb_sync_results.py` and added alternate team abbreviations (TBR, SFG, SDP, KCR). **MLB Park Factors**: Replaced broken FanGraphs park factor scraper with Baseball Savant Statcast SO (strikeout) factors. These are actual K-specific park factors, not derived from run environment. Key values: T-Mobile Park 117 (+17% Ks), American Family Field 109, Coors Field 90 (-10% Ks), Kauffman Stadium 89. Verified projection correctly applies: Paul Skenes @ T-Mobile = 6.14 Ks, @ Coors = 5.38 Ks (+14% swing). New cache file: `cache/mlb_park_factors.json` with 28 parks from Baseball Savant.
- **Mar 28, 2026**: **MLB Data Pipeline Fixes + Live Refresh System**. Fixed multiple pitcher matching issues: (1) Added `normalize_name()` using unicodedata to handle accents (López → Lopez), (2) Added legacy cache fallback (`_check_legacy_cache()`) for pitchers not in pybaseball data, (3) Fixed matching order so legacy cache is checked BEFORE "last name only" match to prevent wrong matches (Eury Perez → Cionel Perez). Added duplicate prevention to both MLB and NBA prediction saving (checks for existing predictions before INSERT). Cleaned up 7 duplicate MLB predictions. **Created `mlb_refresh_data.py`** - comprehensive daily refresh script that pulls LIVE data from pybaseball (FanGraphs) for pitchers, teams, and park factors. Usage: `python mlb_refresh_data.py [--force] [--status]`. Caches refresh if older than 12hr (pitchers/teams) or 7 days (parks). **Known limitation:** Some traded pitchers (Dylan Cease, Sonny Gray, Miles Mikolas, Michael Lorenzen) missing from Odds API - use manual mode with Fanatics lines. **Total MLB predictions:** 29 (after cleanup).
- **Mar 25, 2026**: **MLB Integration Complete**. Added full MLB strikeout props support: pybaseball for automated FanGraphs data (SwStr%, CSW%, K%), Beta-Binomial projection model, separate API endpoints (`/mlb/analyze`, `/mlb/games`, `/mlb/predictions`, `/mlb/stats`), results sync via statsapi. New files: `mlb_fetch_data.py`, `mlb_projection.py`, `mlb_sync_results.py`. New table: `mlb_predictions`. Completely separate from NBA codebase - can run both simultaneously. Season starts tomorrow (Mar 26). **Tested**: Logan Webb vs NYY projects 6.06 Ks, slight edge on over 5.5.
- **Mar 24, 2026**: **Synced Mar 23 results + CLV + DvP refresh**. Mar 23: Rule-based 6/24 (25%) - rough day, CatBoost 15/24 (62.5%) - crushed it. Notable: Alperen Sengun went off (46 PR, 56 PRA), Paolo Banchero 39 pts on 25.5 line. CLV fetched for all 24 predictions (20 via fetch_clv.py, 4 already had data). Most lines didn't move (0 CLV). Refreshed DvP cache. **Added Daily Operations section to CLAUDE.md** with exact commands for syncing results, fetching CLV, and refreshing DvP. **Updated stats:** 334 resolved, Rule-based 47.0% (157/334), CatBoost 57.1% (145/254). Progress to 500: 67%. Credits: 16,858.
- **Mar 23, 2026**: **Synced Mar 22 + DvP refinements**. Mar 22: 3/6 (50%) rule-based, 4/6 (66.7%) CatBoost. Fixed DvP logic to always use position-specific data when available (even for 0% adjustments on average defenses). Refreshed DvP cache. **Updated stats:** 310 resolved, Rule-based 48.7% (151/310), CatBoost 56.5% (130/230). Progress to 500 retrain milestone: 62%. Credits: 17,150.
- **Mar 21, 2026**: **Built Defense vs Position (DvP) Integration**. Created `scrape_dvp.py` to scrape FantasyPros position-specific defensive data. Integrated into `main.py` - now uses position-specific adjustments instead of generic `opp_def_rating`. Adjustment scale: +9% for worst defenses (PG vs Utah), -10% for best (PG vs OKC). Synced Mar 19-21 results: Mar 19: 6/15 (40%), Mar 20: 4/16 (25%), Mar 21: 10/22 (45.5%) - CatBoost crushed it at 63.6% (14/22). 8 voided (DNP). CLV edge holds: Positive 57.7%, Negative 33.3%.
- **Mar 18, 2026**: **Synced Mar 16-17 Results + CLV**. Mar 16: 9/14 (64.3%), Mar 17: 9/15 (60%). Voided 1 DNP (Giannis). CLV fetcher working perfectly - 100% success rate. **Updated stats:** Rule-based 51.3% (117/228), CatBoost 56.7% (97/171). **CLV edge confirmed:** Positive CLV 60.5% vs Negative CLV 31.7% (1.9x better). Credits remaining: 18,311.
- **Mar 15, 2026**: **Built Odds API CLV Fetcher**. Created `fetch_clv.py` to replace BettingPros scraping (blocked by CAPTCHA). Uses The Odds API historical endpoint ($30/month plan) to fetch closing lines ~5 min before game start. Cost: ~11 credits per prediction. Successfully fetched CLV for 42/43 predictions. Also added `status = 'voided'` for DNP predictions and retry logic for NBA API timeouts. **CLV Stats (199 predictions):** Positive CLV 59.4% (19/32), Negative CLV 35.1% (13/37) - 1.7x edge on positive CLV bets.
- **Mar 10, 2026**: **Removed Razzball, Switched to 75/25 Formula**. Razzball projected minutes proved unreliable due to (1) infrequent updates causing stale data, (2) player name matching bug ("Anthony Edwards" matched "Justin Edwards" due to partial last-name matching). Changed minutes projection from 65/35 to **75/25 formula** (75% season avg + 25% last 5 avg) - research baseline, more stable. Synced 121 predictions from Mar 3-8: **CatBoost 56.7% (85/150)**, Rule-based 48.7% (73/150). CatBoost continues to outperform. Both models now use projected_minutes feature consistently. Files `scrape_razzball.py` and `razzball_minutes_cache.json` are deprecated but retained.
- **Mar 9, 2026**: **Razzball Minutes Integration**. Built `scrape_razzball.py` to scrape injury-adjusted projected minutes from Razzball's NBA lineups page. Uses Playwright to handle dynamic tables with HOME/AWAY teams side-by-side. Integrated into `main.py`: checks Razzball cache first, falls back to 65/35 formula if player not found. Added `minutes_source` field to adjustment_factors ("razzball" or "formula") for transparency. Fixed diacritic normalization (Jokić → Jokic) for player name matching. **Why**: Our 65/35 formula over-projected when recent games had inflated minutes due to teammate injuries. Razzball uses 2 injury services and auto-redistributes minutes. **Example**: Drummond showed 18.0 min (formula) vs 12.2 min (Razzball). Console now shows `Proj Min: 33.8 (razzball)` to indicate source.
- **Mar 6, 2026**: Synced results for Mar 3, 4, 5. Fixed closing line scraping issues - LeBron's line was incorrectly scraped as 23.5 (actually 21.5) due to race condition. Discovered BettingPros rate limiting after ~15-20 rapid requests. **Improved daily_sync.py rate limiting**: Changed from 5s delay to 8-12s random jitter, 45s pause every 5 requests, exponential backoff (30s→60s→120s) on consecutive failures. Synced 68 new predictions across 3 days. Current stats: 91 predictions with CLV data. **CLV theory confirmed**: Positive CLV hits 75% (9/12), Negative CLV hits 28% (5/18) - 2.7x difference. Some players not tracked on BettingPros for certain prop types (combo props like pa/pr/pra for lesser-known players).
- **Mar 2, 2026**: Implemented CLV tracking. Upgraded Odds API to 20K credits/month (new key). Built `daily_sync.py` script to scrape closing lines from BettingPros + sync results. Fixed BettingPros blocking with anti-detection. Backfilled CLV for 20 predictions. CLV stats: Positive CLV 2/2 (100%), Negative CLV 0/4 (0%), Zero CLV 9/14 (64%). **Redesigned confidence grading system**: Changed from edge+hit_rate+variance to **projection reliability** (consistency, sample size, minutes volume, data completeness). **Implemented minutes-first projection system**: Research-based approach that projects minutes first (65% season + 35% last 5), calculates filtered stats-per-minute (excludes <20 min games), applies matchup adjustments (defense ±15%, pace ±10%, venue ±2%). Old system was 50% directional accuracy; new system uses industry best practices. Fixed daily_sync.py timeout by changing from `networkidle` to `domcontentloaded`.
- **Mar 1, 2026**: Database cleanup - deleted 166 test entries (IDs 1-205), keeping only 12 real user predictions. Added `test` flag to `/analyze` endpoint (`{"query": "...", "test": true}`) to prevent test queries from being saved to DB. Synced Feb 28 results: 5 new predictions resolved (Tyler Herro, Luka PR, Brandon Ingram RA, Keyonte George, Gui Santos). Rule-based 3/5 (60%), CatBoost 1/5 (20%). Total now 17 predictions: Rule-based 52.9% (9/17), CatBoost 41.2% (7/17). Odds API still exhausted - manual mode active.
- **Feb 26, 2026**: Synced results - 133 total predictions. Rule-based at 49.6% (66/133), CatBoost V2 at 60.0% (9/15). Latest resolved: Jokic points 28.5 → actual 30. CatBoost picked OVER (hit), Rule-based picked UNDER (miss). Both models perform better on unders (61-64%) than overs (33-45%). 34 predictions pending. Tested predictions with V2 model - confidences now in realistic 51-78% range.
- **Feb 25, 2026 (continued)**: **CRITICAL BUG FIX** - Discovered model was outputting 93% confidence on PRA props. Investigation revealed BettingPros scraper bug: URL was `/points-rebounds-assists/` but correct URL is `/points-assists-rebounds/` (PAR order). The wrong URL redirected to `/points/`, so all PRA "closing lines" were actually POINTS lines (mean 18.4 vs actual PRA mean 32.1). This caused 91% over rate in training data. **Fix**: Corrected URL slug in `scrape_closing_lines.py`, re-scraped 2,085 PRA games with correct data, retrained model. **Final results**: 60.7% test accuracy, +28.0% ROI @ 55%, prop_type importance at 5.9%, avg_vs_opponent #1 at 42.0%. PRA over rate now correct at 50.9%. Model outputs realistic confidences (67-77% for PRA) instead of 93%+.
- **Feb 25, 2026**: Implemented CatBoost V2 model with new features focused on line quality and matchup edges. Created `prepare_training_data_v2.py` with new features: `line_vs_last_5`, `line_difficulty`, `consistency`, `avg_vs_opponent`, `opp_def_rating`, `opp_pace`. Created `train_catboost_v2.py` with reduced overfitting (depth=5, L2 reg). Updated `catboost_predictor.py` to use V2 model and compute derived features. Updated `main.py` to add `full_game_log` to `get_player_stats()` for computing `avg_vs_opponent` at prediction time.
- **Feb 22, 2026**: Added Manual Mode fallback for Odds API quota exhaustion. Users can now provide line + odds in query (e.g., "LeBron points 25.5 -110"). Added `_odds_api_available` tracking, `parse_query_with_manual_line()` function, dynamic status indicators in frontend (green=Live Odds, yellow=Manual Mode). Fixed NBA API timeouts by adding browser-like headers to all nba_api calls. Synced results: 126 total predictions, Rule-based at 48.4%, CatBoost at 62.5% (8 predictions). Cleaned up 6 stale predictions with incorrect game dates. Documented Rule-based vs CatBoost model comparison.
- **Feb 21, 2026 (Session 2)**: Implemented full CatBoost ML pipeline. Created 8 new files: prepare_training_data.py (feature engineering), baseline_model.py (logistic regression baseline at 55.6%), train_catboost.py (CatBoost training), evaluate_model.py (calibration & ROI analysis), catboost_predictor.py (production inference). Model achieves 57.1% test accuracy, +24.4% ROI at 55% threshold. Integrated into main.py: added catboost_prob_over, catboost_pick, catboost_confidence, catboost_hit columns to database. Updated /analyze to include CatBoost predictions, /sync-results to compute catboost_hit, /stats to show both model accuracies. Updated frontend (index.html) with purple ML Model Prediction card showing confidence, probability, and model agreement indicator.
- **Feb 21, 2026 (Session 1)**: Built BettingPros closing line scraper with Playwright. Scraped 15,453 historical prop lines (98.7% coverage). Key fixes: (1) season dropdown toggle for lazy-loaded data, (2) date format handling (Oct-Dec = /25, Jan+ = /26), (3) position suffix slugs (alperen-sengun-c, jalen-johnson-f-f), (4) NL markers for games without lines. Backtest data ready for ML training.
- **Feb 13, 2026**: Synced results - now at 121 predictions (44.1% hit rate). Grade B above breakeven at 52.9%. Grade D still best at 71.4%. Grade A still inverted at 38%.
- **Feb 12, 2026 (continued)**: Researched defense vs position feature. Documented for Phase 2. Improved player name matching: normalize_name now handles dots (P.J. → PJ) and suffixes (Jr., III); added fuzzy matching for typos (85%+ similarity). Attempted "ALL" query feature but reverted due to frontend incompatibility with array responses.
- **Feb 12, 2026**: Synced results - now at 110 predictions (45.3% hit rate). Grade A recovered to 40% (Jokić 4/4). Grade D still best at 71.4%. Overall model improving toward breakeven.
- **Feb 10, 2026 (session end)**: Synced again - now at 84 predictions (40.8% hit rate). Grade B improved to 50% breakeven. Grade A still at 20% (fade signal). Grade D at 66.7% (best). De'Aaron Fox injury exit (17 min) caused misses - validates need for injury data.
- **Feb 10, 2026 (continued)**: Researched teammate injury impact feature. Found PBPStats free WOWY API - provides exact WITH/WITHOUT teammate stats needed. Documented implementation plan for phases 2-4 (500+ predictions). Also identified `nbainjuries` package for injury status data. No implementation yet - waiting for more predictions.
- **Feb 10, 2026**: Synced results - now at 64 predictions (36.1% hit rate). Grade A confidence now at 21.7% (worse than coin flip). Over/under disparity holding: overs 30.4%, unders 53.3%. Pattern is clear - Grade A is a fade signal.
- **Feb 8, 2026 (Session 2 continued)**: Synced results - now at 55 predictions (38.5% hit rate). Discovered major model issues: over-projection bias (overs hitting 30% vs unders 66.7%), inverted confidence grading (Grade A worst at 27.8%). These patterns will be valuable training signal for CatBoost.
- **Feb 8, 2026 (Session 2)**: Added 19 ML feature columns to database schema. Updated `save_prediction()` to capture all features automatically. Added `/backfill-features` endpoint. Backfilled all 24 existing predictions with player stats, opponent data, days rest, actual minutes. Now tracking: opponent_team, is_home, vegas_total, spread, season_avg, last_10_avg, std_dev, minutes_avg, opp_def_rating, opp_pace, prob_over, no_vig_prob, days_rest, is_b2b, usage_rate, home_avg, away_avg, model_projection, actual_minutes.
- **Feb 8, 2026 (Session 1)**: Initial exploration and documentation. Synced all results (now 50% hit rate on 24 predictions). Discussed ML roadmap - decided on CatBoost + Platt Scaling as primary approach. Will collect 2,500-5,000 predictions before implementing ML.
