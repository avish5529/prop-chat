import os
import math
import sqlite3
import unicodedata
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# nba_api imports
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    leaguedashteamstats,
    leaguedashplayerstats,
    boxscoretraditionalv2
)

# Defense vs Position data
from scrape_dvp import get_dvp_adjustment, load_cache as load_dvp_cache

load_dotenv()

# Custom headers for NBA API (required to avoid blocking)
NBA_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com'
}

# CatBoost predictor (lazy loaded)
_catboost_predictor = None

def get_catboost_predictor():
    """Get or create the CatBoost predictor instance."""
    global _catboost_predictor
    if _catboost_predictor is None:
        try:
            from catboost_predictor import CatBoostPredictor
            _catboost_predictor = CatBoostPredictor()
        except Exception as e:
            print(f"[CatBoost] Failed to load predictor: {e}")
            _catboost_predictor = None
    return _catboost_predictor

app = FastAPI(title="PropBot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Cache for team stats (refreshed per session)
_team_stats_cache = None
_player_advanced_cache = None

# Odds API status tracking
_odds_api_available = True  # Assume available until proven otherwise
_odds_api_last_error = None


def normalize_position(position: str) -> str:
    """
    Normalize NBA API position string to DvP format (PG, SG, SF, PF, C).
    NBA API returns: "Guard", "Forward", "Center", "Guard-Forward", "Forward-Guard", etc.
    """
    if not position or position == "Unknown":
        return "SF"  # Default to SF as middle ground

    pos = position.upper()

    # Direct mappings
    if pos in ["PG", "POINT GUARD"]:
        return "PG"
    if pos in ["SG", "SHOOTING GUARD"]:
        return "SG"
    if pos in ["SF", "SMALL FORWARD"]:
        return "SF"
    if pos in ["PF", "POWER FORWARD"]:
        return "PF"
    if pos in ["C", "CENTER"]:
        return "C"

    # Handle combo positions
    if "GUARD" in pos and "FORWARD" in pos:
        return "SF"  # Tweeners like wings
    if "FORWARD" in pos and "CENTER" in pos:
        return "PF"  # Big forwards
    if "GUARD" in pos:
        return "SG"  # Default guards to SG
    if "FORWARD" in pos:
        return "SF"  # Default forwards to SF
    if "CENTER" in pos:
        return "C"

    # Fallback
    return "SF"


# Team name to abbreviation mapping for DvP lookups
TEAM_NAME_TO_ABBR = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


def get_team_abbr(team_name: str) -> str:
    """Get team abbreviation from full team name."""
    if not team_name:
        return None
    return TEAM_NAME_TO_ABBR.get(team_name.lower(), None)


# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

# Timezone settings
EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


def utc_to_eastern_date(utc_timestamp: str) -> str:
    """Convert UTC timestamp (ISO format) to Eastern date string (YYYY-MM-DD)"""
    # Parse UTC timestamp like "2026-02-07T00:40:00Z"
    utc_dt = datetime.fromisoformat(utc_timestamp.replace("Z", "+00:00"))
    eastern_dt = utc_dt.astimezone(EASTERN_TZ)
    return eastern_dt.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────
# Database Setup & Functions
# ─────────────────────────────────────────────────────────────────

def init_database():
    """Initialize SQLite database with predictions table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            prop_type TEXT DEFAULT 'points',
            line REAL NOT NULL,
            projection REAL NOT NULL,
            recommended_side TEXT NOT NULL,
            confidence_grade TEXT,
            ev REAL,
            edge REAL,
            best_odds INTEGER,
            best_book TEXT,
            -- Filled in after game completes
            actual_result REAL,
            hit INTEGER,  -- 1 = hit, 0 = miss, NULL = pending
            -- Metadata
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT,
            UNIQUE(player_id, game_id, prop_type)
        )
    """)

    # Add new columns for ML features (if they don't exist)
    new_columns = [
        ("opponent_team", "TEXT"),
        ("is_home", "INTEGER"),
        ("vegas_total", "REAL"),
        ("spread", "REAL"),
        ("season_avg", "REAL"),
        ("last_10_avg", "REAL"),
        ("std_dev", "REAL"),
        ("minutes_avg", "REAL"),
        ("opp_def_rating", "REAL"),
        ("opp_pace", "REAL"),
        ("prob_over", "REAL"),
        ("no_vig_prob", "REAL"),
        ("days_rest", "INTEGER"),
        ("is_b2b", "INTEGER"),
        ("usage_rate", "REAL"),
        ("home_avg", "REAL"),
        ("away_avg", "REAL"),
        ("model_projection", "REAL"),
        ("actual_minutes", "REAL"),
        # CatBoost model columns
        ("catboost_prob_over", "REAL"),
        ("catboost_pick", "TEXT"),
        ("catboost_confidence", "REAL"),
        ("catboost_hit", "INTEGER"),
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()
    print("[Database] Initialized predictions.db")


def save_prediction(
    player_name: str,
    player_id: int,
    game_id: str,
    game_date: str,
    line: float,
    projection: float,
    recommended_side: str,
    confidence_grade: str,
    ev: float,
    edge: float,
    best_odds: int,
    best_book: str,
    prop_type: str = "points",
    # New ML feature columns
    opponent_team: str = None,
    is_home: bool = None,
    vegas_total: float = None,
    spread: float = None,
    season_avg: float = None,
    last_10_avg: float = None,
    std_dev: float = None,
    minutes_avg: float = None,
    opp_def_rating: float = None,
    opp_pace: float = None,
    prob_over: float = None,
    no_vig_prob: float = None,
    days_rest: int = None,
    is_b2b: bool = None,
    usage_rate: float = None,
    home_avg: float = None,
    away_avg: float = None,
    model_projection: float = None,
    # CatBoost model columns
    catboost_prob_over: float = None,
    catboost_pick: str = None,
    catboost_confidence: float = None,
):
    """Save a prediction to the database with all ML features"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check for existing prediction (same player, date, prop_type, and line)
    cursor.execute("""
        SELECT id FROM predictions
        WHERE game_date = ? AND player_name = ? AND prop_type = ? AND line = ?
    """, (game_date, player_name, prop_type, line))
    existing = cursor.fetchone()

    if existing:
        print(f"[Database] Prediction already exists for {player_name} {prop_type} @ {line} - skipping save")
        conn.close()
        return  # Already exists, don't save duplicate

    try:
        cursor.execute("""
            INSERT INTO predictions
            (player_name, player_id, game_id, game_date, prop_type, line, projection,
             recommended_side, confidence_grade, ev, edge, best_odds, best_book, created_at,
             opponent_team, is_home, vegas_total, spread, season_avg, last_10_avg, std_dev,
             minutes_avg, opp_def_rating, opp_pace, prob_over, no_vig_prob, days_rest, is_b2b,
             usage_rate, home_avg, away_avg, model_projection,
             catboost_prob_over, catboost_pick, catboost_confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_name, player_id, game_id, game_date, prop_type, line, projection,
            recommended_side, confidence_grade, ev, edge, best_odds, best_book,
            datetime.now().isoformat(),
            opponent_team,
            1 if is_home else 0 if is_home is not None else None,
            vegas_total, spread, season_avg, last_10_avg, std_dev, minutes_avg,
            opp_def_rating, opp_pace, prob_over, no_vig_prob, days_rest,
            1 if is_b2b else 0 if is_b2b is not None else None,
            usage_rate, home_avg, away_avg, model_projection,
            catboost_prob_over, catboost_pick, catboost_confidence,
            'pending'
        ))
        conn.commit()
        print(f"[Database] Saved prediction: {player_name} {recommended_side} {line} (with ML features)")
    except Exception as e:
        print(f"[Database] Error saving prediction: {e}")
    finally:
        conn.close()


def get_pending_predictions(game_date: str = None) -> list:
    """Get predictions that haven't been resolved yet (excludes voided)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if game_date:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE status = 'pending' AND game_date = ?
            ORDER BY created_at DESC
        """, (game_date,))
    else:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE status = 'pending'
            ORDER BY created_at DESC
        """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_prediction_result(prediction_id: int, actual_result: float, hit: bool,
                             actual_minutes: float = None, catboost_hit: bool = None):
    """Update a prediction with the actual result, minutes played, and CatBoost hit"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE predictions
        SET actual_result = ?, hit = ?, resolved_at = ?, actual_minutes = ?, catboost_hit = ?, status = 'resolved'
        WHERE id = ?
    """, (actual_result, 1 if hit else 0, datetime.now().isoformat(), actual_minutes,
          1 if catboost_hit else 0 if catboost_hit is not None else None, prediction_id))

    conn.commit()
    conn.close()


def get_player_game_stat(player_id: int, game_date: str, prop_type: str = "points") -> tuple[Optional[float], Optional[float]]:
    """
    Fetch actual stat and minutes for a player on a specific date using game log.
    Returns (stat_value, minutes_played) tuple.
    """
    try:
        season = get_current_season()
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
            headers=NBA_API_HEADERS,
            timeout=30
        )
        df = game_log.get_data_frames()[0]

        if df.empty:
            return None, None

        # Map prop types to column names
        prop_to_column = {
            "points": "PTS",
            "rebounds": "REB",
            "assists": "AST",
            "pra": None,  # Calculated
            "pr": None,   # Calculated
            "pa": None,   # Calculated
            "ra": None,   # Calculated
        }

        # Game date format in API is like "JAN 15, 2025"
        # Convert our date format to match
        target_date = datetime.strptime(game_date, "%Y-%m-%d")

        for _, row in df.iterrows():
            game_date_str = row["GAME_DATE"]
            try:
                row_date = datetime.strptime(game_date_str, "%b %d, %Y")
                if row_date.date() == target_date.date():
                    # Get minutes played
                    minutes = float(row["MIN"]) if row["MIN"] else None

                    # Handle combination props
                    if prop_type == "pra":
                        stat = float(row["PTS"]) + float(row["REB"]) + float(row["AST"])
                    elif prop_type == "pr":
                        stat = float(row["PTS"]) + float(row["REB"])
                    elif prop_type == "pa":
                        stat = float(row["PTS"]) + float(row["AST"])
                    elif prop_type == "ra":
                        stat = float(row["REB"]) + float(row["AST"])
                    else:
                        column = prop_to_column.get(prop_type, "PTS")
                        stat = float(row[column])

                    return stat, minutes
            except:
                continue

        return None, None
    except Exception as e:
        print(f"[Database] Error fetching player game stat: {e}")
        return None, None


def sync_results_for_date(game_date: str) -> dict:
    """Sync results for all pending predictions on a specific date"""
    pending = get_pending_predictions(game_date)

    results = {
        "date": game_date,
        "total_pending": len(pending),
        "resolved": 0,
        "hits": 0,
        "misses": 0,
        "catboost_hits": 0,
        "catboost_misses": 0,
        "errors": 0,
        "details": []
    }

    for pred in pending:
        player_id = pred["player_id"]
        player_name = pred["player_name"]
        line = pred["line"]
        recommended_side = pred["recommended_side"]
        prop_type = pred.get("prop_type", "points")
        catboost_pick = pred.get("catboost_pick")

        # Fetch actual stat and minutes based on prop type
        actual_pts, actual_minutes = get_player_game_stat(player_id, game_date, prop_type)

        if actual_pts is None:
            results["errors"] += 1
            results["details"].append({
                "player": player_name,
                "status": "error",
                "message": "Could not fetch game stats"
            })
            continue

        # Determine if prediction hit (original model)
        if recommended_side == "over":
            hit = actual_pts > line
        else:
            hit = actual_pts < line

        # Determine if CatBoost prediction hit
        catboost_hit = None
        if catboost_pick:
            if catboost_pick == "over":
                catboost_hit = actual_pts > line
            else:
                catboost_hit = actual_pts < line

            if catboost_hit:
                results["catboost_hits"] += 1
            else:
                results["catboost_misses"] += 1

        # Update database with actual result, minutes, and CatBoost hit
        update_prediction_result(pred["id"], actual_pts, hit, actual_minutes, catboost_hit)

        results["resolved"] += 1
        if hit:
            results["hits"] += 1
        else:
            results["misses"] += 1

        results["details"].append({
            "player": player_name,
            "prop_type": prop_type,
            "line": line,
            "side": recommended_side,
            "catboost_pick": catboost_pick,
            "projection": pred["projection"],
            "actual": actual_pts,
            "actual_minutes": actual_minutes,
            "hit": hit,
            "catboost_hit": catboost_hit,
            "status": "resolved"
        })

    return results


def get_accuracy_stats() -> dict:
    """Calculate overall accuracy statistics including CatBoost model"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Overall stats for original model (exclude voided)
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END), 0) as hits,
            COALESCE(SUM(CASE WHEN hit = 0 THEN 1 ELSE 0 END), 0) as misses,
            COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) as pending,
            COALESCE(SUM(CASE WHEN status = 'voided' THEN 1 ELSE 0 END), 0) as voided
        FROM predictions
    """)
    row = cursor.fetchone()
    overall = dict(row) if row else {"total": 0, "hits": 0, "misses": 0, "pending": 0, "voided": 0}

    # CatBoost model stats (exclude voided)
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN catboost_hit = 1 THEN 1 ELSE 0 END), 0) as hits,
            COALESCE(SUM(CASE WHEN catboost_hit = 0 THEN 1 ELSE 0 END), 0) as misses
        FROM predictions
        WHERE catboost_hit IS NOT NULL AND status != 'voided'
    """)
    row = cursor.fetchone()
    catboost_overall = dict(row) if row else {"total": 0, "hits": 0, "misses": 0}

    # By confidence grade (exclude voided)
    cursor.execute("""
        SELECT
            confidence_grade,
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END), 0) as hits
        FROM predictions
        WHERE status = 'resolved'
        GROUP BY confidence_grade
        ORDER BY confidence_grade
    """)
    by_confidence = [dict(row) for row in cursor.fetchall()]

    # By recommended side (exclude voided)
    cursor.execute("""
        SELECT
            recommended_side,
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END), 0) as hits
        FROM predictions
        WHERE status = 'resolved'
        GROUP BY recommended_side
    """)
    by_side = [dict(row) for row in cursor.fetchall()]

    # CatBoost by recommended side (exclude voided)
    cursor.execute("""
        SELECT
            catboost_pick,
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN catboost_hit = 1 THEN 1 ELSE 0 END), 0) as hits
        FROM predictions
        WHERE catboost_hit IS NOT NULL AND catboost_pick IS NOT NULL AND status != 'voided'
        GROUP BY catboost_pick
    """)
    catboost_by_side = [dict(row) for row in cursor.fetchall()]

    # Recent predictions (last 20, exclude voided)
    cursor.execute("""
        SELECT player_name, line, recommended_side, catboost_pick, projection,
               actual_result, hit, catboost_hit, confidence_grade, game_date, status
        FROM predictions
        WHERE status = 'resolved'
        ORDER BY resolved_at DESC
        LIMIT 20
    """)
    recent = [dict(row) for row in cursor.fetchall()]

    # CLV Statistics
    cursor.execute("""
        SELECT
            COUNT(*) as total_with_clv,
            COALESCE(AVG(clv), 0) as avg_clv,
            COALESCE(SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END), 0) as positive_clv,
            COALESCE(SUM(CASE WHEN clv = 0 THEN 1 ELSE 0 END), 0) as zero_clv,
            COALESCE(SUM(CASE WHEN clv < 0 THEN 1 ELSE 0 END), 0) as negative_clv,
            COALESCE(SUM(CASE WHEN clv > 0 AND hit = 1 THEN 1 ELSE 0 END), 0) as positive_clv_hits,
            COALESCE(SUM(CASE WHEN clv = 0 AND hit = 1 THEN 1 ELSE 0 END), 0) as zero_clv_hits,
            COALESCE(SUM(CASE WHEN clv < 0 AND hit = 1 THEN 1 ELSE 0 END), 0) as negative_clv_hits
        FROM predictions
        WHERE clv IS NOT NULL AND status = 'resolved'
    """)
    clv_row = cursor.fetchone()
    clv_stats = dict(clv_row) if clv_row else {}

    conn.close()

    # Calculate percentages for original model
    resolved = overall["hits"] + overall["misses"]
    hit_rate = (overall["hits"] / resolved * 100) if resolved > 0 else 0

    # Calculate percentages for CatBoost model
    catboost_resolved = catboost_overall["hits"] + catboost_overall["misses"]
    catboost_hit_rate = (catboost_overall["hits"] / catboost_resolved * 100) if catboost_resolved > 0 else 0

    return {
        "model": {
            "name": "CatBoost ML",
            "total_predictions": catboost_overall["total"],
            "resolved": catboost_resolved,
            "pending": overall["pending"],
            "voided": overall["voided"],
            "hits": catboost_overall["hits"],
            "misses": catboost_overall["misses"],
            "hit_rate": round(catboost_hit_rate, 1)
        },
        "legacy_rule_based": {
            "note": "Retired - kept for historical comparison only",
            "resolved": resolved,
            "hits": overall["hits"],
            "hit_rate": round(hit_rate, 1)
        },
        "by_confidence": [
            {
                "grade": row["confidence_grade"],
                "total": row["total"],
                "hits": row["hits"],
                "hit_rate": round(row["hits"] / row["total"] * 100, 1) if row["total"] > 0 else 0
            }
            for row in by_confidence
        ],
        "by_side": [
            {
                "side": row["catboost_pick"],
                "total": row["total"],
                "hits": row["hits"],
                "hit_rate": round(row["hits"] / row["total"] * 100, 1) if row["total"] > 0 else 0
            }
            for row in catboost_by_side
        ],
        "clv": {
            "total_with_clv": clv_stats.get("total_with_clv", 0),
            "average_clv": round(clv_stats.get("avg_clv", 0), 2),
            "by_clv_bucket": [
                {
                    "bucket": "positive",
                    "total": clv_stats.get("positive_clv", 0),
                    "hits": clv_stats.get("positive_clv_hits", 0),
                    "hit_rate": round(clv_stats.get("positive_clv_hits", 0) / clv_stats.get("positive_clv", 1) * 100, 1) if clv_stats.get("positive_clv", 0) > 0 else 0
                },
                {
                    "bucket": "zero",
                    "total": clv_stats.get("zero_clv", 0),
                    "hits": clv_stats.get("zero_clv_hits", 0),
                    "hit_rate": round(clv_stats.get("zero_clv_hits", 0) / clv_stats.get("zero_clv", 1) * 100, 1) if clv_stats.get("zero_clv", 0) > 0 else 0
                },
                {
                    "bucket": "negative",
                    "total": clv_stats.get("negative_clv", 0),
                    "hits": clv_stats.get("negative_clv_hits", 0),
                    "hit_rate": round(clv_stats.get("negative_clv_hits", 0) / clv_stats.get("negative_clv", 1) * 100, 1) if clv_stats.get("negative_clv", 0) > 0 else 0
                }
            ]
        },
        "recent": recent
    }


# Initialize database on startup
init_database()


# ─────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    test: bool = False  # If True, skip saving to database (for testing)


class PlayerLine(BaseModel):
    book: str
    line: float
    over_odds: int
    under_odds: int


class GameInfo(BaseModel):
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    vegas_total: Optional[float] = None
    spread: Optional[float] = None


class GameHistoryEntry(BaseModel):
    value: float
    date: str
    opponent: str


class PlayerStats(BaseModel):
    season_avg: float
    last_10_avg: float
    last_10_games: list[GameHistoryEntry]
    home_avg: float
    away_avg: float
    std_dev: float
    games_played: int
    minutes_avg: float
    usage_rate: float


class OpponentInfo(BaseModel):
    team_name: str
    def_rating: float
    pace: float
    def_rating_rank: int


class CatBoostPrediction(BaseModel):
    prob_over: Optional[float] = None
    prob_under: Optional[float] = None
    recommended_side: Optional[str] = None
    confidence: Optional[float] = None
    should_bet: Optional[bool] = None


class Analysis(BaseModel):
    model_projection: float
    adjusted_projection: float
    adjustment_factors: dict
    prob_over: float
    prob_under: float
    # No-vig fair market probabilities
    no_vig_prob_over: float
    no_vig_prob_under: float
    ev_over: float
    ev_under: float
    kelly_over: float
    kelly_under: float
    # Half-Kelly for practical bet sizing
    half_kelly_over: float
    half_kelly_under: float
    recommended_side: str
    edge: float
    # True edge vs no-vig market
    market_edge: float
    hit_rate_over: float
    hit_rate_under: float
    variance_rating: str
    confidence_grade: str
    summary: str
    # CatBoost ML prediction
    catboost: Optional[CatBoostPrediction] = None


class PropResponse(BaseModel):
    player: str
    team: str
    position: str
    prop_type: str  # points, rebounds, assists, pra, pr, pa, ra
    prop_label: str  # PTS, REB, AST, PTS+REB+AST, etc.
    game: GameInfo
    opponent: OpponentInfo
    lines: list[PlayerLine]
    best_over: dict
    best_under: dict
    stats: PlayerStats
    analysis: Analysis
    manual_mode: bool = False  # True when user provided line/odds manually




# ─────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────

def parse_query(query: str) -> tuple[str, str]:
    """
    Parse query to extract player name and prop type.
    Returns (player_name, prop_type)

    Supported formats:
    - "LeBron James points" → ("Lebron James", "points")
    - "LeBron James rebounds" → ("Lebron James", "rebounds")
    - "LeBron James assists" → ("Lebron James", "assists")
    - "LeBron James pra" → ("Lebron James", "pra")
    - "LeBron James pr" → ("Lebron James", "pr")
    """
    # First, strip any manual line/odds from the query for basic parsing
    # We'll handle those separately in parse_query_with_manual_line
    query_clean = query.lower().strip()

    # Remove any trailing numbers/odds patterns for basic prop detection
    import re
    # Pattern to match: optional o/u, number, optional odds like -110 or +105
    manual_pattern = r'\s+[ou]?\d+\.?\d*\s*[+-]?\d*\s*$'
    query_clean = re.sub(manual_pattern, '', query_clean).strip()

    # Define prop type patterns - check from END of query
    # Order matters: check longer/combo patterns first
    prop_patterns = [
        # Combinations (longest first)
        ("pra", ["pra", "p+r+a", "pts+reb+ast", "points+rebounds+assists", "pts reb ast", "points rebounds assists"]),
        ("pr", ["pr", "p+r", "pts+reb", "points+rebounds", "pts reb", "points rebounds", "pts+rebs"]),
        ("pa", ["pa", "p+a", "pts+ast", "points+assists", "pts ast", "points assists", "pts+asts"]),
        ("ra", ["ra", "r+a", "reb+ast", "rebounds+assists", "reb ast", "rebounds assists", "rebs+asts"]),
        # Individual props
        ("rebounds", ["rebounds", "rebs", "reb", "boards"]),
        ("assists", ["assists", "ast", "asts", "assist", "dimes"]),
        ("points", ["points", "pts", "point"]),
    ]

    detected_prop = "points"  # default
    player_name = query_clean

    # Check for prop type at the END of the query
    for prop_type, patterns in prop_patterns:
        for pattern in patterns:
            # Check if query ends with the pattern (with optional space before)
            if query_clean.endswith(" " + pattern) or query_clean == pattern:
                detected_prop = prop_type
                # Remove the pattern from the end
                if query_clean.endswith(" " + pattern):
                    player_name = query_clean[: -(len(pattern) + 1)].strip()
                else:
                    player_name = ""
                break
        if detected_prop != "points":
            break

    # Clean up the player name
    player_name = player_name.strip()
    # Remove any leftover + signs or extra spaces
    player_name = player_name.replace("+", " ").strip()
    player_name = " ".join(player_name.split())  # normalize spaces

    # Convert to title case
    player_name = player_name.title()

    return player_name, detected_prop


def parse_query_with_manual_line(query: str) -> tuple[str, str, Optional[float], Optional[int]]:
    """
    Parse query to extract player name, prop type, and optional manual line/odds.
    Returns (player_name, prop_type, manual_line, manual_odds)

    Supported formats:
    - "LeBron James points" → ("LeBron James", "points", None, None)
    - "LeBron James points 25.5" → ("LeBron James", "points", 25.5, None)
    - "LeBron James points 25.5 -115" → ("LeBron James", "points", 25.5, -115)
    - "LeBron points o25.5 -110" → ("LeBron", "points", 25.5, -110)
    - "LeBron points u24.5 +100" → ("LeBron", "points", 24.5, 100)
    """
    import re

    query_lower = query.lower().strip()
    manual_line = None
    manual_odds = None

    # Pattern to extract: optional o/u prefix, line number, optional odds
    # Examples: "25.5", "o25.5", "u24.5 -110", "25.5 +105"
    line_odds_pattern = r'\s+([ou])?(\d+\.?\d*)\s*([+-]\d+)?\s*$'
    match = re.search(line_odds_pattern, query_lower)

    if match:
        # Extract line (ignore o/u prefix for now - we determine side from analysis)
        manual_line = float(match.group(2))

        # Extract odds if present
        if match.group(3):
            manual_odds = int(match.group(3))

    # Get player name and prop type using existing logic
    player_name, prop_type = parse_query(query)

    return player_name, prop_type, manual_line, manual_odds


def parse_player_name(query: str) -> str:
    """Legacy function - extracts just the player name"""
    player_name, _ = parse_query(query)
    return player_name


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize('NFKD', name)
    ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    # Remove dots (P.J. → PJ), hyphens, apostrophes
    ascii_name = ascii_name.lower().replace(".", "").replace("-", " ").replace("'", "").strip()
    # Remove common suffixes
    suffixes = [" jr", " sr", " iii", " ii", " iv", " v"]
    for suffix in suffixes:
        if ascii_name.endswith(suffix):
            ascii_name = ascii_name[:-len(suffix)].strip()
    return ascii_name


def names_match(search_name: str, api_name: str) -> bool:
    search_normalized = normalize_name(search_name)
    api_normalized = normalize_name(api_name)
    if search_normalized == api_normalized:
        return True
    search_parts = search_normalized.split()
    api_parts = api_normalized.split()
    return all(part in api_parts for part in search_parts)


def get_current_season() -> str:
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[2:]}"


# ─────────────────────────────────────────────────────────────────
# Math/Analysis Functions
# ─────────────────────────────────────────────────────────────────

def american_to_implied_prob(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def american_to_decimal(odds: int) -> float:
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)


def calculate_ev(win_prob: float, odds: int) -> float:
    decimal_odds = american_to_decimal(odds)
    return (win_prob * decimal_odds) - 1


def calculate_kelly(win_prob: float, odds: int) -> float:
    decimal_odds = american_to_decimal(odds)
    kelly = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
    return max(0, kelly)


def calculate_no_vig_probability(over_odds: int, under_odds: int) -> tuple[float, float]:
    """
    Remove the vig/juice from sportsbook odds to get fair market probabilities.
    This is the industry standard method for determining true market prices.
    """
    over_implied = american_to_implied_prob(over_odds)
    under_implied = american_to_implied_prob(under_odds)

    # Total implied will be > 100% due to vig
    total_implied = over_implied + under_implied

    # Remove vig by normalizing
    no_vig_over = over_implied / total_implied
    no_vig_under = under_implied / total_implied

    return no_vig_over, no_vig_under


def calculate_minutes_adjusted_projection(
    base_projection: float,
    minutes_avg: float,
    last_10_minutes: float = None
) -> tuple[float, float]:
    """
    Adjust projection based on expected minutes.
    Returns (adjusted_projection, minutes_factor)
    """
    # Estimate points per minute
    if minutes_avg <= 0:
        return base_projection, 0.0

    ppm = base_projection / minutes_avg

    # Use recent minutes if available, otherwise use season avg
    expected_minutes = last_10_minutes if last_10_minutes else minutes_avg

    # Adjust if significant difference from baseline
    baseline_minutes = 32.0  # League average starter minutes
    minutes_diff = expected_minutes - baseline_minutes

    # Only adjust if player's minutes are notably different from baseline
    if abs(minutes_diff) > 3:
        adjustment = (minutes_diff / baseline_minutes) * base_projection * 0.15
        adjustment = min(max(adjustment, -3), 3)  # Cap adjustment
        return base_projection + adjustment, round(adjustment, 2)

    return base_projection, 0.0


def calculate_form_weighted_projection(
    season_avg: float,
    last_10_avg: float,
    last_10_games: list[float],
    std_dev: float
) -> tuple[float, dict]:
    """
    Enhanced form weighting that considers trend and consistency.
    Returns (projection, form_factors)
    """
    factors = {}

    # Base weighting: 60% L10, 40% season
    base_weight_l10 = 0.60
    base_weight_season = 0.40

    # Adjust weights based on form consistency
    if len(last_10_games) >= 5:
        recent_5 = last_10_games[:5]
        older_5 = last_10_games[5:10] if len(last_10_games) >= 10 else last_10_games[5:]

        recent_5_avg = sum(recent_5) / len(recent_5)

        # Detect hot/cold streaks
        if older_5:
            older_5_avg = sum(older_5) / len(older_5)
            trend = recent_5_avg - older_5_avg

            if trend > 3:  # Hot streak
                factors["trend"] = "hot"
                factors["trend_adj"] = round(trend * 0.2, 2)
            elif trend < -3:  # Cold streak
                factors["trend"] = "cold"
                factors["trend_adj"] = round(trend * 0.2, 2)
            else:
                factors["trend"] = "stable"
                factors["trend_adj"] = 0.0
        else:
            factors["trend"] = "stable"
            factors["trend_adj"] = 0.0

        factors["last_5_avg"] = round(recent_5_avg, 1)
    else:
        factors["trend"] = "insufficient_data"
        factors["trend_adj"] = 0.0

    # Calculate weighted projection
    projection = (last_10_avg * base_weight_l10) + (season_avg * base_weight_season)
    projection += factors.get("trend_adj", 0.0)

    factors["l10_weight"] = base_weight_l10
    factors["season_weight"] = base_weight_season

    return round(projection, 1), factors


def calculate_minutes_based_projection(
    prop_type: str,
    full_game_log: list[dict],
    minutes_data: dict,
    opponent_info: Optional[dict] = None,
    is_home: bool = True,
    min_minutes_threshold: int = 20,
    player_position: str = None,
    opponent_abbr: str = None
) -> tuple[float, dict]:
    """
    Minutes-first projection system using 75/25 formula.

    Methodology:
    1. Filter out low-minute games (< threshold) to remove garbage time/injuries
    2. Calculate stats per minute from filtered games
    3. Project minutes using weighted average (65% season, 35% last 5)
    4. Apply matchup adjustments (defense rating, pace)
    5. Final projection = projected_minutes × stats_per_minute × adjustments

    Returns (projection, factors_dict)
    """
    factors = {}

    # Map prop type to stat key
    prop_to_key = {
        "points": "PTS", "rebounds": "REB", "assists": "AST",
        "pra": "PRA", "pr": "PR", "pa": "PA", "ra": "RA"
    }
    stat_key = prop_to_key.get(prop_type, "PTS")

    # STEP 1: Filter games by minutes threshold
    filtered_games = [g for g in full_game_log if g.get("MIN", 0) >= min_minutes_threshold]
    factors["games_filtered"] = len(full_game_log) - len(filtered_games)
    factors["games_used"] = len(filtered_games)

    if len(filtered_games) < 5:
        # Not enough filtered games, fall back to all games
        filtered_games = full_game_log
        factors["filter_applied"] = False
    else:
        factors["filter_applied"] = True

    # STEP 2: Calculate stats per minute with weighted recency
    # Weight: last 5 (50%) + next 5 (30%) + rest of season (20%)
    last_5 = filtered_games[:5]
    next_5 = filtered_games[5:10] if len(filtered_games) >= 10 else filtered_games[5:]
    rest = filtered_games[10:] if len(filtered_games) > 10 else []

    def calc_spm(games):
        """Calculate stats per minute for a set of games"""
        if not games:
            return 0
        total_stat = sum(g.get(stat_key, 0) for g in games)
        total_min = sum(g.get("MIN", 1) for g in games)  # Avoid div by 0
        return total_stat / total_min if total_min > 0 else 0

    spm_last_5 = calc_spm(last_5)
    spm_next_5 = calc_spm(next_5)
    spm_rest = calc_spm(rest)

    # Weighted stats per minute
    if rest:
        stats_per_minute = (spm_last_5 * 0.50) + (spm_next_5 * 0.30) + (spm_rest * 0.20)
    elif next_5:
        stats_per_minute = (spm_last_5 * 0.60) + (spm_next_5 * 0.40)
    else:
        stats_per_minute = spm_last_5

    factors["stats_per_minute"] = round(stats_per_minute, 3)
    factors["spm_last_5"] = round(spm_last_5, 3)

    # STEP 3: Project minutes using 75/25 formula
    # 75% season average + 25% last 5 games (research baseline - stable, less reactive)
    season_minutes = minutes_data.get("season_avg", 30)
    last_5_minutes = minutes_data.get("last_5_avg", season_minutes)

    projected_minutes = (season_minutes * 0.75) + (last_5_minutes * 0.25)

    factors["projected_minutes"] = round(projected_minutes, 1)
    factors["season_minutes"] = season_minutes
    factors["last_5_minutes"] = last_5_minutes

    # STEP 4: Base projection
    base_projection = projected_minutes * stats_per_minute
    factors["base_projection"] = round(base_projection, 1)

    # STEP 5: Apply matchup adjustments
    matchup_multiplier = 1.0

    # 5a. Defense adjustment - use DvP (position-specific) if available, else fall back to generic
    dvp_used = False
    if player_position and opponent_abbr:
        # Try position-specific DvP adjustment
        # Check if we have DvP data for this team/position
        dvp_cache = load_dvp_cache()
        if dvp_cache and opponent_abbr in dvp_cache.get("teams", {}):
            team_dvp = dvp_cache["teams"][opponent_abbr]
            if player_position in team_dvp:
                # DvP data exists - use it (even if multiplier is 1.0 for avg defense)
                dvp_used = True
                dvp_multiplier = get_dvp_adjustment(opponent_abbr, player_position, prop_type)
                matchup_multiplier *= dvp_multiplier
                factors["defense_adj"] = round((dvp_multiplier - 1) * 100, 1)
                factors["dvp_position"] = player_position
                factors["dvp_used"] = True
                factors["dvp_rank"] = team_dvp[player_position].get("pts_rank", "N/A")

    if not dvp_used and opponent_info:
        # Fall back to generic def_rating
        def_rating = opponent_info.get("def_rating", 113.5)
        league_avg_def = 113.5
        # Higher def rating = worse defense = boost projection
        def_diff = def_rating - league_avg_def
        # Scale: every 5 points of def rating diff = ~5% adjustment
        def_adj = (def_diff / 5) * 0.05
        def_adj = max(min(def_adj, 0.15), -0.15)  # Cap at ±15%
        matchup_multiplier *= (1 + def_adj)
        factors["defense_adj"] = round(def_adj * 100, 1)
        factors["opp_def_rating"] = def_rating
        factors["dvp_used"] = False

    # 5b. Pace adjustment
    if opponent_info:
        pace = opponent_info.get("pace", 100.0)
        league_avg_pace = 100.0
        pace_diff = pace - league_avg_pace
        # Scale: every 5 pace points = ~3% adjustment
        pace_adj = (pace_diff / 5) * 0.03
        pace_adj = max(min(pace_adj, 0.10), -0.10)  # Cap at ±10%
        matchup_multiplier *= (1 + pace_adj)
        factors["pace_adj"] = round(pace_adj * 100, 1)
        factors["opp_pace"] = pace

    # 5c. Home/away adjustment (±3%)
    if is_home:
        matchup_multiplier *= 1.02
        factors["venue_adj"] = 2.0
    else:
        matchup_multiplier *= 0.98
        factors["venue_adj"] = -2.0

    factors["total_matchup_adj"] = round((matchup_multiplier - 1) * 100, 1)

    # STEP 6: Final projection
    final_projection = base_projection * matchup_multiplier

    # Detect trend (hot/cold streak)
    if len(last_5) >= 5 and len(next_5) >= 3:
        last_5_avg = sum(g.get(stat_key, 0) for g in last_5) / len(last_5)
        next_5_avg = sum(g.get(stat_key, 0) for g in next_5) / len(next_5)
        trend = last_5_avg - next_5_avg
        if trend > 3:
            factors["trend"] = "hot"
            final_projection += trend * 0.15  # Small trend bonus
        elif trend < -3:
            factors["trend"] = "cold"
            final_projection += trend * 0.15  # Small trend penalty
        else:
            factors["trend"] = "stable"
    else:
        factors["trend"] = "insufficient_data"

    return round(final_projection, 1), factors


def normal_cdf(x: float, mean: float, std: float) -> float:
    if std == 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / std
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def estimate_probability(line: float, projection: float, std: float) -> tuple[float, float]:
    prob_under = normal_cdf(line, projection, std)
    prob_over = 1 - prob_under
    return prob_over, prob_under


def get_variance_rating(std: float, avg: float) -> str:
    if avg == 0:
        return "Unknown"
    cv = std / avg
    if cv < 0.2:
        return "Very Low"
    elif cv < 0.3:
        return "Low"
    elif cv < 0.4:
        return "Medium"
    elif cv < 0.5:
        return "High"
    else:
        return "Very High"


def get_confidence_grade(
    games_played: int,
    std_dev: float,
    avg: float,
    minutes_avg: float,
    has_opponent_data: bool = True
) -> str:
    """
    Confidence grade based on projection RELIABILITY, not edge or probability.

    Answers: "How much can I trust this projection?"

    Factors:
    - Player consistency (CV = std_dev / avg): How predictable is this player?
    - Sample size (games_played): How much data do we have?
    - Minutes volume (minutes_avg): More minutes = more reliable opportunity
    - Data completeness: Do we have opponent matchup data?

    Returns: A, B, C, or D grade
    """
    score = 0

    # 1. Consistency score (40 pts max)
    # Lower coefficient of variation = more consistent = higher confidence
    if avg > 0:
        cv = std_dev / avg
        if cv < 0.2:
            score += 40  # Very consistent
        elif cv < 0.3:
            score += 30  # Consistent
        elif cv < 0.4:
            score += 20  # Moderate
        elif cv < 0.5:
            score += 10  # Variable
        # cv >= 0.5 = Very variable = 0 pts

    # 2. Sample size score (30 pts max)
    # More games = more reliable projection
    if games_played >= 50:
        score += 30
    elif games_played >= 30:
        score += 20
    elif games_played >= 15:
        score += 10
    # < 15 games = 0 pts

    # 3. Minutes volume score (20 pts max)
    # More minutes = starter role = more reliable stats
    if minutes_avg >= 32:
        score += 20  # Starter-level minutes
    elif minutes_avg >= 25:
        score += 15  # Rotation player
    elif minutes_avg >= 18:
        score += 10  # Bench contributor
    elif minutes_avg >= 10:
        score += 5   # Limited role
    # < 10 minutes = 0 pts

    # 4. Data completeness (10 pts)
    if has_opponent_data:
        score += 10

    # Grade thresholds (max 100 pts)
    if score >= 80:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


# ─────────────────────────────────────────────────────────────────
# NBA API Functions
# ─────────────────────────────────────────────────────────────────

def find_nba_player(name: str) -> Optional[dict]:
    from difflib import SequenceMatcher
    all_players = players.get_players()
    search_name = normalize_name(name)

    # Pass 1: Exact match
    for p in all_players:
        if normalize_name(p["full_name"]) == search_name:
            return p

    # Pass 2: Contains match (active players first)
    for p in all_players:
        if search_name in normalize_name(p["full_name"]):
            if p.get("is_active", False):
                return p

    # Pass 3: Contains match (any player)
    for p in all_players:
        if search_name in normalize_name(p["full_name"]):
            return p

    # Pass 4: Fuzzy match for typos (active players only, similarity > 0.85)
    best_match = None
    best_score = 0.85  # Minimum threshold

    for p in all_players:
        if not p.get("is_active", False):
            continue
        player_name = normalize_name(p["full_name"])
        score = SequenceMatcher(None, search_name, player_name).ratio()
        if score > best_score:
            best_score = score
            best_match = p

    if best_match:
        print(f"[Player Match] Fuzzy matched '{name}' → '{best_match['full_name']}' (score: {best_score:.2f})")
        return best_match

    return None


def get_days_rest(player_id: int, game_date: str = None) -> tuple[int, bool]:
    """
    Calculate days of rest for a player before a game.
    Returns (days_rest, is_back_to_back)
    """
    try:
        from datetime import datetime, timedelta
        season = get_current_season()
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            headers=NBA_API_HEADERS,
            timeout=30
        )
        games_df = gamelog.get_data_frames()[0]

        if games_df.empty:
            return None, None

        # Get the most recent game date
        last_game_date_str = games_df.iloc[0]["GAME_DATE"]  # Format: "JAN 15, 2025"
        last_game_date = datetime.strptime(last_game_date_str, "%b %d, %Y")

        # Calculate target date (today if not specified)
        if game_date:
            target_date = datetime.strptime(game_date, "%Y-%m-%d")
        else:
            target_date = datetime.now()

        # Calculate days rest
        days_rest = (target_date.date() - last_game_date.date()).days
        is_b2b = days_rest == 1

        return days_rest, is_b2b
    except Exception as e:
        print(f"[NBA API] Error calculating days rest: {e}")
        return None, None


def get_player_info(player_id: int, max_retries: int = 3) -> dict:
    """Get player team and position info with retry logic for flaky NBA API."""
    for attempt in range(max_retries):
        try:
            info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id,
                headers=NBA_API_HEADERS,
                timeout=30
            )
            data = info.get_data_frames()[0]
            return {
                "team": data["TEAM_NAME"].iloc[0] if not data.empty else "Unknown",
                "team_id": data["TEAM_ID"].iloc[0] if not data.empty else None,
                "team_abbr": data["TEAM_ABBREVIATION"].iloc[0] if not data.empty else "UNK",
                "position": data["POSITION"].iloc[0] if not data.empty else "Unknown"
            }
        except Exception as e:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"[NBA API] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"[NBA API] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[NBA API] All retries exhausted for player {player_id}")
                return {"team": "Unknown", "team_id": None, "team_abbr": "UNK", "position": "Unknown"}


def get_player_stats(player_id: int, max_retries: int = 3) -> dict:
    """Get player stats for all prop types from game log (points, rebounds, assists)"""
    season = get_current_season()
    print(f"[NBA API] Fetching stats for season: {season}")

    # Retry logic for flaky NBA API
    games_df = None
    for attempt in range(max_retries):
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                headers=NBA_API_HEADERS,
                timeout=30
            )
            games_df = gamelog.get_data_frames()[0]
            break  # Success, exit retry loop
        except Exception as e:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"[NBA API] Stats attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"[NBA API] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[NBA API] All retries exhausted for player {player_id}")
                raise HTTPException(status_code=500, detail=f"NBA API timeout after {max_retries} attempts")

    try:

        if games_df.empty:
            raise ValueError("No games found")

        # Add combination columns FIRST before creating views
        games_df["PRA"] = games_df["PTS"] + games_df["REB"] + games_df["AST"]
        games_df["PR"] = games_df["PTS"] + games_df["REB"]
        games_df["PA"] = games_df["PTS"] + games_df["AST"]
        games_df["RA"] = games_df["REB"] + games_df["AST"]

        # Now create home/away views (they'll include the new columns)
        home_games = games_df[games_df["MATCHUP"].str.contains("vs.")].copy()
        away_games = games_df[games_df["MATCHUP"].str.contains("@")].copy()
        minutes_avg = games_df["MIN"].mean() if "MIN" in games_df.columns else 0

        # Extract game details for last 10 games
        def get_game_details(row) -> dict:
            """Extract date and opponent from a game row"""
            matchup = row["MATCHUP"]
            game_date = row["GAME_DATE"]

            # Parse opponent from matchup (e.g., "LAL vs. PHI" or "LAL @ MIA")
            if " vs. " in matchup:
                opponent = matchup.split(" vs. ")[1]
                is_home = True
            elif " @ " in matchup:
                opponent = matchup.split(" @ ")[1]
                is_home = False
            else:
                opponent = "???"
                is_home = True

            # Parse date (format: "JAN 15, 2025" -> "1/15")
            try:
                from datetime import datetime
                dt = datetime.strptime(game_date, "%b %d, %Y")
                date_str = f"{dt.month}/{dt.day}"
            except:
                date_str = game_date[:6]

            return {
                "date": date_str,
                "opponent": ("" if is_home else "@") + opponent,
            }

        # Get details for last 10 games
        last_10_details = []
        for i in range(min(10, len(games_df))):
            row = games_df.iloc[i]
            details = get_game_details(row)
            last_10_details.append(details)

        def calc_stat(column: str) -> dict:
            """Calculate stats for a given column"""
            all_values = games_df[column].tolist()
            last_10_values = all_values[:10] if len(all_values) >= 10 else all_values
            home_avg = home_games[column].mean() if not home_games.empty else 0
            away_avg = away_games[column].mean() if not away_games.empty else 0

            # Build last 10 games with details (including minutes for filtering)
            last_10_games = []
            minutes_list = games_df["MIN"].tolist()[:10] if "MIN" in games_df.columns else []
            for i, val in enumerate(last_10_values):
                game_info = {
                    "value": float(val),
                    "minutes": float(minutes_list[i]) if i < len(minutes_list) and minutes_list[i] else 0,
                    "date": last_10_details[i]["date"] if i < len(last_10_details) else "",
                    "opponent": last_10_details[i]["opponent"] if i < len(last_10_details) else "",
                }
                last_10_games.append(game_info)

            return {
                "season_avg": round(games_df[column].mean(), 1),
                "last_10_avg": round(sum(last_10_values) / len(last_10_values), 1) if last_10_values else 0,
                "last_10_games": last_10_games,
                "home_avg": round(home_avg, 1),
                "away_avg": round(away_avg, 1),
                "std_dev": round(games_df[column].std(), 2),
            }

        # Build full game log with opponent info for avg_vs_opponent calculation
        full_game_log = []
        for i in range(len(games_df)):
            row = games_df.iloc[i]
            matchup = row["MATCHUP"]

            # Parse opponent from matchup
            if " vs. " in matchup:
                opponent = matchup.split(" vs. ")[1]
            elif " @ " in matchup:
                opponent = matchup.split(" @ ")[1]
            else:
                opponent = "???"

            full_game_log.append({
                "opponent": opponent,
                "MIN": float(row["MIN"]) if row["MIN"] else 0,
                "PTS": float(row["PTS"]),
                "REB": float(row["REB"]),
                "AST": float(row["AST"]),
                "PRA": float(row["PRA"]),
                "PR": float(row["PR"]),
                "PA": float(row["PA"]),
                "RA": float(row["RA"]),
            })

        # Calculate minutes statistics for projection
        all_minutes = games_df["MIN"].tolist() if "MIN" in games_df.columns else []
        last_5_minutes = all_minutes[:5] if len(all_minutes) >= 5 else all_minutes
        last_10_minutes = all_minutes[:10] if len(all_minutes) >= 10 else all_minutes

        minutes_data = {
            "season_avg": round(sum(all_minutes) / len(all_minutes), 1) if all_minutes else 0,
            "last_5_avg": round(sum(last_5_minutes) / len(last_5_minutes), 1) if last_5_minutes else 0,
            "last_10_avg": round(sum(last_10_minutes) / len(last_10_minutes), 1) if last_10_minutes else 0,
            "all_games": [float(m) for m in all_minutes],
        }

        return {
            "points": calc_stat("PTS"),
            "rebounds": calc_stat("REB"),
            "assists": calc_stat("AST"),
            "pra": calc_stat("PRA"),
            "pr": calc_stat("PR"),
            "pa": calc_stat("PA"),
            "ra": calc_stat("RA"),
            "games_played": len(games_df),
            "minutes_avg": round(minutes_avg, 1),
            "minutes_data": minutes_data,  # Added for minutes-first projection
            "full_game_log": full_game_log  # Added for avg_vs_opponent calculation
        }
    except Exception as e:
        print(f"[NBA API] Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch player stats: {str(e)}")


def get_player_advanced_stats(player_id: int) -> dict:
    """Get player usage rate and advanced metrics"""
    global _player_advanced_cache

    try:
        if _player_advanced_cache is None:
            season = get_current_season()
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                headers=NBA_API_HEADERS,
                timeout=30
            )
            _player_advanced_cache = stats.get_data_frames()[0]

        player_row = _player_advanced_cache[_player_advanced_cache["PLAYER_ID"] == player_id]

        if player_row.empty:
            return {"usage_rate": 0.20, "pace": 100.0}

        return {
            "usage_rate": round(player_row["USG_PCT"].iloc[0], 3),
            "pace": round(player_row["PACE"].iloc[0], 2),
            "off_rating": round(player_row["OFF_RATING"].iloc[0], 1),
            "def_rating": round(player_row["DEF_RATING"].iloc[0], 1)
        }
    except Exception as e:
        print(f"[NBA API] Error getting advanced stats: {e}")
        return {"usage_rate": 0.20, "pace": 100.0}


def get_team_stats() -> dict:
    """Get all team defensive ratings and pace (cached)"""
    global _team_stats_cache

    try:
        if _team_stats_cache is None:
            season = get_current_season()
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                headers=NBA_API_HEADERS,
                timeout=30
            )
            df = team_stats.get_data_frames()[0]

            _team_stats_cache = {}
            for _, row in df.iterrows():
                _team_stats_cache[row["TEAM_NAME"].lower()] = {
                    "team_id": row["TEAM_ID"],
                    "team_name": row["TEAM_NAME"],
                    "def_rating": round(row["DEF_RATING"], 1),
                    "off_rating": round(row["OFF_RATING"], 1),
                    "pace": round(row["PACE"], 2),
                    "def_rating_rank": int(row["DEF_RATING_RANK"]),
                }

        return _team_stats_cache
    except Exception as e:
        print(f"[NBA API] Error getting team stats: {e}")
        return {}


def get_opponent_stats(opponent_name: str) -> Optional[OpponentInfo]:
    """Get opponent's defensive rating and pace"""
    team_stats = get_team_stats()

    opponent_lower = opponent_name.lower()
    for team_key, stats in team_stats.items():
        if opponent_lower in team_key or team_key in opponent_lower:
            return OpponentInfo(
                team_name=stats["team_name"],
                def_rating=stats["def_rating"],
                pace=stats["pace"],
                def_rating_rank=stats["def_rating_rank"]
            )

    return None


# ─────────────────────────────────────────────────────────────────
# Odds API Functions
# ─────────────────────────────────────────────────────────────────

async def get_nba_events() -> list[dict]:
    """Fetch today's NBA games (FREE)"""
    url = f"{ODDS_API_BASE}/sports/basketball_nba/events"
    params = {"apiKey": ODDS_API_KEY}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Odds API error: {resp.text}")
        return resp.json()


async def get_game_odds(event_id: str) -> dict:
    """Fetch totals and spreads for a game"""
    url = f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals,spreads",
        "oddsFormat": "american",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        if resp.status_code != 200:
            return {}
        return resp.json()


def get_odds_api_market(prop_type: str) -> str:
    """Map prop type to Odds API market key"""
    market_map = {
        "points": "player_points",
        "rebounds": "player_rebounds",
        "assists": "player_assists",
        "pra": "player_points_rebounds_assists",
        "pr": "player_points_rebounds",
        "pa": "player_points_assists",
        "ra": "player_rebounds_assists",
    }
    return market_map.get(prop_type, "player_points")


async def get_player_props(event_id: str, prop_type: str = "points") -> dict:
    """Fetch player props for a game based on prop type"""
    global _odds_api_available, _odds_api_last_error

    market = get_odds_api_market(prop_type)
    url = f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market,
        "oddsFormat": "american",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)

            # Check for quota exceeded or other API errors
            if resp.status_code == 401 or resp.status_code == 429:
                _odds_api_available = False
                _odds_api_last_error = resp.text
                raise HTTPException(status_code=resp.status_code, detail=f"Odds API quota exceeded: {resp.text}")

            if resp.status_code != 200:
                # Check if error message indicates quota issues
                if "quota" in resp.text.lower() or "usage" in resp.text.lower():
                    _odds_api_available = False
                    _odds_api_last_error = resp.text
                raise HTTPException(status_code=resp.status_code, detail=f"Odds API error: {resp.text}")

            remaining = resp.headers.get("x-requests-remaining", "unknown")
            print(f"[Odds API] Requests remaining: {remaining}")

            # Check if we're running low on quota
            try:
                remaining_int = int(remaining)
                if remaining_int <= 0:
                    _odds_api_available = False
                    _odds_api_last_error = "No requests remaining"
                else:
                    _odds_api_available = True
            except (ValueError, TypeError):
                pass

            return resp.json()

    except httpx.TimeoutException:
        _odds_api_last_error = "Request timeout"
        raise HTTPException(status_code=504, detail="Odds API timeout")
    except httpx.RequestError as e:
        _odds_api_last_error = str(e)
        raise HTTPException(status_code=503, detail=f"Odds API connection error: {e}")


def extract_game_lines(odds_data: dict) -> dict:
    """Extract total and spread from game odds"""
    result = {"total": None, "spread": None}

    for bookmaker in odds_data.get("bookmakers", [])[:1]:
        for market in bookmaker.get("markets", []):
            if market.get("key") == "totals":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        result["total"] = outcome.get("point")
                        break
            elif market.get("key") == "spreads":
                for outcome in market.get("outcomes", []):
                    result["spread"] = outcome.get("point")
                    break

    return result


def extract_player_lines(props_data: dict, player_name: str, prop_type: str = "points") -> list[PlayerLine]:
    """Extract main line for each book (filters out alternate lines)

    Main line = the line with odds closest to -110/-110 (standard vig).
    This handles cases where different books have different main lines.
    """
    # Get the expected market key for this prop type
    expected_market = get_odds_api_market(prop_type)

    # First pass: collect ALL lines by book
    all_lines_by_book = {}

    for bookmaker in props_data.get("bookmakers", []):
        book_name = bookmaker.get("title", "Unknown")

        for market in bookmaker.get("markets", []):
            if market.get("key") != expected_market:
                continue

            for outcome in market.get("outcomes", []):
                description = outcome.get("description", "")

                if not names_match(player_name, description):
                    continue

                name = outcome.get("name")
                point = outcome.get("point", 0)
                price = outcome.get("price", 0)

                key = f"{book_name}_{point}"

                if key not in all_lines_by_book:
                    all_lines_by_book[key] = {
                        "book": book_name,
                        "line": point,
                        "over_odds": -110,
                        "under_odds": -110
                    }

                if name == "Over":
                    all_lines_by_book[key]["over_odds"] = price
                else:
                    all_lines_by_book[key]["under_odds"] = price

    if not all_lines_by_book:
        return []

    # Second pass: for each book, find their MAIN line
    # Main line = odds closest to -110/-110 (smallest deviation from standard vig)
    main_lines_by_book = {}

    for data in all_lines_by_book.values():
        book = data["book"]
        over_odds = data["over_odds"]
        under_odds = data["under_odds"]

        # Calculate how "standard" the odds are (distance from -110/-110)
        # Lower score = more likely to be the main line
        vig_score = abs(over_odds + 110) + abs(under_odds + 110)

        if book not in main_lines_by_book:
            main_lines_by_book[book] = (data, vig_score)
        else:
            current_score = main_lines_by_book[book][1]
            # Keep the line with odds closest to -110/-110
            if vig_score < current_score:
                main_lines_by_book[book] = (data, vig_score)

    return [PlayerLine(**data) for data, score in main_lines_by_book.values()]


# ─────────────────────────────────────────────────────────────────
# Enhanced Projection Model
# ─────────────────────────────────────────────────────────────────

def calculate_adjusted_projection(
    base_projection: float,
    player_stats: dict,
    opponent_info: Optional[OpponentInfo],
    game_info: dict,
    is_home: bool
) -> tuple[float, dict]:
    """
    Calculate adjusted projection based on multiple factors.
    Returns (adjusted_projection, adjustment_factors)
    """
    adjustments = {}
    projection = base_projection

    # 1. Home/Away adjustment
    if is_home:
        home_diff = player_stats.get("home_avg", base_projection) - player_stats.get("season_avg", base_projection)
        adj = min(max(home_diff * 0.3, -2), 2)
        adjustments["home_court"] = round(adj, 2)
        projection += adj
    else:
        away_diff = player_stats.get("away_avg", base_projection) - player_stats.get("season_avg", base_projection)
        adj = min(max(away_diff * 0.3, -2), 2)
        adjustments["road_game"] = round(adj, 2)
        projection += adj

    # 2. Opponent defense adjustment
    if opponent_info:
        league_avg_def = 113.5
        def_diff = opponent_info.def_rating - league_avg_def
        # Higher DEF_RATING = worse defense = more points
        adj = def_diff * 0.3
        adj = min(max(adj, -4), 4)
        adjustments["opponent_defense"] = round(adj, 2)
        adjustments["opp_def_rating"] = opponent_info.def_rating
        adjustments["opp_def_rank"] = opponent_info.def_rating_rank
        projection += adj

    # 3. Pace adjustment
    if opponent_info:
        league_avg_pace = 100.0
        pace_diff = opponent_info.pace - league_avg_pace
        adj = pace_diff * 0.2
        adj = min(max(adj, -3), 3)
        adjustments["pace_factor"] = round(adj, 2)
        adjustments["opp_pace"] = opponent_info.pace
        projection += adj

    # 4. Vegas total adjustment
    vegas_total = game_info.get("total")
    if vegas_total:
        avg_total = 225
        total_diff = vegas_total - avg_total
        usage = player_stats.get("usage_rate", 0.20)
        adj = (total_diff / 5) * 0.5 * (usage / 0.20)
        adj = min(max(adj, -2), 2)
        adjustments["vegas_total"] = round(adj, 2)
        adjustments["game_total"] = vegas_total
        projection += adj

    return round(projection, 1), adjustments


# ─────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "PropBot API",
        "version": "2.1.0",
        "status": "Odds API + NBA API + Prediction Tracking",
        "endpoints": {
            "GET /health": "Health check",
            "GET /events": "List today's NBA games",
            "POST /analyze": "Full analysis of a player prop (auto-saved)",
            "GET /predictions": "List all tracked predictions",
            "POST /sync-results": "Sync results for a date (default: yesterday)",
            "GET /stats": "View accuracy statistics",
        }
    }


@app.get("/health")
async def health_check():
    global _odds_api_available
    predictor = get_catboost_predictor()
    return {
        "status": "ok",
        "odds_api_configured": bool(ODDS_API_KEY),
        "odds_api_available": _odds_api_available,
        "ml_model_loaded": predictor is not None and predictor.is_loaded,
        "season": get_current_season(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/events")
async def list_events():
    events = await get_nba_events()
    return {
        "count": len(events),
        "events": [
            {
                "event_id": e["id"],
                "home_team": e["home_team"],
                "away_team": e["away_team"],
                "commence_time": e["commence_time"],
                "matchup": f"{e['away_team']} @ {e['home_team']}"
            }
            for e in events
        ]
    }


def get_prop_type_label(prop_type: str) -> str:
    """Get human-readable label for prop type"""
    labels = {
        "points": "PTS",
        "rebounds": "REB",
        "assists": "AST",
        "pra": "PTS+REB+AST",
        "pr": "PTS+REB",
        "pa": "PTS+AST",
        "ra": "REB+AST",
    }
    return labels.get(prop_type, "PTS")


@app.post("/analyze", response_model=PropResponse)
async def analyze_prop(request: QueryRequest):
    """Full analysis with Odds API + NBA API - supports points, rebounds, assists, and combinations"""
    global _odds_api_available

    # 1. Parse player name, prop type, and optional manual line/odds
    player_name, prop_type, manual_line, manual_odds = parse_query_with_manual_line(request.query)
    prop_label = get_prop_type_label(prop_type)

    # Determine if we're in manual mode (user provided line)
    manual_mode = manual_line is not None
    if manual_mode:
        print(f"[Analyze] MANUAL MODE: {player_name} ({prop_label}) line={manual_line} odds={manual_odds}")
    else:
        print(f"[Analyze] Searching for: {player_name} ({prop_label})")

    # 2. Find player in NBA database
    nba_player = find_nba_player(player_name)
    if not nba_player:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    player_id = nba_player["id"]
    full_name = nba_player["full_name"]
    print(f"[Analyze] Found: {full_name} (ID: {player_id})")

    # 3. Get player info and stats
    player_info = get_player_info(player_id)
    all_stats = get_player_stats(player_id)
    advanced_stats = get_player_advanced_stats(player_id)

    # Get stats for the specific prop type
    prop_stats = all_stats.get(prop_type, all_stats.get("points"))

    stats = PlayerStats(
        season_avg=prop_stats["season_avg"],
        last_10_avg=prop_stats["last_10_avg"],
        last_10_games=prop_stats["last_10_games"],
        home_avg=prop_stats["home_avg"],
        away_avg=prop_stats["away_avg"],
        std_dev=prop_stats["std_dev"],
        games_played=all_stats["games_played"],
        minutes_avg=all_stats["minutes_avg"],
        usage_rate=advanced_stats.get("usage_rate", 0.20)
    )
    print(f"[Analyze] Stats: {stats.season_avg} {prop_label}, {stats.usage_rate:.1%} USG, {stats.games_played} games")

    # 4. Get today's events and find player's game
    events = await get_nba_events()
    if not events:
        raise HTTPException(status_code=404, detail="No NBA games scheduled today")

    found_game = None
    for event in events:
        team_name = player_info["team"].lower()
        if team_name in event["home_team"].lower() or team_name in event["away_team"].lower():
            found_game = event
            break

    if not found_game:
        raise HTTPException(status_code=404, detail=f"No game today for {full_name} ({player_info['team']})")

    is_home = player_info["team"].lower() in found_game["home_team"].lower()
    opponent_name = found_game["away_team"] if is_home else found_game["home_team"]
    print(f"[Analyze] Game: {found_game['away_team']} @ {found_game['home_team']}")

    # 5. Get opponent stats
    opponent_info = get_opponent_stats(opponent_name)
    if opponent_info:
        print(f"[Analyze] Opponent: {opponent_info.team_name} (DEF: {opponent_info.def_rating}, PACE: {opponent_info.pace})")

    # 6. Get game odds (totals, spreads) - works even in manual mode (free endpoint)
    game_odds = await get_game_odds(found_game["id"])
    game_lines = extract_game_lines(game_odds)

    # 7. Get player props - either from API or manual input
    lines = None
    if manual_mode:
        # Manual mode: create a single line entry from user input
        default_odds = manual_odds if manual_odds else -110
        lines = [PlayerLine(
            book="Fanatics (Manual)",
            line=manual_line,
            over_odds=default_odds,
            under_odds=default_odds
        )]
        print(f"[Analyze] Using manual line: {manual_line} at {default_odds}")
    else:
        # Normal mode: fetch from Odds API
        try:
            props_data = await get_player_props(found_game["id"], prop_type)
            lines = extract_player_lines(props_data, full_name, prop_type)
        except HTTPException as e:
            # Check if this is a quota/availability error
            if "quota" in str(e.detail).lower() or "usage" in str(e.detail).lower() or e.status_code in [401, 429]:
                _odds_api_available = False
                raise HTTPException(
                    status_code=503,
                    detail=f"Odds API quota exceeded. Use manual mode by adding line and odds to your query. Example: '{player_name} {prop_type} 25.5 -110'"
                )
            raise e

    if not lines:
        raise HTTPException(status_code=404, detail=f"No {prop_label} prop found for {full_name}")

    # 8. Find best odds
    best_over = max(lines, key=lambda x: x.over_odds)
    best_under = max(lines, key=lambda x: x.under_odds)
    consensus_line = lines[0].line

    # 9. Calculate projections using minutes-first approach
    opponent_dict = {
        "def_rating": opponent_info.def_rating,
        "pace": opponent_info.pace,
        "def_rating_rank": opponent_info.def_rating_rank
    } if opponent_info else None

    # Get player position and opponent abbreviation for DvP lookup
    player_position = normalize_position(player_info.get("position", ""))
    opponent_abbr = get_team_abbr(opponent_name)

    adjusted_projection, projection_factors = calculate_minutes_based_projection(
        prop_type=prop_type,
        full_game_log=all_stats.get("full_game_log", []),
        minutes_data=all_stats.get("minutes_data", {}),
        opponent_info=opponent_dict,
        is_home=is_home,
        min_minutes_threshold=20,
        player_position=player_position,
        opponent_abbr=opponent_abbr
    )

    # Store base projection before matchup adjustments for comparison
    base_projection = projection_factors.get("base_projection", adjusted_projection)

    # Build adjustment_factors dict for display/storage
    adjustment_factors = {
        "projected_minutes": projection_factors.get("projected_minutes", stats.minutes_avg),
        "stats_per_minute": projection_factors.get("stats_per_minute", 0),
        "games_filtered": projection_factors.get("games_filtered", 0),
        "games_used": projection_factors.get("games_used", stats.games_played),
        "defense_adj": projection_factors.get("defense_adj", 0),
        "pace_adj": projection_factors.get("pace_adj", 0),
        "venue_adj": projection_factors.get("venue_adj", 0),
        "total_matchup_adj": projection_factors.get("total_matchup_adj", 0),
        "form_trend": projection_factors.get("trend", "stable"),
    }

    if opponent_info:
        adjustment_factors["opp_def_rating"] = opponent_info.def_rating
        adjustment_factors["opp_def_rank"] = opponent_info.def_rating_rank
        adjustment_factors["opp_pace"] = opponent_info.pace

    # Add DvP info if used
    if projection_factors.get("dvp_used"):
        adjustment_factors["dvp_used"] = True
        adjustment_factors["dvp_position"] = projection_factors.get("dvp_position")

    print(f"[Analyze] Projection: {base_projection} → {adjusted_projection} (75/25 formula)")
    print(f"[Analyze] SPM: {projection_factors.get('stats_per_minute', 0):.3f}, Proj Min: {projection_factors.get('projected_minutes', 0):.1f}, Trend: {projection_factors.get('trend', 'stable')}")

    # Log DvP usage
    if projection_factors.get("dvp_used"):
        print(f"[Analyze] DvP: {player_position} vs {opponent_abbr} → {projection_factors.get('defense_adj', 0):+.1f}% adj")
    else:
        print(f"[Analyze] Using generic DEF rating (DvP unavailable)")

    # 10. Calculate probabilities and EV
    prob_over, prob_under = estimate_probability(consensus_line, adjusted_projection, stats.std_dev)

    # 10b. Calculate no-vig fair market probabilities
    no_vig_over, no_vig_under = calculate_no_vig_probability(best_over.over_odds, best_under.under_odds)

    ev_over = round(calculate_ev(prob_over, best_over.over_odds) * 100, 2)
    ev_under = round(calculate_ev(prob_under, best_under.under_odds) * 100, 2)

    kelly_over = round(calculate_kelly(prob_over, best_over.over_odds) * 100, 2)
    kelly_under = round(calculate_kelly(prob_under, best_under.under_odds) * 100, 2)

    # 10c. Calculate half-Kelly for practical bet sizing
    half_kelly_over = round(kelly_over / 2, 2)
    half_kelly_under = round(kelly_under / 2, 2)

    # 11. Determine recommendation
    if ev_over > ev_under:
        recommended_side = "over"
        edge = prob_over - american_to_implied_prob(best_over.over_odds)
        market_edge = prob_over - no_vig_over
        best_odds = best_over.over_odds
        best_book = best_over.book
    else:
        recommended_side = "under"
        edge = prob_under - american_to_implied_prob(best_under.under_odds)
        market_edge = prob_under - no_vig_under
        best_odds = best_under.under_odds
        best_book = best_under.book

    # 12. Hit rates and confidence
    game_values = [g.value for g in stats.last_10_games]
    hit_rate_over = sum(1 for v in game_values if v > consensus_line) / len(game_values) if game_values else 0
    hit_rate_under = sum(1 for v in game_values if v < consensus_line) / len(game_values) if game_values else 0

    variance_rating = get_variance_rating(stats.std_dev, stats.season_avg)
    hit_rate = hit_rate_over if recommended_side == "over" else hit_rate_under

    # Confidence = projection reliability (not edge or probability)
    confidence = get_confidence_grade(
        games_played=stats.games_played,
        std_dev=stats.std_dev,
        avg=stats.season_avg,
        minutes_avg=stats.minutes_avg,
        has_opponent_data=opponent_info is not None
    )

    # 13. Generate summary
    side_word = "OVER" if recommended_side == "over" else "UNDER"
    odds_str = f"+{best_odds}" if best_odds > 0 else str(best_odds)

    opp_note = ""
    if opponent_info:
        if opponent_info.def_rating_rank <= 10:
            opp_note = f" vs tough defense (#{opponent_info.def_rating_rank})"
        elif opponent_info.def_rating_rank >= 20:
            opp_note = f" vs weak defense (#{opponent_info.def_rating_rank})"

    half_kelly_rec = half_kelly_over if recommended_side == "over" else half_kelly_under

    # Auto-save prediction to database for tracking
    game_date = utc_to_eastern_date(found_game["commence_time"])
    best_ev = ev_over if recommended_side == "over" else ev_under

    # Calculate days rest for ML features
    days_rest, is_b2b = get_days_rest(player_id, game_date)

    # Get the probability for the recommended side
    rec_prob = prob_over if recommended_side == "over" else prob_under
    rec_no_vig = no_vig_over if recommended_side == "over" else no_vig_under

    # Get CatBoost prediction
    catboost_prediction = None
    catboost_prob_over = None
    catboost_pick = None
    catboost_confidence = None

    predictor = get_catboost_predictor()
    if predictor and predictor.is_loaded:
        # Compute player's historical average vs this opponent (V2 feature)
        avg_vs_opponent = stats.season_avg  # Default to season avg
        full_game_log = all_stats.get('full_game_log', [])
        if full_game_log:
            # Filter game log for games vs this opponent
            opp_games = [g for g in full_game_log if g.get('opponent') == opponent_name]
            if opp_games:
                # Use the prop_type key directly (e.g., 'PTS', 'PRA')
                prop_key_map = {
                    'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
                    'pra': 'PRA', 'pr': 'PR', 'pa': 'PA', 'ra': 'RA'
                }
                prop_key = prop_key_map.get(prop_type, 'PTS')
                avg_vs_opponent = sum(g.get(prop_key, 0) for g in opp_games) / len(opp_games)
                print(f"[CatBoost] Historical avg vs {opponent_name}: {avg_vs_opponent:.1f} ({len(opp_games)} games)")

        # Build features for CatBoost V2
        # Use 75/25 projected minutes (season avg weighted)
        projected_minutes = projection_factors.get("projected_minutes", stats.minutes_avg)

        catboost_features = {
            "opponent_team": opponent_name,
            "prop_type": prop_type,
            "closing_line": consensus_line,
            "season_avg": stats.season_avg,
            "last_10_avg": stats.last_10_avg,
            "last_5_avg": sum(g.value for g in stats.last_10_games[:5]) / min(5, len(stats.last_10_games)) if stats.last_10_games else stats.last_10_avg,
            "std_dev": stats.std_dev,  # V2: needed for line_difficulty, consistency
            "minutes_avg": projected_minutes,  # 75% season + 25% last 5
            "days_rest": days_rest if days_rest is not None else 1,
            "is_home": 1 if is_home else 0,
            "is_b2b": 1 if is_b2b else 0,
            "games_played": stats.games_played,
            # V2 matchup features
            "avg_vs_opponent": avg_vs_opponent,
            "opp_def_rating": opponent_info.def_rating if opponent_info else 112.0,
            "opp_pace": opponent_info.pace if opponent_info else 100.0,
        }

        print(f"[CatBoost] Using projected minutes: {projected_minutes:.1f}")

        cb_result = predictor.predict(catboost_features)
        if cb_result:
            catboost_prob_over = cb_result["prob_over"]
            catboost_pick = cb_result["recommended_side"]
            catboost_confidence = cb_result["confidence"]
            catboost_prediction = CatBoostPrediction(
                prob_over=cb_result["prob_over"],
                prob_under=cb_result["prob_under"],
                recommended_side=cb_result["recommended_side"],
                confidence=cb_result["confidence"],
                should_bet=cb_result["should_bet"]
            )
            print(f"[CatBoost] Prediction: {catboost_pick.upper()} ({catboost_confidence:.1%})")

            # Use CatBoost as primary recommendation (retired rule-based)
            recommended_side = catboost_pick
            side_word = "OVER" if recommended_side == "over" else "UNDER"
            # Recalculate edge and best_odds for CatBoost side
            if recommended_side == "over":
                edge = catboost_prob_over - american_to_implied_prob(best_over.over_odds)
                market_edge = catboost_prob_over - no_vig_over
                best_odds = best_over.over_odds
                best_book = best_over.book
                half_kelly_rec = half_kelly_over
            else:
                edge = (1 - catboost_prob_over) - american_to_implied_prob(best_under.under_odds)
                market_edge = (1 - catboost_prob_over) - no_vig_under
                best_odds = best_under.under_odds
                best_book = best_under.book
                half_kelly_rec = half_kelly_under
            odds_str = f"+{best_odds}" if best_odds > 0 else str(best_odds)

    # Build summary with manual mode indicator if applicable
    manual_prefix = "[MANUAL MODE] " if manual_mode else ""

    if catboost_prediction:
        cb_conf = catboost_confidence * 100
        summary = (
            f"{manual_prefix}ML Recommendation: {side_word} {consensus_line} {prop_label} at {best_book} ({odds_str}). "
            f"{full_name} projects to {adjusted_projection} {prop_label}{opp_note}. "
            f"ML Confidence: {cb_conf:.0f}%. Edge: {round(edge * 100, 1)}%. "
            f"Suggested stake: {half_kelly_rec}% (half-Kelly). "
            f"Hit rate: {round(hit_rate * 100)}% L10."
        )
    else:
        # Fallback if CatBoost not available (shouldn't happen normally)
        summary = (
            f"{manual_prefix}Projection: {side_word} {consensus_line} {prop_label} at {best_book} ({odds_str}). "
            f"{full_name} projects to {adjusted_projection} {prop_label}{opp_note}. "
            f"Edge: {round(edge * 100, 1)}%, Market edge: {round(market_edge * 100, 1)}%. "
            f"Suggested stake: {half_kelly_rec}% (half-Kelly). "
            f"Hit rate: {round(hit_rate * 100)}% L10. Confidence: {confidence}."
        )

    analysis = Analysis(
        model_projection=base_projection,
        adjusted_projection=adjusted_projection,
        adjustment_factors=adjustment_factors,
        prob_over=round(prob_over, 4),
        prob_under=round(prob_under, 4),
        no_vig_prob_over=round(no_vig_over, 4),
        no_vig_prob_under=round(no_vig_under, 4),
        ev_over=ev_over,
        ev_under=ev_under,
        kelly_over=kelly_over,
        kelly_under=kelly_under,
        half_kelly_over=half_kelly_over,
        half_kelly_under=half_kelly_under,
        recommended_side=recommended_side,
        edge=round(edge, 4),
        market_edge=round(market_edge, 4),
        hit_rate_over=round(hit_rate_over, 2),
        hit_rate_under=round(hit_rate_under, 2),
        variance_rating=variance_rating,
        confidence_grade=confidence,
        summary=summary,
        catboost=catboost_prediction
    )

    # Only save to database if not a test query
    if not request.test:
        save_prediction(
            player_name=full_name,
            player_id=player_id,
            game_id=found_game["id"],
            game_date=game_date,
            line=consensus_line,
            projection=adjusted_projection,
            recommended_side=recommended_side,
            confidence_grade=confidence,
            ev=best_ev,
            edge=round(edge * 100, 2),
            best_odds=best_odds,
            best_book=best_book,
            prop_type=prop_type,
            opponent_team=opponent_name,
            is_home=is_home,
            vegas_total=game_lines.get("total"),
            spread=game_lines.get("spread"),
            season_avg=stats.season_avg,
            last_10_avg=stats.last_10_avg,
            std_dev=stats.std_dev,
            minutes_avg=projection_factors.get("projected_minutes", stats.minutes_avg),  # 75/25 formula
            opp_def_rating=opponent_info.def_rating if opponent_info else None,
            opp_pace=opponent_info.pace if opponent_info else None,
            prob_over=round(rec_prob, 4),
            no_vig_prob=round(rec_no_vig, 4),
            days_rest=days_rest,
            is_b2b=is_b2b,
            usage_rate=stats.usage_rate,
            home_avg=stats.home_avg,
            away_avg=stats.away_avg,
            model_projection=base_projection,
            catboost_prob_over=catboost_prob_over,
            catboost_pick=catboost_pick,
            catboost_confidence=catboost_confidence,
        )

    return PropResponse(
        player=full_name,
        team=player_info["team"],
        position=player_info["position"],
        prop_type=prop_type,
        prop_label=prop_label,
        game=GameInfo(
            event_id=found_game["id"],
            home_team=found_game["home_team"],
            away_team=found_game["away_team"],
            commence_time=found_game["commence_time"],
            vegas_total=game_lines.get("total"),
            spread=game_lines.get("spread")
        ),
        opponent=opponent_info or OpponentInfo(team_name=opponent_name, def_rating=113.5, pace=100.0, def_rating_rank=15),
        lines=lines,
        best_over={
            "book": best_over.book,
            "line": best_over.line,
            "odds": best_over.over_odds
        },
        best_under={
            "book": best_under.book,
            "line": best_under.line,
            "odds": best_under.under_odds
        },
        stats=stats,
        analysis=analysis,
        manual_mode=manual_mode
    )


# ─────────────────────────────────────────────────────────────────
# Tracking & Stats Endpoints
# ─────────────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    date: Optional[str] = None  # Format: YYYY-MM-DD, defaults to yesterday


@app.post("/sync-results")
async def sync_results(request: SyncRequest = None):
    """
    Sync prediction results for a specific date.
    Call this in the morning to update yesterday's predictions with actual results.
    """
    if request and request.date:
        game_date = request.date
    else:
        # Default to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        game_date = yesterday.strftime("%Y-%m-%d")

    print(f"[Sync] Syncing results for {game_date}")
    results = sync_results_for_date(game_date)

    return {
        "message": f"Synced results for {game_date}",
        "results": results
    }


@app.get("/stats")
async def get_stats():
    """
    Get accuracy statistics for all tracked predictions.
    """
    stats = get_accuracy_stats()
    return stats


@app.get("/predictions")
async def list_predictions(status: str = "all", limit: int = 50):
    """
    List predictions.
    status: 'all', 'pending', 'resolved'
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if status == "pending":
        cursor.execute("""
            SELECT * FROM predictions
            WHERE hit IS NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
    elif status == "resolved":
        cursor.execute("""
            SELECT * FROM predictions
            WHERE hit IS NOT NULL
            ORDER BY resolved_at DESC
            LIMIT ?
        """, (limit,))
    else:
        cursor.execute("""
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return {
        "count": len(rows),
        "predictions": [dict(row) for row in rows]
    }


@app.post("/backfill-features")
async def backfill_features():
    """
    Backfill ML features for existing predictions.
    Attempts to fetch historical data for predictions missing the new columns.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all predictions that are missing ML features
    cursor.execute("""
        SELECT id, player_id, player_name, game_date, prop_type, actual_result
        FROM predictions
        WHERE season_avg IS NULL OR opponent_team IS NULL
    """)
    predictions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    results = {
        "total": len(predictions),
        "updated": 0,
        "failed": 0,
        "details": []
    }

    for pred in predictions:
        try:
            player_id = pred["player_id"]
            player_name = pred["player_name"]
            game_date = pred["game_date"]
            prop_type = pred.get("prop_type", "points")

            # Get player stats
            all_stats = get_player_stats(player_id)
            prop_stats = all_stats.get(prop_type, all_stats.get("points"))
            advanced_stats = get_player_advanced_stats(player_id)

            # Get player info to determine team
            player_info = get_player_info(player_id)

            # Calculate days rest
            days_rest, is_b2b = get_days_rest(player_id, game_date)

            # Try to get actual minutes from game log if resolved
            actual_minutes = None
            if pred["actual_result"] is not None:
                try:
                    season = get_current_season()
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        headers=NBA_API_HEADERS,
                        timeout=30
                    )
                    games_df = gamelog.get_data_frames()[0]

                    # Find the game on this date
                    for _, row in games_df.iterrows():
                        row_date = datetime.strptime(row["GAME_DATE"], "%b %d, %Y").strftime("%Y-%m-%d")
                        if row_date == game_date:
                            actual_minutes = float(row["MIN"]) if row["MIN"] else None

                            # Also try to extract opponent from matchup
                            matchup = row["MATCHUP"]
                            if " vs. " in matchup:
                                opponent_team = matchup.split(" vs. ")[1]
                                is_home = True
                            elif " @ " in matchup:
                                opponent_team = matchup.split(" @ ")[1]
                                is_home = False
                            else:
                                opponent_team = None
                                is_home = None
                            break
                    else:
                        opponent_team = None
                        is_home = None
                except Exception as e:
                    print(f"[Backfill] Error getting game log for {player_name}: {e}")
                    opponent_team = None
                    is_home = None
            else:
                opponent_team = None
                is_home = None

            # Get opponent defensive stats if we have opponent
            opp_def_rating = None
            opp_pace = None
            if opponent_team:
                opponent_info = get_opponent_stats(opponent_team)
                if opponent_info:
                    opp_def_rating = opponent_info.def_rating
                    opp_pace = opponent_info.pace

            # Update the prediction
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions SET
                    opponent_team = COALESCE(opponent_team, ?),
                    is_home = COALESCE(is_home, ?),
                    season_avg = ?,
                    last_10_avg = ?,
                    std_dev = ?,
                    minutes_avg = ?,
                    opp_def_rating = COALESCE(opp_def_rating, ?),
                    opp_pace = COALESCE(opp_pace, ?),
                    days_rest = COALESCE(days_rest, ?),
                    is_b2b = COALESCE(is_b2b, ?),
                    usage_rate = ?,
                    home_avg = ?,
                    away_avg = ?,
                    actual_minutes = COALESCE(actual_minutes, ?)
                WHERE id = ?
            """, (
                opponent_team,
                1 if is_home else 0 if is_home is not None else None,
                prop_stats["season_avg"],
                prop_stats["last_10_avg"],
                prop_stats["std_dev"],
                all_stats["minutes_avg"],
                opp_def_rating,
                opp_pace,
                days_rest,
                1 if is_b2b else 0 if is_b2b is not None else None,
                advanced_stats.get("usage_rate", 0.20),
                prop_stats["home_avg"],
                prop_stats["away_avg"],
                actual_minutes,
                pred["id"]
            ))
            conn.commit()
            conn.close()

            results["updated"] += 1
            results["details"].append({
                "player": player_name,
                "game_date": game_date,
                "status": "updated",
                "actual_minutes": actual_minutes,
                "opponent": opponent_team
            })
            print(f"[Backfill] Updated {player_name} ({game_date})")

        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "player": pred["player_name"],
                "game_date": pred["game_date"],
                "status": "failed",
                "error": str(e)
            })
            print(f"[Backfill] Failed for {pred['player_name']}: {e}")

    return {
        "message": f"Backfill complete: {results['updated']} updated, {results['failed']} failed",
        "results": results
    }


# ============================================================
# MLB ENDPOINTS V2 (Completely separate from NBA)
# ============================================================

# Lazy load MLB modules to avoid import errors if not installed
_mlb_projection = None
_mlb_data = None

def get_mlb_projection():
    """Lazy load MLB projection module (V2)."""
    global _mlb_projection
    if _mlb_projection is None:
        try:
            import mlb_projection_v2
            _mlb_projection = mlb_projection_v2
        except ImportError as e:
            print(f"[MLB] Projection module not available: {e}")
    return _mlb_projection

def get_mlb_data():
    """Lazy load MLB data module."""
    global _mlb_data
    if _mlb_data is None:
        try:
            import mlb_data
            _mlb_data = mlb_data
        except ImportError as e:
            print(f"[MLB] Data module not available: {e}")
    return _mlb_data


class MLBAnalyzeRequest(BaseModel):
    """Request model for MLB analysis - supports simple query or full details."""
    query: Optional[str] = None  # Simple mode: just pitcher name like "Robbie Ray"
    pitcher: Optional[str] = None  # Full mode: pitcher name
    opponent: Optional[str] = None
    line: Optional[float] = None
    over_odds: Optional[int] = -110
    under_odds: Optional[int] = -110
    venue: Optional[str] = None
    vegas_total: Optional[float] = None
    spread: Optional[float] = None
    umpire_name: Optional[str] = None
    test: Optional[bool] = False


# MLB team abbreviation mapping for Odds API matching
MLB_TEAM_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


async def get_mlb_events() -> list:
    """Fetch today's MLB events from Odds API."""
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/events"
    params = {"apiKey": ODDS_API_KEY}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=30)
        if response.status_code == 200:
            remaining = response.headers.get('x-requests-remaining', 'N/A')
            print(f"[MLB] Odds API credits remaining: {remaining}")
            return response.json()
        else:
            print(f"[MLB] Error fetching events: {response.status_code}")
            return []


async def get_mlb_pitcher_props(event_id: str) -> dict:
    """Fetch pitcher strikeout props for an MLB game."""
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "pitcher_strikeouts",
        "oddsFormat": "american",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=30)
        if response.status_code == 200:
            remaining = response.headers.get('x-requests-remaining', 'N/A')
            print(f"[MLB] Odds API credits remaining: {remaining}")
            return response.json()
        else:
            print(f"[MLB] Error fetching props: {response.status_code} - {response.text[:200]}")
            return {}


def find_pitcher_in_props(props_data: dict, pitcher_name: str) -> Optional[dict]:
    """Find a pitcher's strikeout line in the props data."""
    bookmakers = props_data.get('bookmakers', [])
    if not bookmakers:
        return None

    pitcher_lower = pitcher_name.lower()
    pitcher_last = pitcher_lower.split()[-1] if pitcher_lower else ""

    lines_by_book = []

    for bookmaker in bookmakers:
        book_name = bookmaker.get('title', bookmaker.get('key', 'Unknown'))
        markets = bookmaker.get('markets', [])

        for market in markets:
            if market.get('key') != 'pitcher_strikeouts':
                continue
            outcomes = market.get('outcomes', [])

            for outcome in outcomes:
                description = outcome.get('description', '').lower()

                # Match pitcher name
                if pitcher_lower in description or pitcher_last in description:
                    point = outcome.get('point')
                    price = outcome.get('price')
                    name = outcome.get('name', '')  # 'Over' or 'Under'

                    if point is not None:
                        lines_by_book.append({
                            'book': book_name,
                            'line': float(point),
                            'odds': price,
                            'side': name.lower()
                        })

    if not lines_by_book:
        return None

    # Group by line and find consensus
    from collections import defaultdict
    line_groups = defaultdict(list)
    for entry in lines_by_book:
        line_groups[entry['line']].append(entry)

    # Get most common line
    consensus_line = max(line_groups.keys(), key=lambda x: len(line_groups[x]))
    entries = line_groups[consensus_line]

    # Extract over/under odds
    over_odds = -110
    under_odds = -110
    best_book = entries[0]['book'] if entries else 'Unknown'

    for e in entries:
        if e['side'] == 'over':
            over_odds = e['odds']
        elif e['side'] == 'under':
            under_odds = e['odds']

    return {
        'line': consensus_line,
        'over_odds': over_odds,
        'under_odds': under_odds,
        'best_book': best_book
    }


class MLBPropResponse(BaseModel):
    """Response model for MLB prop analysis."""
    sport: str = "mlb"
    pitcher: str
    opponent: str
    projected_ks: float
    line: Optional[float]
    prob_over: float
    prob_under: float
    ev_over: Optional[float]
    ev_under: Optional[float]
    recommended_side: Optional[str]
    edge: Optional[float]
    confidence_grade: str
    inputs: dict
    saved: bool = False


@app.get("/mlb/games")
async def get_mlb_games():
    """Get today's MLB games with probable pitchers."""
    mlb_data = get_mlb_data()
    if not mlb_data:
        raise HTTPException(status_code=500, detail="MLB module not available")

    try:
        games = mlb_data.get_todays_games()
        return {"games": games, "count": len(games)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlb/health")
async def get_mlb_health():
    """Get MLB cache status and health check."""
    mlb_data = get_mlb_data()
    if not mlb_data:
        return {"status": "unavailable", "message": "MLB module not loaded"}

    try:
        cache_status = mlb_data.get_cache_status()
        league_avg = mlb_data.get_league_averages()

        return {
            "status": "ok",
            "caches": cache_status,
            "league_averages": league_avg,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/mlb/pitcher/{name}")
async def get_mlb_pitcher(name: str):
    """Get pitcher stats and recent game logs."""
    mlb_data = get_mlb_data()
    if not mlb_data:
        raise HTTPException(status_code=500, detail="MLB module not available")

    pitcher = mlb_data.get_pitcher(name)
    if not pitcher:
        raise HTTPException(status_code=404, detail=f"Pitcher not found: {name}")

    return pitcher


@app.get("/mlb/team/{abbrev}")
async def get_mlb_team(abbrev: str):
    """Get team batting stats and splits."""
    mlb_data = get_mlb_data()
    if not mlb_data:
        raise HTTPException(status_code=500, detail="MLB module not available")

    team = mlb_data.get_team(abbrev)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team not found: {abbrev}")

    return team


@app.post("/mlb/analyze")
async def analyze_mlb_prop(request: MLBAnalyzeRequest):
    """
    Analyze MLB pitcher strikeout prop using V2 projection engine.

    Simple mode (auto-fetch line from Odds API):
        {"query": "Robbie Ray"}

    Full mode (manual line):
        {"pitcher": "Logan Webb", "opponent": "Yankees", "line": 5.5, ...}
    """
    mlb_proj = get_mlb_projection()
    if not mlb_proj:
        raise HTTPException(status_code=500, detail="MLB projection module not available")

    try:
        # Determine mode: simple query vs full details
        pitcher_name = request.query or request.pitcher
        if not pitcher_name:
            raise HTTPException(status_code=400, detail="Pitcher name required (use 'query' or 'pitcher' field)")

        pitcher_name = pitcher_name.strip()
        print(f"[MLB] Analyzing: {pitcher_name}")

        opponent = request.opponent
        line = request.line
        over_odds = request.over_odds
        under_odds = request.under_odds
        venue = request.venue
        best_book = None

        # Auto-detect opponent and game info if not provided
        # This runs for both simple mode (query) and manual mode (pitcher + line)
        if not request.opponent:
            print(f"[MLB] Auto-detecting opponent for {pitcher_name}...")

            # Normalize accents for matching (López -> Lopez)
            import unicodedata
            def normalize_name(name: str) -> str:
                return unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('ascii').lower()

            pitcher_lower = normalize_name(pitcher_name)
            pitcher_last = pitcher_lower.split()[-1]

            # Check statsapi for pitcher schedule
            import statsapi
            from datetime import datetime
            today = datetime.now().strftime('%m/%d/%Y')
            schedule = statsapi.schedule(start_date=today, end_date=today)

            for game in schedule:
                home_pitcher = normalize_name(game.get('home_probable_pitcher', ''))
                away_pitcher = normalize_name(game.get('away_probable_pitcher', ''))

                if pitcher_lower in home_pitcher or pitcher_last in home_pitcher:
                    opponent = game['away_name']
                    if not venue:
                        venue = game.get('venue_name')
                    break
                elif pitcher_lower in away_pitcher or pitcher_last in away_pitcher:
                    opponent = game['home_name']
                    if not venue:
                        venue = game.get('venue_name')
                    break

            if not opponent:
                raise HTTPException(status_code=404, detail=f"No game found today for pitcher: {pitcher_name}")

            # Convert opponent to abbreviation
            opponent_abbrev = MLB_TEAM_ABBREV.get(opponent, opponent)
            print(f"[MLB] Found game: {pitcher_name} vs {opponent} ({opponent_abbrev}) @ {venue}")
            opponent = opponent_abbrev

            # If line not provided, fetch from Odds API
            if not line:
                print(f"[MLB] Simple mode - fetching line from Odds API...")
                events = await get_mlb_events()
                if events:
                    # Find matching Odds API event
                    found_event = None
                    for game in schedule:
                        home_pitcher = normalize_name(game.get('home_probable_pitcher', ''))
                        away_pitcher = normalize_name(game.get('away_probable_pitcher', ''))
                        if pitcher_lower in home_pitcher or pitcher_last in home_pitcher or \
                           pitcher_lower in away_pitcher or pitcher_last in away_pitcher:
                            for event in events:
                                if game['home_name'].lower() in event.get('home_team', '').lower():
                                    found_event = event
                                    break
                            break

                    if found_event:
                        props_data = await get_mlb_pitcher_props(found_event['id'])
                        pitcher_line = find_pitcher_in_props(props_data, pitcher_name)

                        if pitcher_line:
                            line = pitcher_line['line']
                            over_odds = pitcher_line['over_odds']
                            under_odds = pitcher_line['under_odds']
                            best_book = pitcher_line['best_book']
                            print(f"[MLB] Found line: {line} (Over {over_odds}, Under {under_odds}) @ {best_book}")
                        else:
                            raise HTTPException(status_code=404, detail=f"No strikeout line found for {pitcher_name}. Try manual mode: {{\"pitcher\": \"{pitcher_name}\", \"opponent\": \"{opponent_abbrev}\", \"line\": 5.5}}")

        if line is None:
            raise HTTPException(status_code=400, detail="Line is required. Use simple mode with just pitcher name, or provide line manually.")

        # Try Monte Carlo simulation first (requires confirmed lineup)
        result = None
        model_type = None

        print(f"[MLB] DEBUG: Calling simulation with pitcher={pitcher_name}, opponent={opponent}, line={line}, vegas_total={request.vegas_total}")

        try:
            from mlb_simulation import analyze_prop_simulation
            result = analyze_prop_simulation(
                pitcher_name=pitcher_name,
                opponent_abbrev=opponent,
                line=line,
                over_odds=over_odds or -110,
                under_odds=under_odds or -110,
                venue=venue,
                n_sims=5000,
                vegas_total=request.vegas_total
            )
            if result:
                model_type = 'simulation'
                print(f"[MLB] Using Monte Carlo simulation (lineup confirmed)")
        except Exception as e:
            print(f"[MLB] Simulation failed: {e}, falling back to Beta-Binomial")
            result = None

        # Fall back to Beta-Binomial if simulation not available
        if not result:
            model_type = 'beta_binomial'
            print(f"[MLB] Using Beta-Binomial model (lineup not confirmed)")
            result = mlb_proj.analyze_prop(
                pitcher_name=pitcher_name,
                opponent_name=opponent,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                venue=venue,
                vegas_total=request.vegas_total,
                spread=request.spread,
                umpire_name=request.umpire_name,
            )

        if not result:
            raise HTTPException(status_code=404, detail=f"Pitcher not found: {pitcher_name}")

        # Add model_type to result
        result['model_type'] = model_type

        # Save prediction if not test mode
        saved = False
        if not request.test and result['recommended_side'] != 'pass':
            try:
                # Create a modified request with resolved values for saving
                save_request = MLBAnalyzeRequest(
                    pitcher=pitcher_name,
                    opponent=opponent,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    venue=venue,
                    vegas_total=request.vegas_total,
                    spread=request.spread,
                    test=request.test
                )
                saved = _save_mlb_prediction_v2(result, save_request)
            except Exception as e:
                print(f"[MLB] Error saving prediction: {e}")

        # Get park factor info from matchup_factors
        matchup = result.get('matchup_factors', {})
        park_adj = matchup.get('park_adj', 0)
        park_k_factor = 1.0 + park_adj if park_adj else 1.0

        # Return full response
        response = {
            "sport": "mlb",
            "model_type": result.get('model_type', 'beta_binomial'),
            "pitcher": result['pitcher'],
            "team": result.get('pitcher_stats', {}).get('team'),
            "opponent": result['opponent'],
            "venue": venue,
            "park_k_factor": round(park_k_factor, 3),
            "projected_ks": result['projected_ks'],
            "line": result['line'],
            "over_odds": over_odds,
            "under_odds": under_odds,
            "best_book": best_book,
            "prob_over": result['prob_over'],
            "prob_under": result['prob_under'],
            "ev_over": result['ev_over'],
            "ev_under": result['ev_under'],
            "edge_over": result['edge_over'],
            "edge_under": result['edge_under'],
            "recommended_side": result['recommended_side'],
            "edge": result['edge'],
            "ev": result['ev'],
            "confidence": result['confidence'],
            "kelly_over": result['kelly_over'],
            "kelly_under": result['kelly_under'],
            "confidence_grade": result['confidence_grade'],
            "pitcher_stats": result['pitcher_stats'],
            "matchup_factors": result['matchup_factors'],
            "last_5_starts": result.get('last_5_starts', []),
            "breakdown": result['breakdown'],
            "summary": result['summary'],
            "saved": saved
        }

        # Add simulation-specific fields if using simulation
        if model_type == 'simulation':
            response['sim_stats'] = result.get('sim_stats', {})
            response['lineup'] = result.get('lineup', [])
            response['over_probs'] = result.get('over_probs', {})
            response['lineup_confirmed'] = True
        else:
            response['lineup_confirmed'] = False

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _save_mlb_prediction_v2(result: dict, request: MLBAnalyzeRequest) -> bool:
    """Save MLB prediction to database (V2 format)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    pitcher_name = result['pitcher']
    line = result['line']

    # Check for existing prediction (same pitcher, date, and line)
    cursor.execute("""
        SELECT id FROM mlb_predictions
        WHERE game_date = ? AND pitcher_name = ? AND line = ?
    """, (today, pitcher_name, line))
    existing = cursor.fetchone()

    if existing:
        print(f"[MLB] Prediction already exists for {pitcher_name} @ {line} - skipping save")
        conn.close()
        return False  # Already exists, don't save duplicate

    # Extract data from V2 result structure
    pitcher_stats = result.get('pitcher_stats', {})
    matchup_factors = result.get('matchup_factors', {})
    breakdown = result.get('breakdown', {})
    sim_stats = result.get('sim_stats', {})

    # Model tracking fields
    model_type = result.get('model_type', 'beta_binomial')
    lineup_confirmed = 1 if result.get('lineup_confirmed', False) else 0

    cursor.execute("""
        INSERT INTO mlb_predictions (
            game_date, pitcher_name, pitcher_team, opponent_team, venue,
            line, over_odds, under_odds,
            swstr_pct, csw_pct, k_pct, expected_pitch_count, expected_bf,
            opp_k_pct, opp_p_per_pa, park_factor,
            projected_ks, prob_over, prob_under,
            recommended_side, edge, ev, confidence_grade,
            model_type, lineup_confirmed, sim_mean_k, sim_std_k, sim_avg_ip, sim_avg_pitches, lineup_avg_k_pct,
            status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
    """, (
        today,
        result['pitcher'],
        pitcher_stats.get('team', ''),
        result['opponent'],
        request.venue,
        result['line'],
        request.over_odds,
        request.under_odds,
        pitcher_stats.get('swstr_pct'),
        pitcher_stats.get('csw_pct'),
        pitcher_stats.get('k_pct'),
        pitcher_stats.get('avg_pitches'),
        breakdown.get('expected_bf'),
        matchup_factors.get('opp_k_pct'),
        None,  # opp_p_per_pa not in V2 response
        1 + matchup_factors.get('park_adj', 0),
        result['projected_ks'],
        result['prob_over'],
        result['prob_under'],
        result['recommended_side'],
        result['edge'],
        result['ev'],
        result['confidence_grade'],
        model_type,
        lineup_confirmed,
        sim_stats.get('mean_k'),
        sim_stats.get('std_k'),
        sim_stats.get('avg_ip'),
        sim_stats.get('avg_pitches'),
        matchup_factors.get('lineup_avg_k'),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()
    return True


@app.get("/mlb/predictions")
async def get_mlb_predictions(status: str = "all", limit: int = 50):
    """Get MLB predictions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if status == "all":
        cursor.execute("""
            SELECT * FROM mlb_predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
    else:
        cursor.execute("""
            SELECT * FROM mlb_predictions
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (status, limit))

    rows = cursor.fetchall()
    conn.close()

    return {
        "count": len(rows),
        "predictions": [dict(row) for row in rows]
    }


@app.get("/mlb/stats")
async def get_mlb_stats():
    """Get MLB prediction statistics with CLV breakdown."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Basic stats
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'voided' THEN 1 ELSE 0 END) as voided,
            SUM(hit) as hits,
            ROUND(100.0 * SUM(hit) / NULLIF(SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END), 0), 1) as hit_rate
        FROM mlb_predictions
    """)

    row = cursor.fetchone()
    total, resolved, pending, voided, hits, hit_rate = row

    # CLV breakdown
    cursor.execute("""
        SELECT
            COUNT(*) as total_with_clv,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv_count,
            SUM(CASE WHEN clv > 0 AND hit = 1 THEN 1 ELSE 0 END) as positive_clv_hits,
            SUM(CASE WHEN clv = 0 THEN 1 ELSE 0 END) as zero_clv_count,
            SUM(CASE WHEN clv = 0 AND hit = 1 THEN 1 ELSE 0 END) as zero_clv_hits,
            SUM(CASE WHEN clv < 0 THEN 1 ELSE 0 END) as negative_clv_count,
            SUM(CASE WHEN clv < 0 AND hit = 1 THEN 1 ELSE 0 END) as negative_clv_hits,
            ROUND(AVG(clv), 2) as avg_clv
        FROM mlb_predictions
        WHERE status = 'resolved' AND closing_line IS NOT NULL
    """)

    clv_row = cursor.fetchone()
    conn.close()

    # Build CLV breakdown
    clv_data = {
        "total_with_clv": clv_row[0] or 0,
        "avg_clv": clv_row[7] or 0,
        "positive_clv": {
            "count": clv_row[1] or 0,
            "hits": clv_row[2] or 0,
            "hit_rate": round(100.0 * (clv_row[2] or 0) / clv_row[1], 1) if clv_row[1] else 0
        },
        "zero_clv": {
            "count": clv_row[3] or 0,
            "hits": clv_row[4] or 0,
            "hit_rate": round(100.0 * (clv_row[4] or 0) / clv_row[3], 1) if clv_row[3] else 0
        },
        "negative_clv": {
            "count": clv_row[5] or 0,
            "hits": clv_row[6] or 0,
            "hit_rate": round(100.0 * (clv_row[6] or 0) / clv_row[5], 1) if clv_row[5] else 0
        }
    }

    return {
        "total": total or 0,
        "resolved": resolved or 0,
        "pending": pending or 0,
        "voided": voided or 0,
        "hits": hits or 0,
        "hit_rate": hit_rate or 0,
        "clv": clv_data
    }


@app.post("/mlb/fetch-clv")
async def fetch_mlb_clv(date: str = None):
    """
    Fetch CLV (Closing Line Value) for MLB predictions.

    Uses The Odds API historical endpoint to get closing lines.
    Costs ~11 credits per prediction.

    Args:
        date: Optional date in YYYY-MM-DD format. If not provided,
              fetches CLV for all resolved predictions missing closing_line.
    """
    try:
        from mlb_fetch_clv import fetch_clv_for_date, fetch_all_missing_clv

        if date:
            result = fetch_clv_for_date(date)
        else:
            result = fetch_all_missing_clv()

        return {
            "message": "CLV fetch complete",
            "success": result["success"],
            "failed": result["failed"]
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="mlb_fetch_clv module not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlb/refresh-data")
async def refresh_mlb_data(force: bool = False):
    """
    Refresh MLB data caches (pitcher stats, team batting, park factors, etc.).

    Set force=true to refresh even if caches are not stale.
    """
    mlb_data = get_mlb_data()
    if not mlb_data:
        raise HTTPException(status_code=500, detail="MLB module not available")

    try:
        results = mlb_data.refresh_all(force=force)
        games = mlb_data.get_todays_games()

        return {
            "message": "MLB data refreshed",
            "results": results,
            "games_today": len(games),
            "games": games
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
