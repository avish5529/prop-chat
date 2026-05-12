#!/usr/bin/env python3
"""
MLB CLV Fetcher - Closing Line Value Calculator
------------------------------------------------
Uses The Odds API historical endpoint to fetch closing lines
and calculate CLV for MLB pitcher strikeout predictions.

Usage:
    python mlb_fetch_clv.py                    # Fetch CLV for all resolved predictions missing closing_line
    python mlb_fetch_clv.py 2026-03-26         # Fetch CLV for specific date
    python mlb_fetch_clv.py --test             # Test with one prediction
"""

import sqlite3
import requests
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple
from collections import Counter
import unicodedata

# Database path
DB_PATH = "predictions.db"

# MLB Sport key for Odds API
MLB_SPORT = "baseball_mlb"
MLB_MARKET = "pitcher_strikeouts"

# MLB team abbreviation to full name mapping (for matching Odds API team names)
TEAM_ABBREV_MAP = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
    # Alternate abbreviations (3-letter variants)
    "KCR": "Kansas City Royals", "TBR": "Tampa Bay Rays", "SDP": "San Diego Padres",
    "SFG": "San Francisco Giants", "AZ": "Arizona Diamondbacks", "WAS": "Washington Nationals",
    # Athletics moved from Oakland - API returns "Athletics" not "Oakland Athletics"
    "Athletics": "Athletics",
}


def get_api_key() -> str:
    """Load API key from .env file."""
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('ODDS_API_KEY='):
                return line.strip().split('=', 1)[1]
    raise ValueError("ODDS_API_KEY not found in .env")


API_KEY = get_api_key()


def get_historical_events(date_str: str) -> list:
    """
    Get historical MLB events for a given date.
    date_str should be ISO format like '2026-03-26T22:00:00Z'
    """
    url = f"https://api.the-odds-api.com/v4/historical/sports/{MLB_SPORT}/events"
    params = {
        "apiKey": API_KEY,
        "date": date_str
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        remaining = response.headers.get('x-requests-remaining', 'N/A')
        print(f"[MLB CLV] Credits remaining: {remaining}")
        return response.json().get('data', [])
    else:
        print(f"[MLB CLV] Error fetching events: {response.status_code}")
        return []


def get_historical_pitcher_props(event_id: str, date_str: str) -> dict:
    """
    Get historical pitcher strikeout props for a specific event.
    Returns dict with bookmaker odds data.
    """
    url = f"https://api.the-odds-api.com/v4/historical/sports/{MLB_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "date": date_str,
        "regions": "us",
        "markets": MLB_MARKET,
        "oddsFormat": "american"
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        data = response.json()
        remaining = response.headers.get('x-requests-remaining', 'N/A')
        print(f"[MLB CLV] Credits remaining: {remaining}")
        return data.get('data', {})
    else:
        print(f"[MLB CLV] Error fetching props: {response.status_code} - {response.text[:200]}")
        return {}


def normalize_name(name: str) -> str:
    """Normalize name for matching (lowercase, remove accents)."""
    name_lower = name.lower().strip()
    # Remove accents
    normalized = unicodedata.normalize('NFKD', name_lower).encode('ascii', 'ignore').decode('ascii')
    return normalized


def find_pitcher_closing_line(props_data: dict, pitcher_name: str) -> Optional[float]:
    """
    Find a pitcher's closing line from the props data.
    Returns the consensus line (most common) across bookmakers.
    """
    bookmakers = props_data.get('bookmakers', [])
    if not bookmakers:
        return None

    lines = []
    pitcher_normalized = normalize_name(pitcher_name)

    # Also try just the last name for matching
    pitcher_last = pitcher_normalized.split()[-1] if pitcher_normalized else ""

    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') != MLB_MARKET:
                continue
            outcomes = market.get('outcomes', [])
            for outcome in outcomes:
                description = outcome.get('description', '')
                description_normalized = normalize_name(description)

                # Check if pitcher name matches
                # Try full name match first
                if pitcher_normalized in description_normalized or description_normalized in pitcher_normalized:
                    point = outcome.get('point')
                    if point is not None:
                        lines.append(float(point))
                    continue

                # Try last name match
                if pitcher_last and pitcher_last in description_normalized:
                    point = outcome.get('point')
                    if point is not None:
                        lines.append(float(point))

    if not lines:
        return None

    # Return most common line (consensus)
    most_common = Counter(lines).most_common(1)
    return most_common[0][0] if most_common else None


def calculate_clv(opening_line: float, closing_line: float, recommended_side: str) -> float:
    """
    Calculate CLV (Closing Line Value).

    For OVER bets: CLV = closing_line - opening_line
      - Positive = line moved up (you got value)
    For UNDER bets: CLV = opening_line - closing_line
      - Positive = line moved down (you got value)
    """
    if recommended_side.lower() == 'over':
        return closing_line - opening_line
    else:  # under
        return opening_line - closing_line


def find_event_for_teams(events: list, opponent_team: str) -> Optional[dict]:
    """Find the event matching the opponent team."""
    # Convert abbreviation to full name if needed
    opponent_full = TEAM_ABBREV_MAP.get(opponent_team.upper(), opponent_team) if opponent_team else ""
    opponent_lower = opponent_full.lower()

    for event in events:
        event_home = event.get('home_team', '').lower()
        event_away = event.get('away_team', '').lower()

        # Check for partial matches (e.g., "Yankees" in "New York Yankees")
        if opponent_lower in event_home or opponent_lower in event_away or \
           event_home in opponent_lower or event_away in opponent_lower:
            return event

    return None


def fetch_closing_line_for_prediction(prediction: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch closing line for a prediction and calculate CLV.

    Returns: (closing_line, clv) or (None, None) if not found
    """
    pitcher_name = prediction['pitcher_name']
    game_date = prediction['game_date']
    opening_line = prediction['line']
    recommended_side = prediction['recommended_side']
    opponent_team = prediction.get('opponent_team', '')

    # MLB games happen throughout the day, primarily afternoon/evening
    # Create timestamps for typical game times
    game_dt = datetime.strptime(game_date, "%Y-%m-%d")

    # Try multiple times throughout the day (UTC times) - ORDER MATTERS!
    # Start with early times to catch day games, then move to later times
    # MLB games: 12pm-10pm ET = 4pm-2am UTC (next day)
    query_times = [
        game_dt.replace(hour=16, minute=0),   # 12pm ET (earliest day games)
        game_dt.replace(hour=18, minute=0),   # 2pm ET (afternoon games)
        game_dt.replace(hour=20, minute=0),   # 4pm ET
        game_dt.replace(hour=22, minute=0),   # 6pm ET (evening games)
        game_dt.replace(hour=0, minute=0) + timedelta(days=1),   # 8pm ET
        game_dt.replace(hour=2, minute=0) + timedelta(days=1),   # 10pm ET (west coast)
    ]

    # First, try to find events AND our specific opponent
    events = None
    target_event = None

    for query_time in query_times:
        date_str = query_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[MLB CLV] Searching events at {date_str}...")
        events = get_historical_events(date_str)

        if events:
            print(f"[MLB CLV] Found {len(events)} events")
            # Check if our opponent's game is in this batch
            if opponent_team:
                target_event = find_event_for_teams(events, opponent_team)
                if target_event:
                    print(f"[MLB CLV] Matched opponent '{opponent_team}' → {target_event.get('away_team')} @ {target_event.get('home_team')}")
                    break  # Found our game, stop searching
            else:
                break  # No opponent specified, use whatever events we found

    if not events:
        print(f"[MLB CLV] No events found for {game_date}")
        return None, None

    # If no match found, we'd need to check all events (uses more credits)
    if not target_event:
        if opponent_team:
            print(f"[MLB CLV] No opponent match for '{opponent_team}', checking all events...")
        target_events = events
    else:
        target_events = [target_event]

    closing_line = None

    for event in target_events:
        event_id = event['id']
        commence_time = event.get('commence_time', '')

        # Query 5 minutes before game start for closing line
        if commence_time:
            game_start = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            query_time = game_start - timedelta(minutes=5)
            date_str = query_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Fallback to evening time
            date_str = query_times[0].strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"[MLB CLV] Checking {event.get('away_team')} @ {event.get('home_team')}...")

        # Get pitcher props for this event
        props_data = get_historical_pitcher_props(event_id, date_str)

        if not props_data:
            continue

        # Try to find the pitcher's closing line
        closing_line = find_pitcher_closing_line(props_data, pitcher_name)

        if closing_line is not None:
            print(f"[MLB CLV] Found {pitcher_name}: {closing_line}")
            break

    if closing_line is None:
        print(f"[MLB CLV] Could not find closing line for {pitcher_name}")
        return None, None

    # Calculate CLV
    clv = calculate_clv(opening_line, closing_line, recommended_side)

    return closing_line, clv


def update_prediction_clv(prediction_id: int, closing_line: float, clv: float):
    """Update MLB prediction with closing line and CLV."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE mlb_predictions
        SET closing_line = ?, clv = ?
        WHERE id = ?
    """, (closing_line, clv, prediction_id))
    conn.commit()
    conn.close()


def get_predictions_needing_clv(target_date: str = None) -> list:
    """Get resolved MLB predictions that need CLV data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if target_date:
        cursor.execute("""
            SELECT * FROM mlb_predictions
            WHERE game_date = ? AND status = 'resolved' AND closing_line IS NULL
            ORDER BY id
        """, (target_date,))
    else:
        cursor.execute("""
            SELECT * FROM mlb_predictions
            WHERE status = 'resolved' AND closing_line IS NULL
            ORDER BY game_date, id
        """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_clv_for_date(target_date: str) -> dict:
    """
    Fetch CLV for all predictions on a specific date.
    Returns: {"success": int, "failed": int}
    """
    predictions = get_predictions_needing_clv(target_date)

    if not predictions:
        print(f"No predictions need CLV data for {target_date}")
        return {"success": 0, "failed": 0}

    success = 0
    failed = 0

    for pred in predictions:
        print(f"\n[{pred['id']}] {pred['pitcher_name']} Ks {pred['line']} ({pred['game_date']})")
        print(f"     Pick: {pred['recommended_side'].upper()}")

        closing_line, clv = fetch_closing_line_for_prediction(pred)

        if closing_line is not None:
            update_prediction_clv(pred['id'], closing_line, clv)
            clv_str = f"+{clv:.1f}" if clv > 0 else f"{clv:.1f}"
            print(f"     → Closing: {closing_line}, CLV: {clv_str}")
            success += 1
        else:
            print(f"     → Failed to find closing line")
            failed += 1

    return {"success": success, "failed": failed}


def fetch_all_missing_clv() -> dict:
    """
    Fetch CLV for all predictions missing closing line data.
    Returns: {"success": int, "failed": int}
    """
    predictions = get_predictions_needing_clv()

    if not predictions:
        print("No predictions need CLV data")
        return {"success": 0, "failed": 0}

    print(f"Found {len(predictions)} predictions needing CLV")

    success = 0
    failed = 0

    for pred in predictions:
        print(f"\n[{pred['id']}] {pred['pitcher_name']} Ks {pred['line']} ({pred['game_date']})")
        print(f"     Pick: {pred['recommended_side'].upper()}")

        closing_line, clv = fetch_closing_line_for_prediction(pred)

        if closing_line is not None:
            update_prediction_clv(pred['id'], closing_line, clv)
            clv_str = f"+{clv:.1f}" if clv > 0 else f"{clv:.1f}"
            print(f"     → Closing: {closing_line}, CLV: {clv_str}")
            success += 1
        else:
            print(f"     → Failed to find closing line")
            failed += 1

    return {"success": success, "failed": failed}


def main():
    """Main function to fetch CLV for MLB predictions."""
    # Parse arguments
    target_date = None
    test_mode = False

    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_mode = True
        else:
            target_date = sys.argv[1]

    # Get predictions needing CLV
    predictions = get_predictions_needing_clv(target_date)

    if not predictions:
        print("No MLB predictions need CLV data")
        return

    print(f"Found {len(predictions)} MLB predictions needing CLV")
    print("=" * 60)

    if test_mode:
        predictions = predictions[:1]  # Just test one

    success = 0
    failed = 0

    for pred in predictions:
        print(f"\n[{pred['id']}] {pred['pitcher_name']} Ks {pred['line']} ({pred['game_date']})")
        print(f"     vs {pred['opponent_team']} | Pick: {pred['recommended_side'].upper()}")

        closing_line, clv = fetch_closing_line_for_prediction(pred)

        if closing_line is not None:
            update_prediction_clv(pred['id'], closing_line, clv)
            clv_str = f"+{clv:.1f}" if clv > 0 else f"{clv:.1f}"
            print(f"     → Closing: {closing_line}, CLV: {clv_str}")
            success += 1
        else:
            print(f"     → Failed to find closing line")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {success} success, {failed} failed")

    # Show remaining credits
    url = "https://api.the-odds-api.com/v4/sports"
    r = requests.get(url, params={"apiKey": API_KEY})
    print(f"Remaining credits: {r.headers.get('x-requests-remaining', 'N/A')}")


if __name__ == "__main__":
    main()
