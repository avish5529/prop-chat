#!/usr/bin/env python3
"""
CLV Fetcher - Closing Line Value Calculator
--------------------------------------------
Uses The Odds API historical endpoint to fetch closing lines
and calculate CLV for predictions.

Usage:
    python fetch_clv.py                    # Fetch CLV for all resolved predictions missing closing_line
    python fetch_clv.py 2026-03-14         # Fetch CLV for specific date
    python fetch_clv.py --test             # Test with one prediction
"""

import sqlite3
import requests
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Database path
DB_PATH = "predictions.db"

# Load API key
def get_api_key() -> str:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('ODDS_API_KEY='):
                return line.strip().split('=', 1)[1]
    raise ValueError("ODDS_API_KEY not found in .env")

API_KEY = get_api_key()

# Map our prop types to Odds API market keys
PROP_TYPE_TO_MARKET = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "pra": "player_points_rebounds_assists",
    "pr": "player_points_rebounds",
    "pa": "player_points_assists",
    "ra": "player_rebounds_assists",
}


def get_historical_events(date_str: str) -> list:
    """
    Get historical events for a given date.
    date_str should be ISO format like '2026-03-14T22:00:00Z'
    """
    url = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events"
    params = {
        "apiKey": API_KEY,
        "date": date_str
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"[CLV] Error fetching events: {response.status_code}")
        return []


def get_historical_player_props(event_id: str, date_str: str, market: str) -> dict:
    """
    Get historical player props for a specific event and market.
    Returns dict with bookmaker odds data.
    """
    url = f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "date": date_str,
        "regions": "us",
        "markets": market,
        "oddsFormat": "american"
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        data = response.json()
        remaining = response.headers.get('x-requests-remaining', 'N/A')
        print(f"[CLV] Credits remaining: {remaining}")
        return data.get('data', {})
    else:
        print(f"[CLV] Error fetching props: {response.status_code} - {response.text[:200]}")
        return {}


def find_player_line(props_data: dict, player_name: str) -> Optional[float]:
    """
    Find a player's line from the props data.
    Returns the consensus line (most common) across bookmakers.
    """
    bookmakers = props_data.get('bookmakers', [])
    if not bookmakers:
        return None

    lines = []
    player_lower = player_name.lower()

    # Handle common name variations
    name_variants = [player_lower]
    # Add variant without accents
    import unicodedata
    normalized = unicodedata.normalize('NFKD', player_lower).encode('ascii', 'ignore').decode('ascii')
    if normalized != player_lower:
        name_variants.append(normalized)

    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            outcomes = market.get('outcomes', [])
            for outcome in outcomes:
                description = outcome.get('description', '').lower()
                # Check if player name matches
                for variant in name_variants:
                    if variant in description or description in variant:
                        point = outcome.get('point')
                        if point is not None:
                            lines.append(float(point))
                        break

    if not lines:
        return None

    # Return most common line (consensus)
    from collections import Counter
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


def find_event_for_teams(events: list, home_team: str, away_team: str) -> Optional[dict]:
    """Find the event matching the teams."""
    home_lower = home_team.lower() if home_team else ""
    away_lower = away_team.lower() if away_team else ""

    for event in events:
        event_home = event.get('home_team', '').lower()
        event_away = event.get('away_team', '').lower()

        # Check for partial matches
        if (home_lower in event_home or event_home in home_lower) and \
           (away_lower in event_away or event_away in away_lower):
            return event

        # Also check reversed (in case of data inconsistency)
        if (home_lower in event_away or event_away in home_lower) and \
           (away_lower in event_home or event_home in away_lower):
            return event

    return None


def fetch_closing_line_for_prediction(prediction: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch closing line for a prediction and calculate CLV.

    Returns: (closing_line, clv) or (None, None) if not found
    """
    player_name = prediction['player_name']
    prop_type = prediction['prop_type']
    game_date = prediction['game_date']
    opening_line = prediction['line']
    recommended_side = prediction['recommended_side']
    opponent_team = prediction.get('opponent_team', '')

    # Map prop type to market
    market = PROP_TYPE_TO_MARKET.get(prop_type)
    if not market:
        print(f"[CLV] Unknown prop type: {prop_type}")
        return None, None

    # Create timestamp for ~30 min before typical game time
    # We'll query for evening time since most NBA games are evening
    # Format: 2026-03-14 -> 2026-03-14T23:00:00Z (evening)
    game_dt = datetime.strptime(game_date, "%Y-%m-%d")
    # Try multiple times throughout the evening to find the game
    query_times = [
        game_dt.replace(hour=23, minute=0),  # 11pm UTC (~6-7pm ET)
        game_dt.replace(hour=1, minute=0) + timedelta(days=1),  # 1am UTC next day (~8pm ET)
        game_dt.replace(hour=3, minute=0) + timedelta(days=1),  # 3am UTC (~10pm ET)
    ]

    events = None
    for query_time in query_times:
        date_str = query_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[CLV] Searching events at {date_str}...")
        events = get_historical_events(date_str)
        if events:
            break

    if not events:
        print(f"[CLV] No events found for {game_date}")
        return None, None

    print(f"[CLV] Found {len(events)} events")

    # First, try to find the event by matching opponent_team
    target_events = []
    if opponent_team:
        opponent_lower = opponent_team.lower()
        for event in events:
            event_home = event.get('home_team', '').lower()
            event_away = event.get('away_team', '').lower()
            if opponent_lower in event_home or opponent_lower in event_away or \
               event_home in opponent_lower or event_away in opponent_lower:
                target_events.append(event)
                print(f"[CLV] Matched opponent '{opponent_team}' → {event.get('away_team')} @ {event.get('home_team')}")
                break

    # If no match found, fall back to checking all events (uses more credits)
    if not target_events:
        print(f"[CLV] No opponent match, checking all events...")
        target_events = events

    event_found = None
    closing_line = None

    for event in target_events:
        event_id = event['id']
        commence_time = event.get('commence_time', '')

        # Query 5 minutes before game start
        if commence_time:
            game_start = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            query_time = game_start - timedelta(minutes=5)
            date_str = query_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Fallback to evening time
            date_str = query_times[0].strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"[CLV] Checking {event.get('away_team')} @ {event.get('home_team')}...")

        # Get player props for this event
        props_data = get_historical_player_props(event_id, date_str, market)

        if not props_data:
            continue

        # Try to find the player's line
        closing_line = find_player_line(props_data, player_name)

        if closing_line is not None:
            event_found = event
            print(f"[CLV] Found {player_name} {prop_type}: {closing_line}")
            break

    if closing_line is None:
        print(f"[CLV] Could not find closing line for {player_name} {prop_type}")
        return None, None

    # Calculate CLV
    clv = calculate_clv(opening_line, closing_line, recommended_side)

    return closing_line, clv


def update_prediction_clv(prediction_id: int, closing_line: float, clv: float):
    """Update prediction with closing line and CLV."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE predictions
        SET closing_line = ?, clv = ?
        WHERE id = ?
    """, (closing_line, clv, prediction_id))
    conn.commit()
    conn.close()


def get_predictions_needing_clv(target_date: str = None) -> list:
    """Get resolved predictions that need CLV data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if target_date:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE game_date = ? AND status = 'resolved' AND closing_line IS NULL
            ORDER BY id
        """, (target_date,))
    else:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE status = 'resolved' AND closing_line IS NULL
            ORDER BY game_date, id
        """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def main():
    """Main function to fetch CLV for predictions."""
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
        print("No predictions need CLV data")
        return

    print(f"Found {len(predictions)} predictions needing CLV")
    print("=" * 60)

    if test_mode:
        predictions = predictions[:1]  # Just test one

    success = 0
    failed = 0

    for pred in predictions:
        print(f"\n[{pred['id']}] {pred['player_name']} {pred['prop_type']} {pred['line']} ({pred['game_date']})")
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

    print("\n" + "=" * 60)
    print(f"Results: {success} success, {failed} failed")

    # Show remaining credits
    url = "https://api.the-odds-api.com/v4/sports"
    r = requests.get(url, params={"apiKey": API_KEY})
    print(f"Remaining credits: {r.headers.get('x-requests-remaining', 'N/A')}")


if __name__ == "__main__":
    main()
