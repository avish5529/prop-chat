#!/usr/bin/env python3
"""
Daily Sync Script for Prop.chat
--------------------------------
Runs once daily (morning after games) to:
1. Fetch closing lines from BettingPros
2. Sync actual results from NBA API
3. Calculate CLV (Closing Line Value)
4. Update database

Usage:
    python daily_sync.py              # Sync yesterday's games
    python daily_sync.py 2026-02-28   # Sync specific date
"""

import sqlite3
import requests
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
import time
import sys
import re
import random

# Database path
DB_PATH = "predictions.db"

# NBA API headers (avoid timeouts)
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}


def get_pending_predictions(target_date: str = None):
    """Get predictions that need closing lines or results."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if target_date:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE game_date = ? AND (closing_line IS NULL OR actual_result IS NULL)
            ORDER BY id
        """, (target_date,))
    else:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE closing_line IS NULL OR actual_result IS NULL
            ORDER BY game_date, id
        """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def slugify_player_name(name: str) -> str:
    """Convert player name to BettingPros URL slug."""
    # Handle special cases
    slug = name.lower()
    slug = slug.replace(".", "")  # P.J. -> PJ
    slug = slug.replace("'", "")  # D'Angelo -> DAngelo
    slug = re.sub(r'\s+(jr|sr|ii|iii|iv)$', '', slug)  # Remove suffixes
    slug = slug.replace(" ", "-")
    return slug


def get_prop_url_slug(prop_type: str) -> str:
    """Convert prop type to BettingPros URL slug."""
    mapping = {
        "points": "points",
        "rebounds": "rebounds",
        "assists": "assists",
        "pra": "points-assists-rebounds",  # PAR order, not PRA!
        "pr": "points-rebounds",
        "pa": "points-assists",
        "ra": "rebounds-assists"
    }
    return mapping.get(prop_type, "points")


def scrape_closing_line(player_name: str, prop_type: str, game_date: str) -> float:
    """
    Scrape closing line from BettingPros for a specific player/prop/date.
    Returns the closing line value or None if not found.
    """
    import unicodedata

    # Convert player name to URL slug
    slug = unicodedata.normalize('NFKD', player_name).encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV)$', '', slug, flags=re.IGNORECASE)
    slug = re.sub(r'[.\']', '', slug)
    slug = slug.lower().replace(' ', '-')

    # Map prop type to URL slug
    prop_map = {
        "points": "points",
        "rebounds": "rebounds",
        "assists": "assists",
        "pra": "points-assists-rebounds",  # PAR order!
        "pr": "points-rebounds",
        "pa": "points-assists",
        "ra": "rebounds-assists"
    }
    prop_slug = prop_map.get(prop_type, "points")

    url = f"https://www.bettingpros.com/nba/props/{slug}/{prop_slug}/"

    # Parse target date to match BettingPros format (e.g., "2/26")
    target_dt = datetime.strptime(game_date, "%Y-%m-%d")
    target_month = target_dt.month
    target_day = target_dt.day
    date_pattern = f"{target_month}/{target_day}"

    try:
        with sync_playwright() as p:
            # Launch with anti-detection flags
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                ]
            )

            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
            )

            page = context.new_page()

            # Remove webdriver detection
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            time.sleep(2)

            # Toggle season dropdown to trigger lazy load if needed
            try:
                dropdown_2025 = page.locator('text=2025 Season').first
                if dropdown_2025.is_visible():
                    dropdown_2025.click()
                    time.sleep(0.5)
                    page.locator('text=2024 Season').first.click()
                    time.sleep(1)
                    page.locator('text=2024 Season').first.click()
                    time.sleep(0.5)
                    page.locator('text=2025 Season').first.click()
                    time.sleep(1)
            except:
                pass

            # Find the game log table (has 'Date' and 'Prop Line' headers)
            tables = page.locator('table').all()

            for table in tables:
                headers = table.locator('th').all()
                header_texts = [h.text_content().strip() for h in headers]

                if 'Date' not in header_texts or 'Prop Line' not in header_texts:
                    continue

                date_col = header_texts.index('Date')
                line_col = header_texts.index('Prop Line')

                rows = table.locator('tbody tr').all()

                for row in rows:
                    cells = row.locator('td').all()
                    if len(cells) <= max(date_col, line_col):
                        continue

                    date_text = cells[date_col].text_content().strip()

                    # Check if date matches (format: "2/26")
                    if date_pattern in date_text:
                        line_text = cells[line_col].text_content().strip()

                        # Handle "NL" (No Line)
                        if line_text.upper() == "NL" or not line_text:
                            browser.close()
                            return None

                        # Parse numeric line (remove O/U prefix)
                        try:
                            line_value = float(re.sub(r'[OU]', '', line_text).strip())
                            browser.close()
                            return line_value
                        except ValueError:
                            pass

            browser.close()
            return None

    except Exception as e:
        print(f"  [Scrape Error] {player_name} {prop_type}: {e}")
        return None


def fetch_actual_result(player_name: str, game_date: str, prop_type: str, player_id: int = None):
    """
    Fetch actual stats from NBA API for a player on a specific date.
    Returns the stat value for the given prop type.
    """
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players

    # Get player ID if not provided
    if not player_id:
        player_list = players.find_players_by_full_name(player_name)
        if not player_list:
            return None, None
        player_id = player_list[0]['id']

    try:
        # Determine season
        game_dt = datetime.strptime(game_date, "%Y-%m-%d")
        if game_dt.month >= 10:
            season = f"{game_dt.year}-{str(game_dt.year + 1)[2:]}"
        else:
            season = f"{game_dt.year - 1}-{str(game_dt.year)[2:]}"

        # Fetch game log
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            headers=NBA_HEADERS,
            timeout=30
        )
        df = log.get_data_frames()[0]

        # Find the game
        df['GAME_DATE'] = df['GAME_DATE'].apply(lambda x: datetime.strptime(x, "%b %d, %Y").strftime("%Y-%m-%d"))
        game_row = df[df['GAME_DATE'] == game_date]

        if game_row.empty:
            return None, None

        row = game_row.iloc[0]
        minutes = row['MIN']

        # Calculate stat based on prop type (convert to float to avoid numpy int issues)
        pts = float(row['PTS'])
        reb = float(row['REB'])
        ast = float(row['AST'])

        stat_map = {
            "points": pts,
            "rebounds": reb,
            "assists": ast,
            "pra": pts + reb + ast,
            "pr": pts + reb,
            "pa": pts + ast,
            "ra": reb + ast
        }

        return stat_map.get(prop_type), float(minutes) if minutes else None

    except Exception as e:
        print(f"  [NBA API Error] {player_name}: {e}")
        return None, None


def calculate_clv(opening_line: float, closing_line: float, side: str) -> float:
    """
    Calculate Closing Line Value.
    Positive CLV = you got a better line than closing (good)
    Negative CLV = you got a worse line than closing (bad)

    For OVER bets: CLV = closing - opening (lower opening is better)
    For UNDER bets: CLV = opening - closing (higher opening is better)
    """
    if opening_line is None or closing_line is None:
        return None

    if side.lower() == "over":
        # For overs, you want to bet UNDER a higher number
        # If line went UP, you got value
        return closing_line - opening_line
    else:
        # For unders, you want to bet OVER a lower number
        # If line went DOWN, you got value
        return opening_line - closing_line


def update_prediction(pred_id: int, closing_line: float = None, actual_result: float = None,
                      actual_minutes: float = None, clv: float = None, hit: int = None,
                      catboost_hit: int = None):
    """Update a prediction record with new data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updates = []
    values = []

    if closing_line is not None:
        updates.append("closing_line = ?")
        values.append(closing_line)

    if actual_result is not None:
        updates.append("actual_result = ?")
        values.append(actual_result)
        updates.append("resolved_at = ?")
        values.append(datetime.now().isoformat())

    if actual_minutes is not None:
        updates.append("actual_minutes = ?")
        values.append(actual_minutes)

    if clv is not None:
        updates.append("clv = ?")
        values.append(clv)

    if hit is not None:
        updates.append("hit = ?")
        values.append(hit)

    if catboost_hit is not None:
        updates.append("catboost_hit = ?")
        values.append(catboost_hit)

    if updates:
        values.append(pred_id)
        cursor.execute(f"UPDATE predictions SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()

    conn.close()


def sync_predictions(target_date: str = None):
    """Main sync function."""
    if target_date is None:
        # Default to yesterday
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"DAILY SYNC - {target_date}")
    print(f"{'='*60}\n")

    # Get pending predictions
    predictions = get_pending_predictions(target_date)

    if not predictions:
        print(f"No pending predictions for {target_date}")
        return

    print(f"Found {len(predictions)} predictions to sync\n")

    results = {
        "closing_lines_found": 0,
        "results_synced": 0,
        "clv_calculated": 0,
        "hits": 0,
        "misses": 0,
        "errors": 0
    }

    scrape_count = 0
    consecutive_failures = 0
    for pred in predictions:
        print(f"[{pred['id']}] {pred['player_name']} {pred['prop_type']} {pred['line']}")

        closing_line = pred.get('closing_line')
        actual_result = pred.get('actual_result')
        minutes = pred.get('actual_minutes')

        # Fetch closing line if missing
        if closing_line is None:
            print(f"  Scraping closing line...")
            closing_line = scrape_closing_line(
                pred['player_name'],
                pred['prop_type'],
                pred['game_date']
            )
            if closing_line:
                print(f"  → Closing line: {closing_line}")
                results["closing_lines_found"] += 1
                consecutive_failures = 0
            else:
                print(f"  → No closing line found")
                consecutive_failures += 1

            # Rate limiting to avoid IP blocks
            scrape_count += 1

            # Exponential backoff if we see consecutive failures (likely rate limited)
            if consecutive_failures >= 3:
                backoff = min(120, 30 * (2 ** (consecutive_failures - 3)))
                print(f"  [Backoff {backoff}s - {consecutive_failures} consecutive failures]")
                time.sleep(backoff)
            # Long pause every 5 scrapes
            elif scrape_count % 5 == 0:
                print(f"  [Rate limit pause - 45s after {scrape_count} scrapes]")
                time.sleep(45)
            # Base delay with jitter (8-12s)
            else:
                delay = random.uniform(8, 12)
                time.sleep(delay)

        # Fetch actual result if missing
        if actual_result is None:
            print(f"  Fetching actual result...")
            actual_result, minutes = fetch_actual_result(
                pred['player_name'],
                pred['game_date'],
                pred['prop_type'],
                pred.get('player_id')
            )
            if actual_result is not None:
                print(f"  → Actual: {actual_result} ({minutes} min)")
                results["results_synced"] += 1
            else:
                print(f"  → Could not fetch result")
                results["errors"] += 1

        # Calculate CLV
        clv = None
        if closing_line and pred['line']:
            clv = calculate_clv(pred['line'], closing_line, pred['recommended_side'])
            if clv is not None:
                print(f"  → CLV: {clv:+.1f} ({'good' if clv > 0 else 'bad'})")
                results["clv_calculated"] += 1

        # Calculate hits
        hit = None
        catboost_hit = None
        if actual_result is not None:
            # Rule-based hit
            if pred['recommended_side'] == 'over':
                hit = 1 if actual_result > pred['line'] else 0
            else:
                hit = 1 if actual_result < pred['line'] else 0

            # CatBoost hit
            if pred.get('catboost_pick'):
                if pred['catboost_pick'] == 'over':
                    catboost_hit = 1 if actual_result > pred['line'] else 0
                else:
                    catboost_hit = 1 if actual_result < pred['line'] else 0

            if hit:
                print(f"  → Rule-based: HIT ✓")
                results["hits"] += 1
            else:
                print(f"  → Rule-based: MISS ✗")
                results["misses"] += 1

        # Update database
        update_prediction(
            pred['id'],
            closing_line=closing_line,
            actual_result=actual_result,
            actual_minutes=minutes if actual_result else None,
            clv=clv,
            hit=hit,
            catboost_hit=catboost_hit
        )
        print()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Closing lines found: {results['closing_lines_found']}")
    print(f"Results synced: {results['results_synced']}")
    print(f"CLV calculated: {results['clv_calculated']}")
    print(f"Hits: {results['hits']}")
    print(f"Misses: {results['misses']}")
    print(f"Errors: {results['errors']}")

    if results['hits'] + results['misses'] > 0:
        hit_rate = results['hits'] / (results['hits'] + results['misses']) * 100
        print(f"Hit Rate: {hit_rate:.1f}%")


def show_clv_stats():
    """Show CLV statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv,
            SUM(CASE WHEN clv < 0 THEN 1 ELSE 0 END) as negative_clv,
            AVG(clv) as avg_clv,
            SUM(CASE WHEN clv > 0 AND hit = 1 THEN 1 ELSE 0 END) as positive_clv_hits,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv_total,
            SUM(CASE WHEN clv < 0 AND hit = 1 THEN 1 ELSE 0 END) as negative_clv_hits,
            SUM(CASE WHEN clv < 0 THEN 1 ELSE 0 END) as negative_clv_total
        FROM predictions
        WHERE clv IS NOT NULL
    """)

    row = cursor.fetchone()
    conn.close()

    if row and row[0] > 0:
        print(f"\n{'='*60}")
        print("CLV STATISTICS")
        print(f"{'='*60}")
        print(f"Total with CLV: {row[0]}")
        print(f"Positive CLV: {row[1]} ({row[1]/row[0]*100:.1f}%)")
        print(f"Negative CLV: {row[2]} ({row[2]/row[0]*100:.1f}%)")
        print(f"Average CLV: {row[3]:+.2f}")

        if row[5] > 0:
            print(f"\nPositive CLV hit rate: {row[4]}/{row[5]} ({row[4]/row[5]*100:.1f}%)")
        if row[7] > 0:
            print(f"Negative CLV hit rate: {row[6]}/{row[7]} ({row[6]/row[7]*100:.1f}%)")


if __name__ == "__main__":
    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stats":
            show_clv_stats()
        else:
            # Assume it's a date
            sync_predictions(sys.argv[1])
    else:
        # Default: sync yesterday
        sync_predictions()

    # Always show stats at the end
    show_clv_stats()
