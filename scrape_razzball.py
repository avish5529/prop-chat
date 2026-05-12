#!/usr/bin/env python3
"""
Razzball Projected Minutes Scraper
-----------------------------------
Scrapes projected minutes from Razzball's NBA lineups page.
Minutes are injury-adjusted and updated throughout the day.

Usage:
    python scrape_razzball.py           # Scrape and cache
    python scrape_razzball.py --show    # Show cached data
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright
import time

CACHE_FILE = "razzball_minutes_cache.json"
RAZZBALL_URL = "https://basketball.razzball.com/lineups/"


def scrape_razzball_minutes() -> dict:
    """
    Scrape projected minutes for all players from Razzball.
    Returns dict: {player_name: projected_minutes}
    """
    print(f"Scraping Razzball projected minutes...")
    print(f"URL: {RAZZBALL_URL}\n")

    players = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )

        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )

        page = context.new_page()

        try:
            page.goto(RAZZBALL_URL, timeout=60000, wait_until='domcontentloaded')
            time.sleep(3)

            # Find all tables on the page
            tables = page.locator('table').all()
            print(f"Found {len(tables)} tables (games)\n")

            for table in tables:
                rows = table.locator('tr').all()

                if len(rows) < 6:
                    continue

                # Row 4 has headers: Pos, Inj, Name, MIN, PTS... (repeated for both teams)
                # Row 5+ has player data for both HOME and AWAY

                # Find the header row with MIN
                header_row = None
                header_row_idx = None
                for idx, row in enumerate(rows):
                    ths = row.locator('th').all()
                    if ths:
                        texts = [th.text_content().strip().upper() for th in ths]
                        if 'MIN' in texts and 'NAME' in texts:
                            header_row = row
                            header_row_idx = idx
                            break

                if not header_row:
                    continue

                # Parse header to find column indices
                ths = header_row.locator('th').all()
                headers = [th.text_content().strip().upper() for th in ths]

                # Razzball has HOME team (cols 0-11) and AWAY team (cols 12-23)
                # Find indices for both teams
                try:
                    # HOME team (first occurrence)
                    home_name_idx = headers.index('NAME')
                    home_min_idx = headers.index('MIN')

                    # AWAY team (second occurrence, after first MIN)
                    remaining = headers[home_min_idx + 1:]
                    away_name_idx = home_min_idx + 1 + remaining.index('NAME')
                    away_min_idx = home_min_idx + 1 + remaining.index('MIN')
                except ValueError:
                    continue

                # Parse player rows (after header row)
                for row in rows[header_row_idx + 1:]:
                    cells = row.locator('td').all()
                    if len(cells) < max(home_min_idx, away_min_idx) + 1:
                        continue

                    # Extract HOME player
                    try:
                        home_name = cells[home_name_idx].text_content().strip()
                        home_min = float(cells[home_min_idx].text_content().strip())
                        if home_name and len(home_name) >= 3:
                            players[home_name] = home_min
                    except (ValueError, IndexError):
                        pass

                    # Extract AWAY player
                    try:
                        away_name = cells[away_name_idx].text_content().strip()
                        away_min = float(cells[away_min_idx].text_content().strip())
                        if away_name and len(away_name) >= 3:
                            players[away_name] = away_min
                    except (ValueError, IndexError):
                        pass

            browser.close()

        except Exception as e:
            print(f"Error scraping: {e}")
            browser.close()
            return {}

    print(f"Scraped {len(players)} players\n")
    return players


def save_cache(players: dict):
    """Save scraped data to cache file."""
    cache = {
        "scraped_at": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "players": players
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"Saved to {CACHE_FILE}")


def load_cache() -> dict:
    """Load cached data. Returns empty dict if no cache or stale."""
    if not Path(CACHE_FILE).exists():
        return {}

    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)

    # Check if cache is from today
    if cache.get("date") != datetime.now().strftime("%Y-%m-%d"):
        print("Cache is stale (from different day)")
        return {}

    return cache


def get_projected_minutes(player_name: str) -> float | None:
    """
    Get projected minutes for a player from cache.
    Returns None if player not found.
    """
    cache = load_cache()
    if not cache:
        return None

    players = cache.get("players", {})

    # Try exact match first
    if player_name in players:
        return players[player_name]

    # Try case-insensitive match
    player_lower = player_name.lower()
    for name, minutes in players.items():
        if name.lower() == player_lower:
            return minutes

    # Try matching with first initial + last name (e.g., "A. Edwards" matches "Anthony Edwards")
    # But require BOTH first initial AND last name to match to avoid wrong matches
    parts = player_name.split()
    if len(parts) >= 2:
        first_initial = parts[0][0].lower()
        last_name = parts[-1].lower()
        for name, minutes in players.items():
            name_parts = name.split()
            if len(name_parts) >= 2:
                cache_first_initial = name_parts[0][0].lower()
                cache_last_name = name_parts[-1].lower()
                if first_initial == cache_first_initial and last_name == cache_last_name:
                    return minutes

    return None


def show_cache():
    """Display cached data."""
    cache = load_cache()
    if not cache:
        print("No valid cache found")
        return

    print(f"Scraped at: {cache['scraped_at']}")
    print(f"Date: {cache['date']}")
    print(f"Players: {len(cache['players'])}\n")

    # Sort by minutes descending
    sorted_players = sorted(
        cache['players'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"{'Player':<25} {'Minutes':>8}")
    print("-" * 35)
    for name, mins in sorted_players[:30]:
        print(f"{name:<25} {mins:>8.1f}")

    if len(sorted_players) > 30:
        print(f"... and {len(sorted_players) - 30} more")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        show_cache()
    else:
        players = scrape_razzball_minutes()
        if players:
            save_cache(players)
            print(f"\nSample players:")
            for name, mins in list(players.items())[:10]:
                print(f"  {name}: {mins} min")
