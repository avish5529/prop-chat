"""
BettingPros Closing Line Scraper

Scrapes historical closing lines from BettingPros player game logs
and updates backtest_data.csv with the closing_line values.

Usage:
    pip install playwright pandas
    playwright install chromium
    python scrape_closing_lines.py
"""

import asyncio
import pandas as pd
from playwright.async_api import async_playwright
import re
import unicodedata
import json
from pathlib import Path

# URL patterns for BettingPros
BASE_URL = "https://www.bettingpros.com/nba/props"

# Map our prop types to BettingPros URL slugs
# NOTE: PRA is "points-assists-rebounds" (PAR order), NOT "points-rebounds-assists"
PROP_TYPE_MAP = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "pra": "points-assists-rebounds",  # FIXED: was "points-rebounds-assists" which redirected to points
    "pr": "points-rebounds",
    "pa": "points-assists",
    "ra": "rebounds-assists",
}

# Cache file for progress tracking
CACHE_FILE = "scrape_cache.json"


def player_name_to_slug(name: str) -> str:
    """Convert 'Nikola Jokić' to 'nikola-jokic'"""
    # Normalize unicode (ć -> c, č -> c, ñ -> n, etc.)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Remove suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV)$', '', name, flags=re.IGNORECASE)
    # Remove dots, apostrophes
    name = re.sub(r'[.\']', '', name)
    return name.lower().replace(' ', '-')


def parse_date(date_str: str) -> str:
    """
    Parse '2/19' to '2/19/26' or '10/21' to '10/21/25' format to match backtest_data.
    NBA season runs Oct-Apr, so:
    - Oct, Nov, Dec games are in 2025
    - Jan, Feb, Mar, Apr games are in 2026
    """
    date_str = date_str.strip()
    # BettingPros format: "2/19" (month/day only)
    match = re.match(r'^(\d{1,2})/(\d{1,2})$', date_str)
    if match:
        month = int(match.group(1))
        day = match.group(2)
        # Oct(10), Nov(11), Dec(12) are in 2025; Jan-Sep are in 2026
        year = "25" if month >= 10 else "26"
        return f"{month}/{day}/{year}"
    return None


def load_cache() -> dict:
    """Load cached scraped data"""
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {"scraped_combos": [], "lines": {}}


def save_cache(cache: dict):
    """Save cache to disk"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


async def trigger_game_log_load(page):
    """
    Toggle the season dropdown to trigger lazy loading of game log data.
    BettingPros doesn't load the game log until you interact with the dropdown.
    """
    try:
        # Find and click the season dropdown (shows "2025 Season")
        dropdown = await page.query_selector('text=2025 Season')
        if not dropdown:
            return False

        await dropdown.click()
        await asyncio.sleep(0.5)

        # Click 2024 Season to trigger load
        option_2024 = await page.query_selector('text=2024 Season')
        if option_2024:
            await option_2024.click()
            await asyncio.sleep(1)

            # Click dropdown again (now shows "2024 Season")
            dropdown = await page.query_selector('text=2024 Season')
            if dropdown:
                await dropdown.click()
                await asyncio.sleep(0.5)

                # Select 2025 Season again
                option_2025 = await page.query_selector('text=2025 Season')
                if option_2025:
                    await option_2025.click()
                    await asyncio.sleep(1)
                    return True

        return False
    except Exception:
        return False


async def scrape_player_prop(page, player_name: str, prop_type: str) -> dict:
    """
    Scrape the game log table for a player/prop combination.
    Returns dict mapping game_date -> closing_line
    """
    slug = player_name_to_slug(player_name)
    prop_slug = PROP_TYPE_MAP.get(prop_type)

    if not prop_slug:
        print(f"  Unknown prop type: {prop_type}")
        return {}

    url = f"{BASE_URL}/{slug}/{prop_slug}/"

    try:
        response = await page.goto(url, timeout=20000)

        # Check if page loaded successfully
        if response.status != 200:
            print(f"    HTTP {response.status}")
            return {}

        # Wait for initial content
        await asyncio.sleep(2)

        # Check if we have a game log table already
        tables = await page.query_selector_all('table')
        has_game_log = False
        for table in tables:
            headers = await table.query_selector_all('th')
            header_texts = [await h.inner_text() for h in headers]
            if 'Date' in header_texts and 'Prop Line' in header_texts:
                rows = await table.query_selector_all('tbody tr')
                if len(rows) > 0:
                    has_game_log = True
                    break

        # If no game log, try toggling the season dropdown to trigger load
        if not has_game_log:
            await trigger_game_log_load(page)
            await asyncio.sleep(1)

        # Now find and parse the game log table
        tables = await page.query_selector_all('table')
        results = {}

        for table in tables:
            headers = await table.query_selector_all('th')
            header_texts = [await h.inner_text() for h in headers]

            # The game log table has 'Date' and 'Prop Line' columns
            if 'Date' not in header_texts or 'Prop Line' not in header_texts:
                continue

            date_col = header_texts.index('Date')
            line_col = header_texts.index('Prop Line')

            rows = await table.query_selector_all('tbody tr')

            for row in rows:
                cells = await row.query_selector_all('td')
                if len(cells) <= max(date_col, line_col):
                    continue

                date_text = (await cells[date_col].inner_text()).strip()
                line_text = (await cells[line_col].inner_text()).strip()

                game_date = parse_date(date_text)
                if not game_date:
                    continue

                line_match = re.search(r'[\d.]+', line_text)
                if line_match:
                    closing_line = float(line_match.group())
                    results[game_date] = closing_line

            break  # Found the right table

        return results

    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")
        return {}


async def main():
    # Load the backtest data
    df = pd.read_csv('backtest_data.csv')
    print(f"Loaded {len(df)} rows")
    print(f"Missing closing_line: {df['closing_line'].isna().sum()}")

    # Load cache
    cache = load_cache()
    print(f"Cache has {len(cache['scraped_combos'])} already scraped combos")

    # Get unique player/prop combinations that need data
    missing = df[df['closing_line'].isna()]
    combos = missing[['player_name', 'prop_type']].drop_duplicates()

    # Filter out already scraped combos
    combos_to_scrape = []
    for _, row in combos.iterrows():
        key = f"{row['player_name']}|{row['prop_type']}"
        if key not in cache['scraped_combos']:
            combos_to_scrape.append((row['player_name'], row['prop_type']))

    print(f"Need to scrape {len(combos_to_scrape)} player/prop combinations")

    if not combos_to_scrape:
        print("Nothing new to scrape!")
    else:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1280, "height": 800})

            for i, (player, prop) in enumerate(combos_to_scrape):
                print(f"\n[{i+1}/{len(combos_to_scrape)}] {player} - {prop}")

                lines = await scrape_player_prop(page, player, prop)

                if lines:
                    print(f"    Got {len(lines)} games")
                    for date, line in lines.items():
                        cache_key = f"{player}|{prop}|{date}"
                        cache['lines'][cache_key] = line
                else:
                    print(f"    No data")

                # Mark as scraped (even if no data, so we don't retry)
                cache['scraped_combos'].append(f"{player}|{prop}")

                # Save cache after each player (in case of interruption)
                save_cache(cache)

                # Be nice to the server
                await asyncio.sleep(0.5)

            await browser.close()

    # Update the dataframe with all cached lines
    updates = 0
    for idx, row in df.iterrows():
        if pd.isna(row['closing_line']):
            cache_key = f"{row['player_name']}|{row['prop_type']}|{row['game_date']}"
            if cache_key in cache['lines']:
                df.at[idx, 'closing_line'] = cache['lines'][cache_key]
                updates += 1

    print(f"\nUpdated {updates} rows with closing lines")

    # Save back to CSV
    df.to_csv('backtest_data.csv', index=False)
    print("Saved to backtest_data.csv")

    # Summary
    still_missing = df['closing_line'].isna().sum()
    print(f"\nStill missing: {still_missing} ({still_missing/len(df)*100:.1f}%)")

    # Show which players have no data available
    if still_missing > 0:
        missing_players = df[df['closing_line'].isna()]['player_name'].unique()
        print(f"Players with missing data: {len(missing_players)}")


if __name__ == "__main__":
    asyncio.run(main())
