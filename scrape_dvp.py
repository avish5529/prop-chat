#!/usr/bin/env python3
"""
Defense vs Position Scraper
---------------------------
Scrapes FantasyPros DvP data showing how many points/rebounds/assists
each team allows to each position (PG, SG, SF, PF, C).

Usage:
    python scrape_dvp.py           # Scrape and save to dvp_cache.json
    python scrape_dvp.py --show    # Show current cache without scraping

Data is used to adjust projections based on position-specific matchups.
"""

from playwright.sync_api import sync_playwright
import json
from datetime import datetime
import sys
import os

CACHE_FILE = "dvp_cache.json"

# Team abbreviation mapping (FantasyPros -> standard)
TEAM_ABBR_MAP = {
    "NOR": "NOP",  # New Orleans
    "UTH": "UTA",  # Utah
    "PHO": "PHX",  # Phoenix
    "SAS": "SAS",
    "GSW": "GSW",
    "LAL": "LAL",
    "LAC": "LAC",
    "BKN": "BKN",
    "NYK": "NYK",
    "BOS": "BOS",
    "MIA": "MIA",
    "CHI": "CHI",
    "CLE": "CLE",
    "DET": "DET",
    "IND": "IND",
    "MIL": "MIL",
    "ATL": "ATL",
    "CHA": "CHA",
    "ORL": "ORL",
    "WAS": "WAS",
    "TOR": "TOR",
    "PHI": "PHI",
    "DEN": "DEN",
    "MIN": "MIN",
    "OKC": "OKC",
    "POR": "POR",
    "SAC": "SAC",
    "DAL": "DAL",
    "HOU": "HOU",
    "MEM": "MEM",
}


def scrape_dvp() -> dict:
    """
    Scrape Defense vs Position data from FantasyPros.

    Returns dict with structure:
    {
        "updated_at": "2026-03-21T...",
        "source": "fantasypros",
        "teams": {
            "WAS": {
                "PG": {"pts_allowed": 24.7, "reb_allowed": 6.5, "ast_allowed": 9.1, ...},
                "SG": {...},
                ...
            },
            ...
        },
        "rankings": {
            "PG": {"pts": ["WAS", "DAL", ...], "reb": [...], "ast": [...]},
            ...
        }
    }
    """

    dvp_data = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = context.new_page()

        print("[DvP] Loading FantasyPros page...")
        page.goto(
            "https://www.fantasypros.com/daily-fantasy/nba/fanduel-defense-vs-position.php",
            timeout=60000,
            wait_until="domcontentloaded"
        )
        page.wait_for_timeout(3000)

        positions = ["PG", "SG", "SF", "PF", "C"]

        for pos in positions:
            print(f"[DvP] Scraping {pos}...")

            # Click position tab via JavaScript
            page.evaluate(f"""
                const links = document.querySelectorAll('a');
                for (const link of links) {{
                    if (link.textContent.trim() === '{pos}') {{
                        link.click();
                        break;
                    }}
                }}
            """)
            page.wait_for_timeout(1500)

            # Get visible rows
            rows = page.query_selector_all("table#data-table tbody tr")

            for row in rows:
                # Check visibility
                is_visible = page.evaluate("""(el) => {
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && style.visibility !== 'hidden';
                }""", row)

                if not is_visible:
                    continue

                cells = row.query_selector_all("td")
                if len(cells) >= 9:
                    team_cell = cells[0].inner_text().strip()
                    team_abbr = team_cell[:3].upper()

                    # Normalize team abbreviation
                    team_abbr = TEAM_ABBR_MAP.get(team_abbr, team_abbr)

                    try:
                        data = {
                            "pts_allowed": float(cells[2].inner_text().strip()),
                            "reb_allowed": float(cells[3].inner_text().strip()),
                            "ast_allowed": float(cells[4].inner_text().strip()),
                            "stl_allowed": float(cells[6].inner_text().strip()),
                            "blk_allowed": float(cells[7].inner_text().strip()),
                            "fd_pts_allowed": float(cells[9].inner_text().strip()) if len(cells) > 9 else 0
                        }

                        if team_abbr not in dvp_data:
                            dvp_data[team_abbr] = {}
                        dvp_data[team_abbr][pos] = data

                    except (ValueError, IndexError):
                        pass

            print(f"[DvP]   → {len([t for t in dvp_data if pos in dvp_data[t]])} teams")

        browser.close()

    # Calculate rankings for each position/stat
    rankings = {}
    for pos in positions:
        rankings[pos] = {
            "pts": [],  # Teams sorted by pts allowed (most to least)
            "reb": [],
            "ast": [],
        }

        # Get teams with this position data
        teams_with_pos = [(t, dvp_data[t][pos]) for t in dvp_data if pos in dvp_data[t]]

        # Sort by pts allowed (descending - worst defense first)
        rankings[pos]["pts"] = [t for t, _ in sorted(teams_with_pos, key=lambda x: x[1]["pts_allowed"], reverse=True)]
        rankings[pos]["reb"] = [t for t, _ in sorted(teams_with_pos, key=lambda x: x[1]["reb_allowed"], reverse=True)]
        rankings[pos]["ast"] = [t for t, _ in sorted(teams_with_pos, key=lambda x: x[1]["ast_allowed"], reverse=True)]

    # Add rank to each team's data
    for pos in positions:
        for stat in ["pts", "reb", "ast"]:
            for rank, team in enumerate(rankings[pos][stat], 1):
                if team in dvp_data and pos in dvp_data[team]:
                    dvp_data[team][pos][f"{stat}_rank"] = rank

    output = {
        "updated_at": datetime.now().isoformat(),
        "source": "fantasypros",
        "platform": "fanduel",
        "teams": dvp_data,
        "rankings": rankings
    }

    return output


def save_cache(data: dict):
    """Save DvP data to cache file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[DvP] Saved to {CACHE_FILE}")


def load_cache() -> dict:
    """Load DvP data from cache file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return None


def show_cache():
    """Display current cache data."""
    data = load_cache()
    if not data:
        print("No cache file found. Run without --show to scrape.")
        return

    print(f"DvP Cache (updated: {data['updated_at']})")
    print("=" * 70)

    positions = ["PG", "SG", "SF", "PF", "C"]

    # Show worst defenses per position
    for pos in positions:
        print(f"\n{pos} - Worst Defenses (target):")
        rankings = data["rankings"][pos]["pts"]
        for i, team in enumerate(rankings[:5], 1):
            team_data = data["teams"][team][pos]
            print(f"  {i}. {team}: {team_data['pts_allowed']:.1f} pts, {team_data['reb_allowed']:.1f} reb, {team_data['ast_allowed']:.1f} ast")

    print("\n" + "=" * 70)
    print("Best Defenses (avoid):")
    for pos in positions:
        rankings = data["rankings"][pos]["pts"]
        best = rankings[-3:][::-1]  # Last 3, reversed
        best_str = ", ".join([f"{t} ({data['teams'][t][pos]['pts_allowed']:.1f})" for t in best])
        print(f"  {pos}: {best_str}")


def get_dvp_adjustment(team_abbr: str, position: str, prop_type: str = "points") -> float:
    """
    Get DvP-based adjustment factor for a player.

    Args:
        team_abbr: Opponent team abbreviation (e.g., "BOS")
        position: Player position (PG, SG, SF, PF, C)
        prop_type: Type of prop (points, rebounds, assists, pra, pr, pa, ra)

    Returns:
        Adjustment multiplier (e.g., 1.05 for +5%, 0.95 for -5%)
    """
    data = load_cache()
    if not data or team_abbr not in data["teams"]:
        return 1.0

    team_data = data["teams"].get(team_abbr, {}).get(position, {})
    if not team_data:
        return 1.0

    # Map prop type to relevant stat
    prop_to_stats = {
        "points": ["pts"],
        "rebounds": ["reb"],
        "assists": ["ast"],
        "pra": ["pts", "reb", "ast"],
        "pr": ["pts", "reb"],
        "pa": ["pts", "ast"],
        "ra": ["reb", "ast"],
    }

    stats = prop_to_stats.get(prop_type, ["pts"])

    # Calculate average rank across relevant stats
    ranks = []
    for stat in stats:
        rank_key = f"{stat}_rank"
        if rank_key in team_data:
            ranks.append(team_data[rank_key])

    if not ranks:
        return 1.0

    avg_rank = sum(ranks) / len(ranks)

    # Convert rank to adjustment
    # Rank 1-5 (worst defense): +5% to +10%
    # Rank 6-10: +2% to +5%
    # Rank 11-20: -2% to +2%
    # Rank 21-25: -5% to -2%
    # Rank 26-30 (best defense): -10% to -5%

    if avg_rank <= 5:
        adjustment = 1.05 + (5 - avg_rank) * 0.01  # 1.05 to 1.09
    elif avg_rank <= 10:
        adjustment = 1.02 + (10 - avg_rank) * 0.006  # 1.02 to 1.05
    elif avg_rank <= 20:
        adjustment = 1.0 + (15 - avg_rank) * 0.002  # 0.99 to 1.01
    elif avg_rank <= 25:
        adjustment = 0.98 + (25 - avg_rank) * 0.006  # 0.95 to 0.98
    else:
        adjustment = 0.95 - (avg_rank - 25) * 0.01  # 0.90 to 0.95

    return round(adjustment, 3)


def main():
    if "--show" in sys.argv:
        show_cache()
        return

    print("[DvP] Starting scrape...")
    data = scrape_dvp()
    save_cache(data)

    print("\n" + "=" * 70)
    print("Sample rankings (worst defense = good matchup):")

    for pos in ["PG", "C"]:
        print(f"\n{pos} - Top 5 to target:")
        for i, team in enumerate(data["rankings"][pos]["pts"][:5], 1):
            team_data = data["teams"][team][pos]
            print(f"  {i}. {team}: {team_data['pts_allowed']:.1f} pts allowed")


if __name__ == "__main__":
    main()
