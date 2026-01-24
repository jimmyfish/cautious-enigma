#!/usr/bin/env python3
"""
Stockbit Watchlist API Integration.
Provides functions to manage watchlists via the Stockbit API.

Reads SB_AUTH from .env file automatically.
"""

import os
import requests
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
BASE_URL = "https://exodus.stockbit.com"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://stockbit.com",
    "Referer": "https://stockbit.com/",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.2 Safari/605.1.15",
}


def get_bearer_token() -> str:
    """Get bearer token from .env file (SB_AUTH)."""
    token = os.environ.get("SB_AUTH")
    if not token:
        raise ValueError(
            "SB_AUTH not found in .env file. "
            "Add your Stockbit bearer token: SB_AUTH=your_token"
        )
    return token


def normalize_symbol_for_stockbit(symbol: str) -> str:
    """
    Normalize symbol for Stockbit API.

    Strips exchange suffixes like .JK (Yahoo Finance Indonesian stocks).
    """
    # Remove common exchange suffixes
    suffixes = [".JK", ".SI", ".HK", ".L", ".T", ".AX", ".TO", ".NS", ".BO"]
    symbol_upper = symbol.upper()
    for suffix in suffixes:
        if symbol_upper.endswith(suffix):
            return symbol_upper[:-len(suffix)]
    return symbol_upper


def get_headers() -> Dict[str, str]:
    """Get headers with authorization."""
    headers = DEFAULT_HEADERS.copy()
    headers["Authorization"] = f"Bearer {get_bearer_token()}"
    return headers


def get_watchlist_items(watchlist_id: str) -> List[Dict[str, Any]]:
    """
    Get all items in a watchlist.

    Returns list of items with company_id and symbol.
    """
    url = f"{BASE_URL}/watchlist/{watchlist_id}"
    params = {"page": 1, "limit": 500, "setfincol": 1}

    response = requests.get(url, headers=get_headers(), params=params)
    response.raise_for_status()

    data = response.json()
    items = data.get("data", {}).get("watchlist", [])

    return [
        {
            "company_id": item.get("company_id"),
            "symbol": item.get("symbol"),
            "name": item.get("name"),
        }
        for item in items
    ]


def search_company_id(symbol: str, watchlist_id: str, debug: bool = False) -> Optional[str]:
    """
    Search for company ID by symbol.

    Returns company_id (string) or None if not found.
    """
    url = f"{BASE_URL}/watchlist/search/company"
    params = {
        "keyword": symbol.upper(),
        "page": 1,
        "watchlist_id": watchlist_id,
        "limit": 10,  # Get more results to find exact match
    }

    response = requests.get(url, headers=get_headers(), params=params)
    response.raise_for_status()

    data = response.json()

    if debug:
        print(f"  DEBUG search '{symbol}': {data}")

    # API returns "companies" (plural), not "company"
    companies = data.get("data", {}).get("companies", [])

    if companies:
        # Find exact match for symbol (first result with matching name)
        symbol_upper = symbol.upper()
        for company in companies:
            if company.get("name", "").upper() == symbol_upper:
                return company.get("id")
        # Fallback to first result if no exact match
        return companies[0].get("id")
    return None


def add_to_watchlist(watchlist_id: str, company_id: str) -> bool:
    """
    Add a company to a watchlist.

    Returns True on success.
    """
    url = f"{BASE_URL}/watchlist/{watchlist_id}/company/item"
    # company_id needs to be sent as integer in payload
    payload = {"company_id": int(company_id)}

    response = requests.post(url, headers=get_headers(), json=payload)
    response.raise_for_status()

    return True


def delete_from_watchlist(watchlist_id: str, company_id: str) -> bool:
    """
    Delete a company from a watchlist.

    Returns True on success.
    """
    url = f"{BASE_URL}/watchlist/{watchlist_id}/company/{company_id}/item"

    response = requests.delete(url, headers=get_headers())
    response.raise_for_status()

    return True


def clear_watchlist(watchlist_id: str, verbose: bool = True) -> int:
    """
    Clear all items from a watchlist.

    Returns number of items deleted.
    """
    items = get_watchlist_items(watchlist_id)
    count = 0

    for item in items:
        company_id = item.get("company_id")
        symbol = item.get("symbol", "?")
        if company_id:
            try:
                delete_from_watchlist(watchlist_id, company_id)
                count += 1
                if verbose:
                    print(f"  Removed: {symbol}")
            except Exception as e:
                if verbose:
                    print(f"  Failed to remove {symbol}: {e}")

    return count


def update_watchlist(
    watchlist_id: str,
    symbols: List[str],
    keep_existing: bool = False,
    verbose: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Update watchlist with given symbols.

    Args:
        watchlist_id: The watchlist ID to update
        symbols: List of stock symbols to add
        keep_existing: If False, clear watchlist first (default: False)
        verbose: Print progress messages
        debug: Print debug info for API responses

    Returns:
        Dict with added, failed, and skipped counts
    """
    result = {
        "added": [],
        "failed": [],
        "skipped": [],
        "removed": 0,
    }

    if not symbols:
        if verbose:
            print("  No symbols to add to watchlist")
        return result

    # Clear existing items if not keeping
    if not keep_existing:
        if verbose:
            print(f"\nClearing watchlist {watchlist_id}...")
        result["removed"] = clear_watchlist(watchlist_id, verbose=verbose)
        if verbose:
            print(f"  Cleared {result['removed']} items")

    # Get current watchlist items to check for duplicates
    current_items = get_watchlist_items(watchlist_id)
    current_symbols = {item.get("symbol", "").upper() for item in current_items}

    if verbose:
        print(f"\nAdding {len(symbols)} symbols to watchlist {watchlist_id}...")

    for symbol in symbols:
        # Normalize symbol for Stockbit (strip .JK etc.)
        symbol_normalized = normalize_symbol_for_stockbit(symbol)

        # Skip if already in watchlist
        if symbol_normalized in current_symbols:
            result["skipped"].append(symbol_normalized)
            if verbose:
                print(f"  Skipped (already exists): {symbol_normalized}")
            continue

        # Search for company ID
        try:
            company_id = search_company_id(symbol_normalized, watchlist_id, debug=debug)
            if company_id is None:
                result["failed"].append({"symbol": symbol_normalized, "reason": "Not found"})
                if verbose:
                    print(f"  Failed (not found): {symbol_normalized}")
                continue

            # Add to watchlist
            add_to_watchlist(watchlist_id, company_id)
            result["added"].append(symbol_normalized)
            current_symbols.add(symbol_normalized)  # Update local set
            if verbose:
                print(f"  Added: {symbol_normalized}")

        except Exception as e:
            result["failed"].append({"symbol": symbol_normalized, "reason": str(e)})
            if verbose:
                print(f"  Failed: {symbol_normalized} - {e}")

    return result


def print_watchlist_summary(result: Dict[str, Any]):
    """Print a summary of watchlist update operation."""
    print(f"\nWatchlist Update Summary:")
    print(f"  Added: {len(result['added'])}")
    if result["added"]:
        print(f"         {', '.join(result['added'])}")
    print(f"  Skipped: {len(result['skipped'])}")
    print(f"  Failed: {len(result['failed'])}")
    if result["failed"]:
        for f in result["failed"]:
            print(f"         {f['symbol']}: {f['reason']}")
    if result["removed"] > 0:
        print(f"  Removed: {result['removed']}")


def add_watchlist_args(parser):
    """Add watchlist-related arguments to an argument parser."""
    watchlist_group = parser.add_argument_group("watchlist options")
    watchlist_group.add_argument(
        "-w", "--watchlist",
        action="store_true",
        help="Update watchlist after analysis"
    )
    watchlist_group.add_argument(
        "-bl", "--bullish",
        action="store_true",
        help="Only add bullish symbols to watchlist (requires --watchlist)"
    )
    watchlist_group.add_argument(
        "-br", "--bearish",
        action="store_true",
        help="Only add bearish symbols to watchlist (requires --watchlist)"
    )
    watchlist_group.add_argument(
        "-wid", "--watchlist-id",
        type=str,
        help="Watchlist ID (mandatory with --watchlist)"
    )
    watchlist_group.add_argument(
        "--keep",
        action="store_true",
        help="Keep existing watchlist items before adding new symbols (default: purge)"
    )
    watchlist_group.add_argument(
        "--wl-debug",
        action="store_true",
        help="Debug watchlist API responses"
    )


def validate_watchlist_args(args) -> bool:
    """
    Validate watchlist arguments.

    Returns True if valid, raises ValueError if invalid.
    """
    if args.watchlist:
        if not args.watchlist_id:
            raise ValueError("--watchlist-id (-wid) is required when using --watchlist (-w)")

        if args.bullish and args.bearish:
            raise ValueError("Cannot use both --bullish (-bl) and --bearish (-br) together")

    if (args.bullish or args.bearish) and not args.watchlist:
        raise ValueError("--bullish (-bl) and --bearish (-br) require --watchlist (-w)")

    return True


def filter_symbols_by_outlook(
    symbols_with_outlook: List[Dict[str, Any]],
    bullish_only: bool = False,
    bearish_only: bool = False,
) -> List[str]:
    """
    Filter symbols based on their outlook.

    Args:
        symbols_with_outlook: List of dicts with 'symbol' and 'outlook' keys
        bullish_only: If True, only return bullish symbols
        bearish_only: If True, only return bearish symbols

    Returns:
        List of filtered symbol strings
    """
    if not bullish_only and not bearish_only:
        return [s["symbol"] for s in symbols_with_outlook]

    filtered = []
    for s in symbols_with_outlook:
        outlook = s.get("outlook", "").upper()
        if bullish_only and outlook == "BULLISH":
            filtered.append(s["symbol"])
        elif bearish_only and outlook == "BEARISH":
            filtered.append(s["symbol"])

    return filtered


if __name__ == "__main__":
    # Test the module
    import sys

    if len(sys.argv) < 2:
        print("Usage: python watchlist.py <watchlist_id> [symbol1 symbol2 ...]")
        sys.exit(1)

    wid = sys.argv[1]

    try:
        # List current items
        print(f"Current watchlist {wid}:")
        items = get_watchlist_items(wid)
        for item in items:
            print(f"  {item['symbol']} (id: {item['company_id']})")

        # Add symbols if provided
        if len(sys.argv) > 2:
            symbols = sys.argv[2:]
            result = update_watchlist(wid, symbols, keep_existing=False)
            print_watchlist_summary(result)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
