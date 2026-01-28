#!/usr/bin/env python3
"""
Watchlist Management CLI

Simple utility to add/remove symbols from Stockbit watchlist.

Usage:
    python wl.py -wid 5393656 BBRI BBCA BMRI      # Add symbols (clears existing)
    python wl.py -wid 5393656 BBRI --keep         # Add without clearing
    python wl.py -wid 5393656 --clear             # Clear all items
    python wl.py -wid 5393656 --list              # List current items
"""

import argparse
import sys

from modules.watchlist import (
    get_watchlist_items,
    clear_watchlist,
    update_watchlist,
    print_watchlist_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage Stockbit Watchlist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python wl.py -wid 5393656 BBRI BBCA BMRI      # Add symbols (clears existing first)
    python wl.py -wid 5393656 BBRI BBCA --keep    # Add symbols (keep existing)
    python wl.py -wid 5393656 --clear             # Clear all items
    python wl.py -wid 5393656 --list              # List current items
    python wl.py -wid 5393656 --list --debug      # List with debug info
        """,
    )

    parser.add_argument(
        "symbols",
        nargs="*",
        help="Stock symbols to add (e.g., BBRI BBCA BMRI)",
    )
    parser.add_argument(
        "-wid", "--watchlist-id",
        type=str,
        required=True,
        help="Watchlist ID (required)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep existing items (default: clear before adding)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all items from watchlist",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List current watchlist items",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information",
    )

    args = parser.parse_args()

    try:
        # List current items
        if args.list:
            print(f"Watchlist {args.watchlist_id}:")
            items = get_watchlist_items(args.watchlist_id, debug=args.debug)
            if not items:
                print("  (empty)")
            else:
                for item in items:
                    if args.debug:
                        print(f"  {item['symbol']:<10} (id: {item['company_id']})")
                    else:
                        print(f"  {item['symbol']}")
                print(f"\nTotal: {len(items)} items")
            return 0

        # Clear watchlist
        if args.clear:
            print(f"Clearing watchlist {args.watchlist_id}...")
            count = clear_watchlist(args.watchlist_id, verbose=True)
            print(f"\nCleared {count} items")
            return 0

        # Add symbols
        if args.symbols:
            result = update_watchlist(
                args.watchlist_id,
                args.symbols,
                keep_existing=args.keep,
                verbose=True,
                debug=args.debug,
            )
            print_watchlist_summary(result)
            return 0

        # No action specified
        parser.print_help()
        return 1

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
