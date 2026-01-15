#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROUPS_FILE = Path(__file__).parent / "models" / "groups.json"

# Telegram config (set in .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message: str) -> bool:
    """Send a message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ! Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("  ✓ Telegram notification sent")
        return True
    except Exception as e:
        print(f"  ✗ Telegram error: {e}")
        return False


def normalize_auth(token):
    if not token:
        return None
    return token if token.startswith("Bearer ") else f"Bearer {token}"


def fetch_json(url, headers, output):
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        output.write_text(json.dumps(res.json(), indent=4, ensure_ascii=False))
        print(f"  ✓ {output.name}")
    except Exception as e:
        print(f"  ✗ {output.name}: {e}")
        output.touch()


def get_next_dir(path: Path) -> Path:
    numbers = [int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    next_num = (max(numbers) + 1) if numbers else 1
    new_dir = path / str(next_num)
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def load_groups():
    """Load sector groups from groups.json"""
    if not GROUPS_FILE.exists():
        return {}
    with open(GROUPS_FILE, "r") as f:
        groups = json.load(f)
    # Remove _comment key if present
    groups.pop("_comment", None)
    return groups


def get_symbols_from_input(input_val, groups):
    """
    Returns a list of symbols based on input.
    If input matches a sector name, return all symbols in that sector.
    Otherwise, treat it as a single symbol.
    """
    input_lower = input_val.lower()
    if input_lower in groups:
        return groups[input_lower]
    return [input_val.upper()]


def process_symbol(symbol, sb_auth, headers):
    """Process a single symbol - create directory and fetch data"""
    base_dir = Path(__file__).parent / "sources" / symbol
    base_dir.mkdir(parents=True, exist_ok=True)

    new_dir = get_next_dir(base_dir)
    print(f"\n[{symbol}] Created directory: {new_dir}")

    # Files to always create
    json_files = [
        "market-detector.json",
        "price-feed.json",
        "orderbook.json",
        "running-trade.json",
        "today-running-trade.json",
        "findata.json",
    ]

    # Dynamic date range: from today - 7 days until today (inclusive)
    today = date.today()
    from_date = today - timedelta(days=7)
    from_str = from_date.strftime("%Y-%m-%d")
    to_str = today.strftime("%Y-%m-%d")

    market_url = (
        f"https://exodus.stockbit.com/marketdetectors/{symbol}"
        f"?transaction_type=TRANSACTION_TYPE_NET"
        f"&from={from_str}&to={to_str}"
        f"&market_board=MARKET_BOARD_REGULER"
        f"&investor_type=INVESTOR_TYPE_FOREIGN"
        f"&limit=25"
    )

    price_feed_url = f"https://exodus.stockbit.com/company-price-feed/v2/orderbook/companies/{symbol}"
    orderbook_url = f"https://exodus.stockbit.com/company-price-feed/v2/orderbook/companies/{symbol}"
    running_trade_url = f"https://exodus.stockbit.com/order-trade/running-trade/chart/{symbol}?period=RT_PERIOD_LAST_1_DAY"
    today_running_trade_url = (
        f"https://exodus.stockbit.com/order-trade/running-trade"
        f"?sort=DESC&limit=100&order_by=RUNNING_TRADE_ORDER_BY_TIME&symbols%5B%5D={symbol}"
    )
    findata_url = (
        f"https://exodus.stockbit.com/findata-view/foreign-domestic/v1/chart-data/{symbol}"
        f"?market_type=MARKET_TYPE_REGULAR&period=PERIOD_RANGE_1M"
    )

    # Fetch JSON files if auth exists
    if sb_auth:
        fetch_json(market_url, headers, new_dir / "market-detector.json")
        fetch_json(price_feed_url, headers, new_dir / "price-feed.json")
        fetch_json(orderbook_url, headers, new_dir / "orderbook.json")
        fetch_json(running_trade_url, headers, new_dir / "running-trade.json")
        fetch_json(
            today_running_trade_url, headers, new_dir / "today-running-trade.json"
        )
        fetch_json(findata_url, headers, new_dir / "findata.json")
    else:
        print("  !!! SB_AUTH not set, creating empty files !!!")
        for filename in json_files:
            (new_dir / filename).touch()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch stock data for symbols or sectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python initiate.py BBCA                    # Single symbol
    python initiate.py -g banking              # Single group
    python initiate.py -g banking,health       # Multiple groups (comma-separated)
    python initiate.py -e banking              # All groups EXCEPT banking
    python initiate.py -e banking,health       # All groups EXCEPT banking and health
    python initiate.py --list-sectors          # List available sectors
        """
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Symbol (e.g., BBCA) - for single symbol processing",
    )
    parser.add_argument(
        "-g", "--groups",
        help="Group name(s) to include, comma-separated (e.g., banking,health)",
    )
    parser.add_argument(
        "-e", "--exclude",
        help="Group name(s) to exclude - process all groups EXCEPT these, comma-separated",
    )
    parser.add_argument(
        "--chunk",
        "-c",
        type=int,
        default=10,
        help="Number of symbols to process before pausing (default: 10)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=int,
        default=5,
        help="Seconds to wait between chunks (default: 5)",
    )
    parser.add_argument(
        "--list-sectors", action="store_true", help="List available sectors and exit"
    )

    args = parser.parse_args()
    groups = load_groups()

    # List sectors mode
    if args.list_sectors:
        print("Available sectors:")
        for sector, symbols in groups.items():
            print(f"  {sector}: {len(symbols)} symbols")
        total = sum(len(s) for s in groups.values())
        print(f"\nTotal: {len(groups)} sectors, {total} symbols")
        sys.exit(0)

    # Determine which symbols to process
    symbols = []
    is_sector = False
    selected_groups = []

    if args.exclude:
        # Process all groups EXCEPT the specified ones
        exclude_list = [g.strip().lower() for g in args.exclude.split(",") if g.strip()]
        invalid = [g for g in exclude_list if g not in groups]
        if invalid:
            parser.error(f"Unknown group(s): {', '.join(invalid)}. Use --list-sectors to see available groups.")

        selected_groups = [g for g in groups.keys() if g not in exclude_list]
        for group_name in selected_groups:
            symbols.extend(groups[group_name])
        is_sector = True
        print(f"Excluding groups: {', '.join(exclude_list)}")
        print(f"Processing groups: {', '.join(selected_groups)}")

    elif args.groups:
        # Process specified groups (comma-separated)
        group_list = [g.strip().lower() for g in args.groups.split(",") if g.strip()]
        invalid = [g for g in group_list if g not in groups]
        if invalid:
            parser.error(f"Unknown group(s): {', '.join(invalid)}. Use --list-sectors to see available groups.")

        selected_groups = group_list
        for group_name in group_list:
            symbols.extend(groups[group_name])
        is_sector = True
        print(f"Processing groups: {', '.join(group_list)}")

    elif args.input:
        # Single symbol or legacy sector name input
        symbols = get_symbols_from_input(args.input, groups)
        is_sector = args.input.lower() in groups
        if is_sector:
            selected_groups = [args.input.lower()]

    else:
        parser.error("Please provide: a symbol, -g <groups>, or -e <exclude_groups>")

    sb_auth = normalize_auth(os.getenv("SB_AUTH"))

    headers = {
        "Accept": "application/json",
        "Authorization": sb_auth or "",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
        "Sec-Fetch-Site": "same-site",
        "Origin": "https://stockbit.com",
        "Accept-Language": "EN",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Referer": "https://stockbit.com/",
    }

    total = len(symbols)
    if is_sector:
        print(f"\nProcessing {len(selected_groups)} group(s) with {total} symbols")
        print(f"Chunk size: {args.chunk}, Delay between chunks: {args.delay}s")

    for i, symbol in enumerate(symbols, 1):
        process_symbol(symbol, sb_auth, headers)

        # Chunking logic: pause after every chunk (except at the end)
        if is_sector and i < total and i % args.chunk == 0:
            remaining = total - i
            print(
                f"\n--- Processed {i}/{total} symbols. Waiting {args.delay}s before next chunk ({remaining} remaining)... ---"
            )
            time.sleep(args.delay)

    print(f"\n✓ Done! Processed {total} symbol(s).")

    # Send Telegram notification
    if is_sector:
        groups_str = ', '.join(selected_groups)
        msg = (
            f"<b>✅ Initiate Complete</b>\n\n"
            f"<b>Groups:</b> {groups_str}\n"
            f"<b>Symbols:</b> {total}\n"
            f"<b>Status:</b> Done"
        )
    else:
        msg = (
            f"<b>✅ Initiate Complete</b>\n\n"
            f"<b>Symbol:</b> {symbols[0] if symbols else 'N/A'}\n"
            f"<b>Status:</b> Done"
        )
    send_telegram(msg)


if __name__ == "__main__":
    main()
