#!/usr/bin/env python3

import sys
import os
import json
import re
import requests
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python initiate.py {symbol}")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    sb_auth = normalize_auth(os.getenv("SB_AUTH"))

    base_dir = Path("sources") / symbol
    base_dir.mkdir(parents=True, exist_ok=True)

    new_dir = get_next_dir(base_dir)
    print(f"Created directory: {new_dir}")
    print("Fetching JSON files:")

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

    # For today-running-trade, the symbols parameter uses array notation
    # The curl shows symbols%5C[%5C] but that's likely meant to be symbols[]
    # Using proper URL encoding: symbols[] -> symbols%5B%5D
    today_running_trade_url = (
        f"https://exodus.stockbit.com/order-trade/running-trade"
        f"?sort=DESC&limit=100&order_by=RUNNING_TRADE_ORDER_BY_TIME&symbols%5B%5D={symbol}"
    )

    findata_url = (
        f"https://exodus.stockbit.com/findata-view/foreign-domestic/v1/chart-data/{symbol}"
        f"?market_type=MARKET_TYPE_REGULAR&period=PERIOD_RANGE_1M"
    )

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

    # Fetch JSON files if auth exists
    if sb_auth:
        md_path = new_dir / "market-detector.json"
        fetch_json(market_url, headers, md_path)

        pf_path = new_dir / "price-feed.json"
        fetch_json(price_feed_url, headers, pf_path)

        ob_path = new_dir / "orderbook.json"
        fetch_json(orderbook_url, headers, ob_path)

        rt_path = new_dir / "running-trade.json"
        fetch_json(running_trade_url, headers, rt_path)

        trt_path = new_dir / "today-running-trade.json"
        fetch_json(today_running_trade_url, headers, trt_path)

        fd_path = new_dir / "findata.json"
        fetch_json(findata_url, headers, fd_path)
    else:
        print("!!! SB_AUTH not set, skipping JSON fetches !!!")
        for filename in json_files:
            (new_dir / filename).touch()


if __name__ == "__main__":
    main()