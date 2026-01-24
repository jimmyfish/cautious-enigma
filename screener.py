#!/usr/bin/env python3
"""
Stockbit Screener with optional forecaster integration.

Usage:
    python screener.py                                  # Fetch default screener
    python screener.py -t TEMPLATE_ID                   # Fetch specific screener
    python screener.py --forecast [FORECASTER_ARGS]    # Run forecast.py after screener
    python screener.py --short [FORECASTER_ARGS]       # Run short.py after screener
    python screener.py --yf [FORECASTER_ARGS]          # Run yf.py after screener

Examples:
    python screener.py -t 4475032
    python screener.py --forecast -w -wid 5393656 -bl
    python screener.py --yf -n 5 --period 3mo -w -wid 5393656 -bl
    python screener.py -t 4475032 --short --session1 -w -wid 5393656
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def normalize_auth(token):
    if not token:
        return None
    return token if token.startswith("Bearer ") else f"Bearer {token}"


def fetch_screener(template_id: int, auth_token: str, output_path: Path) -> Optional[List[str]]:
    """
    Fetch screener template results and save to JSON file.

    Returns list of symbols or None on failure.
    """
    url = f"https://exodus.stockbit.com/screener/templates/{template_id}?type=TEMPLATE_TYPE_CUSTOM"

    headers = {
        "Accept": "application/json",
        "Authorization": auth_token or "",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
        "Sec-Fetch-Site": "same-site",
        "Origin": "https://stockbit.com",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Referer": "https://stockbit.com/",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

    try:
        print(f"Fetching screener template {template_id}...")
        res = requests.get(url, headers=headers)
        res.raise_for_status()

        data = res.json()
        output_path.write_text(json.dumps(data, indent=4, ensure_ascii=False))

        # Extract and display summary
        if "data" in data and "calcs" in data["data"]:
            symbols = [calc["company"]["symbol"] for calc in data["data"]["calcs"]]
            print(f"Fetched screener data: {len(symbols)} companies")
            print(f"  Symbols: {', '.join(symbols)}")
            print(f"  Saved to: {output_path}")
            return symbols
        else:
            print(f"Fetched screener data")
            print(f"  Saved to: {output_path}")
            print(f"  Warning: Unexpected response structure")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch screener: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response body: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def run_forecaster(
    forecaster: str,
    symbols: List[str],
    extra_args: List[str],
) -> int:
    """
    Run the selected forecaster with the given symbols.

    Args:
        forecaster: 'forecast', 'short', or 'yf'
        symbols: List of stock symbols from screener
        extra_args: Additional arguments to pass to the forecaster

    Returns:
        Exit code from the forecaster
    """
    workspace_root = Path(__file__).resolve().parent

    if forecaster == "forecast":
        script = workspace_root / "forecast.py"
        # forecast.py processes one symbol at a time, so we run it for each
        return run_single_symbol_forecaster(script, symbols, extra_args)

    elif forecaster == "short":
        script = workspace_root / "short.py"
        # short.py processes one symbol at a time
        return run_single_symbol_forecaster(script, symbols, extra_args)

    elif forecaster == "yf":
        script = workspace_root / "yf.py"
        # yf.py supports comma-separated symbols, need to add .JK suffix
        symbols_jk = [f"{s}.JK" for s in symbols]
        symbols_str = ",".join(symbols_jk)

        cmd = [sys.executable, str(script), symbols_str] + extra_args
        print(f"\n{'=' * 70}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'=' * 70}\n")

        result = subprocess.run(cmd)
        return result.returncode

    else:
        print(f"Unknown forecaster: {forecaster}")
        return 1


def run_single_symbol_forecaster(
    script: Path,
    symbols: List[str],
    extra_args: List[str],
) -> int:
    """
    Run a forecaster that processes one symbol at a time.

    For watchlist args, we use --keep after the first symbol to accumulate results.
    """
    if not symbols:
        print("No symbols to process")
        return 0

    # Check if watchlist args are present
    has_watchlist = "-w" in extra_args or "--watchlist" in extra_args
    has_keep = "--keep" in extra_args

    exit_code = 0

    for i, symbol in enumerate(symbols):
        # Build command
        cmd = [sys.executable, str(script), symbol] + extra_args.copy()

        # Add --keep after the first symbol to accumulate watchlist results
        if has_watchlist and not has_keep and i > 0:
            cmd.append("--keep")

        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{len(symbols)}] Running: {' '.join(cmd)}")
        print(f"{'=' * 70}\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Warning: {symbol} failed with exit code {result.returncode}")
            exit_code = result.returncode

    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Stockbit Screener with optional forecaster integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python screener.py -t 4475032
    python screener.py --forecast -w -wid 5393656 -bl
    python screener.py --yf -n 5 --period 3mo -w -wid 5393656 -bl
    python screener.py -t 4475032 --short --session1 -w -wid 5393656
        """,
    )

    parser.add_argument(
        "-t", "--template",
        type=int,
        default=4475032,
        dest="template_id",
        help="Screener template ID (default: 4475032)",
    )

    # Mutually exclusive forecaster options
    forecaster_group = parser.add_mutually_exclusive_group()
    forecaster_group.add_argument(
        "--forecast",
        action="store_true",
        help="Run forecast.py after screener (daily forecasting)",
    )
    forecaster_group.add_argument(
        "--short",
        action="store_true",
        help="Run short.py after screener (intraday forecasting)",
    )
    forecaster_group.add_argument(
        "--yf",
        action="store_true",
        help="Run yf.py after screener (Yahoo Finance data)",
    )

    # Parse known args, remaining go to forecaster
    args, extra_args = parser.parse_known_args()

    # Get auth token from environment
    sb_auth = normalize_auth(os.getenv("SB_AUTH"))

    if not sb_auth:
        print("!!! SB_AUTH not set in .env file !!!")
        print("Please set SB_AUTH in your .env file:")
        print("  SB_AUTH=Bearer YOUR_TOKEN_HERE")
        sys.exit(1)

    # Output path: screener.json in workspace root
    workspace_root = Path(__file__).resolve().parent
    output_path = workspace_root / "screener.json"

    # Fetch and save screener data
    symbols = fetch_screener(args.template_id, sb_auth, output_path)

    if symbols is None:
        print("\nFailed to fetch screener data")
        sys.exit(1)

    if not symbols:
        print("\nNo symbols found in screener")
        sys.exit(0)

    print(f"\nScreener data fetched successfully!")

    # Run forecaster if requested
    if args.forecast:
        print(f"\nRunning forecast.py for {len(symbols)} symbols...")
        exit_code = run_forecaster("forecast", symbols, extra_args)
        sys.exit(exit_code)

    elif args.short:
        print(f"\nRunning short.py for {len(symbols)} symbols...")
        exit_code = run_forecaster("short", symbols, extra_args)
        sys.exit(exit_code)

    elif args.yf:
        print(f"\nRunning yf.py for {len(symbols)} symbols...")
        exit_code = run_forecaster("yf", symbols, extra_args)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
