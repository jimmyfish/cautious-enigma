#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def normalize_auth(token):
    if not token:
        return None
    return token if token.startswith("Bearer ") else f"Bearer {token}"


def fetch_screener(template_id: int, auth_token: str, output_path: Path):
    """Fetch screener template results and save to JSON file."""
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
            print(f"✓ Fetched screener data: {len(symbols)} companies")
            print(f"  Symbols: {', '.join(symbols)}")
            print(f"  Saved to: {output_path}")
        else:
            print(f"✓ Fetched screener data")
            print(f"  Saved to: {output_path}")
            print(f"  Warning: Unexpected response structure")

        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to fetch screener: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response body: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    # Default screener template ID from the example
    default_template_id = 4475032

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            template_id = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid template ID '{sys.argv[1]}'. Must be a number.")
            print("Usage: python screener.py [TEMPLATE_ID]")
            print(
                f"  TEMPLATE_ID: Screener template ID (default: {default_template_id})"
            )
            sys.exit(1)
    else:
        template_id = default_template_id

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
    success = fetch_screener(template_id, sb_auth, output_path)

    if success:
        print("\n✓ Screener data fetched successfully!")
    else:
        print("\n✗ Failed to fetch screener data")
        sys.exit(1)


if __name__ == "__main__":
    main()
