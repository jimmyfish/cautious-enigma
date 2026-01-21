#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Tuple, Union

import requests
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# Initialize colorama
colorama_init()

# Rate limiting config based on API research
MAX_CONCURRENT_SYMBOLS = 3

Number = Union[int, float]

# Load environment variables from .env file
load_dotenv()

GROUPS_FILE = Path(__file__).parent / "models" / "groups.json"

# Telegram config (set in .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Thread-safe printing
_print_lock = threading.Lock()


def log(msg: str, color: str = ""):
    """Thread-safe colored print."""
    with _print_lock:
        print(f"{color}{msg}{Style.RESET_ALL}" if color else msg)


def send_telegram(message: str) -> bool:
    """Send a message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return True
    except Exception:
        return False


# =============================================================================
# Analysis Generation Functions (from scripts/analyze_market.py)
# =============================================================================

def to_int(value: Optional[Union[str, Number]]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def parse_numeric_string(value: Optional[Union[str, Number]]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    multiplier = 1.0
    if cleaned.endswith("B"):
        multiplier = 1e9
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1e6
        cleaned = cleaned[:-1]
    elif cleaned.endswith("K"):
        multiplier = 1e3
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace("(", "-").replace(")", "")
    if not cleaned:
        return 0.0
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def load_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()
        if not content:
            return None
        return json.loads(content)


def summarize_depth(data: dict) -> Dict[str, dict]:
    pf = data.get("data", {})
    bid_levels = pf.get("bid", []) or []
    offer_levels = pf.get("offer", []) or []

    def side_metrics(levels: Iterable[dict]) -> dict:
        volumes = [to_int(level.get("volume")) or 0 for level in levels]
        prices = [to_int(level.get("price")) or 0 for level in levels]
        total = sum(volumes)
        weighted_sum = sum(price * vol for price, vol in zip(prices, volumes))
        return {
            "total_volume": total,
            "top5_volume": sum(volumes[:5]),
            "top10_volume": sum(volumes[:10]),
            "weighted_price": (weighted_sum / total) if total else None,
            "max_cluster_volume": max(volumes) if volumes else 0,
            "max_cluster_price": prices[volumes.index(max(volumes))] if volumes else None,
        }

    summary: Dict[str, dict] = {"bid": side_metrics(bid_levels), "offer": side_metrics(offer_levels)}

    if bid_levels and offer_levels:
        best_bid = to_int(bid_levels[0].get("price"))
        best_ask = to_int(offer_levels[0].get("price"))
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
        spread_bps = (spread / best_bid * 10000) if spread and best_bid else None
        summary["top_of_book"] = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_vol": to_int(bid_levels[0].get("volume")),
            "ask_vol": to_int(offer_levels[0].get("volume")),
            "spread": spread,
            "spread_bps": spread_bps,
        }
    return summary


def summarize_price_series(data: dict) -> Dict[str, Union[int, float, List[Tuple[int, int]], None]]:
    price_chart = data.get("data", {}).get("price_chart_data", []) or []
    prices: List[int] = []
    for point in price_chart:
        price = to_int(point.get("value", {}).get("raw"))
        if price is not None:
            prices.append(price)

    summary: Dict[str, Union[int, float, List[Tuple[int, int]], None]] = {
        "count": len(prices),
        "start_price": prices[0] if prices else None,
        "end_price": prices[-1] if prices else None,
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
        "mean_price": mean(prices) if prices else None,
        "median_price": median(prices) if prices else None,
        "range": (max(prices) - min(prices)) if prices else None,
    }

    if len(prices) > 1:
        returns = []
        for prev, curr in zip(prices, prices[1:]):
            if prev:
                returns.append(curr / prev - 1)
        if returns:
            summary.update({
                "mean_return": mean(returns),
                "std_return": pstdev(returns) if len(returns) > 1 else 0.0,
                "max_intraday_jump": max(returns),
                "min_intraday_jump": min(returns),
            })

        indices = list(range(len(prices)))
        mean_x = mean(indices)
        mean_y = mean(prices)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, prices))
        denominator = sum((x - mean_x) ** 2 for x in indices)
        slope = numerator / denominator if denominator else 0.0
        summary["slope_per_interval"] = slope
        summary["slope_per_hour_equiv"] = slope * 60
        summary["last_hour_change"] = prices[-1] - prices[-60] if len(prices) >= 60 else None
    summary["top_levels"] = Counter(prices).most_common(5)
    return summary


def summarize_running_trades(data: dict) -> dict:
    trades = data.get("data", {}).get("running_trade", []) or []
    stats = {
        "trade_count": len(trades),
        "start_time": trades[-1].get("time") if trades else None,
        "end_time": trades[0].get("time") if trades else None,
        "lot_total": 0,
        "lot_buy": 0,
        "lot_sell": 0,
        "buy_count": 0,
        "sell_count": 0,
        "avg_trade_price": None,
        "unique_buyers": 0,
        "unique_sellers": 0,
    }

    buyers = set()
    sellers = set()
    price_accumulator = 0

    for trade in trades:
        price = to_int(trade.get("price"))
        lot = to_int(trade.get("lot")) or 0
        action = trade.get("action")
        stats["lot_total"] += lot
        if action == "buy":
            stats["buy_count"] += 1
            stats["lot_buy"] += lot
        elif action == "sell":
            stats["sell_count"] += 1
            stats["lot_sell"] += lot
        if price is not None:
            price_accumulator += price
        buyer = trade.get("buyer")
        seller = trade.get("seller")
        if buyer:
            buyers.add(buyer)
        if seller:
            sellers.add(seller)

    stats["unique_buyers"] = len(buyers)
    stats["unique_sellers"] = len(sellers)
    stats["avg_trade_price"] = price_accumulator / len(trades) if trades else None

    broker_lot = defaultdict(lambda: {"buy_lot": 0, "sell_lot": 0})
    for trade in trades:
        lot = to_int(trade.get("lot")) or 0
        buyer = trade.get("buyer")
        seller = trade.get("seller")
        if buyer:
            broker_lot[buyer]["buy_lot"] += lot
        if seller:
            broker_lot[seller]["sell_lot"] += lot

    broker_activity = []
    for broker, lots in broker_lot.items():
        broker_activity.append({
            "broker": broker,
            "buy_lot": lots["buy_lot"],
            "sell_lot": lots["sell_lot"],
            "net_lot": lots["buy_lot"] - lots["sell_lot"],
        })
    stats["broker_activity"] = sorted(broker_activity, key=lambda entry: entry["net_lot"], reverse=True)[:5]
    return stats


def summarize_broker_charts(data: dict) -> List[dict]:
    groups = data.get("data", {}).get("broker_chart_data", []) or []
    summary = []
    for group in groups:
        group_type = group.get("type")
        for entry in group.get("charts", []):
            chart = entry.get("chart", [])
            if not chart:
                continue
            start_raw = chart[0].get("value", {}).get("raw")
            end_raw = chart[-1].get("value", {}).get("raw")
            start_val = parse_numeric_string(start_raw)
            end_val = parse_numeric_string(end_raw)
            summary.append({
                "group_type": group_type,
                "broker_code": entry.get("broker_code"),
                "points": len(chart),
                "start_value_raw": start_raw,
                "end_value_raw": end_raw,
                "delta": (end_val - start_val) if (start_val is not None and end_val is not None) else None,
            })
    return summary


def summarize_market_detector(data: dict) -> dict:
    md = data.get("data", {})
    bandar = md.get("bandar_detector", {}) or {}
    broker_summary = md.get("broker_summary", {}) or {}

    total_buyer = bandar.get("total_buyer")
    total_seller = bandar.get("total_seller")
    net_brokers = None
    dominant_side = None
    if isinstance(total_buyer, (int, float)) and isinstance(total_seller, (int, float)):
        net_brokers = total_buyer - total_seller
        if net_brokers > 0:
            dominant_side = "buyers"
        elif net_brokers < 0:
            dominant_side = "sellers"
        else:
            dominant_side = "balanced"

    brokers_buy = broker_summary.get("brokers_buy", []) or []
    foreign_buyers = [
        b for b in brokers_buy if str(b.get("type", "")).lower() in {"asing", "foreign", "foreigner"}
    ]
    total_foreign_buy_volume = 0.0
    total_foreign_buy_value = 0.0
    for b in foreign_buyers:
        vol_raw = b.get("blot") or b.get("blotv")
        val_raw = b.get("bval") or b.get("bvalv")
        vol = parse_numeric_string(vol_raw)
        val = parse_numeric_string(val_raw)
        if vol is not None:
            total_foreign_buy_volume += vol
        if val is not None:
            total_foreign_buy_value += val

    return {
        "from": md.get("from"),
        "to": md.get("to"),
        "bandar": {
            "average": bandar.get("average"),
            "top1": bandar.get("top1"),
            "top3": bandar.get("top3"),
            "top5": bandar.get("top5"),
            "top10": bandar.get("top10"),
            "total_buyer": total_buyer,
            "total_seller": total_seller,
            "net_brokers": net_brokers,
            "dominant_side": dominant_side,
            "value": bandar.get("value"),
            "volume": bandar.get("volume"),
        },
        "top_buyers": broker_summary.get("brokers_buy", [])[:5],
        "top_sellers": broker_summary.get("brokers_sell", [])[:5],
        "foreign_focus": {
            "total_foreign_buy_volume": total_foreign_buy_volume or None,
            "total_foreign_buy_value": total_foreign_buy_value or None,
            "top_foreign_buyers": foreign_buyers[:5],
        },
    }


def summarize_findata(data: dict) -> dict:
    findata = data.get("data", {})
    summary: dict = {
        "from": findata.get("from"),
        "to": findata.get("to"),
        "last_updated": findata.get("last_updated"),
        "date_range": findata.get("summary", {}).get("date_range"),
    }

    def extract_value_field(field: dict) -> Optional[dict]:
        if not field or not isinstance(field, dict):
            return None
        value_obj = field.get("value", {})
        if isinstance(value_obj, dict):
            return {"raw": value_obj.get("raw"), "formatted": value_obj.get("formatted")}
        return None

    def extract_value_with_percentage(field: dict) -> Optional[dict]:
        if not field or not isinstance(field, dict):
            return None
        value_obj = field.get("value", {})
        percentage_obj = field.get("percentage", {})
        result = {}
        if isinstance(value_obj, dict):
            result["value"] = {"raw": value_obj.get("raw"), "formatted": value_obj.get("formatted")}
        if isinstance(percentage_obj, dict):
            result["percentage"] = {"raw": percentage_obj.get("raw"), "formatted": percentage_obj.get("formatted")}
        return result if result else None

    summary_section = findata.get("summary", {})
    if summary_section:
        summary["summary"] = {}
        for key in ["foreign_buy", "foreign_sell", "net_foreign", "domestic_buy", "domestic_sell", "net_domestic"]:
            if key in summary_section:
                summary["summary"][key] = extract_value_field(summary_section[key])
        volume_summary = summary_section.get("volume", {})
        if volume_summary:
            summary["summary"]["volume"] = {}
            for key in ["domestic_buy", "domestic_sell", "net_domestic", "foreign_buy", "foreign_sell",
                        "net_foreign_reguler", "net_foreign_tunai_nego", "net_foreign_all_market"]:
                if key in volume_summary:
                    summary["summary"]["volume"][key] = extract_value_field(volume_summary[key])

    for section_name in ["value", "volume", "frequency"]:
        section = findata.get(section_name, {})
        if section:
            summary[section_name] = {"total": extract_value_field(section.get("total", {}))}
            for key in ["foreign_buy", "foreign_sell", "domestic_buy", "domestic_sell", "foreign_total", "domestic_total"]:
                if key in section:
                    summary[section_name][key] = extract_value_with_percentage(section[key])

    def extract_participation(section: dict) -> Optional[dict]:
        if not section:
            return None
        foreign_total = section.get("foreign_total", {})
        domestic_total = section.get("domestic_total", {})
        ft_pct = (foreign_total.get("percentage") or {}).get("raw")
        dt_pct = (domestic_total.get("percentage") or {}).get("raw")
        if ft_pct is None and dt_pct is None:
            return None
        return {"foreign_pct": ft_pct, "domestic_pct": dt_pct}

    for section_name in ["value", "volume", "frequency"]:
        participation = extract_participation(summary.get(section_name, {}))
        if participation:
            summary[f"{section_name}_participation"] = participation

    return summary


def summarize_historical_price(data: dict) -> Dict[str, Union[int, float, List, None]]:
    """Summarize historical daily price data from price-feed API."""
    raw_data = data.get("data", {})

    # API returns data in data.result array
    if isinstance(raw_data, dict):
        history = raw_data.get("result", []) or raw_data.get("summary", []) or []
    elif isinstance(raw_data, list):
        history = raw_data
    else:
        history = []

    if not history or not isinstance(history, list):
        return {"count": 0, "latest": None, "history": []}

    # Extract latest day for backward compatibility with forecast.py
    latest = history[0] if history else {}

    # Parse all close prices and volumes for statistics
    closes = []
    volumes = []
    for item in history:
        close = parse_numeric_string(item.get("close"))
        vol = parse_numeric_string(item.get("volume"))
        if close is not None:
            closes.append(close)
        if vol is not None:
            volumes.append(vol)

    summary: Dict[str, Union[int, float, List, None]] = {
        # Latest day values (backward compatible with old price_feed format)
        "open": parse_numeric_string(latest.get("open")),
        "high": parse_numeric_string(latest.get("high")),
        "low": parse_numeric_string(latest.get("low")),
        "close": parse_numeric_string(latest.get("close")),
        "last": parse_numeric_string(latest.get("close")),
        "volume": parse_numeric_string(latest.get("volume")),
        "value": parse_numeric_string(latest.get("value")),
        "frequency": parse_numeric_string(latest.get("frequency")),
        "average": parse_numeric_string(latest.get("average")),
        "date": latest.get("date"),
        # Foreign flow from latest day (API provides this per day!)
        "foreign_buy": parse_numeric_string(latest.get("foreign_buy")),
        "foreign_sell": parse_numeric_string(latest.get("foreign_sell")),
        "foreign_net": parse_numeric_string(latest.get("net_foreign")),
        # Daily change from API
        "change": parse_numeric_string(latest.get("change")),
        "pct_change": parse_numeric_string(latest.get("change_percentage")),
        # Historical summary statistics
        "count": len(history),
        "price_min": min(closes) if closes else None,
        "price_max": max(closes) if closes else None,
        "price_mean": mean(closes) if closes else None,
        "volume_mean": mean(volumes) if volumes else None,
        # Full history for enhanced forecasting
        "history": history,
    }

    # Calculate multi-day price change if we have enough data
    if len(closes) >= 2:
        summary["price_change_total"] = closes[0] - closes[-1]  # Latest vs oldest
        summary["price_change_total_pct"] = ((closes[0] / closes[-1]) - 1) * 100 if closes[-1] else None

    return summary


def build_analysis_summary(base_dir: Path) -> dict:
    """Build comprehensive analysis summary from all source files."""
    price_feed = load_json_file(base_dir / "price-feed.json")
    running_trade = load_json_file(base_dir / "running-trade.json")
    today_running_trade = load_json_file(base_dir / "today-running-trade.json")
    market_detector = load_json_file(base_dir / "market-detector.json")
    orderbook = load_json_file(base_dir / "orderbook.json")
    findata = load_json_file(base_dir / "findata.json")

    symbol = base_dir.parent.name
    session = base_dir.name

    # Depth data now only comes from orderbook.json (price-feed is historical data)
    has_orderbook = orderbook and orderbook.get("data", {}).get("bid") and orderbook.get("data", {}).get("offer")

    available_files = sorted(p.name for p in base_dir.glob("*.json"))
    sources_analyzed = [
        name for name, data in [
            ("price-feed.json", price_feed),
            ("running-trade.json", running_trade),
            ("today-running-trade.json", today_running_trade),
            ("market-detector.json", market_detector),
            ("orderbook.json", orderbook if has_orderbook else None),
            ("findata.json", findata),
        ] if data is not None
    ]
    missing_sources = [
        name for name, data in [
            ("price-feed.json", price_feed),
            ("running-trade.json", running_trade),
            ("today-running-trade.json", today_running_trade),
            ("market-detector.json", market_detector),
            ("orderbook.json", orderbook if has_orderbook else None),
            ("findata.json", findata),
        ] if data is None
    ]

    md_summary = summarize_market_detector(market_detector) if market_detector else None
    fd_summary = summarize_findata(findata) if findata else None

    # Summarize historical price data
    pf_summary = summarize_historical_price(price_feed) if price_feed else None

    return {
        "metadata": {
            "symbol": symbol,
            "session": session,
            "base_dir": str(base_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "available_files": available_files,
            "sources_analyzed": sources_analyzed,
            "has_orderbook": has_orderbook,
            "missing_sources": missing_sources,
            "time_horizons": {
                "market_detector": {
                    "from": md_summary.get("from") if md_summary else None,
                    "to": md_summary.get("to") if md_summary else None,
                },
                "findata": {
                    "from": fd_summary.get("from") if fd_summary else None,
                    "to": fd_summary.get("to") if fd_summary else None,
                    "date_range": fd_summary.get("date_range") if fd_summary else None,
                },
                "price_feed": {
                    "date": pf_summary.get("date") if pf_summary else None,
                    "count": pf_summary.get("count") if pf_summary else None,
                },
            },
        },
        "price_feed": pf_summary,
        "depth": summarize_depth(orderbook) if has_orderbook else None,
        "price_series": summarize_price_series(running_trade) if running_trade else None,
        "running_trade": summarize_running_trades(today_running_trade) if today_running_trade else None,
        "broker_chart": summarize_broker_charts(running_trade) if running_trade else None,
        "market_detector": md_summary,
        "findata": fd_summary,
        "missing_sources": missing_sources,
    }


def generate_analysis(session_dir: Path) -> Optional[Path]:
    """Generate analysis JSON file for a session directory."""
    try:
        summary = build_analysis_summary(session_dir)
        output_path = session_dir / "analyzed.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return output_path
    except Exception as e:
        print(f"  ✗ Analysis generation failed: {e}")
        return None


def cleanup_source_files(session_dir: Path) -> int:
    """
    Delete source files that are no longer needed after analysis generation.
    Keeps: today-running-trade.json (for short.py) and analysis-data-*.json (for forecast.py)
    """
    files_to_delete = [
        "market-detector.json",
        "price-feed.json",
        "orderbook.json",
        "running-trade.json",
        "findata.json",
    ]
    deleted_count = 0
    for filename in files_to_delete:
        filepath = session_dir / filename
        if filepath.exists():
            filepath.unlink()
            deleted_count += 1
    return deleted_count


# =============================================================================
# Original Functions
# =============================================================================


def check_market_holiday(session: requests.Session, sample_symbol: str = "BBCA") -> Tuple[bool, Optional[str]]:
    """Check if today is a market holiday by comparing latest price data date with today.

    Returns: (is_holiday: bool, last_trading_date: Optional[str])
    """
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    # Request data from past 7 days to ensure we get the last trading date
    from_date = today - timedelta(days=7)
    from_str = from_date.strftime("%Y-%m-%d")

    url = (
        f"https://exodus.stockbit.com/company-price-feed/historical/summary/{sample_symbol}"
        f"?period=HS_PERIOD_DAILY&start_date={from_str}&end_date={today_str}&limit=5&page=1"
    )

    try:
        res = session.get(url, timeout=30)
        res.raise_for_status()
        data = res.json()

        history = data.get("data", {}).get("result", [])
        if not history:
            return True, None

        latest_date = history[0].get("date")
        if not latest_date:
            return True, None

        # If latest data date is not today, it's a holiday
        is_holiday = latest_date != today_str
        return is_holiday, latest_date

    except Exception as e:
        log(f"{Fore.YELLOW}Warning: Could not check market holiday: {e}{Style.RESET_ALL}")
        return False, None  # Proceed if we can't check


def normalize_auth(token):
    if not token:
        return None
    return token if token.startswith("Bearer ") else f"Bearer {token}"


def fetch_json(url, session, output, max_retries=3) -> Tuple[bool, int, Optional[str]]:
    """Fetch JSON from URL and save to file.

    Returns: (success: bool, retries_used: int, failure_reason: Optional[str])
    """
    retries_used = 0

    for attempt in range(max_retries + 1):
        try:
            res = session.get(url, timeout=30)

            # Rate limited - retry with backoff
            if res.status_code == 429:
                if attempt < max_retries:
                    retries_used += 1
                    wait = 2 ** attempt
                    log(f"  {Fore.YELLOW}Rate limited, retry in {wait}s...{Style.RESET_ALL}")
                    time.sleep(wait)
                    continue
                else:
                    output.touch()
                    return False, retries_used, "429 Rate Limited"

            # Server error - retry with backoff
            if res.status_code >= 500:
                if attempt < max_retries:
                    retries_used += 1
                    wait = 2 ** attempt
                    log(f"  {Fore.YELLOW}Server error, retry in {wait}s...{Style.RESET_ALL}")
                    time.sleep(wait)
                    continue
                else:
                    output.touch()
                    return False, retries_used, f"{res.status_code} Server Error"

            res.raise_for_status()
            output.write_text(json.dumps(res.json(), indent=4, ensure_ascii=False))
            return True, retries_used, None

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                retries_used += 1
                wait = 2 ** attempt
                log(f"  {Fore.YELLOW}Timeout, retry in {wait}s...{Style.RESET_ALL}")
                time.sleep(wait)
                continue
            output.touch()
            return False, retries_used, "Timeout"

        except requests.exceptions.HTTPError as e:
            output.touch()
            return False, retries_used, f"{e.response.status_code} HTTP Error"

        except requests.exceptions.ConnectionError:
            output.touch()
            return False, retries_used, "Connection Error"

        except Exception as e:
            output.touch()
            return False, retries_used, str(e)[:50]

    return False, retries_used, "Max Retries"


def fetch_json_task(args) -> Tuple[bool, int, str, Optional[str]]:
    """Wrapper for concurrent fetch. Args: (url, session, output_path)

    Returns: (success, retries, endpoint_name, failure_reason)
    """
    url, session, output = args
    # Extract endpoint name from filename (e.g., "findata.json" -> "Findata")
    endpoint_name = output.stem.replace("-", " ").title()
    success, retries, reason = fetch_json(url, session, output)
    return success, retries, endpoint_name, reason


def get_next_dir(path: Path) -> Path:
    """Get next session directory, using cached counter for efficiency."""
    counter_file = path / ".last_session"

    # Try to read from counter file first (fast path)
    if counter_file.exists():
        try:
            next_num = int(counter_file.read_text().strip()) + 1
        except (ValueError, OSError):
            # Fall back to directory scan
            next_num = None
    else:
        next_num = None

    # Fall back to directory scan if counter file doesn't exist or is invalid
    if next_num is None:
        numbers = [int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
        next_num = (max(numbers) + 1) if numbers else 1

    # Create directory and update counter
    new_dir = path / str(next_num)
    new_dir.mkdir(parents=True, exist_ok=True)
    counter_file.write_text(str(next_num))

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


def process_symbol(symbol: str, index: int, total: int, session: requests.Session) -> Tuple[str, bool, int, int]:
    """Process a single symbol - create directory, fetch data, generate analysis, cleanup.

    Returns: (symbol, success, retries, failures)
    """
    log(f"Start Initiate {Fore.CYAN}{symbol}{Style.RESET_ALL} ({index}/{total})")

    base_dir = Path(__file__).parent / "sources" / symbol
    base_dir.mkdir(parents=True, exist_ok=True)
    new_dir = get_next_dir(base_dir)

    json_files = [
        "market-detector.json",
        "price-feed.json",
        "orderbook.json",
        "running-trade.json",
        "today-running-trade.json",
        "findata.json",
    ]

    if not session.headers.get("Authorization"):
        for filename in json_files:
            (new_dir / filename).touch()
        log(f"{Fore.RED}{symbol} Failed{Style.RESET_ALL} ({index}/{total}) - No auth")
        return symbol, False, 0, 6

    # Dynamic date range
    today = date.today()
    from_date = today - timedelta(days=7)
    from_str = from_date.strftime("%Y-%m-%d")
    to_str = today.strftime("%Y-%m-%d")

    # Historical price data - API max is 50 items per page
    price_history_limit = 50
    price_from_date = today - timedelta(days=365)  # Request 1 year range
    price_from_str = price_from_date.strftime("%Y-%m-%d")

    fetch_tasks = [
        (
            f"https://exodus.stockbit.com/marketdetectors/{symbol}"
            f"?transaction_type=TRANSACTION_TYPE_NET&from={from_str}&to={to_str}"
            f"&market_board=MARKET_BOARD_REGULER&investor_type=INVESTOR_TYPE_FOREIGN&limit=25",
            session,
            new_dir / "market-detector.json",
        ),
        (
            f"https://exodus.stockbit.com/company-price-feed/historical/summary/{symbol}"
            f"?period=HS_PERIOD_DAILY&start_date={price_from_str}&end_date={to_str}&limit={price_history_limit}&page=1",
            session,
            new_dir / "price-feed.json",
        ),
        (
            f"https://exodus.stockbit.com/company-price-feed/v2/orderbook/companies/{symbol}",
            session,
            new_dir / "orderbook.json",
        ),
        (
            f"https://exodus.stockbit.com/order-trade/running-trade/chart/{symbol}?period=RT_PERIOD_LAST_1_DAY",
            session,
            new_dir / "running-trade.json",
        ),
        (
            f"https://exodus.stockbit.com/order-trade/running-trade"
            f"?sort=DESC&limit=100&order_by=RUNNING_TRADE_ORDER_BY_TIME&symbols%5B%5D={symbol}",
            session,
            new_dir / "today-running-trade.json",
        ),
        (
            f"https://exodus.stockbit.com/findata-view/foreign-domestic/v1/chart-data/{symbol}"
            f"?market_type=MARKET_TYPE_REGULAR&period=PERIOD_RANGE_1M",
            session,
            new_dir / "findata.json",
        ),
    ]

    # Concurrent fetch
    total_retries = 0
    total_failures = 0
    failed_endpoints = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_json_task, task) for task in fetch_tasks]
        for future in as_completed(futures):
            success, retries, endpoint_name, reason = future.result()
            total_retries += retries
            if not success:
                total_failures += 1
                failed_endpoints.append(f"{endpoint_name}: {reason}")

    # Generate analysis
    analysis_path = generate_analysis(new_dir)
    if analysis_path:
        cleanup_source_files(new_dir)

    # Log result
    if total_failures == 0:
        log(f"{Fore.GREEN}{symbol} Initiated{Style.RESET_ALL} ({index}/{total})")
    else:
        failures_str = " | ".join(failed_endpoints)
        log(f"{Fore.YELLOW}{symbol} Initiated{Style.RESET_ALL} ({index}/{total}) [{total_failures} failed: {failures_str}]")

    return symbol, total_failures == 0, total_retries, total_failures


def main():
    parser = argparse.ArgumentParser(
        description="Fetch stock data for symbols or sectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python initiate.py BBCA                    # Single symbol
    python initiate.py -g banking              # Single group
    python initiate.py -g banking,health       # Multiple groups
    python initiate.py -e banking              # All groups EXCEPT banking
    python initiate.py --list-sectors          # List available sectors
    python initiate.py -g banking -j 5         # Process 5 symbols concurrently
        """
    )
    parser.add_argument("input", nargs="?", help="Symbol (e.g., BBCA)")
    parser.add_argument("-g", "--groups", help="Group name(s), comma-separated")
    parser.add_argument("-e", "--exclude", help="Exclude group(s), comma-separated")
    parser.add_argument("-j", "--jobs", type=int, default=MAX_CONCURRENT_SYMBOLS,
                        help=f"Concurrent symbols (default: {MAX_CONCURRENT_SYMBOLS})")
    parser.add_argument("--list-sectors", action="store_true", help="List sectors")

    args = parser.parse_args()
    groups = load_groups()

    # List sectors mode
    if args.list_sectors:
        print("Available sectors:")
        for sector, syms in groups.items():
            print(f"  {sector}: {len(syms)} symbols")
        total = sum(len(s) for s in groups.values())
        print(f"\nTotal: {len(groups)} sectors, {total} symbols")
        sys.exit(0)

    # Determine symbols to process
    symbols = []
    selected_groups = []

    if args.exclude:
        exclude_list = [g.strip().lower() for g in args.exclude.split(",") if g.strip()]
        invalid = [g for g in exclude_list if g not in groups]
        if invalid:
            parser.error(f"Unknown group(s): {', '.join(invalid)}")
        selected_groups = [g for g in groups.keys() if g not in exclude_list]
        for g in selected_groups:
            symbols.extend(groups[g])
    elif args.groups:
        group_list = [g.strip().lower() for g in args.groups.split(",") if g.strip()]
        invalid = [g for g in group_list if g not in groups]
        if invalid:
            parser.error(f"Unknown group(s): {', '.join(invalid)}")
        selected_groups = group_list
        for g in group_list:
            symbols.extend(groups[g])
    elif args.input:
        symbols = get_symbols_from_input(args.input, groups)
        if args.input.lower() in groups:
            selected_groups = [args.input.lower()]
    else:
        parser.error("Please provide: a symbol, -g <groups>, or -e <exclude_groups>")

    # Deduplicate symbols while preserving order
    symbols = list(dict.fromkeys(symbols))

    # Create session with connection pooling for better performance
    sb_auth = normalize_auth(os.getenv("SB_AUTH"))
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Authorization": sb_auth or "",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Origin": "https://stockbit.com",
        "Referer": "https://stockbit.com/",
    })

    total = len(symbols)

    # Check for market holiday before processing
    is_holiday, last_trading_date = check_market_holiday(session)
    if is_holiday:
        today_str = date.today().strftime("%Y-%m-%d")
        holiday_msg = (
            f"<b>Market Holiday</b>\n"
            f"Today: {today_str}\n"
            f"Last trading date: {last_trading_date or 'unknown'}\n"
            f"Initiate process skipped."
        )
        log(f"{Fore.YELLOW}Market holiday detected. Last trading date: {last_trading_date}{Style.RESET_ALL}")
        log(f"{Fore.YELLOW}Skipping initiate process.{Style.RESET_ALL}")
        send_telegram(holiday_msg)
        sys.exit(0)

    total_retries = 0
    total_failed = 0
    processed = 0

    print(f"\nInitiating {total} symbol(s)...\n")

    # Process symbols concurrently in batches
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Create indexed tasks
        tasks = [(sym, i + 1, total, session) for i, sym in enumerate(symbols)]

        # Submit all
        futures = {executor.submit(process_symbol, *task): task[0] for task in tasks}

        for future in as_completed(futures):
            symbol, success, retries, failures = future.result()
            processed += 1
            total_retries += retries
            if not success:
                total_failed += 1

    # Summary
    print(f"\n{Fore.GREEN}Done{Style.RESET_ALL}")
    print(f"Processed: {Fore.CYAN}{processed}{Style.RESET_ALL}")
    if total_retries > 0:
        print(f"Retries: {Fore.YELLOW}{total_retries}{Style.RESET_ALL}")
    if total_failed > 0:
        print(f"Failed: {Fore.RED}{total_failed}{Style.RESET_ALL}")

    # Telegram
    if selected_groups:
        msg = (
            f"<b>Initiate Complete</b>\n"
            f"Groups: {', '.join(selected_groups)}\n"
            f"Processed: {processed}\n"
            f"Retries: {total_retries}\n"
            f"Failed: {total_failed}"
        )
    else:
        msg = f"<b>Initiate Complete</b>\n{symbols[0] if symbols else 'N/A'}"
    send_telegram(msg)


if __name__ == "__main__":
    main()
