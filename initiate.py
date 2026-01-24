#!/usr/bin/env python3
"""
Stock Market Data Initiator - Production Grade

Fetches and processes stock market data from Stockbit API with concurrent
processing, automatic retry logic, and comprehensive analysis generation.

Usage:
    python initiate.py BBCA                    # Single symbol
    python initiate.py -g banking              # Single group
    python initiate.py -g banking,health       # Multiple groups
    python initiate.py -e banking              # All groups EXCEPT banking
    python initiate.py --list-sectors          # List available sectors
    python initiate.py -g banking -j 5         # Process 5 symbols concurrently
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Final, List, Optional, Tuple, TypeAlias, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Shared utilities
from modules import (
    GROUPS_FILE,
    SOURCES_DIR,
    setup_logging,
    load_groups,
)

load_dotenv()

Number: TypeAlias = Union[int, float]
JsonDict: TypeAlias = Dict[str, Any]

MAX_CONCURRENT_SYMBOLS: Final[int] = 3
DEFAULT_TIMEOUT: Final[int] = 30
MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_FACTOR: Final[float] = 2.0
PRICE_HISTORY_DAYS: Final[int] = 365
PRICE_HISTORY_LIMIT: Final[int] = 50
MARKET_DETECTOR_LIMIT: Final[int] = 25
MARKET_DETECTOR_DAYS: Final[int] = 7
HOLIDAY_CHECK_DAYS: Final[int] = 7

API_BASE_URL: Final[str] = "https://exodus.stockbit.com"

TELEGRAM_BOT_TOKEN: Final[Optional[str]] = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: Final[Optional[str]] = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API_URL: Final[str] = "https://api.telegram.org"

logger = setup_logging("initiate")

_print_lock = threading.Lock()

def log_safe(msg: str, level: int = logging.INFO) -> None:
    with _print_lock:
        logger.log(level, msg)

class InitiateError(Exception):
    pass

class APIError(InitiateError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class RateLimitError(APIError):
    pass

class AuthenticationError(APIError):
    pass

class MarketHolidayError(InitiateError):
    def __init__(self, last_trading_date: Optional[str] = None):
        super().__init__("Market is closed for holiday")
        self.last_trading_date = last_trading_date

class FetchStatus(Enum):
    SUCCESS = auto()
    RATE_LIMITED = auto()
    SERVER_ERROR = auto()
    TIMEOUT = auto()
    HTTP_ERROR = auto()
    CONNECTION_ERROR = auto()
    UNKNOWN_ERROR = auto()

@dataclass(frozen=True)
class FetchResult:
    success: bool
    retries_used: int
    endpoint_name: str
    failure_reason: Optional[str] = None
    status: FetchStatus = FetchStatus.SUCCESS

@dataclass
class SymbolResult:
    symbol: str
    success: bool
    retries: int
    failures: int
    failed_endpoints: List[str] = field(default_factory=list)
    analysis_path: Optional[Path] = None

@dataclass
class BatchResult:
    total: int
    processed: int
    successful: int
    failed: int
    total_retries: int
    results: List[SymbolResult] = field(default_factory=list)

@dataclass(frozen=True)
class APIEndpoint:
    name: str
    url_template: str
    filename: str

@dataclass
class FetchTask:
    url: str
    output_path: Path
    endpoint_name: str

def get_endpoints(symbol: str, from_date: str, to_date: str,
                  price_from_date: str) -> List[APIEndpoint]:
    return [
        APIEndpoint(
            name="Market Detector",
            url_template=(
                f"{API_BASE_URL}/marketdetectors/{symbol}"
                f"?transaction_type=TRANSACTION_TYPE_NET&from={from_date}&to={to_date}"
                "&market_board=MARKET_BOARD_REGULER&investor_type=INVESTOR_TYPE_FOREIGN"
                f"&limit={MARKET_DETECTOR_LIMIT}"
            ),
            filename="market-detector.json",
        ),
        APIEndpoint(
            name="Price Feed",
            url_template=(
                f"{API_BASE_URL}/company-price-feed/historical/summary/{symbol}"
                f"?period=HS_PERIOD_DAILY&start_date={price_from_date}&end_date={to_date}"
                f"&limit={PRICE_HISTORY_LIMIT}&page=1"
            ),
            filename="price-feed.json",
        ),
        APIEndpoint(
            name="Orderbook",
            url_template=f"{API_BASE_URL}/company-price-feed/v2/orderbook/companies/{symbol}",
            filename="orderbook.json",
        ),
        APIEndpoint(
            name="Running Trade",
            url_template=(
                f"{API_BASE_URL}/order-trade/running-trade/chart/{symbol}"
                "?period=RT_PERIOD_LAST_1_DAY"
            ),
            filename="running-trade.json",
        ),
        APIEndpoint(
            name="Today Running Trade",
            url_template=(
                f"{API_BASE_URL}/order-trade/running-trade"
                "?sort=DESC&limit=100&order_by=RUNNING_TRADE_ORDER_BY_TIME"
                f"&symbols%5B%5D={symbol}"
            ),
            filename="today-running-trade.json",
        ),
        APIEndpoint(
            name="Findata",
            url_template=(
                f"{API_BASE_URL}/findata-view/foreign-domestic/v1/chart-data/{symbol}"
                "?market_type=MARKET_TYPE_REGULAR&period=PERIOD_RANGE_1M"
            ),
            filename="findata.json",
        ),
    ]

def create_session(auth_token: Optional[str] = None) -> requests.Session:
    session = requests.Session()

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Origin": "https://stockbit.com",
        "Referer": "https://stockbit.com/",
    })

    if auth_token:
        session.headers["Authorization"] = auth_token

    return session

def normalize_auth_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = token.strip()
    return token if token.startswith("Bearer ") else f"Bearer {token}"

class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self._enabled = bool(self.bot_token and self.chat_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            return False

        url = f"{TELEGRAM_API_URL}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Telegram notification failed: {e}")
            return False

_notifier = TelegramNotifier()

def send_telegram(message: str) -> bool:
    return _notifier.send(message)

def to_int(value: Optional[Union[str, Number]]) -> Optional[int]:
    """
    Parse a value to integer, handling various formats.

    Handles: commas, percentages, numeric types
    """
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    cleaned = str(value).replace(",", "").strip()
    if not cleaned:
        return None

    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]

    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None

def parse_numeric(value: Optional[Union[str, Number]]) -> Optional[float]:
    """
    Parse a numeric string with support for K/M/B suffixes.

    Examples:
        "1.5M" -> 1_500_000.0
        "2.3B" -> 2_300_000_000.0
        "(500)" -> -500.0
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    cleaned = str(value).replace(",", "").strip()
    if not cleaned:
        return None

    # Handle suffixes
    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3}
    multiplier = 1.0

    for suffix, mult in multipliers.items():
        if cleaned.endswith(suffix):
            multiplier = mult
            cleaned = cleaned[:-1]
            break

    # Handle parentheses for negative numbers
    cleaned = cleaned.replace("(", "-").replace(")", "")

    if not cleaned:
        return 0.0

    try:
        return float(cleaned) * multiplier
    except (ValueError, TypeError):
        return None

def load_json_file(path: Path) -> Optional[JsonDict]:
    """
    Load and parse a JSON file.

    Returns None if file doesn't exist or is empty.
    """
    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        return json.loads(content)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to load {path}: {e}")
        return None

def save_json_file(path: Path, data: JsonDict, indent: int = 2, atomic: bool = True) -> bool:
    """
    Save data to a JSON file with optional atomic write.

    Atomic writes prevent partial/corrupted files by writing to a temp file
    first, then renaming (which is atomic on most filesystems).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=indent, ensure_ascii=False)

        if atomic:
            # Write to temp file then atomically rename
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(path)
        else:
            path.write_text(content, encoding="utf-8")

        return True
    except OSError as e:
        logger.error(f"Failed to save {path}: {e}")
        return False


class SessionDirectory:
    def __init__(self, symbol: str, base_dir: Path = SOURCES_DIR):
        self.symbol = symbol.upper()
        self.base_dir = base_dir / self.symbol
        self._counter_file = self.base_dir / ".last_session"

    def get_next_session(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        next_num = self._read_counter()
        if next_num is None:
            next_num = self._scan_directories()
        session_dir = self.base_dir / str(next_num)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._write_counter(next_num)
        return session_dir

    def get_latest_session(self) -> Optional[Path]:
        if not self.base_dir.exists():
            return None
        session_dirs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        if not session_dirs:
            return None
        return max(session_dirs, key=lambda d: int(d.name))

    def _read_counter(self) -> Optional[int]:
        if not self._counter_file.exists():
            return None

        try:
            return int(self._counter_file.read_text().strip()) + 1
        except (ValueError, OSError):
            return None

    def _scan_directories(self) -> int:
        numbers = [
            int(p.name) for p in self.base_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        return (max(numbers) + 1) if numbers else 1

    def _write_counter(self, num: int) -> None:
        try:
            self._counter_file.write_text(str(num))
        except OSError:
            pass

def summarize_depth(data: JsonDict) -> Dict[str, JsonDict]:
    pf = data.get("data", {})
    bid_levels = pf.get("bid", []) or []
    offer_levels = pf.get("offer", []) or []

    def calculate_side_metrics(levels: List[JsonDict]) -> JsonDict:
        volumes = [to_int(level.get("volume")) or 0 for level in levels]
        prices = [to_int(level.get("price")) or 0 for level in levels]

        total_volume = sum(volumes)
        weighted_sum = sum(p * v for p, v in zip(prices, volumes))

        max_volume = max(volumes) if volumes else 0
        max_idx = volumes.index(max_volume) if volumes and max_volume > 0 else -1

        return {
            "total_volume": total_volume,
            "top5_volume": sum(volumes[:5]),
            "top10_volume": sum(volumes[:10]),
            "weighted_price": (weighted_sum / total_volume) if total_volume else None,
            "max_cluster_volume": max_volume,
            "max_cluster_price": prices[max_idx] if max_idx >= 0 else None,
        }

    summary: Dict[str, JsonDict] = {
        "bid": calculate_side_metrics(bid_levels),
        "offer": calculate_side_metrics(offer_levels),
    }

    if bid_levels and offer_levels:
        best_bid = to_int(bid_levels[0].get("price"))
        best_ask = to_int(offer_levels[0].get("price"))

        spread = None
        spread_bps = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            spread_bps = (spread / best_bid * 10000) if best_bid else None

        summary["top_of_book"] = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_vol": to_int(bid_levels[0].get("volume")),
            "ask_vol": to_int(offer_levels[0].get("volume")),
            "spread": spread,
            "spread_bps": spread_bps,
        }

    return summary

def summarize_price_series(data: JsonDict) -> JsonDict:
    price_chart = data.get("data", {}).get("price_chart_data", []) or []
    prices = [p for pt in price_chart if (p := to_int(pt.get("value", {}).get("raw"))) is not None]

    summary: JsonDict = {
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

        returns = [
            curr / prev - 1
            for prev, curr in zip(prices, prices[1:])
            if prev
        ]

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

def summarize_running_trades(data: JsonDict) -> JsonDict:
    trades = data.get("data", {}).get("running_trade", []) or []

    stats: JsonDict = {
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

    buyers: set = set()
    sellers: set = set()
    price_accumulator = 0
    broker_lots: Dict[str, Dict[str, int]] = defaultdict(lambda: {"buy_lot": 0, "sell_lot": 0})

    for trade in trades:
        price = to_int(trade.get("price"))
        lot = to_int(trade.get("lot")) or 0
        action = trade.get("action")
        buyer = trade.get("buyer")
        seller = trade.get("seller")

        stats["lot_total"] += lot

        if action == "buy":
            stats["buy_count"] += 1
            stats["lot_buy"] += lot
        elif action == "sell":
            stats["sell_count"] += 1
            stats["lot_sell"] += lot

        if price is not None:
            price_accumulator += price

        if buyer:
            buyers.add(buyer)
            broker_lots[buyer]["buy_lot"] += lot
        if seller:
            sellers.add(seller)
            broker_lots[seller]["sell_lot"] += lot

    stats["unique_buyers"] = len(buyers)
    stats["unique_sellers"] = len(sellers)
    stats["avg_trade_price"] = price_accumulator / len(trades) if trades else None

    broker_activity = [
        {
            "broker": broker,
            "buy_lot": lots["buy_lot"],
            "sell_lot": lots["sell_lot"],
            "net_lot": lots["buy_lot"] - lots["sell_lot"],
        }
        for broker, lots in broker_lots.items()
    ]
    stats["broker_activity"] = sorted(
        broker_activity, key=lambda x: x["net_lot"], reverse=True
    )[:5]

    return stats

def summarize_broker_charts(data: JsonDict) -> List[JsonDict]:
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
            start_val = parse_numeric(start_raw)
            end_val = parse_numeric(end_raw)

            summary.append({
                "group_type": group_type,
                "broker_code": entry.get("broker_code"),
                "points": len(chart),
                "start_value_raw": start_raw,
                "end_value_raw": end_raw,
                "delta": (end_val - start_val) if (start_val is not None and end_val is not None) else None,
            })

    return summary

def summarize_market_detector(data: JsonDict) -> JsonDict:
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
    foreign_types = {"asing", "foreign", "foreigner"}
    foreign_buyers = [
        b for b in brokers_buy
        if str(b.get("type", "")).lower() in foreign_types
    ]

    total_foreign_buy_volume = sum(
        parse_numeric(b.get("blot") or b.get("blotv")) or 0
        for b in foreign_buyers
    )
    total_foreign_buy_value = sum(
        parse_numeric(b.get("bval") or b.get("bvalv")) or 0
        for b in foreign_buyers
    )

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

def summarize_findata(data: JsonDict) -> JsonDict:
    findata = data.get("data", {})

    summary: JsonDict = {
        "from": findata.get("from"),
        "to": findata.get("to"),
        "last_updated": findata.get("last_updated"),
        "date_range": findata.get("summary", {}).get("date_range"),
    }

    def extract_value_field(field: JsonDict) -> Optional[JsonDict]:
        if not field or not isinstance(field, dict):
            return None
        value_obj = field.get("value", {})
        if isinstance(value_obj, dict):
            return {"raw": value_obj.get("raw"), "formatted": value_obj.get("formatted")}
        return None

    def extract_value_with_percentage(field: JsonDict) -> Optional[JsonDict]:
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
        flow_keys = ["foreign_buy", "foreign_sell", "net_foreign", "domestic_buy", "domestic_sell", "net_domestic"]
        for key in flow_keys:
            if key in summary_section:
                summary["summary"][key] = extract_value_field(summary_section[key])

        volume_summary = summary_section.get("volume", {})
        if volume_summary:
            summary["summary"]["volume"] = {}
            vol_keys = [
                "domestic_buy", "domestic_sell", "net_domestic",
                "foreign_buy", "foreign_sell", "net_foreign_reguler",
                "net_foreign_tunai_nego", "net_foreign_all_market"
            ]
            for key in vol_keys:
                if key in volume_summary:
                    summary["summary"]["volume"][key] = extract_value_field(volume_summary[key])

    for section_name in ["value", "volume", "frequency"]:
        section = findata.get(section_name, {})
        if section:
            summary[section_name] = {"total": extract_value_field(section.get("total", {}))}
            detail_keys = [
                "foreign_buy", "foreign_sell", "domestic_buy",
                "domestic_sell", "foreign_total", "domestic_total"
            ]
            for key in detail_keys:
                if key in section:
                    summary[section_name][key] = extract_value_with_percentage(section[key])

    def extract_participation(section: JsonDict) -> Optional[JsonDict]:
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

def summarize_historical_price(data: JsonDict) -> JsonDict:
    raw_data = data.get("data", {})

    if isinstance(raw_data, dict):
        history = raw_data.get("result", []) or raw_data.get("summary", []) or []
    elif isinstance(raw_data, list):
        history = raw_data
    else:
        history = []

    if not history or not isinstance(history, list):
        return {"count": 0, "latest": None, "history": []}

    latest = history[0] if history else {}
    closes = [c for item in history if (c := parse_numeric(item.get("close"))) is not None]
    volumes = [v for item in history if (v := parse_numeric(item.get("volume"))) is not None]

    summary: JsonDict = {

        "open": parse_numeric(latest.get("open")),
        "high": parse_numeric(latest.get("high")),
        "low": parse_numeric(latest.get("low")),
        "close": parse_numeric(latest.get("close")),
        "last": parse_numeric(latest.get("close")),
        "volume": parse_numeric(latest.get("volume")),
        "value": parse_numeric(latest.get("value")),
        "frequency": parse_numeric(latest.get("frequency")),
        "average": parse_numeric(latest.get("average")),
        "date": latest.get("date"),

        "foreign_buy": parse_numeric(latest.get("foreign_buy")),
        "foreign_sell": parse_numeric(latest.get("foreign_sell")),
        "foreign_net": parse_numeric(latest.get("net_foreign")),

        "change": parse_numeric(latest.get("change")),
        "pct_change": parse_numeric(latest.get("change_percentage")),

        "count": len(history),
        "price_min": min(closes) if closes else None,
        "price_max": max(closes) if closes else None,
        "price_mean": mean(closes) if closes else None,
        "volume_mean": mean(volumes) if volumes else None,
        "history": history,
    }

    if len(closes) >= 2:
        summary["price_change_total"] = closes[0] - closes[-1]
        summary["price_change_total_pct"] = (
            ((closes[0] / closes[-1]) - 1) * 100 if closes[-1] else None
        )

    return summary

class AnalysisBuilder:

    SOURCE_FILES = [
        "price-feed.json",
        "running-trade.json",
        "today-running-trade.json",
        "market-detector.json",
        "orderbook.json",
        "findata.json",
    ]

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.symbol = session_dir.parent.name
        self.session = session_dir.name

    def build(self) -> JsonDict:

        price_feed = load_json_file(self.session_dir / "price-feed.json")
        running_trade = load_json_file(self.session_dir / "running-trade.json")
        today_running_trade = load_json_file(self.session_dir / "today-running-trade.json")
        market_detector = load_json_file(self.session_dir / "market-detector.json")
        orderbook = load_json_file(self.session_dir / "orderbook.json")
        findata = load_json_file(self.session_dir / "findata.json")

        has_orderbook = (
            orderbook is not None and
            orderbook.get("data", {}).get("bid") and
            orderbook.get("data", {}).get("offer")
        )

        sources = [
            ("price-feed.json", price_feed),
            ("running-trade.json", running_trade),
            ("today-running-trade.json", today_running_trade),
            ("market-detector.json", market_detector),
            ("orderbook.json", orderbook if has_orderbook else None),
            ("findata.json", findata),
        ]

        sources_analyzed = [name for name, data in sources if data is not None]
        missing_sources = [name for name, data in sources if data is None]
        available_files = sorted(p.name for p in self.session_dir.glob("*.json"))

        md_summary = summarize_market_detector(market_detector) if market_detector else None
        fd_summary = summarize_findata(findata) if findata else None
        pf_summary = summarize_historical_price(price_feed) if price_feed else None

        return {
            "metadata": {
                "symbol": self.symbol,
                "session": self.session,
                "base_dir": str(self.session_dir),
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
            "depth": summarize_depth(orderbook) if orderbook and has_orderbook else None,
            "price_series": summarize_price_series(running_trade) if running_trade else None,
            "running_trade": summarize_running_trades(today_running_trade) if today_running_trade else None,
            "broker_chart": summarize_broker_charts(running_trade) if running_trade else None,
            "market_detector": md_summary,
            "findata": fd_summary,
            "missing_sources": missing_sources,
        }

    def generate(self) -> Optional[Path]:
        try:
            summary = self.build()
            output_path = self.session_dir / "analyzed.json"

            if save_json_file(output_path, summary):
                return output_path
            return None

        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return None

    def cleanup(self) -> int:
        """
        Delete source files that are no longer needed after analysis.

        Uses a keep-list approach which is more maintainable than a delete-list.
        """
        KEEP_FILES = {"today-running-trade.json", "analyzed.json"}
        deleted_count = 0

        for filepath in self.session_dir.iterdir():
            if filepath.suffix == ".json" and filepath.name not in KEEP_FILES:
                try:
                    filepath.unlink()
                    deleted_count += 1
                except OSError:
                    pass

        return deleted_count

class DataFetcher:

    def __init__(
        self,
        session: requests.Session,
        max_retries: int = MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.session = session
        self.max_retries = max_retries
        self.timeout = timeout

    def fetch(self, task: FetchTask) -> FetchResult:
        """
        Fetch data from URL and save to file.

        Implements exponential backoff retry on transient failures.
        """
        retries_used = 0

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(task.url, timeout=self.timeout)

                if response.status_code == 429:
                    if attempt < self.max_retries:
                        retries_used += 1
                        wait_time = 2 ** attempt
                        log_safe(
                            f"  Rate limited on {task.endpoint_name}, retry in {wait_time}s...",
                            logging.WARNING
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        task.output_path.touch()
                        return FetchResult(
                            success=False,
                            retries_used=retries_used,
                            endpoint_name=task.endpoint_name,
                            failure_reason="429 Rate Limited",
                            status=FetchStatus.RATE_LIMITED,
                        )

                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        retries_used += 1
                        wait_time = 2 ** attempt
                        log_safe(
                            f"  Server error on {task.endpoint_name}, retry in {wait_time}s...",
                            logging.WARNING
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        task.output_path.touch()
                        return FetchResult(
                            success=False,
                            retries_used=retries_used,
                            endpoint_name=task.endpoint_name,
                            failure_reason=f"{response.status_code} Server Error",
                            status=FetchStatus.SERVER_ERROR,
                        )

                response.raise_for_status()

                task.output_path.write_text(
                    json.dumps(response.json(), indent=4, ensure_ascii=False),
                    encoding="utf-8"
                )

                return FetchResult(
                    success=True,
                    retries_used=retries_used,
                    endpoint_name=task.endpoint_name,
                )

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    retries_used += 1
                    wait_time = 2 ** attempt
                    log_safe(
                        f"  Timeout on {task.endpoint_name}, retry in {wait_time}s...",
                        logging.WARNING
                    )
                    time.sleep(wait_time)
                    continue

                task.output_path.touch()
                return FetchResult(
                    success=False,
                    retries_used=retries_used,
                    endpoint_name=task.endpoint_name,
                    failure_reason="Timeout",
                    status=FetchStatus.TIMEOUT,
                )

            except requests.exceptions.HTTPError as e:
                task.output_path.touch()
                return FetchResult(
                    success=False,
                    retries_used=retries_used,
                    endpoint_name=task.endpoint_name,
                    failure_reason=f"{e.response.status_code} HTTP Error",
                    status=FetchStatus.HTTP_ERROR,
                )

            except requests.exceptions.ConnectionError:
                task.output_path.touch()
                return FetchResult(
                    success=False,
                    retries_used=retries_used,
                    endpoint_name=task.endpoint_name,
                    failure_reason="Connection Error",
                    status=FetchStatus.CONNECTION_ERROR,
                )

            except Exception as e:
                task.output_path.touch()
                return FetchResult(
                    success=False,
                    retries_used=retries_used,
                    endpoint_name=task.endpoint_name,
                    failure_reason=str(e)[:50],
                    status=FetchStatus.UNKNOWN_ERROR,
                )

        return FetchResult(
            success=False,
            retries_used=retries_used,
            endpoint_name=task.endpoint_name,
            failure_reason="Max Retries",
        )

class MarketHolidayChecker:

    def __init__(self, session: requests.Session, sample_symbol: str = "BBCA"):
        self.session = session
        self.sample_symbol = sample_symbol

    def check(self) -> Tuple[bool, Optional[str]]:
        """
        Check if today is a market holiday.

        Returns:
            (is_holiday, last_trading_date)
        """
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")
        from_date = today - timedelta(days=HOLIDAY_CHECK_DAYS)
        from_str = from_date.strftime("%Y-%m-%d")

        url = (
            f"{API_BASE_URL}/company-price-feed/historical/summary/{self.sample_symbol}"
            f"?period=HS_PERIOD_DAILY&start_date={from_str}&end_date={today_str}&limit=5&page=1"
        )

        try:
            response = self.session.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            history = data.get("data", {}).get("result", [])
            if not history:
                return True, None

            latest_date = history[0].get("date")
            if not latest_date:
                return True, None

            is_holiday = latest_date != today_str
            return is_holiday, latest_date

        except Exception as e:
            logger.warning(f"Could not check market holiday: {e}")
            return False, None

def has_existing_data(
    symbol: str,
    target_date: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if data for target date already exists for a symbol.

    Args:
        symbol: Stock symbol
        target_date: Date to check (YYYY-MM-DD). Defaults to today.

    Returns:
        (has_data, data_date)
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    session_mgr = SessionDirectory(symbol)
    latest_session = session_mgr.get_latest_session()

    if not latest_session:
        return False, None

    analyzed_file = latest_session / "analyzed.json"
    if not analyzed_file.exists():
        return False, None

    try:
        analysis = load_json_file(analyzed_file)
        if not analysis:
            return False, None

        pf_date = (
            analysis.get("metadata", {})
            .get("time_horizons", {})
            .get("price_feed", {})
            .get("date")
        )

        return pf_date == target_date, pf_date

    except Exception:
        return False, None

class SymbolProcessor:

    def __init__(
        self,
        session: requests.Session,
        concurrent_fetches: int = 6,
    ):
        self.session = session
        self.concurrent_fetches = concurrent_fetches
        self.fetcher = DataFetcher(session)

    def process(
        self,
        symbol: str,
        index: int,
        total: int
    ) -> SymbolResult:
        """
        Process a single symbol.

        Args:
            symbol: Stock symbol
            index: Current index in batch
            total: Total symbols in batch

        Returns:
            SymbolResult with processing outcome
        """
        log_safe(f"Start Initiate {symbol} ({index}/{total})")

        session_mgr = SessionDirectory(symbol)
        session_dir = session_mgr.get_next_session()

        if not self.session.headers.get("Authorization"):
            self._touch_all_files(session_dir)
            log_safe(f"{symbol} Failed ({index}/{total}) - No auth", logging.ERROR)
            return SymbolResult(
                symbol=symbol,
                success=False,
                retries=0,
                failures=6,
                failed_endpoints=["All endpoints: No auth"],
            )

        today = date.today()
        from_date = today - timedelta(days=MARKET_DETECTOR_DAYS)
        price_from_date = today - timedelta(days=PRICE_HISTORY_DAYS)

        from_str = from_date.strftime("%Y-%m-%d")
        to_str = today.strftime("%Y-%m-%d")
        price_from_str = price_from_date.strftime("%Y-%m-%d")

        endpoints = get_endpoints(symbol, from_str, to_str, price_from_str)
        tasks = [
            FetchTask(
                url=ep.url_template,
                output_path=session_dir / ep.filename,
                endpoint_name=ep.name,
            )
            for ep in endpoints
        ]

        total_retries = 0
        total_failures = 0
        failed_endpoints = []

        with ThreadPoolExecutor(max_workers=self.concurrent_fetches) as executor:
            futures = {executor.submit(self.fetcher.fetch, task): task for task in tasks}

            for future in as_completed(futures):
                result = future.result()
                total_retries += result.retries_used

                if not result.success:
                    total_failures += 1
                    failed_endpoints.append(f"{result.endpoint_name}: {result.failure_reason}")

        builder = AnalysisBuilder(session_dir)
        analysis_path = builder.generate()

        if analysis_path:
            builder.cleanup()

        if total_failures == 0:
            log_safe(f"{symbol} Initiated ({index}/{total})")
        else:
            failures_str = " | ".join(failed_endpoints)
            log_safe(
                f"{symbol} Initiated ({index}/{total}) [{total_failures} failed: {failures_str}]",
                logging.WARNING
            )

        return SymbolResult(
            symbol=symbol,
            success=total_failures == 0,
            retries=total_retries,
            failures=total_failures,
            failed_endpoints=failed_endpoints,
            analysis_path=analysis_path,
        )

    def _touch_all_files(self, session_dir: Path) -> None:
        files = [
            "market-detector.json",
            "price-feed.json",
            "orderbook.json",
            "running-trade.json",
            "today-running-trade.json",
            "findata.json",
        ]
        for filename in files:
            (session_dir / filename).touch()

class BatchProcessor:

    def __init__(
        self,
        session: requests.Session,
        max_concurrent_symbols: int = MAX_CONCURRENT_SYMBOLS,
    ):
        self.session = session
        self.max_concurrent_symbols = max_concurrent_symbols
        self.processor = SymbolProcessor(session)

    def process(self, symbols: List[str]) -> BatchResult:
        """
        Process a batch of symbols concurrently.

        Args:
            symbols: List of stock symbols

        Returns:
            BatchResult with overall statistics
        """
        total = len(symbols)
        results: List[SymbolResult] = []
        total_retries = 0
        total_failed = 0

        print(f"\nInitiating {total} symbol(s)...\n")

        with ThreadPoolExecutor(max_workers=self.max_concurrent_symbols) as executor:
            futures: Dict[Future, str] = {}

            for i, symbol in enumerate(symbols, 1):
                future = executor.submit(self.processor.process, symbol, i, total)
                futures[future] = symbol

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                total_retries += result.retries
                if not result.success:
                    total_failed += 1

        return BatchResult(
            total=total,
            processed=len(results),
            successful=total - total_failed,
            failed=total_failed,
            total_retries=total_retries,
            results=results,
        )

def create_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=MAX_CONCURRENT_SYMBOLS,
        help=f"Concurrent symbols (default: {MAX_CONCURRENT_SYMBOLS})"
    )
    parser.add_argument("--list-sectors", action="store_true", help="List sectors")
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Allow re-initiation even if today's data already exists"
    )
    parser.add_argument(
        "--skip-holiday",
        action="store_true",
        help="Skip market holiday check (for testing)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser

def resolve_symbols(args: argparse.Namespace, groups: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Resolve symbols from command-line arguments.

    Returns:
        (symbols, selected_groups)
    """
    symbols = []
    selected_groups = []

    if args.exclude:
        exclude_list = [g.strip().lower() for g in args.exclude.split(",") if g.strip()]
        invalid = [g for g in exclude_list if g not in groups]
        if invalid:
            raise ValueError(f"Unknown group(s): {', '.join(invalid)}")

        selected_groups = [g for g in groups.keys() if g not in exclude_list]
        for g in selected_groups:
            symbols.extend(groups[g])

    elif args.groups:
        group_list = [g.strip().lower() for g in args.groups.split(",") if g.strip()]
        invalid = [g for g in group_list if g not in groups]
        if invalid:
            raise ValueError(f"Unknown group(s): {', '.join(invalid)}")

        selected_groups = group_list
        for g in group_list:
            symbols.extend(groups[g])

    elif args.input:
        input_lower = args.input.lower()
        if input_lower in groups:
            symbols = groups[input_lower]
            selected_groups = [input_lower]
        else:
            symbols = [args.input.upper()]
    else:
        raise ValueError("Please provide: a symbol, -g <groups>, or -e <exclude_groups>")

    symbols = list(dict.fromkeys(symbols))

    return symbols, selected_groups

def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    groups = load_groups()

    if args.list_sectors:
        print("Available sectors:")
        for sector, syms in groups.items():
            print(f"  {sector}: {len(syms)} symbols")
        total = sum(len(s) for s in groups.values())
        print(f"\nTotal: {len(groups)} sectors, {total} symbols")
        return 0

    try:
        symbols, selected_groups = resolve_symbols(args, groups)
    except ValueError as e:
        parser.error(str(e))
        return 1

    sb_auth = normalize_auth_token(os.getenv("SB_AUTH"))
    session = create_session(sb_auth)

    holiday_checker = MarketHolidayChecker(session)
    is_holiday, last_trading_date = holiday_checker.check()

    if is_holiday and not args.skip_holiday:
        today_str = date.today().strftime("%Y-%m-%d")
        holiday_msg = (
            "<b>Market Holiday</b>\n"
            f"Today: {today_str}\n"
            f"Last trading date: {last_trading_date or 'unknown'}\n"
            "Initiate process skipped."
        )
        logger.warning(f"Market holiday detected. Last trading date: {last_trading_date}")
        logger.warning("Skipping initiate process.")
        send_telegram(holiday_msg)
        return 0

    if is_holiday and args.skip_holiday:
        logger.warning("Market holiday detected but skipping check (--skip-holiday)")
        logger.warning(f"Using last trading date: {last_trading_date}\n")

    target_date = last_trading_date if is_holiday else date.today().strftime("%Y-%m-%d")

    if not args.allow_duplicate:
        skipped_symbols = []
        symbols_to_process = []

        for symbol in symbols:
            has_data, data_date = has_existing_data(symbol, target_date)
            if has_data:
                skipped_symbols.append((symbol, data_date))
            else:
                symbols_to_process.append(symbol)

        if skipped_symbols:
            logger.warning(f"Skipping {len(skipped_symbols)} symbol(s) with data for {target_date}:")
            for sym, dt in skipped_symbols:
                print(f"  {sym} (data from {dt})")
            print("Use --allow-duplicate to re-initiate.\n")

        symbols = symbols_to_process

        if not symbols:
            logger.info(f"All symbols already have data for {target_date}. Nothing to do.")
            return 0

    batch_processor = BatchProcessor(session, args.jobs)
    result = batch_processor.process(symbols)

    print("\nDone")
    print(f"Processed: {result.processed}")
    if result.total_retries > 0:
        print(f"Retries: {result.total_retries}")
    if result.failed > 0:
        print(f"Failed: {result.failed}")

    if selected_groups:
        msg = (
            "<b>Initiate Complete</b>\n"
            f"Groups: {', '.join(selected_groups)}\n"
            f"Processed: {result.processed}\n"
            f"Retries: {result.total_retries}\n"
            f"Failed: {result.failed}"
        )
    else:
        msg = f"<b>Initiate Complete</b>\n{symbols[0] if symbols else 'N/A'}"

    send_telegram(msg)

    return 0 if result.failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
