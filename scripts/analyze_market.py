#!/usr/bin/env python3
"""
Reusable analysis pipeline for market JSON datasets.

Usage
-----
python scripts/analyze_market.py ADMR 1

This script reads all JSON sources found under `sources/{symbol}/{session}/`
and produces a condensed analytics JSON file that other tooling (or prompts)
can consume to generate narrative reports.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Tuple, Union

Number = Union[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate market data sources into a reusable analytics blob."
    )
    parser.add_argument("symbol", help="Trading symbol (e.g., ADMR)")
    parser.add_argument("session", help="Session or batch identifier (e.g., 1 or 20251111)")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="Project root directory (defaults to repository root).",
    )
    parser.add_argument(
        "--out",
        dest="output",
        default=None,
        help="Optional explicit path for the output JSON file.",
    )
    return parser.parse_args()


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


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()
        if not content:
            return None
        return json.loads(content)


def summarize_price_feed(data: dict) -> Dict[str, Union[int, float, dict]]:
    pf = data.get("data", {})
    return {
        "open": pf.get("open"),
        "high": pf.get("high"),
        "low": pf.get("low"),
        "close": pf.get("close"),
        "last": pf.get("lastprice"),
        "prev_close": pf.get("previous"),
        "change": pf.get("change"),
        "pct_change": pf.get("percentage_change"),
        "average": pf.get("average"),
        "value": pf.get("value"),
        "volume": pf.get("volume"),
        "foreign_buy": pf.get("fbuy"),
        "foreign_sell": pf.get("fsell"),
        "foreign_net": pf.get("fnet"),
        "total_bid_offer": pf.get("total_bid_offer"),
        "bid_levels": len(pf.get("bid", [])),
        "offer_levels": len(pf.get("offer", [])),
    }


def summarize_depth(data: dict) -> Dict[str, dict]:
    pf = data.get("data", {})
    bid_levels = pf.get("bid", []) or []
    offer_levels = pf.get("offer", []) or []

    def side_metrics(levels: Iterable[dict]) -> dict:
        volumes = [to_int(level.get("volume")) or 0 for level in levels]
        prices = [to_int(level.get("price")) or 0 for level in levels]
        total = sum(volumes)
        weighted_sum = sum(price * vol for price, vol in zip(prices, volumes))
        side = {
            "total_volume": total,
            "top5_volume": sum(volumes[:5]),
            "top10_volume": sum(volumes[:10]),
            "weighted_price": (weighted_sum / total) if total else None,
            "max_cluster_volume": max(volumes) if volumes else 0,
            "max_cluster_price": prices[volumes.index(max(volumes))] if volumes else None,
        }
        return side

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
            summary.update(
                {
                    "mean_return": mean(returns),
                    "std_return": pstdev(returns) if len(returns) > 1 else 0.0,
                    "max_intraday_jump": max(returns),
                    "min_intraday_jump": min(returns),
                }
            )

        indices = list(range(len(prices)))
        mean_x = mean(indices)
        mean_y = mean(prices)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, prices))
        denominator = sum((x - mean_x) ** 2 for x in indices)
        slope = numerator / denominator if denominator else 0.0
        summary["slope_per_interval"] = slope
        summary["slope_per_hour_equiv"] = slope * 60
        summary["last_hour_change"] = (
            prices[-1] - prices[-60] if len(prices) >= 60 else None
        )
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
        broker_activity.append(
            {
                "broker": broker,
                "buy_lot": lots["buy_lot"],
                "sell_lot": lots["sell_lot"],
                "net_lot": lots["buy_lot"] - lots["sell_lot"],
            }
        )
    stats["broker_activity"] = sorted(
        broker_activity,
        key=lambda entry: entry["net_lot"],
        reverse=True,
    )[:5]

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
            summary.append(
                {
                    "group_type": group_type,
                    "broker_code": entry.get("broker_code"),
                    "points": len(chart),
                    "start_value_raw": start_raw,
                    "end_value_raw": end_raw,
                    "delta": (end_val - start_val) if (start_val is not None and end_val is not None) else None,
                }
            )
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

    # Focus specifically on foreign (e.g. "Asing") buying activity
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

    foreign_focus = {
        "total_foreign_buy_volume": total_foreign_buy_volume or None,
        "total_foreign_buy_value": total_foreign_buy_value or None,
        "top_foreign_buyers": foreign_buyers[:5],
    }

    summary = {
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
        "foreign_focus": foreign_focus,
    }
    return summary


def summarize_findata(data: dict) -> dict:
    """
    Summarize financial trading flow data from findata.json.
    Extracts foreign vs domestic trading activity (value, volume, frequency).
    """
    findata = data.get("data", {})
    summary: dict = {
        "from": findata.get("from"),
        "to": findata.get("to"),
        "last_updated": findata.get("last_updated"),
        "date_range": findata.get("summary", {}).get("date_range"),
    }
    
    # Helper functions
    def extract_value_field(field: dict) -> Optional[dict]:
        if not field or not isinstance(field, dict):
            return None
        value_obj = field.get("value", {})
        if isinstance(value_obj, dict):
            return {
                "raw": value_obj.get("raw"),
                "formatted": value_obj.get("formatted"),
            }
        return None
    
    def extract_value_with_percentage(field: dict) -> Optional[dict]:
        if not field or not isinstance(field, dict):
            return None
        value_obj = field.get("value", {})
        percentage_obj = field.get("percentage", {})
        result = {}
        if isinstance(value_obj, dict):
            result["value"] = {
                "raw": value_obj.get("raw"),
                "formatted": value_obj.get("formatted"),
            }
        if isinstance(percentage_obj, dict):
            result["percentage"] = {
                "raw": percentage_obj.get("raw"),
                "formatted": percentage_obj.get("formatted"),
            }
        return result if result else None
    
    # Extract summary section
    summary_section = findata.get("summary", {})
    if summary_section:
        summary["summary"] = {}
        
        for key in ["foreign_buy", "foreign_sell", "net_foreign", "domestic_buy", "domestic_sell", "net_domestic"]:
            if key in summary_section:
                summary["summary"][key] = extract_value_field(summary_section[key])
        
        # Extract volume summary
        volume_summary = summary_section.get("volume", {})
        if volume_summary:
            summary["summary"]["volume"] = {}
            for key in ["domestic_buy", "domestic_sell", "net_domestic", "foreign_buy", "foreign_sell", 
                       "net_foreign_reguler", "net_foreign_tunai_nego", "net_foreign_all_market"]:
                if key in volume_summary:
                    summary["summary"]["volume"][key] = extract_value_field(volume_summary[key])
    
    # Extract value section (IDR breakdowns)
    value_section = findata.get("value", {})
    if value_section:
        summary["value"] = {
            "total": extract_value_field(value_section.get("total", {})),
        }
        
        for key in ["foreign_buy", "foreign_sell", "domestic_buy", "domestic_sell", "foreign_total", "domestic_total"]:
            if key in value_section:
                summary["value"][key] = extract_value_with_percentage(value_section[key])
    
    # Extract volume section (shares breakdowns)
    volume_section = findata.get("volume", {})
    if volume_section:
        summary["volume"] = {
            "total": extract_value_field(volume_section.get("total", {})),
        }
        
        for key in ["foreign_buy", "foreign_sell", "domestic_buy", "domestic_sell", "foreign_total", "domestic_total"]:
            if key in volume_section:
                summary["volume"][key] = extract_value_with_percentage(volume_section[key])
    
    # Extract frequency section (trade count breakdowns)
    frequency_section = findata.get("frequency", {})
    if frequency_section:
        summary["frequency"] = {
            "total": extract_value_field(frequency_section.get("total", {})),
        }
        
        for key in ["foreign_buy", "foreign_sell", "domestic_buy", "domestic_sell", "foreign_total", "domestic_total"]:
            if key in frequency_section:
                summary["frequency"][key] = extract_value_with_percentage(frequency_section[key])
    
    # High-level participation snapshots to make the report easier to write
    def extract_participation(section: dict) -> Optional[dict]:
        if not section:
            return None
        foreign_total = section.get("foreign_total", {})
        domestic_total = section.get("domestic_total", {})
        ft_pct = (foreign_total.get("percentage") or {}).get("raw")
        dt_pct = (domestic_total.get("percentage") or {}).get("raw")
        if ft_pct is None and dt_pct is None:
            return None
        return {
            "foreign_pct": ft_pct,
            "domestic_pct": dt_pct,
        }

    value_participation = extract_participation(summary.get("value", {}))
    if value_participation:
        summary["value_participation"] = value_participation

    volume_participation = extract_participation(summary.get("volume", {}))
    if volume_participation:
        summary["volume_participation"] = volume_participation

    frequency_participation = extract_participation(summary.get("frequency", {}))
    if frequency_participation:
        summary["frequency_participation"] = frequency_participation

    return summary


def build_summary(base_dir: Path) -> dict:
    price_feed = load_json(base_dir / "price-feed.json")
    running_trade = load_json(base_dir / "running-trade.json")
    today_running_trade = load_json(base_dir / "today-running-trade.json")
    market_detector = load_json(base_dir / "market-detector.json")
    orderbook = load_json(base_dir / "orderbook.json")
    findata = load_json(base_dir / "findata.json")

    symbol = base_dir.parent.name
    session = base_dir.name

    orderbook_source: Optional[str] = None
    if orderbook:
        orderbook_source = "orderbook.json"
    elif price_feed and price_feed.get("data", {}).get("bid") and price_feed.get("data", {}).get("offer"):
        orderbook_source = "price-feed.json"

    available_files = sorted(p.name for p in base_dir.glob("*.json"))
    sources_analyzed = [
        name
        for name, data in [
            ("price-feed.json", price_feed),
            ("running-trade.json", running_trade),
            ("today-running-trade.json", today_running_trade),
            ("market-detector.json", market_detector),
            ("orderbook.json", orderbook if orderbook_source == "orderbook.json" else price_feed if orderbook_source == "price-feed.json" else None),
            ("findata.json", findata),
        ]
        if data is not None
    ]
    missing_sources = [
        name
        for name, data in [
            ("price-feed.json", price_feed),
            ("running-trade.json", running_trade),
            ("today-running-trade.json", today_running_trade),
            ("market-detector.json", market_detector),
            (
                "orderbook.json",
                orderbook if orderbook_source == "orderbook.json" else price_feed if orderbook_source == "price-feed.json" else None,
            ),
            ("findata.json", findata),
        ]
        if data is None
    ]

    md_summary = summarize_market_detector(market_detector) if market_detector else None
    fd_summary = summarize_findata(findata) if findata else None

    summary = {
        "metadata": {
            "symbol": symbol,
            "session": session,
            "base_dir": str(base_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "available_files": available_files,
            "sources_analyzed": sources_analyzed,
            "orderbook_source": orderbook_source,
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
            },
        },
        "price_feed": summarize_price_feed(price_feed) if price_feed else None,
        "depth": summarize_depth(price_feed if orderbook_source == "price-feed.json" else orderbook) if price_feed or orderbook else None,
        "price_series": summarize_price_series(running_trade) if running_trade else None,
        "running_trade": summarize_running_trades(today_running_trade) if today_running_trade else None,
        "broker_chart": summarize_broker_charts(running_trade) if running_trade else None,
        "market_detector": md_summary,
        "findata": fd_summary,
        "missing_sources": missing_sources,
    }
    return summary


def main() -> None:
    args = parse_args()
    base_dir = args.root / "sources" / args.symbol / args.session
    if not base_dir.exists():
        raise SystemExit(f"Data directory not found: {base_dir}")

    summary = build_summary(base_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = base_dir / f"analysis-data-{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote structured analytics to {output_path}")


if __name__ == "__main__":
    main()

