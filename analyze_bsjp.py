#!/usr/bin/env python3
"""
Bulk Market Data Analysis (Screener-Based)
Implements the /analyze-bsjp command workflow.

This script:
1. ALWAYS fetches fresh screener data first (mandatory)
2. Extracts symbols from screener.json
3. For each symbol: initializes data and generates analytics
4. Loads all analytics JSON files
5. Generates comprehensive bulk analysis report focused on 2-3 day short-term trading
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, Optional[str]]:
    """Run a shell command and return success status and output."""
    try:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print(result.stdout)
            return True, result.stdout
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False, result.stderr
    except Exception as e:
        print(f"✗ Exception running {description}: {e}")
        return False, str(e)


def fetch_screener_data(template_id: Optional[int] = None) -> bool:
    """Step 0: ALWAYS fetch fresh screener data first (MANDATORY)."""
    cmd = ["python", "screener.py"]
    if template_id:
        cmd.append(str(template_id))
    
    success, _ = run_command(cmd, "Fetching fresh screener data")
    return success


def extract_symbols_from_screener(screener_path: Path) -> List[Dict]:
    """Step 1: Extract symbols and company info from screener.json."""
    if not screener_path.exists():
        print(f"✗ Screener file not found: {screener_path}")
        return []
    
    with screener_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    symbols_data = []
    calcs = data.get("data", {}).get("calcs", [])
    
    for calc in calcs:
        company = calc.get("company", {})
        symbol = company.get("symbol")
        name = company.get("name", "")
        
        if symbol:
            # Extract screener metrics
            results = calc.get("results", [])
            metrics = {}
            for result in results:
                item = result.get("item", "")
                raw = result.get("raw", "")
                metrics[item] = raw
            
            symbols_data.append({
                "symbol": symbol,
                "name": name,
                "metrics": metrics
            })
    
    print(f"\n✓ Extracted {len(symbols_data)} symbols from screener")
    return symbols_data


def get_latest_session(symbol_dir: Path) -> Optional[int]:
    """Get the latest session number for a symbol."""
    if not symbol_dir.exists():
        return None
    
    sessions = [int(p.name) for p in symbol_dir.iterdir() 
                if p.is_dir() and p.name.isdigit()]
    return max(sessions) if sessions else None


def get_latest_analysis_file(symbol_dir: Path, session: int) -> Optional[Path]:
    """Get the most recent analysis-data-*.json file for a symbol session."""
    session_dir = symbol_dir / str(session)
    if not session_dir.exists():
        return None
    
    analysis_files = list(session_dir.glob("analysis-data-*.json"))
    if not analysis_files:
        return None
    
    # Sort by modification time, return most recent
    analysis_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return analysis_files[0]


def initialize_symbol_data(symbol: str) -> Tuple[bool, Optional[int]]:
    """Step 2: Initialize data for a symbol (creates new session)."""
    cmd = ["python", "initiate.py", symbol]
    success, _ = run_command(cmd, f"Initializing data for {symbol}")
    
    if not success:
        return False, None
    
    # Get the new session number
    symbol_dir = Path("sources") / symbol
    session = get_latest_session(symbol_dir)
    return True, session


def generate_analytics(symbol: str, session: int) -> bool:
    """Step 3: Generate analytics for a symbol session."""
    cmd = ["python", "scripts/analyze_market.py", symbol, str(session)]
    success, _ = run_command(cmd, f"Generating analytics for {symbol} session {session}")
    return success


def load_analytics_data(symbol: str, session: int) -> Optional[Dict]:
    """Step 4: Load the most recent analysis-data-*.json file."""
    symbol_dir = Path("sources") / symbol
    analysis_file = get_latest_analysis_file(symbol_dir, session)
    
    if not analysis_file:
        print(f"✗ No analysis file found for {symbol} session {session}")
        return None
    
    try:
        with analysis_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✓ Loaded analytics for {symbol} from {analysis_file.name}")
        return data
    except Exception as e:
        print(f"✗ Error loading analytics for {symbol}: {e}")
        return None


def classify_trend(slope: Optional[float]) -> str:
    """Classify short-term trend from slope_per_hour_equiv."""
    if slope is None:
        return "Unknown"
    if slope > 1.0:
        return "Uptrend"
    elif slope < -1.0:
        return "Downtrend"
    else:
        return "Sideways"


def classify_volatility(std_return: Optional[float]) -> str:
    """Classify volatility from std_return."""
    if std_return is None:
        return "Unknown"
    if std_return < 0.002:
        return "Low"
    elif std_return < 0.004:
        return "Medium"
    else:
        return "High"


def classify_liquidity(depth: Optional[Dict]) -> str:
    """Classify liquidity from depth metrics."""
    if not depth:
        return "Unknown"
    
    bid_vol = depth.get("bid", {}).get("total_volume", 0) or 0
    offer_vol = depth.get("offer", {}).get("total_volume", 0) or 0
    total_vol = bid_vol + offer_vol
    
    if total_vol > 50_000_000:  # > 50M shares
        return "Very High"
    elif total_vol > 20_000_000:  # > 20M shares
        return "High"
    elif total_vol > 5_000_000:  # > 5M shares
        return "Medium"
    else:
        return "Low"


def get_foreign_flow_status(findata: Optional[Dict], price_feed: Optional[Dict] = None) -> Tuple[str, Optional[float]]:
    """Get foreign flow status and value from findata, with fallback to price_feed."""
    # Try findata first (more comprehensive)
    if findata:
        summary = findata.get("summary", {})
        net_foreign = summary.get("net_foreign", {})
        if net_foreign:
            raw_value = net_foreign.get("value", {}).get("raw")
            if raw_value is not None:
                try:
                    value = float(raw_value)
                    if value > 1_000_000_000:  # > 1B
                        return "Positive", value
                    elif value < -1_000_000_000:  # < -1B
                        return "Negative", value
                    else:
                        return "Neutral", value
                except (ValueError, TypeError):
                    pass
    
    # Fallback to price_feed foreign_net
    if price_feed:
        foreign_net = price_feed.get("foreign_net")
        if foreign_net is not None:
            try:
                value = float(foreign_net)
                if value > 1_000_000_000:  # > 1B
                    return "Positive", value
                elif value < -1_000_000_000:  # < -1B
                    return "Negative", value
                else:
                    return "Neutral", value
            except (ValueError, TypeError):
                pass
    
    return "Unknown", None


def get_bandar_signal(market_detector: Optional[Dict]) -> str:
    """Get bandar signal from market_detector."""
    if not market_detector:
        return "Unknown"
    
    bandar = market_detector.get("bandar", {})
    top5 = bandar.get("top5")
    
    if top5 is None:
        # Try to use dominant_side as fallback
        dominant_side = bandar.get("dominant_side")
        if dominant_side == "buyers":
            return "Accumulation"
        elif dominant_side == "sellers":
            return "Distribution"
        return "Unknown"
    
    # Try to determine from top5 value
    # Positive values typically indicate accumulation, negative distribution
    if isinstance(top5, (int, float)):
        if top5 > 20:
            return "Accumulation"
        elif top5 < -20:
            return "Distribution"
        else:
            return "Neutral"
    
    return "Unknown"


def get_outlook(trend: str, foreign_flow: str, bandar: str, volatility: str) -> str:
    """Determine 2-3 day outlook based on multiple signals."""
    bullish_signals = 0
    bearish_signals = 0
    
    if trend == "Uptrend":
        bullish_signals += 1
    elif trend == "Downtrend":
        bearish_signals += 1
    
    if foreign_flow == "Positive":
        bullish_signals += 1
    elif foreign_flow == "Negative":
        bearish_signals += 1
    
    if bandar == "Accumulation":
        bullish_signals += 1
    elif bandar == "Distribution":
        bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        return "Bullish"
    elif bearish_signals > bullish_signals:
        return "Bearish"
    else:
        return "Neutral"


def format_idr(value: Optional[float]) -> str:
    """Format IDR value."""
    if value is None:
        return "N/A"
    
    if abs(value) >= 1_000_000_000_000:  # Trillion
        return f"{value / 1_000_000_000_000:.2f}T"
    elif abs(value) >= 1_000_000_000:  # Billion
        return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:  # Million
        return f"{value / 1_000_000:.2f}M"
    else:
        return f"{value:,.0f}"


def generate_bulk_report(
    symbols_data: List[Dict],
    analytics_data: Dict[str, Dict],
    screener_path: Path,
    output_path: Path
) -> None:
    """Step 5: Generate comprehensive bulk analysis report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load screener metadata
    with screener_path.open("r", encoding="utf-8") as f:
        screener_json = json.load(f)
    
    screener_name = screener_json.get("data", {}).get("name", "Unknown")
    screener_id = screener_json.get("data", {}).get("id", "Unknown")
    
    # Build report
    report_lines = []
    
    # Header
    report_lines.append("# Bulk Market Analysis Report (Short-Term: 2-3 Trading Days)")
    report_lines.append(f"**Analysis Date**: {timestamp}")
    report_lines.append(f"**Screener Source**: screener.json")
    report_lines.append(f"**Total Symbols Analyzed**: {len(analytics_data)}")
    report_lines.append(f"**Analysis Timeframe**: 2-3 Trading Days")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    # Calculate aggregate metrics
    total_symbols = len(analytics_data)
    positive_performers = 0
    negative_performers = 0
    uptrend_count = 0
    downtrend_count = 0
    
    top_performers = []
    opportunities = []
    
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        if symbol not in analytics_data:
            continue
        
        analytics = analytics_data[symbol]
        price_feed = analytics.get("price_feed", {})
        price_series = analytics.get("price_series", {})
        
        pct_change = price_feed.get("pct_change")
        if pct_change:
            if pct_change > 0:
                positive_performers += 1
            else:
                negative_performers += 1
        
        slope = price_series.get("slope_per_hour_equiv")
        trend = classify_trend(slope)
        if trend == "Uptrend":
            uptrend_count += 1
        elif trend == "Downtrend":
            downtrend_count += 1
        
        # Collect top performers
        if pct_change:
            top_performers.append((symbol, pct_change, price_feed.get("last")))
        
        # Collect opportunities (positive trend + positive foreign flow)
        findata = analytics.get("findata")
        price_feed = analytics.get("price_feed", {})
        foreign_status, foreign_value = get_foreign_flow_status(findata, price_feed)
        if trend == "Uptrend" and foreign_status == "Positive":
            opportunities.append((symbol, trend, foreign_value))
    
    top_performers.sort(key=lambda x: x[1] if x[1] else 0, reverse=True)
    opportunities.sort(key=lambda x: x[2] if x[2] else 0, reverse=True)
    
    # Write executive summary
    report_lines.append(f"This bulk analysis evaluates {total_symbols} stocks from the \"{screener_name}\" screener template (ID: {screener_id}) for short-term trading opportunities over the next 2-3 trading days.")
    report_lines.append("")
    
    sentiment = "Bullish" if positive_performers > negative_performers else "Bearish" if negative_performers > positive_performers else "Mixed"
    report_lines.append(f"**Overall Market Sentiment**: {sentiment.lower()}. {positive_performers} symbols show positive performance today, while {negative_performers} show negative performance.")
    report_lines.append("")
    
    if top_performers:
        top3 = top_performers[:3]
        top_str = ", ".join([f"{s} (+{p:.2f}%)" for s, p, _ in top3 if p])
        report_lines.append(f"**Top Performers Today**: {top_str}")
        report_lines.append("")
    
    if opportunities:
        opp_str = ", ".join([f"{s}" for s, _, _ in opportunities[:5]])
        report_lines.append(f"**Short-Term Opportunities**: {opp_str}")
        report_lines.append("")
    
    # Screener Overview
    report_lines.append("## Screener Overview")
    report_lines.append("")
    report_lines.append(f"**Screener Name**: {screener_name}")
    report_lines.append(f"**Screener ID**: {screener_id}")
    report_lines.append(f"**Total Symbols**: {len(symbols_data)}")
    report_lines.append(f"**Successfully Analyzed**: {len(analytics_data)}")
    report_lines.append(f"**Symbols with Data Issues**: {len(symbols_data) - len(analytics_data)}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Symbol Performance Summary Table
    report_lines.append("## Symbol Performance Summary Table")
    report_lines.append("")
    report_lines.append("| Symbol | Company Name | Today % | Price | Short-Term Trend | Volatility | Liquidity Score | Foreign Flow | Bandar Signal | 2-3 Day Outlook |")
    report_lines.append("|--------|--------------|---------|-------|------------------|------------|-----------------|--------------|---------------|-----------------|")
    
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        name = symbol_info["name"]
        
        if symbol not in analytics_data:
            report_lines.append(f"| {symbol} | {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        
        analytics = analytics_data[symbol]
        price_feed = analytics.get("price_feed", {})
        price_series = analytics.get("price_series", {})
        depth = analytics.get("depth", {})
        findata = analytics.get("findata")
        market_detector = analytics.get("market_detector")
        
        # Extract metrics
        pct_change = price_feed.get("pct_change")
        today_pct = f"{pct_change:+.2f}%" if pct_change is not None else "N/A"
        price = price_feed.get("last") or price_feed.get("close")
        price_str = f"{price:,}" if price else "N/A"
        
        slope = price_series.get("slope_per_hour_equiv")
        trend = classify_trend(slope)
        
        std_return = price_series.get("std_return")
        volatility = classify_volatility(std_return)
        
        liquidity = classify_liquidity(depth)
        
        foreign_status, foreign_value = get_foreign_flow_status(findata, price_feed)
        foreign_str = foreign_status
        if foreign_value:
            foreign_str += f" ({format_idr(foreign_value)})"
        
        bandar = get_bandar_signal(market_detector)
        
        outlook = get_outlook(trend, foreign_status, bandar, volatility)
        
        report_lines.append(f"| {symbol} | {name} | {today_pct} | {price_str} | {trend} | {volatility} | {liquidity} | {foreign_str} | {bandar} | {outlook} |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Top Short-Term Opportunities
    report_lines.append("## Top Short-Term Opportunities (2-3 Days)")
    report_lines.append("")
    
    # Rank opportunities
    ranked_opportunities = []
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        if symbol not in analytics_data:
            continue
        
        analytics = analytics_data[symbol]
        price_feed = analytics.get("price_feed", {})
        price_series = analytics.get("price_series", {})
        depth = analytics.get("depth", {})
        findata = analytics.get("findata")
        market_detector = analytics.get("market_detector")
        
        slope = price_series.get("slope_per_hour_equiv")
        trend = classify_trend(slope)
        price_feed = analytics.get("price_feed", {})
        foreign_status, foreign_value = get_foreign_flow_status(findata, price_feed)
        bandar = get_bandar_signal(market_detector)
        outlook = get_outlook(trend, foreign_status, bandar, classify_volatility(price_series.get("std_return")))
        
        # Score opportunity (higher is better)
        score = 0
        if trend == "Uptrend":
            score += 3
        elif trend == "Sideways":
            score += 1
        
        if foreign_status == "Positive":
            score += 2
        elif foreign_status == "Neutral":
            score += 1
        
        if bandar == "Accumulation":
            score += 2
        elif bandar == "Neutral":
            score += 1
        
        if outlook == "Bullish":
            score += 2
        
        pct_change = price_feed.get("pct_change", 0) or 0
        score += max(0, min(2, pct_change / 2))  # Bonus for positive performance
        
        ranked_opportunities.append((symbol, symbol_info["name"], score, analytics))
    
    ranked_opportunities.sort(key=lambda x: x[2], reverse=True)
    top_opportunities = ranked_opportunities[:5]
    
    for idx, (symbol, name, score, analytics) in enumerate(top_opportunities, 1):
        price_feed = analytics.get("price_feed", {})
        price_series = analytics.get("price_series", {})
        depth = analytics.get("depth", {})
        findata = analytics.get("findata")
        market_detector = analytics.get("market_detector")
        
        price = price_feed.get("last") or price_feed.get("close")
        pct_change = price_feed.get("pct_change")
        slope = price_series.get("slope_per_hour_equiv")
        trend = classify_trend(slope)
        std_return = price_series.get("std_return")
        volatility = classify_volatility(std_return)
        price_feed = analytics.get("price_feed", {})
        foreign_status, foreign_value = get_foreign_flow_status(findata, price_feed)
        bandar = get_bandar_signal(market_detector)
        outlook = get_outlook(trend, foreign_status, bandar, volatility)
        
        # Support/Resistance
        top_levels = price_series.get("top_levels", [])
        support_levels = []
        resistance_levels = []
        
        if top_levels:
            prices = [level[0] for level in top_levels[:5]]
            if price:
                for p in prices:
                    if p < price:
                        support_levels.append(str(p))
                    else:
                        resistance_levels.append(str(p))
        
        bid_cluster = depth.get("bid", {}).get("max_cluster_price")
        offer_cluster = depth.get("offer", {}).get("max_cluster_price")
        
        if bid_cluster:
            support_levels.append(str(bid_cluster))
        if offer_cluster:
            resistance_levels.append(str(offer_cluster))
        
        support_str = ", ".join(support_levels[:3]) if support_levels else "N/A"
        resistance_str = ", ".join(resistance_levels[:3]) if resistance_levels else "N/A"
        
        report_lines.append(f"### {idx}. Symbol: {symbol}")
        report_lines.append(f"- **Company**: {name}")
        report_lines.append(f"- **Current Price**: {price:,} IDR" if price else "- **Current Price**: N/A")
        report_lines.append(f"- **Today's Performance**: {pct_change:+.2f}%" if pct_change is not None else "- **Today's Performance**: N/A")
        report_lines.append(f"- **Short-Term Trend**: {trend} (slope: {slope:+.2f} per hour equivalent)" if slope is not None else "- **Short-Term Trend**: {trend}")
        report_lines.append(f"- **Key Support/Resistance**:")
        report_lines.append(f"  - Support: {support_str}")
        report_lines.append(f"  - Resistance: {resistance_str}")
        report_lines.append(f"- **Volatility**: {volatility} (std_return: {std_return:.4f})" if std_return is not None else f"- **Volatility**: {volatility}")
        report_lines.append(f"- **Liquidity**: {classify_liquidity(depth)}")
        report_lines.append(f"- **Foreign Flow**: {foreign_status} ({format_idr(foreign_value)})" if foreign_value else f"- **Foreign Flow**: {foreign_status}")
        report_lines.append(f"- **Bandar Signal**: {bandar}")
        report_lines.append(f"- **2-3 Day Outlook**: **{outlook}**")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Detailed Analysis by Symbol
    report_lines.append("## Detailed Analysis by Symbol")
    report_lines.append("")
    
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        name = symbol_info["name"]
        
        if symbol not in analytics_data:
            continue
        
        analytics = analytics_data[symbol]
        price_feed = analytics.get("price_feed", {})
        price_series = analytics.get("price_series", {})
        depth = analytics.get("depth", {})
        running_trade = analytics.get("running_trade", {})
        findata = analytics.get("findata")
        market_detector = analytics.get("market_detector")
        
        report_lines.append(f"### {symbol} - {name}")
        report_lines.append("")
        
        # Price Action
        slope = price_series.get("slope_per_hour_equiv")
        trend = classify_trend(slope)
        std_return = price_series.get("std_return")
        price_range = price_series.get("range")
        top_levels = price_series.get("top_levels", [])
        
        report_lines.append("**Price Action (2-3 Day Outlook):**")
        report_lines.append(f"- Current trend: {trend} (slope: {slope:+.2f} per hour equivalent)" if slope is not None else f"- Current trend: {trend}")
        
        if top_levels:
            levels_str = ", ".join([str(level[0]) for level in top_levels[:5]])
            report_lines.append(f"- Key price levels: {levels_str}")
        
        if std_return is not None:
            volatility_class = classify_volatility(std_return)
            report_lines.append(f"- Volatility: {volatility_class} (std_return: {std_return:.4f}, range: {price_range} IDR)" if price_range else f"- Volatility: {volatility_class} (std_return: {std_return:.4f})")
        
        report_lines.append("")
        
        # Order Book
        bid_vol = depth.get("bid", {}).get("total_volume", 0) or 0
        offer_vol = depth.get("offer", {}).get("total_volume", 0) or 0
        spread = depth.get("top_of_book", {}).get("spread")
        spread_bps = depth.get("top_of_book", {}).get("spread_bps")
        
        report_lines.append("**Order Book Insights:**")
        report_lines.append(f"- Liquidity depth: {format_idr(bid_vol)} bid vs {format_idr(offer_vol)} offer")
        if bid_vol and offer_vol:
            ratio = bid_vol / offer_vol
            imbalance = "favoring buyers" if ratio > 1.2 else "favoring sellers" if ratio < 0.8 else "balanced"
            report_lines.append(f"- Order book imbalances: {imbalance} (ratio: {ratio:.2f})")
        if spread is not None:
            report_lines.append(f"- Bid-ask spread: {spread} IDR ({spread_bps:.1f} bps)" if spread_bps else f"- Bid-ask spread: {spread} IDR")
        report_lines.append("")
        
        # Trade Flow
        lot_total = running_trade.get("lot_total", 0) or 0
        lot_buy = running_trade.get("lot_buy", 0) or 0
        lot_sell = running_trade.get("lot_sell", 0) or 0
        
        report_lines.append("**Trade Flow Analysis:**")
        report_lines.append(f"- Trading volume: {format_idr(lot_total)} shares (buy: {format_idr(lot_buy)}, sell: {format_idr(lot_sell)})")
        report_lines.append("")
        
        # Foreign Flow
        price_feed = analytics.get("price_feed", {})
        foreign_status, foreign_value = get_foreign_flow_status(findata, price_feed)
        if foreign_value is not None or findata:
            report_lines.append("**Foreign vs Domestic Flow:**")
            if foreign_value is not None:
                report_lines.append(f"- Net foreign flow: {foreign_status} ({format_idr(foreign_value)})")
            else:
                report_lines.append(f"- Net foreign flow: {foreign_status}")
            report_lines.append("")
        
        # Recommendation
        outlook = get_outlook(trend, foreign_status if findata else "Unknown", get_bandar_signal(market_detector), classify_volatility(std_return))
        
        action = "Buy" if outlook == "Bullish" else "Avoid" if outlook == "Bearish" else "Neutral"
        
        report_lines.append("**Short-Term Trading Recommendation (2-3 Days):**")
        report_lines.append(f"- **Action**: {action}")
        report_lines.append(f"- **Time Horizon**: 2-3 trading days")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Cross-Symbol Patterns
    report_lines.append("## Cross-Symbol Patterns")
    report_lines.append("")
    
    # Analyze common themes
    volatility_low = []
    volatility_medium = []
    volatility_high = []
    foreign_positive = []
    foreign_negative = []
    uptrends = []
    downtrends = []
    
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        if symbol not in analytics_data:
            continue
        
        analytics = analytics_data[symbol]
        price_series = analytics.get("price_series", {})
        findata = analytics.get("findata")
        
        std_return = price_series.get("std_return")
        volatility = classify_volatility(std_return)
        if volatility == "Low":
            volatility_low.append(symbol)
        elif volatility == "Medium":
            volatility_medium.append(symbol)
        elif volatility == "High":
            volatility_high.append(symbol)
        
        price_feed = analytics.get("price_feed", {})
        foreign_status, _ = get_foreign_flow_status(findata, price_feed)
        if foreign_status == "Positive":
            foreign_positive.append(symbol)
        elif foreign_status == "Negative":
            foreign_negative.append(symbol)
        
        slope = price_series.get("slope_per_hour_equiv")
        trend = classify_trend(slope)
        if trend == "Uptrend":
            uptrends.append(symbol)
        elif trend == "Downtrend":
            downtrends.append(symbol)
    
    report_lines.append("**Common Themes:**")
    report_lines.append("")
    
    if foreign_positive or foreign_negative:
        report_lines.append(f"1. **Foreign Flow Patterns**: {len(foreign_positive)} symbols show positive foreign flow, {len(foreign_negative)} show negative flow")
    
    if uptrends or downtrends:
        report_lines.append(f"2. **Trend Distribution**: {len(uptrends)} symbols in uptrend, {len(downtrends)} in downtrend")
    
    if volatility_low or volatility_medium or volatility_high:
        report_lines.append(f"3. **Volatility Clusters**: Low ({len(volatility_low)}), Medium ({len(volatility_medium)}), High ({len(volatility_high)})")
    
    report_lines.append("")
    report_lines.append("**Relative Strength Ranking** (by short-term potential):")
    report_lines.append("")
    
    # Use the ranked opportunities from earlier
    for idx, (symbol, name, score, _) in enumerate(ranked_opportunities[:10], 1):
        report_lines.append(f"{idx}. **{symbol}** - {name}")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Risk Summary
    report_lines.append("## Risk Summary")
    report_lines.append("")
    report_lines.append("**Overall Market Conditions:**")
    report_lines.append("")
    report_lines.append(f"- **Aggregate Volatility**: {len(volatility_high)} high, {len(volatility_medium)} medium, {len(volatility_low)} low volatility symbols")
    report_lines.append(f"- **Foreign Flow Trends**: {len(foreign_positive)} positive, {len(foreign_negative)} negative")
    report_lines.append(f"- **Trend Distribution**: {len(uptrends)} uptrends, {len(downtrends)} downtrends")
    report_lines.append("")
    report_lines.append("**Symbol-Specific Risks:**")
    report_lines.append("")
    
    # Identify high-risk symbols
    high_risk = []
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        if symbol not in analytics_data:
            continue
        
        analytics = analytics_data[symbol]
        price_series = analytics.get("price_series", {})
        std_return = price_series.get("std_return")
        volatility = classify_volatility(std_return)
        
        if volatility == "High":
            high_risk.append(symbol)
    
    if high_risk:
        report_lines.append(f"- **High Volatility Symbols**: {', '.join(high_risk)} - Require careful position sizing")
    
    # Check for missing data
    missing_data_symbols = []
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        if symbol not in analytics_data:
            missing_data_symbols.append(symbol)
        else:
            analytics = analytics_data[symbol]
            metadata = analytics.get("metadata", {})
            missing_sources = metadata.get("missing_sources", [])
            if missing_sources:
                missing_data_symbols.append(f"{symbol} (missing: {', '.join(missing_sources)})")
    
    if missing_data_symbols:
        report_lines.append(f"- **Symbols with Data Quality Issues**: {', '.join(missing_data_symbols)}")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Actionable Recommendations
    report_lines.append("## Actionable Recommendations (2-3 Day Focus)")
    report_lines.append("")
    report_lines.append("**Immediate Actions:**")
    report_lines.append("")
    
    if top_opportunities:
        report_lines.append("1. **Top Symbols to Watch**:")
        for idx, (symbol, name, _, _) in enumerate(top_opportunities[:3], 1):
            report_lines.append(f"   - **{symbol}**: {name}")
        report_lines.append("")
    
    report_lines.append("**Risk Management:**")
    report_lines.append("")
    report_lines.append("- Use appropriate position sizing based on volatility")
    report_lines.append("- Set stop losses based on support levels")
    report_lines.append("- Monitor foreign flow changes daily")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Appendix
    report_lines.append("## Appendix")
    report_lines.append("")
    report_lines.append("### Data Quality")
    report_lines.append("")
    
    if missing_data_symbols:
        report_lines.append("**Symbols with Missing Data Sources:**")
        for symbol in missing_data_symbols:
            report_lines.append(f"- {symbol}")
        report_lines.append("")
    
    report_lines.append("### Screener Details")
    report_lines.append("")
    report_lines.append(f"**Screener Name**: {screener_name}")
    report_lines.append(f"**Screener ID**: {screener_id}")
    report_lines.append("")
    
    # Extract screener metrics
    if symbols_data:
        report_lines.append("**Original Screener Metrics**:")
        report_lines.append("")
        # Get unique metric names
        metric_names = set()
        for s in symbols_data:
            metric_names.update(s.get("metrics", {}).keys())
        
        if metric_names:
            header = "| Symbol | " + " | ".join(metric_names) + " |"
            report_lines.append(header)
            report_lines.append("|" + "|".join(["--------"] * (len(metric_names) + 1)) + "|")
            
            for symbol_info in symbols_data:
                symbol = symbol_info["symbol"]
                metrics = symbol_info.get("metrics", {})
                row = f"| {symbol} | " + " | ".join([str(metrics.get(m, "N/A")) for m in metric_names]) + " |"
                report_lines.append(row)
    
    report_lines.append("")
    report_lines.append("### Limitations")
    report_lines.append("")
    report_lines.append("- Analysis focused on 2-3 trading days only")
    report_lines.append("- Data represents snapshot at time of analysis")
    report_lines.append("- Market conditions can change rapidly")
    report_lines.append("")
    report_lines.append(f"**Report Generated**: {timestamp}")
    report_lines.append("**Analysis Tool**: Bulk Market Data Analysis (Screener-Based)")
    
    # Write report to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✓ Bulk analysis report generated: {output_path}")


def main():
    """Main workflow for /analyze-bsjp command."""
    workspace_root = Path(__file__).resolve().parent
    screener_path = workspace_root / "screener.json"
    
    print("="*60)
    print("Bulk Market Data Analysis (Screener-Based)")
    print("="*60)
    
    # Step 0: ALWAYS fetch fresh screener data first (MANDATORY)
    print("\n[Step 0] Fetching fresh screener data (MANDATORY)...")
    if not fetch_screener_data():
        print("✗ Failed to fetch screener data. Cannot proceed.")
        sys.exit(1)
    
    # Step 1: Extract symbols from screener
    print("\n[Step 1] Extracting symbols from screener...")
    symbols_data = extract_symbols_from_screener(screener_path)
    
    if not symbols_data:
        print("✗ No symbols found in screener. Cannot proceed.")
        sys.exit(1)
    
    print(f"Symbols to analyze: {', '.join([s['symbol'] for s in symbols_data])}")
    
    # Step 2 & 3: Initialize data and generate analytics for each symbol
    print("\n[Step 2-3] Initializing data and generating analytics...")
    analytics_data = {}
    failed_symbols = []
    
    for symbol_info in symbols_data:
        symbol = symbol_info["symbol"]
        print(f"\nProcessing {symbol}...")
        
        # Initialize data
        success, session = initialize_symbol_data(symbol)
        if not success or session is None:
            print(f"✗ Failed to initialize data for {symbol}")
            failed_symbols.append(symbol)
            continue
        
        # Generate analytics
        success = generate_analytics(symbol, session)
        if not success:
            print(f"✗ Failed to generate analytics for {symbol}")
            failed_symbols.append(symbol)
            continue
        
        # Load analytics
        analytics = load_analytics_data(symbol, session)
        if analytics:
            analytics_data[symbol] = analytics
        else:
            print(f"✗ Failed to load analytics for {symbol}")
            failed_symbols.append(symbol)
    
    if not analytics_data:
        print("✗ No analytics data loaded. Cannot generate report.")
        sys.exit(1)
    
    print(f"\n✓ Successfully processed {len(analytics_data)}/{len(symbols_data)} symbols")
    if failed_symbols:
        print(f"✗ Failed symbols: {', '.join(failed_symbols)}")
    
    # Step 5: Generate bulk report
    print("\n[Step 5] Generating bulk analysis report...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = workspace_root / "sources" / "BULK_BSJP" / f"analysis-bulk-{timestamp}.md"
    
    generate_bulk_report(symbols_data, analytics_data, screener_path, output_path)
    
    print("\n" + "="*60)
    print("✓ Bulk analysis complete!")
    print(f"Report saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

