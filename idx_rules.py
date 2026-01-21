#!/usr/bin/env python3
"""
Indonesian Stock Exchange (IDX) Trading Rules

ARA (Auto Rejection Atas) - Upper price limit
ARB (Auto Rejection Bawah) - Lower price limit

Based on IDX regulations (OJK and BEI rules):
- Price limits vary by stock price range
- Limits apply per trading day based on previous closing price
- When a stock hits ARA/ARB, trading continues but price cannot exceed the limit

Reference: https://www.idx.co.id/id/produk/mekanisme-dan-jam-perdagangan
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


# IDX Price Limit Rules (as of 2025)
# These may change - check IDX website for current rules
# Format: (min_price, max_price, ara_pct, arb_pct)
IDX_PRICE_LIMITS = [
    (0, 50, 0.35, 0.35),        # < 50 IDR: ±35% (adjusted for small caps volatility)
    (50, 200, 0.35, 0.35),      # 50-200 IDR: ±35%
    (200, 5000, 0.25, 0.25),    # 200-5000 IDR: ±25%
    (5000, float('inf'), 0.20, 0.20),  # > 5000 IDR: ±20%
]

# Tick size rules for IDX
# Format: (min_price, max_price, tick_size)
IDX_TICK_SIZES = [
    (0, 200, 1),           # < 200 IDR: 1 IDR tick
    (200, 500, 2),         # 200-500 IDR: 2 IDR tick
    (500, 2000, 5),        # 500-2000 IDR: 5 IDR tick
    (2000, 5000, 10),      # 2000-5000 IDR: 10 IDR tick
    (5000, float('inf'), 25),  # > 5000 IDR: 25 IDR tick
]


def get_ara_arb_limits(price: float) -> Tuple[float, float]:
    """
    Get ARA/ARB percentage limits for a given price.

    Args:
        price: Stock price in IDR

    Returns:
        Tuple of (ara_pct, arb_pct) as decimals (e.g., 0.25 for 25%)
    """
    if price <= 0:
        return 0.25, 0.25  # Default

    for min_p, max_p, ara, arb in IDX_PRICE_LIMITS:
        if min_p <= price < max_p:
            return ara, arb

    return 0.20, 0.20  # Default for very high prices


def get_tick_size(price: float) -> int:
    """
    Get tick size for a given price.

    Args:
        price: Stock price in IDR

    Returns:
        Tick size in IDR
    """
    if price <= 0:
        return 1

    for min_p, max_p, tick in IDX_TICK_SIZES:
        if min_p <= price < max_p:
            return tick

    return 25  # Default for very high prices


def round_to_tick(price: float, base_price: float = None) -> float:
    """
    Round price to nearest valid tick.

    Args:
        price: Price to round
        base_price: Reference price for tick size (uses price if None)

    Returns:
        Price rounded to nearest valid tick
    """
    ref = base_price if base_price is not None else price
    tick = get_tick_size(ref)
    return round(price / tick) * tick


def calculate_ara_arb(prev_close: float) -> Tuple[float, float]:
    """
    Calculate ARA (upper limit) and ARB (lower limit) prices for a trading day.

    Args:
        prev_close: Previous day's closing price

    Returns:
        Tuple of (ara_price, arb_price) in IDR
    """
    if prev_close <= 0:
        return 0, 0

    ara_pct, arb_pct = get_ara_arb_limits(prev_close)

    ara = prev_close * (1 + ara_pct)
    arb = prev_close * (1 - arb_pct)

    # Round to valid tick sizes
    ara = round_to_tick(ara, prev_close)
    arb = max(round_to_tick(arb, prev_close), get_tick_size(prev_close))  # ARB can't be below 1 tick

    return ara, arb


def clamp_price_to_limit(predicted: float, prev_close: float) -> float:
    """
    Clamp a predicted price to valid ARA/ARB range.

    Args:
        predicted: Predicted price
        prev_close: Previous day's closing price

    Returns:
        Clamped price within ARA/ARB limits
    """
    ara, arb = calculate_ara_arb(prev_close)
    clamped = max(min(predicted, ara), arb)
    return round_to_tick(clamped, prev_close)


def clamp_forecast_series(forecasts: pd.DataFrame, last_price: float,
                          price_col: str = "ensemble") -> pd.DataFrame:
    """
    Clamp a series of forecasts to valid daily ARA/ARB limits.
    Each day's limit is based on the previous day's (forecasted) close.

    Args:
        forecasts: DataFrame with forecast prices
        last_price: Last known actual closing price
        price_col: Column name for the price to clamp

    Returns:
        DataFrame with clamped prices and ARA/ARB columns added
    """
    result = forecasts.copy()

    # Initialize tracking
    clamped_prices = []
    ara_limits = []
    arb_limits = []

    prev = last_price

    for idx, row in result.iterrows():
        if price_col in row:
            pred = row[price_col]
            ara, arb = calculate_ara_arb(prev)
            clamped = clamp_price_to_limit(pred, prev)

            clamped_prices.append(clamped)
            ara_limits.append(ara)
            arb_limits.append(arb)

            # Next day's limit is based on this day's clamped forecast
            prev = clamped
        else:
            clamped_prices.append(np.nan)
            ara_limits.append(np.nan)
            arb_limits.append(np.nan)

    result[f"{price_col}_clamped"] = clamped_prices
    result["ara_limit"] = ara_limits
    result["arb_limit"] = arb_limits

    # Also clamp confidence intervals if they exist
    if "low" in result.columns:
        result["low_clamped"] = result.apply(
            lambda r: max(r["low"], r["arb_limit"]) if pd.notna(r["arb_limit"]) else r["low"],
            axis=1
        )
    if "high" in result.columns:
        result["high_clamped"] = result.apply(
            lambda r: min(r["high"], r["ara_limit"]) if pd.notna(r["ara_limit"]) else r["high"],
            axis=1
        )

    return result


def detect_ara_arb_hit(df: pd.DataFrame, close_col: str = "close",
                       high_col: str = "high", low_col: str = "low") -> pd.DataFrame:
    """
    Detect days where price hit ARA or ARB limits.

    Args:
        df: DataFrame with OHLC data
        close_col, high_col, low_col: Column names

    Returns:
        DataFrame with ara_hit, arb_hit, and limit_hit columns added
    """
    result = df.copy()

    # Calculate previous close
    result["prev_close"] = result[close_col].shift(1)

    # Calculate limits for each row
    limits = result["prev_close"].apply(
        lambda x: calculate_ara_arb(x) if pd.notna(x) and x > 0 else (np.nan, np.nan)
    )
    result["day_ara"] = limits.apply(lambda x: x[0])
    result["day_arb"] = limits.apply(lambda x: x[1])

    # Detect hits (with small tolerance for rounding)
    tolerance = 0.001  # 0.1%

    result["ara_hit"] = (
        (result[high_col] >= result["day_ara"] * (1 - tolerance)) |
        (result[close_col] >= result["day_ara"] * (1 - tolerance))
    ).astype(int)

    result["arb_hit"] = (
        (result[low_col] <= result["day_arb"] * (1 + tolerance)) |
        (result[close_col] <= result["day_arb"] * (1 + tolerance))
    ).astype(int)

    result["limit_hit"] = ((result["ara_hit"] == 1) | (result["arb_hit"] == 1)).astype(int)

    # Fill NaN for first row
    result.loc[result.index[0], ["ara_hit", "arb_hit", "limit_hit"]] = 0

    return result


def add_ara_arb_features(df: pd.DataFrame, close_col: str = "close",
                         high_col: str = "high", low_col: str = "low") -> pd.DataFrame:
    """
    Add ARA/ARB-related features for model training.

    Features added:
    - ara_proximity: How close high is to ARA (0-1, 1 = at ARA)
    - arb_proximity: How close low is to ARB (0-1, 1 = at ARB)
    - limit_range_pct: Today's price range as % of available limit range
    - ara_hit: Binary flag if ARA was hit
    - arb_hit: Binary flag if ARB was hit
    - limit_hit: Binary flag if any limit was hit
    - pct_to_ara: % move needed to reach ARA from close
    - pct_to_arb: % move needed to reach ARB from close

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with new features added
    """
    result = detect_ara_arb_hit(df, close_col, high_col, low_col)

    # Proximity features (how close did price get to limits)
    # For ARA: (high - prev_close) / (ara - prev_close)
    # For ARB: (prev_close - low) / (prev_close - arb)

    ara_range = result["day_ara"] - result["prev_close"]
    arb_range = result["prev_close"] - result["day_arb"]

    result["ara_proximity"] = np.clip(
        (result[high_col] - result["prev_close"]) / (ara_range + 1e-8),
        0, 1
    )
    result["arb_proximity"] = np.clip(
        (result["prev_close"] - result[low_col]) / (arb_range + 1e-8),
        0, 1
    )

    # How much of the available range was used today
    total_range = ara_range + arb_range
    used_range = result[high_col] - result[low_col]
    result["limit_range_pct"] = used_range / (total_range + 1e-8)

    # Percent move needed to reach limits from current close
    result["pct_to_ara"] = (result["day_ara"] - result[close_col]) / (result[close_col] + 1e-8)
    result["pct_to_arb"] = (result[close_col] - result["day_arb"]) / (result[close_col] + 1e-8)

    # Direction bias: positive = closer to ARA, negative = closer to ARB
    result["limit_bias"] = result["ara_proximity"] - result["arb_proximity"]

    # Fill NaN values
    fill_cols = ["ara_proximity", "arb_proximity", "limit_range_pct",
                 "pct_to_ara", "pct_to_arb", "limit_bias"]
    for col in fill_cols:
        result[col] = result[col].fillna(0)

    # Clean up intermediate columns
    result = result.drop(columns=["prev_close", "day_ara", "day_arb"], errors="ignore")

    return result


def adjust_mask_for_limit_hits(df: pd.DataFrame, mask_col: str = "available_mask",
                               limit_col: str = "limit_hit",
                               penalty: float = 0.5) -> pd.DataFrame:
    """
    Adjust available_mask to down-weight days that hit ARA/ARB limits.
    These days have distorted price signals and should be weighted less in training.

    Args:
        df: DataFrame with available_mask and limit_hit columns
        mask_col: Name of the mask column
        limit_col: Name of the limit hit column
        penalty: Multiplier for limit-hit days (0.5 = half weight)

    Returns:
        DataFrame with adjusted mask
    """
    result = df.copy()

    if mask_col not in result.columns:
        result[mask_col] = 1.0

    if limit_col in result.columns:
        # Reduce weight for limit-hit days
        result[mask_col] = result[mask_col] * (
            1 - (1 - penalty) * result[limit_col]
        )

    return result


def is_indonesian_stock(symbol: str, source: str = "local") -> bool:
    """
    Check if a symbol is an Indonesian stock.
    Indonesian stocks on Yahoo Finance end with .JK
    Local sources (in sources/ directory) are assumed to be Indonesian.

    Args:
        symbol: Stock symbol
        source: Data source ("local", "yahoo", etc.)

    Returns:
        True if Indonesian stock
    """
    symbol = symbol.upper()

    # Yahoo Finance uses .JK suffix for IDX stocks
    if symbol.endswith(".JK"):
        return True

    # For local sources (e.g., sources/ directory), assume Indonesian
    if source == "local":
        return True

    # For Yahoo Finance without .JK, it's not Indonesian
    if source == "yahoo":
        return False

    return False


def get_daily_limit_info(prev_close: float) -> dict:
    """
    Get comprehensive limit information for a trading day.

    Args:
        prev_close: Previous day's closing price

    Returns:
        Dictionary with limit information
    """
    ara, arb = calculate_ara_arb(prev_close)
    ara_pct, arb_pct = get_ara_arb_limits(prev_close)
    tick = get_tick_size(prev_close)

    return {
        "prev_close": prev_close,
        "ara_price": ara,
        "arb_price": arb,
        "ara_pct": ara_pct * 100,
        "arb_pct": arb_pct * 100,
        "tick_size": tick,
        "max_gain_pct": (ara / prev_close - 1) * 100,
        "max_loss_pct": (1 - arb / prev_close) * 100,
    }


# Convenience function for quick checks
def print_limits(price: float):
    """Print limit information for a given price."""
    info = get_daily_limit_info(price)
    print(f"\nPrice Limits for {price:,.0f} IDR:")
    print(f"  ARA (Upper): {info['ara_price']:,.0f} IDR (+{info['max_gain_pct']:.1f}%)")
    print(f"  ARB (Lower): {info['arb_price']:,.0f} IDR (-{info['max_loss_pct']:.1f}%)")
    print(f"  Tick Size: {info['tick_size']} IDR")


if __name__ == "__main__":
    # Test examples
    test_prices = [45, 100, 500, 1500, 3000, 7500, 15000]

    print("IDX ARA/ARB Limit Calculator")
    print("=" * 50)

    for price in test_prices:
        print_limits(price)
