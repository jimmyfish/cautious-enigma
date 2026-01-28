#!/usr/bin/env python3
"""
Telegram Bot Integration for Stock Forecasts.

Provides functions to send forecast results via Telegram Bot API.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env file automatically.
"""

import os
import argparse
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram API base URL
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def get_telegram_config() -> tuple[str, str]:
    """
    Get Telegram bot token and chat ID from environment variables.

    Returns:
        Tuple of (bot_token, chat_id)

    Raises:
        ValueError: If credentials are not configured
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        raise ValueError(
            "Telegram not configured. Add to .env:\n"
            "  TELEGRAM_BOT_TOKEN=your_bot_token\n"
            "  TELEGRAM_CHAT_ID=your_chat_id"
        )

    return bot_token, chat_id


def send_telegram_message(
    message: str,
    parse_mode: str = "HTML",
    disable_notification: bool = False,
) -> bool:
    """
    Send a message via Telegram Bot API.

    Args:
        message: The message text to send
        parse_mode: Message format (HTML, Markdown, MarkdownV2)
        disable_notification: If True, send silently

    Returns:
        True if message was sent successfully

    Raises:
        ValueError: If Telegram is not configured
        requests.RequestException: On API error
    """
    bot_token, chat_id = get_telegram_config()

    url = TELEGRAM_API_URL.format(token=bot_token)
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_notification": disable_notification,
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    if not result.get("ok"):
        raise requests.RequestException(f"Telegram API error: {result.get('description')}")

    return True


def format_forecast_message(
    symbol: str,
    current_price: float,
    forecasts: List[Dict[str, Any]],
    outlook: str,
    change_pct: float,
    currency: str = "IDR",
    ara_arb_info: Optional[Dict[str, Any]] = None,
    script_name: str = "Forecast",
) -> str:
    """
    Format forecast results into a Telegram message.

    Args:
        symbol: Stock symbol
        current_price: Current/last price
        forecasts: List of forecast dicts with 'date', 'price', 'change_pct'
        outlook: BULLISH, BEARISH, or NEUTRAL
        change_pct: Overall change percentage
        currency: Currency code (default: IDR)
        ara_arb_info: Optional dict with 'ara_price', 'arb_price', 'max_gain_pct', 'max_loss_pct'
        script_name: Name of the forecasting script

    Returns:
        Formatted HTML message string
    """
    # Outlook emoji
    outlook_emoji = {
        "BULLISH": "\U0001F7E2",  # Green circle
        "BEARISH": "\U0001F534",  # Red circle
        "NEUTRAL": "\U0001F7E1",  # Yellow circle
    }.get(outlook.upper(), "\u2B55")  # Hollow circle fallback

    lines = []
    lines.append(f"<b>{outlook_emoji} {symbol} - {script_name}</b>")
    lines.append("")
    lines.append(f"<b>Current:</b> {current_price:,.0f} {currency}")

    # ARA/ARB info for Indonesian stocks
    if ara_arb_info:
        lines.append("")
        lines.append("<b>ARA/ARB Limits:</b>")
        lines.append(f"  ARA: {ara_arb_info['ara_price']:,.0f} (+{ara_arb_info['max_gain_pct']:.1f}%)")
        lines.append(f"  ARB: {ara_arb_info['arb_price']:,.0f} (-{ara_arb_info['max_loss_pct']:.1f}%)")

    # Forecast table
    lines.append("")
    lines.append("<b>Forecast:</b>")
    lines.append("<pre>")
    lines.append(f"{'Date':<12}{'Price':>10}{'Chg':>8}")
    lines.append("-" * 30)

    for fc in forecasts:
        date_str = fc.get("date", "")[:10]
        price = fc.get("price", 0)
        pct = fc.get("change_pct", 0)
        lines.append(f"{date_str:<12}{price:>10,.0f}{pct:>+7.1f}%")

    lines.append("</pre>")

    # Summary
    lines.append("")
    lines.append(f"<b>Outlook:</b> {outlook} ({change_pct:+.2f}%)")

    return "\n".join(lines)


def format_batch_summary_message(
    results: List[Dict[str, Any]],
    script_name: str = "Batch Forecast",
) -> str:
    """
    Format batch forecast results into a Telegram message.

    Args:
        results: List of dicts with 'symbol', 'name', 'current_price', 'forecast', 'change_pct', 'outlook'
        script_name: Name of the forecasting script

    Returns:
        Formatted HTML message string
    """
    if not results:
        return f"<b>{script_name}</b>\n\nNo results to report."

    # Count by outlook
    bullish = sum(1 for r in results if r.get("outlook", "").upper() == "BULLISH")
    bearish = sum(1 for r in results if r.get("outlook", "").upper() == "BEARISH")
    neutral = sum(1 for r in results if r.get("outlook", "").upper() == "NEUTRAL")

    lines = []
    lines.append(f"<b>{script_name} - {len(results)} Symbols</b>")
    lines.append(f"\U0001F7E2 Bullish: {bullish} | \U0001F534 Bearish: {bearish} | \U0001F7E1 Neutral: {neutral}")
    lines.append("")
    lines.append("<pre>")
    lines.append(f"{'Symbol':<8}{'Price':>10}{'Fcst':>10}{'Chg':>8}")
    lines.append("-" * 36)

    for r in results:
        symbol = r.get("symbol", "")[:7]
        current = r.get("current_price", 0)
        forecast = r.get("forecast", 0)
        pct = r.get("change_pct", 0)
        outlook = r.get("outlook", "").upper()

        # Emoji prefix based on outlook
        emoji = "\U0001F7E2" if outlook == "BULLISH" else "\U0001F534" if outlook == "BEARISH" else " "
        lines.append(f"{emoji}{symbol:<7}{current:>10,.0f}{forecast:>10,.0f}{pct:>+7.1f}%")

    lines.append("</pre>")

    return "\n".join(lines)


def send_forecast_notification(
    symbol: str,
    current_price: float,
    forecasts: List[Dict[str, Any]],
    outlook: str,
    change_pct: float,
    currency: str = "IDR",
    ara_arb_info: Optional[Dict[str, Any]] = None,
    script_name: str = "Forecast",
    silent: bool = False,
) -> bool:
    """
    Send forecast results via Telegram.

    Args:
        symbol: Stock symbol
        current_price: Current/last price
        forecasts: List of forecast dicts with 'date', 'price', 'change_pct'
        outlook: BULLISH, BEARISH, or NEUTRAL
        change_pct: Overall change percentage
        currency: Currency code
        ara_arb_info: Optional ARA/ARB info for Indonesian stocks
        script_name: Name of the forecasting script
        silent: If True, send without notification sound

    Returns:
        True if sent successfully, False on error
    """
    try:
        message = format_forecast_message(
            symbol=symbol,
            current_price=current_price,
            forecasts=forecasts,
            outlook=outlook,
            change_pct=change_pct,
            currency=currency,
            ara_arb_info=ara_arb_info,
            script_name=script_name,
        )
        return send_telegram_message(message, disable_notification=silent)
    except Exception as e:
        print(f"  Telegram notification failed: {e}")
        return False


def send_batch_notification(
    results: List[Dict[str, Any]],
    script_name: str = "Batch Forecast",
    silent: bool = False,
) -> bool:
    """
    Send batch forecast results via Telegram.

    Args:
        results: List of result dicts
        script_name: Name of the forecasting script
        silent: If True, send without notification sound

    Returns:
        True if sent successfully, False on error
    """
    try:
        message = format_batch_summary_message(results, script_name)
        return send_telegram_message(message, disable_notification=silent)
    except Exception as e:
        print(f"  Telegram notification failed: {e}")
        return False


def add_telegram_args(parser: argparse.ArgumentParser) -> None:
    """Add Telegram-related arguments to an argument parser."""
    telegram_group = parser.add_argument_group("telegram options")
    telegram_group.add_argument(
        "-t", "--telegram",
        action="store_true",
        help="Send forecast results to Telegram"
    )
    telegram_group.add_argument(
        "--tg-silent",
        action="store_true",
        help="Send Telegram notification silently (no sound)"
    )


def validate_telegram_args(args: argparse.Namespace) -> bool:
    """
    Validate Telegram arguments and check configuration.

    Returns True if valid, raises ValueError if invalid.
    """
    if args.telegram:
        # Check if Telegram is configured
        try:
            get_telegram_config()
        except ValueError as e:
            raise ValueError(str(e))

    return True


if __name__ == "__main__":
    # Test the module
    import sys

    print("Testing Telegram module...")

    try:
        # Check configuration
        bot_token, chat_id = get_telegram_config()
        print(f"Bot token: {bot_token[:10]}...{bot_token[-5:]}")
        print(f"Chat ID: {chat_id}")

        # Send test message
        if len(sys.argv) > 1 and sys.argv[1] == "--send":
            test_forecasts = [
                {"date": "2024-01-15", "price": 1050, "change_pct": 2.5},
                {"date": "2024-01-16", "price": 1075, "change_pct": 4.9},
                {"date": "2024-01-17", "price": 1100, "change_pct": 7.3},
            ]

            success = send_forecast_notification(
                symbol="TEST",
                current_price=1025,
                forecasts=test_forecasts,
                outlook="BULLISH",
                change_pct=7.3,
                currency="IDR",
                ara_arb_info={
                    "ara_price": 1256,
                    "arb_price": 858,
                    "max_gain_pct": 22.5,
                    "max_loss_pct": 16.3,
                },
                script_name="Test Forecast",
            )
            print(f"Test message sent: {success}")
        else:
            print("\nRun with --send to send a test message")

    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
