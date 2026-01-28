"""
Stock forecasting modules package.

Provides common utilities shared across forecasting scripts.
"""

from .common import (
    BASE_DIR,
    CSV_DIR,
    GROUPS_FILE,
    MODELS_DIR,
    PLOT_DIR,
    SOURCES_DIR,
    ColoredFormatter,
    JsonDict,
    find_group_for_symbol,
    load_groups,
    setup_logging,
)

from .telegram import (
    add_telegram_args,
    format_batch_summary_message,
    format_forecast_message,
    send_batch_notification,
    send_forecast_notification,
    send_telegram_message,
    validate_telegram_args,
)

__all__ = [
    "JsonDict",
    "BASE_DIR",
    "MODELS_DIR",
    "CSV_DIR",
    "PLOT_DIR",
    "SOURCES_DIR",
    "GROUPS_FILE",
    "ColoredFormatter",
    "setup_logging",
    "load_groups",
    "find_group_for_symbol",
    # Telegram
    "add_telegram_args",
    "validate_telegram_args",
    "send_telegram_message",
    "send_forecast_notification",
    "send_batch_notification",
    "format_forecast_message",
    "format_batch_summary_message",
]
