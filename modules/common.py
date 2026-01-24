"""
Common utilities for stock forecasting modules.

This module contains shared functions and constants used across:
- forecast.py (daily forecasting)
- short.py (intraday forecasting)
- yf.py (Yahoo Finance forecasting)
- initiate.py (data collection)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, TypeAlias

# =============================================================================
# Type Aliases
# =============================================================================

JsonDict: TypeAlias = Dict[str, Any]

# =============================================================================
# Directory Constants
# =============================================================================

BASE_DIR: Final[Path] = Path(__file__).parent.parent
MODELS_DIR: Final[Path] = BASE_DIR / "models"
CSV_DIR: Final[Path] = BASE_DIR / "csv"
PLOT_DIR: Final[Path] = BASE_DIR / "plot"
SOURCES_DIR: Final[Path] = BASE_DIR / "sources"
GROUPS_FILE: Final[Path] = MODELS_DIR / "groups.json"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Logging Configuration
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Logging formatter with color support for terminal output."""

    COLORS: Dict[int, str] = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}" if color else message


def setup_logging(
    name: str,
    level: int = logging.INFO,
    use_color: bool = True
) -> logging.Logger:
    """
    Configure and return an application logger.

    Args:
        name: Logger name (e.g., "forecast", "short", "yf")
        level: Logging level (default: INFO)
        use_color: Whether to use colored output (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter_cls = ColoredFormatter if use_color else logging.Formatter
        handler.setFormatter(formatter_cls(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)

    return logger


# =============================================================================
# Group Management
# =============================================================================

def load_groups() -> Dict[str, List[str]]:
    """
    Load stock groups from the groups.json config file.

    Returns:
        Dictionary mapping group names to lists of stock symbols.
        Returns empty dict if file doesn't exist or is invalid.
    """
    if not GROUPS_FILE.exists():
        return {}

    try:
        with GROUPS_FILE.open("r", encoding="utf-8") as f:
            groups = json.load(f)
        # Remove comment entries (keys starting with "_")
        return {k: v for k, v in groups.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError):
        return {}


def find_group_for_symbol(symbol: str, groups: Dict[str, List[str]]) -> Optional[str]:
    """
    Find which group a symbol belongs to.

    Args:
        symbol: Stock symbol to search for
        groups: Dictionary of group names to symbol lists

    Returns:
        Group name if found, None otherwise
    """
    symbol_upper = symbol.upper()
    for group_name, symbols in groups.items():
        if symbol_upper in (s.upper() for s in symbols):
            return group_name
    return None
