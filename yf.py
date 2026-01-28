#!/usr/bin/env python3
"""
Stock Forecasting using Yahoo Finance Data - Production Grade

Neural network-based stock forecasting using NBEATS, NHITS, and LSTM models.
Supports any stock available on Yahoo Finance with Indonesian stock (IDX) specific
features including ARA/ARB price limit enforcement.

Usage:
    python yf.py AAPL                      # Daily forecast for Apple
    python yf.py BBRI.JK --horizon 10      # Indonesian stock
    python yf.py AAPL --interval 1h        # Hourly data
    python yf.py AAPL --period 6mo --plot  # 6 months history with chart
    python yf.py BBYB.JK,BBCA.JK,BMRI.JK   # Multiple symbols

Examples:
    python yf.py AAPL --horizon 5 --plot
    python yf.py TSLA --interval 1h --period 5d --horizon 24
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    TypeAlias,
)

import numpy as np
import pandas as pd
import yfinance as yf
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM

# Shared utilities
from modules import (
    CSV_DIR,
    PLOT_DIR,
    setup_logging,
)
from modules.idx_rules import (
    add_ara_arb_features,
    clamp_forecast_series,
    get_daily_limit_info,
    is_indonesian_stock,
)
from modules.watchlist import (
    add_watchlist_args,
    filter_symbols_by_outlook,
    print_watchlist_summary,
    update_watchlist,
    validate_watchlist_args,
)
from modules.telegram import (
    add_telegram_args,
    send_batch_notification,
    send_forecast_notification,
    validate_telegram_args,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration & Constants
# =============================================================================

# Type aliases
JsonDict: TypeAlias = Dict[str, Any]

# Model configuration
DEFAULT_HORIZON: Final[int] = 5
DEFAULT_INTERVAL: Final[str] = "1d"
DEFAULT_PERIOD: Final[str] = "3mo"
MIN_DATA_POINTS: Final[int] = 10
MAX_TRAINING_STEPS: Final[int] = 300
RANDOM_SEED: Final[int] = 42
MIN_PRICE_THRESHOLD: Final[float] = 50.0

# Outlook thresholds
BULLISH_THRESHOLD: Final[float] = 2.0
BEARISH_THRESHOLD: Final[float] = -2.0


# =============================================================================
# Interval Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class IntervalConfig:
    """Configuration for a data interval."""
    max_period: str
    freq: str


VALID_INTERVALS: Final[Dict[str, IntervalConfig]] = {
    "1m": IntervalConfig(max_period="7d", freq="1min"),
    "2m": IntervalConfig(max_period="60d", freq="2min"),
    "5m": IntervalConfig(max_period="60d", freq="5min"),
    "15m": IntervalConfig(max_period="60d", freq="15min"),
    "30m": IntervalConfig(max_period="60d", freq="30min"),
    "1h": IntervalConfig(max_period="730d", freq="1h"),
    "1d": IntervalConfig(max_period="max", freq="D"),
    "1wk": IntervalConfig(max_period="max", freq="W"),
    "1mo": IntervalConfig(max_period="max", freq="MS"),
}


# Logger
logger = setup_logging("yf_forecast")


# =============================================================================
# Custom Exceptions
# =============================================================================

class ForecastError(Exception):
    """Base exception for forecasting errors."""
    pass


class DataError(ForecastError):
    """Data loading or processing error."""
    pass


class InsufficientDataError(DataError):
    """Not enough data for forecasting."""
    def __init__(self, available: int, required: int):
        super().__init__(f"Need at least {required} data points (have {available})")
        self.available = available
        self.required = required


class InvalidIntervalError(ForecastError):
    """Invalid interval specified."""
    def __init__(self, interval: str):
        super().__init__(
            f"Invalid interval: {interval}. Valid: {list(VALID_INTERVALS.keys())}"
        )
        self.interval = interval


class LowPriceError(ForecastError):
    """Stock price is below minimum threshold."""
    def __init__(self, symbol: str, price: float, threshold: float):
        super().__init__(f"{symbol}: Price {price:.2f} below threshold {threshold:.2f}")
        self.symbol = symbol
        self.price = price
        self.threshold = threshold


# =============================================================================
# Data Classes
# =============================================================================

class Outlook(Enum):
    """Forecast outlook classification."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()

    @classmethod
    def from_percent_change(cls, pct: float) -> "Outlook":
        """Determine outlook from percentage change."""
        if pct > BULLISH_THRESHOLD:
            return cls.BULLISH
        elif pct < BEARISH_THRESHOLD:
            return cls.BEARISH
        return cls.NEUTRAL


@dataclass(slots=True)
class ForecastMeta:
    """Metadata for a forecast."""
    horizon: int
    interval: str
    input_size: int
    n_records: int
    is_indonesian: bool
    last_price: float
    features: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ForecastSummary:
    """Summary of a single symbol forecast."""
    symbol: str
    name: str
    currency: str
    current_price: float
    final_forecast: float
    change_pct: float
    outlook: Outlook
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    successful: List[ForecastSummary] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.successful) + len(self.skipped) + len(self.errors)

    @property
    def has_failures(self) -> bool:
        """Check if batch had failures with no successes."""
        return bool(self.errors) and not self.successful


# =============================================================================
# Technical Indicators
# =============================================================================

class TechnicalIndicators:
    """Calculate technical analysis indicators."""

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        result = series.rolling(window, min_periods=1).mean()
        return pd.Series(result)

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average."""
        result = series.ewm(span=span, adjust=False).mean()
        return pd.Series(result)

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        return pd.Series(100 - (100 / (1 + gain / (loss + 1e-8))))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """MACD as ratio to price."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow) / series

    @staticmethod
    def bollinger_position(series: pd.Series, window: int = 20) -> pd.Series:
        """Position within Bollinger Bands (-1 to 1)."""
        sma = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std()
        return (series - sma) / (2 * std + 1e-8)

    @staticmethod
    def volatility(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Intraday volatility as ratio."""
        return (high - low) / (close + 1e-8)

    @classmethod
    def add_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical features to DataFrame in a single pass."""
        # Extract columns as Series for type safety
        close: pd.Series = df["close"]  # type: ignore[assignment]
        high: pd.Series = df["high"]  # type: ignore[assignment]
        low: pd.Series = df["low"]  # type: ignore[assignment]
        volume: pd.Series = df["volume"]  # type: ignore[assignment]

        # Price returns
        df["returns"] = close.pct_change()

        # Volatility
        df["volatility"] = cls.volatility(high, low, close)

        # Moving averages
        df["sma_5"] = cls.sma(close, 5)
        df["sma_10"] = cls.sma(close, 10)
        df["sma_20"] = cls.sma(close, 20)

        # Price relative to MAs
        df["price_vs_sma5"] = (close / df["sma_5"]) - 1
        df["price_vs_sma20"] = (close / df["sma_20"]) - 1

        # Volume features
        df["vol_ma"] = cls.sma(volume, 5)
        df["vol_ratio"] = volume / (df["vol_ma"] + 1)

        # Momentum
        df["momentum_5"] = close.pct_change(5)
        df["momentum_10"] = close.pct_change(10)

        # Technical indicators
        df["rsi"] = cls.rsi(close)
        df["macd"] = cls.macd(close)
        df["bb_position"] = cls.bollinger_position(close)

        return df


# =============================================================================
# Yahoo Finance Data Loader
# =============================================================================

class YFinanceLoader:
    """Load and process stock data from Yahoo Finance."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self._info: Optional[Dict] = None
        self._is_indonesian = is_indonesian_stock(self.symbol, source="yahoo")

    @property
    def info(self) -> Dict:
        """Lazily load and cache stock info."""
        if self._info is None:
            try:
                self._info = self.ticker.info
            except Exception:
                self._info = {}
        return self._info

    @property
    def name(self) -> str:
        """Get company name."""
        return self.info.get("shortName", self.info.get("longName", self.symbol))

    @property
    def currency(self) -> str:
        """Get trading currency."""
        return self.info.get("currency", "USD")

    @property
    def is_indonesian(self) -> bool:
        """Check if this is an Indonesian stock."""
        return self._is_indonesian

    def load(self, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
        """
        Load historical data from Yahoo Finance.

        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data

        Raises:
            InvalidIntervalError: If interval is not valid
            DataError: If no data is returned
        """
        if interval not in VALID_INTERVALS:
            raise InvalidIntervalError(interval)

        logger.info(f"Fetching {self.symbol} data: period={period}, interval={interval}")

        df = self.ticker.history(period=period, interval=interval)

        if df.empty:
            raise DataError(f"No data returned for {self.symbol}")

        # Normalize DataFrame
        df = self._normalize_dataframe(df)

        return df

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns and index."""
        df = df.reset_index()

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Normalize datetime column
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ds"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "ds"})

        # Ensure datetime type and remove timezone
        df["ds"] = pd.to_datetime(df["ds"])
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical features to OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional feature columns
        """
        # Add all technical indicators in a single pass
        df = TechnicalIndicators.add_all_features(df)

        # Add ARA/ARB features for Indonesian stocks
        if self._is_indonesian:
            df = add_ara_arb_features(df, close_col="close", high_col="high", low_col="low")
            logger.info("  Added ARA/ARB features for Indonesian stock")

            # Report limit hits
            if "limit_hit" in df.columns:
                limit_days = df["limit_hit"].sum()
                if limit_days > 0:
                    ara_days = df["ara_hit"].sum() if "ara_hit" in df.columns else 0
                    arb_days = df["arb_hit"].sum() if "arb_hit" in df.columns else 0
                    logger.info(f"  Historical ARA/ARB hits: {int(ara_days)} ARA, {int(arb_days)} ARB")

        return df.fillna(0)


# =============================================================================
# Forecaster
# =============================================================================

class YFinanceForecaster:
    """Generate forecasts using NeuralForecast ensemble."""

    # Base exogenous features
    BASE_FEATURES: List[str] = [
        "volatility", "returns", "price_vs_sma5", "price_vs_sma20",
        "vol_ratio", "momentum_5", "rsi", "macd", "bb_position"
    ]

    # Additional features for Indonesian stocks
    IDX_FEATURES: List[str] = [
        "ara_proximity", "arb_proximity", "limit_range_pct",
        "limit_bias", "pct_to_ara", "pct_to_arb"
    ]

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        interval: str = DEFAULT_INTERVAL,
        symbol: str = "",
        max_steps: int = MAX_TRAINING_STEPS,
    ):
        self.horizon = horizon
        self.interval = interval
        self.symbol = symbol
        self.max_steps = max_steps
        self.is_indonesian = is_indonesian_stock(symbol, source="yahoo")
        self.nf: Optional[NeuralForecast] = None

    def _get_exog_features(self) -> List[str]:
        """Get list of exogenous features to use."""
        features = self.BASE_FEATURES.copy()
        if self.is_indonesian:
            features.extend(self.IDX_FEATURES)
        return features

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data in NeuralForecast format.

        Args:
            df: Raw DataFrame with features

        Returns:
            (formatted_df, available_features)
        """
        exog_cols = self._get_exog_features()
        available = [c for c in exog_cols if c in df.columns]

        nf_df = pd.DataFrame({
            "unique_id": "STOCK",
            "ds": df["ds"],
            "y": df["close"]
        })

        for col in available:
            nf_df[col] = df[col].fillna(0)

        # Clean infinities and NaN
        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)

        return nf_df, available

    def _validate_data(self, n: int) -> int:
        """
        Validate data size and adjust horizon if needed.

        Args:
            n: Number of data points

        Returns:
            Adjusted horizon

        Raises:
            InsufficientDataError: If not enough data
        """
        if n < MIN_DATA_POINTS:
            raise InsufficientDataError(n, MIN_DATA_POINTS)

        max_h = n // 2
        if self.horizon > max_h:
            logger.warning(f"  Limiting horizon from {self.horizon} to {max_h}")
            self.horizon = max_h

        return self.horizon

    def _create_models(self, input_size: int, features: List[str]) -> List:
        """Create model instances."""
        return [
            # NBEATS doesn't support exogenous variables
            NBEATS(
                h=self.horizon,
                input_size=input_size,
                max_steps=self.max_steps,
                scaler_type="robust",
                random_seed=RANDOM_SEED,
            ),
            NHITS(
                h=self.horizon,
                input_size=input_size,
                max_steps=self.max_steps,
                scaler_type="robust",
                random_seed=RANDOM_SEED,
                hist_exog_list=features if features else None,
            ),
            LSTM(
                h=self.horizon,
                input_size=input_size,
                max_steps=self.max_steps,
                scaler_type="robust",
                random_seed=RANDOM_SEED,
                hist_exog_list=features if features else None,
            ),
        ]

    def forecast(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, ForecastMeta]:
        """
        Train models and generate forecasts.

        Args:
            df: DataFrame with OHLCV + features

        Returns:
            (forecasts, historical_data, metadata)
        """
        nf_df, features = self._prepare_data(df)
        n = len(nf_df)

        # Print data summary
        self._print_data_summary(nf_df, features)

        # Validate and adjust
        self.horizon = self._validate_data(n)

        # Calculate input size
        input_size = min(n - self.horizon - 1, 30)
        input_size = max(input_size, 5)

        # Print config
        self._print_config(input_size, features)

        # Create and train models
        models = self._create_models(input_size, features)
        freq = VALID_INTERVALS.get(self.interval, IntervalConfig("max", "D")).freq

        logger.info(f"  Frequency: {freq}")
        logger.info(f"\nTraining: {[m.__class__.__name__ for m in models]}")

        self.nf = NeuralForecast(models=models, freq=freq)
        self.nf.fit(df=nf_df)

        logger.info(f"Generating {self.horizon}-step forecast...")
        forecasts = pd.DataFrame(self.nf.predict())

        # Create ensemble
        forecasts = self._create_ensemble(forecasts)

        # Apply ARA/ARB clamping for Indonesian stocks
        close_series: pd.Series = df["close"]  # type: ignore[assignment]
        last_price = float(close_series.iloc[-1])
        if self.is_indonesian:
            forecasts = clamp_forecast_series(forecasts, last_price, price_col="ensemble")

            if "low_clamped" in forecasts.columns:
                forecasts["low"] = forecasts["low_clamped"]
            if "high_clamped" in forecasts.columns:
                forecasts["high"] = forecasts["high_clamped"]

            logger.info("  Forecasts clamped to ARA/ARB limits (Indonesian stock)")

        meta = ForecastMeta(
            horizon=self.horizon,
            interval=self.interval,
            input_size=input_size,
            n_records=n,
            is_indonesian=self.is_indonesian,
            last_price=last_price,
            features=features,
        )

        return forecasts, nf_df, meta

    def _create_ensemble(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble forecast from individual models."""
        model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id"]]

        if model_cols:
            forecasts["ensemble"] = forecasts[model_cols].mean(axis=1)
            forecasts["std"] = forecasts[model_cols].std(axis=1)
            forecasts["low"] = forecasts["ensemble"] - 1.96 * forecasts["std"]
            forecasts["high"] = forecasts["ensemble"] + 1.96 * forecasts["std"]

        return forecasts

    def _print_data_summary(self, nf_df: pd.DataFrame, features: List[str]) -> None:
        """Print data summary."""
        n = len(nf_df)
        print(f"\n{'=' * 60}")
        print("DATA SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Records: {n}")
        print(f"  Period: {nf_df['ds'].min().strftime('%Y-%m-%d')} to {nf_df['ds'].max().strftime('%Y-%m-%d')}")
        print(f"  Price range: {nf_df['y'].min():,.2f} - {nf_df['y'].max():,.2f}")
        print(f"  Last price: {nf_df['y'].iloc[-1]:,.2f}")
        print(f"  Features: {len(features)}")

    def _print_config(self, input_size: int, features: List[str]) -> None:
        """Print forecast configuration."""
        print(f"\n{'=' * 60}")
        print("FORECAST CONFIG")
        print(f"{'=' * 60}")
        print(f"  Horizon: {self.horizon} periods")
        print(f"  Lookback: {input_size} periods")
        print(f"  Interval: {self.interval}")
        print(f"  Exog features: {features}")


# =============================================================================
# Results Printer
# =============================================================================

class ResultsPrinter:
    """Print forecast results in formatted output."""

    @staticmethod
    def print_results(
        forecasts: pd.DataFrame,
        last_price: float,
        symbol: str,
        currency: str,
        meta: Optional[ForecastMeta] = None
    ) -> None:
        """Print detailed forecast results."""
        print(f"\n{'=' * 70}")
        print(f"FORECAST: {symbol}")
        print(f"{'=' * 70}")

        is_indonesian = meta.is_indonesian if meta else symbol.endswith(".JK")

        # Show ARA/ARB limits for Indonesian stocks
        if is_indonesian:
            limit_info = get_daily_limit_info(last_price)
            print(f"\nARA/ARB Limits (Day 1 based on {last_price:,.0f} IDR):")
            print(f"  ARA (Upper): {limit_info['ara_price']:,.0f} (+{limit_info['max_gain_pct']:.1f}%)")
            print(f"  ARB (Lower): {limit_info['arb_price']:,.0f} (-{limit_info['max_loss_pct']:.1f}%)")

        show_clamped = "ensemble_clamped" in forecasts.columns

        # Print header and data rows
        ResultsPrinter._print_table(forecasts, last_price, show_clamped)

        # Print summary
        ResultsPrinter._print_summary(forecasts, last_price, currency, show_clamped)

    @staticmethod
    def _print_table(forecasts: pd.DataFrame, last_price: float, show_clamped: bool) -> None:
        """Print forecast table."""
        print(f"\n{'Date':<12}", end="")
        if show_clamped:
            print(f"{'Forecast':>12}{'Clamped':>12}{'ARA':>10}{'ARB':>10}", end="")
        else:
            print(f"{'Forecast':>12}{'Low':>12}{'High':>12}", end="")
        print(f"{'Change':>10}")
        print("-" * (12 + (44 if show_clamped else 46)))

        for _, row in forecasts.iterrows():
            d = row["ds"]
            if hasattr(d, "strftime"):
                date_str = d.strftime("%Y-%m-%d") if d.hour == 0 else d.strftime("%m-%d %H:%M")  # type: ignore[union-attr]
            else:
                date_str = str(d)[:10]

            print(f"{date_str:<12}", end="")

            if show_clamped:
                ens = row.get("ensemble", 0) or 0
                clamped = row.get("ensemble_clamped", ens) or ens
                ara = row.get("ara_limit", 0) or 0
                arb = row.get("arb_limit", 0) or 0
                pct = (clamped / last_price - 1) * 100  # type: ignore[operator]
                print(f"{ens:>12,.0f}{clamped:>12,.0f}{ara:>10,.0f}{arb:>10,.0f}{pct:>+9.2f}%")
            else:
                ens = row.get("ensemble", 0) or 0
                low = row.get("low", 0) or 0
                high = row.get("high", 0) or 0
                pct = (ens / last_price - 1) * 100  # type: ignore[operator]
                print(f"{ens:>12,.2f}{low:>12,.2f}{high:>12,.2f}{pct:>+9.2f}%")

    @staticmethod
    def _print_summary(
        forecasts: pd.DataFrame,
        last_price: float,
        currency: str,
        show_clamped: bool
    ) -> None:
        """Print forecast summary."""
        if "ensemble" not in forecasts.columns:
            return

        print(f"\n{'=' * 70}")
        print(f"SUMMARY ({currency})")
        print(f"{'=' * 70}")
        print(f"  Current price:   {last_price:>14,.0f}")

        if show_clamped:
            print("\n  Clamped to ARA/ARB limits:")
            fc = forecasts["ensemble_clamped"]
        else:
            fc = forecasts["ensemble"]

        print(f"  Forecast avg:    {fc.mean():>14,.0f} ({(fc.mean()/last_price-1)*100:+.2f}%)")
        print(f"  Forecast high:   {fc.max():>14,.0f} ({(fc.max()/last_price-1)*100:+.2f}%)")
        print(f"  Forecast low:    {fc.min():>14,.0f} ({(fc.min()/last_price-1)*100:+.2f}%)")
        print(f"  Final forecast:  {fc.iloc[-1]:>14,.0f} ({(fc.iloc[-1]/last_price-1)*100:+.2f}%)")

        if "std" in forecasts.columns:
            avg_std = forecasts["std"].mean()
            print(f"  Uncertainty:     {avg_std:>14,.0f} (+/- {avg_std/last_price*100:.2f}%)")

        # Direction
        final = fc.iloc[-1]
        pct_change = (final / last_price - 1) * 100
        outlook = Outlook.from_percent_change(pct_change)
        print(f"\n  Outlook: {outlook.name} ({pct_change:+.2f}%)")


# =============================================================================
# Plotter
# =============================================================================

class ForecastPlotter:
    """Generate forecast visualizations."""

    @staticmethod
    def plot(
        df: pd.DataFrame,
        forecasts: pd.DataFrame,
        symbol: str,
        name: str,
        last_price: float = 0.0
    ) -> Optional[Path]:
        """
        Plot historical prices and forecast.

        Args:
            df: Historical data
            forecasts: Forecast data
            symbol: Stock symbol
            name: Company name
            last_price: Last known price for calculating change %

        Returns:
            Path to saved plot, or None if plotting failed
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                3, 1,
                figsize=(14, 10),
                gridspec_kw={"height_ratios": [3, 1, 1]}
            )

            # Price chart
            ForecastPlotter._plot_price(axes[0], df, forecasts, symbol, name)

            # Volume chart
            ForecastPlotter._plot_volume(axes[1], df)

            # RSI chart
            ForecastPlotter._plot_rsi(axes[2], df)

            # Add forecast table as text annotation
            if last_price > 0:
                ForecastPlotter._add_forecast_table(fig, forecasts, last_price, symbol)

            plt.tight_layout()

            # Adjust layout to make room for the text box
            plt.subplots_adjust(right=0.75)

            # Save plot
            dir_name = symbol.replace(".JK", "")
            plot_dir = PLOT_DIR / dir_name
            plot_dir.mkdir(parents=True, exist_ok=True)

            out_path = plot_dir / f"yf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"\nChart saved: {out_path}")
            plt.close()

            return out_path

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return None

    @staticmethod
    def _add_forecast_table(fig, forecasts: pd.DataFrame, last_price: float, symbol: str) -> None:
        """Add forecast table as text annotation on the right side of the plot."""
        show_clamped = "ensemble_clamped" in forecasts.columns
        is_indonesian = symbol.endswith(".JK")

        lines = []
        lines.append("FORECAST TABLE")
        lines.append("-" * 35)

        # Show ARA/ARB limits for Indonesian stocks
        if is_indonesian:
            limit_info = get_daily_limit_info(last_price)
            lines.append(f"ARA/ARB (Day 1, {last_price:,.0f}):")
            lines.append(f"  ARA: {limit_info['ara_price']:,.0f} (+{limit_info['max_gain_pct']:.1f}%)")
            lines.append(f"  ARB: {limit_info['arb_price']:,.0f} (-{limit_info['max_loss_pct']:.1f}%)")
            lines.append("")

        # Table header
        if show_clamped:
            lines.append(f"{'Date':<10} {'Clamp':>8} {'Chg':>7}")
        else:
            lines.append(f"{'Date':<10} {'Fcst':>8} {'Chg':>7}")
        lines.append("-" * 27)

        # Table rows
        for _, row in forecasts.iterrows():
            d = row["ds"]
            if hasattr(d, "strftime"):
                date_str = d.strftime("%m-%d")
            else:
                date_str = str(d)[5:10]

            if show_clamped:
                val = row.get("ensemble_clamped", row.get("ensemble", 0)) or 0
            else:
                val = row.get("ensemble", 0) or 0

            pct = (val / last_price - 1) * 100
            lines.append(f"{date_str:<10} {val:>8,.0f} {pct:>+6.1f}%")

        # Summary
        lines.append("-" * 27)
        price_col = "ensemble_clamped" if show_clamped else "ensemble"
        if price_col in forecasts.columns:
            fc = forecasts[price_col]
            final = fc.iloc[-1]
            pct_change = (final / last_price - 1) * 100
            outlook = "BULL" if pct_change > 2 else "BEAR" if pct_change < -2 else "NEUT"
            lines.append(f"Final: {final:,.0f} ({pct_change:+.1f}%)")
            lines.append(f"Outlook: {outlook}")

        # Join lines and add to figure
        text = "\n".join(lines)
        fig.text(
            0.78, 0.5, text,
            fontsize=8,
            fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9)
        )

    @staticmethod
    def _plot_price(ax, df: pd.DataFrame, forecasts: pd.DataFrame, symbol: str, name: str) -> None:
        """Plot price chart with forecast."""
        # Historical
        ax.plot(df["ds"], df["close"], "b-", label="Historical", linewidth=1.5)

        # Individual model forecasts
        model_cols = [
            c for c in forecasts.columns
            if c not in ["ds", "unique_id", "ensemble", "std", "low", "high",
                        "ensemble_clamped", "low_clamped", "high_clamped",
                        "ara_limit", "arb_limit"]
        ]
        for col in model_cols:
            ax.plot(forecasts["ds"], forecasts[col], "--", alpha=0.4, linewidth=1, label=col)

        # Ensemble forecast
        if "ensemble" in forecasts.columns:
            price_col = "ensemble_clamped" if "ensemble_clamped" in forecasts.columns else "ensemble"
            label = "Forecast (Clamped)" if "ensemble_clamped" in forecasts.columns else "Ensemble"
            ax.plot(forecasts["ds"], forecasts[price_col], "r-", linewidth=2, label=label)

            if "low" in forecasts.columns:
                ax.fill_between(
                    forecasts["ds"],
                    forecasts["low"],
                    forecasts["high"],
                    alpha=0.2,
                    color="red",
                    label="95% CI"
                )

            # ARA/ARB limits
            if "ara_limit" in forecasts.columns:
                ax.plot(
                    forecasts["ds"], forecasts["ara_limit"],
                    "g--", alpha=0.7, linewidth=1.5, label="ARA (Upper Limit)"
                )
            if "arb_limit" in forecasts.columns:
                ax.plot(
                    forecasts["ds"], forecasts["arb_limit"],
                    "m--", alpha=0.7, linewidth=1.5, label="ARB (Lower Limit)"
                )

        # Moving average
        if "sma_20" in df.columns:
            ax.plot(df["ds"], df["sma_20"], "g--", alpha=0.5, linewidth=1, label="SMA20")

        # Reference lines
        ax.axvline(x=df["ds"].iloc[-1], color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=df["close"].iloc[-1], color="gray", linestyle=":", alpha=0.5)

        ax.set_title(f"{symbol} - {name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_volume(ax, df: pd.DataFrame) -> None:
        """Plot volume chart."""
        colors = [
            "green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red"
            for i in range(len(df))
        ]
        ax.bar(df["ds"], df["volume"], color=colors, alpha=0.7)
        ax.set_ylabel("Volume")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_rsi(ax, df: pd.DataFrame) -> None:
        """Plot RSI chart."""
        if "rsi" in df.columns:
            ax.plot(df["ds"], df["rsi"], "purple", linewidth=1)
            ax.axhline(y=70, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=30, color="green", linestyle="--", alpha=0.5)
            ax.fill_between(df["ds"], 30, 70, alpha=0.1, color="gray")
            ax.set_ylabel("RSI")
            ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Date")


# =============================================================================
# Symbol Processor
# =============================================================================

class SymbolProcessor:
    """Process a single symbol for forecasting."""

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        interval: str = DEFAULT_INTERVAL,
        period: str = DEFAULT_PERIOD,
        plot: bool = True,
    ):
        self.horizon = horizon
        self.interval = interval
        self.period = period
        self.plot = plot

    def process(self, symbol: str) -> ForecastSummary:
        """
        Process a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            ForecastSummary with results

        Raises:
            ForecastError: On any processing error
        """
        loader = YFinanceLoader(symbol)

        print(f"Symbol: {symbol}")
        print(f"Name: {loader.name}")
        print(f"Currency: {loader.currency}")

        # Load data
        df = loader.load(period=self.period, interval=self.interval)

        # Check minimum price
        close_col: pd.Series = df["close"]  # type: ignore[assignment]
        min_price = float(close_col.min())
        if min_price <= MIN_PRICE_THRESHOLD:
            raise LowPriceError(symbol, min_price, MIN_PRICE_THRESHOLD)

        # Add features
        df = loader.add_features(df)
        logger.info(f"Loaded {len(df)} records")

        # Generate forecast
        forecaster = YFinanceForecaster(
            horizon=self.horizon,
            interval=self.interval,
            symbol=symbol
        )
        forecasts, historical, meta = forecaster.forecast(df)

        # Print results
        last_price = float(close_col.iloc[-1])
        ResultsPrinter.print_results(
            forecasts, last_price, symbol, loader.currency, meta
        )

        # Generate plot
        plot_path = None
        if self.plot:
            plot_path = ForecastPlotter.plot(df, forecasts, symbol, loader.name, last_price)

        # Save CSV with Change % column
        csv_path = self._save_csv(forecasts, symbol, last_price)
        logger.info(f"Saved: {csv_path}")

        # Save formatted summary text file
        txt_path = self._save_summary_txt(forecasts, symbol, last_price, loader.currency)
        logger.info(f"Saved: {txt_path}")

        # Calculate summary
        summary_close: pd.Series = df["close"]  # type: ignore[assignment]
        summary_last_price = float(summary_close.iloc[-1])
        price_col = "ensemble_clamped" if "ensemble_clamped" in forecasts.columns else "ensemble"

        if price_col in forecasts.columns:
            fc: pd.Series = forecasts[price_col]  # type: ignore[assignment]
            final_forecast = float(fc.iloc[-1])
            pct_change = (final_forecast / summary_last_price - 1) * 100
            outlook = Outlook.from_percent_change(pct_change)
        else:
            final_forecast = summary_last_price
            pct_change = 0.0
            outlook = Outlook.NEUTRAL

        return ForecastSummary(
            symbol=symbol,
            name=loader.name[:20],  # Truncate long names
            currency=loader.currency,
            current_price=summary_last_price,
            final_forecast=final_forecast,
            change_pct=pct_change,
            outlook=outlook,
            csv_path=csv_path,
            plot_path=plot_path,
        )

    def _save_csv(self, forecasts: pd.DataFrame, symbol: str, last_price: float) -> Path:
        """Save forecasts to CSV with Change % column."""
        dir_name = symbol.replace(".JK", "")
        symbol_csv_dir = CSV_DIR / dir_name
        symbol_csv_dir.mkdir(parents=True, exist_ok=True)

        # Add Change % column
        df_to_save = forecasts.copy()
        price_col = "ensemble_clamped" if "ensemble_clamped" in df_to_save.columns else "ensemble"
        if price_col in df_to_save.columns:
            df_to_save["change_pct"] = (df_to_save[price_col] / last_price - 1) * 100

        out_path = symbol_csv_dir / f"yf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_to_save.to_csv(out_path, index=False)

        return out_path

    def _save_summary_txt(
        self, forecasts: pd.DataFrame, symbol: str, last_price: float, currency: str
    ) -> Path:
        """Save formatted forecast table to text file."""
        dir_name = symbol.replace(".JK", "")
        symbol_csv_dir = CSV_DIR / dir_name
        symbol_csv_dir.mkdir(parents=True, exist_ok=True)

        out_path = symbol_csv_dir / f"yf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.txt"

        lines = []
        lines.append("=" * 70)
        lines.append(f"FORECAST: {symbol}")
        lines.append("=" * 70)

        is_indonesian = symbol.endswith(".JK")
        show_clamped = "ensemble_clamped" in forecasts.columns

        # Show ARA/ARB limits for Indonesian stocks
        if is_indonesian:
            limit_info = get_daily_limit_info(last_price)
            lines.append(f"\nARA/ARB Limits (Day 1 based on {last_price:,.0f} IDR):")
            lines.append(f"  ARA (Upper): {limit_info['ara_price']:,.0f} (+{limit_info['max_gain_pct']:.1f}%)")
            lines.append(f"  ARB (Lower): {limit_info['arb_price']:,.0f} (-{limit_info['max_loss_pct']:.1f}%)")

        # Table header
        if show_clamped:
            lines.append(f"\n{'Date':<12}{'Forecast':>12}{'Clamped':>12}{'ARA':>10}{'ARB':>10}{'Change':>10}")
            lines.append("-" * 66)
        else:
            lines.append(f"\n{'Date':<12}{'Forecast':>12}{'Low':>12}{'High':>12}{'Change':>10}")
            lines.append("-" * 58)

        # Table rows
        for _, row in forecasts.iterrows():
            d = row["ds"]
            if hasattr(d, "strftime"):
                date_str = d.strftime("%Y-%m-%d") if d.hour == 0 else d.strftime("%m-%d %H:%M")
            else:
                date_str = str(d)[:10]

            if show_clamped:
                ens = row.get("ensemble", 0) or 0
                clamped = row.get("ensemble_clamped", ens) or ens
                ara = row.get("ara_limit", 0) or 0
                arb = row.get("arb_limit", 0) or 0
                pct = (clamped / last_price - 1) * 100
                lines.append(f"{date_str:<12}{ens:>12,.0f}{clamped:>12,.0f}{ara:>10,.0f}{arb:>10,.0f}{pct:>+9.2f}%")
            else:
                ens = row.get("ensemble", 0) or 0
                low = row.get("low", 0) or 0
                high = row.get("high", 0) or 0
                pct = (ens / last_price - 1) * 100
                lines.append(f"{date_str:<12}{ens:>12,.2f}{low:>12,.2f}{high:>12,.2f}{pct:>+9.2f}%")

        # Summary
        if "ensemble" in forecasts.columns:
            lines.append(f"\n{'=' * 70}")
            lines.append(f"SUMMARY ({currency})")
            lines.append("=" * 70)
            lines.append(f"  Current price:   {last_price:>14,.0f}")

            if show_clamped:
                lines.append("\n  Clamped to ARA/ARB limits:")
                fc = forecasts["ensemble_clamped"]
            else:
                fc = forecasts["ensemble"]

            lines.append(f"  Forecast avg:    {fc.mean():>14,.0f} ({(fc.mean()/last_price-1)*100:+.2f}%)")
            lines.append(f"  Forecast high:   {fc.max():>14,.0f} ({(fc.max()/last_price-1)*100:+.2f}%)")
            lines.append(f"  Forecast low:    {fc.min():>14,.0f} ({(fc.min()/last_price-1)*100:+.2f}%)")
            lines.append(f"  Final forecast:  {fc.iloc[-1]:>14,.0f} ({(fc.iloc[-1]/last_price-1)*100:+.2f}%)")

            if "std" in forecasts.columns:
                avg_std = forecasts["std"].mean()
                lines.append(f"  Uncertainty:     {avg_std:>14,.0f} (+/- {avg_std/last_price*100:.2f}%)")

            # Direction
            final = fc.iloc[-1]
            pct_change = (final / last_price - 1) * 100
            outlook = Outlook.from_percent_change(pct_change)
            lines.append(f"\n  Outlook: {outlook.name} ({pct_change:+.2f}%)")

        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        return out_path


# =============================================================================
# Batch Processor
# =============================================================================

class BatchProcessor:
    """Process multiple symbols."""

    def __init__(self, processor: SymbolProcessor):
        self.processor = processor

    def process(self, symbols: List[str]) -> BatchResult:
        """
        Process a batch of symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            BatchResult with all outcomes
        """
        result = BatchResult()

        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'#' * 70}")
            print(f"# [{i}/{len(symbols)}] Processing: {symbol}")
            print(f"{'#' * 70}")

            try:
                summary = self.processor.process(symbol)
                result.successful.append(summary)

            except LowPriceError as e:
                print(f"\n  SKIPPING {symbol}: Price touched {e.price:.2f} (threshold: {e.threshold})")
                result.skipped.append({"symbol": symbol, "reason": str(e)})

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                result.errors.append({"symbol": symbol, "error": str(e)})

        return result

    def print_summary(self, result: BatchResult) -> None:
        """Print batch summary."""
        if len(result.successful) + len(result.skipped) + len(result.errors) <= 1:
            return

        print(f"\n{'=' * 100}")
        print("BATCH SUMMARY")
        print(f"{'=' * 100}")
        print(f"  Total: {result.total}")
        print(f"  Success: {len(result.successful)}")
        print(f"  Skipped: {len(result.skipped)}")
        print(f"  Failed: {len(result.errors)}")

        # Forecast summary table
        if result.successful:
            print(f"\n{'=' * 100}")
            print("FORECAST SUMMARY TABLE")
            print(f"{'=' * 100}")
            print(
                f"{'Symbol':<12}{'Name':<22}{'Currency':<10}"
                f"{'Current':>14}{'Forecast':>14}{'Change':>10}{'Outlook':<10}"
            )
            print("-" * 100)

            for s in result.successful:
                print(
                    f"{s.symbol:<12}{s.name:<22}{s.currency:<10}"
                    f"{s.current_price:>14,.0f}{s.final_forecast:>14,.0f}"
                    f"{s.change_pct:>+9.2f}%  {s.outlook.name:<10}"
                )
            print("-" * 100)

        if result.skipped:
            print(f"\n  Skipped (price <= {MIN_PRICE_THRESHOLD}):")
            for s in result.skipped:
                print(f"    - {s['symbol']}: {s['reason']}")

        if result.successful:
            print("\n  Successful:")
            for r in result.successful:
                print(f"    - {r.symbol}: {r.csv_path}")

        if result.errors:
            print("\n  Failed:")
            for e in result.errors:
                print(f"    - {e['symbol']}: {e['error']}")


# =============================================================================
# Watchlist Integration
# =============================================================================

def handle_watchlist_update(args: argparse.Namespace, result: BatchResult) -> None:
    """Handle watchlist update if requested."""
    if not args.watchlist or not result.successful:
        return

    # Convert to outlook format
    symbols_with_outlook = [
        {"symbol": s.symbol, "outlook": s.outlook.name}
        for s in result.successful
    ]

    # Filter based on flags
    symbols_to_add = filter_symbols_by_outlook(
        symbols_with_outlook,
        bullish_only=args.bullish,
        bearish_only=args.bearish,
    )

    if not symbols_to_add:
        filter_type = "bullish" if args.bullish else "bearish" if args.bearish else ""
        print(f"\nNo symbols to add to watchlist (filter: {filter_type})")
        return

    print(f"\n{'=' * 70}")
    print("WATCHLIST UPDATE")
    print(f"{'=' * 70}")

    filter_desc = ""
    if args.bullish:
        filter_desc = " (bullish only)"
    elif args.bearish:
        filter_desc = " (bearish only)"

    print(f"Adding {len(symbols_to_add)} symbols to watchlist{filter_desc}: {', '.join(symbols_to_add)}")

    try:
        wl_result = update_watchlist(
            args.watchlist_id,
            symbols_to_add,
            keep_existing=args.keep,
            debug=getattr(args, 'wl_debug', False),
        )
        print_watchlist_summary(wl_result)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Watchlist update failed: {e}")


# =============================================================================
# Telegram Integration
# =============================================================================

def handle_telegram_notification(args: argparse.Namespace, result: BatchResult) -> None:
    """Handle Telegram notification if requested."""
    if not args.telegram or not result.successful:
        return

    print(f"\n{'=' * 70}")
    print("TELEGRAM NOTIFICATION")
    print(f"{'=' * 70}")

    # For batch processing (multiple symbols), send a summary
    if len(result.successful) > 1:
        tg_results = [
            {
                "symbol": s.symbol,
                "name": s.name,
                "current_price": s.current_price,
                "forecast": s.final_forecast,
                "change_pct": s.change_pct,
                "outlook": s.outlook.name,
            }
            for s in result.successful
        ]

        print(f"Sending batch summary for {len(tg_results)} symbols...")
        success = send_batch_notification(
            results=tg_results,
            script_name="YF Batch Forecast",
            silent=getattr(args, "tg_silent", False),
        )
        if success:
            print("  Telegram notification sent successfully")
    else:
        # For single symbol, send detailed forecast
        s = result.successful[0]

        # We don't have the full forecast data here, so create a simple summary
        tg_forecasts = [{
            "date": "Final",
            "price": s.final_forecast,
            "change_pct": s.change_pct,
        }]

        # Get ARA/ARB info for Indonesian stocks
        ara_arb_info = None
        is_indonesian = s.symbol.endswith(".JK")
        if is_indonesian:
            limit_info = get_daily_limit_info(s.current_price)
            ara_arb_info = {
                "ara_price": limit_info["ara_price"],
                "arb_price": limit_info["arb_price"],
                "max_gain_pct": limit_info["max_gain_pct"],
                "max_loss_pct": limit_info["max_loss_pct"],
            }

        print(f"Sending notification for {s.symbol}...")
        success = send_forecast_notification(
            symbol=s.symbol,
            current_price=s.current_price,
            forecasts=tg_forecasts,
            outlook=s.outlook.name,
            change_pct=s.change_pct,
            currency=s.currency,
            ara_arb_info=ara_arb_info,
            script_name="YF Forecast",
            silent=getattr(args, "tg_silent", False),
        )
        if success:
            print("  Telegram notification sent successfully")


# =============================================================================
# CLI Argument Parser
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Stock Forecasting with Yahoo Finance Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Intervals:
  1m, 2m, 5m, 15m, 30m    Intraday (limited history)
  1h                       Hourly (up to 730 days)
  1d                       Daily (default, max history)
  1wk, 1mo                 Weekly/Monthly

Periods:
  1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

Examples:
    python yf_optimized.py AAPL                          # Apple, daily, 5-day forecast
    python yf_optimized.py BBRI.JK --horizon 10          # Indonesian bank stock
    python yf_optimized.py TSLA --interval 1h --period 5d --horizon 24
    python yf_optimized.py GOOGL --period 1y --plot      # 1 year history with chart
    python yf_optimized.py ^GSPC --horizon 10            # S&P 500 index
    python yf_optimized.py BBYB.JK,BBCA.JK,BMRI.JK       # Multiple symbols
        """
    )

    parser.add_argument(
        "symbol",
        nargs="?",
        help="Stock symbol(s) - single (AAPL) or comma-separated (AAPL,GOOGL,TSLA)"
    )
    parser.add_argument(
        "--horizon", "-n",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Forecast horizon (default: {DEFAULT_HORIZON})"
    )
    parser.add_argument(
        "--interval", "-i",
        default=DEFAULT_INTERVAL,
        help=f"Data interval (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--period", "-P",
        default=DEFAULT_PERIOD,
        help=f"History period (default: {DEFAULT_PERIOD})"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable chart generation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Add watchlist arguments
    add_watchlist_args(parser)

    # Add telegram arguments
    add_telegram_args(parser)

    return parser


def parse_symbols(symbol_arg: str) -> List[str]:
    """Parse and normalize symbol argument."""
    symbols = [s.strip().upper() for s in symbol_arg.split(",") if s.strip()]
    if not symbols:
        raise ValueError("No valid symbols provided")
    return symbols


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate watchlist arguments
    if hasattr(args, 'watchlist') and args.watchlist:
        try:
            validate_watchlist_args(args)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Validate telegram arguments
    if hasattr(args, 'telegram') and args.telegram:
        try:
            validate_telegram_args(args)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    if not args.symbol:
        parser.print_help()
        return 1

    # Parse symbols
    try:
        symbols = parse_symbols(args.symbol)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\n{'=' * 70}")
    print("YAHOO FINANCE FORECASTER")
    print(f"{'=' * 70}")
    print(f"Symbols to process: {', '.join(symbols)}")

    # Create processors
    symbol_processor = SymbolProcessor(
        horizon=args.horizon,
        interval=args.interval,
        period=args.period,
        plot=not args.no_plot,
    )
    batch_processor = BatchProcessor(symbol_processor)

    # Process symbols
    result = batch_processor.process(symbols)

    # Print batch summary
    batch_processor.print_summary(result)

    # Handle watchlist update
    handle_watchlist_update(args, result)

    # Handle Telegram notification
    handle_telegram_notification(args, result)

    # Return appropriate exit code
    return 1 if result.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
