#!/usr/bin/env python3
"""
Intraday Stock Forecasting - Single Session (Production Grade)

Forecasts next trading session based on tick data from running-trade.json.
Supports incremental training, model persistence, and group training.

Usage:
    python short.py SYMBOL --hours 3          # Forecast next 3 hours
    python short.py SYMBOL --session1         # Forecast session 1 (09:00-12:00 Mon-Thu)
    python short.py SYMBOL --session2         # Forecast session 2 (13:30-15:50 Mon-Thu)

Example:
    python short.py ICBP --session1 --plot    # Forecast tomorrow's session 1
    python short.py ICBP --hours 3 --interval 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, HuberLoss
from neuralforecast.models import LSTM, NBEATS, NHITS

# Shared utilities
from modules import (
    MODELS_DIR,
    CSV_DIR,
    PLOT_DIR,
    SOURCES_DIR,
    GROUPS_FILE,
    setup_logging,
    load_groups,
    find_group_for_symbol,
)
from modules.idx_rules import calculate_ara_arb
from modules.watchlist import (
    add_watchlist_args,
    filter_symbols_by_outlook,
    print_watchlist_summary,
    update_watchlist,
    validate_watchlist_args,
)

# Suppress warnings after imports
warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================

# Model configuration
DEFAULT_INTERVAL: Final[int] = 5  # minutes
DEFAULT_SESSION: Final[str] = "1"
MIN_BARS_REQUIRED: Final[int] = 5
MAX_TRAINING_STEPS: Final[int] = 200
FINE_TUNE_STEPS: Final[int] = 50
RANDOM_SEED: Final[int] = 42

# Logger
logger = setup_logging("short_forecast")


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
    """Not enough data points for forecasting."""
    def __init__(self, available: int, required: int = MIN_BARS_REQUIRED):
        super().__init__(f"Need at least {required} bars (have {available})")
        self.available = available
        self.required = required


class SymbolNotFoundError(DataError):
    """Requested symbol not found."""
    def __init__(self, symbol: str, available: List[str]):
        super().__init__(
            f"Symbol '{symbol}' not found. Available: {', '.join(sorted(available))}"
        )
        self.symbol = symbol
        self.available = available


class SessionNotFoundError(DataError):
    """Requested session not found."""
    def __init__(self, session: str, available: List[str]):
        super().__init__(
            f"Session '{session}' not found. Available: {', '.join(available)}"
        )
        self.session = session
        self.available = available


class GroupNotFoundError(DataError):
    """Requested group not found."""
    def __init__(self, group: str, available: List[str]):
        super().__init__(
            f"Group '{group}' not found. Available: {', '.join(available)}"
        )
        self.group = group
        self.available = available


# =============================================================================
# Data Classes
# =============================================================================

class Outlook(Enum):
    """Forecast outlook classification."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()

    @classmethod
    def from_price_change(cls, current: float, forecast: float) -> "Outlook":
        """Determine outlook from price change."""
        if forecast > current:
            return cls.BULLISH
        elif forecast < current:
            return cls.BEARISH
        return cls.NEUTRAL


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for a trading session."""
    start: str
    end: str
    hours: float


@dataclass
class ARAARBLimits:
    """ARA/ARB price limits for a stock."""
    prev_close: float
    ara: float
    arb: float

    @property
    def ara_pct(self) -> float:
        return (self.ara / self.prev_close - 1) * 100

    @property
    def arb_pct(self) -> float:
        return (1 - self.arb / self.prev_close) * 100


# IDX Market Hours (WIB/GMT+7)
# Source: https://www.idx.co.id/id/produk/mekanisme-dan-jam-perdagangan
# Based on: SK Direksi BEI Nomor II-A Kep-00003/BEI/04-2025
MARKET_SESSIONS: Final[Dict[str, SessionConfig]] = {
    "session1": SessionConfig("09:00", "12:00", 3.0),           # Morning (Mon-Thu)
    "session1_fri": SessionConfig("09:00", "11:30", 2.5),       # Morning (Friday)
    "session2": SessionConfig("13:30", "15:50", 2.33),          # Afternoon (Mon-Thu)
    "session2_fri": SessionConfig("14:00", "15:50", 1.83),      # Afternoon (Friday)
    "pre_open": SessionConfig("08:45", "09:00", 0.25),          # Pre-opening
    "pre_close": SessionConfig("15:50", "16:00", 0.17),         # Pre-closing
    "post_trade": SessionConfig("16:02", "16:15", 0.22),        # Post-trading
}


# =============================================================================
# Utility Functions
# =============================================================================

def _require_non_empty(df: pd.DataFrame, message: str) -> None:
    """Validate that DataFrame is not empty."""
    if df.empty:
        raise DataError(message)


def _safe_last(series: pd.Series, default: Any = None) -> Any:
    """Safely get last value from series."""
    if series.empty:
        if default is None:
            raise DataError("Cannot read last value from empty series")
        return default
    return series.iloc[-1]


class TickToOHLCV:
    """Convert tick-level trade data to OHLCV bars."""

    # Feature columns for forecasting
    EXOG_COLUMNS: List[str] = [
        "volume", "volatility", "buy_pressure", "foreign_net", "momentum", "vol_ratio"
    ]

    def __init__(self, base_path: Union[str, Path] = SOURCES_DIR):
        self.base_path = Path(base_path)

    def _load_json(self, filepath: Path) -> Optional[Dict]:
        """Load JSON file with error handling."""
        try:
            with filepath.open("r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.debug(f"File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {filepath}: {e}")
            return None

    @staticmethod
    def _parse_number_series(series: pd.Series) -> pd.Series:
        """Vectorized parsing of number strings (from Gemini)."""
        result = pd.to_numeric(
            series.astype(str).str.replace(",", ""), errors="coerce"
        )
        return pd.Series(result).fillna(0.0)

    def _get_session_date(self, session_path: Path) -> datetime:
        """Extract date from analyzed.json file."""
        analysis = self._load_json(session_path / "analyzed.json")
        if analysis:
            time_horizons = analysis.get("metadata", {}).get("time_horizons", {})
            md_to = time_horizons.get("market_detector", {}).get("to")
            if md_to:
                try:
                    return datetime.strptime(md_to, "%Y-%m-%d")
                except ValueError:
                    pass
        return datetime.now()

    def _parse_trades_vectorized(
        self, trades: List[Dict], base_date: datetime
    ) -> pd.DataFrame:
        """
        Vectorized trade parsing - bulk DataFrame operations instead of loops.
        (Adapted from Gemini's approach)
        """
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # Check required columns
        required_cols = ["price", "lot", "time"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        # Vectorized type conversion
        price_series: pd.Series = df["price"]  # type: ignore[assignment]
        lot_series: pd.Series = df["lot"]  # type: ignore[assignment]
        df["price"] = self._parse_number_series(price_series)
        df["lot"] = self._parse_number_series(lot_series)

        # Filter invalid trades
        df = df[(df["price"] > 0) & (df["lot"] > 0)]
        if df.empty:
            return pd.DataFrame()

        # Vectorized time parsing
        time_col: pd.Series = df["time"]  # type: ignore[assignment]
        time_split = time_col.astype(str).str.split(":", expand=True)

        # Build timestamps from components (faster than string parsing)
        try:
            timestamps = pd.to_datetime(pd.DataFrame({
                "year": base_date.year,
                "month": base_date.month,
                "day": base_date.day,
                "hour": time_split[0].astype(int),
                "minute": time_split[1].astype(int),
                "second": time_split[2].astype(int) if time_split.shape[1] > 2 else 0,
            }))
        except (ValueError, KeyError) as e:
            logger.warning(f"Time parsing error: {e}")
            return pd.DataFrame()

        df["timestamp"] = timestamps
        df["value"] = df["price"] * df["lot"] * 100

        # Use category dtype for string columns (memory optimization from Gemini)
        for col in ["action", "buyer_type", "seller_type"]:
            if col in df.columns:
                col_series: pd.Series = df[col]  # type: ignore[assignment]
                df[col] = col_series.fillna("").astype("category")
            else:
                df[col] = pd.Categorical([""] * len(df))

        return df.sort_values("timestamp").reset_index(drop=True)  # type: ignore[return-value]

    def _aggregate_bars(self, tick_df: pd.DataFrame, interval: int) -> pd.DataFrame:
        """
        Aggregate ticks to OHLCV bars using single-pass aggregation.
        (Adapted from Gemini's approach)
        """
        if tick_df.empty:
            return pd.DataFrame()

        df = tick_df.set_index("timestamp")
        freq = f"{interval}min"

        # Pre-calculate masks for volume breakdown (avoids repeated filtering)
        is_buy = df["action"].astype(str) == "buy"
        is_foreign_buy = df["buyer_type"].astype(str).str.contains("FOREIGN", na=False, case=False)
        is_foreign_sell = df["seller_type"].astype(str).str.contains("FOREIGN", na=False, case=False)

        # Pre-calculate volume columns
        df = df.assign(
            buy_vol=df["lot"].where(is_buy, 0),
            sell_vol=df["lot"].where(~is_buy, 0),
            f_buy=df["lot"].where(is_foreign_buy, 0),
            f_sell=df["lot"].where(is_foreign_sell, 0),
        )

        # Single-pass aggregation (from Gemini - more efficient)
        ohlcv = df.resample(freq).agg({
            "price": ["first", "max", "min", "last", "count"],
            "lot": "sum",
            "value": "sum",
            "buy_vol": "sum",
            "sell_vol": "sum",
            "f_buy": "sum",
            "f_sell": "sum",
        })

        # Flatten MultiIndex columns
        ohlcv.columns = [
            "open", "high", "low", "close", "trades",
            "volume", "value", "buy_vol", "sell_vol", "foreign_buy", "foreign_sell"
        ]

        # Drop empty bars
        ohlcv = ohlcv.dropna(subset=["close"])  # type: ignore[arg-type]
        if ohlcv.empty:
            return pd.DataFrame()

        # Vectorized feature engineering
        ohlcv["returns"] = ohlcv["close"].pct_change().fillna(0)
        ohlcv["volatility"] = (ohlcv["high"] - ohlcv["low"]) / (ohlcv["close"] + 1e-8)
        ohlcv["spread"] = ohlcv["high"] - ohlcv["low"]
        ohlcv["buy_pressure"] = ohlcv["buy_vol"] / (ohlcv["volume"] + 1)
        ohlcv["foreign_net"] = ohlcv["foreign_buy"] - ohlcv["foreign_sell"]
        ohlcv["vwap"] = ohlcv["value"] / (ohlcv["volume"] * 100 + 1)
        ohlcv["momentum"] = ohlcv["close"].pct_change(3).fillna(0)

        vol_ma = ohlcv["volume"].rolling(3, min_periods=1).mean()
        ohlcv["vol_ratio"] = ohlcv["volume"] / (vol_ma + 1)

        return ohlcv.reset_index()

    def load_session(
        self, symbol: str, session: str, interval: int = DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """Load and process a single session from today-running-trade.json."""
        session_path = self.base_path / symbol / session
        trt_path = session_path / "today-running-trade.json"

        if not trt_path.exists():
            return pd.DataFrame()

        base_date = self._get_session_date(session_path)

        trt = self._load_json(trt_path)
        if trt and "data" in trt:
            trades = trt["data"].get("running_trade", [])
            if trades:
                tick_df = self._parse_trades_vectorized(trades, base_date)
                if not tick_df.empty:
                    return self._aggregate_bars(tick_df, interval)

        return pd.DataFrame()

    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def get_sessions(self, symbol: str) -> List[str]:
        """Get list of available sessions for a symbol."""
        path = self.base_path / symbol
        if not path.exists():
            return []
        sessions = [d.name for d in path.iterdir() if d.is_dir()]
        return sorted(sessions, key=lambda x: int(x) if x.isdigit() else 0)

    def load_group_session(
        self, symbols: List[str], session: str, interval: int = DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """Load session data for multiple symbols (for group training)."""
        all_data = []
        available = set(self.get_symbols())

        for symbol in symbols:
            if symbol not in available:
                logger.debug(f"Symbol {symbol} not available")
                continue

            sessions = self.get_sessions(symbol)
            if session not in sessions:
                logger.debug(f"Session {session} not found for {symbol}")
                continue

            try:
                df = self.load_session(symbol, session, interval)
                if not df.empty:
                    df["unique_id"] = symbol
                    all_data.append(df)
                    logger.info(f"  {symbol}: {len(df)} bars")
            except Exception as e:
                logger.warning(f"  {symbol}: Error - {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        return combined.sort_values(["unique_id", "timestamp"]).reset_index(drop=True)


class SessionForecaster:
    """Forecast next trading session using neural network ensemble."""

    # Additional ARA/ARB proximity features
    ARA_ARB_FEATURES: List[str] = [
        "ara_proximity", "arb_proximity", "pct_to_ara", "pct_to_arb"
    ]

    def __init__(
        self,
        interval: int = DEFAULT_INTERVAL,
        symbol: str = "STOCK",
        group: Optional[str] = None
    ):
        self.interval = interval
        self.symbol = symbol
        self.group = group
        self.nf: Optional[NeuralForecast] = None
        self.limits: Optional[ARAARBLimits] = None

        # Model paths
        model_name = (
            f"short_group_{group}_{interval}min"
            if group else f"short_{symbol}_{interval}min"
        )
        self.model_path = MODELS_DIR / model_name
        self.meta_path = MODELS_DIR / f"{model_name}_meta.json"

    def save_model(self, data_count: int, session_date: str):
        """Save trained model and metadata to disk."""
        if self.nf is None:
            return

        # Save the full NeuralForecast model
        self.nf.save(str(self.model_path), overwrite=True)

        # Also save individual model state_dicts for checkpoint resume
        ckpt_dir = self.model_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        import torch
        for i, model in enumerate(self.nf.models):
            model_name = model.__class__.__name__
            ckpt_path = ckpt_dir / f"{model_name}_{i}.pt"
            torch.save(model.state_dict(), ckpt_path)

        meta = {
            "symbol": self.symbol,
            "group": self.group,
            "interval": self.interval,
            "data_count": data_count,
            "session_date": session_date,
            "saved_at": datetime.now().isoformat(),
            "model_names": [m.__class__.__name__ for m in self.nf.models],
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        model_type = f"group '{self.group}'" if self.group else self.symbol
        print(f"  Model saved: {self.model_path} ({model_type})")

    def load_model(self) -> Optional[dict]:
        """Load saved model and metadata if available."""
        if not self.model_path.exists() or not self.meta_path.exists():
            return None

        try:
            with open(self.meta_path, "r") as f:
                meta = json.load(f)

            self.nf = NeuralForecast.load(str(self.model_path))
            print(f"  Loaded saved model (trained on {meta['data_count']} bars from {meta.get('session_date', 'unknown')})")
            return meta
        except Exception as e:
            print(f"  Could not load saved model: {e}")
            return None

    def load_checkpoints_into_models(self, models: list) -> list:
        """Load saved checkpoints into new model instances for warm-start training."""
        ckpt_dir = self.model_path / "checkpoints"
        if not ckpt_dir.exists():
            return models

        import torch
        loaded_count = 0

        for i, model in enumerate(models):
            model_name = model.__class__.__name__
            ckpt_path = ckpt_dir / f"{model_name}_{i}.pt"

            if ckpt_path.exists():
                try:
                    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    model.load_state_dict(state_dict, strict=False)
                    loaded_count += 1
                except Exception as e:
                    print(f"    Warning: Could not load checkpoint for {model_name}: {e}")

        if loaded_count > 0:
            print(f"  Loaded {loaded_count}/{len(models)} model checkpoints (warm-start)")

        return models

    def calculate_horizon(self, hours: float) -> int:
        """Calculate number of bars for given hours."""
        return int(hours * 60 / self.interval)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], float]:
        """Prepare NeuralForecast format with gap handling and ARA/ARB tracking."""
        exog_cols = [
            "volume",
            "volatility",
            "buy_pressure",
            "foreign_net",
            "momentum",
            "vol_ratio",
        ]
        available = [c for c in exog_cols if c in df.columns]

        # Use existing unique_id if available (group training), otherwise use "STOCK"
        if "unique_id" in df.columns:
            unique_ids = df["unique_id"]
        else:
            unique_ids = "STOCK"

        nf_df = pd.DataFrame(
            {"unique_id": unique_ids, "ds": df["timestamp"], "y": df["close"]}
        )

        for col in available:
            nf_df[col] = df[col].fillna(0)

        # Get previous day's close for ARA/ARB calculation
        # For intraday, we use the first bar's open as reference (previous close)
        prev_close = df["open"].iloc[0] if "open" in df.columns else df["close"].iloc[0]
        self.prev_close = prev_close  # Store for later use

        # Calculate and store ARA/ARB limits
        ara, arb = calculate_ara_arb(prev_close)
        self.ara_limit = ara
        self.arb_limit = arb

        # Add ARA/ARB proximity features for each bar
        if "close" in df.columns and "high" in df.columns and "low" in df.columns:
            # Calculate proximity to limits
            nf_df["ara_proximity"] = np.clip(
                (df["high"] - prev_close) / (ara - prev_close + 1e-8), 0, 1
            )
            nf_df["arb_proximity"] = np.clip(
                (prev_close - df["low"]) / (prev_close - arb + 1e-8), 0, 1
            )
            nf_df["pct_to_ara"] = (ara - df["close"]) / (df["close"] + 1e-8)
            nf_df["pct_to_arb"] = (df["close"] - arb) / (df["close"] + 1e-8)

            # Check if any bar hit ARA/ARB
            nf_df["ara_hit"] = (df["high"] >= ara * 0.999).astype(int)
            nf_df["arb_hit"] = (df["low"] <= arb * 1.001).astype(int)

            available.extend(["ara_proximity", "arb_proximity", "pct_to_ara", "pct_to_arb"])

            # Report hits
            ara_hits = nf_df["ara_hit"].sum()
            arb_hits = nf_df["arb_hit"].sum()
            if ara_hits > 0 or arb_hits > 0:
                print(f"  ARA/ARB hits detected: {int(ara_hits)} ARA, {int(arb_hits)} ARB")

        # Handle gaps in intraday time series
        nf_df = self._fill_intraday_gaps(nf_df)

        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)
        return nf_df, available, prev_close

    def _fill_intraday_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps in intraday time series using reindex (faster than merge).
        Adapted from Gemini's approach.
        """
        df = df.sort_values(["unique_id", "ds"])
        result_dfs = []

        for uid, group in df.groupby("unique_id"):
            if len(group) < 2:
                group = group.copy()
                group["available_mask"] = 1.0
                result_dfs.append(group)
                continue

            # Create complete time range using reindex (Gemini's approach)
            full_idx = pd.date_range(
                start=group["ds"].min(),
                end=group["ds"].max(),
                freq=f"{self.interval}min",
            )

            # Reindex fills gaps with NaN (cleaner than merge)
            filled = group.set_index("ds").reindex(full_idx)
            filled["unique_id"] = uid

            # Mark gaps before filling
            filled["available_mask"] = filled["y"].notna().astype(float)

            # Interpolate or ffill/bfill
            filled["y"] = filled["y"].ffill().bfill()

            # Fill remaining numeric columns
            numeric_cols = filled.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if col not in ["available_mask"]:
                    filled[col] = filled[col].ffill().bfill().fillna(0)

            result_dfs.append(filled.reset_index().rename(columns={"index": "ds"}))

        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)
            gap_count = int((result["available_mask"] == 0).sum())
            if gap_count > 0:
                logger.info(f"  Filled {gap_count} gaps (marked with available_mask=0)")
            return result

        return df

    def forecast(
        self, df: pd.DataFrame, hours: float, target_session: Optional[str] = None, force_retrain: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Train and forecast for specified hours. Supports incremental training."""
        nf_df, exog, prev_close = self.prepare_data(df)
        n = len(nf_df)

        horizon = self.calculate_horizon(hours)

        # Store limits for later access
        self.limits = ARAARBLimits(prev_close=prev_close, ara=self.ara_limit, arb=self.arb_limit)

        print(f"\n{'=' * 60}")
        print("INPUT DATA")
        print(f"{'=' * 60}")
        print(f"  Historical bars: {n}")
        print(
            f"  Time range: {nf_df['ds'].min().strftime('%H:%M')} - {nf_df['ds'].max().strftime('%H:%M')}"
        )
        print(f"  Price range: {nf_df['y'].min():,.2f} - {nf_df['y'].max():,.2f}")
        print(f"  Last price: {nf_df['y'].iloc[-1]:,.2f}")

        # Show ARA/ARB limits
        print(f"\n  ARA/ARB Limits (based on prev close {prev_close:,.0f}):")
        print(f"    ARA (Upper): {self.limits.ara:,.0f} (+{self.limits.ara_pct:.1f}%)")
        print(f"    ARB (Lower): {self.limits.arb:,.0f} (-{self.limits.arb_pct:.1f}%)")

        print(f"\n{'=' * 60}")
        print("FORECAST CONFIG")
        print(f"{'=' * 60}")
        print(f"  Forecast duration: {hours} hours")
        print(f"  Bar interval: {self.interval} min")
        print(f"  Forecast bars: {horizon}")

        if target_session:
            sess = MARKET_SESSIONS.get(target_session)
            if sess:
                print(f"  Target session: {target_session.upper()} ({sess.start} - {sess.end})")

        if n < MIN_BARS_REQUIRED:
            raise InsufficientDataError(n, MIN_BARS_REQUIRED)

        # Adjust horizon if very limited data
        max_h = n * 2  # Allow forecasting up to 2x the input data
        if horizon > max_h:
            logger.warning(f"  Limiting horizon from {horizon} to {max_h} bars")
            horizon = max_h

        input_size = min(n - 1, 12)
        input_size = max(input_size, 2)

        print(f"  Lookback window: {input_size} bars")

        # Check for saved model and determine training strategy
        session_date = nf_df['ds'].iloc[0].strftime('%Y-%m-%d') if hasattr(nf_df['ds'].iloc[0], 'strftime') else str(nf_df['ds'].iloc[0])[:10]
        saved_meta = None if force_retrain else self.load_model()
        use_fine_tuning = False
        max_steps = MAX_TRAINING_STEPS

        if saved_meta is not None:
            prev_count = saved_meta.get("data_count", 0)
            new_data = n - prev_count

            if new_data > 0:
                print(f"\n  New data detected: {new_data} bars ({prev_count} -> {n})")
                print("  Strategy: Fine-tuning with reduced steps")
                use_fine_tuning = True
                max_steps = FINE_TUNE_STEPS
            elif new_data == 0:
                print("\n  No new data. Using saved model directly.")
                logger.info(f"Generating {horizon} bars ({hours} hours) using cached model...")

                # Predict only for requested symbol if using group training
                assert self.nf is not None, "Model should be loaded"
                if self.group and self.symbol:
                    symbol_df = nf_df[nf_df["unique_id"] == self.symbol].copy()
                    if not symbol_df.empty:
                        forecasts = pd.DataFrame(self.nf.predict(df=symbol_df))
                        ds_col: pd.Series = symbol_df["ds"]  # type: ignore[assignment]
                        last_date = ds_col.iloc[-1]
                    else:
                        forecasts = pd.DataFrame(self.nf.predict())
                        ds_col = nf_df["ds"]  # type: ignore[assignment]
                        last_date = ds_col.iloc[-1]
                else:
                    forecasts = pd.DataFrame(self.nf.predict())
                    ds_col = nf_df["ds"]  # type: ignore[assignment]
                    last_date = ds_col.iloc[-1]

                # Skip to timestamp generation below
                next_day = last_date + timedelta(days=1)  # type: ignore[operator]
                if target_session:
                    sess = MARKET_SESSIONS.get(target_session)
                    if sess:
                        h, m = map(int, sess.start.split(":"))
                        start_dt = next_day.replace(hour=h, minute=m, second=0)
                    else:
                        start_dt = next_day.replace(hour=9, minute=0, second=0)
                else:
                    start_dt = next_day.replace(hour=9, minute=0, second=0)
                future_times = [start_dt + timedelta(minutes=i * self.interval) for i in range(len(forecasts))]
                forecasts["ds"] = future_times
                model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id"]]
                if model_cols:
                    forecasts["ensemble"] = forecasts[model_cols].mean(axis=1)
                    forecasts["std"] = forecasts[model_cols].std(axis=1)
                    forecasts["low"] = forecasts["ensemble"] - 1.96 * forecasts["std"]
                    forecasts["high"] = forecasts["ensemble"] + 1.96 * forecasts["std"]
                meta = {
                    "horizon": horizon, "hours": hours, "interval": self.interval,
                    "target_session": target_session, "forecast_start": start_dt,
                    "forecast_end": future_times[-1] if future_times else start_dt,
                }
                return forecasts, nf_df, meta
            else:
                print("\n  Data reduced. Retraining from scratch.")
                saved_meta = None

        if saved_meta is None:
            print("\n  Strategy: Training from scratch (no saved model)")

        # Use HuberLoss for robustness to outliers (price gaps, big moves)
        # Use DistributionLoss with StudentT for probabilistic forecasts
        models = [
            # NBEATS with HuberLoss - robust to outliers
            NBEATS(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type="robust",
                loss=HuberLoss(),  # type: ignore[arg-type]
                random_seed=RANDOM_SEED,
            ),
            # NHITS with StudentT distribution - probabilistic forecasts with heavy tails
            NHITS(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type="robust",
                loss=DistributionLoss(distribution="StudentT", level=[80, 90]),  # type: ignore[arg-type]
                random_seed=RANDOM_SEED,
            ),
            # LSTM with HuberLoss - robust sequential model
            LSTM(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type="robust",
                loss=HuberLoss(),  # type: ignore[arg-type]
                random_seed=RANDOM_SEED,
            ),
        ]

        # For fine-tuning: load saved weights into new models (warm-start)
        if use_fine_tuning:
            models = self.load_checkpoints_into_models(models)

        freq = f"{self.interval}min"
        print(f"\nTraining: {[m.__class__.__name__ for m in models]}")
        print(f"Max steps: {max_steps}" + (" (warm-start fine-tuning)" if use_fine_tuning else " (full training)"))

        self.nf = NeuralForecast(models=models, freq=freq)
        self.nf.fit(df=nf_df)

        # Save model for future use
        self.save_model(n, session_date)

        # Predict - only for requested symbol if using group training
        print(f"Forecasting {horizon} bars ({hours} hours)...")
        assert self.nf is not None, "Model should be fitted"
        if self.group and self.symbol:
            symbol_df = nf_df[nf_df["unique_id"] == self.symbol].copy()
            if not symbol_df.empty:
                forecasts = pd.DataFrame(self.nf.predict(df=symbol_df))
                print(f"  Predicting for {self.symbol} only")
            else:
                print(f"  Warning: {self.symbol} not found in group data, predicting all")
                forecasts = pd.DataFrame(self.nf.predict())
        else:
            forecasts = pd.DataFrame(self.nf.predict())

        # Generate future timestamps starting from next trading day 09:00
        # Get the last date for the target symbol
        if self.group and self.symbol and "unique_id" in nf_df.columns:
            symbol_data = nf_df[nf_df["unique_id"] == self.symbol]
            ds_series: pd.Series = symbol_data["ds"] if not symbol_data.empty else nf_df["ds"]  # type: ignore[assignment]
            last_date = ds_series.iloc[-1]
        else:
            ds_series = nf_df["ds"]  # type: ignore[assignment]
            last_date = ds_series.iloc[-1]
        next_day = last_date + timedelta(days=1)  # type: ignore[operator]

        if target_session:
            sess = MARKET_SESSIONS.get(target_session)
            if sess:
                h, m = map(int, sess.start.split(":"))
                start_dt = next_day.replace(hour=h, minute=m, second=0)
            else:
                start_dt = next_day.replace(hour=9, minute=0, second=0)
        else:
            start_dt = next_day.replace(hour=9, minute=0, second=0)

        future_times = [
            start_dt + timedelta(minutes=i * self.interval)
            for i in range(len(forecasts))
        ]
        forecasts["ds"] = future_times

        # Ensemble
        model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id"]]
        if model_cols:
            forecasts["ensemble"] = forecasts[model_cols].mean(axis=1)
            forecasts["std"] = forecasts[model_cols].std(axis=1)
            forecasts["low"] = forecasts["ensemble"] - 1.96 * forecasts["std"]
            forecasts["high"] = forecasts["ensemble"] + 1.96 * forecasts["std"]

            # Clamp forecasts to ARA/ARB limits (for intraday, same limits apply all day)
            forecasts["ensemble_clamped"] = forecasts["ensemble"].clip(
                lower=self.arb_limit, upper=self.ara_limit
            )
            forecasts["low"] = forecasts["low"].clip(lower=self.arb_limit)
            forecasts["high"] = forecasts["high"].clip(upper=self.ara_limit)
            forecasts["ara_limit"] = self.ara_limit
            forecasts["arb_limit"] = self.arb_limit

            print(f"\n  Forecasts clamped to ARA/ARB: {self.arb_limit:,.0f} - {self.ara_limit:,.0f}")

        meta = {
            "horizon": horizon,
            "hours": hours,
            "interval": self.interval,
            "target_session": target_session,
            "forecast_start": start_dt,
            "forecast_end": future_times[-1] if future_times else start_dt,
            "ara_limit": self.ara_limit,
            "arb_limit": self.arb_limit,
            "prev_close": self.prev_close,
        }

        return forecasts, nf_df, meta


def print_results(forecasts: pd.DataFrame, last_price: float, symbol: str, meta: dict) -> None:
    """Print forecast results with ARA/ARB limits."""
    print(f"\n{'=' * 70}")
    print(f"FORECAST: {symbol}")
    if meta.get("target_session"):
        sess = MARKET_SESSIONS.get(meta["target_session"])
        if sess:
            print(f"Target: {meta['target_session'].upper()} ({sess.start}-{sess.end})")
    print(
        f"Period: {meta['forecast_start'].strftime('%Y-%m-%d %H:%M')} to {meta['forecast_end'].strftime('%H:%M')}"
    )

    # Show ARA/ARB limits
    if "ara_limit" in meta:
        prev_close = meta.get("prev_close", last_price)
        print(f"\nARA/ARB Limits (based on prev close {prev_close:,.0f}):")
        print(f"  ARA: {meta['ara_limit']:,.0f} (+{(meta['ara_limit']/prev_close-1)*100:.1f}%)")
        print(f"  ARB: {meta['arb_limit']:,.0f} (-{(1-meta['arb_limit']/prev_close)*100:.1f}%)")

    print(f"{'=' * 70}")

    # Show summary at key intervals
    show_clamped = "ensemble_clamped" in forecasts.columns

    # Show every 15 minutes or so
    step = max(1, len(forecasts) // 12)

    if show_clamped:
        print(f"\n{'Time':<10} {'Forecast':>12} {'Clamped':>12} {'Low':>10} {'High':>10} {'Change':>10}")
        print("-" * 66)
    else:
        print(f"\n{'Time':<10} {'Forecast':>12} {'Low':>12} {'High':>12} {'Change':>10}")
        print("-" * 58)

    indices = list(range(0, len(forecasts), step))
    if len(forecasts) - 1 not in indices:
        indices.append(len(forecasts) - 1)

    for i in indices:
        row = forecasts.iloc[i]
        t = (
            row["ds"].strftime("%H:%M")
            if hasattr(row["ds"], "strftime")
            else str(row["ds"])
        )
        ens = row.get("ensemble", row.get("NBEATS", 0))
        low = row.get("low", ens)
        high = row.get("high", ens)

        if show_clamped:
            clamped = row.get("ensemble_clamped", ens)
            pct = (clamped / last_price - 1) * 100
            print(f"{t:<10} {ens:>12,.0f} {clamped:>12,.0f} {low:>10,.0f} {high:>10,.0f} {pct:>+9.2f}%")
        else:
            pct = (ens / last_price - 1) * 100
            print(f"{t:<10} {ens:>12,.0f} {low:>12,.0f} {high:>12,.0f} {pct:>+9.2f}%")

    # Summary
    if "ensemble" in forecasts.columns:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Last close:      {last_price:>14,.0f}")

        # Use clamped values if available
        price_col = "ensemble_clamped" if "ensemble_clamped" in forecasts.columns else "ensemble"
        fc = forecasts[price_col]

        if "ensemble_clamped" in forecasts.columns:
            print("\n  Clamped to ARA/ARB limits:")

        print(
            f"  Open forecast:   {fc.iloc[0]:>14,.0f} ({(fc.iloc[0] / last_price - 1) * 100:+.2f}%)"
        )
        print(
            f"  High forecast:   {fc.max():>14,.0f} ({(fc.max() / last_price - 1) * 100:+.2f}%)"
        )
        print(
            f"  Low forecast:    {fc.min():>14,.0f} ({(fc.min() / last_price - 1) * 100:+.2f}%)"
        )
        print(
            f"  Close forecast:  {fc.iloc[-1]:>14,.0f} ({(fc.iloc[-1] / last_price - 1) * 100:+.2f}%)"
        )

        direction = (
            "BULLISH"
            if fc.iloc[-1] > last_price
            else "BEARISH"
            if fc.iloc[-1] < last_price
            else "NEUTRAL"
        )
        print(f"\n  Outlook: {direction}")


def plot_forecast(
    ohlcv: pd.DataFrame, forecasts: pd.DataFrame, symbol: str, meta: dict
):
    """Plot historical and forecast."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Price chart
        ax1 = axes[0]

        # Historical
        ax1.plot(
            ohlcv["timestamp"], ohlcv["close"], "b-", label="Historical", linewidth=1.5
        )

        # Forecast
        model_cols = [
            c
            for c in forecasts.columns
            if c not in ["ds", "unique_id", "ensemble", "std", "low", "high"]
        ]
        for col in model_cols:
            ax1.plot(
                forecasts["ds"], forecasts[col], "--", alpha=0.4, linewidth=1, label=col
            )

        if "ensemble" in forecasts.columns:
            # Plot clamped forecast if available
            price_col = "ensemble_clamped" if "ensemble_clamped" in forecasts.columns else "ensemble"
            ax1.plot(
                forecasts["ds"],
                forecasts[price_col],
                "r-",
                linewidth=2,
                label="Forecast (Clamped)" if "ensemble_clamped" in forecasts.columns else "Ensemble Forecast",
            )
            if "low" in forecasts.columns:
                ax1.fill_between(
                    forecasts["ds"],
                    forecasts["low"],
                    forecasts["high"],
                    alpha=0.2,
                    color="red",
                    label="95% CI",
                )

            # Plot ARA/ARB limits
            if "ara_limit" in forecasts.columns:
                ax1.axhline(
                    y=forecasts["ara_limit"].iloc[0],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"ARA ({forecasts['ara_limit'].iloc[0]:,.0f})",
                )
            if "arb_limit" in forecasts.columns:
                ax1.axhline(
                    y=forecasts["arb_limit"].iloc[0],
                    color="magenta",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"ARB ({forecasts['arb_limit'].iloc[0]:,.0f})",
                )

        # Add vertical line at forecast start
        if len(forecasts) > 0:
            ax1.axvline(
                x=forecasts["ds"].iloc[0],
                color="green",
                linestyle="--",
                alpha=0.7,
                label="Forecast Start",
            )

        # Add horizontal line at last price
        last_price = ohlcv["close"].iloc[-1]
        ax1.axhline(y=last_price, color="gray", linestyle=":", alpha=0.5)

        sess_name = meta.get("target_session", "").upper()
        title = f"{symbol} - Next Day Forecast"
        if sess_name and meta.get("target_session"):
            sess = MARKET_SESSIONS.get(meta["target_session"])
            if sess:
                title += f" ({sess_name}: {sess.start}-{sess.end})"

        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Volume
        ax2 = axes[1]
        colors = [
            "green" if ohlcv["close"].iloc[i] >= ohlcv["open"].iloc[i] else "red"
            for i in range(len(ohlcv))
        ]
        ax2.bar(
            ohlcv["timestamp"], ohlcv["volume"], color=colors, alpha=0.7, width=0.001
        )
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date/Time")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        plot_dir = Path("plot") / symbol
        plot_dir.mkdir(parents=True, exist_ok=True)
        out = plot_dir / f"short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {out}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Forecast Next Trading Session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IDX Market Sessions (WIB/GMT+7) - SK Direksi BEI II-A Kep-00003/BEI/04-2025:

  Monday-Thursday (Senin-Kamis):
    --session1    Morning (09:00-12:00) = 3 hours
    --session2    Afternoon (13:30-15:50) = 2h20m

  Friday (Jumat):
    --session1    Morning (09:00-11:30) = 2.5 hours
    --session2    Afternoon (14:00-15:50) = 1h50m

  --hours N       Custom duration in hours

Source: https://www.idx.co.id/id/produk/mekanisme-dan-jam-perdagangan

Examples:
    python short.py ICBP --session1           # Forecast tomorrow's morning session
    python short.py ICBP --session2 --plot    # Forecast afternoon session with chart
    python short.py ICBP --hours 3            # Forecast next 3 hours
    python short.py ICBP --hours 1 --interval 1   # 1 hour with 1-min bars
        """,
    )

    parser.add_argument("symbol", nargs="?", help="Stock symbol")
    parser.add_argument(
        "--session", "-S", default=DEFAULT_SESSION, help=f"Data session folder (default: {DEFAULT_SESSION})"
    )
    parser.add_argument(
        "--session1", action="store_true", help="Forecast session 1 (09:00-12:00 Mon-Thu)"
    )
    parser.add_argument(
        "--session2", action="store_true", help="Forecast session 2 (13:30-15:50 Mon-Thu)"
    )
    parser.add_argument("--hours", "-H", type=float, help="Forecast duration in hours")
    parser.add_argument(
        "--interval", "-i", type=int, default=DEFAULT_INTERVAL,
        help=f"Bar interval in minutes (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    parser.add_argument("--list", "-l", action="store_true", help="List symbols")
    parser.add_argument("--source", "-s", default=str(SOURCES_DIR), help="Data directory")
    parser.add_argument("--retrain", "-r", action="store_true", help="Force retrain from scratch")
    parser.add_argument("--group", "-g", help="Train with symbol group (see models/groups.json)")
    parser.add_argument("--list-groups", action="store_true", help="List available groups")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    # Add watchlist arguments
    add_watchlist_args(parser)

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

    loader = TickToOHLCV(args.source)
    groups = load_groups()

    # List groups
    if args.list_groups:
        if not groups:
            print("No groups defined. Create models/groups.json to define groups.")
        else:
            print("Available groups (edit models/groups.json to modify):")
            for name, symbols in groups.items():
                available = [s for s in symbols if s in loader.get_symbols()]
                print(f"  {name}: {', '.join(symbols)}")
                if available:
                    print(f"    (available in sources: {', '.join(available)})")
        return 0

    if args.list:
        print("Available symbols:")
        for sym in sorted(loader.get_symbols()):
            sessions = loader.get_sessions(sym)
            group = find_group_for_symbol(sym, groups)
            group_str = f" [{group}]" if group else ""
            print(f"  {sym}: sessions {', '.join(sessions)}{group_str}")
        return 0

    if not args.symbol:
        parser.print_help()
        return 1

    symbol = args.symbol.upper()

    # Validate symbol
    available_symbols = loader.get_symbols()
    if symbol not in available_symbols:
        raise SymbolNotFoundError(symbol, available_symbols)

    # Validate session
    sessions = loader.get_sessions(symbol)
    if args.session not in sessions:
        raise SessionNotFoundError(args.session, sessions)

    # Validate group
    group_name = args.group
    if group_name and group_name not in groups:
        raise GroupNotFoundError(group_name, list(groups.keys()))

    # Determine forecast duration
    target_session: Optional[str] = None
    if args.session1:
        hours = MARKET_SESSIONS["session1"].hours
        target_session = "session1"
    elif args.session2:
        hours = MARKET_SESSIONS["session2"].hours
        target_session = "session2"
    elif args.hours:
        hours = args.hours
    else:
        # Default: session 1
        hours = MARKET_SESSIONS["session1"].hours
        target_session = "session1"

    print(f"\n{'=' * 70}")
    print(f"SESSION FORECASTER - {symbol}")
    print(f"{'=' * 70}")
    if group_name:
        print(f"Group training: {group_name} ({', '.join(groups[group_name])})")

    try:
        # Load data based on group or single symbol
        if group_name:
            logger.info(f"Loading group '{group_name}' data from session {args.session}...")
            ohlcv = loader.load_group_session(groups[group_name], args.session, args.interval)
            _require_non_empty(ohlcv, "No tick data found for any symbol in group")
            symbols_loaded = ohlcv["unique_id"].nunique()
            print(f"\nLoaded {len(ohlcv)} total bars from {symbols_loaded} symbols")
        else:
            logger.info(f"Loading tick data from session {args.session}...")
            ohlcv = loader.load_session(symbol, args.session, args.interval)
            _require_non_empty(ohlcv, "No tick data found")
            print(f"Created {len(ohlcv)} bars ({args.interval}-min interval)")

        print(f"Total trades: {ohlcv['trades'].sum():,.0f}")
        print(f"Total volume: {ohlcv['volume'].sum():,.0f} lots")

        # Get last price for the target symbol
        if group_name and "unique_id" in ohlcv.columns:
            symbol_data = ohlcv[ohlcv["unique_id"] == symbol]
            close_series: pd.Series = symbol_data["close"] if not symbol_data.empty else ohlcv["close"]  # type: ignore[assignment]
            last_price = _safe_last(close_series)
        else:
            symbol_data = ohlcv
            close_series = ohlcv["close"]  # type: ignore[assignment]
            last_price = _safe_last(close_series)

        forecaster = SessionForecaster(interval=args.interval, symbol=symbol, group=group_name)
        forecasts, historical, meta = forecaster.forecast(ohlcv, hours, target_session, force_retrain=args.retrain)

        print_results(forecasts, last_price, symbol, meta)

        if not args.no_plot:
            # Use symbol-specific data for plots when using group training
            plot_data: pd.DataFrame = symbol_data if group_name and not symbol_data.empty else ohlcv  # type: ignore[assignment]
            plot_forecast(plot_data, forecasts, symbol, meta)

        symbol_csv_dir = CSV_DIR / symbol
        symbol_csv_dir.mkdir(parents=True, exist_ok=True)
        out = symbol_csv_dir / f"short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        forecasts.to_csv(out, index=False)
        logger.info(f"Saved: {out}")

        # Update watchlist if requested
        if args.watchlist:
            price_col = "ensemble_clamped" if "ensemble_clamped" in forecasts.columns else "ensemble"
            if price_col in forecasts.columns:
                fc = forecasts[price_col]
                final_forecast = fc.iloc[-1]
                pct_change = (final_forecast / last_price - 1) * 100
                outlook = Outlook.from_price_change(last_price, final_forecast)

                print(f"\nForecast outlook for {symbol}: {outlook.name} ({pct_change:+.2f}%)")

                # Filter based on bullish/bearish flags
                symbols_with_outlook = [{"symbol": symbol, "outlook": outlook.name}]
                symbols_to_add = filter_symbols_by_outlook(
                    symbols_with_outlook,
                    bullish_only=args.bullish,
                    bearish_only=args.bearish,
                )

                if symbols_to_add:
                    try:
                        result = update_watchlist(
                            args.watchlist_id,
                            symbols_to_add,
                            keep_existing=args.keep,
                            debug=getattr(args, 'wl_debug', False),
                        )
                        print_watchlist_summary(result)
                    except Exception as e:
                        logger.error(f"Watchlist update failed: {e}")
                else:
                    filter_type = "bullish" if args.bullish else "bearish" if args.bearish else ""
                    print(f"\nNo symbols to add to watchlist (outlook is {outlook.name}, filter: {filter_type})")

        return 0

    except ForecastError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
