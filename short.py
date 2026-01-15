#!/usr/bin/env python3
"""
Intraday Stock Forecasting - Single Session
Forecasts next trading session based on tick data from running-trade.json.

Usage:
    python short.py SYMBOL --hours 3          # Forecast next 3 hours
    python short.py SYMBOL --session1         # Forecast session 1 (09:00-12:00 Mon-Thu)
    python short.py SYMBOL --session2         # Forecast session 2 (13:30-15:50 Mon-Thu)

Example:
    python short.py ICBP --session1 --plot    # Forecast tomorrow's session 1
    python short.py ICBP --hours 3 --interval 5
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATS, NHITS
from neuralforecast.losses.pytorch import HuberLoss, DistributionLoss

warnings.filterwarnings("ignore")

# Model persistence directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Groups config file
GROUPS_FILE = MODELS_DIR / "groups.json"


def load_groups() -> dict:
    """Load stock groups from config file."""
    if not GROUPS_FILE.exists():
        return {}
    try:
        with open(GROUPS_FILE, "r") as f:
            groups = json.load(f)
        return {k: v for k, v in groups.items() if not k.startswith("_")}
    except (json.JSONDecodeError, IOError):
        return {}


def find_group_for_symbol(symbol: str, groups: dict) -> Optional[str]:
    """Find which group a symbol belongs to."""
    for group_name, symbols in groups.items():
        if symbol.upper() in [s.upper() for s in symbols]:
            return group_name
    return None


# IDX Market Hours (WIB/GMT+7)
# Source: https://www.idx.co.id/id/produk/mekanisme-dan-jam-perdagangan
# Berdasarkan SK Direksi BEI Nomor II-A Kep-00003/BEI/04-2025
#
# Monday-Thursday (Senin-Kamis):
#   Pre-opening: 08:45-09:00
#   Session 1: 09:00-12:00 (3 hours)
#   Lunch: 12:00-13:30
#   Session 2: 13:30-15:50 (2h20m)
#   Pre-closing: 15:50-16:00
#   Post-trading: 16:02-16:15
#
# Friday (Jumat):
#   Session 1: 09:00-11:30 (2.5 hours)
#   Session 2: 14:00-15:50 (1h50m)

MARKET_SESSIONS = {
    "session1": {"start": "09:00", "end": "12:00", "hours": 3.0},           # Morning (Mon-Thu)
    "session1_fri": {"start": "09:00", "end": "11:30", "hours": 2.5},       # Morning (Friday)
    "session2": {"start": "13:30", "end": "15:50", "hours": 2.33},          # Afternoon (Mon-Thu)
    "session2_fri": {"start": "14:00", "end": "15:50", "hours": 1.83},      # Afternoon (Friday)
    "pre_open": {"start": "08:45", "end": "09:00", "hours": 0.25},          # Pre-opening
    "pre_close": {"start": "15:50", "end": "16:00", "hours": 0.17},         # Pre-closing
    "post_trade": {"start": "16:02", "end": "16:15", "hours": 0.22},        # Post-trading
}


class TickToOHLCV:
    """Convert tick-level trade data to OHLCV bars."""

    def __init__(self, base_path: str = "sources"):
        self.base_path = Path(base_path)

    def load_json(self, filepath: Path) -> Optional[dict]:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def parse_number(self, val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace(",", "") or 0)

    def get_session_date(self, session_path: Path) -> datetime:
        """Extract date from session files."""
        for f in session_path.glob("analysis-data-*.json"):
            try:
                date_str = f.stem.replace("analysis-data-", "").split("_")[0]
                return datetime.strptime(date_str, "%Y-%m-%d")
            except:
                pass

        md = self.load_json(session_path / "market-detector.json")
        if md and "data" in md:
            date_str = md["data"].get("from")
            if date_str:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass

        fd = self.load_json(session_path / "findata.json")
        if fd and "data" in fd:
            date_str = fd["data"].get("from")
            if date_str:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass

        return datetime.now()

    def parse_trades(self, trades: List[dict], base_date: datetime) -> pd.DataFrame:
        """Parse running trades into a DataFrame."""
        records = []

        for trade in trades:
            try:
                price = self.parse_number(trade.get("price", 0))
                lot = self.parse_number(trade.get("lot", 0))
                time_str = trade.get("time", "09:00:00")

                if price <= 0 or lot <= 0:
                    continue

                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1])
                second = int(parts[2]) if len(parts) > 2 else 0
                timestamp = base_date.replace(hour=hour, minute=minute, second=second)

                records.append(
                    {
                        "timestamp": timestamp,
                        "price": price,
                        "lot": lot,
                        "value": price * lot * 100,
                        "action": trade.get("action", ""),
                        "buyer_type": trade.get("buyer_type", ""),
                        "seller_type": trade.get("seller_type", ""),
                    }
                )
            except:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df.sort_values("timestamp").reset_index(drop=True)

    def aggregate_bars(self, tick_df: pd.DataFrame, interval: int = 5) -> pd.DataFrame:
        """Aggregate ticks to OHLCV bars."""
        if tick_df.empty:
            return pd.DataFrame()

        df = tick_df.set_index("timestamp")

        ohlcv = df["price"].resample(f"{interval}min").ohlc()
        ohlcv.columns = ["open", "high", "low", "close"]

        ohlcv["volume"] = df["lot"].resample(f"{interval}min").sum()
        ohlcv["value"] = df["value"].resample(f"{interval}min").sum()
        ohlcv["trades"] = df["price"].resample(f"{interval}min").count()

        buy_mask = df["action"] == "buy"
        ohlcv["buy_vol"] = df.loc[buy_mask, "lot"].resample(f"{interval}min").sum()
        ohlcv["sell_vol"] = df.loc[~buy_mask, "lot"].resample(f"{interval}min").sum()

        f_buy = df["buyer_type"].str.contains("FOREIGN", na=False)
        f_sell = df["seller_type"].str.contains("FOREIGN", na=False)
        ohlcv["foreign_buy"] = df.loc[f_buy, "lot"].resample(f"{interval}min").sum()
        ohlcv["foreign_sell"] = df.loc[f_sell, "lot"].resample(f"{interval}min").sum()

        ohlcv = ohlcv.dropna(subset=["close"]).fillna(0)

        ohlcv["returns"] = ohlcv["close"].pct_change()
        ohlcv["volatility"] = (ohlcv["high"] - ohlcv["low"]) / (ohlcv["close"] + 1e-8)
        ohlcv["spread"] = ohlcv["high"] - ohlcv["low"]
        ohlcv["buy_pressure"] = ohlcv["buy_vol"] / (ohlcv["volume"] + 1)
        ohlcv["foreign_net"] = ohlcv["foreign_buy"] - ohlcv["foreign_sell"]
        ohlcv["vwap"] = ohlcv["value"] / (ohlcv["volume"] * 100 + 1)

        ohlcv["momentum"] = ohlcv["close"].pct_change(3)
        ohlcv["vol_ma"] = ohlcv["volume"].rolling(3, min_periods=1).mean()
        ohlcv["vol_ratio"] = ohlcv["volume"] / (ohlcv["vol_ma"] + 1)

        return ohlcv.reset_index()

    def load_session(
        self, symbol: str, session: str, interval: int = 5
    ) -> pd.DataFrame:
        """Load and process a single session."""
        session_path = self.base_path / symbol / session
        base_date = self.get_session_date(session_path)

        rt = self.load_json(session_path / "running-trade.json")
        if rt and "data" in rt:
            trades = rt["data"].get("running_trade", [])
            if trades:
                tick_df = self.parse_trades(trades, base_date)
                if not tick_df.empty:
                    return self.aggregate_bars(tick_df, interval)

        trt = self.load_json(session_path / "today-running-trade.json")
        if trt and "data" in trt:
            trades = trt["data"].get("running_trade", [])
            if trades:
                tick_df = self.parse_trades(trades, base_date)
                if not tick_df.empty:
                    return self.aggregate_bars(tick_df, interval)

        return pd.DataFrame()

    def get_symbols(self) -> List[str]:
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def get_sessions(self, symbol: str) -> List[str]:
        path = self.base_path / symbol
        if not path.exists():
            return []
        sessions = [d.name for d in path.iterdir() if d.is_dir()]
        return sorted(sessions, key=lambda x: int(x) if x.isdigit() else 0)

    def load_group_session(
        self, symbols: List[str], session: str, interval: int = 5
    ) -> pd.DataFrame:
        """Load session data for multiple symbols (for group training)."""
        all_data = []
        available = self.get_symbols()

        for symbol in symbols:
            if symbol not in available:
                continue

            sessions = self.get_sessions(symbol)
            if session not in sessions:
                continue

            try:
                df = self.load_session(symbol, session, interval)
                if not df.empty:
                    df["unique_id"] = symbol
                    all_data.append(df)
                    print(f"    {symbol}: {len(df)} bars")
            except Exception as e:
                print(f"    {symbol}: Error - {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        return combined.sort_values(["unique_id", "timestamp"]).reset_index(drop=True)


class SessionForecaster:
    """Forecast next trading session."""

    def __init__(self, interval: int = 5, symbol: str = "STOCK", group: str = None):
        self.interval = interval
        self.symbol = symbol
        self.group = group
        self.nf = None

        # Use group name for model path if training with group
        model_name = f"short_group_{group}_{interval}min" if group else f"short_{symbol}_{interval}min"
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

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare NeuralForecast format with gap handling."""
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

        # Handle gaps in intraday time series
        nf_df = self._fill_intraday_gaps(nf_df)

        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)
        return nf_df, available

    def _fill_intraday_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in intraday time series and add available_mask column."""
        result_dfs = []

        for uid in df["unique_id"].unique():
            uid_df = df[df["unique_id"] == uid].copy()
            uid_df = uid_df.sort_values("ds")

            if len(uid_df) < 2:
                uid_df["available_mask"] = 1.0
                result_dfs.append(uid_df)
                continue

            # Create complete time range at the specified interval
            min_time = uid_df["ds"].min()
            max_time = uid_df["ds"].max()

            # Generate all timestamps at interval frequency
            all_times = pd.date_range(start=min_time, end=max_time, freq=f"{self.interval}min")

            # Create template with all timestamps
            template = pd.DataFrame({"ds": all_times, "unique_id": uid})

            # Merge with actual data
            merged = template.merge(uid_df, on=["ds", "unique_id"], how="left")

            # Add available_mask: 1 for real data, 0 for gaps
            merged["available_mask"] = merged["y"].notna().astype(float)

            # Fill gaps with forward fill then backward fill
            merged["y"] = merged["y"].ffill().bfill()

            # Fill other numeric columns
            for col in merged.columns:
                if col not in ["ds", "unique_id", "available_mask"] and merged[col].dtype in ["float64", "int64"]:
                    merged[col] = merged[col].ffill().bfill().fillna(0)

            result_dfs.append(merged)

        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)
            gap_count = (result["available_mask"] == 0).sum()
            if gap_count > 0:
                print(f"  Filled {gap_count} gaps in intraday data (marked with available_mask=0)")
            return result

        return df

    def forecast(
        self, df: pd.DataFrame, hours: float, target_session: str = None, force_retrain: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Train and forecast for specified hours. Supports incremental training."""
        nf_df, exog = self.prepare_data(df)
        n = len(nf_df)

        horizon = self.calculate_horizon(hours)

        print(f"\n{'=' * 60}")
        print("INPUT DATA")
        print(f"{'=' * 60}")
        print(f"  Historical bars: {n}")
        print(
            f"  Time range: {nf_df['ds'].min().strftime('%H:%M')} - {nf_df['ds'].max().strftime('%H:%M')}"
        )
        print(f"  Price range: {nf_df['y'].min():,.2f} - {nf_df['y'].max():,.2f}")
        print(f"  Last price: {nf_df['y'].iloc[-1]:,.2f}")

        print(f"\n{'=' * 60}")
        print("FORECAST CONFIG")
        print(f"{'=' * 60}")
        print(f"  Forecast duration: {hours} hours")
        print(f"  Bar interval: {self.interval} min")
        print(f"  Forecast bars: {horizon}")

        if target_session:
            sess = MARKET_SESSIONS.get(target_session, {})
            print(
                f"  Target session: {target_session.upper()} ({sess.get('start', '?')} - {sess.get('end', '?')})"
            )

        if n < 5:
            raise ValueError(f"Need at least 5 historical bars (have {n})")

        # Adjust horizon if very limited data
        max_h = n * 2  # Allow forecasting up to 2x the input data
        if horizon > max_h:
            print(f"  Note: Limiting horizon from {horizon} to {max_h} bars")
            horizon = max_h

        input_size = min(n - 1, 12)
        input_size = max(input_size, 2)

        print(f"  Lookback window: {input_size} bars")

        # Check for saved model and determine training strategy
        session_date = nf_df['ds'].iloc[0].strftime('%Y-%m-%d') if hasattr(nf_df['ds'].iloc[0], 'strftime') else str(nf_df['ds'].iloc[0])[:10]
        saved_meta = None if force_retrain else self.load_model()
        use_fine_tuning = False
        max_steps = 200

        if saved_meta is not None:
            prev_count = saved_meta.get("data_count", 0)
            new_data = n - prev_count

            if new_data > 0:
                print(f"\n  New data detected: {new_data} bars ({prev_count} -> {n})")
                print(f"  Strategy: Fine-tuning with reduced steps")
                use_fine_tuning = True
                max_steps = 50  # Reduced steps for fine-tuning
            elif new_data == 0:
                print(f"\n  No new data. Using saved model directly.")
                print(f"\nGenerating {horizon} bars ({hours} hours) using cached model...")

                # Predict only for requested symbol if using group training
                if self.group and self.symbol:
                    symbol_df = nf_df[nf_df["unique_id"] == self.symbol].copy()
                    if not symbol_df.empty:
                        forecasts = self.nf.predict(df=symbol_df)
                        last_date = symbol_df["ds"].iloc[-1]
                    else:
                        forecasts = self.nf.predict()
                        last_date = nf_df["ds"].iloc[-1]
                else:
                    forecasts = self.nf.predict()
                    last_date = nf_df["ds"].iloc[-1]

                # Skip to timestamp generation below
                next_day = last_date + timedelta(days=1)
                if target_session:
                    sess = MARKET_SESSIONS.get(target_session, {})
                    start_time = sess.get("start", "09:00")
                    h, m = map(int, start_time.split(":"))
                    start_dt = next_day.replace(hour=h, minute=m, second=0)
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
                print(f"\n  Data reduced. Retraining from scratch.")
                saved_meta = None

        if saved_meta is None:
            print(f"\n  Strategy: Training from scratch (no saved model)")

        # Use HuberLoss for robustness to outliers (price gaps, big moves)
        # Use DistributionLoss with StudentT for probabilistic forecasts
        models = [
            # NBEATS with HuberLoss - robust to outliers
            NBEATS(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type="robust",
                loss=HuberLoss(),
                random_seed=42,
            ),
            # NHITS with StudentT distribution - probabilistic forecasts with heavy tails
            NHITS(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type="robust",
                loss=DistributionLoss(distribution="StudentT", level=[80, 90]),
                random_seed=42,
            ),
            # LSTM with HuberLoss - robust sequential model
            LSTM(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type="robust",
                loss=HuberLoss(),
                random_seed=42,
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
        if self.group and self.symbol:
            symbol_df = nf_df[nf_df["unique_id"] == self.symbol].copy()
            if not symbol_df.empty:
                forecasts = self.nf.predict(df=symbol_df)
                print(f"  Predicting for {self.symbol} only")
            else:
                print(f"  Warning: {self.symbol} not found in group data, predicting all")
                forecasts = self.nf.predict()
        else:
            forecasts = self.nf.predict()

        # Generate future timestamps starting from next trading day 09:00
        # Get the last date for the target symbol
        if self.group and self.symbol and "unique_id" in nf_df.columns:
            symbol_data = nf_df[nf_df["unique_id"] == self.symbol]
            last_date = symbol_data["ds"].iloc[-1] if not symbol_data.empty else nf_df["ds"].iloc[-1]
        else:
            last_date = nf_df["ds"].iloc[-1]
        next_day = last_date + timedelta(days=1)

        if target_session:
            sess = MARKET_SESSIONS.get(target_session, {})
            start_time = sess.get("start", "09:00")
            h, m = map(int, start_time.split(":"))
            start_dt = next_day.replace(hour=h, minute=m, second=0)
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

        meta = {
            "horizon": horizon,
            "hours": hours,
            "interval": self.interval,
            "target_session": target_session,
            "forecast_start": start_dt,
            "forecast_end": future_times[-1] if future_times else start_dt,
        }

        return forecasts, nf_df, meta


def print_results(forecasts: pd.DataFrame, last_price: float, symbol: str, meta: dict):
    """Print forecast results."""
    print(f"\n{'=' * 70}")
    print(f"FORECAST: {symbol}")
    if meta.get("target_session"):
        sess = MARKET_SESSIONS.get(meta["target_session"], {})
        print(
            f"Target: {meta['target_session'].upper()} ({sess.get('start')}-{sess.get('end')})"
        )
    print(
        f"Period: {meta['forecast_start'].strftime('%Y-%m-%d %H:%M')} to {meta['forecast_end'].strftime('%H:%M')}"
    )
    print(f"{'=' * 70}")

    # Show summary at key intervals
    cols = ["ds", "ensemble", "low", "high"]
    cols = [c for c in cols if c in forecasts.columns]

    # Show every 15 minutes or so
    step = max(1, len(forecasts) // 12)

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
        pct = (ens / last_price - 1) * 100
        print(f"{t:<10} {ens:>12,.2f} {low:>12,.2f} {high:>12,.2f} {pct:>+9.2f}%")

    # Summary
    if "ensemble" in forecasts.columns:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Last close:      {last_price:>14,.2f}")
        print(
            f"  Open forecast:   {forecasts['ensemble'].iloc[0]:>14,.2f} ({(forecasts['ensemble'].iloc[0] / last_price - 1) * 100:+.2f}%)"
        )
        print(
            f"  High forecast:   {forecasts['ensemble'].max():>14,.2f} ({(forecasts['ensemble'].max() / last_price - 1) * 100:+.2f}%)"
        )
        print(
            f"  Low forecast:    {forecasts['ensemble'].min():>14,.2f} ({(forecasts['ensemble'].min() / last_price - 1) * 100:+.2f}%)"
        )
        print(
            f"  Close forecast:  {forecasts['ensemble'].iloc[-1]:>14,.2f} ({(forecasts['ensemble'].iloc[-1] / last_price - 1) * 100:+.2f}%)"
        )

        direction = (
            "BULLISH"
            if forecasts["ensemble"].iloc[-1] > last_price
            else "BEARISH"
            if forecasts["ensemble"].iloc[-1] < last_price
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
            ax1.plot(
                forecasts["ds"],
                forecasts["ensemble"],
                "r-",
                linewidth=2,
                label="Ensemble Forecast",
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
        if sess_name:
            sess = MARKET_SESSIONS.get(meta["target_session"], {})
            title += f" ({sess_name}: {sess.get('start')}-{sess.get('end')})"

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

        plot_dir = Path("plot")
        plot_dir.mkdir(exist_ok=True)
        out = plot_dir / f"short_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {out}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available")


def main():
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
        "--session", "-S", default="1", help="Data session folder (default: 1)"
    )
    parser.add_argument(
        "--session1", action="store_true", help="Forecast session 1 (09:00-12:00 Mon-Thu)"
    )
    parser.add_argument(
        "--session2", action="store_true", help="Forecast session 2 (13:30-15:50 Mon-Thu)"
    )
    parser.add_argument("--hours", "-H", type=float, help="Forecast duration in hours")
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=5,
        help="Bar interval in minutes (default: 5)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    parser.add_argument("--list", "-l", action="store_true", help="List symbols")
    parser.add_argument("--source", "-s", default="sources", help="Data directory")
    parser.add_argument("--retrain", "-r", action="store_true", help="Force retrain from scratch (ignore saved model)")
    parser.add_argument("--group", "-g", help="Train with symbol group (e.g., 'banking', 'mining'). Edit models/groups.json to define groups.")
    parser.add_argument("--list-groups", action="store_true", help="List available groups")

    args = parser.parse_args()

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
        return

    if args.list:
        print("Available symbols:")
        for sym in sorted(loader.get_symbols()):
            sessions = loader.get_sessions(sym)
            group = find_group_for_symbol(sym, groups)
            group_str = f" [{group}]" if group else ""
            print(f"  {sym}: sessions {', '.join(sessions)}{group_str}")
        return

    if not args.symbol:
        parser.print_help()
        sys.exit(1)

    symbol = args.symbol.upper()

    if symbol not in loader.get_symbols():
        print(f"Error: {symbol} not found")
        print(f"Available: {', '.join(sorted(loader.get_symbols()))}")
        sys.exit(1)

    sessions = loader.get_sessions(symbol)
    if args.session not in sessions:
        print(f"Error: Session {args.session} not found")
        print(f"Available: {', '.join(sessions)}")
        sys.exit(1)

    # Determine group to use
    group_name = args.group
    if group_name and group_name not in groups:
        print(f"Error: Group '{group_name}' not found")
        print(f"Available groups: {', '.join(groups.keys())}")
        print("Edit models/groups.json to add groups.")
        sys.exit(1)

    # Determine forecast duration
    target_session = None
    if args.session1:
        hours = MARKET_SESSIONS["session1"]["hours"]
        target_session = "session1"
    elif args.session2:
        hours = MARKET_SESSIONS["session2"]["hours"]
        target_session = "session2"
    elif args.hours:
        hours = args.hours
    else:
        # Default: session 1
        hours = MARKET_SESSIONS["session1"]["hours"]
        target_session = "session1"

    print(f"\n{'=' * 70}")
    print(f"SESSION FORECASTER - {symbol}")
    print(f"{'=' * 70}")
    if group_name:
        print(f"Group training: {group_name} ({', '.join(groups[group_name])})")

    try:
        # Load data based on group or single symbol
        if group_name:
            print(f"\nLoading group '{group_name}' data from session {args.session}...")
            ohlcv = loader.load_group_session(groups[group_name], args.session, args.interval)
            if ohlcv.empty:
                print("Error: No tick data found for any symbol in group")
                sys.exit(1)
            symbols_loaded = ohlcv["unique_id"].nunique()
            print(f"\nLoaded {len(ohlcv)} total bars from {symbols_loaded} symbols")
        else:
            print(f"\nLoading tick data from session {args.session}...")
            ohlcv = loader.load_session(symbol, args.session, args.interval)
            if ohlcv.empty:
                print("Error: No tick data found")
                sys.exit(1)
            print(f"Created {len(ohlcv)} bars ({args.interval}-min interval)")

        print(f"Total trades: {ohlcv['trades'].sum():,.0f}")
        print(f"Total volume: {ohlcv['volume'].sum():,.0f} lots")

        # Get last price for the target symbol
        if group_name and "unique_id" in ohlcv.columns:
            symbol_data = ohlcv[ohlcv["unique_id"] == symbol]
            last_price = symbol_data["close"].iloc[-1] if not symbol_data.empty else ohlcv["close"].iloc[-1]
        else:
            symbol_data = ohlcv
            last_price = ohlcv["close"].iloc[-1]

        forecaster = SessionForecaster(interval=args.interval, symbol=symbol, group=group_name)
        forecasts, historical, meta = forecaster.forecast(ohlcv, hours, target_session, force_retrain=args.retrain)

        print_results(forecasts, last_price, symbol, meta)

        if not args.no_plot:
            # Use symbol-specific data for plots when using group training
            plot_data = symbol_data if group_name and not symbol_data.empty else ohlcv
            plot_forecast(plot_data, forecasts, symbol, meta)

        out = f"short_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        forecasts.to_csv(out, index=False)
        print(f"\nSaved: {out}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
