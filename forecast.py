#!/usr/bin/env python3
"""
Stock Price Forecasting using NeuralForecast
Uses comprehensive market data including orderbook, broker flow, and foreign transactions.

Usage: python forecast.py SYMBOL [--horizon N] [--interval M] [--plot]

Example:
    python forecast.py ARCI --horizon 10 --plot
    python forecast.py ARCI --interval 5 --horizon 20
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, HuberLoss
from neuralforecast.models import LSTM, NBEATS, NHITS, TFT

# Shared utilities
from modules import (
    MODELS_DIR,
    CSV_DIR,
    GROUPS_FILE,
    load_groups,
    find_group_for_symbol,
)
from modules.idx_rules import (
    add_ara_arb_features,
    adjust_mask_for_limit_hits,
    clamp_forecast_series,
    get_daily_limit_info,
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
    send_forecast_notification,
    validate_telegram_args,
)
from modules.idx_rules import get_daily_limit_info

warnings.filterwarnings("ignore")


class MarketAlphaEngine:
    """
    Extract alpha-generating features from analysis-data-*.json files.
    Based on institutional trading patterns and order flow analysis.
    """

    def __init__(self, base_path: str = "sources"):
        self.base_path = Path(base_path)

    def load_json(self, filepath: Path) -> Optional[dict]:
        """Load a JSON file safely."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def parse_number(self, value) -> float:
        """Parse various number formats to float."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.replace(",", "").replace(" ", "") or 0)
        return 0.0

    def extract_session_features(self, symbol: str, session: str) -> Optional[Dict]:
        """Extract all alpha features from analyzed.json."""
        session_path = self.base_path / symbol / session

        # Load analyzed.json
        analysis = self.load_json(session_path / "analyzed.json")
        if not analysis:
            return None

        # Extract price data from price_feed section
        pf = analysis.get("price_feed", {})
        if not pf:
            return None

        close = self.parse_number(pf.get("close", 0) or pf.get("last", 0))
        if close == 0:
            return None

        high = self.parse_number(pf.get("high", 0))
        low = self.parse_number(pf.get("low", 0))
        open_price = self.parse_number(pf.get("open", 0))
        volume = self.parse_number(pf.get("volume", 0))
        value = self.parse_number(pf.get("value", 0))

        # Extract date from metadata
        ds = None
        time_horizons = analysis.get("metadata", {}).get("time_horizons", {})
        md_to = time_horizons.get("market_detector", {}).get("to")
        if md_to:
            try:
                ds = datetime.strptime(md_to, "%Y-%m-%d")
            except:
                pass

        if not ds:
            ds = datetime.now() - timedelta(
                days=int(session) if session.isdigit() else 0
            )

        # Calculate OBI from depth section (pre-computed totals)
        depth = analysis.get("depth") or {}
        bid_data = depth.get("bid") or {}
        ask_data = depth.get("offer") or {}
        bid_vol = self.parse_number(bid_data.get("total_volume", 0))
        ask_vol = self.parse_number(ask_data.get("total_volume", 0))
        total_vol = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        # Extract bandar/market detector data
        md = analysis.get("market_detector", {})
        bandar = md.get("bandar", {}) if md else {}

        # AccDist signal from top1
        top1 = bandar.get("top1", {}) if bandar else {}
        accdist_map = {
            "Big Acc": 2.0,
            "Normal Acc": 1.0,
            "Neutral": 0.0,
            "Normal Dist": -1.0,
            "Big Dist": -2.0,
        }
        accdist = accdist_map.get(top1.get("accdist", "Neutral"), 0.0)
        net_percent = self.parse_number(top1.get("percent", 0))

        # Concentration from top3 percent (already computed as percentage of total)
        top3 = bandar.get("top3", {}) if bandar else {}
        # top3.percent is negative for distribution, positive for accumulation
        # Convert to buy/sell concentration estimate
        top3_pct = abs(self.parse_number(top3.get("percent", 0))) / 100.0
        if top3.get("accdist", "").endswith("Acc"):
            buy_conc = min(top3_pct, 1.0)
            sell_conc = 0.0
        elif top3.get("accdist", "").endswith("Dist"):
            buy_conc = 0.0
            sell_conc = min(top3_pct, 1.0)
        else:
            buy_conc = top3_pct / 2
            sell_conc = top3_pct / 2

        # Foreign flow - prefer price_feed (historical API has per-day data), fallback to findata
        foreign_buy = self.parse_number(pf.get("foreign_buy", 0))
        foreign_sell = self.parse_number(pf.get("foreign_sell", 0))
        foreign_net_val = self.parse_number(pf.get("foreign_net", 0))

        # Fallback to findata if price_feed doesn't have foreign data
        if foreign_buy == 0 and foreign_sell == 0:
            findata = analysis.get("findata", {})
            if findata:
                summary = findata.get("summary", {})
                if summary:
                    net_foreign = summary.get("net_foreign", {})
                    if net_foreign:
                        foreign_net_val = self.parse_number(net_foreign.get("raw", 0))
                    fb = summary.get("foreign_buy", {})
                    if fb:
                        foreign_buy = self.parse_number(fb.get("raw", 0))
                    fs = summary.get("foreign_sell", {})
                    if fs:
                        foreign_sell = self.parse_number(fs.get("raw", 0))

        # Calculate foreign ratio
        total_foreign = foreign_buy + foreign_sell
        foreign_ratio = (
            foreign_net_val / (total_foreign + 1e-5) if total_foreign > 0 else 0.0
        )

        # Volatility
        volatility = (high - low) / (close + 1e-5) if close > 0 else 0.0

        # Price momentum - use API-provided pct_change, fallback to calculation
        pct_change = self.parse_number(pf.get("pct_change", 0))
        if pct_change == 0:
            history = pf.get("history", [])
            if history and len(history) >= 2:
                prev_close = self.parse_number(history[1].get("close", 0))
                if prev_close > 0:
                    pct_change = ((close - prev_close) / prev_close) * 100

        return {
            "ds": ds,
            "unique_id": symbol,
            "session": session,
            # Target
            "y": close,
            # OHLCV
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "value": value,
            "frequency": 0,  # Not available in analysis JSON
            # Alpha Features
            "obi": obi,  # Orderbook Imbalance
            "buy_concentration": buy_conc,  # Top 3 buyer concentration
            "sell_concentration": sell_conc,  # Top 3 seller concentration
            "concentration_diff": buy_conc - sell_conc,  # Net concentration
            "foreign_net": foreign_net_val,  # Foreign net value
            "foreign_ratio": foreign_ratio,  # Foreign as % of total
            "accdist": accdist,  # Accumulation/Distribution signal
            "net_percent": net_percent,  # Bandar net percent
            "volatility": volatility,  # Intraday volatility
            "pct_change": pct_change,  # Price change %
        }

    def load_symbol_data(self, symbol: str, verbose: bool = True) -> pd.DataFrame:
        """Load all session data for a symbol."""
        symbol_path = self.base_path / symbol
        if not symbol_path.exists():
            raise ValueError(f"Symbol not found: {symbol}")

        sessions = [d.name for d in symbol_path.iterdir() if d.is_dir()]
        sessions = sorted(sessions, key=lambda x: int(x) if x.isdigit() else 0)

        if not sessions:
            raise ValueError(f"No sessions found for: {symbol}")

        features_list = []
        for session in sessions:
            features = self.extract_session_features(symbol, session)
            if features:
                features_list.append(features)
                if verbose:
                    print(
                        f"  Session {session}: close={features['y']:,.0f}, OBI={features['obi']:.3f}, "
                        f"BuyConc={features['buy_concentration']:.2f}, ForeignNet={features['foreign_net']:,.0f}"
                    )

        if not features_list:
            raise ValueError(f"No valid data found for: {symbol}")

        df = pd.DataFrame(features_list)
        df = df.sort_values("ds").reset_index(drop=True)

        return df

    def load_group_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load data for multiple symbols (for group training)."""
        all_data = []
        available = self.get_available_symbols()

        for symbol in symbols:
            if symbol not in available:
                print(f"  Warning: {symbol} not found in sources, skipping")
                continue

            try:
                print(f"\n  Loading {symbol}...")
                df = self.load_symbol_data(symbol, verbose=False)
                all_data.append(df)
                print(f"    {len(df)} sessions loaded")
            except ValueError as e:
                print(f"  Warning: {symbol} - {e}")
                continue

        if not all_data:
            raise ValueError("No data loaded for any symbol in group")

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        return combined

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]


class TickDataProcessor:
    """Process tick-level trade data into OHLCV bars."""

    @staticmethod
    def parse_price(price_str) -> float:
        if isinstance(price_str, (int, float)):
            return float(price_str)
        return float(str(price_str).replace(",", "") or 0)

    @staticmethod
    def parse_lot(lot_str) -> int:
        if isinstance(lot_str, (int, float)):
            return int(lot_str)
        return int(str(lot_str).replace(",", "") or 0)

    @staticmethod
    def parse_time(time_str: str, base_date: datetime) -> datetime:
        try:
            parts = time_str.split(":")
            hour, minute = int(parts[0]), int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            return base_date.replace(hour=hour, minute=minute, second=second)
        except:
            return base_date

    def process_running_trades(
        self, trades: List[dict], base_date: datetime
    ) -> pd.DataFrame:
        records = []
        for trade in trades:
            try:
                price = self.parse_price(trade.get("price", 0))
                lot = self.parse_lot(trade.get("lot", 0))
                if price > 0 and lot > 0:
                    records.append(
                        {
                            "timestamp": self.parse_time(
                                trade.get("time", "09:00:00"), base_date
                            ),
                            "price": price,
                            "lot": lot,
                            "action": trade.get("action", "unknown"),
                            "buyer_type": trade.get("buyer_type", ""),
                            "seller_type": trade.get("seller_type", ""),
                        }
                    )
            except:
                continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    def aggregate_to_ohlcv(
        self, tick_df: pd.DataFrame, interval: int = 5
    ) -> pd.DataFrame:
        if tick_df.empty:
            return pd.DataFrame()

        tick_df = tick_df.set_index("timestamp")

        ohlcv = tick_df["price"].resample(f"{interval}min").ohlc()
        ohlcv.columns = ["open", "high", "low", "close"]
        ohlcv["volume"] = tick_df["lot"].resample(f"{interval}min").sum()
        ohlcv["trade_count"] = tick_df["price"].resample(f"{interval}min").count()

        # Buy/Sell flow
        buy_mask = tick_df["action"] == "buy"
        ohlcv["buy_volume"] = (
            tick_df.loc[buy_mask, "lot"].resample(f"{interval}min").sum()
        )
        ohlcv["sell_volume"] = (
            tick_df.loc[~buy_mask, "lot"].resample(f"{interval}min").sum()
        )

        # Foreign flow
        foreign_buy = tick_df["buyer_type"].str.contains("FOREIGN", na=False)
        foreign_sell = tick_df["seller_type"].str.contains("FOREIGN", na=False)
        ohlcv["foreign_buy_vol"] = (
            tick_df.loc[foreign_buy, "lot"].resample(f"{interval}min").sum()
        )
        ohlcv["foreign_sell_vol"] = (
            tick_df.loc[foreign_sell, "lot"].resample(f"{interval}min").sum()
        )

        ohlcv = ohlcv.dropna(subset=["open", "close"]).fillna(0)

        # Derived features
        ohlcv["returns"] = ohlcv["close"].pct_change()
        ohlcv["volatility"] = (ohlcv["high"] - ohlcv["low"]) / (ohlcv["close"] + 1e-5)
        ohlcv["buy_sell_ratio"] = ohlcv["buy_volume"] / (ohlcv["sell_volume"] + 1)
        ohlcv["foreign_net"] = ohlcv["foreign_buy_vol"] - ohlcv["foreign_sell_vol"]

        return ohlcv.reset_index()


class StockForecaster:
    """NeuralForecast-based stock price forecaster using TFT and ensemble."""

    def __init__(self, horizon: int = 5, symbol: str = "STOCK", group: str = None):
        self.horizon = horizon
        self.symbol = symbol
        self.group = group
        self.nf = None

        # Use group name for model path if training with group
        model_name = f"group_{group}" if group else f"forecast_{symbol}"
        self.model_path = MODELS_DIR / model_name
        self.meta_path = MODELS_DIR / f"{model_name}_meta.json"

    def save_model(self, data_count: int):
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
            "horizon": self.horizon,
            "data_count": data_count,
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
            print(f"  Loaded saved model (trained on {meta['data_count']} records)")
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
                    state_dict = torch.load(
                        ckpt_path, map_location="cpu", weights_only=True
                    )
                    # Load with strict=False to handle minor architecture differences
                    model.load_state_dict(state_dict, strict=False)
                    loaded_count += 1
                except Exception as e:
                    print(
                        f"    Warning: Could not load checkpoint for {model_name}: {e}"
                    )

        if loaded_count > 0:
            print(
                f"  Loaded {loaded_count}/{len(models)} model checkpoints (warm-start)"
            )

        return models

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data in NeuralForecast format with alpha features, ARA/ARB features, and gap handling."""

        # Add ARA/ARB features to source data
        df_with_limits = self._add_ara_arb_features(df)

        # Define exogenous features for TFT attention
        # Now includes ARA/ARB proximity features
        exog_features = [
            "obi",
            "concentration_diff",
            "foreign_ratio",
            "volatility",
            "accdist",
            "pct_change",
            # ARA/ARB features
            "ara_proximity",
            "arb_proximity",
            "limit_range_pct",
            "limit_bias",
            "pct_to_ara",
            "pct_to_arb",
        ]

        # Filter to available features
        available_exog = [c for c in exog_features if c in df_with_limits.columns]

        # Build NeuralForecast dataframe
        nf_df = pd.DataFrame(
            {
                "unique_id": df_with_limits["unique_id"],
                "ds": df_with_limits["ds"],
                "y": df_with_limits["y"],
            }
        )

        # Add exogenous features
        for col in available_exog:
            nf_df[col] = df_with_limits[col].fillna(0)

        # Add limit hit flags for mask adjustment
        if "limit_hit" in df_with_limits.columns:
            nf_df["limit_hit"] = df_with_limits["limit_hit"].fillna(0)
        if "ara_hit" in df_with_limits.columns:
            nf_df["ara_hit"] = df_with_limits["ara_hit"].fillna(0)
        if "arb_hit" in df_with_limits.columns:
            nf_df["arb_hit"] = df_with_limits["arb_hit"].fillna(0)

        # Handle gaps in time series by adding available_mask
        # This tells NeuralForecast which rows have real data vs filled gaps
        nf_df = self._fill_gaps_with_mask(nf_df)

        # Adjust mask to down-weight days that hit ARA/ARB limits
        # These days have distorted price signals
        nf_df = adjust_mask_for_limit_hits(nf_df, penalty=0.5)

        # Clean data
        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)

        # Report ARA/ARB statistics
        if "limit_hit" in nf_df.columns:
            limit_days = nf_df["limit_hit"].sum()
            if limit_days > 0:
                print(
                    f"  ARA/ARB limit days detected: {int(limit_days)} ({limit_days / len(nf_df) * 100:.1f}%)"
                )

        return nf_df, available_exog

    def _add_ara_arb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ARA/ARB features for each symbol in the dataframe."""
        result_dfs = []

        for uid in df["unique_id"].unique():
            uid_df = df[df["unique_id"] == uid].copy()
            uid_df = uid_df.sort_values("ds")

            # Need OHLC columns for ARA/ARB calculation
            if all(c in uid_df.columns for c in ["close", "high", "low"]):
                uid_df = add_ara_arb_features(
                    uid_df, close_col="close", high_col="high", low_col="low"
                )
            elif "y" in uid_df.columns:
                # Fallback: use y as close, estimate high/low
                uid_df["_close"] = uid_df["y"]
                uid_df["_high"] = uid_df["y"]
                uid_df["_low"] = uid_df["y"]
                uid_df = add_ara_arb_features(
                    uid_df, close_col="_close", high_col="_high", low_col="_low"
                )
                uid_df = uid_df.drop(
                    columns=["_close", "_high", "_low"], errors="ignore"
                )

            result_dfs.append(uid_df)

        if result_dfs:
            return pd.concat(result_dfs, ignore_index=True)
        return df

    def _fill_gaps_with_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in time series and add available_mask column."""
        result_dfs = []

        for uid in df["unique_id"].unique():
            uid_df = df[df["unique_id"] == uid].copy()
            uid_df = uid_df.sort_values("ds")

            # Create complete date range (business days only for stock data)
            min_date = uid_df["ds"].min()
            max_date = uid_df["ds"].max()

            # Generate all business days in range
            all_dates = pd.date_range(start=min_date, end=max_date, freq="B")

            # Create template with all dates
            template = pd.DataFrame({"ds": all_dates, "unique_id": uid})

            # Merge with actual data
            merged = template.merge(uid_df, on=["ds", "unique_id"], how="left")

            # Add available_mask: 1 for real data, 0 for gaps
            merged["available_mask"] = merged["y"].notna().astype(float)

            # Fill gaps with forward fill then backward fill for y
            merged["y"] = merged["y"].ffill().bfill()

            # Fill other numeric columns
            for col in merged.columns:
                if col not in ["ds", "unique_id", "available_mask"] and merged[
                    col
                ].dtype in ["float64", "int64"]:
                    merged[col] = merged[col].ffill().bfill().fillna(0)

            result_dfs.append(merged)

        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)
            gap_count = (result["available_mask"] == 0).sum()
            if gap_count > 0:
                print(
                    f"  Filled {gap_count} gaps in time series (marked with available_mask=0)"
                )
            return result

        return df

    def create_models(
        self, input_size: int, exog_vars: List[str], max_steps: int = 200
    ):
        """Create models with improved loss functions for robust forecasting."""

        effective_input = max(min(input_size, 24), 2)

        # Use HuberLoss for robustness to outliers (price gaps, big moves)
        # Use DistributionLoss with StudentT for probabilistic forecasts with heavy tails
        # Enable start_padding for short time series (group training with few sessions per symbol)
        models = [
            # TFT: Temporal Fusion Transformer with StudentT distribution
            # Provides probabilistic forecasts with confidence intervals
            # StudentT handles outliers better than Normal distribution
            TFT(
                h=self.horizon,
                input_size=effective_input,
                hidden_size=32,
                max_steps=max_steps,
                scaler_type="robust",
                loss=DistributionLoss(distribution="StudentT", level=[80, 90]),
                random_seed=42,
                start_padding_enabled=True,
            ),
            # NBEATS: Interpretable basis decomposition with HuberLoss
            # HuberLoss is robust to outliers (combines MSE + MAE)
            NBEATS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=max_steps,
                scaler_type="robust",
                loss=HuberLoss(),
                random_seed=42,
                start_padding_enabled=True,
            ),
            # NHITS: Multi-scale hierarchical with HuberLoss
            NHITS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=max_steps,
                scaler_type="robust",
                loss=HuberLoss(),
                random_seed=42,
                start_padding_enabled=True,
            ),
        ]

        return models

    def train_and_forecast(
        self, df: pd.DataFrame, force_retrain: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train models and generate forecasts. Supports incremental training."""

        nf_df, exog_vars = self.prepare_data(df)
        current_data_count = len(nf_df)

        print(f"\n{'=' * 60}")
        print("PREPARED TIME SERIES")
        print(f"{'=' * 60}")
        print(f"  Records: {current_data_count}")
        print(f"  Date range: {nf_df['ds'].min()} to {nf_df['ds'].max()}")
        print(f"  Price range: {nf_df['y'].min():,.2f} to {nf_df['y'].max():,.2f}")
        print(f"  Alpha features: {exog_vars}")

        # Auto-adjust horizon if not enough data
        min_records = 3  # Absolute minimum
        if current_data_count < min_records:
            raise ValueError(
                f"Need at least {min_records} records (have {current_data_count})"
            )

        # Adjust horizon to fit available data
        max_horizon = max(current_data_count - 2, 1)
        if self.horizon > max_horizon:
            print(
                f"  Note: Adjusting horizon from {self.horizon} to {max_horizon} (limited data)"
            )
            self.horizon = max_horizon

        min_required = self.horizon + 2
        if current_data_count < min_required:
            raise ValueError(
                f"Need at least {min_required} records (have {current_data_count})"
            )

        input_size = max(current_data_count - self.horizon - 1, 2)

        # Check for saved model and determine training strategy
        saved_meta = None if force_retrain else self.load_model()
        use_fine_tuning = False
        max_steps = 200

        if saved_meta is not None:
            prev_count = saved_meta.get("data_count", 0)
            new_data = current_data_count - prev_count

            if new_data > 0:
                print(
                    f"\n  New data detected: {new_data} records ({prev_count} -> {current_data_count})"
                )
                print("  Strategy: Fine-tuning with reduced steps")
                use_fine_tuning = True
                max_steps = 50  # Reduced steps for fine-tuning
            elif new_data == 0:
                print("\n  No new data. Using saved model directly.")
                # Just predict with existing model
                print(
                    f"\nGenerating {self.horizon}-step forecast (using cached model)..."
                )

                # Predict only for requested symbol if using group training
                if self.group and self.symbol:
                    symbol_df = nf_df[nf_df["unique_id"] == self.symbol].copy()
                    if not symbol_df.empty:
                        forecasts = self.nf.predict(df=symbol_df)
                    else:
                        forecasts = self.nf.predict()
                else:
                    forecasts = self.nf.predict()

                return forecasts, nf_df
            else:
                print("\n  Data reduced. Retraining from scratch.")
                saved_meta = None

        if saved_meta is None:
            print("\n  Strategy: Training from scratch (no saved model)")

        models = self.create_models(input_size, exog_vars, max_steps=max_steps)

        # For fine-tuning: load saved weights into new models (warm-start)
        if use_fine_tuning:
            models = self.load_checkpoints_into_models(models)

        print(f"\nModels: {[m.__class__.__name__ for m in models]}")
        print(f"Lookback: {min(input_size, 24)} | Horizon: {self.horizon}")
        print(
            f"Max steps: {max_steps}"
            + (" (warm-start fine-tuning)" if use_fine_tuning else " (full training)")
        )

        # Initialize and train
        self.nf = NeuralForecast(models=models, freq="D")
        self.nf.fit(df=nf_df)

        # Save model for future use
        self.save_model(current_data_count)

        # Predict - only for requested symbol if using group training
        print(f"\nGenerating {self.horizon}-step forecast...")
        if self.group and self.symbol:
            # Filter input data to only the requested symbol for prediction
            symbol_df = nf_df[nf_df["unique_id"] == self.symbol].copy()
            if not symbol_df.empty:
                forecasts = self.nf.predict(df=symbol_df)
                print(f"  Predicting for {self.symbol} only")
            else:
                print(
                    f"  Warning: {self.symbol} not found in group data, predicting all"
                )
                forecasts = self.nf.predict()
        else:
            forecasts = self.nf.predict()

        return forecasts, nf_df

    def ensemble_forecast(
        self, forecasts: pd.DataFrame, last_price: float
    ) -> pd.DataFrame:
        """Create ensemble from all models and apply ARA/ARB clamping."""
        model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id"]]

        if model_cols:
            forecasts["ensemble"] = forecasts[model_cols].mean(axis=1)
            forecasts["ensemble_std"] = forecasts[model_cols].std(axis=1)
            forecasts["ensemble_low"] = (
                forecasts["ensemble"] - 1.96 * forecasts["ensemble_std"]
            )
            forecasts["ensemble_high"] = (
                forecasts["ensemble"] + 1.96 * forecasts["ensemble_std"]
            )

            # Apply ARA/ARB clamping to forecasts
            # This ensures predictions don't exceed daily price limits
            forecasts = clamp_forecast_series(
                forecasts, last_price, price_col="ensemble"
            )

            # Also clamp low/high to ARA/ARB
            if "low_clamped" in forecasts.columns:
                forecasts["ensemble_low"] = forecasts["low_clamped"]
            if "high_clamped" in forecasts.columns:
                forecasts["ensemble_high"] = forecasts["high_clamped"]

            print(
                f"\n  Forecasts clamped to ARA/ARB limits (based on {last_price:,.0f} IDR)"
            )

        return forecasts


def print_forecast_table(forecasts: pd.DataFrame, symbol: str, last_price: float):
    """Print formatted forecast results with ARA/ARB limits."""
    print(f"\n{'=' * 80}")
    print(f"FORECAST RESULTS: {symbol}")
    print(f"{'=' * 80}")

    # Show ARA/ARB limits for first day
    limit_info = get_daily_limit_info(last_price)
    print(f"\nARA/ARB Limits (Day 1 based on {last_price:,.0f} IDR):")
    print(
        f"  ARA (Upper): {limit_info['ara_price']:,.0f} (+{limit_info['max_gain_pct']:.1f}%)"
    )
    print(
        f"  ARB (Lower): {limit_info['arb_price']:,.0f} (-{limit_info['max_loss_pct']:.1f}%)"
    )

    # Determine which columns to show
    show_clamped = "ensemble_clamped" in forecasts.columns
    has_limits = "ara_limit" in forecasts.columns

    # Header
    print(f"\n{'Date':<12}", end="")
    if show_clamped:
        print(f"{'Forecast':>12}{'Clamped':>12}", end="")
        if has_limits:
            print(f"{'ARA':>10}{'ARB':>10}", end="")
    else:
        print(f"{'Forecast':>12}", end="")
    print(f"{'Change':>10}")
    print(
        "-" * (12 + (44 if show_clamped and has_limits else 22 if show_clamped else 22))
    )

    # Data - use clamped values if available
    for _, row in forecasts.iterrows():
        date_str = (
            row["ds"].strftime("%Y-%m-%d")
            if hasattr(row["ds"], "strftime")
            else str(row["ds"])[:10]
        )
        print(f"{date_str:<12}", end="")

        if show_clamped:
            orig = row.get("ensemble", 0)
            clamped = row.get("ensemble_clamped", orig)
            print(f"{orig:>12,.0f}{clamped:>12,.0f}", end="")
            if has_limits:
                ara = row.get("ara_limit", 0)
                arb = row.get("arb_limit", 0)
                print(f"{ara:>10,.0f}{arb:>10,.0f}", end="")
            pct = (clamped / last_price - 1) * 100
        else:
            val = row.get("ensemble", 0)
            print(f"{val:>12,.0f}", end="")
            pct = (val / last_price - 1) * 100

        print(f"{pct:>+9.2f}%")

    # Summary
    if "ensemble" in forecasts.columns:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Last price:      {last_price:>14,.0f}")

        if show_clamped:
            print("\n  Original (unclamped):")
            print(f"    Forecast mean:   {forecasts['ensemble'].mean():>12,.0f}")
            print(
                f"    Forecast range:  {forecasts['ensemble'].min():>12,.0f} - {forecasts['ensemble'].max():,.0f}"
            )

            print("\n  Clamped (within ARA/ARB):")
            print(
                f"    Forecast mean:   {forecasts['ensemble_clamped'].mean():>12,.0f}"
            )
            print(
                f"    Forecast range:  {forecasts['ensemble_clamped'].min():>12,.0f} - {forecasts['ensemble_clamped'].max():,.0f}"
            )

            final_forecast = forecasts["ensemble_clamped"].iloc[-1]
        else:
            print(f"  Forecast mean:   {forecasts['ensemble'].mean():>14,.0f}")
            print(
                f"  Forecast range:  {forecasts['ensemble'].min():>14,.0f} - {forecasts['ensemble'].max():,.0f}"
            )
            final_forecast = forecasts["ensemble"].iloc[-1]

        pct = (final_forecast / last_price - 1) * 100
        direction = "UP" if pct > 0 else "DOWN" if pct < 0 else "FLAT"
        print(f"\n  Expected move:   {pct:>+13.2f}% ({direction})")

        if "ensemble_std" in forecasts.columns:
            avg_std = forecasts["ensemble_std"].mean()
            print(
                f"  Uncertainty:     {avg_std:>14,.0f} (+/- {avg_std / last_price * 100:.2f}%)"
            )


def save_forecast_summary(
    forecasts: pd.DataFrame, symbol: str, last_price: float, filepath: Path
) -> None:
    """Save formatted forecast table to text file."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"FORECAST: {symbol}")
    lines.append("=" * 70)

    show_clamped = "ensemble_clamped" in forecasts.columns
    has_limits = "ara_limit" in forecasts.columns

    # Show ARA/ARB limits
    limit_info = get_daily_limit_info(last_price)
    lines.append(f"\nARA/ARB Limits (Day 1 based on {last_price:,.0f} IDR):")
    lines.append(f"  ARA (Upper): {limit_info['ara_price']:,.0f} (+{limit_info['max_gain_pct']:.1f}%)")
    lines.append(f"  ARB (Lower): {limit_info['arb_price']:,.0f} (-{limit_info['max_loss_pct']:.1f}%)")

    # Table header
    if show_clamped and has_limits:
        lines.append(f"\n{'Date':<12}{'Forecast':>12}{'Clamped':>12}{'ARA':>10}{'ARB':>10}{'Change':>10}")
        lines.append("-" * 66)
    elif show_clamped:
        lines.append(f"\n{'Date':<12}{'Forecast':>12}{'Clamped':>12}{'Change':>10}")
        lines.append("-" * 46)
    else:
        lines.append(f"\n{'Date':<12}{'Forecast':>12}{'Change':>10}")
        lines.append("-" * 34)

    # Table rows
    for _, row in forecasts.iterrows():
        d = row["ds"]
        date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]

        if show_clamped:
            ens = row.get("ensemble", 0) or 0
            clamped = row.get("ensemble_clamped", ens) or ens
            pct = (clamped / last_price - 1) * 100
            if has_limits:
                ara = row.get("ara_limit", 0) or 0
                arb = row.get("arb_limit", 0) or 0
                lines.append(f"{date_str:<12}{ens:>12,.0f}{clamped:>12,.0f}{ara:>10,.0f}{arb:>10,.0f}{pct:>+9.2f}%")
            else:
                lines.append(f"{date_str:<12}{ens:>12,.0f}{clamped:>12,.0f}{pct:>+9.2f}%")
        else:
            ens = row.get("ensemble", 0) or 0
            pct = (ens / last_price - 1) * 100
            lines.append(f"{date_str:<12}{ens:>12,.0f}{pct:>+9.2f}%")

    # Summary
    lines.append(f"\n{'=' * 70}")
    lines.append("SUMMARY (IDR)")
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

    if "ensemble_std" in forecasts.columns:
        avg_std = forecasts["ensemble_std"].mean()
        lines.append(f"  Uncertainty:     {avg_std:>14,.0f} (+/- {avg_std/last_price*100:.2f}%)")

    # Outlook
    final = fc.iloc[-1]
    pct_change = (final / last_price - 1) * 100
    outlook = "BULLISH" if pct_change > 2 else "BEARISH" if pct_change < -2 else "NEUTRAL"
    lines.append(f"\n  Outlook: {outlook} ({pct_change:+.2f}%)")

    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _add_forecast_table_to_plot(fig, forecasts: pd.DataFrame, last_price: float, symbol: str) -> None:
    """Add forecast table as text annotation on the right side of the plot."""
    show_clamped = "ensemble_clamped" in forecasts.columns
    has_limits = "ara_limit" in forecasts.columns

    lines = []
    lines.append("FORECAST TABLE")
    lines.append("-" * 30)

    # Show ARA/ARB limits
    limit_info = get_daily_limit_info(last_price)
    lines.append(f"ARA/ARB ({last_price:,.0f}):")
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
        date_str = d.strftime("%m-%d") if hasattr(d, "strftime") else str(d)[5:10]

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

    # Join and add to figure
    text = "\n".join(lines)
    fig.text(
        0.78, 0.5, text,
        fontsize=8,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9)
    )


def plot_results(
    historical: pd.DataFrame,
    forecasts: pd.DataFrame,
    symbol: str,
    alpha_df: pd.DataFrame,
    last_price: float = 0.0,
) -> Optional[Path]:
    """Plot price forecast with alpha features. Returns path to saved plot."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]}
        )

        # Price forecast
        ax1 = axes[0]
        ax1.plot(
            historical["ds"],
            historical["y"],
            "b-o",
            label="Historical",
            linewidth=2,
            markersize=6,
        )

        model_cols = [
            c
            for c in forecasts.columns
            if c
            not in [
                "ds",
                "unique_id",
                "ensemble",
                "ensemble_std",
                "ensemble_low",
                "ensemble_high",
            ]
        ]

        for col in model_cols:
            ax1.plot(forecasts["ds"], forecasts[col], "--", label=col, alpha=0.6)

        if "ensemble" in forecasts.columns:
            # Plot clamped forecast if available, otherwise original
            price_col = (
                "ensemble_clamped"
                if "ensemble_clamped" in forecasts.columns
                else "ensemble"
            )
            ax1.plot(
                forecasts["ds"],
                forecasts[price_col],
                "r-o",
                label="Ensemble (Clamped)"
                if "ensemble_clamped" in forecasts.columns
                else "Ensemble",
                linewidth=2,
                markersize=6,
            )
            if "ensemble_low" in forecasts.columns:
                ax1.fill_between(
                    forecasts["ds"],
                    forecasts["ensemble_low"],
                    forecasts["ensemble_high"],
                    alpha=0.2,
                    color="red",
                    label="95% CI",
                )

            # Plot ARA/ARB limits if available
            if "ara_limit" in forecasts.columns:
                ax1.plot(
                    forecasts["ds"],
                    forecasts["ara_limit"],
                    "g--",
                    alpha=0.7,
                    linewidth=1,
                    label="ARA (Upper Limit)",
                )
            if "arb_limit" in forecasts.columns:
                ax1.plot(
                    forecasts["ds"],
                    forecasts["arb_limit"],
                    "m--",
                    alpha=0.7,
                    linewidth=1,
                    label="ARB (Lower Limit)",
                )

        ax1.set_title(
            f"{symbol} - Price Forecast (TFT + Ensemble)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Orderbook Imbalance
        ax2 = axes[1]
        if "obi" in alpha_df.columns:
            colors = ["green" if x >= 0 else "red" for x in alpha_df["obi"]]
            ax2.bar(alpha_df["ds"], alpha_df["obi"], color=colors, alpha=0.7)
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_ylabel("OBI")
            ax2.set_title("Orderbook Imbalance (Buying Pressure)", fontsize=10)
            ax2.grid(True, alpha=0.3)

        # Concentration & Foreign Flow
        ax3 = axes[2]
        if "concentration_diff" in alpha_df.columns:
            ax3.plot(
                alpha_df["ds"],
                alpha_df["concentration_diff"],
                "b-o",
                label="Bandar Concentration",
                markersize=4,
            )
        if "foreign_ratio" in alpha_df.columns:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(
                alpha_df["ds"],
                alpha_df["foreign_ratio"],
                "g-s",
                label="Foreign Ratio",
                markersize=4,
                alpha=0.7,
            )
            ax3_twin.set_ylabel("Foreign Ratio", color="green")

        ax3.set_ylabel("Concentration Diff", color="blue")
        ax3.set_xlabel("Date")
        ax3.set_title("Bandar Concentration & Foreign Flow", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper left")

        # Add forecast table as text annotation
        if last_price > 0:
            _add_forecast_table_to_plot(fig, forecasts, last_price, symbol)

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)

        plot_dir = Path("plot") / symbol
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            plot_dir / f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {output_path}")
        plt.close()
        return output_path

    except ImportError:
        print("\nMatplotlib not available")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Stock Forecasting with Alpha Features (TFT + NeuralForecast)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Alpha Features Used:
  - OBI (Orderbook Imbalance): Buying/selling pressure from order depth
  - Bandar Concentration: Top 3 broker share (institutional intent)
  - Foreign Flow: Net foreign transaction value
  - Volatility: Intraday price range
  - AccDist Signal: Accumulation/Distribution from broker patterns

Examples:
  python forecast.py ARCI
  python forecast.py ARCI --horizon 10 --plot
  python forecast.py --list
        """,
    )

    parser.add_argument("symbol", nargs="?", help="Stock symbol")
    parser.add_argument("--horizon", "-n", type=int, default=5, help="Forecast horizon")
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable plot generation"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List symbols")
    parser.add_argument("--source", "-s", default="sources", help="Data directory")
    parser.add_argument(
        "--retrain",
        "-r",
        action="store_true",
        help="Force retrain from scratch (ignore saved model)",
    )
    parser.add_argument(
        "--group",
        "-g",
        help="Train with symbol group (e.g., 'banking', 'mining'). Edit models/groups.json to define groups.",
    )
    parser.add_argument(
        "--list-groups", action="store_true", help="List available groups"
    )

    # Add watchlist arguments
    add_watchlist_args(parser)

    # Add telegram arguments
    add_telegram_args(parser)

    args = parser.parse_args()

    # Validate watchlist arguments
    if hasattr(args, "watchlist") and args.watchlist:
        try:
            validate_watchlist_args(args)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Validate telegram arguments
    if hasattr(args, "telegram") and args.telegram:
        try:
            validate_telegram_args(args)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    engine = MarketAlphaEngine(args.source)
    groups = load_groups()

    # List groups
    if args.list_groups:
        if not groups:
            print("No groups defined. Create models/groups.json to define groups.")
        else:
            RED = "\033[31m"
            RESET = "\033[0m"
            available_symbols = engine.get_available_symbols()
            print("Available groups (edit models/groups.json to modify):")
            for name, symbols in groups.items():
                colored = [
                    s if s in available_symbols else f"{RED}{s}{RESET}" for s in symbols
                ]
                print(f"  {name}: {', '.join(colored)}")
        return

    if args.list:
        symbols = engine.get_available_symbols()
        print("Available symbols:")
        for s in sorted(symbols):
            sessions = len([d for d in (engine.base_path / s).iterdir() if d.is_dir()])
            group = find_group_for_symbol(s, groups)
            group_str = f" [{group}]" if group else ""
            print(f"  {s}: {sessions} session(s){group_str}")
        return

    if not args.symbol:
        parser.print_help()
        sys.exit(1)

    symbol = args.symbol.upper()

    if symbol not in engine.get_available_symbols():
        print(f"Error: {symbol} not found")
        print(f"Available: {', '.join(sorted(engine.get_available_symbols()))}")
        sys.exit(1)

    # Determine group to use
    group_name = args.group
    if group_name and group_name not in groups:
        print(f"Error: Group '{group_name}' not found")
        print(f"Available groups: {', '.join(groups.keys())}")
        print("Edit models/groups.json to add groups.")
        sys.exit(1)

    # Auto-detect group if not specified
    if not group_name:
        group_name = find_group_for_symbol(symbol, groups)
        if group_name:
            print(f"Auto-detected group: {group_name}")

    print(f"\n{'=' * 70}")
    print(f"STOCK FORECASTER - {symbol}")
    print(f"{'=' * 70}")
    print("Using: TFT (Temporal Fusion Transformer) + NBEATS + NHITS")
    print(f"Horizon: {args.horizon} periods")
    if group_name:
        print(f"Group training: {group_name} ({', '.join(groups[group_name])})")

    try:
        # Load data based on group or single symbol
        if group_name:
            print(f"\nLoading group '{group_name}' data...")
            df = engine.load_group_data(groups[group_name])
            symbols_loaded = df["unique_id"].nunique()
            print(f"\nLoaded {len(df)} total sessions from {symbols_loaded} symbols")
        else:
            print(f"\nLoading alpha features for {symbol}...")
            df = engine.load_symbol_data(symbol)
            print(f"\nLoaded {len(df)} sessions")

        print(f"Price range: {df['y'].min():,.2f} - {df['y'].max():,.2f}")

        # Get last price for the target symbol
        symbol_data = df[df["unique_id"] == symbol] if group_name else df
        last_price = (
            symbol_data["y"].iloc[-1] if not symbol_data.empty else df["y"].iloc[-1]
        )
        print(f"Last price ({symbol}): {last_price:,.2f}")

        # Show alpha feature summary
        print("\nAlpha Feature Summary:")
        print(f"  OBI range: {df['obi'].min():.3f} to {df['obi'].max():.3f}")
        print(f"  Buy Concentration avg: {df['buy_concentration'].mean():.2%}")
        print(f"  Foreign Net total: {df['foreign_net'].sum():,.0f}")

        forecaster = StockForecaster(
            horizon=args.horizon, symbol=symbol, group=group_name
        )
        forecasts, historical = forecaster.train_and_forecast(
            df, force_retrain=args.retrain
        )
        forecasts = forecaster.ensemble_forecast(forecasts, last_price)

        print_forecast_table(forecasts, symbol, last_price)

        plot_path = None
        if not args.no_plot:
            # Use symbol-specific data for plots when using group training
            plot_data = symbol_data if group_name and not symbol_data.empty else df
            # Filter historical to only show target symbol when using group training
            if group_name and "unique_id" in historical.columns:
                hist_plot = historical[historical["unique_id"] == symbol].copy()
            else:
                hist_plot = historical
            plot_path = plot_results(hist_plot, forecasts, symbol, plot_data, last_price)

        symbol_csv_dir = CSV_DIR / symbol
        symbol_csv_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Add change_pct column before saving
        df_to_save = forecasts.copy()
        price_col = "ensemble_clamped" if "ensemble_clamped" in df_to_save.columns else "ensemble"
        if price_col in df_to_save.columns:
            df_to_save["change_pct"] = (df_to_save[price_col] / last_price - 1) * 100

        output_file = symbol_csv_dir / f"forecast_{timestamp}.csv"
        df_to_save.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

        # Save summary text file
        summary_file = symbol_csv_dir / f"forecast_{timestamp}_summary.txt"
        save_forecast_summary(forecasts, symbol, last_price, summary_file)
        print(f"Saved: {summary_file}")

        # Update watchlist if requested
        if args.watchlist:
            # Determine outlook from forecast
            price_col = (
                "ensemble_clamped"
                if "ensemble_clamped" in forecasts.columns
                else "ensemble"
            )
            if price_col in forecasts.columns:
                final_forecast = forecasts[price_col].iloc[-1]
                pct_change = (final_forecast / last_price - 1) * 100

                if pct_change > 0:
                    outlook = "BULLISH"
                elif pct_change < 0:
                    outlook = "BEARISH"
                else:
                    outlook = "NEUTRAL"

                print(
                    f"\nForecast outlook for {symbol}: {outlook} ({pct_change:+.2f}%)"
                )

                # Filter based on bullish/bearish flags
                symbols_with_outlook = [{"symbol": symbol, "outlook": outlook}]
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
                            debug=getattr(args, "wl_debug", False),
                        )
                        print_watchlist_summary(result)
                    except Exception as e:
                        print(f"\nWatchlist update failed: {e}")
                else:
                    filter_type = (
                        "bullish" if args.bullish else "bearish" if args.bearish else ""
                    )
                    print(
                        f"\nNo symbols to add to watchlist (outlook is {outlook}, filter: {filter_type})"
                    )

        # Send Telegram notification if requested
        if args.telegram:
            price_col = (
                "ensemble_clamped"
                if "ensemble_clamped" in forecasts.columns
                else "ensemble"
            )
            if price_col in forecasts.columns:
                final_forecast = forecasts[price_col].iloc[-1]
                pct_change = (final_forecast / last_price - 1) * 100

                if pct_change > 2:
                    outlook = "BULLISH"
                elif pct_change < -2:
                    outlook = "BEARISH"
                else:
                    outlook = "NEUTRAL"

                # Prepare forecast data for Telegram
                tg_forecasts = []
                for _, row in forecasts.iterrows():
                    d = row["ds"]
                    date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                    price = row.get(price_col, 0) or 0
                    fc_pct = (price / last_price - 1) * 100
                    tg_forecasts.append({
                        "date": date_str,
                        "price": price,
                        "change_pct": fc_pct,
                    })

                # Get ARA/ARB info
                limit_info = get_daily_limit_info(last_price)
                ara_arb_info = {
                    "ara_price": limit_info["ara_price"],
                    "arb_price": limit_info["arb_price"],
                    "max_gain_pct": limit_info["max_gain_pct"],
                    "max_loss_pct": limit_info["max_loss_pct"],
                }

                print(f"\nSending Telegram notification...")
                success = send_forecast_notification(
                    symbol=symbol,
                    current_price=last_price,
                    forecasts=tg_forecasts,
                    outlook=outlook,
                    change_pct=pct_change,
                    currency="IDR",
                    ara_arb_info=ara_arb_info,
                    script_name="Daily Forecast",
                    silent=getattr(args, "tg_silent", False),
                    plot_path=str(plot_path) if plot_path else None,
                )
                if success:
                    print(f"  Telegram notification sent successfully")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
