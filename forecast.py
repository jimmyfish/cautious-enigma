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
from neuralforecast.models import LSTM, NBEATS, NHITS, TFT

warnings.filterwarnings("ignore")


class MarketAlphaEngine:
    """
    Extract alpha-generating features from market data snapshots.
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

    def calculate_orderbook_imbalance(self, data: dict) -> float:
        """
        Calculate Orderbook Imbalance (OBI) - The "Pressure"
        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Range: -1 (all sellers) to +1 (all buyers)
        """
        if not data or "data" not in data:
            return 0.0

        d = data["data"]
        bids = d.get("bid", [])
        asks = d.get("offer", [])

        bid_volume = sum(self.parse_number(b.get("volume", 0)) for b in bids)
        ask_volume = sum(self.parse_number(a.get("volume", 0)) for a in asks)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def calculate_bandar_concentration(self, data: dict) -> Tuple[float, float]:
        """
        Calculate Bandar (Market Maker) Concentration - The "Intent"
        Top 3 brokers' share of total volume indicates institutional activity.
        Returns: (buy_concentration, sell_concentration)
        """
        if not data or "data" not in data:
            return 0.0, 0.0

        broker_summary = data["data"].get("broker_summary", {})

        # Buy concentration
        brokers_buy = broker_summary.get("brokers_buy", [])
        if brokers_buy:
            top3_buy = sum(
                abs(self.parse_number(b.get("blot", 0))) for b in brokers_buy[:3]
            )
            total_buy = sum(
                abs(self.parse_number(b.get("blot", 0))) for b in brokers_buy
            )
            buy_concentration = top3_buy / (total_buy + 1e-5)
        else:
            buy_concentration = 0.0

        # Sell concentration
        brokers_sell = broker_summary.get("brokers_sell", [])
        if brokers_sell:
            top3_sell = sum(
                abs(self.parse_number(b.get("slot", 0))) for b in brokers_sell[:3]
            )
            total_sell = sum(
                abs(self.parse_number(b.get("slot", 0))) for b in brokers_sell
            )
            sell_concentration = top3_sell / (total_sell + 1e-5)
        else:
            sell_concentration = 0.0

        return buy_concentration, sell_concentration

    def calculate_foreign_flow(self, data: dict) -> Tuple[float, float, float]:
        """
        Calculate Foreign Flow Dominance.
        Returns: (foreign_net_value, foreign_net_volume, foreign_ratio)
        """
        if not data or "data" not in data:
            return 0.0, 0.0, 0.0

        d = data["data"]

        # From findata.json structure
        summary = d.get("summary", {})
        if summary:
            fb = summary.get("foreign_buy", {}).get("value", {}).get("raw", 0)
            fs = summary.get("foreign_sell", {}).get("value", {}).get("raw", 0)
            foreign_net_value = self.parse_number(fb) - self.parse_number(fs)

            vol_data = summary.get("volume", {})
            fvb = vol_data.get("foreign_buy", {}).get("value", {}).get("raw", 0)
            fvs = vol_data.get("foreign_sell", {}).get("value", {}).get("raw", 0)
            foreign_net_volume = self.parse_number(fvb) - self.parse_number(fvs)

            total_value = self.parse_number(fb) + self.parse_number(fs)
            foreign_ratio = foreign_net_value / (total_value + 1e-5)

            return foreign_net_value, foreign_net_volume, foreign_ratio

        # Fallback to price-feed structure
        fnet = self.parse_number(d.get("fnet", 0))
        return fnet, 0.0, 0.0

    def calculate_bandar_accdist(self, data: dict) -> Tuple[float, float]:
        """
        Calculate accumulation/distribution signal from bandar detector.
        Returns: (accdist_signal, net_percent)
        """
        if not data or "data" not in data:
            return 0.0, 0.0

        bd = data["data"].get("bandar_detector", {})
        if not bd:
            return 0.0, 0.0

        # Accumulation/Distribution signal
        avg_data = bd.get("avg", {})
        accdist_map = {
            "Big Acc": 2.0,
            "Normal Acc": 1.0,
            "Neutral": 0.0,
            "Normal Dist": -1.0,
            "Big Dist": -2.0,
        }
        accdist = accdist_map.get(avg_data.get("accdist", "Neutral"), 0.0)
        net_percent = self.parse_number(avg_data.get("percent", 0))

        return accdist, net_percent

    def extract_session_features(self, symbol: str, session: str) -> Optional[Dict]:
        """Extract all alpha features from a single session."""
        session_path = self.base_path / symbol / session

        # Load all data files
        price_feed = self.load_json(session_path / "price-feed.json")
        orderbook = self.load_json(session_path / "orderbook.json")
        market_detector = self.load_json(session_path / "market-detector.json")
        findata = self.load_json(session_path / "findata.json")

        # Use price-feed as orderbook fallback (same structure)
        if not orderbook:
            orderbook = price_feed

        if not price_feed or "data" not in price_feed:
            return None

        pf = price_feed["data"]

        # Get price data
        close = self.parse_number(pf.get("close", 0) or pf.get("lastprice", 0))
        if close == 0:
            return None

        high = self.parse_number(pf.get("high", 0))
        low = self.parse_number(pf.get("low", 0))
        open_price = self.parse_number(pf.get("open", 0))
        volume = self.parse_number(pf.get("volume", 0))
        value = self.parse_number(pf.get("value", 0))
        frequency = self.parse_number(pf.get("frequency", 0))

        # Extract date
        ds = None
        for f in session_path.glob("analysis-data-*.json"):
            try:
                date_str = f.stem.replace("analysis-data-", "").split("_")[0]
                ds = datetime.strptime(date_str, "%Y-%m-%d")
                break
            except:
                pass

        if not ds and market_detector and "data" in market_detector:
            date_str = market_detector["data"].get("from")
            if date_str:
                try:
                    ds = datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass

        if not ds:
            ds = datetime.now() - timedelta(days=int(session))

        # Calculate alpha features
        obi = self.calculate_orderbook_imbalance(orderbook)
        buy_conc, sell_conc = self.calculate_bandar_concentration(market_detector)
        foreign_net_val, foreign_net_vol, foreign_ratio = self.calculate_foreign_flow(
            findata or price_feed
        )
        accdist, net_percent = self.calculate_bandar_accdist(market_detector)

        # Volatility
        volatility = (high - low) / (close + 1e-5) if close > 0 else 0.0

        # Price momentum
        pct_change = self.parse_number(pf.get("percentage_change", 0))

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
            "frequency": frequency,
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

    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
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
                print(
                    f"  Session {session}: close={features['y']:,.0f}, OBI={features['obi']:.3f}, "
                    f"BuyConc={features['buy_concentration']:.2f}, ForeignNet={features['foreign_net']:,.0f}"
                )

        if not features_list:
            raise ValueError(f"No valid data found for: {symbol}")

        df = pd.DataFrame(features_list)
        df = df.sort_values("ds").reset_index(drop=True)

        return df

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

    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.nf = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data in NeuralForecast format with alpha features."""

        # Define exogenous features for TFT attention
        exog_features = [
            "obi",
            "concentration_diff",
            "foreign_ratio",
            "volatility",
            "accdist",
            "pct_change",
        ]

        # Filter to available features
        available_exog = [c for c in exog_features if c in df.columns]

        # Build NeuralForecast dataframe
        nf_df = pd.DataFrame(
            {"unique_id": df["unique_id"], "ds": df["ds"], "y": df["y"]}
        )

        # Add exogenous features
        for col in available_exog:
            nf_df[col] = df[col].fillna(0)

        # Clean data
        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)

        return nf_df, available_exog

    def create_models(self, input_size: int, exog_vars: List[str]):
        """Create models including TFT with attention on alpha features."""

        effective_input = max(min(input_size, 24), 2)

        models = [
            # TFT: Temporal Fusion Transformer with attention on alpha features
            # This model learns WHICH features matter most at each timestep
            TFT(
                h=self.horizon,
                input_size=effective_input,
                hidden_size=32,
                max_steps=200,
                scaler_type="robust",
                random_seed=42,
            ),
            # NBEATS: Interpretable basis decomposition
            NBEATS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type="robust",
                random_seed=42,
            ),
            # NHITS: Multi-scale hierarchical
            NHITS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type="robust",
                random_seed=42,
            ),
        ]

        return models

    def train_and_forecast(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train models and generate forecasts."""

        nf_df, exog_vars = self.prepare_data(df)

        print(f"\n{'=' * 60}")
        print("PREPARED TIME SERIES")
        print(f"{'=' * 60}")
        print(f"  Records: {len(nf_df)}")
        print(f"  Date range: {nf_df['ds'].min()} to {nf_df['ds'].max()}")
        print(f"  Price range: {nf_df['y'].min():,.2f} to {nf_df['y'].max():,.2f}")
        print(f"  Alpha features: {exog_vars}")

        # Auto-adjust horizon if not enough data
        min_records = 3  # Absolute minimum
        if len(nf_df) < min_records:
            raise ValueError(f"Need at least {min_records} records (have {len(nf_df)})")

        # Adjust horizon to fit available data
        max_horizon = max(len(nf_df) - 2, 1)
        if self.horizon > max_horizon:
            print(f"  Note: Adjusting horizon from {self.horizon} to {max_horizon} (limited data)")
            self.horizon = max_horizon

        min_required = self.horizon + 2
        if len(nf_df) < min_required:
            raise ValueError(
                f"Need at least {min_required} records (have {len(nf_df)})"
            )

        input_size = max(len(nf_df) - self.horizon - 1, 2)
        models = self.create_models(input_size, exog_vars)

        print(f"\nModels: {[m.__class__.__name__ for m in models]}")
        print(f"Lookback: {min(input_size, 24)} | Horizon: {self.horizon}")

        # Initialize and train
        self.nf = NeuralForecast(models=models, freq="D")
        self.nf.fit(df=nf_df)

        # Predict
        print(f"\nGenerating {self.horizon}-step forecast...")
        forecasts = self.nf.predict()

        return forecasts, nf_df

    def ensemble_forecast(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble from all models."""
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

        return forecasts


def print_forecast_table(forecasts: pd.DataFrame, symbol: str, last_price: float):
    """Print formatted forecast results."""
    print(f"\n{'=' * 70}")
    print(f"FORECAST RESULTS: {symbol}")
    print(f"{'=' * 70}")

    model_cols = [
        c
        for c in forecasts.columns
        if c not in ["ds", "unique_id", "ensemble_std", "ensemble_low", "ensemble_high"]
    ]

    # Header
    print(f"\n{'Date':<14}", end="")
    for col in model_cols:
        print(f"{col[:10]:>12}", end="")
    print()
    print("-" * (14 + 12 * len(model_cols)))

    # Data
    for _, row in forecasts.iterrows():
        date_str = (
            row["ds"].strftime("%Y-%m-%d")
            if hasattr(row["ds"], "strftime")
            else str(row["ds"])[:10]
        )
        print(f"{date_str:<14}", end="")
        for col in model_cols:
            print(f"{row[col]:>12,.2f}", end="")
        print()

    # Summary
    if "ensemble" in forecasts.columns:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Last price:      {last_price:>14,.2f}")
        print(f"  Forecast mean:   {forecasts['ensemble'].mean():>14,.2f}")
        print(
            f"  Forecast range:  {forecasts['ensemble'].min():>14,.2f} - {forecasts['ensemble'].max():,.2f}"
        )

        final_forecast = forecasts["ensemble"].iloc[-1]
        pct = (final_forecast / last_price - 1) * 100
        direction = "UP" if pct > 0 else "DOWN" if pct < 0 else "FLAT"
        print(f"  Expected move:   {pct:>+13.2f}% ({direction})")

        if "ensemble_std" in forecasts.columns:
            avg_std = forecasts["ensemble_std"].mean()
            print(
                f"  Uncertainty:     {avg_std:>14,.2f} (+/- {avg_std / last_price * 100:.2f}%)"
            )


def plot_results(
    historical: pd.DataFrame,
    forecasts: pd.DataFrame,
    symbol: str,
    alpha_df: pd.DataFrame,
):
    """Plot price forecast with alpha features."""
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
            ax1.plot(
                forecasts["ds"],
                forecasts["ensemble"],
                "r-o",
                label="Ensemble",
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

        plt.tight_layout()

        output_path = (
            f"forecast_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {output_path}")
        plt.show()

    except ImportError:
        print("\nMatplotlib not available")


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
    parser.add_argument("--plot", "-p", action="store_true", help="Show plots")
    parser.add_argument("--list", "-l", action="store_true", help="List symbols")
    parser.add_argument("--source", "-s", default="sources", help="Data directory")

    args = parser.parse_args()

    engine = MarketAlphaEngine(args.source)

    if args.list:
        symbols = engine.get_available_symbols()
        print("Available symbols:")
        for s in sorted(symbols):
            sessions = len(list((engine.base_path / s).iterdir()))
            print(f"  {s}: {sessions} session(s)")
        return

    if not args.symbol:
        parser.print_help()
        sys.exit(1)

    symbol = args.symbol.upper()

    if symbol not in engine.get_available_symbols():
        print(f"Error: {symbol} not found")
        print(f"Available: {', '.join(sorted(engine.get_available_symbols()))}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"STOCK FORECASTER - {symbol}")
    print(f"{'=' * 70}")
    print("Using: TFT (Temporal Fusion Transformer) + NBEATS + NHITS")
    print(f"Horizon: {args.horizon} periods")

    try:
        print(f"\nLoading alpha features for {symbol}...")
        df = engine.load_symbol_data(symbol)

        print(f"\nLoaded {len(df)} sessions")
        print(f"Price range: {df['y'].min():,.2f} - {df['y'].max():,.2f}")
        print(f"Last price: {df['y'].iloc[-1]:,.2f}")

        # Show alpha feature summary
        print(f"\nAlpha Feature Summary:")
        print(f"  OBI range: {df['obi'].min():.3f} to {df['obi'].max():.3f}")
        print(f"  Buy Concentration avg: {df['buy_concentration'].mean():.2%}")
        print(f"  Foreign Net total: {df['foreign_net'].sum():,.0f}")

        forecaster = StockForecaster(horizon=args.horizon)
        forecasts, historical = forecaster.train_and_forecast(df)
        forecasts = forecaster.ensemble_forecast(forecasts)

        print_forecast_table(forecasts, symbol, df["y"].iloc[-1])

        if args.plot:
            plot_results(historical, forecasts, symbol, df)

        output_file = (
            f"forecast_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        forecasts.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
