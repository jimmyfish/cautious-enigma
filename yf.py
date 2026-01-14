#!/usr/bin/env python3
"""
Stock Forecasting using Yahoo Finance data.
Simple OHLCV-based forecasting for any stock available on Yahoo Finance.

Usage:
    python yf.py AAPL                      # Daily forecast for Apple
    python yf.py BBRI.JK --horizon 10      # Indonesian stock
    python yf.py AAPL --interval 1h        # Hourly data
    python yf.py AAPL --period 6mo --plot  # 6 months history with chart

Examples:
    python yf.py AAPL --horizon 5 --plot
    python yf.py BBRI.JK --interval 1d --horizon 10
    python yf.py TSLA --interval 1h --period 5d --horizon 24
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM

warnings.filterwarnings("ignore")


# Valid yfinance intervals and their max periods
VALID_INTERVALS = {
    "1m": {"max_period": "7d", "freq": "1min"},
    "2m": {"max_period": "60d", "freq": "2min"},
    "5m": {"max_period": "60d", "freq": "5min"},
    "15m": {"max_period": "60d", "freq": "15min"},
    "30m": {"max_period": "60d", "freq": "30min"},
    "1h": {"max_period": "730d", "freq": "1h"},
    "1d": {"max_period": "max", "freq": "D"},
    "1wk": {"max_period": "max", "freq": "W"},
    "1mo": {"max_period": "max", "freq": "MS"},
}


class YFinanceLoader:
    """Load stock data from Yahoo Finance."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self.info = None

    def get_info(self) -> dict:
        """Get stock info."""
        if self.info is None:
            try:
                self.info = self.ticker.info
            except:
                self.info = {}
        return self.info

    def get_name(self) -> str:
        """Get company name."""
        info = self.get_info()
        return info.get("shortName", info.get("longName", self.symbol))

    def get_currency(self) -> str:
        """Get currency."""
        info = self.get_info()
        return info.get("currency", "USD")

    def load_data(self, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        """
        Load historical data from Yahoo Finance.

        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {list(VALID_INTERVALS.keys())}")

        print(f"Fetching {self.symbol} data: period={period}, interval={interval}")

        df = self.ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {self.symbol}")

        # Clean column names
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Rename datetime column
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ds"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "ds"})

        # Ensure datetime
        df["ds"] = pd.to_datetime(df["ds"])

        # Remove timezone info for NeuralForecast compatibility
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features to OHLCV data."""
        # Returns
        df["returns"] = df["close"].pct_change()

        # Volatility
        df["volatility"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

        # Moving averages
        df["sma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["sma_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()

        # Price vs MA
        df["price_vs_sma5"] = (df["close"] / df["sma_5"]) - 1
        df["price_vs_sma20"] = (df["close"] / df["sma_20"]) - 1

        # Volume features
        df["vol_ma"] = df["volume"].rolling(5, min_periods=1).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma"] + 1)

        # Momentum
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)

        # RSI-like feature (simplified)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

        # MACD-like feature (simplified)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = (ema12 - ema26) / df["close"]

        # Bollinger Band position
        sma20 = df["close"].rolling(20, min_periods=1).mean()
        std20 = df["close"].rolling(20, min_periods=1).std()
        df["bb_position"] = (df["close"] - sma20) / (2 * std20 + 1e-8)

        return df.fillna(0)


class YFinanceForecaster:
    """Forecast using NeuralForecast models."""

    def __init__(self, horizon: int = 5, interval: str = "1d"):
        self.horizon = horizon
        self.interval = interval
        self.nf = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare NeuralForecast format."""
        exog_cols = [
            "volatility", "returns", "price_vs_sma5", "price_vs_sma20",
            "vol_ratio", "momentum_5", "rsi", "macd", "bb_position"
        ]
        available = [c for c in exog_cols if c in df.columns]

        nf_df = pd.DataFrame({
            "unique_id": "STOCK",
            "ds": df["ds"],
            "y": df["close"]
        })

        for col in available:
            nf_df[col] = df[col].fillna(0)

        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)
        return nf_df, available

    def forecast(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Train and generate forecasts."""
        nf_df, features = self.prepare_data(df)
        n = len(nf_df)

        print(f"\n{'=' * 60}")
        print("DATA SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Records: {n}")
        print(f"  Period: {nf_df['ds'].min().strftime('%Y-%m-%d')} to {nf_df['ds'].max().strftime('%Y-%m-%d')}")
        print(f"  Price range: {nf_df['y'].min():,.2f} - {nf_df['y'].max():,.2f}")
        print(f"  Last price: {nf_df['y'].iloc[-1]:,.2f}")
        print(f"  Features: {len(features)}")

        if n < 10:
            raise ValueError(f"Need at least 10 data points (have {n})")

        # Adjust horizon if needed
        max_h = n // 2
        if self.horizon > max_h:
            print(f"  Note: Limiting horizon from {self.horizon} to {max_h}")
            self.horizon = max_h

        input_size = min(n - self.horizon - 1, 30)
        input_size = max(input_size, 5)

        print(f"\n{'=' * 60}")
        print("FORECAST CONFIG")
        print(f"{'=' * 60}")
        print(f"  Horizon: {self.horizon} periods")
        print(f"  Lookback: {input_size} periods")
        print(f"  Interval: {self.interval}")
        print(f"  Exog features: {features}")

        # NBEATS doesn't support exogenous variables, use only for NHITS and LSTM
        models = [
            NBEATS(h=self.horizon, input_size=input_size, max_steps=300, scaler_type="robust", random_seed=42),
            NHITS(h=self.horizon, input_size=input_size, max_steps=300, scaler_type="robust", random_seed=42,
                  hist_exog_list=features if features else None),
            LSTM(h=self.horizon, input_size=input_size, max_steps=300, scaler_type="robust", random_seed=42,
                 hist_exog_list=features if features else None),
        ]

        freq = VALID_INTERVALS.get(self.interval, {}).get("freq", "D")
        print(f"  Frequency: {freq}")
        print(f"\nTraining: {[m.__class__.__name__ for m in models]}")

        self.nf = NeuralForecast(models=models, freq=freq)
        self.nf.fit(df=nf_df)

        print(f"Generating {self.horizon}-step forecast...")
        forecasts = self.nf.predict()

        # Ensemble
        model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id"]]
        if model_cols:
            forecasts["ensemble"] = forecasts[model_cols].mean(axis=1)
            forecasts["std"] = forecasts[model_cols].std(axis=1)
            forecasts["low"] = forecasts["ensemble"] - 1.96 * forecasts["std"]
            forecasts["high"] = forecasts["ensemble"] + 1.96 * forecasts["std"]

        meta = {
            "horizon": self.horizon,
            "interval": self.interval,
            "input_size": input_size,
            "n_records": n,
        }

        return forecasts, nf_df, meta


def print_results(forecasts: pd.DataFrame, last_price: float, symbol: str, currency: str):
    """Print forecast results."""
    print(f"\n{'=' * 70}")
    print(f"FORECAST: {symbol}")
    print(f"{'=' * 70}")

    cols = [c for c in forecasts.columns if c not in ["unique_id", "std"]]

    # Header
    print(f"\n{'Date':<12}", end="")
    for c in cols:
        if c == "ds":
            continue
        print(f"{c[:8]:>12}", end="")
    print()
    print("-" * (12 + 12 * (len(cols) - 1)))

    # Data
    for _, row in forecasts.iterrows():
        d = row["ds"]
        if hasattr(d, "strftime"):
            date_str = d.strftime("%Y-%m-%d") if d.hour == 0 else d.strftime("%m-%d %H:%M")
        else:
            date_str = str(d)[:10]
        print(f"{date_str:<12}", end="")
        for c in cols:
            if c == "ds":
                continue
            print(f"{row[c]:>12,.2f}", end="")
        print()

    # Summary
    if "ensemble" in forecasts.columns:
        print(f"\n{'=' * 70}")
        print(f"SUMMARY ({currency})")
        print(f"{'=' * 70}")
        print(f"  Current price:   {last_price:>14,.2f}")
        print(f"  Forecast avg:    {forecasts['ensemble'].mean():>14,.2f} ({(forecasts['ensemble'].mean()/last_price-1)*100:+.2f}%)")
        print(f"  Forecast high:   {forecasts['ensemble'].max():>14,.2f} ({(forecasts['ensemble'].max()/last_price-1)*100:+.2f}%)")
        print(f"  Forecast low:    {forecasts['ensemble'].min():>14,.2f} ({(forecasts['ensemble'].min()/last_price-1)*100:+.2f}%)")
        print(f"  Final forecast:  {forecasts['ensemble'].iloc[-1]:>14,.2f} ({(forecasts['ensemble'].iloc[-1]/last_price-1)*100:+.2f}%)")

        if "std" in forecasts.columns:
            avg_std = forecasts["std"].mean()
            print(f"  Uncertainty:     {avg_std:>14,.2f} (+/- {avg_std/last_price*100:.2f}%)")

        # Direction
        final = forecasts["ensemble"].iloc[-1]
        pct_change = (final / last_price - 1) * 100
        if pct_change > 2:
            outlook = "BULLISH"
        elif pct_change < -2:
            outlook = "BEARISH"
        else:
            outlook = "NEUTRAL"
        print(f"\n  Outlook: {outlook} ({pct_change:+.2f}%)")


def plot_forecast(df: pd.DataFrame, forecasts: pd.DataFrame, symbol: str, name: str):
    """Plot historical prices and forecast."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

        # Price chart
        ax1 = axes[0]

        # Historical
        ax1.plot(df["ds"], df["close"], "b-", label="Historical", linewidth=1.5)

        # Forecast
        model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id", "ensemble", "std", "low", "high"]]
        for col in model_cols:
            ax1.plot(forecasts["ds"], forecasts[col], "--", alpha=0.4, linewidth=1, label=col)

        if "ensemble" in forecasts.columns:
            ax1.plot(forecasts["ds"], forecasts["ensemble"], "r-", linewidth=2, label="Ensemble")
            if "low" in forecasts.columns:
                ax1.fill_between(forecasts["ds"], forecasts["low"], forecasts["high"],
                                 alpha=0.2, color="red", label="95% CI")

        # Moving averages
        if "sma_20" in df.columns:
            ax1.plot(df["ds"], df["sma_20"], "g--", alpha=0.5, linewidth=1, label="SMA20")

        ax1.axvline(x=df["ds"].iloc[-1], color="gray", linestyle="--", alpha=0.5)
        ax1.axhline(y=df["close"].iloc[-1], color="gray", linestyle=":", alpha=0.5)

        ax1.set_title(f"{symbol} - {name}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Volume
        ax2 = axes[1]
        colors = ["green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red"
                  for i in range(len(df))]
        ax2.bar(df["ds"], df["volume"], color=colors, alpha=0.7)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)

        # RSI
        ax3 = axes[2]
        if "rsi" in df.columns:
            ax3.plot(df["ds"], df["rsi"], "purple", linewidth=1)
            ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5)
            ax3.axhline(y=30, color="green", linestyle="--", alpha=0.5)
            ax3.fill_between(df["ds"], 30, 70, alpha=0.1, color="gray")
            ax3.set_ylabel("RSI")
            ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("Date")

        plt.tight_layout()

        out = f"yf_{symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {out}")
        plt.show()

    except ImportError:
        print("\nMatplotlib not available")


def main():
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
    python yf.py AAPL                          # Apple, daily, 5-day forecast
    python yf.py BBRI.JK --horizon 10          # Indonesian bank stock
    python yf.py TSLA --interval 1h --period 5d --horizon 24
    python yf.py GOOGL --period 1y --plot      # 1 year history with chart
    python yf.py ^GSPC --horizon 10            # S&P 500 index
        """
    )

    parser.add_argument("symbol", nargs="?", help="Stock symbol (e.g., AAPL, BBRI.JK, ^GSPC)")
    parser.add_argument("--horizon", "-n", type=int, default=5, help="Forecast horizon (default: 5)")
    parser.add_argument("--interval", "-i", default="1d", help="Data interval (default: 1d)")
    parser.add_argument("--period", "-P", default="3mo", help="History period (default: 3mo)")
    parser.add_argument("--plot", "-p", action="store_true", help="Show chart")

    args = parser.parse_args()

    if not args.symbol:
        parser.print_help()
        sys.exit(1)

    symbol = args.symbol.upper()

    print(f"\n{'=' * 70}")
    print(f"YAHOO FINANCE FORECASTER")
    print(f"{'=' * 70}")

    try:
        # Load data
        loader = YFinanceLoader(symbol)
        name = loader.get_name()
        currency = loader.get_currency()

        print(f"Symbol: {symbol}")
        print(f"Name: {name}")
        print(f"Currency: {currency}")

        df = loader.load_data(period=args.period, interval=args.interval)
        df = loader.add_features(df)

        print(f"Loaded {len(df)} records")

        # Forecast
        forecaster = YFinanceForecaster(horizon=args.horizon, interval=args.interval)
        forecasts, historical, meta = forecaster.forecast(df)

        # Results
        print_results(forecasts, df["close"].iloc[-1], symbol, currency)

        # Plot
        if args.plot:
            plot_forecast(df, forecasts, symbol, name)

        # Save
        out = f"yf_{symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        forecasts.to_csv(out, index=False)
        print(f"\nSaved: {out}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
