#!/usr/bin/env python3
"""
Intraday Stock Forecasting - Single Session
Forecasts next trading session based on tick data from running-trade.json.

Usage:
    python short.py SYMBOL --hours 3          # Forecast next 3 hours
    python short.py SYMBOL --session1         # Forecast session 1 (09:00-12:00)
    python short.py SYMBOL --session2         # Forecast session 2 (13:30-16:00)

Example:
    python short.py ICBP --session1 --plot    # Forecast tomorrow's session 1
    python short.py ICBP --hours 4 --interval 5
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
from neuralforecast.models import NBEATS, NHITS, LSTM

warnings.filterwarnings("ignore")


# Indonesian Stock Market Sessions
MARKET_SESSIONS = {
    "session1": {"start": "09:00", "end": "12:00", "hours": 3.0},    # Morning session
    "session2": {"start": "13:30", "end": "16:00", "hours": 2.5},    # Afternoon session
    "pre_close": {"start": "16:00", "end": "16:15", "hours": 0.25},  # Pre-closing
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

                records.append({
                    "timestamp": timestamp,
                    "price": price,
                    "lot": lot,
                    "value": price * lot * 100,
                    "action": trade.get("action", ""),
                    "buyer_type": trade.get("buyer_type", ""),
                    "seller_type": trade.get("seller_type", ""),
                })
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

    def load_session(self, symbol: str, session: str, interval: int = 5) -> pd.DataFrame:
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


class SessionForecaster:
    """Forecast next trading session."""

    def __init__(self, interval: int = 5):
        self.interval = interval
        self.nf = None

    def calculate_horizon(self, hours: float) -> int:
        """Calculate number of bars for given hours."""
        return int(hours * 60 / self.interval)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare NeuralForecast format."""
        exog_cols = ["volume", "volatility", "buy_pressure", "foreign_net", "momentum", "vol_ratio"]
        available = [c for c in exog_cols if c in df.columns]

        nf_df = pd.DataFrame({
            "unique_id": "STOCK",
            "ds": df["timestamp"],
            "y": df["close"]
        })

        for col in available:
            nf_df[col] = df[col].fillna(0)

        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)
        return nf_df, available

    def forecast(self, df: pd.DataFrame, hours: float, target_session: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Train and forecast for specified hours."""
        nf_df, exog = self.prepare_data(df)
        n = len(nf_df)

        horizon = self.calculate_horizon(hours)

        print(f"\n{'=' * 60}")
        print("INPUT DATA")
        print(f"{'=' * 60}")
        print(f"  Historical bars: {n}")
        print(f"  Time range: {nf_df['ds'].min().strftime('%H:%M')} - {nf_df['ds'].max().strftime('%H:%M')}")
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
            print(f"  Target session: {target_session.upper()} ({sess.get('start', '?')} - {sess.get('end', '?')})")

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

        models = [
            NBEATS(h=horizon, input_size=input_size, max_steps=200, scaler_type="robust", random_seed=42),
            NHITS(h=horizon, input_size=input_size, max_steps=200, scaler_type="robust", random_seed=42),
            LSTM(h=horizon, input_size=input_size, max_steps=200, scaler_type="robust", random_seed=42),
        ]

        freq = f"{self.interval}min"
        print(f"\nTraining: {[m.__class__.__name__ for m in models]}")

        self.nf = NeuralForecast(models=models, freq=freq)
        self.nf.fit(df=nf_df)

        print(f"Forecasting {horizon} bars ({hours} hours)...")
        forecasts = self.nf.predict()

        # Generate future timestamps starting from next trading day 09:00
        last_date = nf_df["ds"].iloc[-1]
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
        print(f"Target: {meta['target_session'].upper()} ({sess.get('start')}-{sess.get('end')})")
    print(f"Period: {meta['forecast_start'].strftime('%Y-%m-%d %H:%M')} to {meta['forecast_end'].strftime('%H:%M')}")
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
        t = row["ds"].strftime("%H:%M") if hasattr(row["ds"], "strftime") else str(row["ds"])
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
        print(f"  Open forecast:   {forecasts['ensemble'].iloc[0]:>14,.2f} ({(forecasts['ensemble'].iloc[0]/last_price-1)*100:+.2f}%)")
        print(f"  High forecast:   {forecasts['ensemble'].max():>14,.2f} ({(forecasts['ensemble'].max()/last_price-1)*100:+.2f}%)")
        print(f"  Low forecast:    {forecasts['ensemble'].min():>14,.2f} ({(forecasts['ensemble'].min()/last_price-1)*100:+.2f}%)")
        print(f"  Close forecast:  {forecasts['ensemble'].iloc[-1]:>14,.2f} ({(forecasts['ensemble'].iloc[-1]/last_price-1)*100:+.2f}%)")

        direction = "BULLISH" if forecasts['ensemble'].iloc[-1] > last_price else "BEARISH" if forecasts['ensemble'].iloc[-1] < last_price else "NEUTRAL"
        print(f"\n  Outlook: {direction}")


def plot_forecast(ohlcv: pd.DataFrame, forecasts: pd.DataFrame, symbol: str, meta: dict):
    """Plot historical and forecast."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # Price chart
        ax1 = axes[0]

        # Historical
        ax1.plot(ohlcv["timestamp"], ohlcv["close"], "b-", label="Historical", linewidth=1.5)

        # Forecast
        model_cols = [c for c in forecasts.columns if c not in ["ds", "unique_id", "ensemble", "std", "low", "high"]]
        for col in model_cols:
            ax1.plot(forecasts["ds"], forecasts[col], "--", alpha=0.4, linewidth=1, label=col)

        if "ensemble" in forecasts.columns:
            ax1.plot(forecasts["ds"], forecasts["ensemble"], "r-", linewidth=2, label="Ensemble Forecast")
            if "low" in forecasts.columns:
                ax1.fill_between(forecasts["ds"], forecasts["low"], forecasts["high"],
                                 alpha=0.2, color="red", label="95% CI")

        # Add vertical line at forecast start
        if len(forecasts) > 0:
            ax1.axvline(x=forecasts["ds"].iloc[0], color="green", linestyle="--", alpha=0.7, label="Forecast Start")

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
        colors = ["green" if ohlcv["close"].iloc[i] >= ohlcv["open"].iloc[i] else "red"
                  for i in range(len(ohlcv))]
        ax2.bar(ohlcv["timestamp"], ohlcv["volume"], color=colors, alpha=0.7, width=0.001)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date/Time")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        out = f"short_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nChart saved: {out}")
        plt.show()

    except ImportError:
        print("\nMatplotlib not available")


def main():
    parser = argparse.ArgumentParser(
        description="Forecast Next Trading Session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IDX Market Sessions:
  --session1    Morning session (09:00-12:00) = 3 hours
  --session2    Afternoon session (13:30-16:00) = 2.5 hours
  --hours N     Custom duration in hours

Examples:
    python short.py ICBP --session1           # Forecast tomorrow's morning session
    python short.py ICBP --session2 --plot    # Forecast afternoon session with chart
    python short.py ICBP --hours 4            # Forecast next 4 hours
    python short.py ICBP --hours 1 --interval 1   # 1 hour with 1-min bars
        """
    )

    parser.add_argument("symbol", nargs="?", help="Stock symbol")
    parser.add_argument("--session", "-S", default="1", help="Data session folder (default: 1)")
    parser.add_argument("--session1", action="store_true", help="Forecast session 1 (09:00-12:00)")
    parser.add_argument("--session2", action="store_true", help="Forecast session 2 (13:30-16:00)")
    parser.add_argument("--hours", "-H", type=float, help="Forecast duration in hours")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Bar interval in minutes (default: 5)")
    parser.add_argument("--plot", "-p", action="store_true", help="Show plot")
    parser.add_argument("--list", "-l", action="store_true", help="List symbols")
    parser.add_argument("--source", "-s", default="sources", help="Data directory")

    args = parser.parse_args()

    loader = TickToOHLCV(args.source)

    if args.list:
        print("Available symbols:")
        for sym in sorted(loader.get_symbols()):
            sessions = loader.get_sessions(sym)
            print(f"  {sym}: sessions {', '.join(sessions)}")
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

    try:
        print(f"\nLoading tick data from session {args.session}...")
        ohlcv = loader.load_session(symbol, args.session, args.interval)

        if ohlcv.empty:
            print("Error: No tick data found")
            sys.exit(1)

        print(f"Created {len(ohlcv)} bars ({args.interval}-min interval)")
        print(f"Total trades: {ohlcv['trades'].sum():,.0f}")
        print(f"Total volume: {ohlcv['volume'].sum():,.0f} lots")

        forecaster = SessionForecaster(interval=args.interval)
        forecasts, historical, meta = forecaster.forecast(ohlcv, hours, target_session)

        print_results(forecasts, ohlcv["close"].iloc[-1], symbol, meta)

        if args.plot:
            plot_forecast(ohlcv, forecasts, symbol, meta)

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
