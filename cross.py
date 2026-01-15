#!/usr/bin/env python3
"""
Cross-Validation for Stock Forecasting Models
Validates NeuralForecast models using time-series cross-validation.

Reference: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html

Usage:
    python cross.py SYMBOL --source tick              # CV on tick data (like short.py)
    python cross.py SYMBOL --source session           # CV on session data (like forecast.py)
    python cross.py AAPL --source yfinance            # CV on Yahoo Finance data
    python cross.py SYMBOL --n-windows 5              # 5-fold CV

Examples:
    python cross.py ICBP --source tick --n-windows 3
    python cross.py ICBP --source session --horizon 3
    python cross.py AAPL --source yfinance --period 6mo --n-windows 5
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
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATS, NHITS

warnings.filterwarnings("ignore")


# =============================================================================
# DATA LOADERS
# =============================================================================

class TickDataLoader:
    """Load tick data from running-trade.json (same as short.py)."""

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
        for f in session_path.glob("analysis-data-*.json"):
            try:
                date_str = f.stem.replace("analysis-data-", "").split("_")[0]
                return datetime.strptime(date_str, "%Y-%m-%d")
            except:
                pass
        return datetime.now()

    def load_session(self, symbol: str, session: str, interval: int = 5) -> pd.DataFrame:
        """Load and aggregate tick data to OHLCV bars."""
        session_path = self.base_path / symbol / session
        base_date = self.get_session_date(session_path)

        # Try running-trade.json
        rt = self.load_json(session_path / "running-trade.json")
        if rt and "data" in rt:
            trades = rt["data"].get("running_trade", [])
            if trades:
                return self._process_trades(trades, base_date, interval)

        # Try today-running-trade.json
        trt = self.load_json(session_path / "today-running-trade.json")
        if trt and "data" in trt:
            trades = trt["data"].get("running_trade", [])
            if trades:
                return self._process_trades(trades, base_date, interval)

        return pd.DataFrame()

    def _process_trades(self, trades: List[dict], base_date: datetime, interval: int) -> pd.DataFrame:
        records = []
        for trade in trades:
            try:
                price = self.parse_number(trade.get("price", 0))
                lot = self.parse_number(trade.get("lot", 0))
                time_str = trade.get("time", "09:00:00")
                if price <= 0 or lot <= 0:
                    continue
                parts = time_str.split(":")
                hour, minute = int(parts[0]), int(parts[1])
                second = int(parts[2]) if len(parts) > 2 else 0
                timestamp = base_date.replace(hour=hour, minute=minute, second=second)
                records.append({"timestamp": timestamp, "price": price, "lot": lot})
            except:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("timestamp")
        df = df.set_index("timestamp")

        ohlcv = df["price"].resample(f"{interval}min").ohlc()
        ohlcv.columns = ["open", "high", "low", "close"]
        ohlcv["volume"] = df["lot"].resample(f"{interval}min").sum()
        ohlcv = ohlcv.dropna(subset=["close"]).reset_index()
        ohlcv = ohlcv.rename(columns={"timestamp": "ds"})

        return ohlcv

    def get_symbols(self) -> List[str]:
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def get_sessions(self, symbol: str) -> List[str]:
        path = self.base_path / symbol
        if not path.exists():
            return []
        return sorted([d.name for d in path.iterdir() if d.is_dir()],
                      key=lambda x: int(x) if x.isdigit() else 0)


class SessionDataLoader:
    """Load session-aggregated data (like forecast.py)."""

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

    def load_all_sessions(self, symbol: str) -> pd.DataFrame:
        """Load all sessions for a symbol as daily data points."""
        symbol_path = self.base_path / symbol
        if not symbol_path.exists():
            return pd.DataFrame()

        sessions = sorted([d.name for d in symbol_path.iterdir() if d.is_dir()],
                          key=lambda x: int(x) if x.isdigit() else 0)

        records = []
        for session in sessions:
            session_path = symbol_path / session
            pf = self.load_json(session_path / "price-feed.json")

            if pf and "data" in pf:
                data = pf["data"]
                close = self.parse_number(data.get("close", data.get("last", 0)))
                if close > 0:
                    # Try to get date from analysis files
                    ds = None
                    for f in session_path.glob("analysis-data-*.json"):
                        try:
                            date_str = f.stem.replace("analysis-data-", "").split("_")[0]
                            ds = datetime.strptime(date_str, "%Y-%m-%d")
                            break
                        except:
                            pass

                    if ds is None:
                        ds = datetime.now() - timedelta(days=int(session) if session.isdigit() else 0)

                    records.append({
                        "ds": ds,
                        "close": close,
                        "open": self.parse_number(data.get("open", close)),
                        "high": self.parse_number(data.get("high", close)),
                        "low": self.parse_number(data.get("low", close)),
                        "volume": self.parse_number(data.get("volume", 0)),
                    })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("ds").reset_index(drop=True)
        return df

    def get_symbols(self) -> List[str]:
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]


class YFinanceLoader:
    """Load data from Yahoo Finance."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()

    def load_data(self, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {self.symbol}")

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "ds"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "ds"})

        df["ds"] = pd.to_datetime(df["ds"])
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        return df


# =============================================================================
# CROSS-VALIDATION ENGINE
# =============================================================================

class CrossValidator:
    """Cross-validation for NeuralForecast models."""

    def __init__(self, horizon: int = 5, n_windows: int = 3, step_size: Optional[int] = None):
        self.horizon = horizon
        self.n_windows = n_windows
        self.step_size = step_size or horizon  # Default: non-overlapping windows
        self.nf = None
        self.cv_results = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data in NeuralForecast format."""
        nf_df = pd.DataFrame({
            "unique_id": "STOCK",
            "ds": df["ds"],
            "y": df["close"]
        })
        nf_df = nf_df.replace([np.inf, -np.inf], 0).fillna(0)
        return nf_df

    def run_cv(self, df: pd.DataFrame, refit: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Run cross-validation and return results with metrics."""
        nf_df = self.prepare_data(df)
        n = len(nf_df)

        print(f"\n{'=' * 60}")
        print("CROSS-VALIDATION SETUP")
        print(f"{'=' * 60}")
        print(f"  Total records: {n}")
        print(f"  Horizon: {self.horizon}")
        print(f"  N windows: {self.n_windows}")
        print(f"  Step size: {self.step_size}")
        print(f"  Refit: {refit}")

        # Validate parameters
        min_required = self.horizon * (self.n_windows + 1) + 10
        if n < min_required:
            raise ValueError(f"Need at least {min_required} records for {self.n_windows} CV windows "
                             f"with horizon {self.horizon} (have {n})")

        # Adjust parameters if needed
        input_size = min(n // (self.n_windows + 2), 20)
        input_size = max(input_size, 3)

        print(f"  Input size: {input_size}")

        # Define models
        models = [
            NBEATS(h=self.horizon, input_size=input_size, max_steps=200,
                   scaler_type="robust", random_seed=42),
            NHITS(h=self.horizon, input_size=input_size, max_steps=200,
                  scaler_type="robust", random_seed=42),
            LSTM(h=self.horizon, input_size=input_size, max_steps=200,
                 scaler_type="robust", random_seed=42),
        ]

        # Infer frequency
        if len(nf_df) > 1:
            time_diff = (nf_df["ds"].iloc[1] - nf_df["ds"].iloc[0]).total_seconds()
            if time_diff < 3600:  # Less than 1 hour
                freq = f"{int(time_diff / 60)}min"
            elif time_diff < 86400:  # Less than 1 day
                freq = f"{int(time_diff / 3600)}h"
            else:
                freq = "D"
        else:
            freq = "D"

        print(f"  Frequency: {freq}")
        print(f"\nModels: {[m.__class__.__name__ for m in models]}")

        # Create NeuralForecast and run cross-validation
        self.nf = NeuralForecast(models=models, freq=freq)

        print(f"\nRunning {self.n_windows}-fold cross-validation...")

        cv_df = self.nf.cross_validation(
            df=nf_df,
            n_windows=self.n_windows,
            step_size=self.step_size,
            refit=refit
        )

        self.cv_results = cv_df

        # Calculate metrics
        metrics = self._calculate_metrics(cv_df)

        return cv_df, metrics

    def _calculate_metrics(self, cv_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics for each model."""
        model_cols = [c for c in cv_df.columns if c not in ["ds", "unique_id", "y", "cutoff"]]
        metrics = {}

        y_true = cv_df["y"].values

        for model in model_cols:
            y_pred = cv_df[model].values
            mask = ~np.isnan(y_pred) & ~np.isnan(y_true)

            if mask.sum() == 0:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            # MAE
            mae = np.mean(np.abs(y_t - y_p))

            # RMSE
            rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

            # MAPE (avoid division by zero)
            mape_mask = y_t != 0
            if mape_mask.sum() > 0:
                mape = np.mean(np.abs((y_t[mape_mask] - y_p[mape_mask]) / y_t[mape_mask])) * 100
            else:
                mape = np.nan

            # SMAPE
            smape = np.mean(2 * np.abs(y_t - y_p) / (np.abs(y_t) + np.abs(y_p) + 1e-8)) * 100

            # Direction accuracy
            if len(y_t) > 1:
                actual_direction = np.sign(np.diff(y_t))
                pred_direction = np.sign(np.diff(y_p))
                direction_acc = np.mean(actual_direction == pred_direction) * 100
            else:
                direction_acc = np.nan

            metrics[model] = {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "SMAPE": smape,
                "Direction_Acc": direction_acc,
                "n_predictions": mask.sum()
            }

        # Add ensemble metrics
        if len(model_cols) > 1:
            ensemble_pred = cv_df[model_cols].mean(axis=1).values
            mask = ~np.isnan(ensemble_pred) & ~np.isnan(y_true)
            y_t = y_true[mask]
            y_p = ensemble_pred[mask]

            metrics["Ensemble"] = {
                "MAE": np.mean(np.abs(y_t - y_p)),
                "RMSE": np.sqrt(np.mean((y_t - y_p) ** 2)),
                "MAPE": np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100,
                "SMAPE": np.mean(2 * np.abs(y_t - y_p) / (np.abs(y_t) + np.abs(y_p) + 1e-8)) * 100,
                "Direction_Acc": np.mean(np.sign(np.diff(y_t)) == np.sign(np.diff(y_p))) * 100 if len(y_t) > 1 else np.nan,
                "n_predictions": mask.sum()
            }

        return metrics


def print_metrics(metrics: Dict, symbol: str):
    """Print metrics in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"CROSS-VALIDATION RESULTS: {symbol}")
    print(f"{'=' * 70}")

    # Header
    print(f"\n{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE%':>10} {'SMAPE%':>10} {'Dir.Acc%':>10}")
    print("-" * 64)

    # Sort by MAE
    sorted_models = sorted(metrics.items(), key=lambda x: x[1].get("MAE", float("inf")))

    for model, m in sorted_models:
        mae = f"{m['MAE']:,.2f}" if not np.isnan(m.get('MAE', np.nan)) else "N/A"
        rmse = f"{m['RMSE']:,.2f}" if not np.isnan(m.get('RMSE', np.nan)) else "N/A"
        mape = f"{m['MAPE']:.2f}" if not np.isnan(m.get('MAPE', np.nan)) else "N/A"
        smape = f"{m['SMAPE']:.2f}" if not np.isnan(m.get('SMAPE', np.nan)) else "N/A"
        dir_acc = f"{m['Direction_Acc']:.1f}" if not np.isnan(m.get('Direction_Acc', np.nan)) else "N/A"

        print(f"{model:<12} {mae:>10} {rmse:>10} {mape:>10} {smape:>10} {dir_acc:>10}")

    # Best model
    best_model = sorted_models[0][0]
    print(f"\nBest model (by MAE): {best_model}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("METRIC INTERPRETATION")
    print(f"{'=' * 70}")
    print("  MAE:      Mean Absolute Error (lower is better)")
    print("  RMSE:     Root Mean Square Error (lower is better)")
    print("  MAPE:     Mean Absolute Percentage Error (lower is better)")
    print("  SMAPE:    Symmetric MAPE (lower is better, max 200%)")
    print("  Dir.Acc:  Direction Accuracy (higher is better, 50% = random)")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Validation for Stock Forecasting Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  --source tick       Load tick data from sources/{SYMBOL}/{SESSION}/running-trade.json
  --source session    Load session data from sources/{SYMBOL}/{SESSION}/price-feed.json
  --source yfinance   Load from Yahoo Finance (requires yfinance)

Cross-Validation Parameters:
  --n-windows N       Number of CV folds (default: 3)
  --step-size S       Steps between windows (default: horizon)
  --horizon H         Forecast horizon (default: 5)
  --refit             Retrain model at each fold

Examples:
    python cross.py ICBP --source tick --n-windows 3
    python cross.py ICBP --source session --horizon 3
    python cross.py AAPL --source yfinance --period 1y --n-windows 5
    python cross.py BBRI.JK --source yfinance --yf-interval 1h --period 5d
        """
    )

    parser.add_argument("symbol", nargs="?", help="Stock symbol")
    parser.add_argument("--source", "-s", default="tick",
                        choices=["tick", "session", "yfinance"],
                        help="Data source (default: tick)")
    parser.add_argument("--session", "-S", default="1",
                        help="Session folder for tick/session source (default: 1)")
    parser.add_argument("--interval", "-i", type=int, default=5,
                        help="Bar interval in minutes for tick data (default: 5)")
    parser.add_argument("--n-windows", "-w", type=int, default=3,
                        help="Number of CV windows (default: 3)")
    parser.add_argument("--step-size", type=int, default=None,
                        help="Step size between windows (default: horizon)")
    parser.add_argument("--horizon", "-n", type=int, default=5,
                        help="Forecast horizon (default: 5)")
    parser.add_argument("--refit", action="store_true",
                        help="Retrain model at each CV fold")
    parser.add_argument("--period", "-P", default="6mo",
                        help="Period for yfinance (default: 6mo)")
    parser.add_argument("--yf-interval", default="1d",
                        help="Interval for yfinance (default: 1d)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available symbols")
    parser.add_argument("--data-path", default="sources",
                        help="Path to data sources (default: sources)")

    args = parser.parse_args()

    # List symbols
    if args.list:
        if args.source in ["tick", "session"]:
            loader = TickDataLoader(args.data_path)
            print("Available symbols:")
            for sym in sorted(loader.get_symbols()):
                sessions = loader.get_sessions(sym)
                print(f"  {sym}: {len(sessions)} session(s)")
        else:
            print("Use --source tick or --source session to list local symbols")
            print("For yfinance, use any valid ticker (AAPL, BBRI.JK, etc.)")
        return

    if not args.symbol:
        parser.print_help()
        sys.exit(1)

    symbol = args.symbol.upper()

    print(f"\n{'=' * 70}")
    print(f"CROSS-VALIDATION: {symbol}")
    print(f"{'=' * 70}")
    print(f"Source: {args.source}")

    try:
        # Load data based on source
        if args.source == "tick":
            loader = TickDataLoader(args.data_path)
            if symbol not in loader.get_symbols():
                print(f"Error: {symbol} not found in {args.data_path}")
                print(f"Available: {', '.join(sorted(loader.get_symbols()))}")
                sys.exit(1)

            sessions = loader.get_sessions(symbol)
            if args.session not in sessions:
                print(f"Error: Session {args.session} not found")
                print(f"Available: {', '.join(sessions)}")
                sys.exit(1)

            print(f"Loading tick data from session {args.session}...")
            df = loader.load_session(symbol, args.session, args.interval)

            if df.empty:
                print("Error: No tick data found")
                sys.exit(1)

            print(f"Loaded {len(df)} bars ({args.interval}-min interval)")

        elif args.source == "session":
            loader = SessionDataLoader(args.data_path)
            if symbol not in loader.get_symbols():
                print(f"Error: {symbol} not found in {args.data_path}")
                print(f"Available: {', '.join(sorted(loader.get_symbols()))}")
                sys.exit(1)

            print(f"Loading session data...")
            df = loader.load_all_sessions(symbol)

            if df.empty:
                print("Error: No session data found")
                sys.exit(1)

            print(f"Loaded {len(df)} sessions")

        elif args.source == "yfinance":
            print(f"Loading from Yahoo Finance: period={args.period}, interval={args.yf_interval}")
            loader = YFinanceLoader(symbol)
            df = loader.load_data(period=args.period, interval=args.yf_interval)
            print(f"Loaded {len(df)} records")

        else:
            print(f"Unknown source: {args.source}")
            sys.exit(1)

        # Run cross-validation
        cv = CrossValidator(
            horizon=args.horizon,
            n_windows=args.n_windows,
            step_size=args.step_size
        )

        cv_df, metrics = cv.run_cv(df, refit=args.refit)

        # Print results
        print_metrics(metrics, symbol)

        # Save results
        out_file = f"cross_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        cv_df.to_csv(out_file, index=False)
        print(f"\nResults saved: {out_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
