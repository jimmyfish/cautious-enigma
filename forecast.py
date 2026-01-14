#!/usr/bin/env python3
"""
Stock Price Forecasting using NeuralForecast
Extracts intraday OHLCV bars from tick data for richer time series.

Usage: python forecast.py SYMBOL [--horizon N] [--interval M] [--plot]

Example:
    python forecast.py ARCI --horizon 10 --plot
    python forecast.py ARCI --interval 5 --horizon 20
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM, TimesNet, PatchTST

import warnings
warnings.filterwarnings('ignore')


class TickDataProcessor:
    """Process tick-level trade data into OHLCV bars."""

    @staticmethod
    def parse_price(price_str) -> float:
        """Parse price string to float."""
        if isinstance(price_str, (int, float)):
            return float(price_str)
        return float(str(price_str).replace(',', ''))

    @staticmethod
    def parse_lot(lot_str) -> int:
        """Parse lot string to int."""
        if isinstance(lot_str, (int, float)):
            return int(lot_str)
        return int(str(lot_str).replace(',', ''))

    @staticmethod
    def parse_time(time_str: str, base_date: datetime) -> datetime:
        """Parse time string and combine with base date."""
        try:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            return base_date.replace(hour=hour, minute=minute, second=second)
        except:
            return base_date

    def process_running_trades(self, trades: List[dict], base_date: datetime) -> pd.DataFrame:
        """Convert running trades to a DataFrame with parsed values."""
        records = []

        for trade in trades:
            try:
                price = self.parse_price(trade.get('price', 0))
                lot = self.parse_lot(trade.get('lot', 0))
                time_str = trade.get('time', '09:00:00')
                timestamp = self.parse_time(time_str, base_date)

                if price > 0 and lot > 0:
                    records.append({
                        'timestamp': timestamp,
                        'price': price,
                        'lot': lot,
                        'value': price * lot * 100,  # lot = 100 shares
                        'action': trade.get('action', 'unknown'),
                        'buyer_type': trade.get('buyer_type', ''),
                        'seller_type': trade.get('seller_type', ''),
                    })
            except Exception as e:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def aggregate_to_ohlcv(self, tick_df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
        """Aggregate tick data to OHLCV bars."""
        if tick_df.empty:
            return pd.DataFrame()

        tick_df = tick_df.set_index('timestamp')

        # Resample to intervals
        ohlcv = tick_df['price'].resample(f'{interval_minutes}min').ohlc()
        ohlcv.columns = ['open', 'high', 'low', 'close']

        # Volume and value
        ohlcv['volume'] = tick_df['lot'].resample(f'{interval_minutes}min').sum()
        ohlcv['value'] = tick_df['value'].resample(f'{interval_minutes}min').sum()
        ohlcv['trade_count'] = tick_df['price'].resample(f'{interval_minutes}min').count()

        # Buy/Sell pressure
        buy_mask = tick_df['action'] == 'buy'
        ohlcv['buy_volume'] = tick_df.loc[buy_mask, 'lot'].resample(f'{interval_minutes}min').sum()
        ohlcv['sell_volume'] = tick_df.loc[~buy_mask, 'lot'].resample(f'{interval_minutes}min').sum()

        # Foreign flow
        foreign_buy = tick_df['buyer_type'].str.contains('FOREIGN', na=False)
        foreign_sell = tick_df['seller_type'].str.contains('FOREIGN', na=False)
        ohlcv['foreign_buy_vol'] = tick_df.loc[foreign_buy, 'lot'].resample(f'{interval_minutes}min').sum()
        ohlcv['foreign_sell_vol'] = tick_df.loc[foreign_sell, 'lot'].resample(f'{interval_minutes}min').sum()

        # Drop rows with no trades
        ohlcv = ohlcv.dropna(subset=['open', 'close'])
        ohlcv = ohlcv.fillna(0)

        # Calculate derived features
        ohlcv['returns'] = ohlcv['close'].pct_change()
        ohlcv['volatility'] = ohlcv['returns'].rolling(window=3, min_periods=1).std()
        ohlcv['buy_sell_ratio'] = ohlcv['buy_volume'] / (ohlcv['sell_volume'] + 1)
        ohlcv['foreign_net'] = ohlcv['foreign_buy_vol'] - ohlcv['foreign_sell_vol']
        ohlcv['vwap'] = ohlcv['value'] / (ohlcv['volume'] * 100 + 1)

        # Price momentum
        ohlcv['momentum_3'] = ohlcv['close'].pct_change(periods=3)
        ohlcv['momentum_5'] = ohlcv['close'].pct_change(periods=5)

        # Volume momentum
        ohlcv['vol_ma_3'] = ohlcv['volume'].rolling(window=3, min_periods=1).mean()
        ohlcv['vol_ratio'] = ohlcv['volume'] / (ohlcv['vol_ma_3'] + 1)

        return ohlcv.reset_index()


class StockDataLoader:
    """Load and aggregate stock data from JSON files."""

    def __init__(self, base_path: str = "sources"):
        self.base_path = Path(base_path)
        self.tick_processor = TickDataProcessor()

    def get_available_symbols(self) -> list:
        """Get list of available stock symbols."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def get_sessions(self, symbol: str) -> list:
        """Get available sessions for a symbol."""
        symbol_path = self.base_path / symbol
        if not symbol_path.exists():
            return []
        sessions = [d.name for d in symbol_path.iterdir() if d.is_dir()]
        return sorted(sessions, key=lambda x: int(x) if x.isdigit() else 0)

    def load_json(self, filepath: Path) -> Optional[dict]:
        """Load a JSON file safely."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def extract_date_from_session(self, session_path: Path) -> Optional[datetime]:
        """Try to extract date from session files."""
        # Try analysis files first
        for f in session_path.glob('analysis-data-*.json'):
            try:
                date_str = f.stem.replace('analysis-data-', '').split('_')[0]
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                pass

        # Try market-detector
        md = self.load_json(session_path / 'market-detector.json')
        if md and 'data' in md:
            date_str = md['data'].get('from')
            if date_str:
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    pass

        # Try findata
        fd = self.load_json(session_path / 'findata.json')
        if fd and 'data' in fd:
            date_str = fd['data'].get('from')
            if date_str:
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    pass

        return None

    def load_session_ohlcv(self, symbol: str, session: str, interval: int = 5) -> pd.DataFrame:
        """Load a session and convert to OHLCV bars."""
        session_path = self.base_path / symbol / session

        # Get base date
        base_date = self.extract_date_from_session(session_path)
        if not base_date:
            base_date = datetime.now() - timedelta(days=int(session))

        # Try running-trade first (more data)
        running_trade = self.load_json(session_path / 'running-trade.json')
        if running_trade and 'data' in running_trade:
            trades = running_trade['data'].get('running_trade', [])
            if trades:
                tick_df = self.tick_processor.process_running_trades(trades, base_date)
                if not tick_df.empty:
                    ohlcv = self.tick_processor.aggregate_to_ohlcv(tick_df, interval)
                    if not ohlcv.empty:
                        ohlcv['symbol'] = symbol
                        ohlcv['session'] = session
                        return ohlcv

        # Fallback to today-running-trade
        today_trade = self.load_json(session_path / 'today-running-trade.json')
        if today_trade and 'data' in today_trade:
            trades = today_trade['data'].get('running_trade', [])
            if trades:
                tick_df = self.tick_processor.process_running_trades(trades, base_date)
                if not tick_df.empty:
                    ohlcv = self.tick_processor.aggregate_to_ohlcv(tick_df, interval)
                    if not ohlcv.empty:
                        ohlcv['symbol'] = symbol
                        ohlcv['session'] = session
                        return ohlcv

        return pd.DataFrame()

    def load_session_snapshot(self, symbol: str, session: str) -> dict:
        """Load end-of-session snapshot data for additional features."""
        session_path = self.base_path / symbol / session
        snapshot = {}

        # Price feed data
        pf = self.load_json(session_path / 'price-feed.json')
        if pf and 'data' in pf:
            d = pf['data']
            snapshot['eod_close'] = float(d.get('close', 0) or d.get('lastprice', 0))
            snapshot['eod_volume'] = float(d.get('volume', 0))
            snapshot['eod_value'] = float(d.get('value', 0))
            snapshot['eod_frequency'] = float(d.get('frequency', 0))
            snapshot['eod_foreign_net'] = float(d.get('fnet', 0))

            # Order book imbalance
            bid_vol = sum(float(b.get('volume', 0)) for b in d.get('bid', []))
            offer_vol = sum(float(o.get('volume', 0)) for o in d.get('offer', []))
            snapshot['book_imbalance'] = (bid_vol - offer_vol) / (bid_vol + offer_vol + 1e-10)

        # Market detector data
        md = self.load_json(session_path / 'market-detector.json')
        if md and 'data' in md:
            bd = md['data'].get('bandar_detector', {})
            if bd:
                snapshot['buyer_count'] = int(bd.get('total_buyer', 0))
                snapshot['seller_count'] = int(bd.get('total_seller', 0))
                avg_data = bd.get('avg', {})
                accdist_map = {'Big Acc': 2, 'Normal Acc': 1, 'Neutral': 0, 'Normal Dist': -1, 'Big Dist': -2}
                snapshot['accdist_signal'] = accdist_map.get(avg_data.get('accdist', 'Neutral'), 0)

        return snapshot

    def load_symbol_data(self, symbol: str, interval: int = 5) -> pd.DataFrame:
        """Load all session data for a symbol into a single time series."""
        sessions = self.get_sessions(symbol)

        if not sessions:
            raise ValueError(f"No data found for symbol: {symbol}")

        all_ohlcv = []
        session_info = []

        for session in sessions:
            ohlcv = self.load_session_ohlcv(symbol, session, interval)
            if not ohlcv.empty:
                snapshot = self.load_session_snapshot(symbol, session)
                # Add snapshot features to all bars in this session
                for key, value in snapshot.items():
                    ohlcv[key] = value
                all_ohlcv.append(ohlcv)
                session_info.append((session, len(ohlcv)))

        if not all_ohlcv:
            raise ValueError(f"No valid tick data found for symbol: {symbol}")

        # Combine all sessions
        df = pd.concat(all_ohlcv, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Print session info
        print(f"\nData loaded from {len(session_info)} session(s):")
        for sess, count in session_info:
            print(f"  - Session {sess}: {count} bars")

        return df


class StockForecaster:
    """NeuralForecast-based stock price forecaster."""

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.models = None
        self.nf = None

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'close') -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data in NeuralForecast format."""
        # Select exogenous features
        feature_cols = [
            'volume', 'trade_count', 'buy_sell_ratio', 'foreign_net',
            'volatility', 'momentum_3', 'vol_ratio', 'book_imbalance',
            'accdist_signal'
        ]

        # Filter to available columns
        available_features = [c for c in feature_cols if c in df.columns]

        # Create base dataframe
        nf_df = pd.DataFrame({
            'unique_id': df['symbol'],
            'ds': df['timestamp'],
            'y': df[target_col]
        })

        # Add exogenous variables
        for col in available_features:
            nf_df[col] = df[col].fillna(0)

        # Handle missing/infinite values
        nf_df = nf_df.replace([np.inf, -np.inf], 0)
        nf_df = nf_df.fillna(0)

        return nf_df, available_features

    def create_models(self, input_size: int):
        """Create NeuralForecast models."""
        # Adjust input size based on data
        effective_input = min(input_size, 24)  # Max 24 periods lookback

        models = [
            # N-BEATS: Interpretable decomposition
            NBEATS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type='robust',
                random_seed=42
            ),

            # N-HiTS: Multi-scale hierarchical
            NHITS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type='robust',
                random_seed=42
            ),

            # LSTM: Sequential patterns
            LSTM(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type='robust',
                random_seed=42
            ),
        ]

        # Add advanced models if we have enough data
        if input_size >= 20:
            models.append(
                PatchTST(
                    h=self.horizon,
                    input_size=effective_input,
                    max_steps=200,
                    scaler_type='robust',
                    random_seed=42
                )
            )

        return models

    def train_and_forecast(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train models and generate forecasts."""
        nf_df, exog_vars = self.prepare_data(df)

        print(f"\nPrepared time series:")
        print(f"  - Total bars: {len(nf_df)}")
        print(f"  - Time range: {nf_df['ds'].min()} to {nf_df['ds'].max()}")
        print(f"  - Price range: {nf_df['y'].min():.2f} to {nf_df['y'].max():.2f}")
        print(f"  - Exogenous features: {len(exog_vars)} ({', '.join(exog_vars[:5])}...)")

        min_required = self.horizon + 3
        if len(nf_df) < min_required:
            raise ValueError(f"Insufficient data ({len(nf_df)} bars). Need at least {min_required} for horizon={self.horizon}.")

        # Determine input size (lookback)
        input_size = min(len(nf_df) - self.horizon - 1, 24)
        print(f"  - Lookback window: {input_size} bars")

        # Create models
        self.models = self.create_models(input_size)
        print(f"\nTraining {len(self.models)} models: {[m.__class__.__name__ for m in self.models]}")

        # Infer frequency from data
        if len(nf_df) > 1:
            time_diff = (nf_df['ds'].iloc[1] - nf_df['ds'].iloc[0]).total_seconds() / 60
            freq = f'{int(time_diff)}min' if time_diff < 60 else f'{int(time_diff/60)}h'
        else:
            freq = '5min'

        print(f"  - Inferred frequency: {freq}")

        # Initialize NeuralForecast
        self.nf = NeuralForecast(
            models=self.models,
            freq=freq
        )

        # Fit models
        self.nf.fit(df=nf_df)

        # Generate forecasts
        print(f"\nGenerating {self.horizon}-step forecast...")
        forecasts = self.nf.predict()

        return forecasts, nf_df

    def get_ensemble_forecast(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble forecast from multiple models."""
        model_cols = [c for c in forecasts.columns if c not in ['ds', 'unique_id']]

        if model_cols:
            forecasts['ensemble'] = forecasts[model_cols].mean(axis=1)
            forecasts['ensemble_std'] = forecasts[model_cols].std(axis=1)
            forecasts['ensemble_low'] = forecasts['ensemble'] - 1.96 * forecasts['ensemble_std']
            forecasts['ensemble_high'] = forecasts['ensemble'] + 1.96 * forecasts['ensemble_std']

        return forecasts


def plot_results(historical: pd.DataFrame, forecasts: pd.DataFrame, symbol: str, df_raw: pd.DataFrame):
    """Plot historical data and forecasts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Plot 1: Price and forecast
        ax1 = axes[0]

        # Historical prices
        ax1.plot(historical['ds'], historical['y'], 'b-', label='Historical', linewidth=1.5, alpha=0.8)

        # Model forecasts
        model_cols = [c for c in forecasts.columns if c not in ['ds', 'unique_id', 'ensemble', 'ensemble_std', 'ensemble_low', 'ensemble_high']]
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_cols)))

        for col, color in zip(model_cols, colors):
            ax1.plot(forecasts['ds'], forecasts[col], '--', color=color, label=col, alpha=0.6, linewidth=1)

        # Ensemble with confidence interval
        if 'ensemble' in forecasts.columns:
            ax1.plot(forecasts['ds'], forecasts['ensemble'], 'r-', label='Ensemble', linewidth=2)
            if 'ensemble_low' in forecasts.columns:
                ax1.fill_between(
                    forecasts['ds'],
                    forecasts['ensemble_low'],
                    forecasts['ensemble_high'],
                    alpha=0.2, color='red', label='95% CI'
                )

        ax1.set_title(f'{symbol} - Price Forecast ({len(historical)} bars history)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Plot 2: Volume
        ax2 = axes[1]
        if 'volume' in df_raw.columns:
            ax2.bar(df_raw['timestamp'], df_raw['volume'], alpha=0.7, color='steelblue', width=0.002)
            ax2.set_ylabel('Volume (lots)')
            ax2.set_title('Trading Volume', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Plot 3: Buy/Sell ratio or Foreign flow
        ax3 = axes[2]
        if 'foreign_net' in df_raw.columns:
            colors = ['green' if x >= 0 else 'red' for x in df_raw['foreign_net']]
            ax3.bar(df_raw['timestamp'], df_raw['foreign_net'], alpha=0.7, color=colors, width=0.002)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_ylabel('Foreign Net')
            ax3.set_title('Foreign Flow (Net Lots)', fontsize=10)
        elif 'buy_sell_ratio' in df_raw.columns:
            ax3.plot(df_raw['timestamp'], df_raw['buy_sell_ratio'], 'g-', linewidth=1)
            ax3.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
            ax3.set_ylabel('Buy/Sell Ratio')
            ax3.set_title('Buy/Sell Pressure', fontsize=10)

        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Time')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.tight_layout()

        # Save
        output_path = f'forecast_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved: {output_path}")

        plt.show()

    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")


def print_forecast_table(forecasts: pd.DataFrame, symbol: str, last_price: float):
    """Print forecast results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"FORECAST RESULTS: {symbol}")
    print(f"{'='*70}")

    model_cols = [c for c in forecasts.columns if c not in ['ds', 'unique_id', 'ensemble_std', 'ensemble_low', 'ensemble_high']]

    # Header
    print(f"\n{'Time':<20}", end='')
    for col in model_cols:
        col_display = col[:12]
        print(f"{col_display:>12}", end='')
    print()
    print("-" * (20 + 12 * len(model_cols)))

    # Data rows
    for _, row in forecasts.iterrows():
        time_str = row['ds'].strftime('%Y-%m-%d %H:%M') if hasattr(row['ds'], 'strftime') else str(row['ds'])
        print(f"{time_str:<20}", end='')
        for col in model_cols:
            print(f"{row[col]:>12.2f}", end='')
        print()

    # Summary
    if 'ensemble' in forecasts.columns:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Last known price:    {last_price:>12,.2f}")
        print(f"  Forecast mean:       {forecasts['ensemble'].mean():>12,.2f}")
        print(f"  Forecast min:        {forecasts['ensemble'].min():>12,.2f}")
        print(f"  Forecast max:        {forecasts['ensemble'].max():>12,.2f}")
        pct_change = (forecasts['ensemble'].iloc[-1] / last_price - 1) * 100
        direction = "UP" if pct_change > 0 else "DOWN"
        print(f"  Expected move:       {pct_change:>+11.2f}% ({direction})")
        if 'ensemble_std' in forecasts.columns:
            print(f"  Avg uncertainty:     {forecasts['ensemble_std'].mean():>12,.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Stock Price Forecasting using NeuralForecast (Intraday)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python forecast.py ARCI
  python forecast.py ARCI --horizon 20 --interval 5
  python forecast.py ARCI --horizon 10 --plot
  python forecast.py --list
        """
    )

    parser.add_argument('symbol', nargs='?', help='Stock symbol to forecast')
    parser.add_argument('--horizon', '-n', type=int, default=10,
                        help='Forecast horizon (number of bars)')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='Bar interval in minutes (default: 5)')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate and display plots')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available symbols')
    parser.add_argument('--source', '-s', default='sources',
                        help='Path to data source directory')

    args = parser.parse_args()

    # Initialize data loader
    loader = StockDataLoader(args.source)

    # List available symbols
    if args.list:
        symbols = loader.get_available_symbols()
        if symbols:
            print("Available symbols:")
            for sym in sorted(symbols):
                sessions = loader.get_sessions(sym)
                print(f"  {sym}: {len(sessions)} session(s)")
        else:
            print(f"No data found in '{args.source}' directory")
        return

    # Check symbol
    if not args.symbol:
        parser.print_help()
        print("\nError: Symbol is required")
        sys.exit(1)

    symbol = args.symbol.upper()

    available = loader.get_available_symbols()
    if symbol not in available:
        print(f"Error: Symbol '{symbol}' not found")
        print(f"Available: {', '.join(sorted(available))}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"STOCK FORECASTER - {symbol}")
    print(f"{'='*70}")
    print(f"Interval: {args.interval}min | Horizon: {args.horizon} bars")

    try:
        # Load data
        print(f"\nLoading tick data for {symbol}...")
        df = loader.load_symbol_data(symbol, interval=args.interval)

        print(f"\nTotal: {len(df)} OHLCV bars")
        print(f"Price range: {df['close'].min():,.2f} - {df['close'].max():,.2f}")
        print(f"Last price: {df['close'].iloc[-1]:,.2f}")

        # Forecast
        forecaster = StockForecaster(horizon=args.horizon)
        forecasts, historical = forecaster.train_and_forecast(df)
        forecasts = forecaster.get_ensemble_forecast(forecasts)

        # Results
        print_forecast_table(forecasts, symbol, df['close'].iloc[-1])

        # Plot
        if args.plot:
            plot_results(historical, forecasts, symbol, df)

        # Save
        output_file = f'forecast_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        forecasts.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
