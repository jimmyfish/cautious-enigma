#!/usr/bin/env python3
"""
Stock Price Forecasting using NeuralForecast
Usage: python forecast.py SYMBOL [--horizon N] [--plot]

Example: python forecast.py ARCI --horizon 5 --plot
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM, TFT
from neuralforecast.losses.pytorch import MAE, MSE

import warnings
warnings.filterwarnings('ignore')


class StockDataLoader:
    """Load and aggregate stock data from JSON files."""

    def __init__(self, base_path: str = "sources"):
        self.base_path = Path(base_path)

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
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

    def extract_price_features(self, data: dict) -> dict:
        """Extract features from price-feed.json."""
        if not data or 'data' not in data:
            return {}

        d = data['data']
        features = {
            'open': float(d.get('open', 0)),
            'high': float(d.get('high', 0)),
            'low': float(d.get('low', 0)),
            'close': float(d.get('close', 0)) or float(d.get('lastprice', 0)),
            'volume': float(d.get('volume', 0)),
            'value': float(d.get('value', 0)),
            'frequency': float(d.get('frequency', 0)),
            'average': float(d.get('average', 0)),
            'change': float(d.get('change', 0)),
            'pct_change': float(d.get('percentage_change', 0)),
            'foreign_buy': float(d.get('fbuy', 0)),
            'foreign_sell': float(d.get('fsell', 0)),
            'foreign_net': float(d.get('fnet', 0)),
        }

        # Calculate bid/offer imbalance from orderbook
        bid_volume = sum(float(b.get('volume', 0)) for b in d.get('bid', []))
        offer_volume = sum(float(o.get('volume', 0)) for o in d.get('offer', []))
        features['bid_volume'] = bid_volume
        features['offer_volume'] = offer_volume
        features['book_imbalance'] = (bid_volume - offer_volume) / (bid_volume + offer_volume + 1e-10)

        return features

    def extract_market_detector_features(self, data: dict) -> dict:
        """Extract features from market-detector.json."""
        if not data or 'data' not in data:
            return {}

        d = data['data']
        features = {}

        # Extract date
        if 'from' in d:
            features['date'] = d['from']

        # Bandar detector features
        bd = d.get('bandar_detector', {})
        if bd:
            features['bandar_value'] = float(bd.get('value', 0))
            features['bandar_volume'] = float(bd.get('volume', 0))
            features['total_buyers'] = int(bd.get('total_buyer', 0))
            features['total_sellers'] = int(bd.get('total_seller', 0))
            features['buyer_seller_ratio'] = features['total_buyers'] / (features['total_sellers'] + 1)

            # Accumulation/Distribution signals
            avg_data = bd.get('avg', {})
            features['avg_net_amount'] = float(avg_data.get('amount', 0))
            features['avg_net_percent'] = float(avg_data.get('percent', 0))

            # Convert accdist to numeric
            accdist_map = {'Big Acc': 2, 'Normal Acc': 1, 'Neutral': 0, 'Normal Dist': -1, 'Big Dist': -2}
            features['accdist_signal'] = accdist_map.get(avg_data.get('accdist', 'Neutral'), 0)

            # Top broker signals
            top1 = bd.get('top1', {})
            features['top1_net_amount'] = float(top1.get('amount', 0))
            features['top1_accdist'] = accdist_map.get(top1.get('accdist', 'Neutral'), 0)

        return features

    def extract_findata_features(self, data: dict) -> dict:
        """Extract features from findata.json."""
        if not data or 'data' not in data:
            return {}

        d = data['data']
        features = {}

        # Extract date
        if 'from' in d:
            features['date'] = d['from']

        summary = d.get('summary', {})
        if summary:
            # Foreign flow
            fb = summary.get('foreign_buy', {}).get('value', {})
            fs = summary.get('foreign_sell', {}).get('value', {})
            nf = summary.get('net_foreign', {}).get('value', {})

            features['fin_foreign_buy'] = float(fb.get('raw', 0))
            features['fin_foreign_sell'] = float(fs.get('raw', 0))
            features['fin_net_foreign'] = float(nf.get('raw', 0))

            # Domestic flow
            db = summary.get('domestic_buy', {}).get('value', {})
            ds = summary.get('domestic_sell', {}).get('value', {})

            features['fin_domestic_buy'] = float(db.get('raw', 0))
            features['fin_domestic_sell'] = float(ds.get('raw', 0))

        return features

    def extract_running_trade_features(self, data: dict) -> dict:
        """Extract aggregated features from running-trade data."""
        if not data or 'data' not in data:
            return {}

        trades = data['data'].get('running_trade', [])
        if not trades:
            return {}

        features = {}

        # Parse trades
        lots = []
        buy_lots = 0
        sell_lots = 0
        foreign_buy_lots = 0
        foreign_sell_lots = 0

        for trade in trades:
            try:
                lot = int(str(trade.get('lot', '0')).replace(',', ''))
                lots.append(lot)

                action = trade.get('action', '')
                buyer_type = trade.get('buyer_type', '')
                seller_type = trade.get('seller_type', '')

                if action == 'buy':
                    buy_lots += lot
                    if 'FOREIGN' in buyer_type:
                        foreign_buy_lots += lot
                else:
                    sell_lots += lot
                    if 'FOREIGN' in seller_type:
                        foreign_sell_lots += lot
            except (ValueError, TypeError):
                continue

        if lots:
            features['trade_count'] = len(lots)
            features['avg_lot_size'] = np.mean(lots)
            features['max_lot_size'] = np.max(lots)
            features['total_buy_lots'] = buy_lots
            features['total_sell_lots'] = sell_lots
            features['buy_sell_lot_ratio'] = buy_lots / (sell_lots + 1)
            features['foreign_buy_lots'] = foreign_buy_lots
            features['foreign_sell_lots'] = foreign_sell_lots

        return features

    def load_session_data(self, symbol: str, session: str) -> dict:
        """Load all data for a specific session."""
        session_path = self.base_path / symbol / session

        all_features = {'symbol': symbol, 'session': session}

        # Load each type of file
        price_feed = self.load_json(session_path / 'price-feed.json')
        if price_feed:
            all_features.update(self.extract_price_features(price_feed))

        market_detector = self.load_json(session_path / 'market-detector.json')
        if market_detector:
            all_features.update(self.extract_market_detector_features(market_detector))

        findata = self.load_json(session_path / 'findata.json')
        if findata:
            all_features.update(self.extract_findata_features(findata))

        running_trade = self.load_json(session_path / 'running-trade.json')
        if running_trade:
            all_features.update(self.extract_running_trade_features(running_trade))

        # Try today-running-trade if running-trade not available
        if 'trade_count' not in all_features:
            today_trade = self.load_json(session_path / 'today-running-trade.json')
            if today_trade:
                all_features.update(self.extract_running_trade_features(today_trade))

        # Check for analysis files to extract date
        for f in session_path.glob('analysis-data-*.json'):
            # Extract date from filename: analysis-data-2025-12-04_17-29-19.json
            try:
                date_str = f.stem.replace('analysis-data-', '').split('_')[0]
                all_features['date'] = date_str
                break
            except:
                pass

        return all_features

    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load all session data for a symbol into a DataFrame."""
        sessions = self.get_sessions(symbol)

        if not sessions:
            raise ValueError(f"No data found for symbol: {symbol}")

        all_data = []
        for session in sessions:
            session_data = self.load_session_data(symbol, session)
            if session_data.get('close', 0) > 0:  # Only include valid data
                all_data.append(session_data)

        if not all_data:
            raise ValueError(f"No valid price data found for symbol: {symbol}")

        df = pd.DataFrame(all_data)

        # Handle date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date')
        else:
            # Generate synthetic dates if not available
            base_date = datetime.now() - timedelta(days=len(df))
            df['date'] = [base_date + timedelta(days=i) for i in range(len(df))]

        df = df.reset_index(drop=True)
        return df


class StockForecaster:
    """NeuralForecast-based stock price forecaster."""

    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self.models = None
        self.nf = None

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """Prepare data in NeuralForecast format."""
        # Select features
        feature_cols = [
            'volume', 'value', 'frequency', 'pct_change',
            'foreign_net', 'book_imbalance', 'buyer_seller_ratio',
            'accdist_signal', 'avg_net_percent'
        ]

        # Filter to available columns
        available_features = [c for c in feature_cols if c in df.columns]

        # Create base dataframe
        nf_df = pd.DataFrame({
            'unique_id': df['symbol'],
            'ds': df['date'],
            'y': df[target_col]
        })

        # Add exogenous variables
        for col in available_features:
            nf_df[col] = df[col].fillna(0)

        # Handle missing/infinite values
        nf_df = nf_df.replace([np.inf, -np.inf], 0)
        nf_df = nf_df.fillna(0)

        return nf_df, available_features

    def create_models(self, input_size: int, exog_vars: list):
        """Create NeuralForecast models."""

        # Adjust input size based on data availability
        effective_input = min(input_size, 10)

        models = [
            # N-BEATS: Pure time series model
            NBEATS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type='robust',
                random_seed=42
            ),

            # N-HiTS: Hierarchical interpolation
            NHITS(
                h=self.horizon,
                input_size=effective_input,
                max_steps=200,
                scaler_type='robust',
                random_seed=42
            ),
        ]

        # Add LSTM if we have enough data
        if input_size >= 5:
            models.append(
                LSTM(
                    h=self.horizon,
                    input_size=effective_input,
                    max_steps=200,
                    scaler_type='robust',
                    random_seed=42
                )
            )

        return models

    def train_and_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train models and generate forecasts."""

        nf_df, exog_vars = self.prepare_data(df)

        print(f"\nData prepared:")
        print(f"  - Records: {len(nf_df)}")
        print(f"  - Date range: {nf_df['ds'].min()} to {nf_df['ds'].max()}")
        print(f"  - Target (close) range: {nf_df['y'].min():.2f} to {nf_df['y'].max():.2f}")
        print(f"  - Exogenous features: {len(exog_vars)}")

        if len(nf_df) < 5:
            raise ValueError(f"Insufficient data points ({len(nf_df)}). Need at least 5 records.")

        # Create models
        input_size = min(len(nf_df) - 1, 10)
        self.models = self.create_models(input_size, exog_vars)

        print(f"\nTraining {len(self.models)} models...")

        # Initialize NeuralForecast
        self.nf = NeuralForecast(
            models=self.models,
            freq='D'  # Daily frequency
        )

        # Fit models
        self.nf.fit(df=nf_df)

        # Generate forecasts
        print(f"Generating {self.horizon}-step forecast...")
        forecasts = self.nf.predict()

        return forecasts, nf_df

    def get_ensemble_forecast(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble forecast from multiple models."""

        # Get model columns (exclude ds and unique_id)
        model_cols = [c for c in forecasts.columns if c not in ['ds', 'unique_id']]

        if model_cols:
            forecasts['ensemble'] = forecasts[model_cols].mean(axis=1)
            forecasts['ensemble_std'] = forecasts[model_cols].std(axis=1)

        return forecasts


def plot_results(historical: pd.DataFrame, forecasts: pd.DataFrame, symbol: str):
    """Plot historical data and forecasts."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Price history and forecast
        ax1 = axes[0]
        ax1.plot(historical['ds'], historical['y'], 'b-', label='Historical', linewidth=2)

        # Plot each model's forecast
        model_cols = [c for c in forecasts.columns if c not in ['ds', 'unique_id', 'ensemble', 'ensemble_std']]
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_cols)))

        for col, color in zip(model_cols, colors):
            ax1.plot(forecasts['ds'], forecasts[col], '--', color=color, label=col, alpha=0.7)

        # Plot ensemble with confidence interval
        if 'ensemble' in forecasts.columns:
            ax1.plot(forecasts['ds'], forecasts['ensemble'], 'r-', label='Ensemble', linewidth=2)
            if 'ensemble_std' in forecasts.columns:
                ax1.fill_between(
                    forecasts['ds'],
                    forecasts['ensemble'] - 2 * forecasts['ensemble_std'],
                    forecasts['ensemble'] + 2 * forecasts['ensemble_std'],
                    alpha=0.2, color='red', label='95% CI'
                )

        ax1.set_title(f'{symbol} Price Forecast', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume/Value history
        ax2 = axes[1]
        if 'volume' in historical.columns:
            ax2.bar(historical['ds'], historical['volume'], alpha=0.7, label='Volume')
            ax2.set_ylabel('Volume')
        elif 'value' in historical.columns:
            ax2.bar(historical['ds'], historical['value'], alpha=0.7, label='Value')
            ax2.set_ylabel('Value')

        ax2.set_xlabel('Date')
        ax2.set_title('Trading Activity', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = f'forecast_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {output_path}")

        plt.show()

    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")


def print_forecast_table(forecasts: pd.DataFrame, symbol: str):
    """Print forecast results in a formatted table."""

    print(f"\n{'='*60}")
    print(f"FORECAST RESULTS FOR {symbol}")
    print(f"{'='*60}")

    model_cols = [c for c in forecasts.columns if c not in ['ds', 'unique_id']]

    # Header
    print(f"\n{'Date':<12}", end='')
    for col in model_cols:
        print(f"{col:<15}", end='')
    print()
    print("-" * (12 + 15 * len(model_cols)))

    # Data rows
    for _, row in forecasts.iterrows():
        date_str = row['ds'].strftime('%Y-%m-%d') if hasattr(row['ds'], 'strftime') else str(row['ds'])
        print(f"{date_str:<12}", end='')
        for col in model_cols:
            print(f"{row[col]:>14.2f} ", end='')
        print()

    # Summary statistics
    if 'ensemble' in forecasts.columns:
        print(f"\n{'='*60}")
        print("ENSEMBLE FORECAST SUMMARY")
        print(f"{'='*60}")
        print(f"  Mean forecast:  {forecasts['ensemble'].mean():,.2f}")
        print(f"  Min forecast:   {forecasts['ensemble'].min():,.2f}")
        print(f"  Max forecast:   {forecasts['ensemble'].max():,.2f}")
        if 'ensemble_std' in forecasts.columns:
            print(f"  Avg uncertainty: {forecasts['ensemble_std'].mean():,.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Stock Price Forecasting using NeuralForecast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python forecast.py ARCI
  python forecast.py ARCI --horizon 10
  python forecast.py ARCI --horizon 5 --plot
  python forecast.py --list
        """
    )

    parser.add_argument('symbol', nargs='?', help='Stock symbol to forecast')
    parser.add_argument('--horizon', '-n', type=int, default=5,
                        help='Forecast horizon (number of periods)')
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

    # Check symbol argument
    if not args.symbol:
        parser.print_help()
        print("\nError: Symbol is required")
        sys.exit(1)

    symbol = args.symbol.upper()

    # Check if symbol exists
    available = loader.get_available_symbols()
    if symbol not in available:
        print(f"Error: Symbol '{symbol}' not found")
        print(f"Available symbols: {', '.join(sorted(available))}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"STOCK PRICE FORECASTER - {symbol}")
    print(f"{'='*60}")

    try:
        # Load data
        print(f"\nLoading data for {symbol}...")
        df = loader.load_symbol_data(symbol)

        sessions = loader.get_sessions(symbol)
        print(f"  - Found {len(sessions)} session(s): {', '.join(sessions)}")
        print(f"  - Loaded {len(df)} data points")

        # Display data summary
        print(f"\nData Summary:")
        print(f"  - Latest close: {df['close'].iloc[-1]:,.2f}")
        print(f"  - Price range: {df['close'].min():,.2f} - {df['close'].max():,.2f}")
        if 'volume' in df.columns:
            print(f"  - Avg volume: {df['volume'].mean():,.0f}")
        if 'foreign_net' in df.columns:
            print(f"  - Total foreign net: {df['foreign_net'].sum():,.0f}")

        # Initialize forecaster
        forecaster = StockForecaster(horizon=args.horizon)

        # Train and forecast
        forecasts, historical = forecaster.train_and_forecast(df)

        # Add ensemble
        forecasts = forecaster.get_ensemble_forecast(forecasts)

        # Print results
        print_forecast_table(forecasts, symbol)

        # Plot if requested
        if args.plot:
            plot_results(historical, forecasts, symbol)

        # Save forecasts to CSV
        output_file = f'forecast_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        forecasts.to_csv(output_file, index=False)
        print(f"\nForecast saved to: {output_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
