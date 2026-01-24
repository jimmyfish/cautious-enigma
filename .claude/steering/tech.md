# Technology Stack

## Architecture

This is a command-line Python application with script-based workflows. It follows a pipeline architecture:

1. **Data Collection** - Scripts fetch JSON data from Stockbit API endpoints
2. **Data Processing** - Analytics pipeline transforms raw data into structured summaries
3. **Report Generation** - Bulk analysis script produces markdown reports

## Language & Runtime

- **Python 3.x** - Primary language
- **Standard library modules**: `json`, `argparse`, `pathlib`, `datetime`, `statistics`, `collections`

## Key Dependencies

- **requests** - HTTP client for API calls to Stockbit
- **python-dotenv** - Environment variable management for API tokens
- **pandas** - Data manipulation and time series handling
- **NumPy** - Numerical operations
- **yfinance** - Yahoo Finance data for global stock forecasting
- **colorama** - Colored terminal output for progress tracking

### Forecasting Stack

- **NeuralForecast** - Neural network time series forecasting library (Nixtla)
  - TFT (Temporal Fusion Transformer) - Attention-based model with interpretable features
  - NBEATS - Basis decomposition model
  - NHITS - Multi-scale hierarchical interpolation
  - LSTM - Recurrent neural network
- **PyTorch** - Deep learning backend for NeuralForecast
- **PyTorch Lightning** - Training framework (used internally by NeuralForecast)

### Loss Functions (neuralforecast.losses.pytorch)

- **HuberLoss** - Robust to outliers (combines MSE + MAE)
- **DistributionLoss** - Probabilistic forecasts with confidence intervals
  - StudentT distribution - Heavy-tailed, handles market extremes
  - Poisson - For count data (volume forecasting)
  - NegativeBinomial - Overdispersed count data

### IDX Rules Module

- **idx_rules.py** - Indonesian Stock Exchange price limit rules
  - ARA (Auto Rejection Atas) - Upper daily price limit
  - ARB (Auto Rejection Bawah) - Lower daily price limit
  - Tick size calculation based on price bands
  - Automatic forecast clamping to valid price ranges

### Watchlist Integration

- **watchlist.py** - Stockbit watchlist API integration
  - Add/remove symbols from watchlists
  - Filter by bullish/bearish outlook
  - Clear or append to existing watchlists

## External APIs

- **Stockbit Exodus API** (`exodus.stockbit.com`) - Primary data source
  - Screener templates
  - Historical price feed (50 days daily OHLCV + foreign flow per day)
  - Real-time order book depth
  - Running trades (chart + tick data)
  - Market detector (bandar analysis, 7 days)
  - Foreign/domestic flow data (findata)

### API Rate Limiting

- API may return 429 when too many concurrent requests
- `initiate.py` uses retry with exponential backoff
- Each symbol makes 6 concurrent API calls
- `-j` flag controls parallel symbol processing (default: 3)
- Recommended: `-j 2` for safe operation, `-j 1` if experiencing issues

## Development Environment

### Requirements

- Python 3.8+
- Virtual environment recommended
- `.env` file for API credentials

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your SB_AUTH token
```

## Common Commands

```bash
# Fetch screener data (default template)
python screener.py

# Fetch screener data (specific template)
python screener.py 4475032

# Initialize data for a symbol
python initiate.py BBRI

# Initialize data for a group (with concurrency control)
python initiate.py -g banking              # Default concurrency (3 symbols)
python initiate.py -g banking -j 2         # Balanced (2 symbols = 12 requests)
python initiate.py -g banking -j 1         # Safest (1 symbol = 6 requests)
python initiate.py -g banking,mining       # Multiple groups
python initiate.py -e banking              # All except banking

# Generate analytics for a symbol/session
python scripts/analyze_market.py BBRI 1

# Run bulk analysis workflow
python analyze_bsjp.py

# Daily price forecasting
python forecast.py BBRI                    # Basic 5-day forecast
python forecast.py BBRI --horizon 10       # 10-day forecast
python forecast.py BBCA --group banking    # Group training
python forecast.py BBRI --retrain          # Force retrain

# Intraday forecasting
python short.py ICBP --session1            # Morning session forecast
python short.py ICBP --session2            # Afternoon session forecast
python short.py BBCA --group banking       # Group training

# Yahoo Finance forecasting (global stocks)
python yf.py AAPL                          # Apple daily forecast
python yf.py BBRI.JK --horizon 10          # IDX stock via Yahoo Finance
python yf.py AAPL,GOOGL,TSLA               # Multiple symbols (comma-separated)
python yf.py AAPL --interval 1h --period 5d  # Hourly data

# Cross-validation (model evaluation)
python cross.py ICBP --source tick         # CV on tick data
python cross.py AAPL --source yfinance     # CV on Yahoo Finance data
python cross.py ICBP --n-windows 5         # 5-fold CV

# Watchlist integration (requires -wid watchlist ID)
python forecast.py BBRI -w -wid 123456     # Add to watchlist after forecast
python forecast.py BBRI -w -wid 123456 -bl # Add only if bullish
python forecast.py BBRI -w -wid 123456 -br # Add only if bearish
python forecast.py BBRI -w -wid 123456 --keep  # Keep existing items

python short.py ICBP --session1 -w -wid 123456  # Intraday + watchlist
python yf.py AAPL,GOOGL -w -wid 123456 -bl      # Multiple symbols, bullish only

# List available options
python forecast.py --list                  # List symbols
python forecast.py --list-groups           # List stock groups
python initiate.py --list-sectors          # List available groups
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SB_AUTH` | Stockbit API Bearer token | Yes |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for notifications | No |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for notifications | No |

## Data Storage

- All market data stored as JSON files in `sources/{SYMBOL}/{SESSION}/`
- Analysis reports stored as markdown in `sources/BULK_BSJP/`
- Session directories are auto-incremented integers (1, 2, 3, ...)

### Model Storage

- Trained models saved in `models/` directory
- `models/groups.json` - Stock group definitions for group training
- `models/forecast_{SYMBOL}/` - Daily forecast model checkpoints
- `models/short_{SYMBOL}_{interval}min/` - Intraday model checkpoints
- `models/group_{name}/` - Group-trained model checkpoints
- `models/*_meta.json` - Model metadata (data count, training date)

### Output Files

- `csv/{SYMBOL}/forecast_{timestamp}.csv` - Daily forecast results
- `csv/{SYMBOL}/short_{timestamp}.csv` - Intraday forecast results
- `csv/{SYMBOL}/yf_{timestamp}.csv` - Yahoo Finance forecast results
- `plot/{SYMBOL}/` - Generated charts (PNG)
