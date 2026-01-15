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
- **yfinance** - Yahoo Finance data (secondary data source)
- **ccxt** - Cryptocurrency exchange library (available for potential cross-market analysis)

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

## External APIs

- **Stockbit Exodus API** (`exodus.stockbit.com`) - Primary data source
  - Screener templates
  - Price feeds and order books
  - Running trades
  - Market detector (bandar analysis)
  - Foreign/domestic flow data (findata)

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

# List available options
python forecast.py --list                  # List symbols
python forecast.py --list-groups           # List stock groups
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SB_AUTH` | Stockbit API Bearer token | Yes |

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

- `forecast_{SYMBOL}_{timestamp}.csv` - Daily forecast results
- `short_{SYMBOL}_{timestamp}.csv` - Intraday forecast results
- `plot/` - Generated charts (PNG)
