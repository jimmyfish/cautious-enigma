# HelpMe - IDX Stock Market Analysis Toolkit

A command-line toolkit for analyzing Indonesian Stock Exchange (IDX) stocks. Provides automated data collection, technical analysis, and short-term trading signal generation focused on 2-3 day trading opportunities.

## Features

- **Stock Screener Integration** - Fetch filtered stock lists from Stockbit screener templates
- **Market Data Collection** - Automated retrieval of price feeds, order books, running trades, and market detector data
- **Foreign Flow Analysis** - Track foreign vs domestic investor activity and net foreign fund flows
- **Technical Analytics** - Generate structured analytics including price trends, volatility, support/resistance levels
- **Bulk Analysis Reports** - Create comprehensive markdown reports analyzing multiple stocks simultaneously
- **Bandar Detection** - Identify accumulation/distribution signals from market maker activity
- **Price Forecasting** - Neural network-based price predictions using TFT, NBEATS, NHITS models
- **Intraday Forecasting** - Session-level forecasts for next trading day using tick data
- **Yahoo Finance Forecasting** - Forecast global stocks using yfinance with technical indicators
- **Cross-Validation** - Evaluate model accuracy with time-series CV (MAE, RMSE, MAPE, direction accuracy)
- **Group Training** - Train models on 18+ IDX sectors (banking, energy, technology, etc.)
- **Model Persistence** - Save and resume training with checkpoint support for incremental learning
- **Telegram Notifications** - Optional bot integration for automated alerts

## Setup

### Prerequisites

- Python 3.8+
- Stockbit account with API access

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd helpme

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
SB_AUTH=Bearer YOUR_STOCKBIT_TOKEN_HERE

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

To get your Stockbit token:
1. Log in to stockbit.com
2. Open browser developer tools (F12)
3. Go to Network tab
4. Make any action and find a request to `exodus.stockbit.com`
5. Copy the `Authorization` header value

## Usage

### Fetch Screener Data

```bash
# Use default screener template
python screener.py

# Use specific template ID
python screener.py 4475032
```

Output: `screener.json` in project root

### Initialize Data for a Symbol

```bash
# Single symbol
python initiate.py BBRI

# Process a group
python initiate.py -g banking

# Process multiple groups
python initiate.py -g banking,mining

# Exclude groups (process all except)
python initiate.py -e banking

# Control concurrency (default: 3)
python initiate.py -g banking -j 2    # Balanced (12 concurrent requests)
python initiate.py -g banking -j 1    # Safest (6 concurrent requests)
python initiate.py -g banking -j 5    # Aggressive (30 concurrent requests)
```

This creates a new session directory under `sources/BBRI/{session}/` and fetches:
- `market-detector.json` - Bandar detector and broker summary (7 days)
- `price-feed.json` - Historical daily OHLCV + foreign flow (50 days)
- `orderbook.json` - Full order book depth (real-time snapshot)
- `running-trade.json` - Price chart and broker chart data
- `today-running-trade.json` - Recent trade-by-trade data
- `findata.json` - Foreign vs domestic flow breakdown
- `analyzed.json` - Generated summary of all data sources

### Generate Analytics

```bash
python scripts/analyze_market.py BBRI 1
```

Output: `sources/BBRI/1/analysis-data-{timestamp}.json`

### Price Forecasting (Daily)

```bash
# Basic forecast (5-day horizon)
python forecast.py BBRI

# Custom horizon
python forecast.py BBRI --horizon 10

# Group training (train with all banking stocks)
python forecast.py BBCA --group banking

# Force retrain (ignore saved model)
python forecast.py BBRI --retrain

# List available symbols and groups
python forecast.py --list
python forecast.py --list-groups
```

Output: `forecast_BBRI_{timestamp}.csv` with predictions and confidence intervals

### Intraday Forecasting

```bash
# Forecast next session 1 (09:00-12:00)
python short.py ICBP --session1

# Forecast next session 2 (13:30-15:50)
python short.py ICBP --session2

# Custom duration
python short.py ICBP --hours 2

# Group training
python short.py BBCA --group banking --session1

# Disable auto-plot
python short.py ICBP --session1 --no-plot
```

Output: `short_ICBP_{timestamp}.csv` with intraday predictions

### Yahoo Finance Forecasting

Forecast any global stock using Yahoo Finance data:

```bash
# Basic forecast
python yf.py AAPL

# IDX stocks via Yahoo Finance
python yf.py BBRI.JK --horizon 10

# Multiple symbols (comma-separated)
python yf.py AAPL,GOOGL,TSLA

# Intraday data
python yf.py AAPL --interval 1h --period 5d --horizon 24

# Disable auto-plot
python yf.py AAPL --no-plot
```

Output: `yf_{SYMBOL}_{timestamp}.csv` with predictions and charts in `plot/`

### Cross-Validation

Evaluate model accuracy using time-series cross-validation:

```bash
# CV on local tick data
python cross.py ICBP --source tick

# CV on Yahoo Finance data
python cross.py AAPL --source yfinance --period 1y

# Custom CV parameters
python cross.py ICBP --n-windows 5 --horizon 3 --refit
```

Output: `cross_{SYMBOL}_{timestamp}.csv` with MAE, RMSE, MAPE, direction accuracy metrics

### Configure Stock Groups

Edit `models/groups.json` to define your stock groups. The file includes 18+ IDX sectors:

```json
{
  "banking": ["BBCA", "BBRI", "BMRI", "BBNI", "BRIS", ...],
  "energy": ["ADRO", "PGAS", "PTBA", "ITMG", "MEDC", ...],
  "material": ["ANTM", "BRMS", "SMGR", "BRPT", "INTP", ...],
  "technology": ["MTDL", "EMTK", "BUKA", "DCII", ...],
  "property": ["CTRA", "BSDE", "PWON", "SMRA", ...],
  "consumer_non_cyclical": ["GGRM", "HMSP", "UNVR", "INDF", "ICBP", ...],
  "consumer_cyclical": ["AUTO", "MAPI", "ACES", "ERAA", ...],
  "health": ["KLBF", "SIDO", "KAEF", "MIKA", ...],
  "telco": ["TLKM", "EXCL", "ISAT", "TOWR", ...],
  "industrial": ["UNTR", "HEXA", "IMPC", "TOTO", ...],
  ...
}
```

Available sectors: `banking`, `energy`, `material`, `technology`, `property`, `consumer_non_cyclical`, `consumer_cyclical`, `health`, `telco`, `industrial`, `utility`, `infrastructure`, `construction`, `transportation`, `logistic`, `investment`, `lending`, `insurance`, `holding`

### Run Bulk Analysis (Full Workflow)

```bash
python analyze_bsjp.py
```

This runs the complete workflow:
1. Fetches fresh screener data
2. Initializes data for each symbol
3. Generates analytics for each symbol
4. Creates a comprehensive bulk analysis report

Output: `sources/BULK_BSJP/analysis-bulk-{timestamp}.md`

## Project Structure

```
helpme/
├── screener.py           # Fetch screener template results
├── initiate.py           # Initialize market data for a symbol/group
├── analyze_bsjp.py       # Bulk analysis orchestrator
├── forecast.py           # Daily price forecasting (NeuralForecast)
├── short.py              # Intraday session forecasting
├── yf.py                 # Yahoo Finance data forecasting
├── cross.py              # Cross-validation for model evaluation
├── scripts/
│   └── analyze_market.py # Analytics pipeline
├── sources/              # Data storage
│   ├── {SYMBOL}/
│   │   └── {SESSION}/    # Session directories (1, 2, 3...)
│   └── BULK_BSJP/        # Bulk analysis reports
├── models/               # Saved models and config
│   ├── groups.json       # Stock group definitions (18+ IDX sectors)
│   ├── forecast_{SYMBOL}/    # Daily forecast models
│   ├── short_{SYMBOL}_{interval}min/  # Intraday models
│   └── group_{name}/     # Group-trained models
├── plot/                 # Generated charts
├── requirements.txt
├── .env                  # API credentials (not in git)
└── README.md
```

## Data Sources

All data is fetched from **Stockbit Exodus API** (`exodus.stockbit.com`):

| Endpoint | Description |
|----------|-------------|
| `/screener/templates/{id}` | Custom screener results |
| `/marketdetectors/{symbol}` | Market detector / bandar analysis (7 days) |
| `/company-price-feed/historical/summary/{symbol}` | Historical daily OHLCV + foreign flow (50 days) |
| `/company-price-feed/v2/orderbook/companies/{symbol}` | Real-time order book depth |
| `/order-trade/running-trade/chart/{symbol}` | Running trade chart data |
| `/order-trade/running-trade` | Today's running trades |
| `/findata-view/foreign-domestic/v1/chart-data/{symbol}` | Foreign/domestic flow |

### Concurrency Recommendations

The `-j` flag controls how many symbols are processed in parallel. Each symbol makes 6 concurrent API requests.

| Mode | Flag | Concurrent Requests | Risk Level |
|------|------|---------------------|------------|
| Safest | `-j 1` | 6 | None |
| Balanced | `-j 2` | 12 | Low |
| Default | `-j 3` | 18 | Medium |
| Aggressive | `-j 5` | 30 | High (may trigger retries) |

## Analysis Metrics

The analytics pipeline generates these key metrics:

- **Price Series**: Trend slope, volatility (std_return), support/resistance levels
- **Order Book Depth**: Bid/offer volumes, spread, liquidity score
- **Trade Flow**: Buy/sell volume ratio, broker activity
- **Foreign Flow**: Net foreign investment, participation percentage
- **Bandar Signal**: Accumulation/distribution detection
- **Short-Term Outlook**: Bullish/Bearish/Neutral recommendation

## Forecasting Models

The forecasting tools use **NeuralForecast** with these neural network architectures:

| Model | Type | Loss Function | Output |
|-------|------|---------------|--------|
| TFT | Temporal Fusion Transformer | StudentT Distribution | Point + 80%/90% confidence intervals |
| NBEATS | Basis decomposition | HuberLoss | Point forecast (robust to outliers) |
| NHITS | Multi-scale hierarchical | HuberLoss / StudentT | Point + confidence intervals |
| LSTM | Recurrent neural network | HuberLoss | Point forecast |

### Key Features

- **Model Persistence**: Models are saved after training and reused. New data triggers fine-tuning with warm-start (faster training).
- **Gap Handling**: Missing dates/bars are automatically filled with `available_mask` so models can handle incomplete data.
- **Group Training**: Train on multiple related stocks (e.g., all banking stocks) to learn common patterns.
- **Robust Loss Functions**: HuberLoss handles price outliers (gaps, big moves). StudentT distribution handles heavy tails.

### Alpha Features Used

The models learn from these market signals:

- **OBI** (Orderbook Imbalance) - Buy/sell pressure from order depth
- **Bandar Concentration** - Top broker activity indicating institutional moves
- **Foreign Flow** - Net foreign transaction value
- **Volatility** - Intraday price range
- **AccDist Signal** - Accumulation/distribution patterns

## License

Private project - All rights reserved
