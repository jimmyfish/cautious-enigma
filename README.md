# HelpMe - IDX Stock Market Analysis Toolkit

A command-line toolkit for analyzing Indonesian Stock Exchange (IDX) stocks. Provides automated data collection, technical analysis, and short-term trading signal generation focused on 2-3 day trading opportunities.

## Features

- **Stock Screener Integration** - Fetch filtered stock lists from Stockbit screener templates
- **Market Data Collection** - Automated retrieval of price feeds, order books, running trades, and market detector data
- **Foreign Flow Analysis** - Track foreign vs domestic investor activity and net foreign fund flows
- **Technical Analytics** - Generate structured analytics including price trends, volatility, support/resistance levels
- **Bulk Analysis Reports** - Create comprehensive markdown reports analyzing multiple stocks simultaneously
- **Bandar Detection** - Identify accumulation/distribution signals from market maker activity

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
python initiate.py BBRI
```

This creates a new session directory under `sources/BBRI/{session}/` and fetches:
- `market-detector.json` - Bandar detector and broker summary
- `price-feed.json` - Current OHLC and bid/offer levels
- `orderbook.json` - Full order book depth
- `running-trade.json` - Price chart and broker chart data
- `today-running-trade.json` - Recent trade-by-trade data
- `findata.json` - Foreign vs domestic flow breakdown

### Generate Analytics

```bash
python scripts/analyze_market.py BBRI 1
```

Output: `sources/BBRI/1/analysis-data-{timestamp}.json`

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
├── initiate.py           # Initialize market data for a symbol
├── analyze_bsjp.py       # Bulk analysis orchestrator
├── scripts/
│   └── analyze_market.py # Analytics pipeline
├── sources/              # Data storage
│   ├── {SYMBOL}/
│   │   └── {SESSION}/    # Session directories (1, 2, 3...)
│   └── BULK_BSJP/        # Bulk analysis reports
├── requirements.txt
├── .env                  # API credentials (not in git)
└── README.md
```

## Data Sources

All data is fetched from **Stockbit Exodus API** (`exodus.stockbit.com`):

| Endpoint | Description |
|----------|-------------|
| `/screener/templates/{id}` | Custom screener results |
| `/marketdetectors/{symbol}` | Market detector / bandar analysis |
| `/company-price-feed/v2/orderbook/companies/{symbol}` | Price feed and order book |
| `/order-trade/running-trade/chart/{symbol}` | Running trade chart data |
| `/order-trade/running-trade` | Today's running trades |
| `/findata-view/foreign-domestic/v1/chart-data/{symbol}` | Foreign/domestic flow |

## Analysis Metrics

The analytics pipeline generates these key metrics:

- **Price Series**: Trend slope, volatility (std_return), support/resistance levels
- **Order Book Depth**: Bid/offer volumes, spread, liquidity score
- **Trade Flow**: Buy/sell volume ratio, broker activity
- **Foreign Flow**: Net foreign investment, participation percentage
- **Bandar Signal**: Accumulation/distribution detection
- **Short-Term Outlook**: Bullish/Bearish/Neutral recommendation

## License

Private project - All rights reserved
