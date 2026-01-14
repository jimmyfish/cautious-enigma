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
- **pandas** - Data manipulation (available but sparingly used)
- **NumPy** - Numerical operations
- **yfinance** - Yahoo Finance data (secondary data source)
- **ccxt** - Cryptocurrency exchange library (available for potential cross-market analysis)

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
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SB_AUTH` | Stockbit API Bearer token | Yes |

## Data Storage

- All market data stored as JSON files in `sources/{SYMBOL}/{SESSION}/`
- Analysis reports stored as markdown in `sources/BULK_BSJP/`
- Session directories are auto-incremented integers (1, 2, 3, ...)
