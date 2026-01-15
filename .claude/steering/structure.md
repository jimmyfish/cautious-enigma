# Project Structure

## Root Directory Organization

```
helpme/
├── screener.py           # Fetch screener template results
├── initiate.py           # Initialize market data for a symbol
├── analyze_bsjp.py       # Bulk analysis orchestrator
├── forecast.py           # Daily price forecasting (NeuralForecast)
├── short.py              # Intraday session forecasting
├── yf.py                 # Yahoo Finance data forecasting
├── cross.py              # Cross-validation for model evaluation
├── scripts/
│   └── analyze_market.py # Analytics pipeline
├── sources/              # Data storage directory
├── models/               # Saved models and config
│   └── groups.json       # Stock group definitions (18+ sectors)
├── plot/                 # Generated charts
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (gitignored)
└── .claude/
    └── steering/         # Project steering documents
```

## Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `screener.py` | Fetch stock screener results | `python screener.py [TEMPLATE_ID]` |
| `initiate.py` | Collect market data for a symbol/group | `python initiate.py {SYMBOL}` or `-g {group}` |
| `analyze_bsjp.py` | Full bulk analysis workflow | `python analyze_bsjp.py` |
| `forecast.py` | Daily price forecasting | `python forecast.py {SYMBOL} [--horizon N]` |
| `short.py` | Intraday session forecasting | `python short.py {SYMBOL} --session1/--session2` |
| `yf.py` | Yahoo Finance forecasting | `python yf.py {SYMBOL} [--interval 1h]` |
| `cross.py` | Model cross-validation | `python cross.py {SYMBOL} --source tick/yfinance` |
| `scripts/analyze_market.py` | Generate analytics JSON | `python scripts/analyze_market.py {SYMBOL} {SESSION}` |

## Data Directory Structure

```
sources/
├── {SYMBOL}/             # e.g., BBRI, ADMR, UNVR
│   ├── 1/               # Session 1
│   │   ├── market-detector.json
│   │   ├── price-feed.json
│   │   ├── orderbook.json
│   │   ├── running-trade.json
│   │   ├── today-running-trade.json
│   │   ├── findata.json
│   │   └── analysis-data-{timestamp}.json
│   ├── 2/               # Session 2
│   └── ...
└── BULK_BSJP/           # Bulk analysis reports
    └── analysis-bulk-{timestamp}.md
```

## Data File Types

| File | Source | Content |
|------|--------|---------|
| `market-detector.json` | Stockbit API | Bandar detector, broker summary (7 days) |
| `price-feed.json` | Stockbit API | Historical daily OHLCV + foreign flow (50 days) |
| `orderbook.json` | Stockbit API | Real-time order book depth (bid/offer levels) |
| `running-trade.json` | Stockbit API | Price chart data, broker chart data |
| `today-running-trade.json` | Stockbit API | Recent trade-by-trade tick data |
| `findata.json` | Stockbit API | Foreign vs domestic flow breakdown |
| `analyzed.json` | Generated | Comprehensive summary of all data sources |
| `.last_session` | Generated | Session counter for efficient directory numbering |
| `analysis-data-*.json` | Generated | Structured analytics summary (legacy) |
| `analysis-bulk-*.md` | Generated | Markdown report for bulk analysis |

### price-feed.json Structure (Historical API)

The historical price feed contains rich daily data:

```json
{
  "data": {
    "result": [
      {
        "date": "2026-01-15",
        "open": 8000, "high": 8175, "low": 7975, "close": 8075,
        "volume": 1517531, "value": 1227841697500, "frequency": 26293,
        "foreign_buy": 736957717500, "foreign_sell": 793577152500,
        "net_foreign": -56619435000,
        "change": 75, "change_percentage": 0.9375
      }
    ],
    "paginate": { "next_page": "2" }
  }
}
```

- **Pagination**: Max 50 items per page, use `page=2` for more
- **Foreign data**: Per-day foreign_buy, foreign_sell, net_foreign included

## Code Organization Patterns

- **Single-file scripts** - Each entry point is self-contained
- **Utility functions** - Helper functions defined at top of each script
- **Pipeline pattern** - Data flows through collection, analysis, and reporting stages
- **JSON-based data interchange** - All intermediate data stored as JSON

## Key Architectural Principles

1. **Session-based organization** - Each data collection creates a new numbered session directory
2. **Timestamped outputs** - Analysis files include timestamps to preserve history
3. **Fail-soft data collection** - Missing data sources are noted but don't block analysis
4. **Environment-based auth** - API credentials loaded from `.env` file
5. **Markdown reports** - Human-readable output format with tables and structure
6. **Connection pooling** - `requests.Session()` reuses TCP connections for efficiency
7. **Symbol deduplication** - Symbols appearing in multiple groups are processed only once
8. **Efficient session counting** - `.last_session` file avoids full directory scans

## initiate.py Architecture

```
main()
├── Parse args (-g groups, -j jobs, -e exclude)
├── Load groups from groups.json
├── Deduplicate symbols
├── Create requests.Session() with headers
└── ThreadPoolExecutor(max_workers=jobs)
    └── process_symbol() for each symbol
        ├── Create session directory
        ├── ThreadPoolExecutor(max_workers=6)
        │   └── fetch_json_task() × 6 endpoints
        ├── generate_analysis() → analyzed.json
        └── Cleanup source JSON files
```

### Concurrency Model

- **Outer loop**: `jobs` symbols processed in parallel (default: 3)
- **Inner loop**: 6 API endpoints fetched in parallel per symbol
- **Total concurrent requests**: `jobs × 6`
- **Rate limiting**: Exponential backoff on 429/5xx errors
