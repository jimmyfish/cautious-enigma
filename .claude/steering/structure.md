# Project Structure

## Root Directory Organization

```
helpme/
├── screener.py           # Fetch screener template results
├── initiate.py           # Initialize market data for a symbol
├── analyze_bsjp.py       # Bulk analysis orchestrator
├── scripts/
│   └── analyze_market.py # Analytics pipeline
├── sources/              # Data storage directory
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (gitignored)
└── .claude/
    └── steering/         # Project steering documents
```

## Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `screener.py` | Fetch stock screener results | `python screener.py [TEMPLATE_ID]` |
| `initiate.py` | Collect market data for a symbol | `python initiate.py {SYMBOL}` |
| `analyze_bsjp.py` | Full bulk analysis workflow | `python analyze_bsjp.py` |
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
| `market-detector.json` | Stockbit API | Bandar detector, broker summary, foreign buyers |
| `price-feed.json` | Stockbit API | OHLC prices, bid/offer levels, foreign net |
| `orderbook.json` | Stockbit API | Full order book depth |
| `running-trade.json` | Stockbit API | Price chart data, broker chart data |
| `today-running-trade.json` | Stockbit API | Recent trade-by-trade data |
| `findata.json` | Stockbit API | Foreign vs domestic flow breakdown |
| `analysis-data-*.json` | Generated | Structured analytics summary |
| `analysis-bulk-*.md` | Generated | Markdown report for bulk analysis |

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
