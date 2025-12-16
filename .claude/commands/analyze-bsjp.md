# Bulk Market Data Analysis Command (Screener-Based)

## Command Usage

```bash
/analyze-bsjp
```

**Prerequisites:**
- `SB_AUTH` environment variable must be set in `.env` file (for fetching screener data)

**Important:** This command **ALWAYS** runs `python screener.py` first to fetch fresh screener results before performing any analysis. This ensures the analysis uses the most up-to-date screened symbols.

**Example:**
```bash
# The command automatically runs screener.py first, then performs analysis
/analyze-bsjp
```

**Note:** The screener data is automatically fetched using `python screener.py [TEMPLATE_ID]` which reads the auth token from `.env`. This ensures fresh screener results for accurate analysis.

---

## Workflow Pipeline

### Step 0: Fetch Screener Results (MANDATORY FIRST STEP)

**ALWAYS fetch fresh screener template results before analysis:**

This step is **automatically executed** at the start of the analysis workflow. The screener script is run to fetch and save the latest screener data:

```bash
python screener.py [TEMPLATE_ID]
```

**Parameters:**
- `[TEMPLATE_ID]` = Optional screener template ID (default: 4475032). If not provided, uses the default template.

**Why this is mandatory:**
- Ensures analysis uses the most current screened symbols
- Screener results change over time as market conditions evolve
- Prevents analysis of stale or outdated symbol lists
- Guarantees accuracy of screened symbol selection

**Requirements:**
- `SB_AUTH` environment variable must be set in `.env` file
- The script will automatically normalize the token (adds "Bearer " prefix if missing).

**Screener JSON Structure:**
The `screener.json` contains:
- `data.calcs[]`: Array of company results
- Each company has:
  - `company.symbol`: Stock symbol (e.g., "PGAS", "MDKA", "FUTR")
  - `company.name`: Company name
  - `results[]`: Screener metrics (Bandar Accum/Dist, Value, Price, Net Foreign Flow, etc.)

**Output:** `screener.json` saved in workspace root (overwrites any existing file).

**Note:** The `screener.py` script is **ALWAYS** run first to fetch `screener.json` with actual and updated symbols for correct screened symbols. This ensures the analysis is based on the latest screener criteria results.

### Step 1: Extract Symbols from Screener

**Parse `screener.json` to extract all company symbols:**

Extract symbols from `data.calcs[].company.symbol` and create a list of unique symbols to analyze.

**Example symbols extracted:**
- PGAS
- MDKA
- FUTR
- BUMI
- IMPC
- EMTK
- MYOR

### Step 2: Bulk Data Initialization

**For each symbol from the screener, run data initialization:**

```bash
python initiate.py {SYMBOL}
```

This script:
- Creates a new session directory in `sources/{SYMBOL}/` (auto-increments the session number)
- Fetches and saves the following JSON files to the new session directory:
  - `market-detector.json`
  - `price-feed.json`
  - `orderbook.json`
  - `running-trade.json`
  - `today-running-trade.json`
  - `findata.json`

**Note:** Process all symbols sequentially or in parallel batches. Track which symbols succeeded and which failed for reporting.

### Step 3: Bulk Analytics Generation

**For each successfully initialized symbol, run the analytics exporter:**

```bash
python scripts/analyze_market.py {SYMBOL} {SESSION}
```

Where `{SESSION}` is the session number created by `initiate.py` (or the most recent session if not specified).

This processes all JSON files in `sources/{SYMBOL}/{SESSION}/` and generates:
- `sources/{SYMBOL}/{SESSION}/analysis-data-{TIMESTAMP}.json`

**Note:** Order book data is automatically sourced from:
- `orderbook.json` if present, OR
- `price-feed.json` (from `data.bid` and `data.offer` sections) if `orderbook.json` is missing

The analytics JSON includes `metadata.orderbook_source` indicating which file provided the depth data.

### Step 4: Load Pre-Processed Analytics

**DO NOT read raw JSON files.** Only load:
- ✅ `sources/{SYMBOL}/{SESSION}/analysis-data-*.json` (the most recent one for each symbol)

**Skip these raw files:**
- ❌ `price-feed.json`
- ❌ `running-trade.json`
- ❌ `today-running-trade.json`
- ❌ `market-detector.json`
- ❌ `orderbook.json` (if present)
- ❌ `findata.json` (if present)

All required metrics are pre-computed in the analytics JSON:
- `metadata`: Symbol, session, timestamps, data sources, orderbook source
- `price_feed`: OHLC, volume, foreign flows, bid/offer counts
- `depth`: Order book metrics, spread, weighted prices, volume clusters
- `price_series`: Intraday statistics, trend slope, volatility, support/resistance
- `running_trade`: Trade counts, lot totals, broker activity
- `broker_chart`: Broker value/volume deltas, flow direction
- `market_detector`: Bandar metrics, top buyers/sellers
- `findata`: Foreign vs domestic trading flow breakdowns (value, volume, frequency)

### Step 5: Generate Bulk Analysis Report

Use the structured analytics from all symbols to write a comprehensive bulk market analysis report focused on **short-term (2-3 trading days) performance**.

---

## Input Files Structure

### Screener Data
- `screener.json` (workspace root): Contains list of companies from screener template

### Per-Symbol Analytics
For each symbol `{SYMBOL}`:
- `sources/{SYMBOL}/{SESSION}/analysis-data-*.json` - Pre-processed analytics (required)

---

## Analysis Framework (Short-Term Focus: 2-3 Trading Days)

### 1. Data Structure Assessment
- Reference `metadata.sources_analyzed` and `metadata.missing_sources` for each symbol
- Check `metadata.orderbook_source` to know where depth data originated
- Validate data completeness from analytics metrics
- Track symbols with missing or incomplete data

### 2. Price Feed Analysis (Short-Term Performance)
From `price_feed` and `price_series` sections:
- **Today's Performance**: Current price vs previous close, percentage change
- **Short-Term Trend**: Identify trend direction (uptrend, downtrend, sideways) using `price_series.slope_per_hour_equiv` projected over 2-3 days
- **Volatility Assessment**: Analyze price volatility using `price_series.std_return` and `price_series.range` - critical for short-term trading
- **Support and Resistance**: Detect key levels from `price_series.top_levels` for entry/exit planning
- **Price Momentum**: Calculate momentum indicators from `price_series` for 2-3 day outlook
- **Intraday Patterns**: Identify patterns from `price_series` that may repeat in next 2-3 days

### 3. Order Book Analysis (Short-Term Liquidity)
From `depth` section:
- **Immediate Liquidity**: Analyze bid-ask spread from `depth.top_of_book.spread` and `spread_bps` - affects short-term execution
- **Support/Resistance Levels**: Identify from `depth.bid.max_cluster_price` and `depth.offer.max_cluster_price` for next 2-3 days
- **Order Book Imbalances**: Detect from `depth.bid.total_volume` vs `depth.offer.total_volume` - indicates short-term pressure
- **Large Orders (Walls)**: Spot from `depth.bid.max_cluster_volume` and `depth.offer.max_cluster_volume` - potential short-term barriers
- **Depth Analysis**: Calculate depth metrics from `depth.bid.top5_volume`, `top10_volume` for short-term trading capacity

### 4. Trade Execution Analysis (Short-Term Flow)
From `running_trade` section:
- **Today's Trade Flow**: Analyze from `running_trade.lot_buy` vs `lot_sell` - indicates short-term sentiment
- **Trade Velocity**: Calculate from `running_trade.trade_count` and time range - shows short-term activity
- **Large Trades**: Identify from `running_trade.broker_activity` - institutional interest for next 2-3 days
- **Aggressive vs Passive**: Detect from trade patterns - shows short-term buying/selling pressure

### 5. Foreign vs Domestic Flow Analysis (Short-Term Sentiment)
From `findata` section (if available):
- **Today's Foreign Flow**: Analyze from `findata.value` (IDR breakdowns) - indicates short-term foreign interest
- **Net Foreign Flow**: Calculate from `findata.summary.net_foreign` - shows 2-3 day foreign sentiment
- **Volume Participation**: Compare foreign vs domestic from `findata.volume` (shares breakdowns) - short-term liquidity source
- **Trading Frequency**: Assess from `findata.frequency` (trade count breakdowns) - shows short-term activity level

### 6. Market Microstructure Analysis (Short-Term Execution)
- **Liquidity Risk**: Analyze from `depth` metrics and `price_feed.volume` - critical for 2-3 day trades
- **Slippage Estimation**: Calculate from spread and depth imbalances - affects short-term execution
- **Market Impact**: Assess from `running_trade` patterns - shows short-term price sensitivity
- **Execution Quality**: Evaluate from spread metrics and depth - important for short-term entries/exits

### 7. Short-Term Risk Assessment (2-3 Days)
- **Volatility Risk**: Analyze from `price_series.std_return` - affects short-term position sizing
- **Liquidity Risk**: Assess from `depth` imbalances - impacts 2-3 day exit capability
- **Execution Risk**: Evaluate from `running_trade` patterns - affects short-term fills
- **Gap Risk**: Assess from spread metrics and volatility - potential overnight gaps
- **Foreign Flow Dependency**: Evaluate if foreign participation is high (from `findata`) - affects 2-3 day stability

### 8. Cross-Symbol Comparison
- **Rank symbols by short-term potential** based on:
  - Today's performance (percentage change)
  - Short-term trend strength (slope)
  - Volatility (risk-adjusted returns)
  - Liquidity (depth and volume)
  - Foreign flow momentum
  - Bandar accumulation/distribution signals
- **Identify top candidates** for 2-3 day trading opportunities
- **Group by risk profile** (low/medium/high volatility)
- **Compare relative strength** across screener results

---

## Output Requirements

### File Output
**Location**: `sources/BULK_BSJP/analysis-bulk-{timestamp}.md`

Where:
- `{timestamp}` = ISO 8601 format `YYYY-MM-DD_HH-MM-SS` (e.g., `2025-11-11_14-30-45`)

### Content Structure

#### 1. Header Section
```markdown
# Bulk Market Analysis Report (Short-Term: 2-3 Trading Days)
**Analysis Date**: {timestamp}
**Screener Source**: screener.json
**Total Symbols Analyzed**: {count}
**Analysis Timeframe**: 2-3 Trading Days
**Screener Criteria**: [Extract from screener.json rules/columns]
```

#### 2. Executive Summary
Provide a brief 2-3 paragraph overview:
- **Overall Market Sentiment**: Aggregate view across all symbols
- **Top Performers Today**: Symbols with best today's performance
- **Short-Term Opportunities**: Top 3-5 symbols with best 2-3 day outlook
- **Risk Assessment**: Overall volatility and liquidity conditions
- **Key Themes**: Common patterns across screener results

#### 3. Screener Overview
Table showing:
- Screener name and criteria from `screener.json`
- Total symbols in screener
- Symbols successfully analyzed
- Symbols with data issues
- Screener metrics (Bandar Accum/Dist thresholds, Value thresholds, etc.)

#### 4. Symbol Performance Summary Table

| Symbol | Company Name | Today % | Price | Short-Term Trend | Volatility | Liquidity Score | Foreign Flow | Bandar Signal | 2-3 Day Outlook |
|--------|--------------|---------|-------|------------------|------------|-----------------|--------------|---------------|-----------------|
| PGAS   | Perusahaan Gas Negara | +2.5% | 1,825 | Uptrend | Medium | High | Positive | Accumulation | Bullish |
| MDKA   | Merdeka Copper Gold | -1.2% | 2,200 | Sideways | High | Medium | Positive | Accumulation | Neutral |

**Columns:**
- **Symbol**: Stock symbol
- **Company Name**: From screener data
- **Today %**: Percentage change from `price_feed.percentage_change`
- **Price**: Current price from `price_feed` or screener
- **Short-Term Trend**: Uptrend/Downtrend/Sideways from `price_series.slope_per_hour_equiv`
- **Volatility**: Low/Medium/High from `price_series.std_return`
- **Liquidity Score**: Low/Medium/High from `depth` metrics
- **Foreign Flow**: Positive/Negative/Neutral from `findata.summary.net_foreign`
- **Bandar Signal**: Accumulation/Distribution/Neutral from `market_detector.bandar`
- **2-3 Day Outlook**: Bullish/Bearish/Neutral based on combined signals

#### 5. Top Short-Term Opportunities (2-3 Days)

For each top candidate (top 3-5 symbols), provide:

##### 5.1 Symbol: {SYMBOL}
- **Company**: {name}
- **Current Price**: {price}
- **Today's Performance**: {percentage_change}%
- **Short-Term Trend**: {trend} (from `price_series.slope_per_hour_equiv`)
- **Key Support/Resistance**: 
  - Support: {levels from `price_series.top_levels` and `depth.bid.max_cluster_price`}
  - Resistance: {levels from `depth.offer.max_cluster_price`}
- **Volatility**: {std_return} (Low/Medium/High)
- **Liquidity**: {assessment from `depth` metrics}
- **Foreign Flow**: {net_foreign from `findata`}
- **Bandar Signal**: {bandar metrics from `market_detector`}
- **2-3 Day Outlook**: {Bullish/Bearish/Neutral with reasoning}
- **Entry Zones**: {specific price levels from support/resistance}
- **Stop Loss Recommendation**: {level based on support}
- **Target Zones**: {levels based on resistance and trend}
- **Risk Level**: Low/Medium/High
- **Confidence**: Low/Medium/High

#### 6. Detailed Analysis by Symbol

For each symbol analyzed, provide a concise summary:

##### 6.1 {SYMBOL} - {Company Name}

**Price Action (2-3 Day Outlook):**
- Current trend and strength from `price_series.slope_per_hour_equiv`
- Key price levels (support/resistance) from `price_series.top_levels` and `depth` clusters
- Volatility metrics from `price_series.std_return`, `range`
- Short-term momentum indicators

**Order Book Insights:**
- Liquidity depth analysis from `depth.bid.total_volume` vs `depth.offer.total_volume`
- Order book imbalances from volume ratios
- Significant order walls from `depth.bid.max_cluster_volume`, `depth.offer.max_cluster_volume`
- Bid-ask spread analysis from `depth.top_of_book.spread` and `spread_bps`

**Trade Flow Analysis:**
- Trading volume patterns from `running_trade.lot_total`, `lot_buy`, `lot_sell`
- Large trade identification from `running_trade.broker_activity`
- Trade aggression metrics from buy/sell ratios

**Foreign vs Domestic Flow (if available):**
- Foreign vs domestic value breakdowns from `findata.value` (IDR)
- Net foreign flow analysis from `findata.summary.net_foreign`
- Foreign vs domestic buy/sell imbalances

**Short-Term Trading Recommendation (2-3 Days):**
- **Action**: Buy/Neutral/Avoid
- **Entry Zone**: {price levels}
- **Stop Loss**: {level}
- **Target**: {price levels}
- **Position Size**: Small/Medium/Large (based on volatility and liquidity)
- **Time Horizon**: 2-3 trading days
- **Key Risks**: {specific risks}

#### 7. Cross-Symbol Patterns

**Common Themes:**
- Sector trends (if applicable)
- Foreign flow patterns across symbols
- Volatility clusters
- Liquidity conditions

**Relative Strength Ranking:**
- Rank all symbols by short-term potential
- Group by risk profile
- Identify best risk-adjusted opportunities

#### 8. Risk Summary

**Overall Market Conditions:**
- Aggregate volatility assessment
- Liquidity conditions
- Foreign flow trends
- Market sentiment

**Symbol-Specific Risks:**
- High volatility symbols
- Low liquidity symbols
- Symbols with data quality issues
- Symbols with conflicting signals

#### 9. Actionable Recommendations (2-3 Day Focus)

**Immediate Actions:**
- Top 3 symbols to watch in next 2-3 trading days
- Key levels to monitor for each top symbol
- Entry/exit timing considerations

**Risk Management:**
- Suggested stop levels for top opportunities
- Position sizing guidelines based on volatility
- Diversification recommendations

**Monitoring Checklist:**
- What to watch for each top symbol in next 2-3 days
- Key indicators that would change outlook
- Data refresh recommendations

#### 10. Appendix

**Data Quality:**
- Symbols with missing data sources
- Symbols with incomplete analytics
- Data freshness timestamps

**Screener Details:**
- Full screener criteria from `screener.json`
- Screener rules and filters
- Original screener metrics for each symbol

**Limitations:**
- Assumptions made in analysis
- Timeframe limitations (2-3 days focus)
- Data snapshot vs live data considerations

---

## Analysis Criteria (Short-Term Focus)

- **Time Horizon**: All analysis focused on 2-3 trading days, NOT longer-term
- **Be objective and data-driven**: Use specific numbers from analytics JSON
- **Prioritize short-term signals**: Trend, momentum, volatility, liquidity for 2-3 day trades
- **Acknowledge limitations**: Missing data sources, snapshot data
- **Provide actionable recommendations**: Specific entry/exit levels, stop losses, targets
- **Rank and prioritize**: Identify best opportunities for short-term trading
- **Cross-validate**: Compare findings across symbols and data sources
- **Risk-first approach**: Highlight risks prominently for short-term trading

---

## Usage Example

**Command:**
```bash
/analyze-bsjp
```

**This will:**
1. **ALWAYS** fetch fresh screener data first:
   ```bash
   python screener.py
   ```
   Creates/updates: `screener.json` in workspace root with latest screener results
2. Load `screener.json` and extract symbols: PGAS, MDKA, FUTR, BUMI, IMPC, EMTK, MYOR (example symbols)
3. For each symbol, run data initialization:
   ```bash
   python initiate.py PGAS
   python initiate.py MDKA
   # ... etc
   ```
   Creates: `sources/{SYMBOL}/{NEW_SESSION}/` with fresh JSON data files

4. For each symbol, run analytics exporter:
   ```bash
   python scripts/analyze_market.py PGAS {NEW_SESSION}
   python scripts/analyze_market.py MDKA {NEW_SESSION}
   # ... etc
   ```
   Generates: `sources/{SYMBOL}/{NEW_SESSION}/analysis-data-{TIMESTAMP}.json`

5. Load all analytics JSON files and generate bulk report:
   - Input: All `sources/{SYMBOL}/{SESSION}/analysis-data-*.json` files
   - Output: `sources/BULK_BSJP/analysis-bulk-2025-11-11_18-30-45.md`

**With custom screener template:**
```bash
python screener.py 4475032
/analyze-bsjp
```
- Fetches screener result using template ID 4475032 and saves to `screener.json`
- Continues with steps 2-5 above

---

## Quick Reference

**Next Steps:**
1. **ALWAYS** fetch fresh screener data first (automatically executed):
   ```bash
   python screener.py [TEMPLATE_ID]
   ```
   Creates/updates: `screener.json` in workspace root (requires `SB_AUTH` in `.env`)
2. Extract symbols from `screener.json` → `data.calcs[].company.symbol`
3. For each symbol:
   - Run: `python initiate.py {SYMBOL}` (creates new session)
   - Run: `python scripts/analyze_market.py {SYMBOL} {SESSION}` (generates analytics)
4. Load all `analysis-data-*.json` files
5. Generate bulk report focused on **2-3 day short-term trading opportunities**

**Notes:**
- The bulk analysis focuses on **short-term (2-3 trading days)** performance and opportunities
- All recommendations should be actionable for 2-3 day trading horizon
- Rank symbols by short-term potential, not long-term value
- Track data quality issues per symbol for accurate reporting
- Compare relative strength across all screener symbols

---

**Begin your bulk analysis by automatically running `python screener.py` first to fetch fresh screener data, then extracting symbols, initializing data for each symbol, generating analytics, and finally creating a comprehensive bulk market analysis report focused on short-term (2-3 trading days) opportunities. The screener step is mandatory and ensures you're analyzing the most current screened symbols.**
