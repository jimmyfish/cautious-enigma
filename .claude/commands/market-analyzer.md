# Market Data Analysis Command

## Command Usage

```bash
/analyze-market {SYMBOL} {SESSION}
```

**Parameters:**
- `{SYMBOL}` = Trading pair symbol (e.g., BTCUSDT, ETHUSDT, ADMR, AAPL)
- `{SESSION}` = Session/batch identifier (any unsigned integer, e.g., 1, 12345, 99999). If no SESSION is provided, use the most recent ones.

**Example:**
```bash
/analyze-market ADMR 1
/analyze-market BTCUSDT 1
/analyze-market ETHUSDT 99999
```

---

## Workflow Pipeline

### Step 1: Generate Structured Analytics

**First, run the analytics exporter script:**
```bash
python scripts/analyze_market.py {SYMBOL} {SESSION}
```

This processes all JSON files in `sources/{SYMBOL}/{SESSION}/` and generates:
- `sources/{SYMBOL}/{SESSION}/analysis-data-{TIMESTAMP}.json`

**Note:** Order book data is automatically sourced from:
- `orderbook.json` if present, OR
- `price-feed.json` (from `data.bid` and `data.offer` sections) if `orderbook.json` is missing

The analytics JSON includes `metadata.orderbook_source` indicating which file provided the depth data.

### Step 2: Load Pre-Processed Analytics

**DO NOT read raw JSON files.** Only load:
- ✅ `sources/{SYMBOL}/{SESSION}/analysis-data-*.json` (the most recent one)

**Skip these raw files:**
- ❌ `price-feed.json`
- ❌ `running-trade.json`
- ❌ `today-running-trade.json`
- ❌ `market-detector.json`
- ❌ `orderbook.json` (if present)

All required metrics are pre-computed in the analytics JSON:
- `metadata`: Symbol, session, timestamps, data sources, orderbook source
- `price_feed`: OHLC, volume, foreign flows, bid/offer counts
- `depth`: Order book metrics, spread, weighted prices, volume clusters
- `price_series`: Intraday statistics, trend slope, volatility, support/resistance
- `running_trade`: Trade counts, lot totals, broker activity
- `broker_chart`: Broker value/volume deltas, flow direction
- `market_detector`: Bandar metrics, top buyers/sellers

### Step 3: Generate Analysis Report

Use the structured analytics to write the comprehensive market analysis report.

---

## Input Files Structure

The analytics script processes ALL available data sources from the directory:

**Required Sources (if available):**
1. `sources/{symbol}/{session}/price-feed.json` - Price tick data and OHLCV (includes bid/offer depth)
2. `sources/{symbol}/{session}/orderbook.json` - Order book snapshots and depth (optional, falls back to price-feed.json)
3. `sources/{symbol}/{session}/running-trade.json` - Executed trades and fills
4. `sources/{symbol}/{session}/today-running-trade.json` - Intraday price chart and broker flows
5. `sources/{symbol}/{session}/market-detector.json` - Broker accumulation/distribution signals
6. `sources/{symbol}/{session}/*.json` - Any other JSON files present

**Path Variables:**
- `{symbol}` = Trading pair symbol (e.g., BTCUSDT, ETHUSDT, AAPL, EUR_USD)
- `{session}` = Session/batch identifier (any unsigned integer, e.g., 1, 2, 12345, 99999)

---

## Analysis Framework

### 1. Data Structure Assessment
- Reference `metadata.sources_analyzed` and `metadata.missing_sources`
- Check `metadata.orderbook_source` to know where depth data originated
- Validate data completeness from analytics metrics

### 2. Price Feed Analysis
From `price_feed` and `price_series` sections:
- Identify trend direction (uptrend, downtrend, sideways) using `price_series.slope_per_hour_equiv`
- Detect support and resistance levels from `price_series.top_levels`
- Analyze price volatility using `price_series.std_return` and `price_series.range`
- Calculate price statistics (mean, median from `price_series`)
- Compute percentage changes from `price_feed.percentage_change`
- Identify outliers from `price_series.max_intraday_jump` and `min_intraday_jump`

### 3. Order Book Analysis
From `depth` section:
- Analyze bid-ask spread from `depth.top_of_book.spread` and `spread_bps`
- Identify support/resistance levels from `depth.bid.max_cluster_price` and `depth.offer.max_cluster_price`
- Detect order book imbalances from `depth.bid.total_volume` vs `depth.offer.total_volume`
- Calculate depth and liquidity metrics from `depth.bid.top5_volume`, `top10_volume`
- Spot large orders (walls) from `depth.bid.max_cluster_volume` and `depth.offer.max_cluster_volume`
- Analyze order book pressure from weighted prices and volume distributions

### 4. Trade Execution Analysis
From `running_trade` section:
- Analyze trade flow and direction from `running_trade.lot_buy` vs `lot_sell`
- Calculate trade velocity from `running_trade.trade_count` and time range
- Identify large trades from `running_trade.broker_activity`
- Detect aggressive vs passive orders from trade patterns
- Analyze trade size distribution from lot totals
- Spot unusual trading patterns from broker activity deltas

### 5. Cross-Data Pattern Recognition
- Correlate price movements with broker flows from `broker_chart` deltas
- Identify order flow patterns from `market_detector.bandar` metrics
- Detect market manipulation signals from `market_detector.top1`, `top3`, `top5` accumulation/distribution
- Recognize institutional vs retail trading patterns from broker types
- Spot arbitrage opportunities or anomalies from cross-source discrepancies

### 6. Market Microstructure Analysis
- Liquidity analysis from `depth` metrics and `price_feed.volume`
- Slippage estimation from spread and depth imbalances
- Market impact assessment from `running_trade` patterns
- Tick velocity from `price_series` statistics
- Quote stuffing or other HFT patterns from trade frequency

### 7. Risk Assessment
- Volatility analysis from `price_series.std_return`
- Liquidity risk from `depth` imbalances
- Execution risk from `running_trade` patterns
- Gap and slippage risks from spread metrics
- Potential manipulation risks from `market_detector` signals

---

## Output Requirements

### File Output
**Location**: `sources/{symbol}/{session}/analysis-{timestamp}.md`

Where:
- `{symbol}` = Same as input directory
- `{session}` = Same as input directory
- `{timestamp}` = ISO 8601 format `YYYY-MM-DD_HH-MM-SS` (e.g., `2025-11-11_14-30-45`)

### Content Structure

#### 1. Header Section
```markdown
# Market Analysis Report
**Symbol**: {symbol}
**Session ID**: {session}
**Analysis Date**: {timestamp}
**Data Sources**: [List from metadata.sources_analyzed]
**Order Book Source**: {metadata.orderbook_source}
```

#### 2. Executive Summary
Provide a brief 2-3 paragraph overview using:
- `price_feed` for daily performance
- `price_series` for intraday trends
- `market_detector.bandar` for accumulation/distribution signals

#### 3. Data Inventory
Table showing:
- File names from `metadata.sources_analyzed`
- Record counts from analytics sections
- Time range from `metadata` timestamps
- Data quality assessment
- Missing sources from `metadata.missing_sources` (if any)

#### 4. Detailed Analysis

##### 4.1 Price Action Analysis
- Current trend and strength from `price_series.slope_per_hour_equiv`
- Key price levels (support/resistance) from `price_series.top_levels`
- Volatility metrics from `price_series.std_return`, `range`
- Price patterns identified from `price_series` statistics

##### 4.2 Order Book Insights
- Liquidity depth analysis from `depth.bid.total_volume` vs `depth.offer.total_volume`
- Order book imbalances from volume ratios
- Significant order walls from `depth.bid.max_cluster_volume`, `depth.offer.max_cluster_volume`
- Bid-ask spread analysis from `depth.top_of_book.spread` and `spread_bps`

##### 4.3 Trade Flow Analysis
- Trading volume patterns from `running_trade.lot_total`, `lot_buy`, `lot_sell`
- Large trade identification from `running_trade.broker_activity`
- Trade aggression metrics from buy/sell ratios
- Execution patterns from `running_trade.trade_count` and time range

##### 4.4 Market Microstructure
- Cross-data correlations from `broker_chart` deltas vs `price_series`
- Order flow → price impact from `market_detector` vs price movements
- Market efficiency signals from spread and depth metrics
- Unusual patterns detected from `market_detector.bandar` accumulation/distribution

##### 4.5 Statistical Summary
Present key metrics in tables using pre-computed values:
- Price statistics from `price_feed` and `price_series`
- Volume statistics from `price_feed.volume` and `running_trade`
- Liquidity metrics from `depth` section
- Volatility measures from `price_series.std_return`

#### 5. Second Opinion Section

**Market Bias**: Your independent view on likely near-term direction
- Bullish/Bearish/Neutral with reasoning based on analytics
- Time horizon for the bias

**Confidence Level**: Low / Medium / High
- Explain confidence based on `metadata.missing_sources` and signal strength

**Key Risks**: What could invalidate your analysis
- Technical risks from `price_series` and `depth` metrics
- Liquidity risks from `depth` imbalances
- External factors

**Trading Implications**: 
- Optimal entry/exit zones from `price_series.top_levels` and `depth` clusters
- Position sizing considerations from volatility metrics
- Stop-loss recommendations from support/resistance levels
- Time-of-day considerations from `price_series` patterns

**Alternative Scenarios**: 
- Bear case analysis
- Bull case analysis
- Sideways scenario

#### 6. Actionable Recommendations
- **Immediate Actions**: What to watch in next 1-4 hours
- **Key Levels to Monitor**: Specific price points from `price_series.top_levels` and `depth` clusters
- **Risk Management**: Suggested stop levels and position sizes
- **Further Analysis**: What additional data would be helpful (reference `metadata.missing_sources`)

#### 7. Appendix
- Data anomalies or issues found
- Assumptions made in analysis (e.g., lot size = 100 shares)
- Limitations of the analysis (missing sources, static snapshot vs live data)

#### 8. Second opinion input
- You are the expert market analyzer, and have the one and only job is to make a deep and wide analysis on the market.
- As the expert market analyzer, you want to make a crucial decision about is it the best time to buy or not based on your expert knowledge and the data provided.
- You are able to create a deep and wide market analysis that can easily understand by your trader if it's the best time to buy
- You have vast knowledge about many trade behavior executioner, you are easily undestand the market and output the decision are made.
- You are able to output the analysis in the best decision possible, like is it the best time buy (a generous amount of buy), volatile one (small amount of buy to test the market) or just dont buy it at all.
- As a top-notch market analyzer, your opinion is 100% rely on data and ignore any bias.

---

## Analysis Criteria

- Be objective and data-driven
- Acknowledge limitations from `metadata.missing_sources`
- Provide both bullish and bearish perspectives
- Use specific numbers and percentages from the analytics JSON
- Highlight any unusual or significant findings
- Consider multiple timeframes if data allows
- Cross-validate findings across different analytics sections
- Note any discrepancies between data sources

---

## Usage Example

**Command:**
```bash
/analyze-market ADMR 1
```

**This will:**
1. First run the analytics exporter:
   ```bash
   python scripts/analyze_market.py ADMR 1
   ```
   Generates: `sources/ADMR/1/analysis-data-2025-11-11_18-16-24.json`

2. Load the analytics JSON and generate report:
   - Input: `sources/ADMR/1/analysis-data-*.json`
   - Output: `sources/ADMR/1/analysis-2025-11-11_18-30-45.md`

**Another example:**
```bash
/analyze-market BTCUSDT 1
```
- Run: `python scripts/analyze_market.py BTCUSDT 1`
- Load: `sources/BTCUSDT/1/analysis-data-*.json`
- Output: `sources/BTCUSDT/1/analysis-2025-11-11_14-30-45.md`

---

## Quick Reference

**Next Steps:**
1. Re-run the analytics exporter (regenerates the summarized JSON with the new metadata):
   ```bash
   python scripts/analyze_market.py ADMR 1
   ```
2. When drafting future reports off `analysis-data-*.json`, refer to `metadata.orderbook_source` to know where the depth data originated.

**Notes:**
- The exporter automatically records which source supplied order-book data (`price-feed.json` or `orderbook.json`).
- If `analysis-data-*.json` flags missing inputs in `metadata.missing_sources`, call them out in your report's risk and limitations sections.
- Regenerate analytics after any upstream data change before writing a new report.

---

**Begin your analysis by running the analytics exporter, then load the structured data and generate the comprehensive market analysis report.**
