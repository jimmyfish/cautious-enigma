# Market Data Analysis Prompt for Gemini Web

## Instructions for Use

1. **Attach your JSON data files** using Gemini web's file attachment feature
2. **Copy this entire prompt** and submit it along with your attached files
3. Gemini will analyze the attached files and generate the analysis report

---

## Files to Attach

### Required File:
- **analysis-data-*.json** - Pre-processed analytics (REQUIRED)
  - This file contains all the structured metrics needed for analysis
  - It should include: `metadata`, `price_feed`, `depth`, `price_series`, `running_trade`, `broker_chart`, `market_detector`, and `findata` sections

### Optional Files (for additional context):
You may also attach any of these raw source files if available:
- **findata.json** - Foreign vs domestic trading flow data
- **price-feed.json** - Price tick data and OHLCV
- **orderbook.json** - Order book snapshots
- **running-trade.json** - Executed trades
- **today-running-trade.json** - Intraday price chart
- **market-detector.json** - Broker accumulation/distribution signals

**Note:** The primary data source is the `analysis-data-*.json` file. The optional raw files are only for additional context if needed.

---

# Analysis Framework

## Your Task

You are an expert market analyzer. Analyze the attached market data files and generate a comprehensive market analysis report. 

**Primary Data Source:**
- Use the attached `analysis-data-*.json` file as your primary data source
- This file contains pre-processed analytics with all structured metrics

**IMPORTANT**: 
- **DO NOT read raw JSON files directly** - use the pre-processed analytics from `analysis-data-*.json`
- All required metrics are pre-computed in the analytics JSON
- Reference `metadata.sources_analyzed` and `metadata.missing_sources` to understand data completeness
- Check `metadata.orderbook_source` to know where depth data originated
- If other JSON files are attached, they are only for additional context - prioritize the analytics data

---

## Analysis Framework

### 1. Data Structure Assessment
- Reference `metadata.sources_analyzed` and `metadata.missing_sources` from the analytics JSON
- Check `metadata.orderbook_source` to know where depth data originated
- Validate data completeness from analytics metrics
- Note any missing data sources and their impact on analysis quality

### 2. Price Feed Analysis
From `price_feed` and `price_series` sections:
- Identify trend direction (uptrend, downtrend, sideways) using `price_series.slope_per_hour_equiv`
- Detect support and resistance levels from `price_series.top_levels`
- Analyze price volatility using `price_series.std_return` and `price_series.range`
- Calculate price statistics (mean, median from `price_series`)
- Compute percentage changes from `price_feed.percentage_change` or `price_feed.pct_change`
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

### 5. Foreign vs Domestic Flow Analysis
From `findata` section (if available in analytics JSON):
- Analyze foreign vs domestic trading activity from `findata.value` (IDR breakdowns)
- Compare foreign vs domestic volume participation from `findata.volume` (shares breakdowns)
- Assess trading frequency patterns from `findata.frequency` (trade count breakdowns)
- Calculate net foreign flow from `findata.summary.net_foreign`
- Identify foreign vs domestic buy/sell imbalances
- Cross-reference `findata` flows with `price_feed.foreign_buy/foreign_sell` for consistency
- Assess market participation distribution (foreign vs domestic percentages)

### 6. Cross-Data Pattern Recognition
- Correlate price movements with broker flows from `broker_chart` deltas
- Identify order flow patterns from `market_detector.bandar` metrics
- Detect market manipulation signals from `market_detector.top1`, `top3`, `top5` accumulation/distribution
- Recognize institutional vs retail trading patterns from broker types and `findata` foreign/domestic breakdowns
- Spot arbitrage opportunities or anomalies from cross-source discrepancies
- Correlate `findata` foreign flows with `market_detector` top buyers/sellers

### 7. Market Microstructure Analysis
- Liquidity analysis from `depth` metrics and `price_feed.volume`
- Slippage estimation from spread and depth imbalances
- Market impact assessment from `running_trade` patterns
- Tick velocity from `price_series` statistics
- Quote stuffing or other HFT patterns from trade frequency
- Foreign vs domestic participation impact on liquidity (from `findata` if available)

### 8. Risk Assessment
- Volatility analysis from `price_series.std_return`
- Liquidity risk from `depth` imbalances
- Execution risk from `running_trade` patterns
- Gap and slippage risks from spread metrics
- Potential manipulation risks from `market_detector` signals
- Foreign flow dependency risk (if foreign participation is high, from `findata`)

---

## Output Requirements

Generate a comprehensive market analysis report in Markdown format with the following structure:

### 1. Header Section
```markdown
# Market Analysis Report
**Symbol**: {extract from metadata or analysis-data filename}
**Session ID**: {extract from metadata or analysis-data filename}
**Analysis Date**: {current date/time or from metadata.generated_at}
**Data Sources**: [List from metadata.sources_analyzed]
**Order Book Source**: {metadata.orderbook_source}
```

### 2. Executive Summary
Provide a brief 2-3 paragraph overview using:
- `price_feed` for daily performance
- `price_series` for intraday trends
- `market_detector.bandar` for accumulation/distribution signals
- `findata` for foreign vs domestic flow context (if available)

### 3. Data Inventory
Table showing:
- File names from `metadata.sources_analyzed`
- Record counts from analytics sections
- Time range from `metadata` timestamps
- Data quality assessment
- Missing sources from `metadata.missing_sources` (if any)

### 4. Detailed Analysis

#### 4.1 Price Action Analysis
- Current trend and strength from `price_series.slope_per_hour_equiv`
- Key price levels (support/resistance) from `price_series.top_levels`
- Volatility metrics from `price_series.std_return`, `range`
- Price patterns identified from `price_series` statistics

#### 4.2 Order Book Insights
- Liquidity depth analysis from `depth.bid.total_volume` vs `depth.offer.total_volume`
- Order book imbalances from volume ratios
- Significant order walls from `depth.bid.max_cluster_volume`, `depth.offer.max_cluster_volume`
- Bid-ask spread analysis from `depth.top_of_book.spread` and `spread_bps`

#### 4.3 Trade Flow Analysis
- Trading volume patterns from `running_trade.lot_total`, `lot_buy`, `lot_sell`
- Large trade identification from `running_trade.broker_activity`
- Trade aggression metrics from buy/sell ratios
- Execution patterns from `running_trade.trade_count` and time range

#### 4.4 Foreign vs Domestic Flow Analysis (if `findata` available)
- Foreign vs domestic value breakdowns from `findata.value` (IDR)
- Foreign vs domestic volume participation from `findata.volume` (shares)
- Trading frequency patterns from `findata.frequency` (trade counts)
- Net foreign flow analysis from `findata.summary.net_foreign`
- Foreign vs domestic buy/sell imbalances
- Market participation distribution (percentages)
- Cross-reference with `price_feed.foreign_buy/foreign_sell` for consistency

#### 4.5 Market Microstructure
- Cross-data correlations from `broker_chart` deltas vs `price_series`
- Order flow → price impact from `market_detector` vs price movements
- Market efficiency signals from spread and depth metrics
- Unusual patterns detected from `market_detector.bandar` accumulation/distribution
- Foreign vs domestic participation impact (from `findata` if available)

#### 4.6 Statistical Summary
Present key metrics in tables using pre-computed values:
- Price statistics from `price_feed` and `price_series`
- Volume statistics from `price_feed.volume` and `running_trade`
- Liquidity metrics from `depth` section
- Volatility measures from `price_series.std_return`
- Foreign vs domestic flow statistics from `findata` (if available)

### 5. Second Opinion Section

**Market Bias**: Your independent view on likely near-term direction
- Bullish/Bearish/Neutral with reasoning based on analytics
- Time horizon for the bias (e.g., daily, 1-3 days, weekly) - appropriate for daily trading decisions, NOT hourly trades

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

### 6. Actionable Recommendations
- **Immediate Actions**: What to watch in the next trading day(s) (1-3 days timeframe)
- **Key Levels to Monitor**: Specific price points from `price_series.top_levels` and `depth` clusters
- **Risk Management**: Suggested stop levels and position sizes
- **Further Analysis**: What additional data would be helpful (reference `metadata.missing_sources`)

### 7. Appendix
- Data anomalies or issues found
- Assumptions made in analysis (e.g., lot size = 100 shares)
- Limitations of the analysis (missing sources, static snapshot vs live data)

### 8. Expert Market Analysis Decision

As the expert market analyzer, provide your crucial decision:

**Is it the best time to buy?**

Based on your expert knowledge and the data provided, make a clear recommendation:

- **GENEROUS BUY**: Strong buy signal with high confidence - recommend buying a generous amount
- **VOLATILE BUY**: Moderate buy signal with some risk - recommend buying a small amount to test the market
- **DON'T BUY**: Weak or bearish signals - recommend not buying at all

Provide detailed reasoning for your decision based on:
- Data-driven analysis (100% rely on data, ignore any bias)
- Market behavior patterns you recognize
- Risk assessment
- Trading execution considerations

**IMPORTANT**: The data provided is for **daily trading decisions**, not hourly or intraday trades. All time horizons in your analysis should reflect daily trading (e.g., daily, 1-3 days, weekly), NOT short-term hourly timeframes (1-4 hours).

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
- Your opinion must be 100% data-driven and ignore any bias

---

## Begin Analysis

Now analyze the attached JSON files (especially the `analysis-data-*.json` file) and generate the comprehensive market analysis report following the framework outlined above.

**Remember:**
- Primary data source: `analysis-data-*.json` file
- Extract symbol and session information from the filename or `metadata` section
- Use specific numbers and percentages from the analytics data
- Be objective and data-driven in your analysis
