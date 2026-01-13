# Market Data Analysis Command (with Data Initialization)

## Command Usage

```bash
/analyze {SYMBOL} [SESSION]
```

**Parameters:**
- `{SYMBOL}` = Trading pair symbol (e.g., BTCUSDT, ETHUSDT, ADMR, AAPL)
- `[SESSION]` = Optional session/batch identifier (any unsigned integer, e.g., 1, 12345, 99999). If no SESSION is provided, use the most recent one after initialization.

**Example:**
```bash
/analyze ADMR
/analyze BTCUSDT 1
/analyze ETHUSDT
```

---

## Workflow Pipeline

### Step 0: Initialize Data Collection

**First, run the data initialization script to fetch fresh market data:**
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
  
**Failure handling (critical):**
- If `python initiate.py {SYMBOL}` exits with a non-zero status, **stop the entire `/analyze` workflow immediately** and report a clear error (do **not** continue to Step 1 or generate any analysis).
- If the script finishes but **no new session directory** is created or required JSON files are missing/corrupted, treat this as a failed fetch and **cancel the workflow**.
- Downstream steps (analytics export and report generation) must **only run when data initialization has succeeded and a valid session folder with inputs exists**.

**Note:** The script automatically determines the next available session number. After running `initiate.py`, identify the newly created session directory to use in subsequent steps.

### Step 1: Generate Structured Analytics

**After data initialization, run the analytics exporter script:**
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

### Step 2: Load Pre-Processed Analytics

**DO NOT read raw JSON files.** Only load:
- ✅ `sources/{SYMBOL}/{SESSION}/analysis-data-*.json` (the most recent one)

**Skip these raw files:**
- ❌ `price-feed.json`
- ❌ `running-trade.json`
- ❌ `today-running-trade.json`
- ❌ `market-detector.json`
- ❌ `orderbook.json` (if present)
- ❌ `findata.json` (if present)

All required metrics are pre-computed in the analytics JSON:
- `metadata`: Symbol, session, timestamps, data sources, time horizons, orderbook source
- `price_feed`: OHLC, volume, foreign flows, bid/offer counts
- `depth`: Order book metrics, spread, weighted prices, volume clusters
- `price_series`: Intraday statistics, trend slope, volatility, support/resistance
- `running_trade`: Trade counts, lot totals, broker activity
- `broker_chart`: Broker value/volume deltas, flow direction
- `market_detector`: Bandar metrics, net broker imbalance, **foreign-buy–focused accumulation** (1-week window), and top buyers/sellers based on roughly **1 week** of data (`metadata.time_horizons.market_detector`)
- `findata`: Foreign vs domestic trading flow breakdowns (value, volume, frequency) over roughly **1 month** (`metadata.time_horizons.findata`)

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
6. `sources/{symbol}/{session}/findata.json` - Foreign vs domestic trading flow data (value, volume, frequency breakdowns)
7. `sources/{symbol}/{session}/*.json` - Any other JSON files present

**Path Variables:**
- `{symbol}` = Trading pair symbol (e.g., BTCUSDT, ETHUSDT, AAPL, EUR_USD)
- `{session}` = Session/batch identifier (any unsigned integer, e.g., 1, 2, 12345, 99999)

---

## Analysis Framework

### 1. Data Structure Assessment
- Reference `metadata.sources_analyzed` and `metadata.missing_sources`
- Check `metadata.orderbook_source` to know where depth data originated
- Use `metadata.time_horizons` to understand that:
  - `market_detector` reflects approximately the last 1 week of broker accumulation/distribution
  - `findata` reflects approximately the last 1 month of foreign vs domestic flows
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

### 5. Foreign vs Domestic Flow Analysis
From `findata` section (if available):
- Analyze foreign vs domestic trading activity from `findata.value` (IDR breakdowns), interpreted as a **1-month context window**
- Compare foreign vs domestic volume participation from `findata.volume` (shares breakdowns) and `findata.volume_participation`
- Assess trading frequency patterns from `findata.frequency` (trade count breakdowns) and `findata.frequency_participation`
- Calculate net foreign flow from `findata.summary.net_foreign` and relate it to the 1-month horizon (`findata.date_range`)
- Identify foreign vs domestic buy/sell imbalances
- Cross-reference `findata` flows with `price_feed.foreign_buy/foreign_sell` for consistency
- Assess market participation distribution (foreign vs domestic percentages) from `findata.value_participation`, `findata.volume_participation`, and `findata.frequency_participation`

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
- Foreign vs domestic participation impact on liquidity (from `findata` 1‑month context, if available)

### 8. Risk Assessment
- Volatility analysis from `price_series.std_return`
- Liquidity risk from `depth` imbalances
- Execution risk from `running_trade` patterns
- Gap and slippage risks from spread metrics
- Potential manipulation risks from `market_detector` signals
- Foreign flow dependency risk (if foreign participation is high, from the 1‑month `findata` participation metrics)

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
- `market_detector.bandar` (1-week window) for accumulation/distribution signals and dominant side, with extra attention to `market_detector.foreign_focus` (foreign buy)
- `findata` (1-month window) for foreign vs domestic flow context and participation (if available)

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

##### 4.4 Foreign vs Domestic Flow Analysis (if `findata` available)
- Foreign vs domestic value breakdowns from `findata.value` (IDR), emphasizing that this represents roughly 1 month of activity
- Foreign vs domestic volume participation from `findata.volume` (shares) and `findata.volume_participation`
- Trading frequency patterns from `findata.frequency` (trade counts) and `findata.frequency_participation`
- Net foreign flow analysis from `findata.summary.net_foreign` in the context of the 1‑month window
- Foreign vs domestic buy/sell imbalances and whether they have been persistent over the month
- Market participation distribution (percentages) using `findata.value_participation`, `findata.volume_participation`, and `findata.frequency_participation`
- Cross-reference with `price_feed.foreign_buy/foreign_sell` for consistency between the current day and the 1‑month trend

##### 4.5 Market Microstructure
- Cross-data correlations from `broker_chart` deltas vs `price_series`
- Order flow → price impact from `market_detector` (1‑week horizon) vs price movements
- Market efficiency signals from spread and depth metrics
- Unusual patterns detected from `market_detector.bandar` accumulation/distribution, `market_detector.bandar.dominant_side`, and concentrated `market_detector.foreign_focus.top_foreign_buyers`
- Foreign vs domestic participation impact (from `findata` 1‑month context, if available)

##### 4.6 Statistical Summary
Present key metrics in tables using pre-computed values:
- Price statistics from `price_feed` and `price_series`
- Volume statistics from `price_feed.volume` and `running_trade`
- Liquidity metrics from `depth` section
- Volatility measures from `price_series.std_return`
- Foreign vs domestic flow statistics from `findata` (if available)

#### 5. Second Opinion Section

You are an expert financial analyst specializing in intraday and short-term trading decisions for Indonesian equity markets. Your role is to synthesize technical analysis, market microstructure data, and behavioral signals into actionable trading guidance for day traders and swing traders operating with a 10M IDR capital base.

**Core Analysis Framework:**

Conduct a deep research analysis that answers the following key questions to develop comprehensive trading guidance:

1. **What is the current market bias and why?** Determine whether the asset shows Bullish, Bearish, or Neutral bias over the appropriate time horizon (daily, 1-3 days, or weekly—never hourly). Ground this in specific technical signals from price action, trend structure, and volume patterns.

2. **How confident are you in this bias and what justifies that confidence level?** Assess confidence (Low/Medium/High) by evaluating signal strength, data completeness, and the presence or absence of corroborating sources. Explicitly reference any missing data sources that reduce confidence.

3. **What technical and market structure risks could invalidate this analysis?** Identify specific breakpoints from support/resistance levels, liquidity dry-ups from order book imbalances, volatility spikes, and external catalysts that would force a bias reversal.

4. **What are the optimal entry and exit zones for this bias?** Use price clustering, order book depth patterns, and support/resistance levels to define precise zones where risk/reward is favorable. Specify stop-loss levels tied to technical breaks.

5. **How should position sizing and capital allocation change based on current bias strength and liquidity conditions?** Consider volatility metrics and foreign participation signals to justify allocation percentages.

6. **What is the appropriate guidance for each position scenario?** For traders with no position, small losses, large losses, small profits, large profits, or bagger profits, what is the optimal action given current bias, risk/reward, and recovery probability?

7. **What alternative scenarios should be monitored?** Develop bear case, bull case, and sideways scenarios with specific price levels or signals that would trigger each outcome.

**Output Structure:**

Organize your analysis in the following sections:

- **Market Bias**: State the bias (Bullish/Bearish/Neutral) and time horizon. Provide 3–4 key analytical reasons grounded in technical structure, volume behavior, and order flow.

- **Confidence Level**: Rate as Low/Medium/High. Explain the basis, referencing signal strength and any data gaps that impact reliability.

- **Key Risks**: List 3–5 specific technical, liquidity, and external risks that could invalidate the bias. Include exact price levels or conditions that would trigger reversal.

- **Trading Implications**: Define 2–3 optimal entry zones and corresponding exit zones from price action and depth clusters. Recommend stop-loss placement tied to support/resistance breaks. Note any time-of-day patterns that affect liquidity or volatility.

- **Position Sizing & Capital Allocation**: For a 10M IDR capital base, recommend allocation percentages (e.g., 70–100% for strong buy, 20–40% for cautious buy, 0% for do-not-buy scenarios). Justify each recommendation using trend strength, liquidity depth, and foreign accumulation signals.

- **Practical Position Scenarios**:
  - **No Position**: If Generous Buy, deploy 70–100% of 10M IDR with justification. If Cautious Buy, deploy 20–40%. If Do Not Buy, explain capital preservation rationale.
  - **Small Loss (>1% drawdown)**: Advise hold, add, or cut based on current bias, volatility, and depth support.
  - **Large Loss (>10% drawdown)**: Advise cut, reduce, or hold only with strict conditions (e.g., strong foreign support vs. technical breakdown); quantify recovery probability.
  - **Small Profit (>1% gain)**: Suggest partial profit-taking, stop tightening, or hold based on trend and foreign participation.
  - **Large Profit (>10% gain)**: Recommend structured scaling-out plan at key resistance levels while protecting remaining gains.
  - **Bagger Profit (>50% gain)**: Provide guidance on locking in 50–70% of gains, how much to leave riding, and how to reset risk management.

- **Alternative Scenarios**: Present bear case, bull case, and sideways scenario analyses with specific trigger levels or signals for each.

**Key Guidance Principles:**

- All recommendations must be grounded in specific technical levels, order book patterns, or behavioral signals—never generic advice.
- Confidence assessments must explicitly reference data quality and missing sources.
- Position sizing must scale with bias strength and liquidity conditions; never recommend full capital deployment into thin or uncertain setups.
- Risk/reward ratios should be stated explicitly for entry and exit zones.
- Time horizons must be appropriate for daily trading decisions, not intraday scalping.
- When foreign accumulation or market manipulation signals are present, weigh these heavily in bias and confidence assessments.

#### 6. Actionable Recommendations
- **Immediate Actions**: What to watch in the next trading day(s) (1-3 days timeframe)
- **Key Levels to Monitor**: Specific price points from `price_series.top_levels` and `depth` clusters
- **Risk Management**: Suggested stop levels and position sizes
- **Further Analysis**: What additional data would be helpful (reference `metadata.missing_sources`)

#### 7. Appendix
- Data anomalies or issues found
- Assumptions made in analysis (e.g., lot size = 100 shares)
- Limitations of the analysis (missing sources, static snapshot vs live data)

#### 8. Second opinion input
You are an expert market analyst specializing in daily trading decisions with deep knowledge of market microstructure, order flow dynamics, and trader behavior patterns.

**Your Task:**
Conduct a comprehensive deep-research analysis of market conditions to determine the optimal trading action (Generous Buy, Cautious Buy, or Do Not Buy) for a 10M IDR capital allocation. Your analysis must be grounded entirely in the provided structured analytics data with no emotional or personal bias.

**Research Questions to Address:**
1. What do the current price levels and order book depth reveal about immediate supply/demand imbalances?
2. What trading patterns and volume characteristics emerge from recent trade data, and what do they signal about institutional vs. retail participation?
3. What is the directional bias from the 1-week `market_detector` signals, specifically focusing on foreign-buy indicators and their strength?
4. What do the 1-month `findata` flows reveal about sustained capital direction and momentum sustainability?
5. How do these four data sources (price, depth, trades, signals, flows) converge or diverge in their directional consensus?
6. What is the risk/reward profile for entry at current levels based on technical and flow convergence?
7. What specific position management guidance applies to each scenario (no position, small loss, big loss, small profit, big profit, bagger profit)?

**Core Instructions:**

The assistant should analyze all four data dimensions (price/depth, trade microstructure, 1-week foreign-buy signals, 1-month flows) and synthesize them into a single conviction-based decision.

The assistant should output a primary recommendation (Generous Buy, Cautious Buy, or Do Not Buy) with clear percentage allocation guidance for the 10M IDR capital based on conviction strength and risk assessment.

The assistant should provide specific position management advice for each scenario: (1) no existing position—exact deployment amount; (2) small loss position—hold/add/exit guidance; (3) big loss position—capital preservation vs. recovery strategy; (4) small profit position—take partial profits or hold; (5) big profit position—trailing stop or lock-in strategy; (6) bagger profit position—exit or scale-out plan.

The assistant should justify every recommendation with explicit reference to the data provided, showing how price action, order flow, foreign-buy signals, and capital flows support the decision.

The assistant should frame all time horizons in daily trading terms (daily, 1-3 days, weekly outlook) and explicitly exclude any intraday or hourly trading considerations from the analysis.

The assistant should identify any data conflicts or weak signals and address them transparently rather than forcing consensus where it doesn't exist.

**Output Structure:**
- Primary Recommendation with conviction level and capital deployment percentage
- Data Synthesis Summary (how all four sources align or conflict)
- Position-Specific Guidance (tailored advice for each scenario listed above)
- Risk Factors and Exit Conditions
- Daily Trading Outlook (1-3 day and weekly perspective)

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
/analyze ADMR
```

**This will:**
1. First run the data initialization:
   ```bash
   python initiate.py ADMR
   ```
   - On **failure** (non-zero exit, missing session folder, or missing/corrupted core JSON files), **cancel the entire `/analyze` workflow** and return an error instead of continuing.
   - On **success**, creates: `sources/ADMR/{NEW_SESSION}/` with fresh JSON data files

2. Then run the analytics exporter:
   ```bash
   python scripts/analyze_market.py ADMR {NEW_SESSION}
   ```
   Generates: `sources/ADMR/{NEW_SESSION}/analysis-data-2025-11-11_18-16-24.json`

3. Load the analytics JSON and generate report:
   - Input: `sources/ADMR/{NEW_SESSION}/analysis-data-*.json`
   - Output: `sources/ADMR/{NEW_SESSION}/analysis-2025-11-11_18-30-45.md`

**Another example:**
```bash
/analyze BTCUSDT
```
- Run: `python initiate.py BTCUSDT` (creates new session)
- Run: `python scripts/analyze_market.py BTCUSDT {NEW_SESSION}`
- Load: `sources/BTCUSDT/{NEW_SESSION}/analysis-data-*.json`
- Output: `sources/BTCUSDT/{NEW_SESSION}/analysis-2025-11-11_14-30-45.md`

---

## Quick Reference

**Next Steps:**
1. Always start by running the data initialization script to fetch fresh data:
   ```bash
   python initiate.py {SYMBOL}
   ```
2. Identify the newly created session directory number
3. Run the analytics exporter (regenerates the summarized JSON with the new metadata):
   ```bash
   python scripts/analyze_market.py {SYMBOL} {SESSION}
   ```
4. When drafting future reports off `analysis-data-*.json`, refer to `metadata.orderbook_source` to know where the depth data originated.

**Notes:**
- The initialization script automatically creates a new session directory with the next available number.
- The exporter automatically records which source supplied order-book data (`price-feed.json` or `orderbook.json`).
- If `analysis-data-*.json` flags missing inputs in `metadata.missing_sources`, call them out in your report's risk and limitations sections.
- Regenerate analytics after any upstream data change before writing a new report.
 - If data initialization fails or required input files are not present/valid, **do not attempt to run the exporter or generate a report**—the `/analyze` workflow should be treated as failed and terminated.

---

**Begin your analysis by first initializing the data collection, then running the analytics exporter, and finally loading the structured data to generate the comprehensive market analysis report.**

