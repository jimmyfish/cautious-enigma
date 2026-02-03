import json
from datetime import datetime

import pandas as pd
import yfinance as yf

# 1. Load your models/groups.json
try:
    with open("models/groups.json", "r") as f:
        groups = json.load(f)
except FileNotFoundError:
    print("Error: models/groups.json not found.")
    exit()

# 2. Map and Batch Download
ticker_map = {f"{t}.JK": t for cat in groups.values() for t in cat}
symbols = list(ticker_map.keys())

print(f"Batch fetching {len(symbols)} tickers from 2015...")
# Fetching 'Low' and 'Close' for verification
df = yf.download(symbols, start="2015-01-01", progress=False)

results = []

# 3. Analyze the MultiIndex DataFrame
# Structure is df['Low']['TICKER.JK']
for symbol, original_ticker in ticker_map.items():
    try:
        # Get the series and drop NaNs for accurate date indexing
        low_series = df["Low"][symbol].dropna()

        if low_series.empty:
            continue

        min_price = float(low_series.min())

        if min_price <= 50:
            # Get the first date it hit that minimum price
            impact_date = low_series.idxmin().strftime("%Y-%m-%d")
            results.append(
                {
                    "ticker": original_ticker,
                    "status": "HIT <= 50",
                    "min": min_price,
                    "date": impact_date,
                }
            )
        else:
            results.append(
                {
                    "ticker": original_ticker,
                    "status": "ABOVE 50",
                    "min": min_price,
                    "date": "N/A",
                }
            )

    except KeyError:
        print(f"Could not process {symbol}")

# 4. Clean Output Table
print("\n" + "=" * 55)
print(f"{'TICKER':<10} | {'STATUS':<10} | {'MIN PRICE':<10} | {'DATE REACHED'}")
print("-" * 55)
for res in results:
    print(
        f"{res['ticker']:<10} | {res['status']:<10} | {res['min']:<10.2f} | {res['date']}"
    )
print("=" * 55)
