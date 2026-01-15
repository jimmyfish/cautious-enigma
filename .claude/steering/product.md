# Product Overview

## Description

**HelpMe** is a stock market analysis toolkit for Indonesian Stock Exchange (IDX) stocks. It provides automated data collection, technical analysis, and short-term trading signal generation focused on 2-3 day trading opportunities.

## Core Features

- **Stock Screener Integration**: Fetch filtered stock lists from Stockbit screener templates based on custom criteria
- **Market Data Collection**: Automated retrieval of price feeds, order books, running trades, and market detector data from Stockbit API
- **Foreign Flow Analysis**: Track foreign vs domestic investor activity and net foreign fund flows
- **Technical Analytics Pipeline**: Generate structured analytics including price trends, volatility metrics, support/resistance levels, and liquidity scores
- **Bulk Analysis Reports**: Create comprehensive markdown reports analyzing multiple stocks simultaneously with trading recommendations
- **Bandar Detection**: Identify accumulation/distribution signals from market maker (bandar) activity patterns
- **Price Forecasting**: Neural network-based daily price predictions using TFT, NBEATS, NHITS ensemble with probabilistic confidence intervals
- **Intraday Forecasting**: Session-level forecasts for next trading day using tick-level data aggregated into OHLCV bars
- **Yahoo Finance Forecasting**: Forecast any global stock using Yahoo Finance data with technical indicators (RSI, MACD, Bollinger Bands)
- **Cross-Validation**: Time-series cross-validation to evaluate model accuracy before deployment (MAE, RMSE, MAPE, direction accuracy)
- **Group Training**: Train models on related stocks (18+ IDX sectors including banking, energy, technology, property, etc.) to learn common market patterns
- **Incremental Learning**: Model persistence with warm-start training - saves checkpoints and fine-tunes on new data instead of retraining from scratch
- **Telegram Notifications**: Optional Telegram bot integration for automated alerts

## Target Use Case

- **Short-term traders** seeking 2-3 day trading opportunities in IDX stocks
- **Foreign flow watchers** monitoring institutional investor movements
- **Screener-based workflows** where users want to analyze multiple stocks matching specific criteria simultaneously
- **Quantitative traders** wanting ML-based price forecasts with confidence intervals
- **Sector analysts** analyzing stock groups (banking, mining, etc.) with shared models

## Key Value Proposition

- Automates the tedious process of collecting and analyzing market data from multiple sources
- Provides actionable insights by combining price action, order flow, foreign activity, and market maker signals
- Generates ready-to-read reports with clear trading recommendations and risk assessments
- Saves time by processing entire screener results in a single workflow
- **Probabilistic forecasts** with 80%/90% confidence intervals for risk management
- **Incremental training** - models improve over time without full retraining
- **Robust to market anomalies** - HuberLoss and StudentT distribution handle outliers, gaps, and extreme moves
- **Sector-aware forecasting** - group training captures common patterns across related stocks
