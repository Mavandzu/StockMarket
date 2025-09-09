#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# -----------------------------
# User Settings
# -----------------------------
tickers = ["AAPL", "MSFT", "GOOGL"]  # Replace/add any tickers
start = "2021-01-01"
end = "2026-01-01"

# -----------------------------
# 1️⃣ Download Stock Data
# -----------------------------
data = {}
failed_tickers = []

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if df.empty:
            failed_tickers.append(ticker)
        else:
            df.index = pd.to_datetime(df.index)  # Ensure datetime index
            data[ticker] = df
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"Failed {ticker}: {e}")

if failed_tickers:
    print("Failed tickers:", failed_tickers)

# -----------------------------
# 2️⃣ Prepare Normalized & Returns CSV
# -----------------------------
if data:
    # Normalize close prices
    normalized = pd.concat({t: df["Close"] for t, df in data.items()}, axis=1)
    normalized = normalized / normalized.iloc[0] * 100
    normalized.to_csv("normalized_prices.csv", index_label="Date")

    # Daily returns
    returns = pd.concat({t: df["Close"].pct_change() for t, df in data.items()}, axis=1)
    returns.to_csv("daily_returns.csv", index_label="Date")

    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns.to_csv("cumulative_returns.csv", index_label="Date")

    # Risk & Return Metrics
    mean_daily_return = returns.mean()
    volatility = returns.std()
    annual_return = mean_daily_return * 252
    annual_volatility = volatility * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility

    risk_return = pd.DataFrame({
        "Ticker": tickers,
        "Annual Return (%)": (annual_return * 100).values,
        "Annual Volatility (%)": (annual_volatility * 100).values,
        "Sharpe Ratio": sharpe_ratio.values
    })
    risk_return.to_csv("risk_return_metrics.csv", index=False)

# -----------------------------
# 3️⃣ Download Macro Indicators
# -----------------------------
fred_series = {
    "CPI": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "Interest Rate": "FEDFUNDS"
}

macro_data = {}
for name, code in fred_series.items():
    try:
        df = web.DataReader(code, "fred", start, end)
        df.rename(columns={code: name}, inplace=True)
        df.index = pd.to_datetime(df.index)
        macro_data[name] = df
    except Exception as e:
        print(f"Failed to fetch {name}: {e}")

if macro_data:
    macro_df = pd.concat(macro_data.values(), axis=1).fillna(method="ffill").dropna()
    macro_df.to_csv("macro_indicators.csv", index_label="Date")

# -----------------------------
# 4️⃣ Monthly Resample for Forecasting
# -----------------------------
stock_monthly = pd.concat({t: df["Close"].resample("M").mean() for t, df in data.items()}, axis=1)
macro_monthly = macro_df.resample("M").mean()
combined_monthly = pd.concat([stock_monthly, macro_monthly], axis=1).dropna()
combined_monthly.to_csv("combined_monthly.csv", index_label="Date")

# -----------------------------
# 5️⃣ Machine Learning Forecasts
# -----------------------------
ml_forecasts = {}

for ticker in stock_monthly.columns:
    series = stock_monthly[ticker].dropna()
    if len(series) < 24:
        continue

    df_feat = pd.DataFrame({
        "y": series,
        "lag1": series.shift(1),
        "lag2": series.shift(2),
        "lag3": series.shift(3)
    }).dropna()

    X = df_feat[["lag1", "lag2", "lag3"]]
    y = df_feat["y"]

    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        actuals.extend(y_test)

    model.fit(X, y)
    forecast = model.predict(X.iloc[[-1]])[0]

    ml_forecasts[ticker] = {
        "MAE": mean_absolute_error(actuals, preds),
        "MAPE": mean_absolute_percentage_error(actuals, preds),
        "R2": r2_score(actuals, preds),
        "Next Forecast": forecast
    }

# Save ML forecasts
forecast_df = pd.DataFrame(ml_forecasts).T.reset_index().rename(columns={"index": "Ticker"})
forecast_df.to_csv("ml_forecasts.csv", index=False)

print("✅ All CSVs exported:")
print("- normalized_prices.csv")
print("- daily_returns.csv")
print("- cumulative_returns.csv")
print("- risk_return_metrics.csv")
print("- macro_indicators.csv")
print("- combined_monthly.csv")
print("- ml_forecasts.csv")

