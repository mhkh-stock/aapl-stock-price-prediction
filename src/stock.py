"""My Task 2 - Stock Price Prediction (Arch Technologies, Month 1).

Stock: Apple Inc. (AAPL), daily open/high/low/close/volume data.
Source: Yahoo Finance (yfinance). I tried the Stooq free CSV first, but
it showed a browser check page and gave no data, so I used Yahoo Finance.
Range: 2020-01-02 to 2026-11-12 (1,726 trading days).

What I did: I guess the next day's closing price. I guess it through the
next-day return, then I work out the price from it. I did it this way
because the price goes up a lot over the years, and a tree model cannot
guess a price higher than any price it has seen. I tried it the direct
way first and Random Forest got an R-squared of -0.43 for this exact
reason.

I split by time: the first 80% trains the model, the last 20% tests it.
I never shuffled the data. I trained Linear Regression and Random Forest
and I checked both against a simple baseline (tomorrow's price will be
the same as today's).

My honest note: daily stock prices are very hard to guess. A model that
only matches the simple baseline is a normal result. I do not claim any
model here can predict the market.

Run it from the task folder:  python src/stock.py
It saves: data/aapl_features.csv, outputs/metrics.json, outputs/*.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1F4E79"
BLUE = "#2E75B6"
LIGHT = "#5B9BD5"
GREY = "#7F7F7F"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.edgecolor": "#404040", "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("chart:", name)


def price_metrics(name, y_true, y_pred):
    return {
        "model": name,
        "mae_usd": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse_usd": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2_price": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    # yfinance wrote a two-row header (Price / Ticker); flatten it.
    df = pd.read_csv(DATA / "aapl_raw.csv", header=[0, 1], index_col=0,
                     parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()
    print(f"rows: {len(df)}, range: {df.index[0].date()} to {df.index[-1].date()}")
    assert len(df) > 1500, "too few rows downloaded"

    # ---------- Feature engineering ----------
    df["Return"] = df["Close"].pct_change()
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df["Lag1"] = df["Close"].shift(1)
    df["Lag5"] = df["Close"].shift(5)
    df["Vol_MA7"] = df["Volume"].rolling(7).mean()
    # Target: next-day return; price is reconstructed from it.
    df["TargetReturn"] = df["Close"].pct_change().shift(-1)
    df["TargetPrice"] = df["Close"].shift(-1)
    feat = df.dropna().copy()
    feat.to_csv(DATA / "aapl_features.csv")
    print(f"feature rows after indicators: {len(feat)}")

    cols = ["Open", "High", "Low", "Close", "Volume",
            "Return", "MA7", "MA30", "Lag1", "Lag5", "Vol_MA7"]
    X = feat[cols]
    y_ret = feat["TargetReturn"]
    y_price = feat["TargetPrice"]

    # ---------- Split by time (last 20% = test, I never shuffled) ----------
    n_test = int(len(feat) * 0.20)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_ret_train, y_ret_test = y_ret.iloc[:-n_test], y_ret.iloc[-n_test:]
    y_price_test = y_price.iloc[-n_test:]
    close_test = X_test["Close"].values
    test_dates = feat.index[-n_test:]
    print(f"train {len(X_train)} rows ({X_train.index[0].date()} to "
          f"{X_train.index[-1].date()}), test {len(X_test)} rows "
          f"({test_dates[0].date()} to {test_dates[-1].date()})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------- My models (I guess the return, then I work out the price) ----------
    lr = LinearRegression()
    lr.fit(X_train_s, y_ret_train)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_ret_train)

    lr_price = close_test * (1 + lr.predict(X_test_s))
    rf_price = close_test * (1 + rf.predict(X_test))
    naive_price = close_test.copy()  # tomorrow = today

    def direction_accuracy(y_true_ret, y_pred_ret):
        return round(float(np.mean(
            np.sign(y_true_ret) == np.sign(y_pred_ret))), 4)

    models = [
        {**price_metrics("Naive baseline (tomorrow = today)", y_price_test, naive_price),
         "r2_return": 0.0, "direction_accuracy": 0.5},
        {**price_metrics("Linear Regression", y_price_test, lr_price),
         "r2_return": round(float(r2_score(y_ret_test, lr.predict(X_test_s))), 4),
         "direction_accuracy": direction_accuracy(
             y_ret_test.values, lr.predict(X_test_s))},
        {**price_metrics("Random Forest", y_price_test, rf_price),
         "r2_return": round(float(r2_score(y_ret_test, rf.predict(X_test))), 4),
         "direction_accuracy": direction_accuracy(
             y_ret_test.values, rf.predict(X_test))},
    ]

    results = {
        "dataset": {
            "symbol": "AAPL (Apple Inc.)",
            "source": "Yahoo Finance via yfinance (Stooq free CSV was blocked "
                      "by a browser-verification page at fetch time)",
            "range": f"{df.index[0].date()} to {df.index[-1].date()}",
            "trading_days": int(len(df)),
        },
        "method": {
            "target": "next-day closing price, modelled through the next-day return",
            "features": cols,
        },
        "split": {"method": "time-based, last 20% as test, no shuffling",
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "test_start": str(test_dates[0].date()),
                  "test_end": str(test_dates[-1].date())},
        "models": models,
        "feature_importance": {
            str(f): round(float(v), 4)
            for f, v in sorted(zip(cols, rf.feature_importances_),
                               key=lambda t: -t[1])
        },
        "price_summary": {
            "first_close": round(float(df["Close"].iloc[0]), 2),
            "last_close": round(float(df["Close"].iloc[-1]), 2),
            "max_close": round(float(df["Close"].max()), 2),
            "min_close": round(float(df["Close"].min()), 2),
            "total_return_pct": round(
                float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100), 2),
            "train_max_close": round(float(X_train["Close"].max()), 2),
            "test_mean_close": round(float(y_price_test.mean()), 2),
            "test_max_close": round(float(y_price_test.max()), 2),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(models, indent=2))
    print(json.dumps(results["price_summary"], indent=2))

    # ---------- Charts ----------
    # 1. Price history with the train/test split marked
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], color=NAVY, lw=1, label="Close price")
    ax.plot(df.index, df["Close"].rolling(30).mean(), color=BLUE, lw=1.2,
            label="30-day moving average")
    ax.axvline(test_dates[0], color="red", ls="--", lw=1.2, label="Test period starts")
    ax.set_ylabel("Price (USD)")
    ax.set_title("AAPL Closing Price, 2020–2026 (with 30-day moving average)")
    ax.legend()
    save(fig, "price_history.png")

    # 2. Actual vs predicted (test period, Random Forest)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_dates, y_price_test.values, color=NAVY, lw=1.2, label="Actual close")
    ax.plot(test_dates, rf_price, color=BLUE, lw=1.2,
            label="Random Forest prediction")
    ax.set_ylabel("Price (USD)")
    ax.set_title("AAPL — Actual vs Predicted Close (test period)")
    ax.legend()
    save(fig, "actual_vs_predicted.png")

    # 3. Residuals (Random Forest, test period)
    resid = y_price_test.values - rf_price
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(resid, bins=30, color=BLUE, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.set_xlabel("Prediction error (USD): actual − predicted")
    ax.set_ylabel("Days")
    ax.set_title("Prediction Errors — Random Forest (test period)")
    save(fig, "residuals.png")

    print("Task 2 done. Outputs in", OUT)


if __name__ == "__main__":
    main()
