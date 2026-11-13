# Task 2 — Stock Price Prediction (AAPL)

**Name:** Muhammad Hashir
**Internship:** Data Science, Arch Technologies (Month 1)
**Stock:** Apple Inc. (AAPL), daily open/high/low/close/volume data
**Source:** Yahoo Finance (with `yfinance`). I tried the Stooq free CSV
first, but it showed a browser check page and gave no data, so I used
Yahoo Finance instead. Both are free public sources.
**Range:** 2020-01-02 to 2026-11-12 — 1,726 trading days

## What I did

I built a model to guess the next day's closing price of Apple stock from
its past prices and volume. I made new features from the data, trained two
models and checked them against a simple baseline.

## How I did it

1. **My target.** I guess the next-day closing price. I did it through the
   next-day return: price tomorrow = price today × (1 + return). I did it
   this way because the price goes up a lot over the years. Apple went from
   $75.09 to $354.88. A tree model that learns from past prices cannot guess
   a price higher than any price it has seen. I tried it the direct way
   first and Random Forest got an R² of −0.43 for this exact reason. The
   test period went up to $355.63, but the training data never went above
   $259.02.
2. **My features.** Open, High, Low, Close, Volume, daily return, 7-day and
   30-day moving averages of the close, the close 1 day and 5 days back,
   and a 7-day average of volume.
3. **My split.** I split by time, which is the only fair way for time
   data. The first 80% (1,357 rows) is for training. The last 20% (339
   rows, 2025-07-10 to 2026-11-11) is for testing. I did not shuffle, so
   no future data leaks into training.
4. **My models.** Linear Regression (with scaled features) and Random Forest
   (300 trees, seed 42). I also used a simple baseline: tomorrow's close
   will be the same as today's close.

## My results (test set, 339 trading days)

| Model | MAE (USD) | RMSE (USD) | R² (price) | R² (return) | Direction |
|---|---|---|---|---|---|
| Simple baseline (tomorrow = today) | **3.12** | **4.42** | 0.986 | 0.00 | 50.0% |
| Linear Regression | 3.59 | 5.11 | 0.981 | −0.31 | 50.4% |
| Random Forest | 4.82 | 6.08 | 0.974 | −0.87 | 48.7% |

## My honest conclusion

**Neither model beat the simple baseline.** The simple guess — tomorrow's
price will be the same as today's — had the lowest error (MAE $3.12,
RMSE $4.42). Linear Regression was close (MAE $3.59). Random Forest was
worse (MAE $4.82). The R² on returns is below zero for both models. That
means they guess daily moves worse than just guessing the average return.
And they got the up/down direction right only about half the time, which
is the same as a coin toss.

The high R² on price (0.97–0.99) looks good, but I will not use it alone.
It is high mostly because today's price is close to tomorrow's price. It
does not mean the model can predict the market. This is the result I
expected for one-day prices. Daily price moves are very hard to guess from
past prices alone. I do not use these models to trade, and I do not claim
they can predict the market.

## How to run it

```bash
pip install -r requirements.txt
python src/stock.py
```

## Folder structure

```
task2-stock/
├── README.md
├── requirements.txt
├── src/
│   └── stock.py            # my full code, runs end to end
├── data/
│   ├── aapl_raw.csv        # Yahoo Finance data, 2020-01-02 to 2026-11-12
│   └── aapl_features.csv   # features I made + targets
├── outputs/
│   ├── metrics.json        # all scores and numbers
│   ├── price_history.png
│   ├── actual_vs_predicted.png
│   └── residuals.png
└── report/
    ├── MuhammadHashir_DataScience_Month1.docx
    └── MuhammadHashir_DataScience_Month1.pdf
```

## Tools I used

Python 3 · pandas · scikit-learn · matplotlib · NumPy · yfinance
