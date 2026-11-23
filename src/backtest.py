"""
Vectorized backtester.

Works for a single asset (Series) or many assets (DataFrame of weights).
Signals/weights decided at day t are traded at the close of t and earn
the return of day t+1 (enforced by shift(1) -> no look-ahead).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def backtest(weights, prices, cost=0.001):
    """
    weights : Series or DataFrame (dates x tickers) of target weights.
              1 = fully long, -1 = fully short, 0 = flat.
    prices  : Series or DataFrame of prices aligned with weights.
    cost    : cost per unit of turnover (0.001 = 0.1%).

    Returns daily net returns (Series).
    """
    if isinstance(weights, pd.Series):
        weights = weights.to_frame()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    weights = weights.reindex_like(prices).fillna(0)
    weights.columns = prices.columns

    rets = prices.pct_change().fillna(0)
    position = weights.shift(1).fillna(0)          # trade on next day
    turnover = position.diff().abs().fillna(position.abs())
    gross = (position * rets).sum(axis=1)
    net = gross - turnover.sum(axis=1) * cost
    return net


def cagr(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    growth = (1 + r).prod()
    return growth ** (TRADING_DAYS / len(r)) - 1


def sharpe(r):
    r = r.dropna()
    sd = r.std()
    return np.sqrt(TRADING_DAYS) * r.mean() / sd if sd > 0 else np.nan


def max_drawdown(r):
    equity = (1 + r.fillna(0)).cumprod()
    return (equity / equity.cummax() - 1).min()


def summary(r, name="strategy"):
    return pd.Series({
        "total_return": (1 + r.fillna(0)).prod() - 1,
        "cagr": cagr(r),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "days": int(r.dropna().shape[0]),
    }, name=name)


def compare(results: dict):
    """results = {"name": daily_returns, ...} -> formatted table"""
    table = pd.DataFrame([summary(r, k) for k, r in results.items()])
    fmt = {"total_return": "{:.1%}", "cagr": "{:.1%}",
           "sharpe": "{:.2f}", "max_drawdown": "{:.1%}", "days": "{:.0f}"}
    out = table.copy()
    for c, f in fmt.items():
        out[c] = table[c].map(f.format)
    return out
