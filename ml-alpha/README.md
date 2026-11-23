# ml-alpha

Predicting cross-sectional stock returns with machine learning, evaluated with an
honest backtest (walk-forward validation, no look-ahead, transaction costs).

## Structure
```
src/backtest.py      vectorized backtester + metrics (CAGR, Sharpe, max drawdown)
src/data.py          price download with local cache
scripts/             experiments (baseline: MA crossover)
results/             charts and tables
```

## Quick start
```
pip install -r requirements.txt
python scripts/ma_crossover.py
```

## Roadmap
- [x] Week 1 — data pipeline, backtester, baseline strategy
- [ ] Week 2 — features (momentum, volatility, volume) + Ridge baseline
- [ ] Week 3 — LightGBM with walk-forward validation
- [ ] Week 4 — long-short portfolio, IC, cost analysis, write-up
