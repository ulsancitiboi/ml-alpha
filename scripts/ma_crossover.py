"""
Baseline: 50/200 moving-average crossover on SPY (reproduces the June notebook).
Run from the repo root:  python scripts/ma_crossover.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
from data import get_prices
from backtest import backtest, compare

TICKER, SHORT, LONG = "SPY", 50, 200
SPLIT = "2017-01-01"            # out-of-sample starts here

price = get_prices(TICKER, start="2005-01-01", end="2024-01-01")[TICKER]
signal = (price.rolling(SHORT).mean() > price.rolling(LONG).mean()).astype(int)

strat = backtest(signal, price, cost=0.001)
hold = backtest(signal * 0 + 1, price, cost=0.0)

print("Full period")
print(compare({"MA 50/200": strat, "Buy & hold": hold}).to_string(index=False))
print(f"\nOut-of-sample (from {SPLIT})")
print(compare({"MA 50/200": strat[SPLIT:], "Buy & hold": hold[SPLIT:]}).to_string(index=False))

Path("results").mkdir(exist_ok=True)
(1 + strat).cumprod().plot(label="MA 50/200 (net of costs)", figsize=(10, 5))
(1 + hold).cumprod().plot(label="Buy & hold")
plt.axvline(SPLIT, ls="--", c="gray")
plt.legend(); plt.title(f"{TICKER}: MA crossover vs buy & hold")
plt.tight_layout(); plt.savefig("results/ma_crossover.png", dpi=120)
print("\nSaved results/ma_crossover.png")
