"""Price data loading with a local cache (so you don't re-download every run)."""
from pathlib import Path
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"


def get_prices(tickers, start="2005-01-01", end=None, refresh=False):
    """
    Adjusted close prices, dates x tickers.
    tickers: str or list, e.g. "SPY" or ["AAPL", "MSFT"]
    """
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]
    CACHE.mkdir(exist_ok=True)
    key = f"{'_'.join(sorted(tickers))[:80]}_{start}_{end}.parquet"
    path = CACHE / key
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df = df.dropna(how="all")
    df.to_parquet(path)
    return df
