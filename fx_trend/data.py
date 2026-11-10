"""資料層：下載 / 快取 / 模擬資料。

正式環境用 yfinance；每次只補抓缺的天數並寫回 data/raw/*.csv。
抓取失敗時退回快取；連快取都沒有才丟錯。
--demo 模式使用模擬資料，僅供測試，絕不可作為真實預測。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

OHLC = ["Open", "High", "Low", "Close"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLC if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.dropna(subset=["Close"])
    for c in OHLC:                      # 少數標的缺 O/H/L，以 Close 補
        if c not in df.columns:
            df[c] = df["Close"]
        df[c] = df[c].fillna(df["Close"])
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _download(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, start=start, interval="1d", auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"{ticker}: 無資料")
    return _clean(raw)


def _cached_fetch(name: str, ticker: str, fetch: bool) -> pd.DataFrame:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RAW_DIR / f"{name}.csv"
    cached = None
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
    if fetch:
        try:
            start = C.HISTORY_START
            if cached is not None and len(cached):
                start = (cached.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            new = _download(ticker, start)
            cached = new if cached is None else _clean(pd.concat([cached, new]))
            cached.to_csv(path)
        except Exception as e:  # noqa: BLE001
            if cached is None:
                raise
            print(f"[warn] {name} 更新失敗，改用快取：{e}")
    if cached is None:
        raise RuntimeError(f"{name}: 沒有快取且未允許下載")
    return cached


def load_pair(key: str, fetch: bool = True, demo: bool = False) -> pd.DataFrame:
    if demo:
        return synthetic_ohlc(key)
    return _cached_fetch(key, C.PAIRS[key]["ticker"], fetch)


def load_macro(fetch: bool = True, demo: bool = False) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name, ticker in C.MACRO.items():
        try:
            if demo:
                out[name] = synthetic_ohlc("MACRO_" + name, base=100.0)["Close"]
            else:
                out[name] = _cached_fetch("macro_" + name, ticker, fetch)["Close"]
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 總經因子 {name} 不可用（以 0 補）：{e}")
    return out


def synthetic_ohlc(key: str, n: int = 1600, base: float | None = None) -> pd.DataFrame:
    """帶有緩慢趨勢切換與波動叢聚的模擬日線（僅測試用）。"""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(key))   # 固定種子，結果可重現
    rng = np.random.default_rng(seed)
    base = base or {"USDTWD": 32.0, "EURUSD": 1.1, "USDJPY": 150.0,
                    "GBPUSD": 1.3, "USDCHF": 0.9}.get(key, 1.0)
    regime = np.repeat(rng.normal(0, 0.0004, n // 60 + 1), 60)[:n]
    vol = 0.004 * np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.2)
    r = regime + rng.normal(0, 1, n) * vol
    close = base * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    rng_hl = np.abs(rng.normal(0, 1, n)) * vol * close * 0.6
    high = np.maximum(open_, close) + rng_hl
    low = np.minimum(open_, close) - rng_hl
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)
