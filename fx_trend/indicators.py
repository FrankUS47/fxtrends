"""技術指標（純 pandas，無外部依賴）。"""
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(dn != 0, 100.0).where(up.notna(), np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stochastic(df: pd.DataFrame, n: int = 14) -> pd.Series:
    ll, hh = df["Low"].rolling(n).min(), df["High"].rolling(n).max()
    return 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
