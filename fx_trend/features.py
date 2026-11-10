"""特徵工程：把價格與總經資料轉成與價格尺度無關的特徵，供模型使用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

FEATURE_LABELS = {
    "r1": "1 日動能",
    "r5": "5 日動能",
    "r20": "20 日動能",
    "rsi": "RSI(14)",
    "macd": "MACD 柱狀",
    "bb": "布林通道位置",
    "dist20": "距 20 日均線",
    "trend": "均線多空排列",
    "stoch": "KD 隨機指標",
    "volr": "波動度變化",
    "gold": "國際黃金(5日)",
    "twii": "台股(5日)",
    "dxy": "美元指數(5日)",
    "us10y": "美債殖利率變化(5日)",
}
FEATURES = list(FEATURE_LABELS)


def build_features(df: pd.DataFrame, macro: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    c = df["Close"]
    r1 = np.log(c).diff()
    vol20 = r1.rolling(20).std()
    mid, up, lo = ind.bollinger(c, 20)
    _, _, hist = ind.macd(c)
    atr14 = ind.atr(df)

    f = pd.DataFrame(index=df.index)
    f["r1"] = r1 / vol20
    f["r5"] = np.log(c / c.shift(5)) / (vol20 * np.sqrt(5))
    f["r20"] = np.log(c / c.shift(20)) / (vol20 * np.sqrt(20))
    f["rsi"] = (ind.rsi(c) - 50) / 50
    f["macd"] = hist / atr14
    f["bb"] = (c - mid) / ((up - lo) / 2)
    f["dist20"] = (c / mid - 1) / (vol20 * np.sqrt(10))
    f["trend"] = (mid / ind.sma(c, 60) - 1) / (vol20 * np.sqrt(20))
    f["stoch"] = (ind.stochastic(df) - 50) / 50
    f["volr"] = vol20 / r1.rolling(100).std() - 1

    macro = macro or {}
    for name in ("gold", "twii", "dxy", "us10y"):
        s = macro.get(name)
        if s is None or len(s) < 30:
            f[name] = 0.0
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        x = s.diff(5) if name == "us10y" else np.log(s).diff(5)
        # 落後 1 天：避免用到與外匯收盤時間不一致的未來資訊
        f[name] = x.reindex(f.index.union(x.index)).ffill().shift(1).reindex(f.index).fillna(0.0)

    return f[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
