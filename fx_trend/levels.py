"""支撐/壓力、樞軸點、趨勢線與區間預測（皆由實際價格計算）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def swing_points(df: pd.DataFrame, w: int = 5):
    """回傳已確認的波段高/低點 [(位置, 價格)]；最後 w 根尚未確認，不納入。"""
    hi, lo = df["High"], df["Low"]
    is_hi = hi == hi.rolling(2 * w + 1, center=True).max()
    is_lo = lo == lo.rolling(2 * w + 1, center=True).min()
    highs = [(i, hi.iloc[i]) for i in np.flatnonzero(is_hi.values)]
    lows = [(i, lo.iloc[i]) for i in np.flatnonzero(is_lo.values)]
    return highs, lows


def _cluster(levels: list[float], tol: float) -> list[tuple[float, int]]:
    out: list[list[float]] = []
    for v in sorted(levels):
        if out and v - np.mean(out[-1]) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [(float(np.mean(g)), len(g)) for g in out]


def compute_levels(df: pd.DataFrame, lookback: int = 250) -> dict:
    d = df.iloc[-lookback:].reset_index(drop=True)
    close = float(d["Close"].iloc[-1])
    a = float(ind.atr(df).iloc[-1])
    h, l, c = (float(d[k].iloc[-1]) for k in ("High", "Low", "Close"))
    p = (h + l + c) / 3
    pivot = {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l)}

    highs, lows = swing_points(d)
    tol = 0.6 * a
    res_c = [x for x in _cluster([v for _, v in highs], tol) if x[0] > close + 0.2 * a]
    sup_c = [x for x in _cluster([v for _, v in lows], tol) if x[0] < close - 0.2 * a]

    def pick(cands, above: bool):
        if not cands:
            return None
        near = sorted(cands, key=lambda x: abs(x[0] - close))
        strong = [x for x in near[:3] if x[1] >= 2]
        return (strong or near)[0]

    r, s = pick(res_c, True), pick(sup_c, False)
    resistance = r[0] if r else close + 2 * a
    support = s[0] if s else close - 2 * a
    return {
        "close": close, "atr": a, "pivot": pivot,
        "resistance": float(resistance), "resistance_touches": r[1] if r else 0,
        "support": float(support), "support_touches": s[1] if s else 0,
    }


def trend_and_forecast(df: pd.DataFrame, horizon: int, win: int = 60) -> dict:
    c = np.log(df["Close"].iloc[-win:].values)
    x = np.arange(len(c))
    slope, icpt = np.polyfit(x, c, 1)
    fit = slope * x + icpt
    resid = c - fit
    ss_res, ss_tot = (resid**2).sum(), ((c - c.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sigma_d = float(np.log(df["Close"]).diff().iloc[-20:].std())
    last = float(df["Close"].iloc[-1])
    s = sigma_d * np.sqrt(horizon)
    return {
        "slope_pct_per_day": float(np.expm1(slope) * 100), "r2": float(r2),
        "channel_sigma": float(resid.std()),
        "channel_upper": float(np.exp(fit[-1] + 2 * resid.std())),
        "channel_lower": float(np.exp(fit[-1] - 2 * resid.std())),
        "range68": [last * float(np.exp(-s)), last * float(np.exp(s))],
        "range95": [last * float(np.exp(-1.96 * s)), last * float(np.exp(1.96 * s))],
        "horizon": horizon,
    }
