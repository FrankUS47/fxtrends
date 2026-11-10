"""型態辨識：由實際 K 線偵測（取代原版的隨機文字）。

K 線組合：吞噬、晨星/夜星、錘子/流星
波段型態：雙底/雙頂（含頸線突破判斷）、收斂三角/上升三角/下降三角
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .levels import swing_points


def _candles(df: pd.DataFrame, a: float) -> list[dict]:
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    out = []
    body = lambda i: abs(c[i] - o[i])  # noqa: E731
    bull = lambda i: c[i] > o[i]       # noqa: E731
    bear = lambda i: c[i] < o[i]       # noqa: E731
    if len(df) < 4:
        return out
    # 吞噬
    if bear(-2) and bull(-1) and o[-1] <= c[-2] and c[-1] >= o[-2] and body(-1) > body(-2):
        out.append(dict(name="多頭吞噬", bias="bull", kind="reversal",
                        desc="陽線完全吞噬前一根陰線，短線買盤轉強。"))
    if bull(-2) and bear(-1) and o[-1] >= c[-2] and c[-1] <= o[-2] and body(-1) > body(-2):
        out.append(dict(name="空頭吞噬", bias="bear", kind="reversal",
                        desc="陰線完全吞噬前一根陽線，短線賣壓轉強。"))
    # 晨星 / 夜星
    small = body(-2) < 0.35 * max(body(-3), 1e-12)
    if bear(-3) and body(-3) > 0.5 * a and small and bull(-1) and c[-1] > (o[-3] + c[-3]) / 2:
        out.append(dict(name="晨星", bias="bull", kind="reversal",
                        desc="長陰、小實體、長陽三根組合，跌勢可能力竭。"))
    if bull(-3) and body(-3) > 0.5 * a and small and bear(-1) and c[-1] < (o[-3] + c[-3]) / 2:
        out.append(dict(name="夜星", bias="bear", kind="reversal",
                        desc="長陽、小實體、長陰三根組合，漲勢可能力竭。"))
    # 錘子 / 流星（需出現在近 10 日高低點附近）
    rng = h[-1] - l[-1]
    if rng > 0 and body(-1) < 0.35 * rng:
        lower, upper = min(o[-1], c[-1]) - l[-1], h[-1] - max(o[-1], c[-1])
        if lower > 2 * body(-1) and upper < 0.25 * rng and l[-1] <= l[-10:].min():
            out.append(dict(name="錘子線", bias="bull", kind="reversal",
                            desc="低檔長下影線，下方承接力道明顯。"))
        if upper > 2 * body(-1) and lower < 0.25 * rng and h[-1] >= h[-10:].max():
            out.append(dict(name="流星線", bias="bear", kind="reversal",
                            desc="高檔長上影線，上方賣壓沉重。"))
    return out


def _swing_patterns(df: pd.DataFrame, a: float) -> list[dict]:
    d = df.iloc[-150:].reset_index(drop=True)
    highs, lows = swing_points(d, w=4)
    close = float(d["Close"].iloc[-1])
    out = []
    tol = 1.2 * a

    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        mids = [v for i, v in highs if i1 < i < i2]
        if mids and abs(p1 - p2) <= tol and i2 - i1 >= 8 and len(d) - i2 <= 40:
            neck = max(mids)
            if neck - max(p1, p2) >= 2 * a:
                done = close > neck
                out.append(dict(name="雙底(W底)", bias="bull", kind="reversal",
                                desc=f"兩低點 {p1:.4f}/{p2:.4f}，頸線 {neck:.4f}，"
                                     + ("已收盤突破頸線，反轉成立。" if done else "尚未突破頸線，需等待確認。")))
    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        mids = [v for i, v in lows if i1 < i < i2]
        if mids and abs(p1 - p2) <= tol and i2 - i1 >= 8 and len(d) - i2 <= 40:
            neck = min(mids)
            if min(p1, p2) - neck >= 2 * a:
                done = close < neck
                out.append(dict(name="雙頂(M頭)", bias="bear", kind="reversal",
                                desc=f"兩高點 {p1:.4f}/{p2:.4f}，頸線 {neck:.4f}，"
                                     + ("已收盤跌破頸線，反轉成立。" if done else "尚未跌破頸線，需等待確認。")))

    # 三角形：用最近 3 個高/低點的斜率
    if len(highs) >= 3 and len(lows) >= 3:
        hx, hy = zip(*highs[-3:])
        lx, ly = zip(*lows[-3:])
        span = max(hx[-1], lx[-1]) - min(hx[0], lx[0])
        if 10 <= span <= 120 and min(hx[0], lx[0]) > len(d) - 130:
            sh, sl = np.polyfit(hx, hy, 1)[0] * span, np.polyfit(lx, ly, 1)[0] * span
            flat = 0.8 * a
            if sh < -flat and sl > flat:
                out.append(dict(name="對稱收斂三角形", bias="neutral", kind="continuation",
                                desc="高點降低、低點墊高，波動收斂，突破方向即為後續趨勢。"))
            elif abs(sh) <= flat and sl > flat:
                out.append(dict(name="上升三角形", bias="bull", kind="continuation",
                                desc="壓力水平、低點墊高，偏多整理，放量突破壓力有利續漲。"))
            elif sh < -flat and abs(sl) <= flat:
                out.append(dict(name="下降三角形", bias="bear", kind="continuation",
                                desc="支撐水平、高點降低，偏空整理，跌破支撐有利續跌。"))
    return out


def detect_patterns(df: pd.DataFrame) -> list[dict]:
    a = float(ind.atr(df).iloc[-1])
    found = _swing_patterns(df, a) + _candles(df.iloc[-12:], a)
    if not found:
        found = [dict(name="無明確型態", bias="neutral", kind="none",
                      desc="目前未偵測到符合條件的反轉或整理型態。")]
    return found
