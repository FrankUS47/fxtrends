"""單一貨幣對的完整分析流程（資料 → 特徵 → 調參 → 預測 → 紀錄）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import indicators as ind
from .features import FEATURE_LABELS, FEATURES, build_features
from .learner import (ParamStore, PredictionLog, degraded, evaluate,
                      forward_returns, signals, tune, walk_forward)
from .levels import compute_levels, trend_and_forecast
from .model import LogitModel
from .patterns import detect_patterns


def candle_pack(df: pd.DataFrame, keep: int) -> dict:
    c = df["Close"]
    mid, up, lo = ind.bollinger(c, 20)
    t = pd.DataFrame({"o": df["Open"], "h": df["High"], "l": df["Low"], "c": c,
                      "s20": ind.sma(c, 20), "s60": ind.sma(c, 60), "bu": up, "bl": lo}).tail(keep)
    return {"x": t.index.strftime("%Y-%m-%d").tolist(),
            **{k: [None if pd.isna(v) else round(float(v), 6) for v in t[k]] for k in t.columns}}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()


def action_from(p: float, margin: float) -> tuple[str, str]:
    if p >= 0.5 + margin:
        return "BUY", "建議買進（模型偏多）"
    if p <= 0.5 - margin:
        return "SELL", "建議賣出（模型偏空）"
    return "HOLD", "觀望（無明顯優勢）"


def analyze_pair(key: str, df: pd.DataFrame, macro: dict, store: ParamStore,
                 log: PredictionLog, force_retune: bool = False) -> dict:
    meta = C.PAIRS[key]
    X = build_features(df, macro)
    if len(X) < C.DEFAULT_PARAMS["min_train"] + 150:
        raise RuntimeError(f"{key}: 資料太少（{len(X)} 列可用），至少需要約 {C.DEFAULT_PARAMS['min_train'] + 150}")
    close = df["Close"]

    # 1) 先對答案：把已到期的歷史預測結算，檢查實盤表現
    log.resolve(key, close)
    live = log.live_stats(key)

    # 2) 調參：首次、每週、手動、或實盤劣化
    entry = store.get(key)
    why = None
    if force_retune: why = "手動要求"
    elif entry is None: why = "首次執行"
    elif store.due(key): why = f"超過 {C.RETUNE_EVERY_DAYS} 天未調參"
    elif degraded(live): why = f"實盤近 {live['n_recent']} 筆命中率 {live['hit_recent']:.0%} 低於門檻"
    if why:
        params, rep = tune(X, close, (entry or {}).get("params"))
        rep["reason"] = why
        store.set(key, params, rep)
        print(f"[{key}] 重新調參（{why}）→ h={params['horizon']} l2={params['l2']} margin={params['buy_margin']}")
    else:
        params = entry["params"]
    p = {**C.DEFAULT_PARAMS, **params}
    h, margin = p["horizon"], p["buy_margin"]

    # 3) 樣本外評估（調參區 70% vs holdout 30%）
    fwd = forward_returns(close, h)
    oos = walk_forward(X, fwd, p)
    valid = oos.dropna()
    cut = valid.index[int(len(valid) * 0.7)] if len(valid) else None
    stats = {
        "fit": evaluate(valid[valid.index < cut], fwd, margin, p["cost_bps"], h) if cut is not None else None,
        "holdout": evaluate(valid[valid.index >= cut], fwd, margin, p["cost_bps"], h) if cut is not None else None,
        "live": live,
    }

    # 4) 以全部已揭曉標籤訓練最終模型，預測最新一根
    lab = fwd.reindex(X.index).notna().values
    model = LogitModel(p["l2"], p["lr"], p["epochs"]).fit(X.values[lab], (fwd.reindex(X.index).values[lab] > 0).astype(float))
    x_last = X.values[-1]
    prob = float(model.predict_proba(x_last)[0])
    action, action_text = action_from(prob, margin)
    contrib = model.contributions(x_last)
    factors = sorted(({"name": FEATURE_LABELS[f], "value": float(v)} for f, v in zip(FEATURES, contrib)),
                     key=lambda d: -abs(d["value"]))[:6]

    # 5) 價位、型態、趨勢
    levels = compute_levels(df)
    patterns = detect_patterns(df)
    trend = trend_and_forecast(df, h)

    # 6) 寫入今日預測（日期以最新一根 K 線為準）
    bar_date = df.index[-1].strftime("%Y-%m-%d")
    log.upsert(key, bar_date, h, prob, action, float(close.iloc[-1]))

    # 圖表資料
    sig = signals(valid, margin).tail(250)
    daily = candle_pack(df, 250)
    cl = close.reindex(sig.index)
    markers = {"buy": [{"x": d.strftime("%Y-%m-%d"), "y": float(df.Low.loc[d])} for d in sig.index[sig == 1]],
               "sell": [{"x": d.strftime("%Y-%m-%d"), "y": float(df.High.loc[d])} for d in sig.index[sig == -1]]}
    prev = float(close.iloc[-2])
    return {
        "key": key, "label": meta["label"], "tv": meta["tv"], "decimals": meta["decimals"],
        "price": float(close.iloc[-1]), "change_pct": float(close.iloc[-1] / prev - 1) * 100,
        "bar_date": bar_date, "proba": prob, "score": round(prob * 100, 1),
        "action": action, "action_text": action_text, "horizon": h, "margin": margin,
        "params": params, "factors": factors, "levels": levels, "patterns": patterns,
        "trend": trend, "stats": stats, "log": log.recent(key),
        "tuned_at": store.get(key)["tuned_at"], "tune_reason": store.get(key)["report"].get("reason"),
        "candles": {"D": daily, "W": candle_pack(resample(df, "W-FRI"), 156), "M": candle_pack(resample(df, "ME"), 96)},
        "markers": markers,
    }
