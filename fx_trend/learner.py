"""學習層：walk-forward 回測、調參、實盤預測紀錄與自我檢討。

「讓預測更接近」的做法（全部是樣本外驗證，不是看訓練集）：
1. 每次更新都用最新資料重新訓練模型權重。
2. walk-forward：每段預測只用該時點之前的資料訓練，避免偷看未來。
3. 調參（horizon / L2 / 買賣門檻）只用前 70% 的樣本外結果挑選，
   後 30% 留作 holdout 報告，降低過度擬合的自我欺騙。
4. 每次預測都寫入 predictions.csv，到期後自動對答案，統計「實盤」命中率；
   若實盤命中率持續低於門檻，立刻觸發重新調參。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from .model import LogitModel


# ---------------------------------------------------------------- walk-forward
def forward_returns(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1


def walk_forward(X: pd.DataFrame, fwd: pd.Series, p: dict) -> pd.Series:
    """回傳樣本外機率序列（index 與 X 相同的後段）。"""
    h, n = p["horizon"], len(X)
    Xv, fv = X.values, fwd.reindex(X.index).values
    out = pd.Series(np.nan, index=X.index)
    t = p["min_train"]
    while t < n:
        tr = np.arange(0, max(t - h, 0))            # 標籤須在 t 之前已揭曉
        tr = tr[~np.isnan(fv[tr])]
        if len(tr) >= 100:
            y = (fv[tr] > 0).astype(float)
            if 0 < y.mean() < 1:
                m = LogitModel(p["l2"], p["lr"], p["epochs"]).fit(Xv[tr], y)
                end = min(t + p["retrain_every"], n)
                out.iloc[t:end] = m.predict_proba(Xv[t:end])
        t += p["retrain_every"]
    return out


def signals(proba: pd.Series, margin: float) -> pd.Series:
    s = pd.Series(0, index=proba.index)
    s[proba >= 0.5 + margin] = 1
    s[proba <= 0.5 - margin] = -1
    return s.where(proba.notna(), 0)


def evaluate(proba: pd.Series, fwd: pd.Series, margin: float, cost_bps: float, h: int) -> dict:
    f = fwd.reindex(proba.index)
    ok = proba.notna() & f.notna()
    sig = signals(proba, margin)[ok]
    f = f[ok]
    trades = sig != 0
    n = int(trades.sum())
    base = float((f > 0).mean()) if len(f) else float("nan")
    if n == 0:
        return {"n": 0, "samples": int(ok.sum()), "hit": None, "mean_ret": None,
                "score": -1e9, "base_up": base}
    r = sig[trades] * f[trades] - cost_bps / 1e4
    hit = float((sig[trades] * f[trades] > 0).mean())
    mean = float(r.mean())
    return {"n": n, "samples": int(ok.sum()), "hit": hit, "mean_ret": mean,
            "sharpe": float(mean / r.std() * np.sqrt(252 / h)) if r.std() > 0 else 0.0,
            "score": mean * np.sqrt(n), "base_up": base}


# ------------------------------------------------------------------- 調參
def tune(X: pd.DataFrame, close: pd.Series, base: dict | None = None) -> tuple[dict, dict]:
    base = dict(base or C.DEFAULT_PARAMS)
    best, best_score, tried = None, -1e18, 0
    for h in C.TUNE_HORIZONS:
        fwd = forward_returns(close, h)
        for l2 in C.TUNE_L2:
            p = {**base, "horizon": h, "l2": l2}
            proba = walk_forward(X, fwd, p).dropna()
            if len(proba) < 120:
                continue
            cut = proba.index[int(len(proba) * 0.7)]
            fit_part = proba[proba.index < cut]
            for mg in C.TUNE_MARGINS:
                ev = evaluate(fit_part, fwd, mg, p["cost_bps"], h)
                tried += 1
                if ev["n"] >= C.TUNE_MIN_TRADES and ev["score"] > best_score:
                    best_score, best = ev["score"], ({**p, "buy_margin": mg}, ev)
    if best is None:
        return base, {"note": "樣本不足或無有效組合，沿用預設參數", "tried": tried}
    params, ev = best
    return params, {"tuned_on": ev, "tried": tried}


# ------------------------------------------------------------- 參數儲存
class ParamStore:
    def __init__(self):
        self.d = {}
        if C.PARAMS_FILE.exists():
            self.d = json.loads(C.PARAMS_FILE.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict | None:
        return self.d.get(key)

    def set(self, key: str, params: dict, report: dict) -> None:
        self.d[key] = {"params": params, "tuned_at": date.today().isoformat(), "report": report}

    def due(self, key: str) -> bool:
        e = self.d.get(key)
        if not e:
            return True
        age = (date.today() - date.fromisoformat(e["tuned_at"])).days
        return age >= C.RETUNE_EVERY_DAYS

    def save(self) -> None:
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        C.PARAMS_FILE.write_text(json.dumps(self.d, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------- 實盤預測紀錄
COLS = ["pair", "bar_date", "horizon", "proba", "action", "close", "fwd_ret", "hit", "resolved"]


class PredictionLog:
    def __init__(self):
        if C.PRED_LOG_FILE.exists():
            self.df = pd.read_csv(C.PRED_LOG_FILE, dtype={"bar_date": str})
        else:
            self.df = pd.DataFrame(columns=COLS)
        self.df["resolved"] = self.df["resolved"].astype(bool)

    def upsert(self, pair: str, bar_date: str, horizon: int, proba: float, action: str, close: float):
        m = (self.df.pair == pair) & (self.df.bar_date == bar_date)
        row = {"pair": pair, "bar_date": bar_date, "horizon": horizon, "proba": round(proba, 4),
               "action": action, "close": close, "fwd_ret": np.nan, "hit": np.nan, "resolved": False}
        if m.any():
            if self.df.loc[m, "resolved"].any():      # 已揭曉者不覆蓋
                return
            self.df = self.df[~m]
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def resolve(self, pair: str, close: pd.Series) -> None:
        idx = close.index.strftime("%Y-%m-%d")
        pos_map = {d: i for i, d in enumerate(idx)}
        for i in self.df.index[(self.df.pair == pair) & (~self.df.resolved)]:
            pos = pos_map.get(self.df.at[i, "bar_date"])
            h = int(self.df.at[i, "horizon"])
            if pos is None or pos + h >= len(close):
                continue
            fwd = close.iloc[pos + h] / close.iloc[pos] - 1
            act = self.df.at[i, "action"]
            self.df.at[i, "fwd_ret"] = fwd
            self.df.at[i, "hit"] = (1.0 if (act == "BUY" and fwd > 0) or (act == "SELL" and fwd < 0)
                                    else 0.0) if act in ("BUY", "SELL") else np.nan
            self.df.at[i, "resolved"] = True

    def live_stats(self, pair: str, last: int = 30) -> dict:
        d = self.df[(self.df.pair == pair) & self.df.resolved & self.df.hit.notna()]
        rec = d.tail(last)
        return {"n": int(len(d)), "hit_all": float(d.hit.mean()) if len(d) else None,
                "n_recent": int(len(rec)), "hit_recent": float(rec.hit.mean()) if len(rec) else None}

    def recent(self, pair: str, k: int = 12) -> list[dict]:
        d = self.df[self.df.pair == pair].sort_values("bar_date").tail(k).iloc[::-1]
        return [{"date": r.bar_date, "action": r.action, "p": float(r.proba), "h": int(r.horizon),
                 "fwd": None if pd.isna(r.fwd_ret) else float(r.fwd_ret),
                 "hit": None if pd.isna(r.hit) else bool(r.hit)} for r in d.itertuples()]

    def save(self) -> None:
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(C.PRED_LOG_FILE, index=False)


def degraded(live: dict) -> bool:
    return (live["n_recent"] >= C.LIVE_MIN_RESOLVED and live["hit_recent"] is not None
            and live["hit_recent"] < C.LIVE_DEGRADE_HIT)
