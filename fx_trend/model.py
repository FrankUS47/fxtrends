"""預測模型：帶 L2 的邏輯迴歸（numpy 實作，Adam 優化，類別平衡）。

輸出機率 p = P(未來 horizon 日報酬 > 0)；score = 100·p。
類別平衡可避免模型只學到「這段期間 USD 一直漲」的長期漂移。
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogitModel:
    def __init__(self, l2: float = 0.3, lr: float = 0.05, epochs: int = 300):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w = self.b = self.mu = self.sd = None

    def _z(self, X):
        return np.clip((np.asarray(X, float) - self.mu) / self.sd, -4, 4)

    def fit(self, X, y) -> "LogitModel":
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = self._z(X)
        n, d = Z.shape
        pos = max(y.mean(), 1e-6)
        sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-6))
        sw = sw / sw.mean()
        w, b = np.zeros(d), 0.0
        mw, vw, mb, vb = np.zeros(d), np.zeros(d), 0.0, 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            err = (_sigmoid(Z @ w + b) - y) * sw
            gw, gb = Z.T @ err / n + self.l2 * w, err.mean()
            mw, vw = b1 * mw + (1 - b1) * gw, b2 * vw + (1 - b2) * gw**2
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb**2
            w -= self.lr * (mw / (1 - b1**t)) / (np.sqrt(vw / (1 - b2**t)) + eps)
            b -= self.lr * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + eps)
        self.w, self.b = w, b
        return self

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self._z(np.atleast_2d(X)) @ self.w + self.b)

    def contributions(self, x) -> np.ndarray:
        """單一樣本各特徵對 logit 的貢獻（>0 偏多、<0 偏空）。"""
        return self._z(np.atleast_2d(x))[0] * self.w
