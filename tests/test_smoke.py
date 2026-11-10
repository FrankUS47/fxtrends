import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fx_trend import config as C
from fx_trend import indicators as ind
from fx_trend.data import synthetic_ohlc
from fx_trend.features import FEATURES, build_features
from fx_trend.learner import PredictionLog, forward_returns, walk_forward
from fx_trend.levels import compute_levels
from fx_trend.model import LogitModel
from fx_trend.patterns import detect_patterns


class Smoke(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_ohlc("USDTWD", n=900)

    def test_indicators(self):
        r = ind.rsi(self.df.Close).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())
        self.assertGreater(ind.atr(self.df).iloc[-1], 0)

    def test_features_no_nan(self):
        X = build_features(self.df, {})
        self.assertEqual(list(X.columns), FEATURES)
        self.assertFalse(X.isna().any().any())

    def test_model_learns_signal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(800, 3))
        y = (X[:, 0] + 0.3 * rng.normal(size=800) > 0).astype(float)
        p = LogitModel().fit(X, y).predict_proba(X)
        self.assertGreater(((p > .5) == y).mean(), 0.8)

    def test_walk_forward_has_no_lookahead(self):
        X = build_features(self.df, {})
        p = {**C.DEFAULT_PARAMS}
        fwd = forward_returns(self.df.Close, p["horizon"])
        o = walk_forward(X, fwd, p)
        self.assertTrue(o.iloc[:p["min_train"]].isna().all())
        self.assertGreater(o.notna().sum(), 100)

    def test_levels_and_patterns(self):
        lv = compute_levels(self.df)
        self.assertLess(lv["support"], lv["close"])
        self.assertGreater(lv["resistance"], lv["close"])
        self.assertTrue(detect_patterns(self.df))

    def test_prediction_log_resolves(self):
        old = C.PRED_LOG_FILE
        with tempfile.TemporaryDirectory() as d:
            C.PRED_LOG_FILE = Path(d) / "p.csv"
            try:
                log = PredictionLog()
                close = self.df.Close
                day = close.index[-10].strftime("%Y-%m-%d")
                log.upsert("X", day, 5, 0.7, "BUY", float(close.iloc[-10]))
                log.resolve("X", close)
                self.assertTrue(bool(log.df.resolved.iloc[0]))
                self.assertEqual(log.live_stats("X")["n"], 1)
            finally:
                C.PRED_LOG_FILE = old


if __name__ == "__main__":
    unittest.main()
