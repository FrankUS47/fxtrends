"""命令列入口：update（一次更新）、watch（本機持續更新並提供網頁）。"""
from __future__ import annotations

import argparse
import functools
import http.server
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config as C
from .data import load_macro, load_pair
from .learner import ParamStore, PredictionLog
from .pipeline import analyze_pair
from .report import write_outputs


def run_update(pairs: list[str], demo: bool = False, fetch: bool = True, retune: bool = False) -> int:
    macro = load_macro(fetch and not demo, demo)
    store, log = ParamStore(), PredictionLog()
    results, errors = {}, {}
    for key in pairs:
        try:
            df = load_pair(key, fetch and not demo, demo)
            results[key] = analyze_pair(key, df, macro, store, log, retune)
            r = results[key]
            print(f"[{key}] {r['bar_date']} 收 {r['price']:.4f}｜分數 {r['score']}｜{r['action_text']}")
        except Exception as e:  # noqa: BLE001
            errors[key] = str(e)
            print(f"[error] {key}: {e}")
    if not results:
        print("所有貨幣對皆失敗，未更新輸出。")
        return 1
    store.save()
    log.save()
    now = datetime.now(ZoneInfo(C.TIMEZONE)).strftime("%Y-%m-%d %H:%M")
    write_outputs(results, now + (" [DEMO 模擬資料]" if demo else ""), errors)
    print(f"完成：{C.DOCS_DIR / 'index.html'}")
    return 0


def run_watch(pairs, interval: int, port: int, demo: bool, retune: bool) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(C.DOCS_DIR))
    C.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"網頁：http://localhost:{port}/  （每 {interval} 秒更新，Ctrl+C 結束）")
    while True:
        try:
            run_update(pairs, demo=demo, retune=retune)
            retune = False
        except Exception as e:  # noqa: BLE001
            print(f"[error] 本輪更新失敗：{e}")
        time.sleep(interval)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fx_trend", description="匯率 K 線分析與買賣預測")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("update", "watch"):
        sp = sub.add_parser(name)
        sp.add_argument("--pairs", default=",".join(C.PAIRS), help="以逗號分隔，如 USDTWD,USDJPY")
        sp.add_argument("--demo", action="store_true", help="使用模擬資料（僅測試）")
        sp.add_argument("--retune", action="store_true", help="強制重新調參")
        if name == "update":
            sp.add_argument("--no-fetch", action="store_true", help="只用快取資料，不連網")
        else:
            sp.add_argument("--interval", type=int, default=300)
            sp.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    pairs = [p.strip().upper() for p in a.pairs.split(",") if p.strip()]
    bad = [p for p in pairs if p not in C.PAIRS]
    if bad:
        ap.error(f"未知貨幣對：{bad}；可用：{list(C.PAIRS)}")
    if a.cmd == "update":
        return run_update(pairs, a.demo, not a.no_fetch, a.retune)
    run_watch(pairs, a.interval, a.port, a.demo, a.retune)
    return 0
