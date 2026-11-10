"""全域設定：貨幣對、資料來源、預設參數。要新增貨幣對只需改這裡。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = ROOT / "docs"          # GitHub Pages 發佈目錄
PARAMS_FILE = DATA_DIR / "params.json"
PRED_LOG_FILE = DATA_DIR / "predictions.csv"
LATEST_FILE = DATA_DIR / "latest.json"

TIMEZONE = "Asia/Taipei"
HISTORY_START = "2018-01-01"      # 首次下載的起始日

# key -> Yahoo Finance 代號、顯示名稱、TradingView 代號、小數位數
PAIRS = {
    "USDTWD": {"ticker": "TWD=X",    "label": "USD/TWD (美金/台幣)", "tv": "FX_IDC:USDTWD", "decimals": 3},
    "EURUSD": {"ticker": "EURUSD=X", "label": "EUR/USD (歐元/美金)", "tv": "FX_IDC:EURUSD", "decimals": 5},
    "USDJPY": {"ticker": "JPY=X",    "label": "USD/JPY (美金/日圓)", "tv": "FX_IDC:USDJPY", "decimals": 3},
    "GBPUSD": {"ticker": "GBPUSD=X", "label": "GBP/USD (英鎊/美金)", "tv": "FX_IDC:GBPUSD", "decimals": 5},
    "USDCHF": {"ticker": "CHF=X",    "label": "USD/CHF (美金/瑞郎)", "tv": "FX_IDC:USDCHF", "decimals": 5},
}

# 跨市場總經因子（自動抓取；抓不到時以 0 補，不影響其他功能）
MACRO = {
    "gold":  "GC=F",       # 國際黃金
    "twii":  "^TWII",      # 台股加權指數
    "dxy":   "DX-Y.NYB",   # 美元指數
    "us10y": "^TNX",       # 美國 10 年期公債殖利率
}

# 模型/學習預設參數（會被每個貨幣對的調參結果覆蓋，存於 data/params.json）
DEFAULT_PARAMS = {
    "horizon": 5,          # 預測未來幾根日線的漲跌
    "l2": 0.3,             # 邏輯迴歸 L2 正則
    "buy_margin": 0.05,    # p >= 0.5+margin 買進；p <= 0.5-margin 賣出
    "lr": 0.05,
    "epochs": 300,
    "min_train": 250,      # walk-forward 最少訓練樣本
    "retrain_every": 20,   # walk-forward 每幾根重訓
    "cost_bps": 2.0,       # 單次進出場成本（基點）
}

# 調參搜尋空間
TUNE_HORIZONS = [3, 5, 10]
TUNE_L2 = [0.03, 0.3, 3.0]
TUNE_MARGINS = [0.02, 0.04, 0.07]
TUNE_MIN_TRADES = 15

RETUNE_EVERY_DAYS = 7      # 每週重新調參
LIVE_MIN_RESOLVED = 20     # 實盤預測累積到此數量才做劣化判斷
LIVE_DEGRADE_HIT = 0.45    # 近期實盤方向命中率低於此值 → 立即重新調參
