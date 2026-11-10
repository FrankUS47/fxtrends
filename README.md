# fx_trend — 匯率 K 線分析與買賣預測系統（Python 版）

把原本的 `trend.html` 改寫為 Python：資料由 Yahoo Finance 下載，分析與預測在 Python 內完成，
結果輸出成靜態網頁 `docs/index.html`，透過 GitHub Actions 定時更新、GitHub Pages 發佈。

## 一、部署到 GitHub（一次性設定）

1. 在 GitHub 建立新 repo，把本資料夾**全部內容**推上去（含隱藏資料夾 `.github`）。
2. repo → **Settings → Pages** → Source 改選 **GitHub Actions**（不要選 Deploy from a branch）。
3. repo → **Actions → update-fx-trend → Run workflow** 手動跑第一次（首次會下載 2018 起的歷史資料並調參，約數分鐘）。
4. 網址：`https://<你的帳號>.github.io/<repo 名稱>/`

workflow 檔內已宣告 `contents: write` 權限，通常**不需要**再去改 Settings → Actions → General → Workflow permissions。
只有當第一次執行在「Commit data」步驟出現 403 時，才到那一頁（往下捲到最底）改成 Read and write。

### 更新速度

- 分析：每 5 分鐘一次（GitHub 排程的最小間隔，實際常延遲數分鐘），週日至週五。
- 圖表與報價：頁面預設用 TradingView 即時圖與即時報價，是真正的即時。
- 頁面每 5 分鐘自動重新整理，並記住你選的貨幣對、週期與圖表。
- 用 Actions 部署頁面不受 Pages「每小時 10 次建置」限制；預測紀錄與參數每小時才 commit 一次。
- 要更即時的 Python 分析，請在自己的電腦/主機上跑 `python run.py watch --interval 60`。
- 私有 repo 的 Actions 有免費額度（每月 2000 分鐘），每 5 分鐘一次會超過，請改成 Public 或把 cron 放寬。

## 二、本機使用

```bash
pip install -r requirements.txt
python run.py update                 # 更新一次 → docs/index.html
python run.py watch --interval 300   # 每 5 分鐘更新並開 http://localhost:8000
python run.py update --pairs USDTWD,USDJPY
python run.py update --retune        # 強制重新調參
python run.py update --demo          # 模擬資料（僅測試介面，千萬別當真）
python -m unittest discover -s tests
```

## 三、模組

| 檔案 | 用途 |
|---|---|
| `fx_trend/config.py` | 貨幣對、資料代號、預設參數、調參範圍（新增幣別改這裡） |
| `fx_trend/data.py` | 下載 / 增量快取（`data/raw/*.csv`）/ 模擬資料 |
| `fx_trend/indicators.py` | RSI、MACD、布林、ATR、KD |
| `fx_trend/features.py` | 特徵工程（含黃金、台股、美元指數、美債殖利率，抓不到則補 0） |
| `fx_trend/model.py` | 邏輯迴歸（numpy，類別平衡） |
| `fx_trend/levels.py` | 支撐壓力、樞軸點、趨勢通道、區間預測 |
| `fx_trend/patterns.py` | 雙底/雙頂、三角形、吞噬、晨星/夜星、錘子/流星 |
| `fx_trend/learner.py` | walk-forward 回測、調參、預測紀錄與對答案 |
| `fx_trend/pipeline.py` | 單一貨幣對的完整流程 |
| `fx_trend/report.py` | 產生網頁與 `data/latest.json` |
| `fx_trend/cli.py`、`run.py` | 命令列入口 |

要修某個功能，只需替換對應的單一檔案。

## 四、「持續調整參數、讓預測更接近」是怎麼做的

- **每次更新都重訓**：用最新資料重新訓練模型權重。
- **walk-forward**：歷史訊號每一段都只用「當時之前」的資料訓練，避免偷看未來。
- **調參**：預測天數（3/5/10）、正則強度、買賣門檻，只用前 70% 的樣本外結果挑選；後 30% 為 holdout，只報告、不參與選擇。
- **實盤對答案**：每次預測寫入 `data/predictions.csv`，到期後自動結算命中與否，頁面顯示實盤命中率。
- **自動修正**：每 7 天重新調參；若近 20 筆以上的實盤方向命中率低於 45%，立刻重新調參。

## 五、與原版 HTML 的差異與限制

- 原版的支撐壓力是「現價 ×1.008」、型態是隨機挑選的文字、分數是固定加減分；現在全部改為由實際 K 線與模型計算。
- 沒有收錄 1 小時與 15 分鐘週期（免費資料的日內歷史太短、不適合訓練），改為日/週/月線。
- 原版「開啟買賣通知」按鈕未保留；靜態網頁無法在頁面關閉時推播。可另加 GitHub Actions 寄信或 LINE/Telegram 通知。
- 外匯接近有效市場，**樣本外命中率常只略高於 50%**；請看頁面「模型可信度」，尤其是 Holdout 與實盤欄位，並注意單純上漲率作對照。訊號少、樣本小時數字不穩。本工具不是投資建議。
- Yahoo 偶爾會擋雲端主機：抓取失敗時會自動沿用快取並在記錄中警告，下一輪再補抓。
- GitHub 排程是盡力而為，可能延遲；公開 repo 若 60 天完全沒有活動，排程可能被自動停用，需到 Actions 頁面重新啟用。
- 每小時一次 commit，歷史會逐漸變多；不在意的話無需處理。
