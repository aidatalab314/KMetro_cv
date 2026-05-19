# KMetro CV — 捷運站智慧監控系統

捷運站攝影機電腦視覺系統，支援：
- **行李大小件判別**：在指定 ROI 電子圍籬內偵測大件 / 小件行李
- **行人跌倒偵測**：即時偵測 ROI 內人員跌倒事件

所有偵測僅在 ROI 電子圍籬範圍內啟動；未設定 ROI 時自動切換為全幀偵測模式。

---

## 環境需求

- Python 3.10+
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## 環境建立

```bash
conda create -n kmetro python=3.10 -y
conda activate kmetro
pip install -r requirements.txt
```

> 每次開發前先執行 `conda activate kmetro`。

---

## 專案架構

```
KMetro_cv/
├── configs/
│   ├── model_config.yaml       # 模型參數、影片路徑、大小件判斷閾值
│   └── roi_records.json        # 各影片 ROI 座標紀錄（自動維護）
├── data/
│   ├── test_videos/
│   │   ├── metro/              # 捷運行李偵測影片（IMG_XXXX.MOV）
│   │   └── fall/               # 跌倒測試影片（fall5.mp4 等）
│   ├── raw/                    # 原始訓練資料
│   ├── annotated/              # 標註資料（YOLO 格式）
│   └── splits/train|val|test/
├── models/
│   ├── luggage/
│   │   └── yolo_luggage_best.pt    # 行李偵測模型（自訓練）
│   └── fall_detection/
│       └── yolov10s.pt             # 跌倒偵測模型（COCO person）
├── src/
│   ├── detection/
│   │   ├── luggage_detector.py     # 行李偵測 + 大小件分類
│   │   └── fall_detector.py        # 跌倒偵測（YOLOv10s）
│   ├── roi/
│   │   └── roi_manager.py          # ROI 管理、互動繪製工具、紀錄 I/O
│   └── pipeline/
│       └── run_inference.py        # 主推論入口
├── scripts/
│   ├── setup_roi.py                # 獨立 ROI 繪製 CLI
│   └── test_fall.py                # 跌倒模型獨立測試
├── outputs/
│   └── videos/                     # 推論結果影片輸出
└── requirements.txt
```

---

## 模型說明

| 功能 | 模型 | 路徑 |
|------|------|------|
| 行李偵測（大/小件） | 自訓練 YOLOv8 | `models/luggage/yolo_luggage_best.pt` |
| 行人偵測 + 跌倒判斷 | YOLOv10s（COCO） | `models/fall_detection/yolov10s.pt` |

> 兩個任務共用同一個 YOLOv10s 推論結果，不重複執行。

---

## Config 設定說明

`configs/model_config.yaml`：

```yaml
video:
  test_dir: "data/test_videos/metro"  # 捷運影片目錄
  target: "IMG_2734.MOV"              # 指定影片；留空則批次處理全部
  output_dir: "outputs/videos"        # 推論結果輸出目錄
  display_scale: 0.5                  # 預覽視窗縮放比例（1.0 = 原始大小）

luggage:
  weight: "models/luggage/yolo_luggage_best.pt"
  conf_threshold: 0.4                 # 行李偵測信心值門檻

  size_method: "person_ratio"         # 大小件分類方法：
                                      #   person_ratio → 以最近行人 bbox 面積為基準
                                      #   frame_ratio  → 以畫面面積為基準（備援）

  large_person_area_ratio: 0.22       # [person_ratio] 行李面積 / 行人面積 >= 此值 → 大件
  max_match_distance_px: 400          # [person_ratio] 行李與最近行人的最大配對距離（px）
  large_luggage_area_ratio: 0.01      # [frame_ratio] 行李面積 / 畫面面積 >= 此值 → 大件

fall_detection:
  weight: "models/fall_detection/yolov10s.pt"
  conf_threshold: 0.5                 # 人體偵測信心值門檻
  fallen_aspect_ratio: 1.2            # 人體 bbox 寬/高比 >= 此值視為橫倒（跌倒）
  alert_frames: 5                     # 連續 N 幀偵測到跌倒才觸發警報
```

---

## ROI 電子圍籬

### 首次執行（自動開啟繪製工具）

第一次對某支影片執行推論時，系統會自動讀取第一幀並開啟互動繪製視窗：

```bash
conda activate kmetro
cd /Users/jaysu/KMetro_cv
python src/pipeline/run_inference.py
```

### ROI 繪製操作說明

| 操作 | 動作 |
|------|------|
| 左鍵點擊 | 新增頂點 |
| 右鍵點擊 | 刪除最後一個頂點 |
| `C` | 確認當前多邊形（至少 3 點），可繼續畫下一個 ROI |
| `R` | 重置當前未完成的多邊形 |
| `ESC` / `Q` | 儲存並結束（若有未確認的點 ≥ 3，自動完成最後一個 ROI） |

- 達到 3 個點後，圍籬邊界會自動閉合並顯示半透明填色預覽
- 可在同一支影片設定多個 ROI 區域
- 設定完成後儲存至 `configs/roi_records.json`，下次執行直接載入

### 重新繪製 ROI

```bash
# 方法一：執行推論時加上 --reset-roi
python src/pipeline/run_inference.py --reset-roi

# 方法二：使用獨立工具（直接覆蓋舊紀錄）
python scripts/setup_roi.py --video IMG_2734.MOV
```

### 查看已設定 ROI 的影片清單

```bash
python scripts/setup_roi.py --list
```

---

## 執行推論

```bash
conda activate kmetro
cd /Users/jaysu/KMetro_cv

# 執行 config 指定的影片（預設 IMG_2734.MOV）
python src/pipeline/run_inference.py

# 指定影片檔名
python src/pipeline/run_inference.py --video IMG_2731.MOV

# 批次處理 metro/ 下所有影片（target 留空）
python src/pipeline/run_inference.py

# 即時攝影機
python src/pipeline/run_inference.py --camera 0

# 強制重新設定 ROI 後執行
python src/pipeline/run_inference.py --reset-roi
```

推論結果影片輸出至 `outputs/videos/`，檔名格式：`{影片名}_{時間戳}.mp4`。

### 推論畫面說明

| 顯示元素 | 說明 |
|----------|------|
| 半透明多邊形填色 | ROI 電子圍籬範圍 |
| 橘色 bbox | 大件行李（`Large luggage`） |
| 綠色 bbox | 小件行李（`Small luggage`） |
| 紅色 bbox | 偵測到跌倒（`FALL`） |
| 綠色 bbox | 正常行走（`person`） |
| 畫面左上紅色警報 | `!! FALL DETECTED !!`（連續 5 幀跌倒） |
| `ESC` | 中止推論 |

---

## 跌倒模型獨立測試

```bash
# 使用預設測試影片 fall5.mp4
python scripts/test_fall.py

# 指定影片
python scripts/test_fall.py --video data/test_videos/fall/fall5.mp4

# 調整參數
python scripts/test_fall.py --conf 0.4 --aspect_ratio 1.0 --alert_frames 3
```

---

## 更換模型

### 行李偵測模型（自訓練 YOLOv8）

1. 將新的 `.pt` 檔放入 `models/luggage/`，例如 `yolo_luggage_v2.pt`
2. 修改 `configs/model_config.yaml`：

```yaml
luggage:
  weight: "models/luggage/yolo_luggage_v2.pt"
```

3. 同步確認 `conf_threshold` 是否需要跟著調整（新模型信心分佈可能不同）

> 行李模型為自訓練模型，輸出只有行李類別（無 class index 限制），直接替換權重即可。

---

### 跌倒偵測模型（YOLOv10s / COCO）

跌倒偵測使用通用 COCO 模型的 **class 0（person）**，可替換為任何相容格式的 YOLO 模型：

1. 將新的 `.pt` 檔放入 `models/fall_detection/`，例如 `yolov8n.pt`
2. 修改 `configs/model_config.yaml`：

```yaml
fall_detection:
  weight: "models/fall_detection/yolov8n.pt"
```

3. 若新模型的 person class index 不是 `0`，需同步修改 `src/detection/fall_detector.py`：

```python
PERSON_CLASS = 0   # 改成新模型對應的 person class index
```

> 若改用姿態估計模型（如 YOLOv8-Pose），需重寫 `fall_detector.py` 的 `detect()` 與 `_is_fallen()` 邏輯（改為關鍵點判斷）。

---

### 驗證新模型是否正常運作

**行李模型：**
```bash
conda activate kmetro
cd /Users/jaysu/KMetro_cv
python src/pipeline/run_inference.py --video IMG_2734.MOV
```

**跌倒模型：**
```bash
python scripts/test_fall.py --video data/test_videos/fall/fall5.mp4
```

---

## 大小件判斷邏輯

**主要方法（`person_ratio`）**：

```
size_ratio = 行李 bbox 面積 / 最近行人 bbox 面積
size_ratio >= 0.22  →  大件（Large）
size_ratio <  0.22  →  小件（Small）
```

- 若畫面內最近行人距離 > 400px，自動切換備援方法
- 若畫面內無行人，切換備援方法

**備援方法（`frame_ratio`）**：

```
frame_ratio = 行李 bbox 面積 / 畫面總面積
frame_ratio >= 0.01  →  大件
frame_ratio <  0.01  →  小件
```

---

## roi_records.json 格式說明

```json
{
  "IMG_2734.MOV": [
    {
      "id": "roi_0",
      "label": "ROI 0",
      "color": [0, 255, 255],
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

- key 為影片檔名（不含路徑）
- 每支影片可設定多個 ROI
- 座標為原始影片解析度下的像素座標
