# KMetro CV — 捷運站智慧監控系統

捷運站攝影機電腦視覺系統，支援：
- **行李大小件判別**：在指定 ROI 電子圍籬內偵測大件 / 小件行李
- **行人跌倒偵測**：ROI 內結合 YOLO 偵測與 RTMO 姿態估計雙軌判斷

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
│   ├── model_config.yaml       # 模型參數、影片路徑、偵測閾值
│   └── roi_records.json        # 各影片 ROI 座標紀錄（自動維護）
├── data/
│   └── test_videos/
│       ├── metro/              # 捷運行李偵測影片（IMG_XXXX.MOV）
│       └── fall/               # 跌倒測試影片
├── models/
│   ├── luggage/
│   │   └── yolo_luggage_best.pt    # 行李偵測模型（自訓練 YOLOv8）
│   └── fall_detection/
│       └── yolo12l.pt              # 人員偵測模型（COCO，YOLO12l）
├── src/
│   ├── detection/
│   │   ├── luggage_detector.py     # 行李偵測 + 大小件分類
│   │   ├── fall_detector.py        # YOLO 人員偵測 + 跌倒判斷（aspect ratio）
│   │   └── pose_detector.py        # RTMO 姿態估計 + 跌倒判斷（骨架向量）
│   ├── roi/
│   │   └── roi_manager.py          # ROI 管理、互動繪製、紀錄 I/O
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
| 人員偵測 + 跌倒判斷（主） | YOLO12l（COCO） | `models/fall_detection/yolo12l.pt` |
| 人員偵測 + 跌倒判斷（補） | RTMO-m（bottom-up） | 自動下載至 `~/.cache/rtmlib/` |

---

## 跌倒偵測雙軌架構

```
每幀流程：
  YOLO12l → ROI 內有 person → aspect ratio 判跌倒（主）
  RTMO    → 永遠掃 ROI → 骨架向量判跌倒（視覺化 + 補位）

  警報計算：YOLO 有人用 YOLO，YOLO 沒人才以 RTMO 結果補位
```

**YOLO（主）**：速度快，在一般光線下偵測穩定；CLAHE 增強後改善暗區表現。

**RTMO（補）**：one-stage bottom-up 模型，不需先有 person bbox，直接從全圖找骨架關節點。當 YOLO 因低亮度漏偵測時介入。跌倒判斷改用「肩-髖向量角度」（主）或「鼻-肩向量角度」（下半身不可見時的 fallback）。

---

## Config 設定說明

`configs/model_config.yaml`：

```yaml
video:
  test_dir: "data/test_videos/metro"
  target: "IMG_2734.MOV"           # 留空則批次處理全部
  output_dir: "outputs/videos"
  display_scale: 0.5

luggage:
  weight: "models/luggage/yolo_luggage_best.pt"
  conf_threshold: 0.4
  size_method: "person_ratio"      # person_ratio（主）/ frame_ratio（備援）
  large_person_area_ratio: 0.22    # 行李面積 / 行人面積 >= 此值 → 大件
  max_match_distance_px: 400
  large_luggage_area_ratio: 0.01   # frame_ratio 備援門檻

pose_fallback:
  enabled: false                   # true 啟用 RTMO，false 完全關閉
  mode: "balanced"                 # lightweight / balanced / performance
  kp_conf: 0.3                     # 關節點信心門檻
  min_kp: 3                        # 最少有效關節點數（電扶梯遮蔽場景）
  fall_angle_deg: 50.0             # 肩-髖向量與垂直軸夾角超過此值 → 橫倒
  device: "cpu"

fall_detection:
  weight: "models/fall_detection/yolo12l.pt"
  conf_threshold: 0.3
  fallen_aspect_ratio: 1.2
  alert_frames: 5
  clahe: true                      # CLAHE 對比增強，改善暗區偵測率
  gamma: 0.5                       # gamma < 1 提亮暗部（0.5 = sqrt），1.0 = 關閉
  imgsz: 1280                      # YOLO 推論解析度（640 = 速度優先 / 1280 = 細節優先）
```

---

## ROI 電子圍籬

### 首次執行（自動開啟繪製工具）

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
| `ESC` / `Q` | 儲存並結束 |

### 重新繪製 ROI

```bash
# 推論時強制重畫
python src/pipeline/run_inference.py --reset-roi

# 獨立工具
python scripts/setup_roi.py --video IMG_2734.MOV

# 列出已設定的影片
python scripts/setup_roi.py --list
```

---

## 執行推論

```bash
conda activate kmetro
cd /Users/jaysu/KMetro_cv

# config 指定的影片
python src/pipeline/run_inference.py

# 指定影片
python src/pipeline/run_inference.py --video IMG_2731.MOV

# 即時攝影機
python src/pipeline/run_inference.py --camera 0

# 強制重設 ROI
python src/pipeline/run_inference.py --reset-roi
```

推論結果輸出至 `outputs/videos/`，檔名格式：`{影片名}_{時間戳}.mp4`。

### 推論畫面說明

| 顯示元素 | 說明 |
|----------|------|
| 半透明多邊形 | ROI 電子圍籬 |
| 橘色 bbox | 大件行李 |
| 綠色 bbox | 小件行李 |
| 綠色 bbox（細） | YOLO 偵測到的正常行人 |
| 紅色 bbox | YOLO 偵測到跌倒 |
| 橘色 bbox + 骨架 | RTMO 偵測到的正常行人 |
| 紅橘色 bbox + 骨架 | RTMO 偵測到跌倒 |
| 畫面左上紅色警報 | `!! FALL DETECTED !!`（連續 5 幀） |
| `ESC` | 中止推論 |

---

## 跌倒模型獨立測試

```bash
# 預設測試影片
python scripts/test_fall.py

# 指定影片與參數
python scripts/test_fall.py --video data/test_videos/fall/fall5.mp4
python scripts/test_fall.py --conf 0.3 --aspect_ratio 1.0 --alert_frames 3
```

---

## 大小件判斷邏輯

**主要方法（`person_ratio`）**：

```
size_ratio = 行李 bbox 面積 / 最近行人 bbox 面積
size_ratio >= 0.22  →  大件
size_ratio <  0.22  →  小件
```

若最近行人距離 > 400px 或畫面內無行人，自動切換備援。

**備援方法（`frame_ratio`）**：

```
frame_ratio = 行李 bbox 面積 / 畫面總面積
frame_ratio >= 0.01  →  大件
```

---

## roi_records.json 格式

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
