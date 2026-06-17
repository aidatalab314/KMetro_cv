# KMetro CV — 技術評估報告（歷史存檔）

> 版本：2026-06-16（凍結）
> 此文件為初期技術選型評估記錄，不再更新。
> 現況與待處理事項請見 [`STATUS.md`](STATUS.md)；架構說明見 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 1. 部署環境與開發策略

### 硬體環境

| 角色 | 機器 | 用途 |
|------|------|------|
| 主開發機 | Apple M1 Mac mini（16GB） | 演算法開發、試錯、本地影片測試 |
| 部署機 | Jetson Orin | 現場安裝，4路RTSP實際推論 |

### 現場架構
- **1 台 Jetson Orin 對接 4 台 RTSP 攝影機（multistream）**
- 每台攝影機依架設角度啟用不同功能組合
- 所有功能均需在各攝影機的電子圍籬（ROI）內生效

### 開發環境策略
- **Mac**：conda env `kmetro`，`device: mps`，以本地測試影片開發
- **Jetson**：`device: 0`（CUDA），RTSP 串流，TensorRT 加速
- 平台差異透過 `configs/cameras.local.yaml` 覆蓋，不進版控

---

## 2. 加速器評估

### GStreamer / TensorRT（採用）
```
RTSP → GStreamer nvv4l2decoder（Jetson HW解碼）→ Python frame queue
     → ultralytics YOLO（TensorRT .engine）→ 各功能演算法 → 事件輸出
```
- GStreamer 只用於 **RTSP 解碼層**（已在 csas_poc `rtsp_reader.py` 實作）
- TensorRT 透過 `model.export(format="engine")` 轉換，ultralytics 原生支援
- Mac 開發階段用 `mps` backend，不需額外處理

### DeepStream（不採用）
**結論：本專案不適合 DeepStream。**

| 項目 | 說明 |
|------|------|
| 自訂邏輯難度 | ROI 繪製、電扶梯角度校正、滯留演算法需寫 C++ plugin |
| ReID 彈性 | NvDCF 封裝過深，無法插入自訂 appearance embedding |
| 開發成本 | 學習曲線陡，與本專案 Python 架構不相容 |
| 替代方案 | GStreamer（解碼） + Python pipeline（推論 + 邏輯）效果等同 |

---

## 3. 功能評估

### 功能 1：月台旅客滯留通報

**核心需求**：辨識每個人、指派 ID、計算 ROI 內滯留時間、超時告警。

**Tracker 選型評估**

| 選項 | 說明 | 結論 |
|------|------|------|
| CentroidTracker（現有） | 純距離配對，人離開畫面即掉 ID | ❌ 不適用 |
| SORT | Kalman + Hungarian，可處理短暫離開 | ✅ 可用 |
| **ByteTrack（ultralytics 內建）** | `model.track(tracker="bytetrack.yaml")`，無需額外模型，Jetson 友好 | ✅ **首選** |
| BoT-SORT（ultralytics 內建） | 加外觀嵌入，更準確，稍重 | 若 ByteTrack 不夠再升級 |
| OSNet 外觀嵌入 | 人離開 30 秒以上仍能認回 | 依商業需求決定是否加 |

**跨攝影機 Re-ID 評估**：用戶直覺正確，跨鏡頭 Re-ID 資源消耗明顯更大。
本場景（月台候車區）用單攝影機即可，不需要跨鏡頭。

**待釐清商業邏輯**：人離開畫面較長時間後回來，滯留計時是否繼續或重置？
MVP 階段先以「重置」處理，ByteTrack 可自然處理短暫消失（<5秒）不重置。

**實作方向**：用 `model.track()` 取代 `CentroidTracker`，在 ByteTrack 回傳的 track ID 上層計算 dwell time。

---

### 功能 2：火光煙霧偵測

**評估結論**：**標準 YOLO COCO 80 類別不含 fire/smoke，必須用獨立模型。**

| 方案 | 說明 | 結論 |
|------|------|------|
| COCO pretrain 直接用 | 無 fire/smoke 類別 | ❌ 不可行 |
| 自行 finetune | 需資料集 + 訓練時間 | 最後手段 |
| **公開 fire/smoke YOLO weight** | D-Fire 資料集的 YOLOv8/11 weight，Roboflow/HuggingFace 可取得 | ✅ **首選** |

- 放置路徑：`models/fire_smoke/fire_smoke.pt`
- 告警 2 秒內：連續 N 幀確認觸發，YOLO 推論延遲完全達得到
- Jetson 部署：export 成 TensorRT .engine

---

### 功能 3：區域人流偵測

**評估結論**：**不需要 Tracker，直接計數 ROI 內的 person bbox 即可。**

**理由**：
- 每 15 秒取一次計數，時間粒度夠粗，tracker 的 ID 一致性優勢不顯著
- Tracker 引入 ID switch、ghost track 等誤差來源
- 純偵測 + ROI 過濾最簡單且準確，10% 誤差規格完全可達

**例外**：若未來需要「累計進出人次」（而非即時人數），才需要 tracker。目前規格是即時人數，跳過。

**ROI 邏輯**：直接沿用 `roi_manager.py` 的 `is_inside()` + `get_containing_rois()`。

---

### 功能 4：電扶梯人員跌倒、行李箱滾落

#### 4a. 跌倒偵測

**Pose 架構評估**：

現有 RTMO bottom-up fallback 架構**合適**。原因：
- 電扶梯有側板遮蔽、部分骨架不可見，top-down（先找 person 再找 keypoint）因 YOLO 漏偵而全失
- Bottom-up 直接從全圖找關節點，遮蔽容忍度高
- 主要問題是「每幀都跑」造成資源浪費，不是底層方法錯

**主要問題：電扶梯傾斜角**

當前 pose 判斷以「垂直軸」為基準，但電扶梯傾斜約 30°，正常站立的人肩-髖向量已偏離垂直。

解決方案：
- ROI 設定時加入 `escalator_angle_deg` 欄位
- PoseDetector 接收此參數，以「電扶梯法線方向」取代垂直軸

**已知 Bug（待修）**：

`pose_detector.py:33` 的 `_fall_cos` 計算錯誤：
```python
# 現在（錯誤）：實際 threshold 是 40°，而非設定的 50°
self._fall_cos = np.cos(np.radians(90 - fall_angle_deg))  # cos(40) ≈ 0.766

# 正確：
self._fall_cos = np.cos(np.radians(fall_angle_deg))  # cos(50) ≈ 0.643
```

**run_inference.py Bug（待修）**：ROI overlay 在 YOLO 推論前繪製，影響模型輸入。應先偵測、再繪製。

**開發模型**：`yolo12n` 或 `yolo11s`，等另一組 finetune 完再替換 weight path。

#### 4b. 行李箱滾落偵測

**演算法設計**：
```
每幀：YOLO 偵測 suitcase → ByteTrack 取得 per-ID 速度向量
判斷滾落（同時滿足）：
  1. 行李 ID 垂直位移速度 > speed_threshold_px_per_frame
  2. 行李與最近人的速度向量差 > independence_threshold  （行李獨立移動）
  3. 持續 N 幀 → 觸發告警
```

Tracker 在此必要（需要速度向量），ByteTrack 統一處理人和行李。

---

### 功能 5：大件行李與輪椅偵測

**輪椅**：COCO 無 wheelchair 類別。選項：
- 等另一組 finetune 模型（若包含輪椅類別）
- 或用 Roboflow 公開輪椅偵測 weight

**大件行李判斷改善方向**：
1. 加 ByteTrack tracking，讓同一行李的大小標籤不在幀間跳動（最大改善點）
2. `_nearest_person` 排除 `fallen=True` 的人（跌倒者 bbox 形狀異常）
3. 電梯前近拍場景可考慮以 `frame_ratio` 取代 `person_ratio`（更穩定）

---

## 4. 整體架構設計

### 目錄結構

```
KMetro_cv/
├── configs/
│   ├── cameras.yaml              # 攝影機清單 + 功能設定 + 模型參數
│   ├── cameras.local.yaml        # 本機覆蓋（gitignore）
│   ├── cameras.local.yaml.example# 各平台設定範本
│   ├── model_config.yaml         # 保留，供舊版單影片模式使用
│   └── roi_records.json          # ROI 座標（key = camera_id）
├── docs/
│   └── TECH_EVALUATION.md        # 本文件
├── data/
│   ├── test_videos/
│   ├── snapshots/                # 事件截圖
│   └── logs/                    # 事件 log
├── models/
│   ├── fall_detection/           # YOLO 人員偵測模型
│   ├── luggage/                  # 行李偵測模型
│   └── fire_smoke/               # 火煙偵測模型（待取得）
├── src/
│   ├── detection/                # 偵測器（保留現有）
│   │   ├── fall_detector.py
│   │   ├── luggage_detector.py
│   │   └── pose_detector.py
│   ├── roi/
│   │   └── roi_manager.py
│   ├── event/
│   │   └── event_manager.py      # 事件輸出（port from csas_poc + 擴充）
│   ├── pipeline/
│   │   ├── run_inference.py      # 保留，單影片 backward compat
│   │   ├── camera_worker.py      # 單攝影機執行緒
│   │   └── multistream.py        # 主入口：4路 multistream
│   ├── rtsp_reader.py            # 影像來源抽象（port from csas_poc）
│   └── utils.py                  # YAML loader + logger（port from csas_poc）
├── scripts/
│   ├── setup_roi.py
│   └── test_fall.py
└── outputs/
    └── videos/
```

### 每幀處理流程

```
RTSPReader.read()
     ↓
[clean frame]
     ├── FallDetector.detect()  → all_persons
     ├── LuggageDetector.detect()  → all_luggage
     └── （FireSmokeDetector.detect()）→ fire_smoke_dets  [TODO]
     ↓
ROI 過濾：persons_in_roi / luggage_in_roi
     ↓
功能處理：
  ├── [功能1] DwellMonitor   → track ID + dwell time → 超時告警 [TODO ByteTrack]
  ├── [功能2] FireSmoke      → 存在即告警 [TODO model]
  ├── [功能3] ZoneCounter    → count(persons_in_roi) → 超量告警 [TODO interval]
  ├── [功能4a] FallDetector  → compute_alert() → 跌倒告警 ✓
  ├── [功能4b] LuggageRoll   → 速度向量 → 滾落告警 [TODO ByteTrack]
  └── [功能5] SizeClassifier → person_ratio → Large/Small ✓
     ↓
ROI.draw() + 各功能繪製 → annotated frame
     ↓
EventManager.trigger() → log + snapshot
VideoWriter.write()     → 輸出影片（本地來源強制，RTSP可選）
Queue.put()             → 顯示執行緒
```

---

## 5. 開發路徑（優先順序）

| 階段 | 項目 | 狀態 |
|------|------|------|
| 1 | **多流架構骨架**（multistream.py + camera_worker.py） | ✅ 完成 |
| 2 | **ByteTrack 整合**（FallDetector + LuggageDetector tracking） | ✅ 完成 |
| 3 | **功能 3**：zone_counter（每 N 秒計數 + overlay） | ✅ 完成 |
| 4 | **功能 1**：dwell_monitor（ByteTrack ID + 滯留計時） | ✅ 完成 |
| 5 | **功能 5**：size_classifier（person_ratio，LuggageDetector） | ✅ 完成 |
| 6 | **功能 4b**：luggage_roll（速度向量 + independence check） | ✅ 完成 |
| 7 | **功能 4a**：fall_detector（YOLO aspect ratio + RTMO pose） | ✅ 完成 |
| 8 | **功能 2**：fire_smoke（取得 weight 後掛上） | ⏳ 待模型 |
| 9 | Jetson TensorRT export + 效能調校 | ⏳ 待開始 |

---

## 6. 已知 Bug 清單

| 編號 | 檔案 | 說明 | 嚴重度 |
|------|------|------|--------|
| B1 | `pose_detector.py:33` | `_fall_cos` 角度計算錯誤，實際 threshold 比設定少 10° | High |
| B2 | `run_inference.py:117` | ROI overlay 在 YOLO 推論前繪製，影響模型輸入 | Medium |
| B3 | `run_inference.py:129` | PoseDetector 每幀都跑，但只有 YOLO 沒人時才用結果 | Medium |
| B4 | `pose_detector.py:4` | 引入 FallDetector 只為借用 `_build_gamma_lut`，耦合不當 | Low |
| B5 | `luggage_detector.py:25` | `_nearest_person` 不排除跌倒者，bbox 異常影響計算 | Low |

> B1、B2 優先修復；B3-B5 在多流架構重構時一併解決。

---

## 7. 預留擴充空間

以下功能目前不實作，但在架構中預留 hook：

- **喇叭告警**：`EventManager.trigger()` 回傳 `True` 時呼叫 `_on_alert_hook()`（預設 no-op）
- **事件錄影**：`camera_worker.py` 的 `_on_event_record()` 預留介面
- **API/WebSocket 推播**：`EventManager` 的 `_on_remote_notify()` 預留介面
- **跨攝影機 Re-ID**：架構設計預留 `cross_cam_reid` 參數，目前 disabled
