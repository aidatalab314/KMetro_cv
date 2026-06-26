# KMetro CV — 專案現況

> 最後更新：2026-06-26

---

## 實作狀態

### 功能模組

| 功能 | 模組 | 狀態 | 備註 |
|------|------|------|------|
| 功能1 旅客滯留 | `DwellMonitor` + ByteTrack | ✓ | Re-ID 限制見 D1 |
| 功能2 火光煙霧 | `FireSmokeDetector`（YOLOv8n） | ✓ | luminous0219/fire-and-smoke-detection-yolov8 |
| 功能3 人流偵測 | `ZoneCounter` | ✓ | |
| 功能4a 電扶梯跌倒 | `FallDetector` + `PoseDetector` | ✓ | escalator_angle 尚未整合（見 D2） |
| 功能4b 行李箱滾落 | `LuggageRollMonitor` + ByteTrack | ✓ | |
| 功能5 大件行李 | `LuggageDetector` size_classifier | ✓ | 輪椅待模型（見 D5） |

### 架構 / Pipeline

| 項目 | 狀態 | 備註 |
|------|------|------|
| 4路 multistream pipeline | ✓ | `multistream.py` + `camera_worker.py` |
| InferenceBus（batch 推論） | ✓ | 4路×3模型→3次batch GPU呼叫，見效能章節 |
| ByteTrack 整合 | ✓ | Bus 模式：per-camera tracker 狀態隔離 |
| Fire/Smoke 偵測 | ✓ | ROI 控制（無 ROI 則略過），YOLOv8n PyTorch |
| ROI Manager（多邊形 + 全畫面） | ✓ | `[F]` 鍵設全畫面 |
| ROI 雙軌 key（RTSP / Local 各自獨立） | ✓ | `camera_id_rtsp` / `camera_id_local` |
| RTSP 自動探測（`--source auto`） | ✓ | 可達→攝影機；不可達→本地 fallback |
| GStreamer RTSP reader | ✓ | SW(avdec) → HW(nvv4l2) fallback，TCP 模式 |
| TRT engine（Ubuntu） | ✓ | batch=4 imgsz=640 FP16，已驗證 |
| 雙機設定（Mac / Ubuntu） | ✓ | `cameras.local.yaml` gitignore，各機獨立 |
| EventManager（log + snapshot） | ✓ | hook 預留 no-op |
| Per-camera 影片輸出 | ✓ | 只寫推論幀（避免慢動作），fps=src/skip |
| Mosaic 四格合成影片 | ✓ | dev: display loop；op: 獨立 thread |
| Display 流暢度優化 | ✓ | cache last_mosaic，drain queue，sleep(2ms) |
| FPS 量測（rolling window） | ✓ | 1s 滾動視窗；log 含 read_ms / infer_ms |
| Jetson TensorRT 部署 | ⏳ | 架構相同，尚未現場驗證 |

---

## 效能基準（Ubuntu RTX 5060，2026-06-26）

### 優化前 vs 優化後

| 指標 | 優化前 | 優化後 | 說明 |
|------|--------|--------|------|
| fps per camera | 4.9 | **13.9** | 2.8× 提升 |
| 瓶頸 | 相機 5fps RTSP | GPU batch 推論 | 瓶頸已轉移至 GPU（正常） |
| read_ms | 195ms | 2ms | 相機確認為 30fps，GStreamer drop 舊幀 |
| infer_ms（per worker）| 8ms | 77ms | worker 等 Bus 結果（含湊幀 32ms + 推論 41ms） |
| GPU 呼叫次數 | 12次/輪（4路×3模型） | 3次/輪（batch×4） | 75% 減少 |
| TRT context 數量 | 8（每路各自載模型） | 2（person + luggage 各1） | 顯著降低 VRAM 與 context 切換 |

### InferenceBus 計時細節

```
avg_infer = 41ms   ← 3 模型 batch×4 的 GPU 時間
avg_wait  = 32ms   ← 等 4 路都提交幀的時間差
per-camera fps = 1 / (41ms + 32ms) ≈ 13.7fps  ✓
```

### 三層優化說明

| 層級 | 內容 | 效果 |
|------|------|------|
| Level 1 | `fall_detector.imgsz: 1280 → 640` | GPU 計算量減少 4× |
| Level 2 | 共用 model instance（1 個 TRT context） | 消除冗餘 context 切換 |
| Level 3 | InferenceBus batch×4 | 4路同幀→單次 GPU 呼叫，吞吐量 ~4× |

### TRT Engine Export 規格

```bash
# 在 Ubuntu/Jetson 上執行（約 5–10 分鐘）
python -c "from ultralytics import YOLO; YOLO('models/fall_detection/yolo12l.pt').export(format='engine', batch=4, imgsz=640, device=0, half=True)"
python -c "from ultralytics import YOLO; YOLO('models/luggage/yolo_luggage_best.pt').export(format='engine', batch=4, imgsz=640, device=0, half=True)"
# fire_smoke 使用 .pt（PyTorch 支援動態 batch，無需 export）
```

> **注意**：Engine 與 GPU 架構綁定，每台機器須各自 export。

---

## 已知 Bug

| 編號 | 檔案 | 說明 | 嚴重度 | 狀態 |
|------|------|------|--------|------|
| B4 | `pose_detector.py:4` | 引入 FallDetector 只為借用 `_build_gamma_lut`，耦合不當 | Low | ⏳ |
| B5 | `luggage_detector.py` | `_nearest_person` 不排除 `fallen=True` 的人 | Low | ⏳ |

---

## 討論議題

### D1：滯留偵測 Re-ID 策略

**現況**：ByteTrack 有效重入窗口約 **2 秒**（`max_disappeared_frames=30` ÷ `effective_fps=15`）。

| 選項 | 做法 | 代價 |
|------|------|------|
| A 接受歸零（現行） | 不改動 | 長時間離開後回來不計入 |
| B 延長 ByteTrack 保留 | `max_disappeared_frames` 拉到 150+ | Ghost track 風險 |
| C 外觀 ReID | BoT-SORT + OSNet embedding | 效能增加，需額外模型 |

**待決定**：業務上「人離開再回來」是否要繼續計時？

---

### D2：PoseDetector 電扶梯角度整合

**現況**：`cameras.yaml` 的 `escalator_angle_deg`（±30°）已定義，但 `PoseDetector` 尚未使用。
目前以垂直軸為基準判斷跌倒，電扶梯正常乘客可能誤判。

**待做**：
- `camera_worker.py` 讀取 `escalator_angle_deg` 並傳給 `PoseDetector`
- `PoseDetector` 以電扶梯法線方向取代垂直軸計算夾角

---

### D5：輪椅偵測模型

**現況**：功能5 大件行李可偵測，輪椅因 COCO 無此類別而未實作。

**選項**：
- 從 Roboflow Universe 取得 wheelchair detection weight
- 或等另一組 finetune 完成

---

## 下一步（優先順序）

1. **端對端測試**：接上 4 台真實 30fps 攝影機，驗證 ROI / 偵測 / 告警完整流程
2. **Jetson Orin 部署驗證**：在 Jetson 上 export engine + 測量效能（架構相同，應可直接移植）
3. **整合 escalator_angle_deg**（D2）：補正電扶梯傾斜誤判
4. **決定 Re-ID 策略**（D1）
5. **EventManager hook 實作**：喇叭告警 / API 推播

---

## 預留擴充介面

| Hook | 位置 | 說明 |
|------|------|------|
| 喇叭告警 | `EventManager._on_alert_hook()` | 目前 no-op |
| 事件錄影 | `camera_worker._on_event_record()` | 目前 no-op |
| API/WebSocket 推播 | `EventManager._on_remote_notify()` | 目前 no-op |
| 跨攝影機 Re-ID | `camera_worker` cross_cam_reid 參數 | 目前 disabled |
