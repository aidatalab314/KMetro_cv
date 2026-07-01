# KMetro CV — 專案現況

> 最後更新：2026-07-01

---

## 實作狀態

### 功能模組

| 功能 | 模組 | 狀態 | 備註 |
|------|------|------|------|
| 功能1 旅客滯留 | `DwellMonitor` + ByteTrack | ✓ 已驗（Ubuntu RTSP 單路觸發正常） | |
| 功能1+ 跨攝影機 Re-ID | `ReIDGallery` + OSNet-ain | ⏳ 進行中 | 單路 enroll 確認；跨鏡繼承驗證中，見 D1 |
| 功能2 火光煙霧 | `FireSmokeDetector`（YOLOv8n） | ✓ | luminous0219/fire-and-smoke-detection-yolov8 |
| 功能3 人流偵測 | `ZoneCounter` | ✓ | |
| 功能4a 電扶梯跌倒 | `FallDetector` + `PoseDetector` | ✓ | escalator_angle 尚未整合（見 D2） |
| 功能4b 行李箱滾落 | `LuggageRollMonitor` + ByteTrack | ✓ | |
| 功能5 大件行李 | `LuggageDetector` size_classifier | ✓ | 輪椅待模型（見 D5） |

### 架構 / Pipeline

| 項目 | 狀態 | 備註 |
|------|------|------|
| 4路 multistream pipeline | ✓ | `multistream.py` + `camera_worker.py` |
| InferenceBus（共用推論） | ✓ | 現為串行模式（seq）；batch×4 架構保留但停用，見效能章節 |
| ByteTrack 整合 | ✓ | 串行模式：`_track_one()` save/restore `pred.trackers`，各路狀態完全隔離 |
| Fire/Smoke 偵測 | ✓ | ROI 控制（無 ROI 則略過），YOLOv8n PyTorch |
| ROI Manager（多邊形 + 全畫面） | ✓ | `[F]` 鍵設全畫面 |
| ROI 雙軌 key（RTSP / Local 各自獨立） | ✓ | `camera_id_rtsp` / `camera_id_local` |
| RTSP 自動探測（`--source auto`） | ✓ | 可達→攝影機；不可達→本地 fallback |
| GStreamer RTSP reader | ✓ | SW(avdec) → HW(nvv4l2) fallback，TCP 模式 |
| TRT engine（Ubuntu） | ✓ | batch=4 imgsz=640 FP16，已驗證 |
| 雙機設定（Mac / Ubuntu） | ✓ | `cameras.local.yaml` + `roi_records.json` gitignore，各機獨立；不再衝突 |
| EventManager（log + snapshot） | ✓ | hook 預留 no-op |
| Per-camera 影片輸出 | ✓ | 只寫推論幀（避免慢動作），fps=src/skip |
| Mosaic 四格合成影片 | ✓ | dev: display loop；op: 獨立 thread |
| Display 流暢度優化 | ✓ | cache last_mosaic，drain queue，sleep(2ms) |
| FPS 量測（rolling window） | ✓ | 1s 滾動視窗；log 含 read_ms / infer_ms |
| 跨攝影機 Re-ID（OSNet-ain） | ✓ | `reid.enabled: true` 啟用，Ubuntu 推論機專用 |

---

## 效能基準（Ubuntu RTX 5060，2026-06-26）

### 優化前 vs 優化後

| 指標 | 優化前 | 優化後（串行模式） | 說明 |
|------|--------|--------|------|
| fps per camera | 4.9 | **13.9**（參考值） | 2.8× 提升；來自 Level 1+2 |
| 瓶頸 | 相機 5fps RTSP | GPU 串行推論 | 瓶頸已轉移至 GPU（正常） |
| read_ms | 195ms | 2ms | 相機確認為 30fps，GStreamer drop 舊幀 |
| GPU 呼叫次數 | 12次/輪（4路×3模型各載） | 12次/輪（4路×3模型串行） | 次數相同，但共用 TRT context |
| TRT context 數量 | 8（每路各自載模型） | 2（person + luggage 各1） | 顯著降低 VRAM 與 context 切換 |

> **串行模式原因**：`model.track([cam0,cam1,cam2,cam3])` batch 模式下 Ultralytics 共用單一 ByteTracker，跨攝影機畫面混入同一流 → track_id 大量變 -1，dwell / fall / luggage_roll 全部失效。強制串行後各路 tracker 狀態完全隔離，track_id 正常延續。
>
> batch×4 架構（Level 3）程式碼保留，條件：確認 Ultralytics 支援 per-batch tracker 隔離後可切回（`_seq_mode = False`）。

### 三層優化說明

| 層級 | 內容 | 效果 |
|------|------|------|
| Level 1 | `fall_detector.imgsz: 1280 → 640` | GPU 計算量減少 4× |
| Level 2 | 共用 model instance（1 個 TRT context） | 消除冗餘 context 切換 |
| Level 3 | InferenceBus batch×4（保留，暫停用） | 架構具備；待 tracker 隔離確認後啟用 |

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

**現況**：已實作跨攝影機 Re-ID（`ReIDGallery` + OSNet-ain-x1_0）。

| 層次 | 機制 | 有效範圍 |
|------|------|----------|
| 同路短暫消失 | ByteTrack grace period（`max_disappeared_frames=30`，2 秒窗口） | 遮蔽 / 偵測失敗 |
| 跨路重新入鏡 | OSNet embedding + cosine similarity | 同機 4 路之間 |

**已知限制**：
- 同一路攝影機離開 > 2 秒後回來 → ByteTrack 分配新 ID → Re-ID 查詢補救（若外觀特徵夠穩定）
- embedding 使用 ImageNet 預訓練（非行人資料集）；換 MSMT17 checkpoint 可提升精度
  ```bash
  # 下載 MSMT17 checkpoint（可選）
  wget -P ~/.cache/torch/checkpoints/ \
    https://download.openmmlab.com/mmtracking/reid/tracktor_reid_r50_iter25245.pth
  # 或從 torchreid model zoo 取得 osnet_ain_x1_0_msmt17.pth
  # 取得後在 cameras.local.yaml 設定：reid.model_path: "path/to/osnet_ain_x1_0_msmt17.pth"
  ```

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

1. **跨鏡 Re-ID 驗證**：人員 cam_platform_north → cam_escalator_up 移動，確認第二路出現 `ReID 匹配 gid=xxx` 並繼承計時（X-CAM 標記）
2. **alert_seconds 還原**：驗證完成後 Ubuntu `cameras.local.yaml` 從 5.0 改回 60.0
3. **整合 escalator_angle_deg**（D2）：補正電扶梯傾斜誤判
4. **Re-ID checkpoint 升級**（D1）：換用 MSMT17 預訓練提升跨視角精度
5. **EventManager hook 實作**：喇叭告警 / API 推播

---

## 預留擴充介面

| Hook | 位置 | 說明 |
|------|------|------|
| 喇叭告警 | `EventManager._on_alert_hook()` | 目前 no-op |
| 事件錄影 | `camera_worker._on_event_record()` | 目前 no-op |
| API/WebSocket 推播 | `EventManager._on_remote_notify()` | 目前 no-op |
| 跨攝影機 Re-ID | `camera_worker` cross_cam_reid 參數 | 目前 disabled |
