# KMetro CV — 專案現況

> 最後更新：2026-06-23

---

## 實作狀態

### 功能模組

| 功能 | 模組 | 狀態 | 備註 |
|------|------|------|------|
| 功能1 旅客滯留 | `DwellMonitor` + ByteTrack | ✓ | Re-ID 限制見討論議題 |
| 功能2 火光煙霧 | `camera_worker._fire_det` | ⏳ | 等待 fire_smoke.pt 模型 |
| 功能3 人流偵測 | `ZoneCounter` | ✓ | |
| 功能4a 電扶梯跌倒 | `FallDetector` + `PoseDetector` | ✓ | escalator_angle 尚未整合 |
| 功能4b 行李箱滾落 | `LuggageRollMonitor` + ByteTrack | ✓ | |
| 功能5 大件行李 | `LuggageDetector` size_classifier | ✓ | 輪椅待模型 |

### 架構 / Pipeline

| 項目 | 狀態 | 備註 |
|------|------|------|
| 4路 multistream pipeline | ✓ | `multistream.py` + `camera_worker.py` |
| ByteTrack 整合 | ✓ | FallDetector + LuggageDetector tracking=True |
| ROI Manager（多邊形 + 全畫面） | ✓ | `[F]` 鍵設全畫面，已更新 roi_records 格式 |
| ROI 雙軌 key（RTSP / Local 各自獨立） | ✓ | `camera_id_rtsp` / `camera_id_local`，不互蓋 |
| RTSP 自動探測（`--source auto`） | ✓ | 探測可達 → 攝影機；不可達 → 本地 fallback |
| ROI 選單顯示全部功能 | ✓ | 不再依 camera 啟用功能過濾選單 |
| 全攝影機啟用全功能 | ✓ | 4 台均開 dwell/zone/fall/luggage/size/fire |
| GStreamer RTSP reader | ✓ | HW/SW fallback，port from csas_poc |
| EventManager（log + snapshot） | ✓ | hook 預留 no-op |
| Per-camera 影片輸出 | ✓ | |
| Mosaic 四格合成影片 | ✓ | dev: display loop 寫入；op: 獨立 thread |
| Jetson TensorRT 部署 | ⏳ | 尚未驗證 |

---

## 已知 Bug

| 編號 | 檔案 | 說明 | 嚴重度 | 狀態 |
|------|------|------|--------|------|
| B1 | `pose_detector.py:33` | `_fall_cos` 計算：已確認使用 `cos(angle)`，無誤 | High | ✓ 已確認正確 |
| B2 | `run_inference.py:117` | ROI overlay 在 YOLO 推論前繪製，影響模型輸入 | Medium | ✓ 新架構 camera_worker 已修正，舊版保留不維護 |
| B4 | `pose_detector.py:4` | 引入 FallDetector 只為借用 `_build_gamma_lut`，耦合不當 | Low | ⏳ |
| B5 | `luggage_detector.py` | `_nearest_person` 不排除 `fallen=True` 的人，bbox 異常影響面積計算 | Low | ⏳ |

---

## 討論議題

### D1：滯留偵測 Re-ID 策略

**現況**：ByteTrack 的有效重入窗口約 **2 秒**（`max_disappeared_frames=30` ÷ `effective_fps=15`）。
超過 2 秒後離開再回來，分配新 ID，滯留計時歸零。

**選項**：

| 選項 | 做法 | 代價 |
|------|------|------|
| A 接受歸零（現行） | 不改動 | 長時間離開後回來不計入 |
| B 延長 ByteTrack 保留 | `max_disappeared_frames` 拉到 150+ | Ghost track 風險（其他人繼承舊 ID） |
| C 外觀 ReID | BoT-SORT + OSNet embedding | 顯著效能增加，需額外模型 |

**待決定**：業務上「人離開月台再回來」是否要繼續計時？還是視為新一次滯留？

---

### D2：PoseDetector 電扶梯角度整合

**現況**：`cameras.yaml` 的 `escalator_angle_deg`（±30°）已定義，但 `PoseDetector` 尚未使用此參數。
目前以垂直軸為基準判斷跌倒，電扶梯正常乘客的肩-髖向量已偏離垂直，可能誤判。

**待做**：
- `camera_worker.py` 讀取 `escalator_angle_deg` 並傳給 `PoseDetector`
- `PoseDetector` 以電扶梯法線方向取代垂直軸計算夾角

---

### D3：Fire Smoke 模型取得

**現況**：`camera_worker` 已預留 `_fire_det` 介面，載入時若模型不存在自動略過。

**待做**：
- 從 HuggingFace / Roboflow 取得 D-Fire 資料集訓練的 YOLOv8 weight
- 放置路徑：`models/fire_smoke/fire_smoke.pt`
- 格式確認：需可被 `ultralytics.YOLO()` 直接載入

---

### D4：Mosaic FPS 設定

**現況**：`mosaic_fps` 必須手動設定為 `source_fps ÷ skip_frames`，設錯會造成播放速度偏差。

| 場景 | source_fps | skip_frames | 正確 mosaic_fps |
|------|------------|-------------|-----------------|
| Mac 開發（現行） | 30 | 2 | **15** |
| Jetson 部署 | 30 | 1 | **30** |
| RTSP 25fps | 25 | 1 | **25** |

**待改善**：可考慮讓 `multistream.py` 自動從 worker 取得 `src_fps` 並計算，免除人工設定。

---

### D5：輪椅偵測模型

**現況**：功能5 大件行李可偵測，輪椅因 COCO 無此類別而未實作。

**選項**：
- 等另一組 finetune 完成（若模型含 wheelchair 類別）
- 或：從 Roboflow Universe 取得 wheelchair detection weight（需確認格式相容性）

---

## 下一步（優先順序）

1. **端對端測試**：以 2 路真實攝影機跑 RTSP 模式，確認 ROI 設定、偵測結果、告警觸發
2. **整合 escalator_angle_deg**（D2）：`camera_worker` 讀取後傳給 FallDetector，補正電扶梯傾斜誤判
3. **取得 fire_smoke 模型**，完成功能2
4. **Jetson TensorRT 部署驗證**：export `.engine` + 效能測試
5. **決定 Re-ID 策略**（D1），視業務需求決定是否升級 BoT-SORT
6. **EventManager hook 實作**：喇叭告警 / API 推播
7. **Mosaic FPS 自動計算**（D4，非緊急）

---

## 預留擴充介面（未來實作）

| Hook | 位置 | 說明 |
|------|------|------|
| 喇叭告警 | `EventManager._on_alert_hook()` | 目前 no-op |
| 事件錄影 | `camera_worker._on_event_record()` | 目前 no-op |
| API/WebSocket 推播 | `EventManager._on_remote_notify()` | 目前 no-op |
| 跨攝影機 Re-ID | `camera_worker` cross_cam_reid 參數 | 目前 disabled |
