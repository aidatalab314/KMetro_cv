# KMetro CV — 設定說明

## 設定檔結構

```
configs/
├── cameras.yaml              # 主設定（版控）
├── cameras.local.yaml        # 本機覆蓋（gitignore，各平台自建）
├── cameras.local.yaml.example# 平台設定範本
├── model_config.yaml         # 舊版單影片模式設定（保留）
└── roi_records.json          # ROI 座標紀錄（自動寫入）
                              # key 格式：{camera_id}_rtsp / {camera_id}_local
                              # RTSP 與本地影片各自獨立，互不覆蓋
```

`utils.load_yaml()` 會自動合併 `cameras.yaml` 和 `cameras.local.yaml`（後者覆蓋前者）。

---

## cameras.yaml 完整結構

### 攝影機（cameras）

```yaml
cameras:
  - id: cam_platform_north        # 唯一 ID（ROI 紀錄的 key）
    name: "北月台候車區"           # 顯示名稱（允許中文）
    source: "rtsp://..."          # RTSP URL（op 模式 / --source rtsp）
    fallback: "data/test_videos/metro/IMG_2733.MOV"  # dev 模式本地影片
    escalator_angle_deg: 30.0     # 電扶梯傾斜角（±，僅電扶梯攝影機需要）
    features:
      dwell_monitor:   true       # 功能1：滯留通報
      fire_smoke:      true       # 功能2：火光煙霧
      zone_counter:    true       # 功能3：人流計數
      fall_detector:   true       # 功能4a：跌倒偵測
      luggage_roll:    true       # 功能4b：行李滾落
      size_classifier: true       # 功能5：大件行李
```

### 模型路徑（models）

```yaml
models:
  person:     "models/fall_detection/yolo12l.pt"
  luggage:    "models/luggage/yolo_luggage_best.pt"
  fire_smoke: "models/fire_smoke/fire_smoke.pt"   # 待取得
```

### 推論設定（detector）

```yaml
detector:
  conf:         0.4     # YOLO 信心門檻
  device:       "mps"   # Mac→"mps" | Ubuntu RTX (CUDA)→"0" | CPU→"cpu"
  imgsz:        640     # 推論解析度（640 平衡 / 1280 細節）
  skip_frames:  2       # 每 N 幀推論一次（減少計算量）
```

### 功能閾值（features）

```yaml
features:

  dwell_monitor:
    alert_seconds:          60.0   # 滯留超過此秒數 → 告警
    max_disappeared_frames: 30     # ByteTrack：ID 消失幾幀後才刪除
                                   # 有效重入窗口 = 此值 ÷ effective_fps

  zone_counter:
    interval_seconds:   15         # 每 N 秒計數一次
    crowd_alert_count:  20         # 人數超過此值 → 人潮告警

  fire_smoke:
    conf:         0.5              # 火煙偵測信心門檻
    alert_frames: 3                # 連續幾幀才告警

  fall_detector:
    fallen_aspect_ratio: 1.2       # bbox 寬/高 > 此值視為橫倒
    alert_frames:        5         # 連續幾幀才發告警
    clahe:               true      # CLAHE 對比增強（暗區建議開）
    gamma:               0.5       # Gamma 校正（<1 提亮，1.0=關閉）
    imgsz:               1280      # 跌倒偵測建議較高解析度
    pose_fallback:
      enabled:        false        # RTMO 補位偵測（效能較重，預設關）
      mode:           "balanced"   # lightweight / balanced / performance
      kp_conf:        0.3
      min_kp:         3
      fall_angle_deg: 50.0         # 肩-髖向量偏離垂直軸超過此角 → 跌倒

  luggage_roll:
    speed_threshold_px_per_frame:        15
    independence_threshold_px_per_frame: 10
    alert_frames:                        5

  size_classifier:
    size_method:             "person_ratio"   # 主要方法
    large_person_area_ratio: 0.22
    max_match_distance_px:   400
    large_area_ratio:        0.01             # frame_ratio 備援門檻
```

### 顯示設定（display）

```yaml
display:
  panel_h: 540     # 每路攝影機顯示高度（px），4路並排時的每格高度
  window:  "KMetro CV — Multistream"
```

### 輸出設定（output）

```yaml
output:
  snapshot_dir:     "data/snapshots"
  log_dir:          "data/logs"
  video_dir:        "outputs/videos"
  save_snapshots:   true
  save_video_local: true    # 本地影片來源 → 強制存 per-camera 影片
  save_video_rtsp:  false   # RTSP 來源 → 預設不存（可覆蓋）
  save_mosaic:      true    # 四格合成影片
  mosaic_fps:       15      # ⚠️ 必須 = source_fps ÷ skip_frames
                            # 30fps ÷ 2 = 15；設錯會導致慢動作或快轉
```

---

## 平台設定（cameras.local.yaml）

不同平台的差異透過 `cameras.local.yaml` 覆蓋，此檔案不進版控。

```bash
cp configs/cameras.local.yaml.example configs/cameras.local.yaml
```

### Mac M1（開發）

```yaml
detector:
  device: "mps"
  skip_frames: 2

features:
  fall_detector:
    imgsz: 640    # 覆蓋預設 1280（Mac MPS 效能考量）

output:
  save_video_rtsp:  false
  save_video_local: true
  mosaic_fps:       15    # 30fps ÷ skip_frames(2) = 15
```

### Ubuntu RTX 5060（推論機）

```yaml
detector:
  device:      "0"     # CUDA GPU 0
  skip_frames: 1       # TRT 夠快，不需跳幀

models:
  person:  "models/fall_detection/yolo12l.engine"      # TensorRT FP16
  luggage: "models/luggage/yolo_luggage_best.engine"   # TensorRT FP16

features:
  fall_detector:
    imgsz: 640

output:
  save_video_rtsp: false
  save_video_local: false
  mosaic_fps:      30    # 30fps ÷ skip_frames(1) = 30

reid:
  enabled: true          # 啟用跨攝影機 Re-ID（需 torchreid 已安裝）
```

---

## 平台對照表

| 項目 | Mac M1（開發） | Ubuntu RTX 5060（推論） |
|------|--------|-------------|
| `device` | `"mps"` | `"0"` |
| RTSP 解碼 | FFmpeg（cv2 預設） | GStreamer avdec（SW） |
| 模型格式 | `.pt`（PyTorch MPS） | `.engine`（TensorRT FP16） |
| 推論模式 | 直接模式（bus=None） | InferenceBus batch×4 |
| Re-ID | 不支援 | `reid.enabled: true` 啟用 |
| `skip_frames` | 2 | 1 |
| `mosaic_fps` | 15（30fps÷2） | 30（30fps÷1） |

---

## Reid 設定（`reid:`）

跨攝影機滯留計時繼承，需搭配 Ubuntu 推論模式（InferenceBus）使用。

```yaml
reid:
  enabled:             false   # 預設關閉
  sim_threshold:       0.75    # cosine 相似度門檻（0.7=寬鬆，0.80=嚴格）
  ttl_sec:             300.0   # gallery 記錄 TTL（秒）；超過視為人員已離站
  model_path:          ""      # 留空 → ImageNet 預訓練；填路徑 → 指定 checkpoint
  update_interval_sec: 10.0   # 持續滯留者更新 embedding 的間隔（秒）
```

| 參數 | 說明 |
|------|------|
| `enabled` | 開啟後 InferenceBus 載入 OSNet-ain-x1_0 模型（~11MB） |
| `sim_threshold` | cosine similarity 超過此值才判定為同一人；建議 0.72–0.80 |
| `ttl_sec` | gallery 記錄存活時間；5 分鐘足以覆蓋站內移動距離 |
| `model_path` | 空 = ImageNet 預訓練；MSMT17 checkpoint 精度更高 |
| `update_interval_sec` | 每 N 秒更新 embedding（防止外觀變化導致後續比對失敗） |

**依賴套件**（Ubuntu 需額外安裝）：

```bash
pip install torchreid gdown
```

---

## --source 參數行為

| `--source` | 行為 | ROI key 後綴 |
|------------|------|-------------|
| `auto`（預設） | 每台攝影機探測 RTSP（6s timeout）；可達 → 攝影機，不可達 → fallback 本地影片 | 依實際使用來源決定（`_rtsp` 或 `_local`） |
| `rtsp` | 強制使用 RTSP | `_rtsp` |
| `local` | 強制使用 fallback 本地影片 | `_local` |

## ROI 紀錄 key 格式

每台攝影機的 ROI 以 `{camera_id}_{source_type}` 為 key 寫入 `roi_records.json`，RTSP 與本地影片各自儲存，切換來源不覆蓋對方：

```json
{
  "cam_platform_north_rtsp":  [...],   // --source rtsp 時使用
  "cam_platform_north_local": [...],   // --source local 時使用
  "cam_escalator_up_rtsp":    [...]
}
```

首次以某模式啟動時，若對應 key 不存在，系統自動開啟互動式 ROI 繪製視窗。

## Dev / Op 模式對照

| 項目 | Dev | Op |
|------|-----|----|
| 顯示視窗 | ✓ split-screen mosaic | ✗ headless |
| 預設影像來源（`auto`） | 探測 RTSP，不可達則本地影片 | 探測 RTSP，不可達則本地影片 |
| 本地影片 → 存影片 | 強制 | 強制 |
| RTSP → 存影片 | 依 `save_video_rtsp` | 依設定 |
| Mosaic 錄影 | Display loop 直接寫 | 獨立 `_MosaicOpThread` |
| Debug overlay | 右上角 FPS | 不顯示 |
