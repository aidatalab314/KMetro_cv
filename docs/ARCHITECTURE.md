# KMetro CV — 系統架構

## 硬體環境

| 角色 | 機器 | 用途 |
|------|------|------|
| 開發機 | Apple M1 Mac mini（16GB） | 演算法開發、本地影片測試 |
| 推論機 | Ubuntu 24.04 + GeForce RTX 5060 | RTSP 整合測試、TensorRT 推論 |

---

## 整體架構

```
4 台 RTSP 攝影機
  │
  ├─ cam_platform_north  → CameraWorker Thread ─┐
  ├─ cam_escalator_up    → CameraWorker Thread ─┤  submit frame
  ├─ cam_escalator_down  → CameraWorker Thread ─┤──→ InferenceBus ──→ GPU (serial×4)
  └─ cam_elevator_lobby  → CameraWorker Thread ─┘  future.result()
                                                 │
                          multistream.py ─────── frame queues
                          ├─ Dev: Display Loop        (consume + show)
                          │   └─ Mosaic Writer         (4-panel video)
                          └─ Op:  Mosaic Thread        (consume + record)
```

---

## 模組結構

```
src/
├── pipeline/
│   ├── multistream.py      # 主入口：啟動 InferenceBus + workers、顯示迴圈、mosaic 錄影
│   └── camera_worker.py    # 單攝影機執行緒（preprocess → bus.submit → parse → 功能模組）
├── inference/
│   └── inference_bus.py    # 共用 GPU 推論 hub（串行模式，ByteTrack per-camera 完全隔離）
├── detection/
│   ├── fall_detector.py    # YOLO12l 人員偵測：preprocess() / parse_result() / detect()
│   ├── luggage_detector.py # 行李 YOLO + 大小分類：parse_result() / detect()
│   ├── fire_smoke_detector.py  # YOLOv8n 火煙偵測：parse_result() / detect()
│   └── pose_detector.py    # RTMO bottom-up 姿態估計（跌倒補位，直接模式）
├── features/
│   ├── dwell_monitor.py    # ByteTrack ID + ROI 滯留計時 + 告警 + inherit_timer
│   ├── reid_gallery.py     # 跨攝影機外觀特徵庫（OSNet embedding + TTL）
│   ├── zone_counter.py     # ROI 內人數計數（每 N 秒）
│   └── luggage_roll_monitor.py  # 行李速度向量 + independence check
├── roi/
│   └── roi_manager.py      # ROI 互動繪製、I/O、多邊形 + 全畫面模式
├── event/
│   └── event_manager.py    # 事件 log + snapshot + 預留 hook
├── rtsp_reader.py          # 影像來源抽象（本地/RTSP，GStreamer SW→HW fallback）
└── utils.py                # YAML loader（.local.yaml 覆蓋）+ logger
```

---

## 每幀處理流程（CameraWorker._process）

### Bus 模式（Ubuntu / 部署，預設）

```
RTSPReader.read()  →  [clean frame]
     │
     ├─ FallDetector.preprocess()         # CLAHE + Gamma（CPU，在 worker 執行）
     │
     ├─ InferenceBus.submit(preprocessed, orig)
     │        ↓  等待 future.result()
     │   InferenceBus（單一 GPU 執行緒）：
     │     ① 收集 N 路幀（最多等 40ms）
     │     ② 串行逐路：person_model.track([fi])   # 每路各自獨立的 ByteTrack 狀態
     │     ③ 串行逐路：luggage_model.track([fi])
     │     ④ 串行逐路：fire_model([fi])            # YOLOv8n .pt
     │     ⑤ 分發 Results 給各 worker future
     │
     │   ⚠ batch×4 架構保留程式碼但停用（_seq_mode=True）：
     │     model.track([f0,f1,f2,f3]) 使用單一全域 ByteTracker，跨攝影機畫面
     │     混入同一流 → track_id 大量變 -1 → dwell/fall/luggage_roll 全部失效
     │
     ├─ parse_result()                     # CPU，各 worker 獨立解析
     │    FallDetector.parse_result(raw["person"])
     │    LuggageDetector.parse_result(raw["luggage"], h, w, persons)
     │    FireSmokeDetector.parse_result(raw["fire"])   # 僅有 ROI 時
     │
     ├─ ROI 過濾（per feature）
     │    _in_roi(dets, feature)：
     │      有 ROI 設定 → 只回傳 ROI 內偵測結果
     │      無 ROI 設定 → 回傳空 list（不執行此功能）
     │
     ├─ 功能模組
     │    DwellMonitor.update()        → 超時告警
     │    ZoneCounter.update()         → 人數 + 人潮告警
     │    FallDetector.compute_alert() → 跌倒告警
     │    LuggageRollMonitor.update()  → 滾落告警
     │    LuggageDetector（size）      → 大件告警
     │    FireSmokeDetector.compute_alerts() → 火煙告警
     │
     ├─ ROI.draw() + 各功能 draw()  →  annotated frame
     │
     └─ EventManager.trigger()  →  log + snapshot（+ 預留 hook）
          VideoWriter.write()   →  per-camera 輸出影片
          Queue.put()           →  display / mosaic thread
```

### 直接模式（Mac 開發 / bus=None）

Bus 不存在時，`_process()` 改為各偵測器直接 `detect()` → 行為與舊版相同。

---

## 執行緒模型

| 執行緒 | 數量 | 職責 |
|--------|------|------|
| `InferenceBus` | 1 | 湊幀 → batch 推論 → 分發結果（所有路共享） |
| `CameraWorker` | N（每攝影機一條） | 讀幀 → preprocess → 等 Bus → parse → 功能模組 |
| Dev display loop | 1（main thread） | 讀 queue → 組 mosaic → imshow → 可選 mosaic 寫檔 |
| `_MosaicOpThread` | 1（op 模式） | 讀所有 queue → 組 mosaic → 寫檔 |

> Queue maxsize=4，`put_nowait` 滿則丟幀（顯示端不阻塞推論端）。

---

## 推論加速策略

| 平台 | RTSP 解碼 | 推論模式 | imgsz | skip_frames | 實測 fps |
|------|-----------|----------|-------|-------------|---------|
| Mac M1 | FFmpeg（cv2 預設） | PyTorch MPS，直接模式 | 640 | 2 | ~8fps |
| Ubuntu RTX 5060 | GStreamer avdec（SW） | TRT FP16 + InferenceBus 串行 | 640 | 1 | **13–14fps** |

### InferenceBus 效能數字（Ubuntu RTX 5060，2026-06-26）

```
batch#50  avg_infer=41ms  avg_wait=32ms  mode=seq
fps=13.9  read=2ms  infer=77ms   ← GPU 瓶頸（正常），非相機限制

優化前（4.9fps）: read=195ms（相機 5fps RTSP 限制），infer=8ms，GPU 閒置
優化後（13.9fps）: read=2ms（GStreamer drop 舊幀），infer=77ms（等 Bus 結果）
```

> 串行模式下 GPU 呼叫為 4路×3模型=12次，但共用同一 TRT context（Level 2 優化保留），VRAM 佔用不增加。

### TensorRT Engine 規格

| 模型 | 來源 | Export 參數 | 檔案大小 |
|------|------|-------------|---------|
| `yolo12l.engine` | yolo12l.pt | batch=4, imgsz=640, half=True | ~53 MB |
| `yolo_luggage_best.engine` | yolo_luggage_best.pt | batch=4, imgsz=640, half=True | ~7 MB |
| `fire_smoke.pt` | luminous0219/fire-and-smoke-detection-yolov8 | YOLOv8n，PyTorch（不需 TRT） | ~6 MB |

> **注意**：engine 需在目標機器上 export（TRT engine 與 GPU 架構綁定）。
> Export 指令詳見 `docs/INSTALL_UBUNTU.md`。

---

## 輸出管道

| 輸出 | 路徑 | 說明 |
|------|------|------|
| 事件 log | `data/logs/events_{cam}_{ts}.txt` | JSON，每事件一行 |
| 事件 snapshot | `data/snapshots/{evt}_{cam}_{ts}.jpg` | 告警當幀截圖 |
| Per-camera 影片 | `outputs/videos/{cam}_{ts}.mp4` | 標注後影片（可選） |
| Mosaic 影片 | `outputs/videos/mosaic_{ts}.mp4` | 4 格合成影片（可選） |

---

## 雙機開發流程

| 項目 | Mac M1（開發） | Ubuntu RTX 5060（推論） |
|------|--------------|------------------------|
| 設定檔 | `configs/cameras.local.yaml` | `configs/cameras.local.yaml` |
| 模型格式 | `.pt`（PyTorch MPS） | `.engine`（TRT FP16） |
| 推論模式 | 直接模式（bus=None） | InferenceBus 串行（seq） |
| 影像來源 | `--source local`（本地影片） | `--source rtsp`（RTSP 攝影機） |
| Re-ID | 不支援（bus=None → gallery=None） | 啟用（`reid.enabled: true`） |
| 設定同步 | `cameras.local.yaml` 不進版控，各機獨立 | 同左 |

```bash
# 兩端同步方式
git pull   # 拉程式碼更新（不影響 local.yaml）

# Mac 啟動
python src/pipeline/multistream.py --source local

# Ubuntu 啟動
python src/pipeline/multistream.py --source rtsp --mode op
```

---

## 跨攝影機 Re-ID 架構（Ubuntu 推論機）

```
Cam1 Zone0：person A 滯留 ≥ 60s
  ├─ extract_reid(crop) → OSNet-ain embedding (512d)
  └─ ReIDGallery.enroll(emb, first_dwell_time, cam1) → gid=0

person A 移動到 Cam2 Zone1（ByteTrack 分配新 local tid=99）
  ├─ query ReIDGallery with crop embedding
  ├─ cosine_sim > 0.75 → match gid=0
  └─ DwellMonitor.inherit_timer(99, first_dwell_time) → 立即觸發告警
```

| 元件 | 功能 |
|------|------|
| `OSNet-ain-x1_0` | 外觀 embedding 提取（512d，跨視角設計） |
| `ReIDGallery` | thread-safe 特徵庫，cosine 比對，TTL 自動清除 |
| `enroll()` | 首次達滯留門檻時入庫 |
| `query()` | 新人入 dwell ROI 時查詢是否為跨鏡滯留者 |
| `update_embedding()` | 每 10s 滑動平均更新（應對外觀變化） |

> ReID 僅在 Ubuntu 推論模式（InferenceBus）下啟用。Mac 直接模式 bus=None → gallery=None → 自動略過。
