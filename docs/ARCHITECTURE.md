# KMetro CV — 系統架構

## 硬體環境

| 角色 | 機器 | 用途 |
|------|------|------|
| 開發機 | Apple M1 Mac mini（16GB） | 演算法開發、本地影片測試 |
| 部署機 | Jetson Orin | 現場 RTSP 推論，TensorRT 加速 |

---

## 整體架構

```
4 台 RTSP 攝影機
  │
  ├─ cam_platform_north  → CameraWorker Thread ─┐
  ├─ cam_escalator_up    → CameraWorker Thread ─┤
  ├─ cam_escalator_down  → CameraWorker Thread ─┤  frame queues
  └─ cam_elevator_lobby  → CameraWorker Thread ─┤
                                                 │
                          multistream.py ────────┤
                          ├─ Dev: Display Loop   │ (consume + show)
                          │   └─ Mosaic Writer   │ (4-panel video)
                          └─ Op:  Mosaic Thread ─┘ (consume + record)
```

---

## 模組結構

```
src/
├── pipeline/
│   ├── multistream.py      # 主入口：啟動 workers、顯示迴圈、mosaic 錄影
│   ├── camera_worker.py    # 單攝影機執行緒（推論 + 功能模組）
│   └── run_inference.py    # 舊版單影片模式（backward compat，保留）
├── detection/
│   ├── fall_detector.py    # YOLO12l 人員偵測 + 跌倒 aspect ratio 判斷
│   ├── luggage_detector.py # 行李 YOLO + 大小分類（person_ratio）
│   └── pose_detector.py    # RTMO bottom-up 姿態估計（跌倒補位）
├── features/
│   ├── dwell_monitor.py    # ByteTrack ID + ROI 滯留計時 + 告警
│   ├── zone_counter.py     # ROI 內人數計數（每 N 秒）
│   └── luggage_roll_monitor.py  # 行李速度向量 + independence check
├── roi/
│   └── roi_manager.py      # ROI 互動繪製、I/O、多邊形 + 全畫面模式
├── event/
│   └── event_manager.py    # 事件 log + snapshot + 預留 hook
├── rtsp_reader.py          # 影像來源抽象（本地/RTSP，GStreamer HW fallback）
└── utils.py                # YAML loader（.local.yaml 覆蓋）+ logger
```

---

## 每幀處理流程（CameraWorker._process）

```
RTSPReader.read()  →  [clean frame（未標注）]
     │
     ├─ FallDetector.detect()    → all_persons（含 track_id, fallen, bbox）
     └─ LuggageDetector.detect() → all_luggage（含 track_id, size, bbox）
     │
     ├─ ROI 過濾（per feature）
     │    _in_roi(dets, feature)：
     │      有 ROI 設定 → 只回傳 ROI 內的偵測結果
     │      無 ROI 設定 → 回傳空 list（不執行此功能）
     │      全畫面 ROI（[F] 設定）→ 全部回傳
     │
     ├─ Pose 補位（fall_detector ROI 內無 YOLO person 時）
     │    PoseDetector.detect() → skeleton → 跌倒角度判斷
     │
     ├─ 功能模組
     │    DwellMonitor.update()       → 超時告警
     │    ZoneCounter.update()        → 人數 + 人潮告警
     │    FallDetector.compute_alert() → 跌倒告警
     │    LuggageRollMonitor.update() → 滾落告警
     │    LuggageDetector（size）     → 大件告警
     │
     ├─ ROI.draw() + 各功能 draw()  →  annotated frame
     │
     └─ EventManager.trigger()  →  log + snapshot（+ 預留 hook）
          VideoWriter.write()   →  per-camera 輸出影片
          Queue.put()           →  display / mosaic thread
```

---

## 執行緒模型

| 執行緒 | 數量 | 職責 |
|--------|------|------|
| `CameraWorker` | N（每攝影機一條） | 讀幀 → 推論 → 功能模組 → 寫幀到 queue |
| Dev display loop | 1（main thread） | 讀 queue → 組 mosaic → imshow → mosaic 寫檔 |
| `_MosaicOpThread` | 1（op 模式） | 讀所有 queue → 組 mosaic → 寫檔 |

> queue maxsize=2，`put_nowait` 滿則丟幀，顯示端不阻塞推論端。

---

## 輸出管道

| 輸出 | 路徑 | 說明 |
|------|------|------|
| 事件 log | `data/logs/events_{cam}_{ts}.txt` | JSON，每事件一行 |
| 事件 snapshot | `data/snapshots/{evt}_{cam}_{ts}.jpg` | 告警當幀截圖 |
| Per-camera 影片 | `outputs/videos/{cam}_{ts}.mp4` | 標注後影片（可選） |
| Mosaic 影片 | `outputs/videos/mosaic_{ts}.mp4` | 4 格合成影片（可選） |

---

## 推論加速策略

| 平台 | RTSP 解碼 | 模型推論 | skip_frames |
|------|-----------|----------|-------------|
| Mac M1 | FFmpeg（cv2 預設） | PyTorch MPS | 2 |
| Jetson Orin | GStreamer nvv4l2decoder（H.265 HW） | TensorRT `.engine` | 1 |

Jetson TensorRT 轉換：

```python
from ultralytics import YOLO
YOLO("models/fall_detection/yolo12l.pt").export(format="engine", device=0)
# 產生 models/fall_detection/yolo12l.engine
# 在 cameras.local.yaml 改用 .engine 路徑
```
