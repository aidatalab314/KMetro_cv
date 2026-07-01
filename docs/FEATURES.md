# KMetro CV — 功能說明

## 功能總覽

| ID | 功能 | 攝影機 | 狀態 |
|----|------|--------|------|
| 功能1 | 月台旅客滯留通報 | cam_platform_north | ✓ 實作 |
| 功能2 | 火光煙霧偵測 | 全部 | ⏳ 待模型 |
| 功能3 | 區域人流偵測 | 全部 | ✓ 實作 |
| 功能4a | 電扶梯人員跌倒 | cam_escalator_up/down | ✓ 實作 |
| 功能4b | 行李箱滾落偵測 | cam_escalator_up/down | ✓ 實作 |
| 功能5 | 大件行李／輪椅偵測 | cam_elevator_lobby | ✓ 實作 |

---

## 功能 1：旅客滯留通報

**觸發條件**：同一 track_id 在 ROI 內累計停留超過 `alert_seconds`（預設 60 秒）。

**算法**：
```
ByteTrack 指派 track_id
  → DwellMonitor 記錄 first_seen[id]
  → 每幀更新 last_in_roi[id]
  → dwell_seconds = now - first_seen[id]
  → dwell_seconds >= alert_seconds → 告警
```

**Grace period（`grace_period_sec=10`）**：
人短暫離開 ROI 但 track_id 仍存在（ByteTrack 未刪除）→ 計時暫停，回來繼續。
ByteTrack 刪除 ID 後 10 秒 → DwellMonitor 清除該 ID 計時器。

**Re-ID 限制**：
ByteTrack 的 `max_disappeared_frames`（預設 30）決定 ID 保留時間。
有效重入窗口 ≈ `max_disappeared_frames ÷ effective_fps`（目前約 2 秒）。
超過後回來分配新 ID，計時從零開始。詳見 [STATUS.md](STATUS.md) 討論議題。

**ROI 需求**：需設定多邊形 ROI（代表監控等候區域）。

**關鍵參數**（`cameras.yaml`）：
```yaml
features:
  dwell_monitor:
    alert_seconds:          60.0   # 滯留告警門檻
    max_disappeared_frames: 30     # ByteTrack ID 保留幀數
    grace_period_sec:       10.0   # DwellMonitor ID 清除緩衝
```

---

## 功能 2：火光煙霧偵測

**觸發條件**：fire/smoke 偵測結果連續 `alert_frames` 幀 conf ≥ 設定門檻。

**模型**：需獨立 YOLO 模型（D-Fire 資料集訓練）。
放置路徑：`models/fire_smoke/fire_smoke.pt`。
（COCO 80 類別不含 fire/smoke，不可用通用模型。）

**ROI 需求**：建議設為**全畫面**（ROI 設定介面按 `[F]`），整個畫面都是偵測範圍。

**關鍵參數**：
```yaml
features:
  fire_smoke:
    conf:         0.5
    alert_frames: 3
```

---

## 功能 3：區域人流偵測

**觸發條件**：ROI 內即時人數每 `interval_seconds` 秒輸出一次；超過 `crowd_alert_count` → 人潮告警。

**算法**：純計數（無 tracker），ROI 內 person bbox 數量。不需要 ID 連續性。

**ROI 需求**：設定多邊形 ROI（代表計數區域）。可設多個 ROI 分區計數。

**關鍵參數**：
```yaml
features:
  zone_counter:
    interval_seconds:   15
    crowd_alert_count:  20
```

---

## 功能 4a：電扶梯人員跌倒

**觸發條件**：ROI 內有 person bbox 的 w/h ≥ `fallen_aspect_ratio` 且持續 `alert_frames` 幀。

**雙軌架構**：
```
YOLO12l 偵測 person → ROI 過濾
  有人 → aspect ratio 判斷（主要）
  無人 → RTMO bottom-up 姿態估計（補位，處理遮蔽暗區）
    └→ 肩-髖向量與垂直軸夾角 > fall_angle_deg → 跌倒
```

**電扶梯角度**：電扶梯傾斜約 ±30°，`cameras.yaml` 有 `escalator_angle_deg` 欄位。
> ⚠️ 目前 PoseDetector 尚未整合此參數（見 STATUS.md）。

**ROI 需求**：設定多邊形 ROI 覆蓋電扶梯區域。

**關鍵參數**：
```yaml
fall_detector:
  fallen_aspect_ratio: 1.2    # bbox 寬/高比門檻
  alert_frames:        5
  imgsz:               1280   # 跌倒偵測建議用較高解析度
  pose_fallback:
    enabled:        false     # RTMO 補位（預設關閉，效能較重）
    fall_angle_deg: 50.0
```

---

## 功能 4b：行李箱滾落偵測

**觸發條件**：行李在電扶梯上獨立移動（不跟隨旁邊乘客），持續 `alert_frames` 幀。

**算法**：
```
YOLO 偵測 suitcase → ByteTrack 速度向量
  ① 行李本身速度 > speed_threshold_px_per_frame
  ② 行李速度 - 最近人物速度 > independence_threshold
  ① + ② 同時成立 → 滾落告警
```

**ROI 需求**：設定多邊形 ROI 覆蓋電扶梯。

**關鍵參數**：
```yaml
luggage_roll:
  speed_threshold_px_per_frame:        15
  independence_threshold_px_per_frame: 10
  alert_frames:                        5
```

---

## 功能 5：大件行李／輪椅偵測

**觸發條件**：行李 bbox 面積超過人體 bbox 面積的 `large_person_area_ratio` 倍。

**大小判斷邏輯**：
```
主要（person_ratio）：
  ratio = 行李 bbox 面積 / 最近人物 bbox 面積
  ratio >= 0.22 → 大件

備援（frame_ratio，無人或人距 > 400px 時）：
  ratio = 行李 bbox 面積 / 畫面總面積
  ratio >= 0.01 → 大件
```

**輪椅**：COCO 無 wheelchair 類別，待自訓或公開模型。

**ROI 需求**：設定多邊形 ROI（代表電梯前乘場區域）。

**關鍵參數**：
```yaml
size_classifier:
  large_person_area_ratio: 0.22
  max_match_distance_px:   400
  large_area_ratio:        0.01
```

---

## ROI 設定操作指南

### 啟動設定

```bash
# 初次執行自動開啟（無 ROI 紀錄時）
python src/pipeline/multistream.py

# 強制重繪
python src/pipeline/multistream.py --reset-roi cam_platform_north
python src/pipeline/multistream.py --reset-roi all
```

### 操作按鍵

| 按鍵 | 動作 |
|------|------|
| 左鍵點擊 | 新增多邊形頂點 |
| 右鍵點擊 | 刪除最後一個頂點 |
| `C` | 確認當前多邊形（≥3 點）→ 進入功能選擇選單 |
| **`F`** | **全畫面模式**：不畫多邊形，直接選擇功能，整幀為 ROI |
| `R` | 重置當前未完成的多邊形 |
| `ESC` / `Q` | 儲存並結束 |

### 功能選擇選單（`C` 或 `F` 後出現）

| 按鍵 | 動作 |
|------|------|
| 數字鍵 `1`-`6` | 切換對應功能開/關 |
| `A` | 全選所有功能 |
| `C` | 清除所有選取 |
| `Enter` / `Space` | 確認（未選則預設全選） |
| `ESC` | 取消此 ROI |

### 全畫面模式（`F` 鍵）

適用功能：火光煙霧偵測（整個畫面都是偵測範圍，不需要框選特定區域）。
顯示方式：畫面邊緣彩色邊框 + 左上角 `[Full Frame][feat]` 標籤。

### ROI 與功能的對應關係

```
有 ROI 設定（多邊形或全畫面）→ 只在 ROI 內執行對應功能
無 ROI 設定 → 此功能不執行（不顯示偵測框）
```

一個攝影機可設多個 ROI，每個 ROI 可指定不同功能組合。

### 儲存格式（`configs/roi_records.json`）

key 格式為 `{camera_id}_rtsp`（RTSP 模式）或 `{camera_id}_local`（本地影片模式），兩者互不覆蓋。
此檔案已加入 `.gitignore`，各機器自行維護，不進版控。

```json
{
  "cam_platform_north_rtsp": [
    {
      "id": "roi_0",
      "label": "ROI 0",
      "features": ["dwell_monitor", "zone_counter"],
      "color": [0, 255, 255],
      "full_frame": false,
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    },
    {
      "id": "roi_1",
      "label": "Full Frame",
      "features": ["fire_smoke"],
      "color": [255, 80, 0],
      "full_frame": true,
      "points": [[0,0], [W,0], [W,H], [0,H]]
    }
  ],
  "cam_platform_north_local": [...]
}
```
