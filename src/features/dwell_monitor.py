import time
import cv2
import numpy as np


class DwellMonitor:
    """
    ROI 內旅客滯留監測。
    依賴 ByteTrack 提供的 track_id，計算每個 ID 在 ROI 內的累計停留時間。

    Grace period 設計：
      - 人短暫離開 ROI（track_id 仍存在）→ 計時暫停，回來後繼續
      - ByteTrack 把 track_id 刪掉（真正離開）→ grace_period 秒後清除計時器
    """

    def __init__(self, alert_seconds: float = 60.0,
                 grace_period_sec: float = 10.0):
        self._alert_sec  = alert_seconds
        self._grace      = grace_period_sec
        # track_id → 首次進入 ROI 的時間
        self._first_seen: dict[int, float] = {}
        # track_id → 上次在 ROI 內的時間（用於 grace period 計算）
        self._last_in_roi: dict[int, float] = {}

    # ── 主更新介面 ───────────────────────────────────────────────────────────

    def update(self, persons_in_roi: list[dict]) -> list[dict]:
        """
        傳入當前幀 ROI 內的追蹤人員列表，回傳超過滯留門檻的人員列表。
        每個 dict 額外帶有 'dwell_seconds' 欄位。
        """
        now = time.time()
        current_ids = set()

        for p in persons_in_roi:
            tid = p.get("track_id", -1)
            if tid < 0:
                continue
            current_ids.add(tid)
            if tid not in self._first_seen:
                self._first_seen[tid] = now
            self._last_in_roi[tid] = now

        # 清除超過 grace period 且已離開 ROI 的 track
        for tid in list(self._first_seen):
            if tid not in current_ids:
                if now - self._last_in_roi.get(tid, 0) > self._grace:
                    del self._first_seen[tid]
                    self._last_in_roi.pop(tid, None)

        # 回傳超過門檻的人
        alerts = []
        for p in persons_in_roi:
            tid = p.get("track_id", -1)
            if tid < 0:
                continue
            dwell = self.get_dwell(tid)
            if dwell >= self._alert_sec:
                alerts.append({**p, "dwell_seconds": dwell})

        return alerts

    def get_dwell(self, track_id: int) -> float:
        """回傳指定 track_id 在 ROI 內的滯留秒數（0.0 if unknown）。"""
        if track_id not in self._first_seen:
            return 0.0
        return time.time() - self._first_seen[track_id]

    def inherit_timer(self, track_id: int, first_seen_wall_time: float):
        """
        跨攝影機 Re-ID 匹配後繼承原始計時起點。
        僅在本機尚未記錄此 track_id 時生效（避免覆蓋已有計時）。
        """
        if track_id not in self._first_seen:
            self._first_seen[track_id]  = first_seen_wall_time
            self._last_in_roi[track_id] = time.time()

    # ── 繪製 ─────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, persons_in_roi: list[dict],
             reid_gids: "dict[int, int] | None" = None):
        """
        在每個追蹤人員的 bbox 上繪製 ID + 滯留時間。
        顏色：綠 = 正常 / 橘 = 接近門檻(>50%) / 紅 = 超過門檻
        reid_gids: local track_id → global_id 映射（有值時顯示 G:XXX，並加 ← 跨鏡標記）
        """
        for p in persons_in_roi:
            tid = p.get("track_id", -1)
            if tid < 0:
                continue

            dwell  = self.get_dwell(tid)
            x1, y1, x2, y2 = p["bbox"]

            if dwell >= self._alert_sec:
                color = (0, 0, 255)      # 紅：超過門檻
            elif dwell >= self._alert_sec * 0.5:
                color = (0, 140, 255)    # 橘：接近門檻
            else:
                color = (0, 220, 0)      # 綠：正常

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            gid = reid_gids.get(tid) if reid_gids else None
            if gid is not None:
                label = f"G:{gid:03d}  {dwell:.0f}s"
            else:
                label = f"ID:{tid}  {dwell:.0f}s"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            # 跨鏡繼承標記（右下角小字）
            if gid is not None:
                mark = "X-CAM"
                (mw, mh), _ = cv2.getTextSize(mark, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(frame, (x2 - mw - 4, y2 - mh - 4), (x2, y2),
                              (0, 0, 180), -1)
                cv2.putText(frame, mark, (x2 - mw - 2, y2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
