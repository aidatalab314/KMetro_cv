import time
import cv2
import numpy as np


class ZoneCounter:
    """
    ROI 區域人流計數。
    每 interval_seconds 秒紀錄一次人數（log），
    人數達到 crowd_alert_count 立即觸發告警。
    不需要 tracker：直接計算當前幀 ROI 內 person bbox 數量。
    """

    def __init__(self, interval_seconds: float = 15.0,
                 crowd_alert_count: int = 20):
        self._interval   = interval_seconds
        self._threshold  = crowd_alert_count
        self._last_tick  = time.time()
        self._last_count = 0

    # ── 主更新介面 ───────────────────────────────────────────────────────────

    def update(self, persons_in_roi: list[dict]) -> tuple[int, bool, bool]:
        """
        傳入當前幀 ROI 內人員清單。
        回傳 (count, is_interval_tick, crowd_alert)
          count           : 當前人數
          is_interval_tick: 是否到了紀錄間隔（每 N 秒 True 一次）
          crowd_alert     : 是否超過人數門檻
        """
        now   = time.time()
        count = len(persons_in_roi)
        self._last_count = count

        tick = (now - self._last_tick) >= self._interval
        if tick:
            self._last_tick = now

        return count, tick, count >= self._threshold

    # ── 繪製 ─────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, count: int, roi_label: str = ""):
        """
        左下角繪製目前人數。
        count < threshold → 綠色；>= threshold → 紅色。
        """
        h, w = frame.shape[:2]
        color = (0, 0, 255) if count >= self._threshold else (0, 220, 0)
        prefix = f"{roi_label}: " if roi_label else ""
        text   = f"{prefix}Count:{count}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x, y = 12, h - 16
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 6, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
