"""
行李箱滾落偵測。

演算法：
  1. 依 ByteTrack track_id 維護行李與人員的位置歷史
  2. 以最近 3 幀位移計算平滑平均速度向量
  3. 找最近人員，比較速度向量差（Independence Score）
  4. 速度超過門檻 AND 行動獨立 → 該幀標記 rolling
  5. 連續 alert_frames 幀皆 rolling → 觸發告警

Independence 設計：
  - 無附近人員 → 行李獨自移動，直接視為 rolling
  - 有附近人員且速度向量差 >= independence_threshold → 獨立移動 = rolling
  - 行李與人員速度相近（被攜帶中）→ 不告警
"""

import math
import time
import cv2
import numpy as np
from collections import deque


class LuggageRollMonitor:

    def __init__(self,
                 speed_threshold_px:        float = 15.0,
                 independence_threshold_px: float = 10.0,
                 alert_frames:              int   = 5,
                 max_match_distance_px:     float = 300.0):
        self._speed_thr  = speed_threshold_px
        self._indep_thr  = independence_threshold_px
        self._n_alert    = alert_frames
        self._max_dist   = max_match_distance_px

        # track_id → deque[(cx, cy)]，maxlen=3 供平均速度計算
        self._lug_pos:    dict[int, deque] = {}
        # person track_id → deque[(cx, cy)]
        self._person_pos: dict[int, deque] = {}
        # track_id → deque[bool]（逐幀是否判為 rolling）
        self._alert_hist: dict[int, deque] = {}
        # 最後看到時間，供 track 清理
        self._last_seen:  dict[int, float] = {}

    # ── 主更新介面 ───────────────────────────────────────────────────────────

    def update(self, luggage_in_roi: list[dict],
               all_persons: list[dict]) -> list[dict]:
        """
        傳入當前幀 ROI 內行李清單 + 全幀人員清單。
        回傳觸發滾落告警的行李 dict 清單（額外帶 roll_speed、roll_vel 欄位）。
        """
        now = time.time()
        current_lug_ids = set()

        # Step 1：更新人員位置歷史（供速度比較）
        for p in all_persons:
            ptid = p.get("track_id", -1)
            if ptid < 0:
                continue
            if ptid not in self._person_pos:
                self._person_pos[ptid] = deque(maxlen=3)
            self._person_pos[ptid].append((p["cx"], p["cy"]))

        # Step 2：逐件行李判斷
        alerts = []
        for lug in luggage_in_roi:
            tid = lug.get("track_id", -1)
            if tid < 0:
                continue
            current_lug_ids.add(tid)
            self._last_seen[tid] = now

            if tid not in self._lug_pos:
                self._lug_pos[tid]    = deque(maxlen=3)
                self._alert_hist[tid] = deque(maxlen=self._n_alert)

            self._lug_pos[tid].append((lug["cx"], lug["cy"]))

            pos_hist = list(self._lug_pos[tid])
            if len(pos_hist) < 2:
                self._alert_hist[tid].append(False)
                continue

            # Step 3：平均速度向量
            avg_vel, speed = self._avg_velocity(pos_hist)

            # Step 4：判斷 rolling
            rolling = self._is_rolling(lug["cx"], lug["cy"],
                                       avg_vel, speed, all_persons)
            self._alert_hist[tid].append(rolling)

            # Step 5：連續 N 幀皆 rolling → 告警
            hist = list(self._alert_hist[tid])
            if len(hist) == self._n_alert and all(hist):
                alerts.append({
                    **lug,
                    "roll_speed": round(speed, 1),
                    "roll_vel":   avg_vel,
                })

        # Step 6：清理消失超過 10 秒的 track
        cutoff = now - 10.0
        for tid in list(self._lug_pos):
            if tid not in current_lug_ids and self._last_seen.get(tid, 0) < cutoff:
                self._lug_pos.pop(tid, None)
                self._alert_hist.pop(tid, None)
                self._last_seen.pop(tid, None)

        return alerts

    # ── 繪製 ────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray,
             luggage_in_roi: list[dict],
             alerts: list[dict]):
        """
        繪製追蹤行李的 bbox、ID、速度、速度向量箭頭。
        顏色：黃色 = 追蹤中 / 紅色 = 滾落告警
        """
        alert_ids = {a["track_id"] for a in alerts}

        for lug in luggage_in_roi:
            tid = lug.get("track_id", -1)
            if tid < 0:
                continue

            x1, y1, x2, y2 = lug["bbox"]
            rolling = tid in alert_ids
            color   = (0, 0, 255) if rolling else (0, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            pos_hist = list(self._lug_pos.get(tid, []))
            if len(pos_hist) >= 2:
                avg_vel, speed = self._avg_velocity(pos_hist)
                cx, cy = (x1+x2)//2, (y1+y2)//2

                # 速度向量箭頭
                if speed > 0.5:
                    arrow_len = min(50, int(speed * 2))
                    norm = (avg_vel[0]/speed, avg_vel[1]/speed)
                    end  = (int(cx + norm[0]*arrow_len),
                            int(cy + norm[1]*arrow_len))
                    cv2.arrowedLine(frame, (cx, cy), end, color, 2, tipLength=0.35)

                label = f"ID:{tid} {speed:.0f}px/f"
                if rolling:
                    label += " ROLL"
            else:
                label = f"ID:{tid}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame,
                          (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # ── 內部工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _avg_velocity(pos_hist: list) -> tuple[tuple, float]:
        """由位置歷史計算平均速度向量與速率。"""
        vels = [
            (pos_hist[i+1][0] - pos_hist[i][0],
             pos_hist[i+1][1] - pos_hist[i][1])
            for i in range(len(pos_hist) - 1)
        ]
        avg = (sum(v[0] for v in vels) / len(vels),
               sum(v[1] for v in vels) / len(vels))
        return avg, math.hypot(*avg)

    def _is_rolling(self, cx: int, cy: int,
                    lug_vel: tuple, speed: float,
                    persons: list[dict]) -> bool:
        if speed < self._speed_thr:
            return False

        # 找最近人員
        nearest_ptid = -1
        nearest_dist = self._max_dist
        for p in persons:
            d = math.hypot(p["cx"] - cx, p["cy"] - cy)
            if d < nearest_dist:
                nearest_dist = d
                nearest_ptid = p.get("track_id", -1)

        # 無附近人員 → 獨立移動，視為 rolling
        if nearest_dist >= self._max_dist:
            return True

        # 與最近人員速度向量比較
        per_vel = self._get_person_vel(nearest_ptid)
        indep   = math.hypot(lug_vel[0] - per_vel[0],
                             lug_vel[1] - per_vel[1])
        return indep >= self._indep_thr

    def _get_person_vel(self, track_id: int) -> tuple:
        pos = self._person_pos.get(track_id)
        if not pos or len(pos) < 2:
            return (0.0, 0.0)
        hist = list(pos)
        return (hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])
