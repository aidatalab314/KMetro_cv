import json
import time
import cv2
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ_TAIPEI = timezone(timedelta(hours=8))
_COOLDOWN_SEC = 3.0


class EventManager:
    """
    事件輸出管理器。
    - Console JSON 輸出
    - TXT 事件 log（每攝影機獨立檔案）
    - Snapshot 截圖（可選）
    - 預留擴充 hook：喇叭告警、事件錄影、遠端推播（目前 no-op）
    """

    def __init__(self, camera_id: str,
                 snapshot_dir: str = "data/snapshots",
                 log_dir: str = "data/logs",
                 save_snapshots: bool = True):
        self.camera_id = camera_id
        self._snapshot_dir = Path(snapshot_dir)
        self._log_dir = Path(log_dir)
        self._save_snapshots = save_snapshots

        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self._log_dir / f"events_{camera_id}_{ts}.txt"

        # cooldown per (event_type, roi_id) 防重複刷屏
        self._last_triggered: dict[tuple, float] = {}

    def trigger(self, event_type: str, roi_id: str = "global",
                severity: str = "WARNING", confidence: float = 1.0,
                frame=None, extra: dict = None) -> bool:
        """
        觸發事件。
        回傳 True 表示事件成功記錄（不在冷卻中）。
        """
        key = (event_type, roi_id)
        now = time.time()
        if now - self._last_triggered.get(key, 0) < _COOLDOWN_SEC:
            return False
        self._last_triggered[key] = now

        ts = datetime.now(tz=TZ_TAIPEI).isoformat()
        event = {
            "event_type": event_type,
            "camera_id":  self.camera_id,
            "roi_id":     roi_id,
            "timestamp":  ts,
            "severity":   severity,
            "confidence": round(confidence, 4),
        }
        if extra:
            event.update(extra)

        line = json.dumps(event, ensure_ascii=False)
        print(f"[EVENT] {line}")

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if self._save_snapshots and frame is not None:
            snap_name = (
                f"{event_type}_{self.camera_id}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )
            cv2.imwrite(str(self._snapshot_dir / snap_name), frame)

        # ── 預留擴充 hook（未來實作時在此加入）──────────────────────────────
        self._on_alert_hook(event_type, roi_id, severity, frame)   # 喇叭告警
        self._on_event_record(event_type, frame)                   # 事件錄影
        self._on_remote_notify(event)                              # API/WebSocket 推播

        return True

    # ── 預留 hook（no-op，未來擴充）─────────────────────────────────────────

    def _on_alert_hook(self, event_type: str, roi_id: str,
                       severity: str, frame) -> None:
        """喇叭/廣播告警 hook（預留）"""
        pass

    def _on_event_record(self, event_type: str, frame) -> None:
        """事件錄影 hook（預留）"""
        pass

    def _on_remote_notify(self, event: dict) -> None:
        """遠端 API / WebSocket 推播 hook（預留）"""
        pass
