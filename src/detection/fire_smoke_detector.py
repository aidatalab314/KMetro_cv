from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


class FireSmokeDetector:
    """
    YOLOv8n 火焰／煙霧偵測器（luminous0219/fire-and-smoke-detection-yolov8）。
    Classes: 0=fire, 1=smoke。

    weight_path=None：Bus 模式，不載模型；僅提供 parse_result() 與 compute_alerts()。
    """

    CLASS_NAMES = {0: "fire", 1: "smoke"}
    _COLORS = {
        "fire":  (0,   0,   255),
        "smoke": (100, 100, 100),
    }

    def __init__(self, weight_path: str | None = "models/fire_smoke/fire_smoke.pt",
                 conf: float = 0.5, alert_frames: int = 3):
        self.model = YOLO(weight_path) if weight_path else None
        self.conf  = conf
        self.alert_frames = alert_frames
        self._history: dict[int, deque] = {
            cls: deque(maxlen=alert_frames) for cls in self.CLASS_NAMES
        }

    def parse_result(self, result) -> list[dict]:
        """從單個 ultralytics Results 物件解析偵測結果（Bus 模式使用）。"""
        if result is None or result.boxes is None:
            return []
        detections = []
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox":  (x1, y1, x2, y2),
                "cx":    (x1 + x2) // 2,
                "cy":    (y1 + y2) // 2,
                "conf":  float(box.conf[0]),
                "class": cls,
                "label": self.CLASS_NAMES.get(cls, "unknown"),
            })
        return detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        """直接推論（非 Bus 模式，向下相容）。"""
        results = self.model(frame, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            dets.extend(self.parse_result(r))
        return dets

    def compute_alerts(self, detections: list[dict]) -> dict[str, bool]:
        """
        傳入「在 ROI 內」的偵測結果，更新逐幀歷史並回傳各類別是否觸發告警。
        """
        detected_classes = {d["class"] for d in detections}
        alerts: dict[str, bool] = {}
        for cls, name in self.CLASS_NAMES.items():
            self._history[cls].append(1 if cls in detected_classes else 0)
            alerts[name] = (
                len(self._history[cls]) == self.alert_frames and
                sum(self._history[cls]) == self.alert_frames
            )
        return alerts

    def draw(self, frame: np.ndarray,
             detections: list[dict],
             alerts: dict[str, bool] | None = None) -> None:
        if alerts is None:
            alerts = {}
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            color = self._COLORS.get(label, (255, 255, 255))
            thickness = 3 if alerts.get(label) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame,
                        f"{label} {d['conf']:.2f}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                        cv2.LINE_AA)
