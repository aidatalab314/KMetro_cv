import numpy as np
import cv2
import cvzone
from ultralytics import YOLO


class LuggageDetector:
    def __init__(self, weight_path: str, conf: float = 0.4,
                 size_method: str = "person_ratio",
                 large_person_area_ratio: float = 0.22,
                 max_match_distance_px: float = 400,
                 large_area_ratio: float = 0.01):
        self.model = YOLO(weight_path)
        self.conf = conf
        self.size_method = size_method
        self.large_person_area_ratio = large_person_area_ratio
        self.max_match_distance_px = max_match_distance_px
        self.large_area_ratio = large_area_ratio

    def _nearest_person(self, cx: int, cy: int,
                        persons: list[dict]) -> dict | None:
        """找中心點距離最近且在 max_match_distance_px 內的行人"""
        best, best_dist = None, float("inf")
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, p
        return best if best_dist <= self.max_match_distance_px else None

    def _classify_size(self, x1, y1, x2, y2,
                       frame_h: int, frame_w: int,
                       persons: list[dict]) -> tuple[str, str]:
        """
        Returns (size_label, method_used)
        size_label : "Large" | "Small"
        method_used: "person_ratio" | "frame_ratio"
        """
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        luggage_area = (x2 - x1) * (y2 - y1)

        if self.size_method == "person_ratio" and persons:
            person = self._nearest_person(cx, cy, persons)
            if person:
                px1, py1, px2, py2 = person["bbox"]
                person_area = (px2 - px1) * (py2 - py1)
                if person_area > 0:
                    ratio = luggage_area / person_area
                    size = "Large" if ratio >= self.large_person_area_ratio else "Small"
                    return size, "person_ratio"

        # 備援：frame ratio
        ratio = luggage_area / (frame_h * frame_w)
        size = "Large" if ratio >= self.large_area_ratio else "Small"
        return size, "frame_ratio"

    def detect(self, frame: np.ndarray,
               persons: list[dict] = None) -> list[dict]:
        """
        persons: FallDetector.detect() 回傳的行人偵測結果（共用 yolov10s）
        """
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                size, method = self._classify_size(
                    x1, y1, x2, y2, h, w, persons or [])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "size": size,
                    "method": method,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                })
        return detections

    def draw(self, frame: np.ndarray, detections: list[dict],
             roi_labels: list[str] = None) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 100, 255) if det["size"] == "Large" else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (det["cx"], det["cy"]), 5, (0, 0, 255), -1)

            label = f"{det['size']} luggage {det['conf']:.2f}"
            cvzone.putTextRect(frame, label, (x1, y1), scale=2, thickness=2,
                               colorR=color, colorT=(0, 0, 0))

            if roi_labels:
                for i, rl in enumerate(roi_labels):
                    cv2.putText(frame, rl, (x1, y1 - 40 - i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return frame
