import numpy as np
import cv2
import cvzone
from ultralytics import YOLO
from collections import deque


class FallDetector:
    """
    以 YOLO12l 通用偵測器偵測 person 類別，
    判斷邏輯：人體 bbox 寬 > 高（橫倒）且持續 alert_frames 幀。
    參考：github.com/freedomwebtech/person_fall_detection
    """

    PERSON_CLASS = 0   # COCO class index for person

    def __init__(self, weight_path: str = "models/fall_detection/yolo12l.pt",
                 conf: float = 0.5, fallen_aspect_ratio: float = 1.2,
                 alert_frames: int = 5, clahe: bool = True,
                 gamma: float = 1.0, imgsz: int = 640):
        self.model = YOLO(weight_path)
        self.conf = conf
        self.fallen_aspect_ratio = fallen_aspect_ratio
        self._history: deque[int] = deque(maxlen=alert_frames)
        self.alert_frames = alert_frames
        self.imgsz = imgsz
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if clahe else None
        # gamma < 1 提亮暗部（0.5 = sqrt），gamma=1 不做任何處理
        self._gamma_lut = self._build_gamma_lut(gamma) if gamma != 1.0 else None

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        return np.array([(i / 255.0) ** gamma * 255 for i in range(256)],
                        dtype=np.uint8)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self._gamma_lut is not None:
            frame = cv2.LUT(frame, self._gamma_lut)
        if self._clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return frame

    def _is_fallen(self, x1, y1, x2, y2) -> bool:
        w = x2 - x1
        h = y2 - y1
        return h > 0 and (w / h) >= self.fallen_aspect_ratio

    def detect(self, frame: np.ndarray) -> list[dict]:
        """偵測全幀所有 person，不更新 alert 歷史（交由 compute_alert 處理）。"""
        results = self.model(self._preprocess(frame), conf=self.conf,
                             imgsz=self.imgsz, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != self.PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                fallen = self._is_fallen(x1, y1, x2, y2)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "fallen": fallen,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                })
        return detections

    def compute_alert(self, detections: list[dict]) -> bool:
        """根據已過濾（ROI 內）的偵測結果更新歷史並回傳是否觸發警報。"""
        fallen_count = sum(1 for d in detections if d["fallen"])
        self._history.append(fallen_count)
        return (len(self._history) == self.alert_frames and
                all(c > 0 for c in self._history))

    def draw(self, frame: np.ndarray, detections: list[dict],
             alert: bool) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 0, 255) if det["fallen"] else (0, 255, 0)
            label = f"FALL {det['conf']:.2f}" if det["fallen"] else f"person {det['conf']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, (0, 0, 255), -1)
            cvzone.putTextRect(frame, label, (x1, y1), scale=2, thickness=2,
                               colorR=color, colorT=(0, 0, 0))

        if alert:
            cvzone.putTextRect(frame, "!! FALL DETECTED !!", (30, 60),
                               scale=2, thickness=3,
                               colorR=(0, 0, 255), colorT=(255, 255, 255))
        return frame
