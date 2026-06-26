import numpy as np
import cv2
import cvzone
from ultralytics import YOLO
from collections import deque


class FallDetector:
    """
    以 YOLO12l 通用偵測器偵測 person 類別，
    判斷邏輯：人體 bbox 寬 > 高（橫倒）且持續 alert_frames 幀。

    weight_path=None：Bus 模式，不載模型；僅提供 preprocess() 與 parse_result()。
    tracking=True  ：使用 ByteTrack（直接模式時生效；Bus 模式追蹤由 InferenceBus 處理）。
    """

    PERSON_CLASS = 0   # COCO class index for person

    def __init__(self, weight_path: str | None = "models/fall_detection/yolo12l.pt",
                 conf: float = 0.5, fallen_aspect_ratio: float = 1.2,
                 alert_frames: int = 5, clahe: bool = True,
                 gamma: float = 1.0, imgsz: int = 640,
                 tracking: bool = False):
        self.model = YOLO(weight_path) if weight_path else None
        self.conf = conf
        self.fallen_aspect_ratio = fallen_aspect_ratio
        self._history: deque[int] = deque(maxlen=alert_frames)
        self.alert_frames = alert_frames
        self.imgsz = imgsz
        self._tracking = tracking
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if clahe else None
        self._gamma_lut = self._build_gamma_lut(gamma) if gamma != 1.0 else None

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        return np.array([(i / 255.0) ** gamma * 255 for i in range(256)],
                        dtype=np.uint8)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE + Gamma 前處理（Bus 模式在 submit 前呼叫）。"""
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

    def parse_result(self, result) -> list[dict]:
        """從單個 ultralytics Results 物件解析偵測結果（Bus 模式使用）。"""
        if result is None or result.boxes is None:
            return []
        detections = []
        for box in result.boxes:
            if int(box.cls[0]) != self.PERSON_CLASS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append({
                "bbox":     (x1, y1, x2, y2),
                "conf":     float(box.conf[0]),
                "fallen":   self._is_fallen(x1, y1, x2, y2),
                "cx":       (x1 + x2) // 2,
                "cy":       (y1 + y2) // 2,
                "track_id": track_id,
            })
        return detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        """直接推論（非 Bus 模式，向下相容）。"""
        processed = self.preprocess(frame)
        if self._tracking:
            results = self.model.track(processed, conf=self.conf, imgsz=self.imgsz,
                                       verbose=False, tracker="bytetrack.yaml",
                                       persist=True)
        else:
            results = self.model(processed, conf=self.conf,
                                 imgsz=self.imgsz, verbose=False)
        dets = []
        for r in results:
            dets.extend(self.parse_result(r))
        return dets

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
            tid   = det.get("track_id", -1)
            id_prefix = f"ID:{tid} " if tid >= 0 else ""
            label = (f"{id_prefix}FALL {det['conf']:.2f}"
                     if det["fallen"]
                     else f"{id_prefix}person {det['conf']:.2f}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, (0, 0, 255), -1)
            cvzone.putTextRect(frame, label, (x1, y1), scale=2, thickness=2,
                               colorR=color, colorT=(0, 0, 0))

        if alert:
            cvzone.putTextRect(frame, "!! FALL DETECTED !!", (30, 60),
                               scale=2, thickness=3,
                               colorR=(0, 0, 255), colorT=(255, 255, 255))
        return frame
