import numpy as np
import cv2
from rtmlib import Body

# COCO-17 keypoint indices
_NOSE                    = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP,      _R_HIP      = 11, 12


class PoseDetector:
    """
    RTMO bottom-up fallback detector。
    當 YOLO 在 ROI 內未偵測到 person 時，用 RTMO 掃全圖。
    RTMO 是 one-stage 模型，不需要 person bbox，直接從全圖找骨架關節點，
    可偵測到 YOLO 因暗區低 confidence 漏掉的人。
    """

    def __init__(self,
                 mode: str = 'balanced',
                 kp_conf: float = 0.3,
                 min_kp: int = 3,
                 fall_angle_deg: float = 50.0,
                 gamma: float = 1.0,
                 device: str = 'cpu'):
        # pose='rtmo' 觸發 rtmlib 使用 RTMO one-stage bottom-up 模式
        self.body = Body(pose='rtmo', mode=mode,
                         backend='onnxruntime', device=device)
        self.kp_conf = kp_conf
        self.min_kp = min_kp
        # 肩-髖向量與垂直方向夾角超過此值 → 橫倒
        # cos(fall_angle_deg)：夾角 θ > fall_angle_deg 時 cos(θ) < cos(fall_angle_deg)
        self._fall_cos = np.cos(np.radians(fall_angle_deg))
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._gamma_lut = self._build_gamma_lut(gamma) if gamma != 1.0 else None

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        return np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self._gamma_lut is not None:
            frame = cv2.LUT(frame, self._gamma_lut)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """掃描全幀，回傳與 fall_detector.detect() 相同格式的 list。"""
        keypoints, scores = self.body(self._preprocess(frame))   # (N,17,2), (N,17)
        detections = []
        for kps, scs in zip(keypoints, scores):
            valid = scs > self.kp_conf
            if valid.sum() < self.min_kp:
                continue

            xs, ys = kps[valid, 0], kps[valid, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            detections.append({
                "bbox":      (x1, y1, x2, y2),
                "conf":      float(scs[valid].mean()),
                "fallen":    self._is_fallen(kps, scs),
                "cx":        (x1 + x2) // 2,
                "cy":        (y1 + y2) // 2,
                "source":    "pose",
                "keypoints": kps,
                "kp_scores": scs,
            })
        return detections

    def _is_fallen(self, kps: np.ndarray, scs: np.ndarray) -> bool:
        """
        主要：肩中點 → 髖中點向量與垂直軸夾角 > fall_angle_deg → 橫倒。
        上半身 fallback：髖部不可見時，改用鼻尖 → 肩中點向量判斷
                        （電扶梯側板常遮住下半身）。
        """
        shoulders_ok = scs[_L_SHOULDER] > self.kp_conf and scs[_R_SHOULDER] > self.kp_conf
        hips_ok      = scs[_L_HIP]      > self.kp_conf and scs[_R_HIP]      > self.kp_conf

        if shoulders_ok and hips_ok:
            shoulder = (kps[_L_SHOULDER] + kps[_R_SHOULDER]) / 2
            hip      = (kps[_L_HIP]      + kps[_R_HIP])      / 2
            return self._vec_is_fallen(hip - shoulder)

        # 上半身 fallback：鼻尖可見 + 雙肩可見
        if shoulders_ok and scs[_NOSE] > self.kp_conf:
            shoulder = (kps[_L_SHOULDER] + kps[_R_SHOULDER]) / 2
            # 鼻尖 → 肩中點：站立時向下，跌倒時趨近水平
            return self._vec_is_fallen(shoulder - kps[_NOSE])

        return False

    def _vec_is_fallen(self, vec: np.ndarray) -> bool:
        norm = np.linalg.norm(vec)
        if norm < 1e-3:
            return False
        return (abs(vec[1]) / norm) < self._fall_cos

    # COCO-17 骨架連線
    _SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # 頭部
        (5, 6),                                    # 肩膀
        (5, 7), (7, 9), (6, 8), (8, 10),          # 手臂
        (5, 11), (6, 12), (11, 12),               # 軀幹
        (11, 13), (13, 15), (12, 14), (14, 16),   # 腿部
    ]

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            color = (0, 64, 255) if det["fallen"] else (0, 165, 255)  # 橘→紅橘區分跌倒
            x1, y1, x2, y2 = det["bbox"]
            label = f"[pose] {'FALL' if det['fallen'] else 'person'} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            kps, scs = det["keypoints"], det["kp_scores"]

            # 骨架連線
            for a, b in self._SKELETON:
                if scs[a] > self.kp_conf and scs[b] > self.kp_conf:
                    pt1 = (int(kps[a, 0]), int(kps[a, 1]))
                    pt2 = (int(kps[b, 0]), int(kps[b, 1]))
                    cv2.line(frame, pt1, pt2, color, 2)

            # 關節點
            for i, (x, y) in enumerate(kps):
                if scs[i] > self.kp_conf:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        return frame
