"""
單攝影機處理執行緒。

功能實作進度：
  ✓ 功能1: dwell_monitor（ByteTrack ID + DwellMonitor）
  ✓ 功能3: zone_counter（每 N 秒計數 + 人數 overlay）
  ✓ 功能4a: fall_detector（YOLO aspect ratio + RTMO pose 補位）
  ✓ 功能4b: luggage_roll（ByteTrack 速度向量 + LuggageRollMonitor）
  ✓ 功能5: size_classifier（LuggageDetector person_ratio）
  TODO 功能2: fire_smoke（等待模型）
"""

import threading
import queue
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path

from src.utils import log
from src.rtsp_reader import RTSPReader
from src.roi.roi_manager import ROIManager
from src.event.event_manager import EventManager
from src.detection.fall_detector import FallDetector
from src.detection.luggage_detector import LuggageDetector
from src.detection.pose_detector import PoseDetector
from src.features.dwell_monitor import DwellMonitor
from src.features.zone_counter import ZoneCounter
from src.features.luggage_roll_monitor import LuggageRollMonitor


class CameraWorker(threading.Thread):

    def __init__(self, cam_cfg: dict, global_cfg: dict, source,
                 mode: str, records_path: str, frame_queue: queue.Queue,
                 roi_key: str | None = None):
        super().__init__(daemon=True, name=f"worker-{cam_cfg['id']}")

        self.camera_id   = cam_cfg["id"]
        self.camera_name = cam_cfg.get("name", self.camera_id)
        self._stop_event = threading.Event()
        self._queue      = frame_queue
        self._mode       = mode

        enabled   = cam_cfg.get("features", {})
        det_cfg   = global_cfg.get("detector", {})
        feat_cfg  = global_cfg.get("features", {})
        model_cfg = global_cfg.get("models", {})
        out_cfg   = global_cfg.get("output", {})

        # ── 影像來源 ────────────────────────────────────────────────────────
        self._reader = RTSPReader(source, fallback=cam_cfg.get("fallback"))

        # ── ROI ──────────────────────────────────────────────────────────────
        self._roi = ROIManager(roi_key or self.camera_id, records_path)

        # ── 事件管理 ─────────────────────────────────────────────────────────
        self._events = EventManager(
            camera_id=self.camera_id,
            snapshot_dir=out_cfg.get("snapshot_dir", "data/snapshots"),
            log_dir=out_cfg.get("log_dir", "data/logs"),
            save_snapshots=out_cfg.get("save_snapshots", True),
        )

        # ── 推論設定 ─────────────────────────────────────────────────────────
        self._skip    = max(1, det_cfg.get("skip_frames", 2))
        self._frame_n = 0

        # ── 偵測器（feature-gated）──────────────────────────────────────────
        needs_person           = any(enabled.get(f) for f in (
            "fall_detector", "dwell_monitor", "zone_counter",
            "luggage_roll", "size_classifier",
        ))
        needs_person_tracking  = (enabled.get("dwell_monitor", False) or
                                  enabled.get("luggage_roll",  False))
        needs_luggage          = (enabled.get("size_classifier", False) or
                                  enabled.get("luggage_roll",   False))
        needs_luggage_tracking = enabled.get("luggage_roll", False)

        self._fall_det: FallDetector | None = None
        if needs_person:
            fd = feat_cfg.get("fall_detector", {})
            self._fall_det = FallDetector(
                weight_path=model_cfg.get("person", "models/fall_detection/yolo12l.pt"),
                conf=det_cfg.get("conf", 0.4),
                fallen_aspect_ratio=fd.get("fallen_aspect_ratio", 1.2),
                alert_frames=fd.get("alert_frames", 5),
                clahe=fd.get("clahe", True),
                gamma=fd.get("gamma", 1.0),
                imgsz=fd.get("imgsz", det_cfg.get("imgsz", 640)),
                tracking=needs_person_tracking,
            )

        self._luggage_det: LuggageDetector | None = None
        if needs_luggage:
            sc = feat_cfg.get("size_classifier", {})
            lr = feat_cfg.get("luggage_roll", {})
            self._luggage_det = LuggageDetector(
                weight_path=model_cfg.get("luggage", "models/luggage/yolo_luggage_best.pt"),
                conf=det_cfg.get("conf", 0.4),
                size_method=sc.get("size_method", "person_ratio"),
                large_person_area_ratio=sc.get("large_person_area_ratio", 0.22),
                max_match_distance_px=sc.get("max_match_distance_px",
                                             lr.get("max_match_distance_px", 400)),
                large_area_ratio=sc.get("large_area_ratio", 0.01),
                tracking=needs_luggage_tracking,
            )

        self._pose_det: PoseDetector | None = None
        if enabled.get("fall_detector"):
            pf = feat_cfg.get("fall_detector", {}).get("pose_fallback", {})
            if pf.get("enabled", False):
                self._pose_det = PoseDetector(
                    mode=pf.get("mode", "balanced"),
                    kp_conf=pf.get("kp_conf", 0.3),
                    min_kp=pf.get("min_kp", 3),
                    fall_angle_deg=pf.get("fall_angle_deg", 50.0),
                    gamma=feat_cfg.get("fall_detector", {}).get("gamma", 1.0),
                    device=pf.get("device", "cpu"),
                )

        self._fire_det = None
        if enabled.get("fire_smoke"):
            fire_path = model_cfg.get("fire_smoke", "")
            if not Path(fire_path).exists():
                log("WARN", f"[{self.camera_id}] fire_smoke 模型不存在，略過: {fire_path}")

        # ── 功能模組 ─────────────────────────────────────────────────────────
        self._dwell_mon: DwellMonitor | None = None
        if enabled.get("dwell_monitor"):
            dm = feat_cfg.get("dwell_monitor", {})
            self._dwell_mon = DwellMonitor(
                alert_seconds=dm.get("alert_seconds", 60.0),
                grace_period_sec=dm.get("grace_period_sec", 10.0),
            )

        self._zone_ctr: ZoneCounter | None = None
        if enabled.get("zone_counter"):
            zc = feat_cfg.get("zone_counter", {})
            self._zone_ctr = ZoneCounter(
                interval_seconds=zc.get("interval_seconds", 15.0),
                crowd_alert_count=zc.get("crowd_alert_count", 20),
            )

        self._roll_mon: LuggageRollMonitor | None = None
        if enabled.get("luggage_roll"):
            lr = feat_cfg.get("luggage_roll", {})
            self._roll_mon = LuggageRollMonitor(
                speed_threshold_px=lr.get("speed_threshold_px_per_frame", 15.0),
                independence_threshold_px=lr.get("independence_threshold_px_per_frame", 10.0),
                alert_frames=lr.get("alert_frames", 5),
                max_match_distance_px=lr.get("max_match_distance_px", 300.0),
            )

        self._feat = enabled

        # ── 輸出影片 ─────────────────────────────────────────────────────────
        self._writer           = None
        self._video_dir        = Path(out_cfg.get("video_dir", "outputs/videos"))
        self._save_video_local = out_cfg.get("save_video_local", True)
        self._save_video_rtsp  = out_cfg.get("save_video_rtsp", False)

        # ── 狀態（供 multistream display thread 讀取，不需加鎖）────────────
        self.fps:   float = 0.0
        self._last_annotated: np.ndarray | None = None   # skip-frame cache
        # 最近 5 筆告警，deque 在 CPython append 操作是 thread-safe
        self.recent_alerts: deque = deque(maxlen=5)
        self.status_info: dict = {
            "name":         self.camera_id,    # ASCII-only for cv2.putText
            "fps":          0.0,
            "skip":         self._skip,
            "person_track": needs_person_tracking,
            "zone_count":   0,
            "dwell_active": 0,
            "features":     [k for k, v in enabled.items() if v],
        }

    # ── 執行緒主迴圈 ─────────────────────────────────────────────────────────

    def run(self):
        if not self._reader.open():
            log("ERROR", f"[{self.camera_id}] 無法開啟影像來源，worker 結束")
            return

        w, h    = self._reader.get_size()
        src_fps = self._reader.get_fps()
        is_file = self._reader.is_file()

        if (is_file and self._save_video_local) or \
           (not is_file and self._save_video_rtsp):
            self._init_writer(w, h, src_fps)

        active = ", ".join(k for k, v in self._feat.items() if v) or "（無）"
        log("INFO", f"[{self.camera_id}] 開始推論 {w}x{h}@{src_fps:.1f}fps "
                    f"skip={self._skip} 功能=[{active}]")

        t_frame = time.time()

        while not self._stop_event.is_set():
            ret, frame = self._reader.read()
            if not ret:
                if is_file:
                    log("INFO", f"[{self.camera_id}] 影片結束")
                else:
                    time.sleep(0.05)
                    continue
                break

            self._frame_n += 1

            if self._frame_n % self._skip == 0:
                # Full inference frame: run detection + draw
                annotated = self._process(frame)
                self._last_annotated = annotated
            else:
                # Skip frame: reuse last annotated overlay to avoid flickering.
                # We re-apply only the cheap ROI draw onto the current raw frame
                # so the background pixels are fresh but detection boxes stay visible.
                if self._last_annotated is not None:
                    annotated = self._last_annotated
                else:
                    # First few frames before first inference: just show raw
                    annotated = frame

            now = time.time()
            self.fps = 1.0 / max(now - t_frame, 1e-6)
            self.status_info["fps"] = self.fps
            t_frame = now

            if self._writer is not None:
                self._writer.write(annotated)

            try:
                self._queue.put_nowait((self.camera_id, annotated, self.fps))
            except queue.Full:
                pass

        self._cleanup()

    # ── 推論核心 ─────────────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()

        # Step 1：在 clean frame 偵測（ROI overlay 不影響模型輸入）
        all_persons: list[dict] = []
        if self._fall_det is not None:
            all_persons = self._fall_det.detect(frame)

        all_luggage: list[dict] = []
        if self._luggage_det is not None:
            all_luggage = self._luggage_det.detect(frame, persons=all_persons)

        # Step 2：各功能使用各自 ROI 過濾
        #  - 有對應 ROI → 在 ROI 內的偵測結果
        #  - 無對應 ROI → 空 list（跳過此功能）
        fall_in_roi   = self._in_roi(all_persons, "fall_detector")
        dwell_in_roi  = self._in_roi(all_persons, "dwell_monitor")
        zone_in_roi   = self._in_roi(all_persons, "zone_counter")
        lug_roll_roi  = self._in_roi(all_luggage, "luggage_roll")
        lug_size_roi  = self._in_roi(all_luggage, "size_classifier")

        # Step 3：Pose 補位（僅 fall ROI 內無 YOLO person 時）
        pose_in_roi: list[dict] = []
        if (self._pose_det is not None and not fall_in_roi and
                self._roi.has_roi_for_feature("fall_detector")):
            pose_dets   = self._pose_det.detect(frame)
            pose_in_roi = self._in_roi(pose_dets, "fall_detector")
            self._pose_det.draw(annotated, pose_in_roi)

        alert_persons = fall_in_roi if fall_in_roi else pose_in_roi

        # Step 4：功能模組
        self._run_dwell_monitor(annotated, dwell_in_roi)
        self._run_zone_counter(annotated, zone_in_roi)
        self._run_fall_logic(annotated, fall_in_roi, alert_persons)
        self._run_luggage_roll(annotated, lug_roll_roi, all_persons)
        self._run_size_classifier(annotated, lug_size_roi)

        # Step 5：ROI overlay（最上層）
        self._roi.draw(annotated)
        if self._mode == "dev":
            self._draw_dev_corner(annotated)

        return annotated

    def _in_roi(self, detections: list[dict], feature: str) -> list[dict]:
        """回傳在此 feature ROI 內的偵測結果；無 ROI 設定則回傳空 list（不執行此功能）。"""
        if not self._roi.has_roi_for_feature(feature):
            return []
        return [d for d in detections
                if self._roi.is_inside_for_feature(d["cx"], d["cy"], feature)]

    # ── 功能模組 ─────────────────────────────────────────────────────────────

    def _run_dwell_monitor(self, annotated: np.ndarray, persons: list[dict]):
        if self._dwell_mon is None:
            return
        alerts = self._dwell_mon.update(persons)
        self._dwell_mon.draw(annotated, persons)
        self.status_info["dwell_active"] = len(alerts)
        for p in alerts:
            rois = self._roi.get_containing_rois_for_feature(p["cx"], p["cy"], "dwell_monitor")
            fired = self._events.trigger(
                event_type="dwell_alert",
                roi_id=rois[0] if rois else "global",
                severity="WARNING",
                confidence=p["conf"],
                frame=annotated,
                extra={"track_id": p["track_id"],
                       "dwell_seconds": round(p["dwell_seconds"], 1)},
            )
            if fired:
                self._push_alert("dwell_alert",
                                 rois[0] if rois else "global", "WARNING")
                if self._mode == "dev":
                    x1, y1, x2, y2 = p["bbox"]
                    cv2.rectangle(annotated, (x1-3, y1-3), (x2+3, y2+3),
                                  (0, 0, 255), 3)

    def _run_zone_counter(self, annotated: np.ndarray, persons: list[dict]):
        if self._zone_ctr is None:
            return
        count, tick, crowd = self._zone_ctr.update(persons)
        self._zone_ctr.draw(annotated, count)
        self.status_info["zone_count"] = count
        if tick:
            log("INFO", f"[{self.camera_id}] 區域人數: {count}")
        if crowd:
            rois = (self._roi.get_containing_rois_for_feature(
                persons[0]["cx"], persons[0]["cy"], "zone_counter"
            ) if persons else [])
            fired = self._events.trigger(
                event_type="crowd_alert",
                roi_id=rois[0] if rois else "global",
                severity="WARNING",
                confidence=1.0,
                frame=annotated,
                extra={"count": count, "threshold": self._zone_ctr._threshold},
            )
            if fired:
                self._push_alert("crowd_alert",
                                 rois[0] if rois else "global", "WARNING",
                                 f"{count} persons")

    def _run_fall_logic(self, annotated: np.ndarray,
                        fall_in_roi: list[dict],
                        alert_persons: list[dict]):
        if not self._feat.get("fall_detector") or self._fall_det is None:
            return
        alert = self._fall_det.compute_alert(alert_persons)
        self._fall_det.draw(annotated, fall_in_roi, alert)
        if alert:
            rois = (self._roi.get_containing_rois_for_feature(
                alert_persons[0]["cx"], alert_persons[0]["cy"], "fall_detector"
            ) if alert_persons else [])
            fired = self._events.trigger(
                event_type="fall_detected",
                roi_id=rois[0] if rois else "global",
                severity="CRITICAL",
                confidence=max((d["conf"] for d in alert_persons), default=1.0),
                frame=annotated,
            )
            if fired:
                self._push_alert("fall_detected",
                                 rois[0] if rois else "global", "CRITICAL")

    def _run_luggage_roll(self, annotated: np.ndarray,
                          luggage_in_roi: list[dict],
                          all_persons: list[dict]):
        if self._roll_mon is None:
            return
        alerts = self._roll_mon.update(luggage_in_roi, all_persons)
        self._roll_mon.draw(annotated, luggage_in_roi, alerts)
        for lug in alerts:
            rois = self._roi.get_containing_rois_for_feature(
                lug["cx"], lug["cy"], "luggage_roll")
            fired = self._events.trigger(
                event_type="luggage_roll_detected",
                roi_id=rois[0] if rois else "global",
                severity="CRITICAL",
                confidence=lug["conf"],
                frame=annotated,
                extra={"track_id": lug["track_id"],
                       "roll_speed": lug["roll_speed"]},
            )
            if fired:
                self._push_alert("luggage_roll",
                                 rois[0] if rois else "global", "CRITICAL",
                                 f"{lug['roll_speed']}px/f")

    def _run_size_classifier(self, annotated: np.ndarray,
                             luggage_in_roi: list[dict]):
        if not self._feat.get("size_classifier") or self._luggage_det is None:
            return
        for det in luggage_in_roi:
            rois = self._roi.get_containing_rois_for_feature(
                det["cx"], det["cy"], "size_classifier")
            self._luggage_det.draw(annotated, [det], roi_labels=rois)
            if det["size"] == "Large":
                fired = self._events.trigger(
                    event_type="large_luggage_detected",
                    roi_id=rois[0] if rois else "global",
                    severity="WARNING",
                    confidence=det["conf"],
                    frame=annotated,
                    extra={"method": det["method"]},
                )
                if fired:
                    self._push_alert("large_luggage",
                                     rois[0] if rois else "global", "WARNING")

    # ── 工具 ─────────────────────────────────────────────────────────────────

    def _push_alert(self, event: str, roi: str, severity: str, detail: str = ""):
        """把最新告警推入 recent_alerts（multistream display thread 讀取）。"""
        self.recent_alerts.append({
            "event":    event,
            "roi":      roi,
            "severity": severity,
            "detail":   detail,
            "time":     datetime.now().strftime("%H:%M:%S"),
        })

    def _draw_dev_corner(self, frame: np.ndarray):
        """右上角顯示 FPS（主要資訊已在 multistream 底部列顯示）。"""
        h, w = frame.shape[:2]
        text = f"{self.fps:.1f}fps"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (w - tw - 10, 4), (w - 4, th + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (w - tw - 6, th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    def _init_writer(self, w: int, h: int, fps: float):
        self._video_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self._video_dir / f"{self.camera_id}_{ts}.mp4"
        self._writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        log("INFO", f"[{self.camera_id}] 輸出影片: {out_path}")

    def stop(self):
        self._stop_event.set()

    def _cleanup(self):
        if self._writer:
            self._writer.release()
        self._reader.release()
        log("INFO", f"[{self.camera_id}] worker 已結束")
