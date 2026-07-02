"""
單攝影機處理執行緒。

功能實作進度：
  ✓ 功能1: dwell_monitor（ByteTrack ID + DwellMonitor）
  ✓ 功能3: zone_counter（每 N 秒計數 + 人數 overlay）
  ✓ 功能4a: fall_detector（YOLO aspect ratio + RTMO pose 補位）
  ✓ 功能4b: luggage_roll（ByteTrack 速度向量 + LuggageRollMonitor）
  ✓ 功能5: size_classifier（LuggageDetector person_ratio）
  ✓ 功能2: fire_smoke（YOLOv8n，luminous0219/fire-and-smoke-detection-yolov8）
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
from src.detection.fire_smoke_detector import FireSmokeDetector
from src.detection.luggage_detector import LuggageDetector
from src.detection.pose_detector import PoseDetector
from src.features.dwell_monitor import DwellMonitor
from src.features.zone_counter import ZoneCounter
from src.features.luggage_roll_monitor import LuggageRollMonitor

# TYPE_CHECKING 避免循環 import（InferenceBus 在 inference 模組）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.inference.inference_bus import InferenceBus


class CameraWorker(threading.Thread):

    def __init__(self, cam_cfg: dict, global_cfg: dict, source,
                 mode: str, records_path: str, frame_queue: queue.Queue,
                 roi_key: str | None = None,
                 bus: "InferenceBus | None" = None):
        super().__init__(daemon=True, name=f"worker-{cam_cfg['id']}")
        self._bus = bus

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

        # Bus 模式：偵測器不載模型（GPU 推論由 InferenceBus 統一負責）
        _bus_mode        = bus is not None
        _person_path     = None if _bus_mode else model_cfg.get("person",  "models/fall_detection/yolo12l.pt")
        _luggage_path    = None if _bus_mode else model_cfg.get("luggage", "models/luggage/yolo_luggage_best.pt")

        self._fall_det: FallDetector | None = None
        if needs_person:
            fd = feat_cfg.get("fall_detector", {})
            self._fall_det = FallDetector(
                weight_path=_person_path,
                conf=det_cfg.get("conf", 0.4),
                fallen_aspect_ratio=fd.get("fallen_aspect_ratio", 1.2),
                alert_frames=fd.get("alert_frames", 5),
                clahe=fd.get("clahe", True),
                gamma=fd.get("gamma", 1.0),
                imgsz=fd.get("imgsz", det_cfg.get("imgsz", 640)),
                tracking=(needs_person_tracking and not _bus_mode),
            )

        self._luggage_det: LuggageDetector | None = None
        if needs_luggage:
            sc = feat_cfg.get("size_classifier", {})
            lr = feat_cfg.get("luggage_roll", {})
            self._luggage_det = LuggageDetector(
                weight_path=_luggage_path,
                conf=det_cfg.get("conf", 0.4),
                size_method=sc.get("size_method", "person_ratio"),
                large_person_area_ratio=sc.get("large_person_area_ratio", 0.22),
                max_match_distance_px=sc.get("max_match_distance_px",
                                             lr.get("max_match_distance_px", 400)),
                large_area_ratio=sc.get("large_area_ratio", 0.01),
                tracking=(needs_luggage_tracking and not _bus_mode),
            )

        self._pose_det: PoseDetector | None = None
        if enabled.get("fall_detector") and not _bus_mode:
            # Pose 補位只在直接模式使用；Bus 模式跳過（效能考量）
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

        self._fire_det: FireSmokeDetector | None = None
        if enabled.get("fire_smoke"):
            fs = feat_cfg.get("fire_smoke", {})
            if _bus_mode:
                # Bus 模式：建立解析器（無模型），fire inference 由 InferenceBus 負責
                self._fire_det = FireSmokeDetector(
                    weight_path=None,
                    conf=fs.get("conf", 0.5),
                    alert_frames=fs.get("alert_frames", 3),
                )
            else:
                fire_path = model_cfg.get("fire_smoke", "")
                if not Path(fire_path).exists():
                    log("WARN", f"[{self.camera_id}] fire_smoke 模型不存在，略過: {fire_path}")
                else:
                    self._fire_det = FireSmokeDetector(
                        weight_path=fire_path,
                        conf=fs.get("conf", 0.5),
                        alert_frames=fs.get("alert_frames", 3),
                    )

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

        # ── ReID 狀態（跨攝影機滯留計時繼承）────────────────────────────────
        reid_cfg = global_cfg.get("reid", {})
        self._reid_update_interval  = reid_cfg.get("update_interval_sec", 10.0)
        self._enrolled_tids: set[int]         = set()   # 已 enroll 的本機 track_id
        self._reid_gids: dict[int, int]       = {}      # local tid → global_id
        self._last_reid_update: dict[int, float] = {}   # tid → 上次更新 embedding 的時間
        self._last_gallery_cleanup = 0.0

        # ── 輸出影片 ─────────────────────────────────────────────────────────
        self._writer           = None
        self._video_dir        = Path(out_cfg.get("video_dir", "outputs/videos"))
        self._save_video_local = out_cfg.get("save_video_local", True)
        self._save_video_rtsp  = out_cfg.get("save_video_rtsp", False)

        # ── 狀態（供 multistream display thread 讀取，不需加鎖）────────────
        self.fps:          float = 0.0
        self._fps_frames:  int   = 0    # 1 秒滾動視窗計數
        self._fps_t0:      float = 0.0
        self._fps_log_n:   int   = 0    # 每 10 秒輸出一次 log
        # 計時診斷（視窗平均）
        self._diag_read_sum:  float = 0.0
        self._diag_infer_sum: float = 0.0
        self._diag_n:         int   = 0
        self.source_failed: bool = False                  # RTSP + fallback 均失敗
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
            self.source_failed = True
            return

        w, h    = self._reader.get_size()
        src_fps = self._reader.get_fps()
        is_file = self._reader.is_file()

        # 輸出 fps = 實際推論 fps（只在 inference 幀寫入，避免跳幀複製造成慢動作）
        out_fps = max(1.0, src_fps / self._skip)
        if (is_file and self._save_video_local) or \
           (not is_file and self._save_video_rtsp):
            self._init_writer(w, h, out_fps)

        active = ", ".join(k for k, v in self._feat.items() if v) or "（無）"
        log("INFO", f"[{self.camera_id}] 開始推論 {w}x{h}@{src_fps:.1f}fps "
                    f"skip={self._skip} 功能=[{active}]")

        while not self._stop_event.is_set():
            t_read0 = time.time()
            ret, frame = self._reader.read()
            t_read1 = time.time()

            if not ret:
                if is_file:
                    log("INFO", f"[{self.camera_id}] 影片結束")
                else:
                    time.sleep(0.05)
                    continue
                break

            self._frame_n += 1
            read_ms = (t_read1 - t_read0) * 1000.0

            if self._frame_n % self._skip == 0:
                # Inference frame: run detection + draw + write to file
                t_inf0 = time.time()
                annotated = self._process(frame)
                infer_ms = (time.time() - t_inf0) * 1000.0
                self._last_annotated = annotated
                if self._writer is not None:
                    self._writer.write(annotated)
                self._diag_infer_sum += infer_ms
            else:
                # Skip frame: display only（不寫入影片，避免重複幀造成慢動作）
                infer_ms = 0.0
                if self._last_annotated is not None:
                    annotated = self._last_annotated
                else:
                    annotated = frame

            self._diag_read_sum += read_ms
            self._diag_n += 1

            # ── 1 秒滾動視窗 FPS（穩定、無逐幀抖動）────────────────────────
            now = time.time()
            self._fps_frames += 1
            if self._fps_t0 == 0.0:
                self._fps_t0 = now
            elif now - self._fps_t0 >= 1.0:
                self.fps = self._fps_frames / (now - self._fps_t0)
                self.status_info["fps"] = round(self.fps, 1)
                self._fps_frames = 0
                self._fps_t0 = now
                self._fps_log_n += 1
                if self._fps_log_n % 5 == 0:    # 每 5 秒 log 一次
                    avg_read  = self._diag_read_sum  / max(1, self._diag_n)
                    avg_infer = self._diag_infer_sum / max(1, self._diag_n)
                    log("INFO", f"[{self.camera_id}] [{self._mode}] "
                                f"fps={self.fps:.1f} skip={self._skip} "
                                f"read={avg_read:.0f}ms infer={avg_infer:.0f}ms")
                    self._diag_read_sum = self._diag_infer_sum = 0.0
                    self._diag_n = 0

            try:
                self._queue.put_nowait((self.camera_id, annotated, self.fps))
            except queue.Full:
                pass

        self._cleanup()

    # ── 推論核心 ─────────────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Step 1：偵測（Bus 批次路徑 or 直接推論路徑）
        if self._bus is not None:
            # ── Bus 模式：前處理 → 提交幀 → 等 batch 推論結果 ────────────
            preprocessed = (self._fall_det.preprocess(frame)
                            if self._fall_det else frame)
            future = self._bus.submit(self.camera_id, preprocessed, frame)
            try:
                raw = future.result(timeout=2.0)
            except Exception as e:
                log("WARN", f"[{self.camera_id}] InferenceBus 推論超時/失敗: {e}")
                raw = {"person": None, "luggage": None, "fire": None}

            all_persons: list[dict] = (
                self._fall_det.parse_result(raw["person"])
                if self._fall_det and raw.get("person") is not None else []
            )
            all_luggage: list[dict] = (
                self._luggage_det.parse_result(raw["luggage"], h, w, all_persons)
                if self._luggage_det and raw.get("luggage") is not None else []
            )
            all_fire_smoke: list[dict] = []
            if (self._fire_det and raw.get("fire") is not None
                    and self._roi.has_roi_for_feature("fire_smoke")):
                all_fire_smoke = self._fire_det.parse_result(raw["fire"])
        else:
            # ── 直接模式（向下相容，無 Bus）────────────────────────────────
            all_persons = []
            if self._fall_det is not None:
                all_persons = self._fall_det.detect(frame)

            all_luggage = []
            if self._luggage_det is not None:
                all_luggage = self._luggage_det.detect(frame, persons=all_persons)

            all_fire_smoke = []
            if self._fire_det is not None and self._roi.has_roi_for_feature("fire_smoke"):
                all_fire_smoke = self._fire_det.detect(frame)

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
        self._run_dwell_monitor(annotated, dwell_in_roi, frame)
        self._run_zone_counter(annotated, zone_in_roi)
        self._run_fall_logic(annotated, fall_in_roi, alert_persons)
        self._run_luggage_roll(annotated, lug_roll_roi, all_persons)
        self._run_size_classifier(annotated, lug_size_roi)
        self._run_fire_smoke(annotated, all_fire_smoke)

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

    def _run_dwell_monitor(self, annotated: np.ndarray,
                           persons: list[dict], frame: np.ndarray):
        if self._dwell_mon is None:
            return

        gallery = self._bus.gallery if self._bus is not None else None
        now = time.time()

        # ① 新人入鏡：查 gallery 看是否為跨攝影機滯留者
        if gallery is not None:
            for p in persons:
                tid = p.get("track_id", -1)
                if tid < 0 or tid in self._dwell_mon._first_seen:
                    continue  # 已在本機計時，跳過
                crop = self._extract_crop(frame, p["bbox"])
                if crop is None:
                    continue
                emb = self._bus.extract_reid(crop)
                if emb is None:
                    continue
                match = gallery.query(emb, debug_label=f"{self.camera_id}/tid={tid}")
                if match is not None:
                    self._dwell_mon.inherit_timer(tid, match.first_dwell_time)
                    # 繼承原始 gid，並標記已處理，避免後面 enroll 邏輯再產生新 gid
                    self._enrolled_tids.add(tid)
                    self._reid_gids[tid] = match.global_id
                    log("INFO",
                        f"[{self.camera_id}] ReID 匹配 "
                        f"local_tid={tid} gid={match.global_id} "
                        f"from={match.cam_id} "
                        f"繼承滯留={now - match.first_dwell_time:.0f}s")

        alerts = self._dwell_mon.update(persons)
        self._dwell_mon.draw(annotated, persons, reid_gids=self._reid_gids)
        self.status_info["dwell_active"] = len(alerts)

        # ② 達到門檻：入庫 / 定期更新 embedding
        if gallery is not None:
            for p in persons:
                tid = p.get("track_id", -1)
                if tid < 0:
                    continue
                dwell = self._dwell_mon.get_dwell(tid)
                if dwell < self._dwell_mon._alert_sec:
                    continue

                if tid not in self._enrolled_tids:
                    crop = self._extract_crop(frame, p["bbox"])
                    if crop is not None:
                        emb = self._bus.extract_reid(crop)
                        if emb is not None:
                            first_t = self._dwell_mon._first_seen.get(tid, now - dwell)
                            gid = gallery.enroll(emb, first_t, self.camera_id)
                            self._enrolled_tids.add(tid)
                            self._reid_gids[tid] = gid
                            log("INFO",
                                f"[{self.camera_id}] ReID enroll "
                                f"tid={tid} → gid={gid} dwell={dwell:.0f}s")
                else:
                    last_upd = self._last_reid_update.get(tid, 0.0)
                    if now - last_upd >= self._reid_update_interval:
                        gid = self._reid_gids.get(tid)
                        if gid is not None:
                            crop = self._extract_crop(frame, p["bbox"])
                            if crop is not None:
                                emb = self._bus.extract_reid(crop)
                                if emb is not None:
                                    gallery.update_embedding(gid, emb, self.camera_id)
                                    self._last_reid_update[tid] = now

            if now - self._last_gallery_cleanup > 60.0:
                gallery.cleanup()
                self._last_gallery_cleanup = now

        # ③ 告警觸發
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

    @staticmethod
    def _extract_crop(frame: np.ndarray, bbox: tuple,
                      pad: float = 0.1) -> "np.ndarray | None":
        """從 frame 裁出 person bbox（含 pad），太小的 crop 回傳 None。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None
        px, py = max(1, int(bw * pad)), max(1, int(bh * pad))
        return frame[max(0, y1 - py):min(h, y2 + py),
                     max(0, x1 - px):min(w, x2 + px)]

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

    def _run_fire_smoke(self, annotated: np.ndarray,
                        all_detections: list[dict]):
        if self._fire_det is None:
            return
        detections = self._in_roi(all_detections, "fire_smoke")

        alerts = self._fire_det.compute_alerts(detections)
        self._fire_det.draw(annotated, detections, alerts)

        for label, triggered in alerts.items():
            if not triggered:
                continue
            best = max(
                (d for d in detections if d["label"] == label),
                key=lambda d: d["conf"], default=None,
            )
            if best is None:
                continue
            rois = self._roi.get_containing_rois_for_feature(
                best["cx"], best["cy"], "fire_smoke")
            event_type = f"{label}_detected"
            fired = self._events.trigger(
                event_type=event_type,
                roi_id=rois[0] if rois else "global",
                severity="CRITICAL",
                confidence=best["conf"],
                frame=annotated,
                extra={"label": label},
            )
            if fired:
                self._push_alert(event_type,
                                 rois[0] if rois else "global", "CRITICAL")

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
