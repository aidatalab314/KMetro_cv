"""
InferenceBus — 單 GPU 推論執行緒，收集多路攝影機幀後批次推論。

原本架構：4 路 × 3 模型 = 12 次 GPU 呼叫（各自串行）
優化後  ：3 次 batch GPU 呼叫（batch_size=N cameras）

ByteTrack 一致性保證：
  - 永遠以固定順序（cameras 列表順序）打 full batch
  - 尚未收到幀的攝影機使用最後已知幀補位
  - model.predictor.trackers[i] 永遠對應第 i 路，狀態不錯位
"""

import threading
import concurrent.futures
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.utils import log


class InferenceBus:
    """
    共用 GPU 推論 hub。
    Worker 呼叫 submit() 非阻塞提交幀，等 Future 結果。
    單一背景執行緒湊齊 batch → 三模型串行 batch 推論 → 分發結果。
    """

    def __init__(self, cameras: list[dict], cfg: dict, batch_timeout: float = 0.040):
        """
        cameras     : 與 multistream 相同的 cameras 列表（決定 batch 順序）
        cfg         : 合併後的 cameras.yaml 設定
        batch_timeout: 等待湊滿 batch 的最大秒數（預設 40ms）
        """
        n = len(cameras)
        self._n         = n
        self._cam_order = [c["id"] for c in cameras]
        self._cam_idx   = {c["id"]: i for i, c in enumerate(cameras)}
        self._timeout   = batch_timeout

        # ── 載入共用模型（所有路共享，不重複載入）─────────────────────────
        det_cfg    = cfg.get("detector", {})
        feat_cfg   = cfg.get("features", {})
        models_cfg = cfg.get("models", {})

        person_path  = models_cfg.get("person",     "models/fall_detection/yolo12l.pt")
        luggage_path = models_cfg.get("luggage",    "models/luggage/yolo_luggage_best.pt")
        fire_path    = models_cfg.get("fire_smoke", "")

        log("INFO", "[InferenceBus] 載入共用模型（Level 2：單一 TRT context）...")
        self._person_model  = YOLO(person_path)
        self._luggage_model = YOLO(luggage_path)
        self._fire_model    = (YOLO(fire_path)
                               if fire_path and Path(fire_path).exists() else None)
        log("INFO", "[InferenceBus] 模型就緒  "
                    f"person={Path(person_path).name}  "
                    f"luggage={Path(luggage_path).name}  "
                    f"fire={'OK' if self._fire_model else '略過（檔案不存在）'}")

        # ── 推論參數 ────────────────────────────────────────────────────────
        fd_cfg = feat_cfg.get("fall_detector", {})
        fs_cfg = feat_cfg.get("fire_smoke", {})
        self._person_conf   = det_cfg.get("conf", 0.4)
        self._person_imgsz  = fd_cfg.get("imgsz", det_cfg.get("imgsz", 640))
        self._luggage_conf  = det_cfg.get("conf", 0.4)
        self._luggage_imgsz = det_cfg.get("imgsz", 640)
        self._fire_conf     = fs_cfg.get("conf", 0.5)
        self._fire_imgsz    = det_cfg.get("imgsz", 640)

        # ── 批次協調 ────────────────────────────────────────────────────────
        # slots[i] = (preprocessed_frame, orig_frame, future) | None
        self._slots: list = [None] * n
        self._lock  = threading.Lock()
        self._event = threading.Event()

        # 最後已知幀（保持 full-batch，維持 ByteTrack 位置一致性）
        self._last_pre:  list[np.ndarray | None] = [None] * n
        self._last_orig: list[np.ndarray | None] = [None] * n
        self._ready = False  # 等所有路都有第一幀才啟動批次推論

        # ── 計時 log ────────────────────────────────────────────────────────
        self._batch_count = 0
        self._t_infer_sum = 0.0
        self._t_wait_sum  = 0.0

        # ── 推論模式：固定串行，保證每路 ByteTrack 狀態獨立 ─────────────────
        # batch mode 下 model.track([cam0,cam1,cam2,cam3]) 共用單一 tracker，
        # 跨攝影機畫面混入同一流 → track_id 大量變 -1，dwell/fall 失效。
        # TRT engine (batch=N) 也建議串行以維持 track 正確性；僅在確認正確後再改 False。
        self._seq_mode = True
        # 每路獨立的 ByteTrack 狀態（串行模式用）
        self._seq_person_trackers:  list = [[] for _ in range(n)]
        self._seq_luggage_trackers: list = [[] for _ in range(n)]

        threading.Thread(target=self._run, daemon=True,
                         name="inference-bus").start()
        # ── ReID extractor + gallery（跨攝影機滯留繼承）────────────────────
        from src.features.reid_gallery import ReIDGallery
        reid_cfg = cfg.get("reid", {})
        self._reid_extractor = None
        self.gallery: "ReIDGallery | None" = None
        if reid_cfg.get("enabled", False):
            try:
                import torchreid as _torchreid
            except ImportError as e:
                log("WARN", f"[InferenceBus] reid.enabled=true 但 torchreid 無法載入（{e}）"
                            "；ReID 停用。請安裝：pip install torchreid gdown tensorboard")
                _torchreid = None

            if _torchreid is not None:
                _dev = str(det_cfg.get("device", "0"))
                if _dev == "mps":
                    _reid_dev = "cpu"   # torchreid MPS 支援不穩定
                elif _dev.isdigit():
                    _reid_dev = f"cuda:{_dev}"
                else:
                    _reid_dev = _dev    # "cuda" / "cpu"
                _model_path = reid_cfg.get("model_path", "") or ""
                self._reid_extractor = _torchreid.utils.FeatureExtractor(
                    model_name="osnet_ain_x1_0",
                    model_path=_model_path,
                    device=_reid_dev,
                )
                self.gallery = ReIDGallery(
                    sim_threshold=reid_cfg.get("sim_threshold", 0.75),
                    ttl_sec=reid_cfg.get("ttl_sec", 300.0),
                )
                log("INFO", f"[InferenceBus] ReID extractor 就緒  "
                            f"device={_reid_dev}  sim_threshold={reid_cfg.get('sim_threshold', 0.75)}  "
                            f"ttl={reid_cfg.get('ttl_sec', 300):.0f}s")

        log("INFO", f"[InferenceBus] 啟動完成  cameras={self._cam_order}  "
                    f"person_imgsz={self._person_imgsz}  timeout={batch_timeout*1000:.0f}ms")

    # ── 外部介面（Worker 呼叫）──────────────────────────────────────────────

    def extract_reid(self, crop_bgr: np.ndarray) -> "np.ndarray | None":
        """提取 person crop 的外觀 embedding（512d numpy array）。ReID 未啟用時回傳 None。"""
        if self._reid_extractor is None:
            return None
        return self._reid_extractor([crop_bgr]).cpu().numpy()[0]

    def submit(self, cam_id: str,
               preprocessed: np.ndarray,
               orig: np.ndarray) -> "concurrent.futures.Future[dict]":
        """
        Worker 提交一幀並取得 Future。
        Future 結果格式：{"person": Results, "luggage": Results, "fire": Results | None}
        """
        idx = self._cam_idx[cam_id]
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._slots[idx] = (preprocessed, orig, fut)
            full = all(s is not None for s in self._slots)
        if full:
            self._event.set()  # 提前喚醒（不等 timeout）
        return fut

    # ── 推論方法 ──────────────────────────────────────────────────────────────

    def _infer(self, pre_frames: list, ori_frames: list):
        """
        嘗試 batch 推論；若 TRT engine 不支援目前 batch size 則自動切換至串行模式。
        回傳 (pr_list, lr_list, fr_list)，各長度為 self._n。
        """
        if self._seq_mode:
            return self._infer_seq(pre_frames, ori_frames)

        try:
            pr = self._person_model.track(
                pre_frames,
                conf=self._person_conf, imgsz=self._person_imgsz,
                classes=[0], verbose=False,
                persist=True, tracker="bytetrack.yaml",
            )
            lr = self._luggage_model.track(
                ori_frames,
                conf=self._luggage_conf, imgsz=self._luggage_imgsz,
                verbose=False,
                persist=True, tracker="bytetrack.yaml",
            )
            fr = ([None] * self._n if self._fire_model is None
                  else self._fire_model(
                      ori_frames,
                      conf=self._fire_conf, imgsz=self._fire_imgsz,
                      verbose=False,
                  ))
            return pr, lr, fr

        except RuntimeError as e:
            err = str(e)
            if "not equal to" in err or "max model size" in err or "batch" in err.lower():
                self._seq_mode = True
                log("WARN",
                    "[InferenceBus] TRT engine batch_size=1，自動切換串行模式\n"
                    "  tracking 功能維持正常，但吞吐量較低\n"
                    "  建議重新 export engine（batch=4 imgsz=640）後重啟：\n"
                    "    python -c \"from ultralytics import YOLO; "
                    "YOLO('models/fall_detection/yolo12l.pt').export("
                    "format='engine', batch=4, imgsz=640, device=0, half=True)\"\n"
                    "    python -c \"from ultralytics import YOLO; "
                    "YOLO('models/luggage/yolo_luggage_best.pt').export("
                    "format='engine', batch=4, imgsz=640, device=0, half=True)\"")
                return self._infer_seq(pre_frames, ori_frames)
            raise

    def _infer_seq(self, pre_frames: list, ori_frames: list):
        """
        串行 batch=1 推論，逐路執行 model.track() 並隔離各路 ByteTrack 狀態。
        保證 track_id 按攝影機正確延續（不互相污染）。
        """
        pr, lr, fr = [], [], []
        for i in range(self._n):
            pr.append(self._track_one(
                self._person_model, pre_frames[i],
                self._seq_person_trackers, i,
                conf=self._person_conf, imgsz=self._person_imgsz, classes=[0],
            ))
            lr.append(self._track_one(
                self._luggage_model, ori_frames[i],
                self._seq_luggage_trackers, i,
                conf=self._luggage_conf, imgsz=self._luggage_imgsz,
            ))
            fr.append(
                self._fire_model(
                    [ori_frames[i]], conf=self._fire_conf,
                    imgsz=self._fire_imgsz, verbose=False,
                )[0] if self._fire_model else None
            )
        return pr, lr, fr

    def _track_one(self, model, frame: np.ndarray,
                   tracker_cache: list, cam_idx: int, **kwargs) -> object:
        """
        以 cam_idx 專屬的 ByteTrack 狀態執行單幀 track()。
        透過 save/restore model.predictor.trackers 隔離各路狀態。
        """
        pred  = getattr(model, 'predictor', None)
        saved = None

        if pred is not None and hasattr(pred, 'trackers'):
            saved = list(pred.trackers)
            pred.trackers = list(tracker_cache[cam_idx])

        result = model.track(
            [frame], persist=True, tracker="bytetrack.yaml",
            verbose=False, **kwargs,
        )[0]

        # 取回更新後的 tracker 狀態
        pred2 = getattr(model, 'predictor', None)
        if pred2 is not None and hasattr(pred2, 'trackers'):
            tracker_cache[cam_idx] = list(pred2.trackers)
            if saved is not None:
                pred2.trackers = saved

        return result

    # ── 推論執行緒 ────────────────────────────────────────────────────────────

    def _collect_batch(self) -> list:
        """阻塞直到所有 slot 滿或超時，返回 slots snapshot 並清空。"""
        self._event.wait(timeout=self._timeout)
        with self._lock:
            slots = list(self._slots)
            self._slots = [None] * self._n
            self._event.clear()
        return slots

    def _run(self):
        while True:
            t_wait0 = time.time()
            slots   = self._collect_batch()
            t_wait1 = time.time()

            # 更新最後已知幀；收集有 future 的 index
            futures_map: list[tuple[int, concurrent.futures.Future]] = []
            any_new = False
            for i, s in enumerate(slots):
                if s is not None:
                    self._last_pre[i]  = s[0]
                    self._last_orig[i] = s[1]
                    futures_map.append((i, s[2]))
                    any_new = True

            if not any_new:
                continue

            # 啟動前等所有路至少送過一幀（避免 ByteTrack 因 zero batch 出錯）
            if not self._ready:
                if all(lf is not None for lf in self._last_pre):
                    self._ready = True
                    log("INFO", "[InferenceBus] 所有路就緒，進入批次推論模式")
                else:
                    # 尚未就緒：以空結果讓 worker 繼續（跳過功能）
                    for _, fut in futures_map:
                        if not fut.done():
                            fut.set_result({"person": None, "luggage": None, "fire": None})
                    continue

            # ── batch 推論（3 模型串行，每次 N 路一起）──────────────────────
            try:
                t_inf0 = time.time()

                pr, lr, fr = self._infer(self._last_pre, self._last_orig)

                t_inf1 = time.time()

                # 分發結果給各 worker 的 future
                for i, fut in futures_map:
                    fut.set_result({
                        "person":  pr[i],
                        "luggage": lr[i],
                        "fire":    fr[i],
                    })

                # 計時 log（每 50 batch 印一次）
                self._batch_count  += 1
                self._t_infer_sum  += (t_inf1  - t_inf0)  * 1000
                self._t_wait_sum   += (t_wait1 - t_wait0) * 1000
                if self._batch_count % 50 == 0:
                    avg_i = self._t_infer_sum / self._batch_count
                    avg_w = self._t_wait_sum  / self._batch_count
                    mode  = "seq" if self._seq_mode else f"batch×{self._n}"
                    log("INFO", f"[InferenceBus] batch#{self._batch_count}  "
                                f"avg_infer={avg_i:.0f}ms  avg_wait={avg_w:.0f}ms  "
                                f"mode={mode}")
                    self._t_infer_sum = self._t_wait_sum = 0.0
                    self._batch_count = 0

            except Exception as e:
                log("ERROR", f"[InferenceBus] 推論失敗: {e}")
                for _, fut in futures_map:
                    if not fut.done():
                        fut.set_exception(e)
