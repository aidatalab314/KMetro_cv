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

        threading.Thread(target=self._run, daemon=True,
                         name="inference-bus").start()
        log("INFO", f"[InferenceBus] 啟動完成  cameras={self._cam_order}  "
                    f"person_imgsz={self._person_imgsz}  timeout={batch_timeout*1000:.0f}ms")

    # ── 外部介面（Worker 呼叫）──────────────────────────────────────────────

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

                # person + ByteTrack（batch_size=N, 順序固定）
                pr = self._person_model.track(
                    self._last_pre,
                    conf=self._person_conf, imgsz=self._person_imgsz,
                    classes=[0], verbose=False,
                    persist=True, tracker="bytetrack.yaml",
                )
                # luggage + ByteTrack
                lr = self._luggage_model.track(
                    self._last_orig,
                    conf=self._luggage_conf, imgsz=self._luggage_imgsz,
                    verbose=False,
                    persist=True, tracker="bytetrack.yaml",
                )
                # fire/smoke（不需 tracking）
                fr: list = ([None] * self._n if self._fire_model is None
                            else self._fire_model(
                                self._last_orig,
                                conf=self._fire_conf, imgsz=self._fire_imgsz,
                                verbose=False,
                            ))

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
                    log("INFO", f"[InferenceBus] batch#{self._batch_count}  "
                                f"avg_infer={avg_i:.0f}ms  avg_wait={avg_w:.0f}ms  "
                                f"n_cam={self._n}  active={len(futures_map)}")
                    self._t_infer_sum = self._t_wait_sum = 0.0
                    self._batch_count = 0

            except Exception as e:
                log("ERROR", f"[InferenceBus] 推論失敗: {e}")
                for _, fut in futures_map:
                    if not fut.done():
                        fut.set_exception(e)
