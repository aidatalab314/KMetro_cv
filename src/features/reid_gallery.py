import threading
import time

import numpy as np

from src.utils import log


def _normalize(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)


class GalleryRecord:
    __slots__ = ("global_id", "embedding", "first_dwell_time", "cam_id", "last_seen_time")

    def __init__(self, global_id: int, embedding: np.ndarray,
                 first_dwell_time: float, cam_id: str, last_seen_time: float):
        self.global_id        = global_id
        self.embedding        = embedding        # (512,) float32, L2 normalized
        self.first_dwell_time = first_dwell_time # wall-clock when first entered dwell zone
        self.cam_id           = cam_id
        self.last_seen_time   = last_seen_time


class ReIDGallery:
    """
    跨攝影機行人外觀特徵庫（單機共享，thread-safe）。

    只存放已達滯留門檻的人員 embedding。
    新路攝影機發現未知人員時，查詢此庫繼承原始計時，使告警不因跨鏡重置。

    相似度計算：cosine similarity（embedding 已 L2 normalize → dot product 即 cosine）。
    滑動平均更新：避免單幀噪音，外觀穩定性更好。
    TTL 機制：離站人員自動清除，避免庫膨脹與誤匹配。
    """

    def __init__(self, sim_threshold: float = 0.75, ttl_sec: float = 300.0):
        self._sim_threshold = sim_threshold
        self._ttl           = ttl_sec
        self._records: dict[int, GalleryRecord] = {}
        self._lock    = threading.Lock()
        self._next_gid = 0

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    def enroll(self, embedding: np.ndarray,
               first_dwell_time: float, cam_id: str) -> int:
        """首次達到滯留門檻時入庫。回傳 global_id。"""
        emb = _normalize(embedding)
        with self._lock:
            gid = self._next_gid
            self._next_gid += 1
            self._records[gid] = GalleryRecord(
                global_id=gid,
                embedding=emb,
                first_dwell_time=first_dwell_time,
                cam_id=cam_id,
                last_seen_time=time.time(),
            )
        return gid

    def query(self, embedding: np.ndarray,
              debug_label: str = "") -> "GalleryRecord | None":
        """
        尋找 cosine similarity 最高且超過門檻的記錄。
        過期（TTL）記錄不參與比對。
        """
        emb = _normalize(embedding)
        best: GalleryRecord | None = None
        best_sim = self._sim_threshold
        top_candidate: GalleryRecord | None = None
        top_sim = -1.0
        now = time.time()
        with self._lock:
            for rec in self._records.values():
                if now - rec.last_seen_time > self._ttl:
                    continue
                sim = float(np.dot(emb, rec.embedding))
                if sim > top_sim:
                    top_sim = sim
                    top_candidate = rec
                if sim > best_sim:
                    best_sim = sim
                    best = rec
        # 始終記錄最高分（不論是否過門檻），方便調整 threshold
        if top_candidate is not None:
            hit = best is not None
            log("DEBUG",
                f"[ReIDGallery] query {debug_label} "
                f"best_sim={top_sim:.3f} threshold={self._sim_threshold:.2f} "
                f"gid={top_candidate.global_id} cam={top_candidate.cam_id} "
                f"{'✓ match' if hit else '✗ miss'}")
        return best

    def update_embedding(self, global_id: int,
                         embedding: np.ndarray, cam_id: str):
        """
        以滑動平均更新已入庫人員的 embedding（每 N 秒呼叫一次）。
        0.7 舊 + 0.3 新：穩定外觀特徵，防單幀突變污染庫。
        """
        emb = _normalize(embedding)
        with self._lock:
            if global_id not in self._records:
                return
            rec = self._records[global_id]
            rec.embedding       = _normalize(0.7 * rec.embedding + 0.3 * emb)
            rec.cam_id          = cam_id
            rec.last_seen_time  = time.time()

    def cleanup(self):
        """清除超過 TTL 的記錄（由 CameraWorker 定期呼叫）。"""
        now = time.time()
        with self._lock:
            expired = [gid for gid, rec in self._records.items()
                       if now - rec.last_seen_time > self._ttl]
            for gid in expired:
                del self._records[gid]
        if expired:
            log("DEBUG", f"[ReIDGallery] 清除 {len(expired)} 筆過期記錄，剩餘 {len(self)}")

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
