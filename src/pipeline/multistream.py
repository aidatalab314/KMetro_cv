"""
KMetro CV — 多路攝影機主入口

每個 panel 版面：
  ┌──────────────────────────────────┐
  │  告警列（44px）紅/橘/深背景      │
  │  !! 事件類型  ROI  時間          │
  ├──────────────────────────────────┤
  │                                  │
  │  影像內容（縮放至剩餘高度）      │
  │                                  │
  ├──────────────────────────────────┤
  │  資訊列第1行（29px）             │
  │  攝影機名稱  fps  功能清單       │
  │  資訊列第2行（29px）             │
  │  zone人數  dwell滯留數  追蹤狀態 │
  └──────────────────────────────────┘

用法：
    python src/pipeline/multistream.py                            # auto：探測 RTSP，不可達則本地影片
    python src/pipeline/multistream.py --source rtsp             # 強制 RTSP（ROI key = camera_id）
    python src/pipeline/multistream.py --source local            # 強制本地影片（ROI key = camera_id_local）
    python src/pipeline/multistream.py --mode op                 # op 模式，headless
    python src/pipeline/multistream.py --cameras cam_platform_north,cam_escalator_up
    python src/pipeline/multistream.py --reset-roi cam_platform_north
    python src/pipeline/multistream.py --reset-roi all
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_yaml, log
from src.roi.roi_manager import has_roi_record, draw_roi_interactive, save_roi_record
from src.pipeline.camera_worker import CameraWorker


def _is_rtsp_url(url: str) -> bool:
    return isinstance(url, str) and (url.startswith("rtsp://") or url.startswith("rtmp://"))


def _probe_rtsp(url: str, timeout: float = 6.0) -> bool:
    """背景執行緒探測 RTSP 是否可連接，timeout 秒內有回應回傳 True。"""
    reachable = [False]

    def _try():
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, _ = cap.read()
            reachable[0] = ret
        cap.release()

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return reachable[0]


def _roi_key(camera_id: str, use_rtsp: bool) -> str:
    """ROI record key：RTSP → camera_id_rtsp，本地影片 → camera_id_local。"""
    return f"{camera_id}_rtsp" if use_rtsp else f"{camera_id}_local"

RECORDS_PATH = "configs/roi_records.json"
CONFIG_PATH  = "configs/cameras.yaml"

# 版面常數
_ALERT_H = 44   # 頂部告警列高度
_INFO_H  = 58   # 底部資訊列高度（2行 × 29px）


# ── ROI 設定流程 ──────────────────────────────────────────────────────────────

def ensure_roi(cam_cfg: dict, records_path: str, reset: bool = False,
               use_rtsp: bool = False):
    camera_id = cam_cfg["id"]
    key       = _roi_key(camera_id, use_rtsp)
    if reset or not has_roi_record(key, records_path):
        src_type = "RTSP" if use_rtsp else "local"
        action   = "重新繪製" if reset else "尚無紀錄，開啟繪製工具"
        log("ROI", f"{camera_id}（{src_type}）{action}  [key={key}]")
        source  = (cam_cfg["source"] if use_rtsp
                   else cam_cfg.get("fallback") or cam_cfg["source"])
        title = f"{camera_id}  [{src_type}]"
        try:
            rois = draw_roi_interactive(str(source),
                                        enabled_features=None,   # 永遠顯示全部功能
                                        title=title)
        except RuntimeError as e:
            log("ERROR", f"[{camera_id}] ROI 畫面擷取失敗: {e}")
            rois = []
        if rois:
            save_roi_record(key, rois, records_path)
        else:
            log("ROI", f"{camera_id} 未設定 ROI（跳過所有功能）")
    else:
        key_hint = "(RTSP)" if use_rtsp else "(local)"
        log("ROI", f"{camera_id} 已有 ROI 設定，直接載入 {key_hint}")


# ── Panel 版面繪製 ────────────────────────────────────────────────────────────

_SEV_COLOR = {
    "CRITICAL": (0, 0, 200),
    "WARNING":  (0, 110, 200),
    "INFO":     (100, 70, 0),
}

# ASCII-only: cv2.putText does not support Unicode/CJK
_EVT_LABEL = {
    "fall_detected":         "!! FALL DETECTED",
    "luggage_roll_detected": "!! LUGGAGE ROLLING",
    "luggage_roll":          "!! LUGGAGE ROLLING",
    "dwell_alert":           ">> DWELL ALERT",
    "crowd_alert":           ">> CROWD ALERT",
    "large_luggage":         ">> LARGE LUGGAGE",
    "large_luggage_detected":">> LARGE LUGGAGE",
}


def _draw_alert_bar(panel: np.ndarray, worker: CameraWorker, pw: int):
    """頂部告警列：有告警 → 深色背景 + 事件資訊；無告警 → 攝影機名稱。"""
    alerts = list(worker.recent_alerts)  # thread-safe snapshot

    if alerts:
        last  = alerts[-1]
        sev   = last.get("severity", "INFO")
        bg    = _SEV_COLOR.get(sev, (40, 40, 40))
        cv2.rectangle(panel, (0, 0), (pw, _ALERT_H), bg, -1)
        label = _EVT_LABEL.get(last["event"], last["event"])
        detail = f"  {last['detail']}" if last.get("detail") else ""
        text  = f"{label}  {last['roi']}  {last['time']}{detail}"
        cv2.putText(panel, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    else:
        cv2.rectangle(panel, (0, 0), (pw, _ALERT_H), (22, 22, 22), -1)
        cv2.putText(panel, worker.camera_id,   # camera_name may be non-ASCII
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (90, 90, 90), 1)

    # 分隔線
    cv2.line(panel, (0, _ALERT_H - 1), (pw, _ALERT_H - 1), (55, 55, 55), 1)


def _draw_info_bar(panel: np.ndarray, worker: CameraWorker,
                   pw: int, ph: int):
    """底部資訊列：2 行資訊。"""
    info  = worker.status_info
    bar_y = ph - _INFO_H

    cv2.rectangle(panel, (0, bar_y), (pw, ph), (18, 18, 18), -1)
    cv2.line(panel, (0, bar_y), (pw, bar_y), (55, 55, 55), 1)

    fps_str   = f"{info.get('fps', 0):.1f}fps"
    track_str = "track:ON" if info.get("person_track") else "track:OFF"
    feats_str = "|".join(f.split("_")[0][:4] for f in info.get("features", []))
    row1 = f"{info.get('name', '')}  {fps_str}  skip={info.get('skip', 1)}  {track_str}"
    cv2.putText(panel, row1, (10, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

    zone_str  = f"zone:{info.get('zone_count', 0)}"
    dwell_str = f"dwell_alert:{info.get('dwell_active', 0)}"
    row2 = f"{zone_str}  {dwell_str}  [{feats_str}]"
    cv2.putText(panel, row2, (10, bar_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (110, 110, 110), 1)


def make_panel(frame: np.ndarray, worker: CameraWorker,
               panel_h: int) -> np.ndarray:
    """
    建立單路攝影機 panel：告警列 + 縮放影像 + 資訊列。
    固定寬度 = panel_h * 16/9，與 waiting/nosource panel 一致，mosaic 格子均等。
    非 16:9 來源以 letterbox 置中填入。
    """
    pw        = int(panel_h * 16 / 9)
    content_h = panel_h - _ALERT_H - _INFO_H
    h, w      = frame.shape[:2]
    scale     = min(pw / w, content_h / h)
    sw, sh    = max(1, int(w * scale)), max(1, int(h * scale))

    panel  = np.zeros((panel_h, pw, 3), np.uint8)
    scaled = cv2.resize(frame, (sw, sh))
    x_off  = (pw - sw) // 2
    panel[_ALERT_H: _ALERT_H + sh, x_off: x_off + sw] = scaled

    _draw_alert_bar(panel, worker, pw)
    _draw_info_bar(panel, worker, pw, panel_h)

    return panel


def _make_waiting_panel(cam_name: str, panel_h: int) -> np.ndarray:
    """尚未收到第一幀時的佔位 panel。"""
    pw = int(panel_h * 16 / 9)
    panel = np.zeros((panel_h, pw, 3), np.uint8)
    cv2.rectangle(panel, (0, 0), (pw, _ALERT_H), (22, 22, 22), -1)
    cv2.putText(panel, cam_name, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (70, 70, 70), 1)
    cv2.putText(panel, "Connecting...",
                (pw // 2 - 60, panel_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 60), 1)
    return panel


def _make_nosource_panel(cam_name: str, panel_h: int) -> np.ndarray:
    """RTSP 與 fallback 均無法開啟時的黑畫面 panel。"""
    pw = int(panel_h * 16 / 9)
    panel = np.zeros((panel_h, pw, 3), np.uint8)
    cv2.rectangle(panel, (0, 0), (pw, _ALERT_H), (30, 10, 10), -1)
    cv2.putText(panel, cam_name, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 50, 50), 1)
    cy = panel_h // 2
    cv2.putText(panel, "NO SOURCE",
                (pw // 2 - 70, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (60, 40, 40), 2)
    cv2.putText(panel, "RTSP and fallback unavailable",
                (pw // 2 - 130, cy + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 35, 35), 1)
    return panel


# ── Mosaic 工具 ───────────────────────────────────────────────────────────────

def build_mosaic(panels: list[np.ndarray]) -> np.ndarray:
    n = len(panels)
    if n == 1:
        return panels[0]
    if n == 2:
        return np.hstack(_eq_h(panels))
    row1 = np.hstack(_eq_h(panels[:2]))
    p2   = list(panels[2:4])
    if len(p2) == 1:
        p2.append(np.zeros_like(panels[0]))
    row2 = np.hstack(_eq_h(p2))
    tw   = max(row1.shape[1], row2.shape[1])
    return np.vstack([_pad_w(row1, tw), _pad_w(row2, tw)])


def _eq_h(panels: list[np.ndarray]) -> list[np.ndarray]:
    mh = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        if p.shape[0] < mh:
            pad = np.zeros((mh - p.shape[0], p.shape[1], 3), np.uint8)
            p   = np.vstack([p, pad])
        out.append(p)
    return out


def _pad_w(img: np.ndarray, target_w: int) -> np.ndarray:
    if img.shape[1] >= target_w:
        return img
    pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), np.uint8)
    return np.hstack([img, pad])


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KMetro CV Multistream")
    parser.add_argument("--cameras",   type=str, default=None)
    parser.add_argument("--mode",      type=str, default="dev",
                        choices=["dev", "op"])
    parser.add_argument("--source",    type=str, default="auto",
                        choices=["auto", "local", "rtsp"])
    parser.add_argument("--config",    type=str, default=CONFIG_PATH)
    parser.add_argument("--reset-roi", type=str, default=None,
                        metavar="CAM_ID|all")
    args = parser.parse_args()

    cfg         = load_yaml(args.config)
    all_cameras = cfg.get("cameras", [])
    panel_h     = cfg.get("display", {}).get("panel_h", 540)
    win_title   = cfg.get("display", {}).get("window", "KMetro CV — Multistream")

    if args.cameras:
        req     = set(args.cameras.split(","))
        cameras = [c for c in all_cameras if c["id"] in req]
        missing = req - {c["id"] for c in cameras}
        if missing:
            log("ERROR", f"找不到 camera: {missing}")
            return
    else:
        cameras = all_cameras

    if not cameras:
        log("ERROR", "未設定任何攝影機")
        return

    # ── 決定每台攝影機來源（RTSP 或本地影片）──────────────────────────────
    use_rtsp_per_cam: dict[str, bool] = {}
    for cam in cameras:
        cid = cam["id"]
        if args.source == "rtsp":
            use_rtsp_per_cam[cid] = True
        elif args.source == "local":
            use_rtsp_per_cam[cid] = False
        else:  # auto：探測 RTSP 是否可達
            if _is_rtsp_url(cam.get("source", "")):
                log("INFO", f"[{cid}] 探測 RTSP...")
                reachable = _probe_rtsp(cam["source"])
                use_rtsp_per_cam[cid] = reachable
                log("INFO" if reachable else "WARN",
                    f"[{cid}] RTSP {'可達，使用攝影機' if reachable else '不可達，切換本地影片'}")
            else:
                use_rtsp_per_cam[cid] = False

    reset_ids = set()
    if args.reset_roi:
        reset_ids = ({c["id"] for c in cameras} if args.reset_roi == "all"
                     else set(args.reset_roi.split(",")))
    for cam in cameras:
        ensure_roi(cam, RECORDS_PATH, reset=(cam["id"] in reset_ids),
                   use_rtsp=use_rtsp_per_cam[cam["id"]])

    frame_queues = {cam["id"]: queue.Queue(maxsize=2) for cam in cameras}
    workers: list[CameraWorker] = []

    for cam in cameras:
        cid      = cam["id"]
        use_rtsp = use_rtsp_per_cam[cid]
        src      = (cam["source"] if use_rtsp
                    else cam.get("fallback") or cam["source"])
        roi_k    = _roi_key(cid, use_rtsp)
        w = CameraWorker(cam_cfg=cam, global_cfg=cfg, source=src,
                         mode=args.mode, records_path=RECORDS_PATH,
                         roi_key=roi_k,
                         frame_queue=frame_queues[cid])
        w.start()
        workers.append(w)

    rtsp_cams  = [c["id"] for c in cameras if use_rtsp_per_cam[c["id"]]]
    local_cams = [c["id"] for c in cameras if not use_rtsp_per_cam[c["id"]]]
    log("INFO", f"已啟動 {len(workers)} 路攝影機 [{args.mode}]  "
                f"rtsp={rtsp_cams}  local={local_cams}")

    worker_map = {w.camera_id: w for w in workers}

    out_cfg     = cfg.get("output", {})
    save_mosaic = out_cfg.get("save_mosaic", False)
    mosaic_fps  = float(out_cfg.get("mosaic_fps", 25.0))
    mosaic_dir  = Path(out_cfg.get("video_dir", "outputs/videos"))
    mosaic_path = mosaic_dir / f"mosaic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    if args.mode == "op":
        mosaic_rec: _MosaicOpThread | None = None
        if save_mosaic:
            mosaic_rec = _MosaicOpThread(
                cameras, worker_map, frame_queues, panel_h,
                str(mosaic_path), fps=mosaic_fps,
            )
            mosaic_rec.start()
        try:
            while any(w.is_alive() for w in workers):
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if mosaic_rec:
                mosaic_rec.stop()
            _shutdown(workers)
        return

    # ── Dev 模式顯示迴圈 ──────────────────────────────────────────────────────
    last_frames: dict[str, tuple] = {}
    mosaic_writer: cv2.VideoWriter | None = None
    mosaic_next_t: float = 0.0   # 下一次寫幀的時間戳
    _win_initialized = False

    try:
        while True:
            panels = []
            got_new_frame = False
            for cam in cameras:
                cid = cam["id"]
                try:
                    _, frame, fps = frame_queues[cid].get_nowait()
                    last_frames[cid] = (frame, fps)
                    got_new_frame = True
                except queue.Empty:
                    pass

                wk = worker_map[cid]
                if cid in last_frames:
                    frame, _ = last_frames[cid]
                    panels.append(make_panel(frame, wk, panel_h))
                elif wk.source_failed:
                    panels.append(_make_nosource_panel(cid, panel_h))
                else:
                    panels.append(_make_waiting_panel(cid, panel_h))

            mosaic = build_mosaic(panels)

            # mosaic 錄影：等所有 camera 都有真實影格 + 有新幀才寫入
            # （避免 waiting panel 尺寸不一致 / 重複寫舊幀造成慢動作）
            if save_mosaic and got_new_frame and len(last_frames) == len(cameras):
                now = time.time()
                if mosaic_writer is None:
                    h, w = mosaic.shape[:2]
                    mosaic_dir.mkdir(parents=True, exist_ok=True)
                    mosaic_writer = cv2.VideoWriter(
                        str(mosaic_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        mosaic_fps, (w, h),
                    )
                    mosaic_next_t = now
                    log("INFO", f"[Mosaic] 開始錄影: {mosaic_path}")
                if now >= mosaic_next_t:
                    mosaic_writer.write(mosaic)
                    mosaic_next_t += 1.0 / mosaic_fps

            if not _win_initialized:
                cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_title, 1280, 720)
                _win_initialized = True
            cv2.imshow(win_title, mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            # 若所有 worker 都結束且沒有任何來源失敗（影片播完），才自動退出
            # 有 source_failed 的 worker：保持顯示 NO SOURCE 黑畫面，等使用者按 q
            if (not any(w.is_alive() for w in workers) and
                    not any(w.source_failed for w in workers)):
                log("INFO", "所有 worker 已結束")
                break

    except KeyboardInterrupt:
        pass
    finally:
        if mosaic_writer is not None:
            mosaic_writer.release()
            log("INFO", f"[Mosaic] 錄影完成: {mosaic_path}")
        cv2.destroyAllWindows()
        _shutdown(workers)


# ── Op 模式 Mosaic 錄影 thread ────────────────────────────────────────────────

class _MosaicOpThread(threading.Thread):
    """
    Op 模式專用：消費所有 camera frame queue → 合成 mosaic → 寫入影片。
    Dev 模式改由 display loop 直接寫入，不用此 thread。
    """

    def __init__(self, cameras: list, worker_map: dict,
                 frame_queues: dict, panel_h: int,
                 out_path: str, fps: float = 25.0):
        super().__init__(daemon=True, name="mosaic-recorder")
        self._cameras     = cameras
        self._worker_map  = worker_map
        self._queues      = frame_queues
        self._panel_h     = panel_h
        self._out_path    = out_path
        self._fps         = fps
        self._stop        = threading.Event()
        self._writer: cv2.VideoWriter | None = None

    def run(self):
        interval    = 1.0 / self._fps
        last_frames: dict = {}

        while not self._stop.is_set():
            t0 = time.time()

            got_new = False
            for cam in self._cameras:
                cid = cam["id"]
                try:
                    _, frame, _ = self._queues[cid].get_nowait()
                    last_frames[cid] = frame
                    got_new = True
                except queue.Empty:
                    pass

            # 等所有 camera 都有真實影格 + 本輪有新幀才寫入
            if got_new and len(last_frames) == len(self._cameras):
                panels = []
                for cam in self._cameras:
                    cid = cam["id"]
                    wk  = self._worker_map[cid]
                    panels.append(make_panel(last_frames[cid], wk, self._panel_h))

                mosaic = build_mosaic(panels)

                if self._writer is None:
                    h, w = mosaic.shape[:2]
                    Path(self._out_path).parent.mkdir(parents=True, exist_ok=True)
                    self._writer = cv2.VideoWriter(
                        self._out_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self._fps, (w, h),
                    )
                    log("INFO", f"[Mosaic] 開始錄影: {self._out_path}")

                self._writer.write(mosaic)

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

        if self._writer:
            self._writer.release()
            log("INFO", f"[Mosaic] 錄影完成: {self._out_path}")

    def stop(self):
        self._stop.set()
        self.join(timeout=5.0)


def _shutdown(workers):
    log("INFO", "正在關閉所有 worker...")
    for w in workers:
        w.stop()
    for w in workers:
        w.join(timeout=5.0)
    log("INFO", "系統已關閉")


if __name__ == "__main__":
    main()
