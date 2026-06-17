"""
Multistream worker 整合測試（無 GUI）。
直接初始化 CameraWorker 並讓它跑 N 秒，驗證 ByteTrack + 功能模組的端到端流程。
"""
import sys, queue, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_yaml, log
from src.pipeline.camera_worker import CameraWorker

cfg = load_yaml("configs/cameras.yaml")
cam = next(c for c in cfg["cameras"] if c["id"] == "cam_platform_north")

q = queue.Queue(maxsize=4)
w = CameraWorker(
    cam_cfg=cam,
    global_cfg=cfg,
    source=cam["fallback"],
    mode="dev",
    records_path="configs/roi_records.json",
    frame_queue=q,
)

w.start()
log("TEST", "Worker 已啟動，等待 15 秒...")

received = 0
start = time.time()
while time.time() - start < 15:
    try:
        cam_id, frame, fps = q.get(timeout=1.0)
        received += 1
        if received % 10 == 0:
            log("TEST", f"收到第 {received} 幀  fps={fps:.1f}")
    except queue.Empty:
        if not w.is_alive():
            log("TEST", "Worker 已結束（影片播完）")
            break

w.stop()
w.join(timeout=5.0)
log("TEST", f"完成，共收到 {received} 幀")
