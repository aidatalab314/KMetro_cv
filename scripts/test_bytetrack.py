"""ByteTrack + DwellMonitor 整合測試"""
import sys, cv2, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.fall_detector import FallDetector
from src.features.dwell_monitor import DwellMonitor
from src.roi.roi_manager import ROIManager

fd  = FallDetector(weight_path="models/fall_detection/yolo12l.pt",
                   conf=0.3, tracking=True, clahe=False, gamma=1.0)
dm  = DwellMonitor(alert_seconds=3.0, grace_period_sec=5.0)
roi = ROIManager("IMG_2732.MOV", "configs/roi_records.json")

cap     = cv2.VideoCapture("data/test_videos/metro/IMG_2732.MOV")
frame_n = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_n >= 120:
        break
    frame_n += 1
    if frame_n % 2 != 0:
        continue

    persons = fd.detect(frame)
    in_roi  = [p for p in persons if roi.is_inside(p["cx"], p["cy"])]
    alerts  = dm.update(in_roi)
    alert_ids = {a["track_id"] for a in alerts}

    for p in in_roi:
        tid = p.get("track_id", -1)
        if tid < 0:
            continue
        dwell = dm.get_dwell(tid)
        if dwell > 1.0:
            flag = " <<ALERT>>" if tid in alert_ids else ""
            print(f"  frame {frame_n:3d}: ID={tid} dwell={dwell:.1f}s{flag}")
            break

cap.release()
print("tracked IDs:", list(dm._first_seen.keys()))
