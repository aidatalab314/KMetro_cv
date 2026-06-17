"""
行李箱滾落偵測獨立測試。

用法：
    python scripts/test_luggage_roll.py
    python scripts/test_luggage_roll.py --video data/test_videos/metro/IMG_2732.MOV
    python scripts/test_luggage_roll.py --scale 0.4
"""
import sys, argparse, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.fall_detector import FallDetector
from src.detection.luggage_detector import LuggageDetector
from src.features.luggage_roll_monitor import LuggageRollMonitor
from src.roi.roi_manager import ROIManager
from src.utils import log

# ── 設定 ─────────────────────────────────────────────────────────────────────
PERSON_MODEL  = "models/fall_detection/yolo12l.pt"
LUGGAGE_MODEL = "models/luggage/yolo_luggage_best.pt"
DEFAULT_VIDEO = "data/test_videos/metro/IMG_2732.MOV"
ROI_KEY       = "IMG_2732.MOV"     # 用現有 ROI 紀錄

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scale", type=float, default=0.45)
    parser.add_argument("--speed_thr",    type=float, default=8.0,
                        help="行李速度門檻 px/frame（調低可看更多追蹤）")
    parser.add_argument("--indep_thr",    type=float, default=5.0,
                        help="獨立速度差門檻 px/frame")
    parser.add_argument("--alert_frames", type=int,   default=5)
    parser.add_argument("--save", action="store_true",
                        help="儲存標注影片至 outputs/videos/")
    args = parser.parse_args()

    # ── 初始化 ───────────────────────────────────────────────────────────────
    person_det  = FallDetector(PERSON_MODEL, conf=0.3,
                               tracking=True, clahe=True, gamma=0.5)
    luggage_det = LuggageDetector(LUGGAGE_MODEL, conf=0.3, tracking=True)
    roll_mon    = LuggageRollMonitor(
        speed_threshold_px=args.speed_thr,
        independence_threshold_px=args.indep_thr,
        alert_frames=args.alert_frames,
    )
    roi = ROIManager(ROI_KEY, "configs/roi_records.json")
    log("INFO", f"ROI zones: {len(roi.rois)}")
    log("INFO", f"speed_thr={args.speed_thr}  indep_thr={args.indep_thr}  alert_frames={args.alert_frames}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log("ERROR", f"無法開啟: {args.video}")
        return

    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.save:
        from datetime import datetime
        import os
        os.makedirs("outputs/videos", exist_ok=True)
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_p  = f"outputs/videos/luggage_roll_{ts}.mp4"
        writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        log("INFO", f"輸出: {out_p}")

    frame_n = 0
    total_alerts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % 2 != 0:      # skip_frames=2
            continue

        annotated = frame.copy()

        # 偵測
        persons     = person_det.detect(frame)
        luggage_all = luggage_det.detect(frame, persons=persons)

        # ROI 過濾
        luggage_in_roi = [d for d in luggage_all
                          if roi.is_inside(d["cx"], d["cy"])]

        # 滾落判斷
        alerts = roll_mon.update(luggage_in_roi, persons)
        roll_mon.draw(annotated, luggage_in_roi, alerts)

        for a in alerts:
            log("ALERT", f"frame {frame_n}: luggage ID={a['track_id']} "
                         f"speed={a['roll_speed']}px/f  ROLL DETECTED")
            total_alerts += 1

        # ROI overlay
        roi.draw(annotated)

        # 統計 overlay
        cv2.putText(annotated,
                    f"luggage_in_roi={len(luggage_in_roi)}  roll_alerts={total_alerts}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if writer:
            writer.write(annotated)

        preview = cv2.resize(annotated, None, fx=args.scale, fy=args.scale)
        cv2.imshow("Luggage Roll Test", preview)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log("INFO", f"完成 {frame_n} 幀，滾落告警累計: {total_alerts}")

if __name__ == "__main__":
    main()
