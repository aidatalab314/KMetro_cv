"""
跌倒偵測獨立測試腳本。
用法：
    python scripts/test_fall.py                          # 預設 fall5.mp4
    python scripts/test_fall.py --video path/to/xxx.mp4
"""
import argparse
import os
import sys
import cv2
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.fall_detector import FallDetector

WEIGHT     = "models/fall_detection/yolo12l.pt"
DEFAULT_VIDEO = "data/test_videos/fall/fall5.mp4"
OUTPUT_DIR = "outputs/videos"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--aspect_ratio", type=float, default=1.2)
    parser.add_argument("--alert_frames", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.5, help="預覽縮放比例")
    args = parser.parse_args()

    detector = FallDetector(
        weight_path=WEIGHT,
        conf=args.conf,
        fallen_aspect_ratio=args.aspect_ratio,
        alert_frames=args.alert_frames,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟影片: {args.video}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"fall_test_{ts}.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"[INFO] 影片: {args.video}")
    print(f"[INFO] 輸出: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets  = detector.detect(frame)
        alert = detector.compute_alert(dets)
        detector.draw(frame, dets, alert)

        writer.write(frame)

        preview = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        cv2.imshow("Fall Detection Test", preview)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 完成，輸出: {out_path}")


if __name__ == "__main__":
    main()
