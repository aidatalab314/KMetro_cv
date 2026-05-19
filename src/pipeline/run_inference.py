"""
主推論入口。
優先順序：--camera > --video > configs/model_config.yaml 的 video.target > 批次全部

用法：
    python src/pipeline/run_inference.py                             # 依 config target 或批次全部
    python src/pipeline/run_inference.py --video IMG_2734.MOV        # 指定檔名
    python src/pipeline/run_inference.py --video IMG_2734.MOV --reset-roi  # 強制重畫 ROI
    python src/pipeline/run_inference.py --camera 0                  # 即時攝影機
"""
import argparse
import os
import sys
import yaml
import cv2
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detection.luggage_detector import LuggageDetector
from src.detection.fall_detector import FallDetector
from src.roi.roi_manager import (
    ROIManager, has_roi_record, draw_roi_interactive, save_roi_record
)

RECORDS_PATH = "configs/roi_records.json"


def load_config(path="configs/model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_output_path(cfg: dict, source_name: str) -> str:
    out_dir = cfg["video"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{source_name}_{ts}.mp4")


def ensure_roi(video_path: str, reset: bool = False):
    """
    確保該影片有 ROI 紀錄。
    - 若無紀錄或 reset=True → 開啟互動繪製工具並儲存。
    """
    video_name = Path(video_path).name
    if reset or not has_roi_record(video_name, RECORDS_PATH):
        action = "重新繪製" if reset else "尚無紀錄，開啟繪製工具"
        print(f"[ROI] {video_name} {action}...")
        rois = draw_roi_interactive(video_path)
        if rois:
            save_roi_record(video_name, rois, RECORDS_PATH)
        else:
            print("[ROI] 未確認任何 ROI，將以無圍籬模式執行。")
    else:
        print(f"[ROI] {video_name} 已有 ROI 紀錄，直接使用。")


def process(source, cfg: dict, reset_roi: bool = False):
    luggage = LuggageDetector(
        weight_path=cfg["luggage"]["weight"],
        conf=cfg["luggage"]["conf_threshold"],
        size_method=cfg["luggage"]["size_method"],
        large_person_area_ratio=cfg["luggage"]["large_person_area_ratio"],
        max_match_distance_px=cfg["luggage"]["max_match_distance_px"],
        large_area_ratio=cfg["luggage"]["large_luggage_area_ratio"],
    )
    fall = FallDetector(
        weight_path=cfg["fall_detection"]["weight"],
        conf=cfg["fall_detection"]["conf_threshold"],
        fallen_aspect_ratio=cfg["fall_detection"]["fallen_aspect_ratio"],
        alert_frames=cfg["fall_detection"]["alert_frames"],
    )

    # ROI：攝影機串流不需要 ROI 設定流程
    video_name = Path(str(source)).name if isinstance(source, str) else None
    if video_name:
        ensure_roi(str(source), reset=reset_roi)
    roi = ROIManager(video_name or "", RECORDS_PATH)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟來源: {source}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    source_name = Path(str(source)).stem if isinstance(source, str) else "camera"
    out_path = build_output_path(cfg, source_name)
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    scale = cfg["video"]["display_scale"]
    print(f"[INFO] 來源: {source}  →  輸出: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi.draw(frame)

        # 1. 全幀偵測 person（供跌倒判斷與行李大小件比例共用）
        all_persons = fall.detect(frame)

        # 2. 過濾：只保留 ROI 內的 person
        persons_in_roi = [d for d in all_persons
                          if roi.is_inside(d["cx"], d["cy"])]

        # 3. 跌倒警報只依 ROI 內的人計算
        alert = fall.compute_alert(persons_in_roi)
        fall.draw(frame, persons_in_roi, alert)

        # 4. 行李偵測（全幀偵測後過濾 ROI）
        all_luggage = luggage.detect(frame, persons=all_persons)
        luggage_in_roi = [d for d in all_luggage
                          if roi.is_inside(d["cx"], d["cy"])]
        for det in luggage_in_roi:
            in_rois = roi.get_containing_rois(det["cx"], det["cy"])
            luggage.draw(frame, [det], roi_labels=in_rois)

        writer.write(frame)

        preview = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow("KMetro CV", preview)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 完成，影片儲存至 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     type=str, default=None)
    parser.add_argument("--camera",    type=int, default=None)
    parser.add_argument("--config",    type=str, default="configs/model_config.yaml")
    parser.add_argument("--reset-roi", action="store_true",
                        help="強制重新繪製 ROI（覆蓋舊紀錄）")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    test_dir = Path(cfg["video"]["test_dir"])
    exts     = {".mp4", ".mov", ".avi", ".mkv"}

    if args.camera is not None:
        process(args.camera, cfg)
    elif args.video:
        p = Path(args.video)
        video_path = p if p.is_absolute() else test_dir / p
        process(str(video_path), cfg, reset_roi=args.reset_roi)
    else:
        target = cfg["video"].get("target") or None
        if target:
            process(str(test_dir / target), cfg, reset_roi=args.reset_roi)
        else:
            videos = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in exts)
            if not videos:
                print(f"[WARN] {test_dir} 下沒有影片。")
                return
            for v in videos:
                process(str(v), cfg, reset_roi=args.reset_roi)


if __name__ == "__main__":
    main()
