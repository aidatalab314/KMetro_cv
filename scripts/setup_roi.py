"""
互動式 ROI 設定 CLI。
用法：
    python scripts/setup_roi.py --video IMG_2734.MOV        # 有紀錄 → 直接覆蓋重畫
    python scripts/setup_roi.py --video IMG_2734.MOV --force # 同上（明確指定）
    python scripts/setup_roi.py --list                       # 列出所有已有 ROI 紀錄的影片
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roi.roi_manager import draw_roi_interactive, save_roi_record

RECORDS  = "configs/roi_records.json"
TEST_DIR = "data/test_videos/metro"


def list_records():
    p = Path(RECORDS)
    if not p.exists() or p.stat().st_size <= 2:
        print("[INFO] 目前沒有任何 ROI 紀錄。")
        return
    with open(p) as f:
        records = json.load(f)
    print(f"[INFO] 共 {len(records)} 筆 ROI 紀錄：")
    for name, rois in records.items():
        print(f"  {name}  ({len(rois)} 個 ROI)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="影片檔名（在 metro 目錄下）或完整路徑")
    parser.add_argument("--force", action="store_true",
                        help="強制重畫（即使已有紀錄）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有已有 ROI 紀錄的影片")
    args = parser.parse_args()

    if args.list:
        list_records()
        return

    if not args.video:
        parser.print_help()
        return

    p = Path(args.video)
    video_path = p if p.is_absolute() else Path(TEST_DIR) / p
    if not video_path.exists():
        print(f"[ERROR] 找不到影片: {video_path}")
        return

    video_name = video_path.name
    rois = draw_roi_interactive(str(video_path))

    if not rois:
        print("[WARN] 未確認任何 ROI，不儲存。")
        return

    save_roi_record(video_name, rois, RECORDS)


if __name__ == "__main__":
    main()
