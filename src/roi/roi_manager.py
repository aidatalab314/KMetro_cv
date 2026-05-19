import json
import numpy as np
import cv2
from pathlib import Path


# ── 互動式繪製工具 ────────────────────────────────────────────────────────────

def draw_roi_interactive(video_path: str) -> list[dict]:
    """
    從影片第一幀開啟互動式 ROI 繪製。
    操作說明：
      左鍵       加點
      右鍵       刪除最後一點
      C          確認當前多邊形（至少 3 點）
      R          重置當前未完成多邊形
      ESC / Q    儲存並結束
    回傳已完成的 ROI 列表。
    """
    cap = cv2.VideoCapture(video_path)
    ret, original = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"無法讀取影片第一幀: {video_path}")

    H, W = original.shape[:2]
    scale = min(1.0, 1280 / W)
    win   = f"ROI Setup  [{Path(video_path).name}]"

    COLORS = [
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (0, 165, 255),
        (255, 100, 0),
    ]

    state = {"pts": [], "rois": []}

    def _color(idx: int):
        return COLORS[idx % len(COLORS)]

    def _redraw():
        frame = original.copy()

        # 已完成的 ROI：半透明填色 + 邊框 + 標籤
        for roi in state["rois"]:
            poly  = np.array(roi["points"], np.int32)
            color = tuple(roi["color"])
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            # 使用獨立 result 變數避免 src/dst 衝突
            result = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            np.copyto(frame, result)
            cv2.polylines(frame, [poly], True, color, 3)
            cv2.putText(frame, roi["label"], tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 繪製中的多邊形
        cur_color = _color(len(state["rois"]))
        pts = state["pts"]
        for pt in pts:
            cv2.circle(frame, pt, 7, cur_color, -1)
        if len(pts) >= 2:
            poly = np.array(pts, np.int32)
            if len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], cur_color)
                result = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                np.copyto(frame, result)
                cv2.polylines(frame, [poly], True, cur_color, 2)
            else:
                cv2.polylines(frame, [poly], False, cur_color, 2)

        # 操作說明（白字黑邊）
        tips = [
            "Left click : add point",
            "Right click: undo point",
            "[C]        : confirm ROI (>=3 pts)",
            "[R]        : reset current ROI",
            "[ESC/Q]    : save & quit (auto-confirm if pending)",
            f"ROIs done  : {len(state['rois'])}",
        ]
        for i, t in enumerate(tips):
            y = 28 + i * 24
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

        # 未確認點達 3 個時顯示提示
        if len(state["pts"]) >= 3:
            hint = ">> Press [C] to confirm  or  [ESC] to auto-confirm & quit <<"
            cv2.putText(frame, hint, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, hint, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        display = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow(win, display)
        cv2.waitKey(1)   # 強制刷新畫面，避免 imshow 未即時生效

    def _mouse(event, x, y, flags, param):
        ox, oy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((ox, oy))
            _redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["pts"]:
                state["pts"].pop()
                _redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _mouse)
    _redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('c'), ord('C')):
            if len(state["pts"]) >= 3:
                idx = len(state["rois"])
                state["rois"].append({
                    "id":     f"roi_{idx}",
                    "label":  f"ROI {idx}",
                    "color":  list(_color(idx)),
                    "points": [list(p) for p in state["pts"]],
                })
                state["pts"] = []
                _redraw()
                print(f"[ROI] ROI {idx} confirmed  ({len(state['rois'][-1]['points'])} pts)")
            else:
                print("[ROI] 至少需要 3 個點才能確認 ROI")

        elif key in (ord('r'), ord('R')):
            state["pts"] = []
            _redraw()

        elif key in (27, ord('q'), ord('Q')):
            # 有未確認的點（≥3）自動完成最後一個 ROI
            if len(state["pts"]) >= 3:
                idx = len(state["rois"])
                state["rois"].append({
                    "id":     f"roi_{idx}",
                    "label":  f"ROI {idx}",
                    "color":  list(_color(idx)),
                    "points": [list(p) for p in state["pts"]],
                })
                print(f"[ROI] ROI {idx} auto-confirmed on exit  ({len(state['pts'])} pts)")
            break

    cv2.destroyWindow(win)
    return state["rois"]


# ── 紀錄檔 I/O ────────────────────────────────────────────────────────────────

def has_roi_record(video_name: str, records_path: str) -> bool:
    p = Path(records_path)
    if not p.exists():
        return False
    with open(p) as f:
        return video_name in json.load(f)


def save_roi_record(video_name: str, rois: list[dict], records_path: str):
    p = Path(records_path)
    records = {}
    if p.exists():
        with open(p) as f:
            records = json.load(f)
    records[video_name] = rois
    with open(p, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[ROI] 已儲存 {len(rois)} 個 ROI → {records_path}  (key: {video_name})")


# ── ROI 管理器 ────────────────────────────────────────────────────────────────

class ROIManager:
    def __init__(self, video_name: str, records_path: str):
        self.rois = []
        p = Path(records_path)
        if p.exists():
            with open(p) as f:
                records = json.load(f)
            for r in records.get(video_name, []):
                self.rois.append({
                    "id":      r["id"],
                    "label":   r["label"],
                    "color":   tuple(r["color"]),
                    "polygon": np.array(r["points"], np.int32),
                })

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for roi in self.rois:
            poly  = roi["polygon"]
            color = roi["color"]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            result = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)
            np.copyto(frame, result)
            cv2.polylines(frame, [poly], True, color, 3)
            cv2.putText(frame, roi["label"], tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def get_containing_rois(self, cx: int, cy: int) -> list[str]:
        return [
            roi["label"]
            for roi in self.rois
            if cv2.pointPolygonTest(roi["polygon"], (float(cx), float(cy)), False) >= 0
        ]

    def is_inside(self, cx: int, cy: int) -> bool:
        """若無 ROI 設定則視為全幀皆在範圍內（無圍籬模式）"""
        if not self.rois:
            return True
        return any(
            cv2.pointPolygonTest(roi["polygon"], (float(cx), float(cy)), False) >= 0
            for roi in self.rois
        )
