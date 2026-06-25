import json
import numpy as np
import cv2
from pathlib import Path


# ── Feature definitions ───────────────────────────────────────────────────────
# key, feature_name, short_english_label (cv2.putText is ASCII-only)

_FEATURE_MENU = [
    ("1", "dwell_monitor",   "Dwell Monitor"),
    ("2", "zone_counter",    "Zone Counter"),
    ("3", "fall_detector",   "Fall Detect"),
    ("4", "luggage_roll",    "Luggage Roll"),
    ("5", "size_classifier", "Size Classify"),
    ("6", "fire_smoke",      "Fire/Smoke"),
]

_FEATURE_COLORS = {
    "dwell_monitor":   (0, 255, 255),
    "zone_counter":    (255, 0, 255),
    "fall_detector":   (0, 165, 255),
    "luggage_roll":    (0, 0, 255),
    "size_classifier": (0, 255, 0),
    "fire_smoke":      (255, 80,  0),
}

_DEFAULT_COLOR = (200, 200, 200)


def _feature_color(features: list[str]) -> tuple:
    """Pick the color of the first known feature; fallback to default."""
    for f in features:
        if f in _FEATURE_COLORS:
            return _FEATURE_COLORS[f]
    return _DEFAULT_COLOR


# ── Multi-select feature menu (OpenCV overlay, ASCII-only text) ───────────────

def _select_features(win: str, display_frame: np.ndarray,
                     enabled_features: list[str] | None) -> list[str]:
    """
    Show a toggle-style multi-select feature menu after an ROI is confirmed.

    Controls:
      Number keys  toggle individual features ON/OFF
      [A]          select ALL enabled features
      [C]          clear all selections
      [Enter/Spc]  confirm (empty selection defaults to all enabled)
      [ESC]        cancel this ROI (returns empty list)

    Returns list of selected feature names, or [] to cancel.
    """
    options = [
        (key, feat, label)
        for key, feat, label in _FEATURE_MENU
        if not enabled_features or feat in enabled_features
    ]
    if not options:
        return []

    selected: set[str] = set()

    while True:
        frame = display_frame.copy()
        fh, fw = frame.shape[:2]

        menu_w = 520
        menu_h = 88 + len(options) * 38
        mx = (fw - menu_w) // 2
        my = (fh - menu_h) // 2

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (mx - 12, my - 12),
                      (mx + menu_w + 12, my + menu_h + 12), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (mx - 12, my - 12),
                      (mx + menu_w + 12, my + menu_h + 12), (70, 70, 70), 1)

        # Title
        cv2.putText(frame, "Select features for this ROI:",
                    (mx, my + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, "Press keys to toggle, Enter to confirm",
                    (mx, my + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1)

        # Feature rows
        for i, (key, feat, label) in enumerate(options):
            y     = my + 78 + i * 38
            is_on = feat in selected
            color = _FEATURE_COLORS.get(feat, _DEFAULT_COLOR) if is_on else (70, 70, 70)
            check = "[X]" if is_on else "[ ]"
            row   = f"[{key.upper()}]  {check}  {feat:<20}  {label}"
            cv2.putText(frame, row, (mx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1 if not is_on else 2)

        # Bottom hints
        y_bot = my + menu_h
        sel_str = ", ".join(sorted(selected)) if selected else "(none - Enter=all)"
        cv2.putText(frame, f"Selected: {sel_str}",
                    (mx, y_bot - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)
        cv2.putText(frame, "[A]=all  [C]=clear  [Enter/Spc]=confirm  [ESC]=cancel ROI",
                    (mx, y_bot - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(30) & 0xFF

        # Toggle individual
        for key, feat, _ in options:
            if k == ord(key):
                if feat in selected:
                    selected.discard(feat)
                else:
                    selected.add(feat)
                break

        if k in (ord('a'), ord('A')):
            selected = {feat for _, feat, _ in options}
        elif k in (ord('c'), ord('C')):
            selected.clear()
        elif k in (13, ord(' ')):       # Enter or Space → confirm
            return list(selected) if selected else [feat for _, feat, _ in options]
        elif k == 27:                   # ESC → cancel this ROI
            return []


# ── Interactive ROI drawing tool ──────────────────────────────────────────────

def draw_roi_interactive(source: str,
                          enabled_features: list[str] | None = None,
                          title: str | None = None) -> list[dict]:
    """
    Open interactive ROI drawing on a frame from source (file path or RTSP URL).

    Controls:
      Left-click     add vertex
      Right-click    remove last vertex
      [C]            confirm polygon (>=3 pts) -> feature selection menu
      [R]            reset current unfinished polygon
      [ESC / Q]      save & quit  (auto-confirms pending polygon, features=all)

    Each ROI stores a list of features in the "features" key.
    """
    is_rtsp = source.startswith("rtsp://") or source.startswith("rtmp://")
    cap = cv2.VideoCapture(source)
    original = None
    # RTSP：跳過開頭幾幀（緩衝區不穩），取第一幀穩定畫面
    n_warmup = 8 if is_rtsp else 0
    for _ in range(n_warmup + 1):
        ret, frame = cap.read()
        if ret:
            original = frame
    cap.release()
    if original is None:
        raise RuntimeError(f"Cannot read frame: {source}")

    H, W  = original.shape[:2]
    scale = min(1.0, 1280 / W)
    default_title = source if is_rtsp else Path(source).name
    win   = f"ROI Setup  [{title or default_title}]"

    state: dict = {"pts": [], "rois": []}

    def _roi_color(roi: dict) -> tuple:
        return tuple(roi["color"])

    def _redraw():
        frame = original.copy()

        # Confirmed ROIs
        full_idx = 0
        for roi in state["rois"]:
            poly     = np.array(roi["points"], np.int32)
            color    = _roi_color(roi)
            feats    = roi.get("features", ["all"])
            is_full  = roi.get("full_frame", False)
            feat_str = "+".join(f.split("_")[0][:4] for f in feats)
            if is_full:
                cv2.rectangle(frame, (4 + full_idx * 3, 4 + full_idx * 3),
                              (W - 4 - full_idx * 3, H - 4 - full_idx * 3), color, 3)
                caption = f"[Full Frame] [{feat_str}]"
                cv2.putText(frame, caption, (14, 40 + full_idx * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2)
                full_idx += 1
            else:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], color)
                result  = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)
                np.copyto(frame, result)
                cv2.polylines(frame, [poly], True, color, 3)
                caption  = f"{roi['label']} [{feat_str}]"
                cv2.putText(frame, caption, tuple(poly[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2)

        # Current in-progress polygon
        pts = state["pts"]
        cur_color = (180, 180, 180)
        for pt in pts:
            cv2.circle(frame, pt, 7, cur_color, -1)
        if len(pts) >= 2:
            poly = np.array(pts, np.int32)
            if len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], cur_color)
                result  = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
                np.copyto(frame, result)
                cv2.polylines(frame, [poly], True, cur_color, 2)
            else:
                cv2.polylines(frame, [poly], False, cur_color, 2)

        # Instructions (ASCII only)
        tips = [
            "Left-click : add point",
            "Right-click: undo point",
            "[C]        : confirm polygon -> pick feature(s)",
            "[F]        : full frame (no polygon needed)",
            "[R]        : reset current polygon",
            "[ESC/Q]    : save & quit",
            f"ROIs done  : {len(state['rois'])}",
        ]
        for i, t in enumerate(tips):
            y = 28 + i * 24
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
            cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

        if len(pts) >= 3:
            hint = ">> [C] confirm & select feature(s) <<"
            y = frame.shape[0] - 15
            cv2.putText(frame, hint, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
            cv2.putText(frame, hint, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1)

        display = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow(win, display)
        cv2.waitKey(1)

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
                pts_snap = list(state["pts"])
                # Build display for feature menu
                disp = cv2.resize(original.copy(), None, fx=scale, fy=scale)
                poly_disp = (np.array(pts_snap, np.float32) * scale).astype(np.int32)
                cv2.polylines(disp, [poly_disp], True, (220, 220, 220), 2)

                feats = _select_features(win, disp, enabled_features)
                if not feats:           # ESC → cancel this polygon
                    print("[ROI] Cancelled.")
                    _redraw()
                    continue

                color = list(_feature_color(feats))
                idx   = len(state["rois"])
                state["rois"].append({
                    "id":       f"roi_{idx}",
                    "label":    f"ROI {idx}",
                    "features": feats,
                    "color":    color,
                    "points":   [list(p) for p in pts_snap],
                })
                state["pts"] = []
                _redraw()
                print(f"[ROI] ROI {idx} confirmed  features={feats}  ({len(pts_snap)} pts)")
            else:
                print("[ROI] Need at least 3 points.")

        elif key in (ord('f'), ord('F')):
            disp  = cv2.resize(original.copy(), None, fx=scale, fy=scale)
            feats = _select_features(win, disp, enabled_features)
            if feats:
                color = list(_feature_color(feats))
                idx   = len(state["rois"])
                state["rois"].append({
                    "id":         f"roi_{idx}",
                    "label":      "Full Frame",
                    "features":   feats,
                    "color":      color,
                    "full_frame": True,
                    "points":     [[0, 0], [W, 0], [W, H], [0, H]],
                })
                _redraw()
                print(f"[ROI] Full-frame ROI confirmed  features={feats}")

        elif key in (ord('r'), ord('R')):
            state["pts"] = []
            _redraw()

        elif key in (27, ord('q'), ord('Q')):
            if len(state["pts"]) >= 3:
                idx = len(state["rois"])
                all_feats = [f for _, f, _ in _FEATURE_MENU
                             if not enabled_features or f in enabled_features]
                state["rois"].append({
                    "id":       f"roi_{idx}",
                    "label":    f"ROI {idx}",
                    "features": all_feats or ["all"],
                    "color":    list(_DEFAULT_COLOR),
                    "points":   [list(p) for p in state["pts"]],
                })
                print(f"[ROI] ROI {idx} auto-confirmed on exit  (features=all, {len(state['pts'])} pts)")
            break

    cv2.destroyWindow(win)
    return state["rois"]


# ── Records I/O ───────────────────────────────────────────────────────────────

def has_roi_record(camera_id: str, records_path: str) -> bool:
    p = Path(records_path)
    if not p.exists():
        return False
    with open(p) as f:
        return camera_id in json.load(f)


def save_roi_record(camera_id: str, rois: list[dict], records_path: str):
    p = Path(records_path)
    records = {}
    if p.exists():
        with open(p) as f:
            records = json.load(f)
    records[camera_id] = rois
    with open(p, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[ROI] Saved {len(rois)} ROI(s) -> {records_path}  (key: {camera_id})")


# ── ROI Manager ───────────────────────────────────────────────────────────────

class ROIManager:
    """
    Loads and manages ROIs for a given camera.

    Each ROI has a "features" list (new format) or legacy "feature" string.
    - is_inside_for_feature(cx, cy, feat):
        True if point is inside any ROI tagged for this feat.
        No ROI for feat -> False (skip this feature, no full-frame fallback).
    - is_inside(cx, cy):
        Legacy: True if inside any ROI, or True when no ROIs at all (full-frame).
    """

    def __init__(self, camera_id: str, records_path: str):
        self.rois: list[dict] = []
        self._by_feature: dict[str, list[dict]] = {}

        p = Path(records_path)
        if not p.exists():
            return
        with open(p) as f:
            records = json.load(f)

        for r in records.get(camera_id, []):
            # Support both new "features" (list) and legacy "feature" (str)
            if "features" in r:
                features = r["features"]
            elif "feature" in r:
                features = [r["feature"]]
            else:
                features = ["all"]

            entry = {
                "id":         r["id"],
                "label":      r["label"],
                "features":   features,
                "color":      tuple(r["color"]),
                "full_frame": r.get("full_frame", False),
                "polygon":    np.array(r["points"], np.int32),
            }
            self.rois.append(entry)
            for feat in features:
                self._by_feature.setdefault(feat, []).append(entry)

    # ── Feature-specific queries ──────────────────────────────────────────────

    def has_roi_for_feature(self, feature: str) -> bool:
        return bool(self._by_feature.get(feature))

    def is_inside_for_feature(self, cx: int, cy: int, feature: str) -> bool:
        candidates = self._by_feature.get(feature, [])
        if not candidates:
            return False
        pt = (float(cx), float(cy))
        return any(cv2.pointPolygonTest(r["polygon"], pt, False) >= 0
                   for r in candidates)

    def get_containing_rois_for_feature(self, cx: int, cy: int,
                                         feature: str) -> list[str]:
        pt = (float(cx), float(cy))
        return [r["label"] for r in self._by_feature.get(feature, [])
                if cv2.pointPolygonTest(r["polygon"], pt, False) >= 0]

    # ── Legacy ────────────────────────────────────────────────────────────────

    def is_inside(self, cx: int, cy: int) -> bool:
        if not self.rois:
            return True
        pt = (float(cx), float(cy))
        return any(cv2.pointPolygonTest(r["polygon"], pt, False) >= 0
                   for r in self.rois)

    def get_containing_rois(self, cx: int, cy: int) -> list[str]:
        pt = (float(cx), float(cy))
        return [r["label"] for r in self.rois
                if cv2.pointPolygonTest(r["polygon"], pt, False) >= 0]

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w     = frame.shape[:2]
        full_idx = 0
        for roi in self.rois:
            poly     = roi["polygon"]
            color    = roi["color"]
            feats    = roi.get("features", ["all"])
            is_full  = roi.get("full_frame", False)
            feat_str = "+".join(f.split("_")[0][:4] for f in feats)
            if is_full:
                cv2.rectangle(frame, (3 + full_idx * 3, 3 + full_idx * 3),
                              (w - 3 - full_idx * 3, h - 3 - full_idx * 3), color, 2)
                cv2.putText(frame, f"[Full] [{feat_str}]", (10, 28 + full_idx * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
                full_idx += 1
            else:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], color)
                result  = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
                np.copyto(frame, result)
                cv2.polylines(frame, [poly], True, color, 2)
                cv2.putText(frame, f"{roi['label']} [{feat_str}]", tuple(poly[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        return frame
