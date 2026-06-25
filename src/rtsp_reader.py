import cv2
from pathlib import Path
from src.utils import log


def _is_file_source(src) -> bool:
    if isinstance(src, int):
        return False
    s = str(src)
    return not s.startswith("rtsp://") and not s.startswith("rtmp://") and not s.isdigit()


def _is_rtsp_source(src) -> bool:
    if isinstance(src, int):
        return False
    s = str(src)
    return s.startswith("rtsp://") or s.startswith("rtmp://")


def _resolve_source(src):
    if isinstance(src, int):
        return src
    return int(src) if str(src).isdigit() else src


def _guess_codec(rtsp_url: str) -> str:
    """從 URL 關鍵字推測串流 codec（h264 / h265）。"""
    low = rtsp_url.lower()
    if any(k in low for k in ("h264", "264", "avc")):
        return "h264"
    return "h265"   # 預設 H.265（IP cam 較常見）


def _build_gst_rtsp_pipeline(rtsp_url: str, hw_accel: bool = True) -> str:
    codec  = _guess_codec(rtsp_url)
    depay  = "rtph264depay" if codec == "h264" else "rtph265depay"
    parse  = "h264parse"    if codec == "h264" else "h265parse"

    if hw_accel:
        # Jetson：nvv4l2decoder 硬體解碼（H.264 / H.265 均支援）
        return (
            f"rtspsrc location={rtsp_url} latency=0 ! "
            f"{depay} ! {parse} ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
    # Ubuntu x86 / 非 Jetson：avdec 軟體解碼
    avdec = "avdec_h264" if codec == "h264" else "avdec_h265"
    return (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        f"{depay} ! {parse} ! {avdec} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class RTSPReader:
    """
    影像來源讀取器。
    RTSP 來源依序嘗試：
      1. GStreamer HW（Jetson nvv4l2decoder，H.264/H.265 自動偵測）
      2. GStreamer SW（avdec_h264 / avdec_h265，需 gstreamer1.0-libav）
      3. FFmpeg（Mac / Ubuntu 無 GStreamer 環境）
    codec 由 URL 關鍵字自動推測（含 h264/264/avc → H.264，否則 H.265）。
    本地檔案 / webcam 直接使用 cv2.VideoCapture 預設 backend。
    """

    def __init__(self, source, fallback=None):
        self.source = source
        self.fallback = fallback
        self.cap = None
        self.active_source = None
        self._prefetched = None

    def open(self) -> bool:
        if self._try_open(self.source, label="主要來源"):
            return True
        if self.fallback is not None:
            log("WARN", f"主要來源失敗，切換備援: {self.fallback}")
            if self._try_open(self.fallback, label="備援"):
                return True
        log("ERROR", "無法開啟任何影像來源")
        return False

    def _try_open(self, src, label: str) -> bool:
        src = _resolve_source(src)

        if _is_file_source(src):
            if not Path(str(src)).exists():
                log("WARN", f"{label} 檔案不存在，跳過: {src}")
                return False

        if self.cap:
            self.cap.release()

        if _is_rtsp_source(src):
            for hw_accel, decode_label in ((True, "HW nvv4l2"), (False, "SW avdec_h265")):
                gst_pipe = _build_gst_rtsp_pipeline(str(src), hw_accel=hw_accel)
                cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    self.cap = cap
                    self.active_source = src
                    log("INFO", f"已開啟 {label}（GStreamer {decode_label}）: {src}")
                    return True
                cap.release()
            log("WARN", f"{label}: GStreamer 不可用，改用 FFmpeg: {src}")

        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.active_source = src
            if _is_rtsp_source(src):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log("INFO", f"已開啟 {label}: {src}")
            return True

        self.cap.release()
        self.cap = None
        log("WARN", f"{label} 無法開啟: {src}")
        return False

    def read(self):
        if self.cap is None:
            return False, None
        if self._prefetched is not None:
            frame, self._prefetched = self._prefetched, None
            return True, frame
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def is_file(self) -> bool:
        return _is_file_source(self.active_source) if self.active_source is not None else False

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def get_size(self) -> tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            return w, h
        # GStreamer live source：需預讀一幀才取得真實尺寸
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._prefetched = frame
            return frame.shape[1], frame.shape[0]
        return 640, 640

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
