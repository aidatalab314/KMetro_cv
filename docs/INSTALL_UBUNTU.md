# KMetro CV — Ubuntu 24.04 + GeForce RTX 安裝指南

> 適用環境：Ubuntu 24.04 LTS，NVIDIA GeForce RTX 5060（Blackwell GB206）
> 不適用 Jetson 系列（Jetson 使用 nvv4l2decoder / JetPack，路徑不同）

---

## 目錄

1. [系統需求確認](#1-系統需求確認)
2. [NVIDIA 驅動程式](#2-nvidia-驅動程式)
3. [CUDA Toolkit 12.8](#3-cuda-toolkit-128)
4. [cuDNN 9](#4-cudnn-9)
5. [TensorRT 10](#5-tensorrt-10)
6. [GStreamer（RTSP 解碼）](#6-gstreamer)
7. [Conda 環境與 Python 套件](#7-conda-環境與-python-套件)
8. [OpenCV 驗證（GStreamer + CUDA）](#8-opencv-驗證)
9. [TensorRT 模型匯出](#9-tensorrt-模型匯出)
10. [cameras.local.yaml 設定](#10-cameraslocal-yaml)
11. [完整驗證清單](#11-完整驗證清單)
12. [啟動](#12-啟動)

---

## 1. 系統需求確認

```bash
# Ubuntu 版本
lsb_release -a

# GPU 確認（應看到 RTX 5060）
lspci | grep -i nvidia

# 確認已插顯示卡且 BIOS 未停用
```

預期 GPU compute capability：**12.0**（Blackwell GB206）

---

## 2. NVIDIA 驅動程式

RTX 5060（Blackwell）需要 **Driver 570+** 且必須使用 **open kernel module**。

```bash
# 更新套件清單
sudo apt update

# 自動安裝建議版本（open kernel module 版）
sudo ubuntu-drivers install --gpgpu nvidia:570-open

# 或手動指定
sudo apt install -y nvidia-driver-570-open

# 重開機
sudo reboot
```

重開機後確認：

```bash
nvidia-smi
# 應看到 Driver Version: 570.x  CUDA Version: 12.8
```

---

## 3. CUDA Toolkit 12.8

RTX 5060（Blackwell, sm_120）需要 **CUDA 12.8 以上**。

```bash
# 加入 NVIDIA CUDA 套件庫
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安裝 CUDA Toolkit 12.8
sudo apt install -y cuda-toolkit-12-8

# 設定環境變數（加入 ~/.bashrc 或 ~/.zshrc）
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認
nvcc --version
# nvcc: release 12.8
```

> **注意**：`cuda-12-8` 套件僅安裝 toolkit，不含驅動。若要 toolkit + 驅動一次安裝：
> `sudo apt install -y cuda-12-8`（包含驅動）但需確認版本不衝突。

---

## 4. cuDNN 9

```bash
# 使用第 3 步加入的 CUDA repo（已含 cuDNN）
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# 確認
dpkg -l | grep cudnn
```

---

## 5. TensorRT 10

> **重要**：Ubuntu x86 使用 pip 安裝 TensorRT，**不是** Jetson 的 `tensorrt` deb 套件。

```bash
# 在 conda env 外先確認系統 pip 路徑，或在 kmetro env 內安裝（建議後者，見第 7 節）

# TensorRT 10.x（需 CUDA 12.8）
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs

# 確認
python -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
```

---

## 6. GStreamer

本專案 RTSP 使用 GStreamer，H.264/H.265 由 URL 自動判斷（`rtsp_reader.py`）。
Ubuntu x86 使用軟體解碼（`avdec_h264` / `avdec_h265`），GPU 資源留給推論。

```bash
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev
```

驗證：

```bash
# H.264 測試（需替換為實際 RTSP 位址）
gst-launch-1.0 rtspsrc location="rtsp://user:pass@ip/stream" latency=0 ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

# H.265 測試
gst-launch-1.0 rtspsrc location="rtsp://user:pass@ip/stream" latency=0 ! \
  rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! autovideosink
```

---

## 7. Conda 環境與 Python 套件

### 7.1 安裝 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash   # 或 zsh
source ~/.bashrc
```

### 7.2 建立環境

```bash
conda create -n kmetro python=3.10 -y
conda activate kmetro
```

### 7.3 PyTorch（CUDA 12.8）

```bash
# PyTorch cu128 build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 確認 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
# 預期: CUDA: True  12.8
```

### 7.4 TensorRT（環境內安裝）

```bash
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs
```

### 7.5 Ultralytics 與其他依賴

```bash
pip install ultralytics[export]   # 含 onnx, onnxruntime 等 export 工具

# 其他專案依賴
pip install \
  opencv-python \        # 含 GStreamer 支援
  cvzone \
  rtmlib \
  pyyaml \
  numpy
```

### 7.6 驗證整合

```bash
python - <<'EOF'
import torch, cv2, ultralytics
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0))
print("OpenCV:", cv2.__version__, "| GStreamer:", cv2.getBuildInformation().split("GStreamer:")[1].split("\n")[0].strip())
print("Ultralytics:", ultralytics.__version__)
EOF
```

---

## 8. OpenCV 驗證

`pip install opencv-python` 在 Linux 上通常已含 GStreamer。驗證：

```bash
python -c "
import cv2
info = cv2.getBuildInformation()
print('GStreamer:', 'YES' if 'GStreamer: YES' in info else 'NO')
print('CUDA:', 'YES' if 'CUDA: YES' in info else 'NO (推論用 PyTorch CUDA，此項非必要)')
"
```

> **說明**：OpenCV GStreamer YES → RTSP 解碼正常。OpenCV CUDA 非必要，推論已由 PyTorch / TensorRT 負責。
> 若 GStreamer 為 NO，需從原始碼編譯 OpenCV（見附錄 A）。

---

## 9. TensorRT 模型匯出

在 Ubuntu 目標機上執行（TensorRT engine 綁定 GPU 架構，不可跨機器使用）：

```bash
conda activate kmetro
cd ~/KMetro_cv

python - <<'EOF'
from ultralytics import YOLO

# Person 偵測模型（FP16 加速）
YOLO("models/fall_detection/yolo12l.pt").export(
    format="engine", device=0, half=True, imgsz=1280
)

# 行李偵測模型
YOLO("models/luggage/yolo_luggage_best.pt").export(
    format="engine", device=0, half=True, imgsz=640
)
EOF

# 確認 engine 檔案生成
ls -lh models/fall_detection/*.engine models/luggage/*.engine
```

匯出時間：yolo12l 約 3–10 分鐘（首次需 calibration）。

> **注意**：`.engine` 檔案綁定 GPU 型號與 TensorRT 版本，不可從 Mac / Jetson 複製過來。

---

## 10. cameras.local.yaml 設定

```bash
cp configs/cameras.local.yaml.example configs/cameras.local.yaml
```

編輯 `configs/cameras.local.yaml`，填入以下 Ubuntu CUDA 設定：

```yaml
# Ubuntu x86 + GeForce RTX — 覆蓋設定

detector:
  device:       "0"      # CUDA GPU 0（非 "mps"，非 "cpu"）
  skip_frames:  1        # TensorRT 夠快，不需跳幀；若 fps 仍不足可改 2
  imgsz:        640      # 一般場景；fall_detector 可在 features 區覆蓋 1280

models:
  person:     "models/fall_detection/yolo12l.engine"      # TensorRT FP16
  luggage:    "models/luggage/yolo_luggage_best.engine"   # TensorRT FP16

output:
  save_video_rtsp: true   # 部署環境建議錄影存證
  mosaic_fps:      25     # 依 RTSP 來源 fps 調整（25fps 來源 ÷ 1 = 25）
                          # ⚠️ 必須 = source_fps ÷ skip_frames
```

---

## 11. 完整驗證清單

```bash
conda activate kmetro
cd ~/KMetro_cv

# 1. CUDA + GPU
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"

# 2. TensorRT engine 載入
python -c "from ultralytics import YOLO; m=YOLO('models/fall_detection/yolo12l.engine'); print('Engine OK')"

# 3. RTSP 連線（替換為實際 IP）
python -c "
import cv2
cap = cv2.VideoCapture('rtsp://admin:123456@192.168.6.91/stream0')
ok, f = cap.read(); cap.release()
print('RTSP OK' if ok else 'RTSP 連線失敗')
"

# 4. 本地影片 fallback
python -c "
import cv2
cap = cv2.VideoCapture('data/test_videos/metro/IMG_2733.MOV')
ok, _ = cap.read(); cap.release()
print('Fallback video OK' if ok else '影片讀取失敗')
"

# 5. 全系統啟動測試（auto 模式，headless）
python src/pipeline/multistream.py --mode op --cameras cam_platform_north &
sleep 10 && kill %1
```

---

## 12. 啟動

```bash
conda activate kmetro
cd ~/KMetro_cv

# auto 模式（探測 RTSP，不可達則 fallback 本地影片）
python src/pipeline/multistream.py

# 強制 RTSP + headless op 模式
python src/pipeline/multistream.py --source rtsp --mode op

# 指定攝影機
python src/pipeline/multistream.py --cameras cam_platform_north,cam_escalator_up

# 重設 ROI（RTSP 模式下從真實畫面重設）
python src/pipeline/multistream.py --source rtsp --reset-roi all
```

> **來源失敗行為**：若 RTSP 與 fallback 影片均無法開啟，該路顯示黑畫面並標示
> `NO SOURCE / RTSP and fallback unavailable`，不影響其他路正常運作，按 `q` 可退出。

---

## 附錄 A：從原始碼編譯 OpenCV（GStreamer 為 NO 時）

```bash
sudo apt install -y cmake build-essential libgtk-3-dev pkg-config

git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GSTREAMER=ON \
  -DWITH_CUDA=OFF \           # OpenCV CUDA 非必要
  -DBUILD_opencv_python3=ON \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DINSTALL_PYTHON_EXAMPLES=OFF

make -j$(nproc)
sudo make install
```

---

## 附錄 B：常見問題

| 問題 | 原因 | 解法 |
|------|------|------|
| `nvidia-smi` 顯示正常但 `torch.cuda.is_available()` 為 False | PyTorch 版本與 CUDA 不符 | 確認安裝 `cu128` build |
| TensorRT export 失敗：`sm_120 not supported` | TensorRT 版本太舊 | 升級至 TensorRT 10.x |
| GStreamer RTSP 失敗，但 FFmpeg 可以 | `avdec_h264/h265` 未安裝 | `sudo apt install gstreamer1.0-libav` |
| ROI 視窗開不起來（headless）| 無 X display | 加 `--mode op`（headless）或設定 `DISPLAY` |
| `nvv4l2decoder` 錯誤訊息 | Jetson 專屬，Ubuntu 正常現象 | 可無視，程式自動 fallback 到 avdec |
