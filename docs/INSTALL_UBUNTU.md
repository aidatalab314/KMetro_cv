# KMetro CV — Ubuntu 24.04 + GeForce RTX 安裝指南

> 適用環境：Ubuntu 24.04 LTS，NVIDIA GeForce RTX 5060（Blackwell GB206）
> 不適用 Jetson 系列（Jetson 使用 nvv4l2decoder / JetPack，路徑不同）

---

## 目錄

1. [系統需求確認](#1-系統需求確認)
2. [NVIDIA 驅動程式](#2-nvidia-驅動程式)
3. [CUDA Toolkit 12.8](#3-cuda-toolkit-128)
4. [cuDNN 9](#4-cudnn-9)
5. [GStreamer（RTSP 解碼）](#5-gstreamer)
6. [Python 環境（venv）](#6-python-環境venv)
7. [OpenCV 驗證](#7-opencv-驗證)
8. [專案取得（git clone + LFS）](#8-專案取得)
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
sudo apt update

# 自動安裝建議版本（open kernel module 版）
sudo ubuntu-drivers install --gpgpu nvidia:570-open

# 或手動指定
# sudo apt install -y nvidia-driver-570-open

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

# 安裝 CUDA Toolkit 12.8（不含驅動，驅動已在第 2 步安裝）
sudo apt install -y cuda-toolkit-12-8

# 設定環境變數
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認
nvcc --version
# nvcc: release 12.8
```

---

## 4. cuDNN 9

```bash
# 使用第 3 步已加入的 CUDA repo
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# 確認
dpkg -l | grep cudnn
```

---

## 5. GStreamer

本專案 RTSP 使用 GStreamer，H.264/H.265 由 URL 自動判斷（`rtsp_reader.py`）。
Ubuntu x86 使用軟體解碼（`avdec_h264` / `avdec_h265`），GPU 資源保留給推論。

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

驗證（需替換為實際 RTSP 位址）：

```bash
# H.264
gst-launch-1.0 rtspsrc location="rtsp://user:pass@ip/stream" latency=0 ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! fakesink

# H.265
gst-launch-1.0 rtspsrc location="rtsp://user:pass@ip/stream" latency=0 ! \
  rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! fakesink
```

---

## 6. Python 環境（venv）

Ubuntu 24.04 內建 Python 3.12，與本專案相容（本專案最低需求 Python 3.10+）。

### 6.1 系統套件

```bash
sudo apt install -y python3-venv python3-dev python3-pip
```

### 6.2 建立 venv

```bash
cd ~/KMetro_cv

# 建立 .venv（已在 .gitignore 排除）
python3 -m venv .venv

# 啟動（後續所有 pip / python 指令都在啟動後執行）
source .venv/bin/activate

# 確認 Python 版本
python --version
# Python 3.12.x
```

> **每次開新 terminal 都需要執行** `source ~/KMetro_cv/.venv/bin/activate`
> 建議加入 alias：`echo "alias kmetro='source ~/KMetro_cv/.venv/bin/activate'" >> ~/.bashrc`

### 6.3 PyTorch（CUDA 12.8）

```bash
# cu128 專用 build（務必使用此 index-url，不可用預設 PyPI）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 確認 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| CUDA ver:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))"
# 預期: CUDA: True | CUDA ver: 12.8 | GPU: NVIDIA GeForce RTX 5060
```

### 6.4 TensorRT 10

```bash
# TensorRT Python bindings（需 CUDA 12.8）
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs

# 確認
python -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
```

### 6.5 專案依賴

```bash
# 安裝 requirements.txt（含 ultralytics, opencv, rtmlib 等）
pip install -r requirements.txt

# ultralytics export 工具（TensorRT engine 匯出需要）
pip install "ultralytics[export]"
```

### 6.6 驗證整合

```bash
python -c "
import torch, cv2, ultralytics, tensorrt
print('PyTorch   :', torch.__version__)
print('CUDA      :', torch.cuda.is_available(), '|', torch.version.cuda)
print('GPU       :', torch.cuda.get_device_name(0))
gst = cv2.getBuildInformation().split('GStreamer:')[1].split('\n')[0].strip()
print('OpenCV    :', cv2.__version__, '| GStreamer:', gst)
print('Ultralytics:', ultralytics.__version__)
print('TensorRT  :', tensorrt.__version__)
"
```

---

## 7. OpenCV 驗證

`pip install opencv-python` 在 Linux 上通常已含 GStreamer。若結果為 NO：

```bash
python -c "
import cv2
info = cv2.getBuildInformation()
print('GStreamer:', 'YES' if 'GStreamer: YES' in info else 'NO')
"
```

GStreamer 為 NO 時，需從原始碼編譯（見附錄 A）。

---

## 8. 專案取得

本專案使用 **Git LFS** 儲存模型權重（`.pt`），clone 前需先安裝 git-lfs。

```bash
# 安裝 git-lfs
sudo apt install git-lfs
git lfs install

# Clone（LFS 物件會一併下載）
git clone https://github.com/aidatalab314/KMetro_cv.git ~/KMetro_cv
cd ~/KMetro_cv

# 確認模型已下載（不是 LFS pointer 文字檔）
ls -lh models/fall_detection/yolo12l.pt       # 應約 51MB
ls -lh models/luggage/yolo_luggage_best.pt    # 應約 5.2MB

# 建立 venv（接續第 6 節）
python3 -m venv .venv
source .venv/bin/activate
```

> **若已 clone 但 .pt 是 pointer 文字（幾百 bytes）**：
> ```bash
> git lfs pull
> ```

---

## 9. TensorRT 模型匯出

`.engine` 檔綁定 GPU 型號與 TensorRT 版本，**必須在 Ubuntu 目標機上執行**，不可從 Mac / Jetson 複製。

```bash
cd ~/KMetro_cv
source .venv/bin/activate

python - <<'EOF'
from ultralytics import YOLO

# Person 偵測模型（FP16 加速，1280 解析度）
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

匯出時間：yolo12l 約 3–10 分鐘（首次 TensorRT calibration）。

---

## 10. cameras.local.yaml 設定

```bash
cp configs/cameras.local.yaml.example configs/cameras.local.yaml
```

編輯 `configs/cameras.local.yaml`，填入 Ubuntu CUDA 設定：

```yaml
# Ubuntu x86 + GeForce RTX — 覆蓋設定

detector:
  device:       "0"      # CUDA GPU 0
  skip_frames:  1        # TensorRT 夠快，不需跳幀；fps 不足可改 2

models:
  person:     "models/fall_detection/yolo12l.engine"      # TensorRT FP16
  luggage:    "models/luggage/yolo_luggage_best.engine"   # TensorRT FP16

output:
  save_video_rtsp: true
  mosaic_fps:      25    # ⚠️ = source_fps ÷ skip_frames（25fps ÷ 1 = 25）
```

> 若 TensorRT engine 尚未匯出（第 9 步），先用 `.pt` 測試：直接不設 models 覆蓋，
> 或設 `device: "cuda"` 讓 PyTorch 用 CUDA 推論。

---

## 11. 完整驗證清單

```bash
cd ~/KMetro_cv
source .venv/bin/activate

# 1. CUDA + GPU
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"

# 2. TensorRT engine 載入（engine 匯出後才可執行）
python -c "from ultralytics import YOLO; YOLO('models/fall_detection/yolo12l.engine'); print('Engine OK')"

# 3. RTSP 連線（替換為實際 IP）
python -c "
import cv2
cap = cv2.VideoCapture('rtsp://admin:123456@192.168.6.91/stream0')
ok, _ = cap.read(); cap.release()
print('RTSP OK' if ok else 'RTSP 連線失敗')
"

# 4. GStreamer RTSP（透過 rtsp_reader）
python -c "
from src.rtsp_reader import RTSPReader
r = RTSPReader('rtsp://admin:123456@192.168.6.91/stream0')
print('open:', r.open(), '| file:', r.is_file())
r.release()
"

# 5. 全系統啟動測試（headless，10 秒後自動停止）
timeout 10 python src/pipeline/multistream.py --mode op --cameras cam_platform_north || true
```

---

## 12. 啟動

```bash
cd ~/KMetro_cv
source .venv/bin/activate

# auto 模式（探測 RTSP，不可達則 fallback 本地影片）
python src/pipeline/multistream.py

# 強制 RTSP + headless op 模式
python src/pipeline/multistream.py --source rtsp --mode op

# 指定攝影機
python src/pipeline/multistream.py --cameras cam_platform_north,cam_escalator_up

# 重設 ROI（RTSP 模式下從真實畫面重設）
python src/pipeline/multistream.py --source rtsp --reset-roi all
```

> **來源失敗行為**：RTSP 與 fallback 均無法開啟時，該路顯示黑畫面
> `NO SOURCE / RTSP and fallback unavailable`，按 `q` 退出。

---

## 附錄 A：從原始碼編譯 OpenCV（GStreamer 為 NO 時）

```bash
sudo apt install -y cmake build-essential libgtk-3-dev pkg-config

git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GSTREAMER=ON \
  -DWITH_CUDA=OFF \
  -DBUILD_opencv_python3=ON \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DINSTALL_PYTHON_EXAMPLES=OFF

make -j$(nproc)
sudo make install

# 重新安裝 venv 內的 opencv（指向系統編譯版）
pip uninstall opencv-python -y
# 系統 site-packages 路徑需加入 venv 或改用 --system-site-packages 建立 venv
```

---

## 附錄 B：常見問題

| 問題 | 原因 | 解法 |
|------|------|------|
| `nvidia-smi` 正常但 `torch.cuda.is_available()` 為 False | PyTorch 版本與 CUDA 不符 | 確認安裝 `cu128` build（`--index-url .../cu128`） |
| TensorRT export 失敗：`sm_120 not supported` | TensorRT 版本太舊 | 升級至 TensorRT 10.x |
| GStreamer RTSP 失敗，但 FFmpeg 可以 | `avdec_h264/h265` 未安裝 | `sudo apt install gstreamer1.0-libav` |
| `nvv4l2decoder` 錯誤訊息出現 | Jetson 專屬元件，Ubuntu 正常觸發 | 可忽略，程式自動 fallback 到 avdec |
| `.pt` 只有幾百 bytes | git-lfs 未啟用就 clone | `git lfs install && git lfs pull` |
| `pip install` 後 `import` 失敗 | venv 未啟動 | `source ~/KMetro_cv/.venv/bin/activate` |
| ROI 視窗無法顯示（headless server）| 無 X display | 加 `--mode op` 改用 headless |
