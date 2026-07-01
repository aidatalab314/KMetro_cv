# KMetro CV — Ubuntu 24.04 + GeForce RTX 安裝指南

> 適用環境：Ubuntu 24.04 LTS，NVIDIA GeForce RTX 5060（Blackwell GB206）
> 不適用 Jetson 系列（Jetson 使用 nvv4l2decoder / JetPack，路徑不同）

**確認測試環境**

| 元件 | 測試版本 | 最低需求 |
|------|----------|----------|
| Ubuntu | 24.04 LTS | 24.04 |
| NVIDIA Driver | 580.x | 570+ (open kernel) |
| CUDA Toolkit (`nvcc`) | 12.8 | 12.8+ |
| GPU | RTX 5060 (sm_120) | Blackwell (sm_120) |

> **`nvidia-smi` 顯示的 CUDA Version（13.0）是驅動最高支援版本，不是 Toolkit 版本。**
> 實際安裝的 CUDA Toolkit 版本由 `nvcc --version` 確認。

---

## 目錄

1. [系統需求確認](#1-系統需求確認)
2. [NVIDIA 驅動程式](#2-nvidia-驅動程式)
3. [CUDA Toolkit 12.8](#3-cuda-toolkit-128)
4. [cuDNN](#4-cudnn)
5. [GStreamer（RTSP 解碼）](#5-gstreamer)
6. [Python 環境（venv）](#6-python-環境venv)
7. [OpenCV（含 GStreamer 支援）](#7-opencv含-gstreamer-支援)
8. [專案取得（git clone + LFS）](#8-專案取得)
9. [TensorRT 模型匯出](#9-tensorrt-模型匯出)
10. [cameras.local.yaml 設定（雙機開發）](#10-cameraslocal-yaml-設定雙機開發)
11. [完整驗證清單](#11-完整驗證清單)
12. [啟動](#12-啟動)

---

## 1. 系統需求確認

```bash
# Ubuntu 版本
lsb_release -a

# GPU 確認（應看到 RTX 5060）
lspci | grep -i nvidia

# 若驅動已裝，確認版本
nvidia-smi
```

預期 GPU compute capability：**12.0**（Blackwell GB206）

---

## 2. NVIDIA 驅動程式

RTX 5060（Blackwell）需要 **Driver 570+**，必須使用 **open kernel module**。
測試環境使用 Driver 580，效果更佳。

```bash
sudo apt update

# 自動安裝最新建議版本（open kernel module 版）
sudo ubuntu-drivers install --gpgpu

# 或手動指定版本
# sudo apt install -y nvidia-driver-580-open

sudo reboot
```

重開機後確認：

```bash
nvidia-smi
# 應看到 Driver Version: 580.x  CUDA Version: 13.0
```

> `nvidia-smi` 顯示的 CUDA Version 代表該驅動**最高支援**的 CUDA 版本，
> 向下相容所有舊版 CUDA runtime（PyTorch cu128 在此環境下可正常使用）。

---

## 3. CUDA Toolkit 12.8

RTX 5060（sm_120）需要 **CUDA 12.8 以上**。測試環境為 CUDA Toolkit 12.8（Driver 580）。

> **已有 CUDA Toolkit 的機器（`nvcc --version` 回傳 12.8+）可跳至步驟 3c**

### 3a. 加入 NVIDIA 套件庫

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### 3b. 安裝 CUDA Toolkit

```bash
# CUDA 12.8（測試確認版本）
sudo apt install -y cuda-toolkit-12-8
```

### 3c. 設定環境變數

```bash
# 確認實際安裝路徑（應看到 cuda-12.8）
ls /usr/local/ | grep cuda

# 設定 PATH（依 nvcc --version 的實際版本調整）
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認（應顯示 release 12.8）
nvcc --version
```

---

## 4. cuDNN

PyTorch（cu128 wheel）和 TensorRT pip 套件均**自帶** cuDNN，對推論無需額外安裝。
以下為選擇性安裝（自行編譯 CUDA 程式碼時才需要）：

```bash
# 確認可用的 cuDNN 套件版本
apt-cache search cudnn | grep cuda

# 依 CUDA 版本選擇安裝（以下為範例，請對應實際版本）
# CUDA 13.0：
sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13
# CUDA 12.x：
# sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

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

驗證（替換為實際 RTSP 位址）：

```bash
# H.264 — protocols=tcp 避免 UDP 封包遺失；location 必須加引號（含 @ / 字元）
gst-launch-1.0 rtspsrc location="rtsp://user:pass@ip/stream" \
  latency=0 protocols=tcp ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! fakesink

# H.265
gst-launch-1.0 rtspsrc location="rtsp://user:pass@ip/stream" \
  latency=0 protocols=tcp ! \
  rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! fakesink
```

> **codec 自動判斷規則**：URL 含 `h265/265/hevc` → H.265，其餘預設 H.264。
> `rtsp_reader.py` 先嘗試 GStreamer SW（avdec，< 1s），再嘗試 HW（nvv4l2）。

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

# 建立 kmetro venv（已在 .gitignore 排除）
python3 -m venv kmetro

# 啟動（後續所有 pip / python 指令都在啟動後執行）
source kmetro/bin/activate

# 確認 Python 版本
python --version
# Python 3.12.x
```

> **每次開新 terminal 都需要執行** `source ~/KMetro_cv/kmetro/bin/activate`
> 建議加入 alias，之後只要打 `kmetro` 即可啟動：
> ```bash
> echo "alias kmetro='source ~/KMetro_cv/kmetro/bin/activate'" >> ~/.bashrc
> source ~/.bashrc
> ```

### 6.3 PyTorch

PyTorch wheel 自帶 CUDA runtime，Driver 版本只需 ≥ wheel 所需的最低版本。

```bash
# 確認目前最新可用的 CUDA wheel（優先選擇最新）
# cu128 = CUDA 12.8 build，Driver 570+ 皆可，在 580 / CUDA 13.0 環境下完全相容
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 若 cu130 或更新版本已上架，可改用（版本更新 = 效能更好）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 確認 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| runtime:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))"
# 預期: CUDA: True | runtime: 12.8 (或 13.0) | GPU: NVIDIA GeForce RTX 5060
```

### 6.4 TensorRT

TensorRT pip 套件同樣自帶 CUDA 12 runtime，與系統 CUDA 13.0 相容。

```bash
# cu12 bindings（與系統 CUDA 13.0 相容，自帶 cu12 libs）
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

### 6.6 torchreid（跨攝影機 Re-ID，可選）

跨攝影機滯留計時繼承功能需要 OSNet-ain 外觀特徵模型。

```bash
pip install torchreid gdown

# 驗證（首次執行會自動下載 ~11MB checkpoint）
python - <<'EOF'
import torchreid, numpy as np
ext = torchreid.utils.FeatureExtractor(model_name='osnet_ain_x1_0', device='cuda')
dummy = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
emb = ext([dummy])
print(f"ReID OK: shape={emb.shape}, device={emb.device}")
# 預期: ReID OK: shape=torch.Size([1, 512]), device=cuda:0
EOF
```

啟用方式（在 `cameras.local.yaml` 加入）：

```yaml
reid:
  enabled: true
```

> checkpoint 自動快取至 `~/.cache/torch/checkpoints/osnet_ain_x1_0_imagenet.pth`。
> 若需更高精度，可自行取得 MSMT17 版本並設定 `reid.model_path`。

### 6.7 驗證整合

```bash
python -c "
import torch, cv2, ultralytics, tensorrt
print('PyTorch   :', torch.__version__)
print('CUDA      :', torch.cuda.is_available(), '| runtime:', torch.version.cuda)
print('GPU       :', torch.cuda.get_device_name(0))
gst = cv2.getBuildInformation().split('GStreamer:')[1].split('\n')[0].strip()
print('OpenCV    :', cv2.__version__, '| GStreamer:', gst)
print('Ultralytics:', ultralytics.__version__)
print('TensorRT  :', tensorrt.__version__)
"
```

---

## 7. OpenCV（含 GStreamer 支援）

Ubuntu 24.04 的 pip `opencv-python` 以 NumPy **1.x** 編譯，但 venv 內裝的是 NumPy 2.x，
執行時會出現 ABI 錯誤：`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`。

**解法：從原始碼編譯 OpenCV**（詳見附錄 A），編譯後以 `.pth` 掛入 venv。

快速確認（附錄 A 完成後才跑）：

```bash
python -c "
import cv2
gst = cv2.getBuildInformation().split('GStreamer:')[1].split('\n')[0].strip()
print('OpenCV:', cv2.__version__, '| GStreamer:', gst)
"
# 預期：OpenCV: 4.11.0 | GStreamer:  YES (1.24.2)
```

---

## 8. 專案取得

本專案使用 **Git LFS** 儲存模型權重（`.pt`），clone 前需先安裝 git-lfs。

```bash
# 安裝必要套件
sudo apt install -y git git-lfs python3-venv python3-dev python3-pip
git lfs install

# Clone（LFS 物件會一併下載，應看到 Filtering content: ~56MB）
git clone https://github.com/aidatalab314/KMetro_cv.git ~/KMetro_cv

# ⚠️ 必須 cd 進專案目錄再確認檔案
cd ~/KMetro_cv
ls -lh models/fall_detection/yolo12l.pt       # 應約 51MB
ls -lh models/luggage/yolo_luggage_best.pt    # 應約 5.2MB

# 建立 venv（接續第 6 節）
python3 -m venv kmetro
source kmetro/bin/activate
```

> **若已 clone 但 .pt 是 pointer 文字（幾百 bytes）**：
> ```bash
> cd ~/KMetro_cv && git lfs pull
> ```

---

## 9. TensorRT 模型匯出

`.engine` 檔綁定 GPU 型號與 TensorRT 版本，**必須在 Ubuntu 目標機上執行**，不可從 Mac / Jetson 複製。

```bash
cd ~/KMetro_cv
source kmetro/bin/activate

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

## 10. cameras.local.yaml 設定（雙機開發）

`cameras.local.yaml` 已加入 `.gitignore`，**不進版控**。Mac 和 Ubuntu 各自維護，
`git pull` 不會互相覆蓋，可安心在兩台機器之間來回同步程式碼。

```bash
cp configs/cameras.local.yaml.example configs/cameras.local.yaml
```

Ubuntu 的 `cameras.local.yaml` 填入 CUDA + RTSP 設定：

```yaml
# Ubuntu 24.04 + RTX 5060 — 覆蓋設定

detector:
  device:       "0"      # CUDA GPU 0
  skip_frames:  1        # TensorRT 夠快；fps 不足可改 2

models:
  person:     "models/fall_detection/yolo12l.engine"      # TensorRT FP16
  luggage:    "models/luggage/yolo_luggage_best.engine"   # TensorRT FP16

features:
  fall_detector:
    imgsz: 640

output:
  save_video_rtsp:  false
  save_video_local: false
  mosaic_fps:       30    # source_fps(30) ÷ skip_frames(1) = 30

reid:
  enabled: true           # 啟用跨攝影機 Re-ID（需已完成 6.6 torchreid 安裝）
```

**各平台啟動指令**：

| 機器 | 指令 |
|------|------|
| Mac Mini（本地影片） | `python src/pipeline/multistream.py --source local` |
| Ubuntu（RTSP 攝影機） | `python src/pipeline/multistream.py --source rtsp` |
| 兩台都可（自動探測） | `python src/pipeline/multistream.py` |

> 若 TensorRT engine 尚未匯出，暫時不設 `models` 覆蓋，PyTorch 會直接用 `.pt` + CUDA 推論。

---

## 11. 完整驗證清單

```bash
cd ~/KMetro_cv
source kmetro/bin/activate

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

# 5. ReID extractor（torchreid 已安裝時）
python - <<'EOF'
import torchreid, numpy as np
ext = torchreid.utils.FeatureExtractor(model_name='osnet_ain_x1_0', device='cuda')
dummy = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
emb = ext([dummy])
print(f"ReID OK: shape={emb.shape}, device={emb.device}")
EOF

# 6. 全系統啟動測試（headless，10 秒後自動停止）
timeout 10 python src/pipeline/multistream.py --mode op --cameras cam_platform_north || true
```

---

## 12. 啟動

```bash
cd ~/KMetro_cv
source kmetro/bin/activate

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

## 附錄 A：從原始碼編譯 OpenCV 4.11.0（含 GStreamer）

Ubuntu 24.04 + NumPy 2.x venv 的必要步驟。編譯後用 `.pth` 掛入 venv，不需重建 venv。

### A.1 安裝編譯依賴

```bash
sudo apt install -y \
  cmake build-essential pkg-config \
  libgtk-3-dev libcanberra-gtk3-module \
  libjpeg-dev libpng-dev libtiff-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### A.2 Clone OpenCV 4.11.0

```bash
cd ~
git clone --depth 1 -b 4.11.0 https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
```

### A.3 CMake 設定

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_GSTREAMER=ON \
  -DWITH_CUDA=OFF \
  -DBUILD_opencv_python3=ON \
  -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") \
  -DPYTHON3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
  -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.12/site-packages \
  -DINSTALL_PYTHON_EXAMPLES=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF
```

### A.4 編譯與安裝

```bash
make -j$(nproc)     # 約 10–20 分鐘
sudo make install
sudo ldconfig
```

確認安裝位置：

```bash
python3 -c "import sys; sys.path.insert(0, '/usr/local/lib/python3.12/site-packages'); import cv2; print(cv2.__version__)"
# 應顯示 4.11.0
```

### A.5 掛入 venv（.pth 方式）

venv 內的 `cv2/` stub 會優先蓋掉系統安裝。先移除 stub，再加 `.pth` 讓 venv 找到系統版本：

```bash
cd ~/KMetro_cv
source kmetro/bin/activate

SITE=$(python -c "import site; print(site.getsitepackages()[0])")

# 移除 venv 內的 opencv stub（若存在）
pip uninstall opencv-python -y 2>/dev/null || true
rm -rf "$SITE/cv2"

# 加入指向系統 site-packages 的 .pth
echo "/usr/local/lib/python3.12/site-packages" > "$SITE/compiled-cv2.pth"
```

驗證：

```bash
python -c "
import cv2
gst = cv2.getBuildInformation().split('GStreamer:')[1].split('\n')[0].strip()
print('OpenCV:', cv2.__version__, '| GStreamer:', gst)
"
# 預期：OpenCV: 4.11.0 | GStreamer:  YES (1.24.2)
```

---

## 附錄 B：常見問題

| 問題 | 原因 | 解法 |
|------|------|------|
| `nvidia-smi` 正常但 `torch.cuda.is_available()` 為 False | PyTorch wheel 與 Driver 不符 | 確認安裝 `cu128`（或更新）build；`--index-url .../cu128` |
| `nvidia-smi` 顯示 CUDA 13.0，`nvcc` 顯示 12.8 | 正常現象 | `nvidia-smi` 顯示驅動最高支援版本；實際 Toolkit 版本以 `nvcc --version` 為準 |
| TensorRT export 失敗：`sm_120 not supported` | TensorRT 版本太舊 | 升級至 TensorRT 10.x+（`pip install --upgrade tensorrt`） |
| `A module compiled using NumPy 1.x cannot be run in NumPy 2.x` | pip opencv 以 NumPy 1.x 編譯，venv 是 NumPy 2.x | 必須從原始碼編譯 OpenCV（附錄 A） |
| GStreamer RTSP 失敗，但 `ffplay` 可以播放 | `avdec_h264/h265` 未安裝 | `sudo apt install gstreamer1.0-libav` |
| GStreamer `udpsrc Internal data stream error` | rtspsrc 預設 UDP，封包遺失 | pipeline 加 `protocols=tcp`（程式碼已修正） |
| GStreamer 測試出現 `Generic error` | 使用了假 URL（user:pass@ip） | 換成實際攝影機 IP 和帳密 |
| `nvv4l2decoder` 錯誤訊息 | 第一個嘗試的 HW pipeline 失敗 | 可忽略，程式自動 fallback 到 SW（avdec），速度更快 |
| `cv2.__file__` 回傳 None / 找到 venv stub | venv 內有 opencv stub 蓋掉系統安裝 | 執行附錄 A.5 的 `.pth` 修正步驟 |
| `ls models/...` 找不到檔案 | 在 `~` 而非專案目錄執行 | 先 `cd ~/KMetro_cv` 再操作 |
| `.pt` 只有幾百 bytes | clone 前未執行 `git lfs install` | `cd ~/KMetro_cv && git lfs pull` |
| `pip install` 後 `import` 失敗 | venv 未啟動 | `source ~/KMetro_cv/kmetro/bin/activate` 或打 `kmetro` |
| alias `kmetro` 無效 | 直接在 shell 賦值而非寫入 `.bashrc` | `echo "alias kmetro='source ~/KMetro_cv/kmetro/bin/activate'" >> ~/.bashrc && source ~/.bashrc` |
| SSH 執行 dev 模式出現 `qt.qpa.xcb: could not connect to display` | SSH session 無 DISPLAY 設定 | 見附錄 C |
| ROI 視窗出現、但 setMouseCallback 失效（NULL handler） | 視窗名稱含非 ASCII 字元 | window name 已修正為純 ASCII，確認使用最新版 |
| cuDNN apt 套件找不到 | 套件名隨 CUDA 版本變 | `apt-cache search cudnn` 確認可用版本後安裝 |

---

## 附錄 C：SSH 遠端連線顯示設定（Dev 模式）

從 Mac SSH 連入 Ubuntu，要讓 OpenCV 視窗顯示在 Ubuntu 的實體螢幕上：

```bash
# Ubuntu 上先確認目前的 display session
who
# 範例輸出：lab314  :1  ...

# 設定 DISPLAY（:1 對應 who 輸出的 display 號）
export DISPLAY=:1
export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority

# 驗證
xdpyinfo -display :1 | head -3
```

建議加入 `~/.bashrc` 以免每次手動設定：

```bash
cat >> ~/.bashrc << 'EOF'

# KMetro dev: SSH 連入時設定 Ubuntu 實體螢幕 display
if [ -z "$DISPLAY" ]; then
  export DISPLAY=:1
  export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
fi
EOF
source ~/.bashrc
```

> **Op 模式（`--mode op`）不需要 DISPLAY**，適合純 headless 推論。
> Dev 模式才需要此設定（顯示四格 mosaic 視窗）。
