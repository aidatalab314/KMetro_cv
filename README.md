# KMetro CV — 高捷站智慧監控系統

高雄捷運站 AI 影像監控 PoC。1 台 Jetson Orin 對接 4 台 RTSP 攝影機，各攝影機依架設位置啟用不同功能模組。

---

## 安裝

```bash
conda create -n kmetro python=3.10 -y
conda activate kmetro
pip install -r requirements.txt
```

---

## 快速啟動

```bash
conda activate kmetro
cd KMetro_cv

# auto 模式（預設）：探測 RTSP 是否可達，可達用攝影機，否則用本地影片
python src/pipeline/multistream.py

# 強制 RTSP
python src/pipeline/multistream.py --source rtsp

# 強制本地影片
python src/pipeline/multistream.py --source local

# Op 模式（headless）
python src/pipeline/multistream.py --mode op

# 指定攝影機
python src/pipeline/multistream.py --cameras cam_platform_north,cam_escalator_up

# 強制重繪 ROI（ROI 紀錄以 camera_id_rtsp / camera_id_local 分開儲存）
python src/pipeline/multistream.py --reset-roi cam_platform_north
python src/pipeline/multistream.py --reset-roi all
```

---

## 文件索引

| 文件 | 內容 |
|------|------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 系統架構、模組說明、執行緒模型、輸出流 |
| [`docs/FEATURES.md`](docs/FEATURES.md) | 6 大功能演算法、ROI 設定操作指南 |
| [`docs/CONFIG.md`](docs/CONFIG.md) | cameras.yaml 完整說明、平台設定 |
| [`docs/STATUS.md`](docs/STATUS.md) | 現況、已知 Bug、待討論議題、下一步 |
| [`docs/TECH_EVALUATION.md`](docs/TECH_EVALUATION.md) | 技術選型評估（歷史存檔） |
