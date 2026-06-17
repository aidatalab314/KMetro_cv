import os
import yaml
from datetime import datetime


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_yaml(path: str) -> dict:
    """載入 YAML，若存在 .local.yaml 則自動 deep merge（local 優先）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    local_path = path.replace(".yaml", ".local.yaml")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        data = _deep_merge(data, local)
        log("CONFIG", f"已載入本機覆蓋設定：{local_path}")

    return data


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, msg: str):
    print(f"[{now_str()}] [{level}] {msg}")
