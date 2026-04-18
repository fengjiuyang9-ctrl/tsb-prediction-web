import json
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    # utf-8-sig 兼容带 BOM 的配置文件，避免 JSONDecodeError
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(project_root: str, p: str) -> str:
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(project_root, p))


class StandardScaler1D:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.fitted = False

    def fit(self, values: np.ndarray) -> None:
        vals = np.asarray(values, dtype=np.float64)
        self.mean = float(np.mean(vals))
        self.std = float(np.std(vals))
        if self.std < 1e-8:
            self.std = 1.0
        self.fitted = True

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StandardScaler1D not fitted.")
        vals = np.asarray(values, dtype=np.float64)
        return (vals - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StandardScaler1D not fitted.")
        vals = np.asarray(values, dtype=np.float64)
        return vals * self.std + self.mean
