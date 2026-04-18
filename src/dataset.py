import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import StandardScaler1D


@dataclass
class MetaProcessor:
    age_col: str
    sex_col: str
    race_col: str
    weight_col: str
    use_weight: bool
    sex_categories: List[str]
    race_categories: List[str]
    extra_numeric_cols: List[str]

    def __post_init__(self):
        self.age_scaler = StandardScaler1D()
        self.weight_scaler = StandardScaler1D()
        self.extra_scalers = {c: StandardScaler1D() for c in self.extra_numeric_cols}
        self.age_fitted = False
        self.weight_fitted = False
        self.extra_fitted = False

    @property
    def output_dim(self) -> int:
        base = 1 + len(self.sex_categories) + len(self.race_categories)
        if self.use_weight:
            base += 1
        base += len(self.extra_numeric_cols)
        return base

    def fit(self, df_train: pd.DataFrame) -> None:
        age_vals = pd.to_numeric(df_train[self.age_col], errors="coerce").fillna(0.0).values
        self.age_scaler.fit(age_vals)
        self.age_fitted = True
        if self.use_weight:
            weight_vals = pd.to_numeric(df_train[self.weight_col], errors="coerce").fillna(0.0).values
            self.weight_scaler.fit(weight_vals)
            self.weight_fitted = True
        for c in self.extra_numeric_cols:
            vals = pd.to_numeric(df_train[c], errors="coerce").fillna(0.0).values
            self.extra_scalers[c].fit(vals)
        self.extra_fitted = True

    def _one_hot(self, token: str, categories: List[str]) -> np.ndarray:
        vec = np.zeros(len(categories), dtype=np.float32)
        if token not in categories:
            token = "UNK" if "UNK" in categories else categories[-1]
        vec[categories.index(token)] = 1.0
        return vec

    def transform_row(self, row: pd.Series) -> np.ndarray:
        if not self.age_fitted:
            raise RuntimeError("MetaProcessor not fitted on train set.")
        feats = []

        age = pd.to_numeric(row[self.age_col], errors="coerce")
        if pd.isna(age):
            age = 0.0
        age_z = self.age_scaler.transform(np.array([age], dtype=np.float64))[0]
        feats.append(float(age_z))

        sex = str(row[self.sex_col]) if not pd.isna(row[self.sex_col]) else "UNK"
        race = str(row[self.race_col]) if not pd.isna(row[self.race_col]) else "UNK"
        feats.extend(self._one_hot(sex, self.sex_categories).tolist())
        feats.extend(self._one_hot(race, self.race_categories).tolist())

        if self.use_weight:
            if not self.weight_fitted:
                raise RuntimeError("weight scaler expected fitted.")
            weight = pd.to_numeric(row[self.weight_col], errors="coerce")
            if pd.isna(weight):
                weight = 0.0
            weight_z = self.weight_scaler.transform(np.array([weight], dtype=np.float64))[0]
            feats.append(float(weight_z))

        if self.extra_numeric_cols:
            if not self.extra_fitted:
                raise RuntimeError("extra numeric scalers expected fitted.")
            for c in self.extra_numeric_cols:
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.isna(v):
                    v = 0.0
                vz = self.extra_scalers[c].transform(np.array([v], dtype=np.float64))[0]
                feats.append(float(vz))

        return np.asarray(feats, dtype=np.float32)


def build_transforms(image_size: int, is_train: bool):
    if is_train:
        return T.Compose(
            [
                T.Resize(int(image_size * 1.14)),
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class E2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        image_col: str,
        subject_col: str,
        target_col: str,
        meta_processor: MetaProcessor,
        transform,
        target_scaler: StandardScaler1D,
        target_standardize: bool,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_root = image_root
        self.image_col = image_col
        self.subject_col = subject_col
        self.target_col = target_col
        self.meta_processor = meta_processor
        self.transform = transform
        self.target_scaler = target_scaler
        self.target_standardize = target_standardize

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_rel_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            raise ValueError(f"Absolute image path is not allowed: {rel_path}")
        return os.path.normpath(os.path.join(self.image_root, rel_path))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row[self.image_col])
        abs_path = self._resolve_rel_path(rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image missing: {abs_path}")

        with Image.open(abs_path) as im:
            im = im.convert("RGB")
            image = self.transform(im)

        meta = self.meta_processor.transform_row(row)
        y = float(row[self.target_col])
        if self.target_standardize:
            y_model = float(self.target_scaler.transform(np.array([y], dtype=np.float64))[0])
        else:
            y_model = y

        return {
            "image": image,
            "meta": torch.tensor(meta, dtype=torch.float32),
            "target": torch.tensor(y_model, dtype=torch.float32),
            "target_umol": torch.tensor(y, dtype=torch.float32),
            "subject_id": str(row[self.subject_col]),
            "image_path": rel_path,
        }
