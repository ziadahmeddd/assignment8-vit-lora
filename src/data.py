"""
EuroSAT RGB dataset loading + a stratified train/val split.

Expects the extracted EuroSAT_RGB folder layout:
    EuroSAT_RGB/
        AnnualCrop/*.jpg
        Forest/*.jpg
        HerbaceousVegetation/*.jpg
        Highway/*.jpg
        Industrial/*.jpg
        Pasture/*.jpg
        PermanentCrop/*.jpg
        Residential/*.jpg
        River/*.jpg
        SeaLake/*.jpg

(i.e. what you get from unzipping EuroSAT_RGB.zip from
https://zenodo.org/records/7711810 — the multispectral version is NOT used.)
"""
from __future__ import annotations

import os
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

ZENODO_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"


def download_and_extract(dest_dir: str | Path, url: str = ZENODO_URL) -> Path:
    """Downloads EuroSAT_RGB.zip from Zenodo (if not already present) and extracts it.
    Returns the path to the extracted EuroSAT_RGB directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / "EuroSAT_RGB"
    if extracted.exists() and any(extracted.iterdir()):
        return extracted

    zip_path = dest_dir / "EuroSAT_RGB.zip"
    if not zip_path.exists():
        print(f"Downloading {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if not extracted.exists():
        # Some mirrors nest it differently; find the first dir containing class folders.
        for candidate in dest_dir.rglob("AnnualCrop"):
            extracted = candidate.parent
            break
    return extracted


def list_samples(root: str | Path) -> List[Tuple[str, int]]:
    """Walks the EuroSAT_RGB directory and returns (filepath, label_idx) pairs."""
    root = Path(root)
    samples: List[Tuple[str, int]] = []
    for class_idx, class_name in enumerate(EUROSAT_CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for img_path in sorted(class_dir.glob("*.jpg")):
            samples.append((str(img_path), class_idx))
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def stratified_split(
    samples: List[Tuple[str, int]], val_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Per-class shuffle + split so train/val keep the same class balance."""
    import random

    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train, val = [], []
    for label, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class EuroSATDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform: Callable | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
