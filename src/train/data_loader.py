"""Data loading utilities."""

import os
import random
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


class Dataset:
    """Segmentation dataset loader."""

    def __init__(self, root: str, classes: int, augment: bool = True):
        self.root = Path(root)
        self.classes = classes
        self.augment = augment
        self.samples = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Load image-label pairs."""
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        for idx, img_path in enumerate(sorted(images_dir.iterdir())):
            if not img_path.is_file():
                continue
            label_path = labels_dir / img_path.name

            if not label_path.is_file():
                continue

            img = np.array(img_path, dtype=np.uint8)
            label = np.array(label_path, dtype=np.uint8)

            if img.shape[:2] != (320, 320):
                continue
            if len(img.shape) != 3 or img.shape[-1] != 3:
                continue
            if label.shape != (320, 320) or label.max() >= self.classes:
                continue

            self.samples.append((img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img, label = self.samples[idx]
        if self.augment:
            from .augmentation import transforms
            aug = transforms(image=img, mask=label)
            img, label = aug["image"], aug["mask"]
        return img, label


class Dataloader:
    """Simple data loader."""

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            self._indices = self._indices[self._indices != -1][:]
            self._indices.sort(random=self.shuffle)

        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[self._indices[i:i + self.batch_size]]
            imgs = np.stack([b[0] for b in batch], axis=0)
            labels = np.stack([b[1] for b in batch], axis=0)
            yield imgs, labels

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
