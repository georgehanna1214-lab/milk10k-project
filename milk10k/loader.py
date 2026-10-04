"""Part 4 (Session 2) - a simple numpy data loader that feeds (image, label) batches.

Usage:
    loader = MilkLoader(metadata, batch_size=32, shuffle=True, size=224, normalize="scale")
    for images, labels, ids in loader:      # one pass = one epoch
        ...                                 # images: (B, H, W, C) float32, labels: (B,) int64

Design:
    * FAIL LOUDLY (Milestone 1): if a metadata row has no image file the loader raises
      FileNotFoundError listing the missing files. ``allow_missing=True`` explicitly
      switches back to the "available subset" behaviour (only rows with a file are used,
      and unreadable files are skipped and recorded in ``loader.skipped``);
    * images are loaded and preprocessed batch by batch when the batch is requested,
      never all at once (10,480 full-size images would not fit comfortably in RAM);
    * labels are integers 0..K-1; ``loader.classes[i]`` gives the name of label i;
    * shuffling uses a seeded random generator: a new order every epoch, but the
      same sequence of orders every time the program is run (reproducible).

The PyTorch version used for training is ``milk10k.dataset.MilkDataset``.
"""
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .data import available_subset
from .preprocessing import check_options, preprocess_batch


class MilkLoader:
    """Iterable over batches ``(images, labels, ids)`` of MILK10k images.

    Args:
        metadata: table with one row per image; needs ``isic_id`` and the label column.
        img_dir: folder with the .jpg files (default: config.IMG_DIR).
        label_col: column used as label, e.g. "diagnosis_1" (3 classes) or "dx" (11 classes).
        batch_size: number of images per batch.
        shuffle: shuffle the order of images at the start of every epoch.
        seed: seed of the shuffling, for reproducibility.
        classes: fixed list of class names (defines the label integers); default = sorted names.
        drop_last: drop the last batch if it is smaller than ``batch_size``.
        allow_missing: False (default) = raise if any image file is missing or unreadable;
            True = silently use only the available images and skip unreadable ones.
        **preprocess_options: passed to ``preprocess_image`` (size, color_mode, normalize, ...).
    """

    def __init__(self, metadata: pd.DataFrame, img_dir: Optional[Path] = None,
                 label_col: str = config.TARGET_COL, batch_size: int = 32, shuffle: bool = True,
                 seed: int = config.SEED, classes: Optional[Sequence[str]] = None,
                 drop_last: bool = False, allow_missing: bool = False, **preprocess_options):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        check_options(preprocess_options.get("color_mode", "rgb"),
                      preprocess_options.get("normalize", "scale"), preprocess_options.get("size", 1))
        available = available_subset(metadata, img_dir)
        self.n_unavailable = len(metadata) - len(available)
        if self.n_unavailable and not allow_missing:
            missing = sorted(set(metadata["isic_id"]) - set(available["isic_id"]))
            raise FileNotFoundError(f"{len(missing)} metadata rows have no image file "
                                    f"(e.g. {missing[:5]}); pass allow_missing=True to use only the "
                                    f"available images")
        if available.empty:
            raise ValueError("none of the metadata rows has an image file on disk")
        self.table = available

        self.classes = sorted(self.table[label_col].unique()) if classes is None else list(classes)
        unknown = set(self.table[label_col]) - set(self.classes)
        if unknown:
            raise ValueError(f"labels not in `classes`: {sorted(unknown)}")
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}

        self.img_dir = img_dir
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.allow_missing = allow_missing
        self.preprocess_options = preprocess_options
        self.epoch = 0
        self.skipped = []            # (isic_id, reason) for every image that failed to load

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.table)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        """Yield ``(images, labels, ids)`` for every batch of one epoch."""
        order = np.arange(len(self.table))
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(order)
        self.epoch += 1
        for b in range(len(self)):
            rows = self.table.iloc[order[b * self.batch_size:(b + 1) * self.batch_size]]
            result = preprocess_batch(rows, self.img_dir, skip_errors=self.allow_missing,
                                      **self.preprocess_options)
            self.skipped.extend(result.skipped)
            if not result.ids:               # every image of this batch failed
                continue
            label_of = dict(zip(rows["isic_id"], rows[self.label_col]))
            labels = np.array([self.class_to_index[label_of[i]] for i in result.ids], dtype=np.int64)
            yield result.images, labels, result.ids

    def class_counts(self) -> pd.Series:
        """Number of available images per class (in the order of ``self.classes``)."""
        return self.table[self.label_col].value_counts().reindex(self.classes, fill_value=0)

    def __repr__(self) -> str:
        return (f"MilkLoader({len(self.table)} images, {self.n_unavailable} unavailable, "
                f"{len(self.classes)} classes, batch_size={self.batch_size}, "
                f"{len(self)} batches/epoch, shuffle={self.shuffle})")
