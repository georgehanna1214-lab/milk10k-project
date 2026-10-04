"""PyTorch datasets and data loaders (Milestone 1, B8) - the upgrade of the Session 2 MilkLoader.

    MilkDataset     one item = one IMAGE -> (image_tensor, label, isic_id)      [project choice]
    LesionDataset   one item = one LESION -> dict with one tensor per view       [Part A, A3.6]
    make_loaders    train (augment + shuffle or weighted sampler) / val / test (no shuffle)
    aggregate_predictions   image-level probabilities -> one prediction per lesion

How the two images of a lesion are used (B6): both images are training samples that
carry the lesion's label (option a: twice the training images, both views learned).
Evaluation is done PER LESION by averaging the probabilities of its two images, so a
lesion counts once in every metric.
"""
import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config
from .data import image_path

TableLike = Union[pd.DataFrame, str, Path]


def seed_everything(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible NumPy / random seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_table(table: TableLike) -> pd.DataFrame:
    return pd.read_csv(table) if isinstance(table, (str, Path)) else table.reset_index(drop=True).copy()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as f:
        return f.convert("RGB")


class MilkDataset(Dataset):
    """One item = one image: ``(image_tensor, label, isic_id)``.

    Args:
        table: split CSV path or DataFrame with ``isic_id`` and the label column.
        label_map: class name -> integer, e.g. ``label_map["diagnosis_1"]`` from label_map.json.
        label_col: column holding the class names.
        transform: torchvision transform applied to the PIL image (train OR eval transform).
        img_dir: image folder (default: config.IMG_DIR).
        allow_missing: False (default) = raise FileNotFoundError if any image file is missing;
            True = drop those rows explicitly.
    """

    def __init__(self, table: TableLike, label_map: Dict[str, int], label_col: str = config.TARGET_COL,
                 transform=None, img_dir: Optional[Path] = None, allow_missing: bool = False):
        df = _read_table(table)
        paths = [image_path(i, img_dir) for i in df["isic_id"]]
        exists = np.array([p.is_file() for p in paths])
        if not exists.all():
            missing = df.loc[~exists, "isic_id"].tolist()
            if not allow_missing:
                raise FileNotFoundError(f"{len(missing)} image files are missing, e.g. {missing[:5]} "
                                        f"(pass allow_missing=True to drop them)")
            df = df[exists].reset_index(drop=True)
            paths = [p for p, ok in zip(paths, exists) if ok]
        unknown = set(df[label_col]) - set(label_map)
        if unknown:
            raise ValueError(f"labels missing from label_map: {sorted(unknown)}")
        self.table = df
        self.paths = paths
        self.labels = df[label_col].map(label_map).to_numpy(dtype=np.int64)
        self.ids = df["isic_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = _load_rgb(self.paths[i])
        x = self.transform(img) if self.transform is not None else torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, int(self.labels[i]), self.ids[i]


class LesionDataset(Dataset):
    """One item = one lesion: ``{"derm": tensor, "clinical": tensor, "label": int, "lesion_id": str}``.

    ``lesion_table`` needs the columns ``lesion_id``, ``derm_id``, ``clinical_id`` and the label
    column (see ``labels.lesion_table``). The transform is applied to each view separately,
    so each view gets its own random augmentation.
    """

    VIEW_COLUMNS = {"derm": "derm_id", "clinical": "clinical_id"}

    def __init__(self, lesion_table: TableLike, img_dir: Optional[Path] = None,
                 label_col: str = config.TARGET_COL, transform=None,
                 views: Sequence[str] = ("derm", "clinical"), label_map: Optional[Dict[str, int]] = None):
        df = _read_table(lesion_table)
        bad_views = set(views) - set(self.VIEW_COLUMNS)
        if bad_views:
            raise ValueError(f"unknown views {sorted(bad_views)}; choose from {list(self.VIEW_COLUMNS)}")
        missing = [df.at[r, self.VIEW_COLUMNS[v]] for v in views for r in df.index
                   if not image_path(df.at[r, self.VIEW_COLUMNS[v]], img_dir).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image files are missing, e.g. {missing[:5]}")
        self.label_map = label_map or {name: i for i, name in enumerate(sorted(df[label_col].unique()))}
        self.table, self.views, self.img_dir, self.transform = df, tuple(views), img_dir, transform
        self.labels = df[label_col].map(self.label_map).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> dict:
        row = self.table.iloc[i]
        item = {}
        for view in self.views:
            img = _load_rgb(image_path(row[self.VIEW_COLUMNS[view]], self.img_dir))
            item[view] = self.transform(img) if self.transform is not None else torch.from_numpy(
                np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        item["label"] = int(self.labels[i])
        item["lesion_id"] = row["lesion_id"]
        return item


def aggregate_predictions(image_probs: np.ndarray, image_table: pd.DataFrame) -> pd.DataFrame:
    """Average the class probabilities of each lesion's images -> one prediction per lesion.

    ``image_probs`` is (n_images, n_classes), row i belonging to row i of ``image_table``
    (which needs a ``lesion_id`` column). Returns one row per lesion: the mean probability
    of every class and ``pred`` (the class with the highest mean probability).
    """
    probs = np.asarray(image_probs, dtype=float)
    if len(probs) != len(image_table):
        raise ValueError("image_probs and image_table must have the same number of rows")
    frame = pd.DataFrame(probs, columns=list(range(probs.shape[1])))
    frame["lesion_id"] = image_table["lesion_id"].to_numpy()
    lesion = frame.groupby("lesion_id").mean()
    lesion["pred"] = lesion.to_numpy().argmax(axis=1)
    return lesion


def class_weights(labels: Sequence[int], n_classes: int) -> np.ndarray:
    """'Balanced' loss weights: n_samples / (n_classes * count_of_class). Rare class -> big weight."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"a class has no training samples: counts={counts.tolist()}")
    return counts.sum() / (n_classes * counts)


def save_class_weights(weights: np.ndarray, class_names: Sequence[str],
                       path: Path = config.ARTIFACTS_DIR / "class_weights.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"classes": list(class_names), "weights": [round(float(w), 4) for w in weights],
                                "formula": "n_train / (n_classes * count_in_train)",
                                "computed_on": "train split only"}, indent=2) + "\n")
    return path


def weighted_sampler(labels: Sequence[int], seed: int = config.SEED) -> WeightedRandomSampler:
    """Sampler that draws every class equally often on average (rare images are repeated)."""
    labels = np.asarray(labels)
    per_class = 1.0 / np.bincount(labels)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(per_class[labels], dtype=torch.double),
                                 num_samples=len(labels), replacement=True, generator=generator)


def make_loaders(datasets: Dict[str, Dataset], batch_size: int = 32, num_workers: int = 0,
                 use_sampler: bool = False, seed: int = config.SEED) -> Dict[str, DataLoader]:
    """DataLoaders for 'train', 'val' and 'test'.

    train: shuffled every epoch (or drawn with the class-balancing ``weighted_sampler`` when
           ``use_sampler=True``); its dataset should carry the TRAIN transform.
    val / test: fixed order, no augmentation (their datasets carry the EVAL transform).
    Use EITHER the weighted sampler OR the class-weighted loss, not both: together they
    correct the imbalance twice.
    """
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker,
                  persistent_workers=num_workers > 0)
    train_ds = datasets["train"]
    if use_sampler:
        train = DataLoader(train_ds, sampler=weighted_sampler(train_ds.labels, seed), **common)
    else:
        train = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    loaders = {"train": train}
    for name in ("val", "test"):
        if name in datasets:
            loaders[name] = DataLoader(datasets[name], shuffle=False, **common)
    return loaders
