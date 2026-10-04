"""Loading the MILK10k tables and linking metadata rows to image files."""
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def image_path(isic_id: str, img_dir: Optional[Path] = None) -> Path:
    """Return the path of the .jpg file for one image id (e.g. 'ISIC_0051817')."""
    folder = Path(img_dir) if img_dir is not None else config.IMG_DIR
    return folder / f"{isic_id}.jpg"


def load_metadata(path: Path = config.METADATA_PATH) -> pd.DataFrame:
    """Read metadata.csv: one row per IMAGE (10,480 rows, 2 images per lesion)."""
    return pd.read_csv(path)


def load_ground_truth(path: Path = config.GT_PATH) -> pd.DataFrame:
    """Read training_gt.csv: one row per LESION, one-hot columns for the 11 classes."""
    return pd.read_csv(path)


def onehot_to_label(gt: pd.DataFrame) -> pd.Series:
    """Turn the one-hot class columns into one string label per lesion (e.g. 'BCC')."""
    return gt.drop(columns="lesion_id").idxmax(axis=1)


def load_dataset() -> pd.DataFrame:
    """Metadata (per image) plus the 11-class label ``dx`` from training_gt.csv.

    The 11-class label is defined per lesion, so it is joined on ``lesion_id``.
    """
    meta = load_metadata()
    gt = load_ground_truth()
    labels = pd.DataFrame({"lesion_id": gt["lesion_id"], config.CLASS_COL: onehot_to_label(gt)})
    df = meta.merge(labels, on="lesion_id", how="left", validate="many_to_one")
    assert df[config.CLASS_COL].notna().all(), "some images have no 11-class label"
    return df


def available_subset(df: pd.DataFrame, img_dir: Optional[Path] = None) -> pd.DataFrame:
    """Keep only the rows whose image file exists on disk.

    The metadata describes all 10,480 images, but a local copy may hold only some
    of them. With the full dataset this keeps every row.
    """
    folder = Path(img_dir) if img_dir is not None else config.IMG_DIR
    on_disk = {p.stem for p in folder.glob("*.jpg")}   # one directory listing, not 10k lookups
    mask = df["isic_id"].isin(on_disk)
    return df[mask].reset_index(drop=True)
