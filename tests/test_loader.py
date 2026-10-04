"""Tests for milk10k.loader.MilkLoader on a tiny fake dataset (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from milk10k.loader import MilkLoader


@pytest.fixture
def fake_dataset(tmp_path):
    """7 metadata rows but only 5 image files on disk."""
    rng = np.random.default_rng(0)
    ids = [f"ISIC_{i:07d}" for i in range(7)]
    for iid in ids[:5]:
        pixels = rng.integers(0, 256, size=(45, 60, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / f"{iid}.jpg")
    labels = ["Benign", "Malignant", "Malignant", "Indeterminate", "Benign", "Benign", "Malignant"]
    return pd.DataFrame({"isic_id": ids, "diagnosis_1": labels}), tmp_path


def test_missing_files_fail_loudly_by_default(fake_dataset):
    meta, img_dir = fake_dataset
    with pytest.raises(FileNotFoundError):
        MilkLoader(meta, img_dir=img_dir, batch_size=2, size=16)


def test_allow_missing_uses_only_available_images(fake_dataset):
    meta, img_dir = fake_dataset
    loader = MilkLoader(meta, img_dir=img_dir, batch_size=2, size=16, allow_missing=True)
    assert len(loader.table) == 5 and loader.n_unavailable == 2


def test_batches_cover_every_image_once(fake_dataset):
    meta, img_dir = fake_dataset
    loader = MilkLoader(meta.head(5), img_dir=img_dir, batch_size=2, size=16)
    batches = list(loader)
    assert [len(ids) for _, _, ids in batches] == [2, 2, 1]
    assert batches[0][0].shape == (2, 16, 16, 3)
    seen = [i for _, _, ids in batches for i in ids]
    assert sorted(seen) == sorted(loader.table["isic_id"])


def test_labels_match_the_metadata(fake_dataset):
    meta, img_dir = fake_dataset
    loader = MilkLoader(meta.head(5), img_dir=img_dir, batch_size=5, size=16)
    _, labels, ids = next(iter(loader))
    expected = meta.set_index("isic_id").loc[ids, "diagnosis_1"].tolist()
    assert [loader.classes[k] for k in labels] == expected


def test_shuffle_is_reproducible_and_changes_each_epoch(fake_dataset):
    meta, img_dir = fake_dataset

    def epoch_order(loader):
        return [i for _, _, ids in loader for i in ids]

    a = MilkLoader(meta.head(5), img_dir=img_dir, batch_size=5, size=16, seed=1)
    b = MilkLoader(meta.head(5), img_dir=img_dir, batch_size=5, size=16, seed=1)
    first_a, first_b = epoch_order(a), epoch_order(b)
    assert first_a == first_b                                   # same seed -> same order
    orders = {tuple(first_a)} | {tuple(epoch_order(a)) for _ in range(4)}
    assert len(orders) > 1                                      # order changes across epochs
