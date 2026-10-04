"""Tests for the Milestone 1 pipeline: splits, transforms, datasets (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from milk10k import config
from milk10k.data import load_ground_truth, load_metadata
from milk10k.dataset import LesionDataset, MilkDataset, aggregate_predictions, class_weights
from milk10k.labels import lesion_table
from milk10k.splits import check_no_overlap, split_lesions
from milk10k.transforms import eval_transform, train_transform

HAS_DATA = config.METADATA_PATH.is_file() and config.GT_PATH.is_file()


@pytest.fixture(scope="module")
def lesions():
    return lesion_table(load_metadata(), load_ground_truth())


@pytest.mark.skipif(not HAS_DATA, reason="MILK10k metadata not available")
def test_lesion_table_shape(lesions):
    assert len(lesions) == 5240 and lesions["lesion_id"].is_unique
    assert lesions[["derm_id", "clinical_id"]].notna().all().all()


@pytest.mark.skipif(not HAS_DATA, reason="MILK10k metadata not available")
@pytest.mark.parametrize("seed", range(10))
def test_split_properties_for_10_seeds(lesions, seed):
    train, val, test = split_lesions(lesions, val_size=0.15, test_size=0.15, seed=seed)
    check_no_overlap({"train": train, "val": val, "test": test})               # no shared lesion
    assert len(train) + len(val) + len(test) == len(lesions)                   # nothing lost
    for ids, wanted in ((val, 0.15), (test, 0.15), (train, 0.70)):
        assert abs(len(ids) / len(lesions) - wanted) <= 0.01                   # +-1 percentage point
    assert split_lesions(lesions, 0.15, 0.15, seed=seed) == (train, val, test)  # reproducible


def test_eval_transform_is_deterministic():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (450, 600, 3), dtype=np.uint8))
    tf = eval_transform([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    assert torch.equal(tf(img), tf(img))
    assert tf(img).shape == (3, 224, 224)


def test_train_transform_output_shape():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (450, 600, 3), dtype=np.uint8))
    assert train_transform([0.5] * 3, [0.25] * 3)(img).shape == (3, 224, 224)


def test_dataset_fails_loudly_on_missing_file(tmp_path):
    table = pd.DataFrame({"isic_id": ["ISIC_MISSING"], "diagnosis_1": ["Benign"]})
    with pytest.raises(FileNotFoundError):
        MilkDataset(table, {"Benign": 0}, img_dir=tmp_path)
    assert len(MilkDataset(table, {"Benign": 0}, img_dir=tmp_path, allow_missing=True)) == 0


def test_lesion_dataset_returns_both_views(tmp_path):
    for iid in ("D1", "C1"):
        Image.fromarray(np.zeros((45, 60, 3), dtype=np.uint8)).save(tmp_path / f"{iid}.jpg")
    table = pd.DataFrame({"lesion_id": ["L1"], "derm_id": ["D1"], "clinical_id": ["C1"],
                          "diagnosis_1": ["Malignant"]})
    item = LesionDataset(table, img_dir=tmp_path)[0]
    assert item["derm"].shape == item["clinical"].shape == (3, 45, 60)
    assert item["lesion_id"] == "L1"


def test_aggregate_predictions_averages_the_two_views():
    table = pd.DataFrame({"lesion_id": ["A", "A", "B", "B"]})
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.4, 0.6]])
    out = aggregate_predictions(probs, table)
    assert np.allclose(out.loc["A", [0, 1]], [0.7, 0.3]) and out.loc["A", "pred"] == 0
    assert np.allclose(out.loc["B", [0, 1]], [0.3, 0.7]) and out.loc["B", "pred"] == 1


def test_class_weights_are_balanced():
    w = class_weights([0, 0, 0, 1], n_classes=2)          # 3 vs 1 sample
    assert np.allclose(w, [4 / 6, 4 / 2])
