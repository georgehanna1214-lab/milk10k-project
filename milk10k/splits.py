"""Leak-free train / validation / test splits at LESION level.

The two images of a lesion are near-copies of the same information, so they must
always land in the same split. ``split_lesions`` works on the lesion table (one row
per lesion) with StratifiedGroupKFold: grouped by lesion_id (no lesion in two splits)
and stratified on the 11-class label (every split keeps the class proportions).
Images are assigned afterwards, by lesion_id.
"""
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


def split_lesions(lesions: pd.DataFrame, val_size: float = config.VAL_SIZE,
                  test_size: float = config.TEST_SIZE, seed: int = config.SEED,
                  label_col: str = config.CLASS_COL, n_folds: int = 20) -> Tuple[List[str], List[str], List[str]]:
    """Return (train, val, test) lists of lesion_id.

    StratifiedGroupKFold cuts the lesions into ``n_folds`` small stratified folds (5% each
    by default); ``test_size`` and ``val_size`` are then made of whole folds (15% = 3
    folds), chosen at random with ``seed``. Same seed -> identical split.
    """
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(lesions), dtype=int)
    with warnings.catch_warnings():   # expected: MAL_OTH has 9 lesions, fewer than the 20 folds
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        for k, (_, fold_idx) in enumerate(sgkf.split(lesions, lesions[label_col], groups=lesions["lesion_id"])):
            fold[fold_idx] = k
    n_test, n_val = round(test_size * n_folds), round(val_size * n_folds)
    order = np.random.default_rng(seed).permutation(n_folds)
    test_folds, val_folds = order[:n_test], order[n_test:n_test + n_val]
    ids = lesions["lesion_id"].to_numpy()
    test = ids[np.isin(fold, test_folds)]
    val = ids[np.isin(fold, val_folds)]
    train = ids[~np.isin(fold, np.concatenate([test_folds, val_folds]))]
    return sorted(train), sorted(val), sorted(test)


def check_no_overlap(splits: Dict[str, List[str]]) -> None:
    """Assert that no lesion_id appears in two splits."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]) & set(splits[b])
            assert not shared, f"{len(shared)} lesions in both {a} and {b}, e.g. {sorted(shared)[:3]}"


def images_of(meta: pd.DataFrame, lesion_ids: List[str]) -> pd.DataFrame:
    """All image rows of the given lesions (the split is decided per lesion, images follow)."""
    return meta[meta["lesion_id"].isin(set(lesion_ids))].reset_index(drop=True)


def proportions(lesions: pd.DataFrame, splits: Dict[str, List[str]], col: str) -> pd.DataFrame:
    """Class proportions (%) of ``col`` per split, the global proportions, and the max deviation."""
    table = pd.DataFrame({name: lesions[lesions["lesion_id"].isin(set(ids))][col]
                          .value_counts(normalize=True).mul(100) for name, ids in splits.items()})
    table["global"] = lesions[col].value_counts(normalize=True).mul(100)
    table = table.fillna(0.0)
    table["max_abs_dev_pp"] = table[list(splits)].sub(table["global"], axis=0).abs().max(axis=1)
    return table.round(2)
