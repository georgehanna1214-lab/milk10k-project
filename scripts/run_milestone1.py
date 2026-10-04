"""Milestone 1 - leak-free, reproducible data pipeline. Runs steps B1-B8 and saves every output.

    python scripts/run_milestone1.py

figures -> figures/milestone1/    tables and logs -> outputs/milestone1/
splits  -> splits/{train,val,test}.csv    label map, norm stats, class weights -> artifacts/
Takes about 3 minutes (verifies all 10,480 images and loads one full training epoch).
"""
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from milk10k import color_analysis as ca
from milk10k import config, quality
from milk10k import labels as lb
from milk10k import splits as sp
from milk10k import transforms as tr
from milk10k.data import image_path, load_dataset, load_ground_truth, load_metadata
from milk10k.dataset import (MilkDataset, class_weights, make_loaders, save_class_weights,
                             seed_everything)
from milk10k.visualize import CLASS_ORDER, apply_style, plot_class_balance, save_figure, show_grid

FIG = config.FIGURES_DIR / "milestone1"
OUT = config.OUTPUTS_DIR / "milestone1"
SPLIT_DATE = "2026-10-04"
BATCH_SIZE = 32
NUM_WORKERS = 4
T, C = config.TARGET_COL, config.CLASS_COL


def b1_integrity(meta: pd.DataFrame) -> None:
    print("\n=== B1: full-dataset integrity ===")
    missing = quality.missing_files(meta)
    print(f"metadata rows without an image file: {len(missing)} {missing[:10]}")
    assert not missing, "some metadata rows have no image file"
    start = time.perf_counter()
    verified = quality.verify_images(meta)
    bad = verified[~verified["readable"]]
    print(f"Image.verify() on {len(verified):,} images: {len(bad)} unreadable "
          f"({time.perf_counter() - start:.0f} s)")
    if len(bad):
        bad.to_csv(OUT / "unreadable_images.csv", index=False)
    summary = quality.size_summary(verified)
    summary.to_csv(OUT / "image_size_summary.csv", index=False)
    print(summary.to_string(index=False))


def b2_eda(df: pd.DataFrame, lesions: pd.DataFrame) -> None:
    print("\n=== B2: label-centred EDA ===")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1, 2.6]})
    plot_class_balance(lesions[T], ax=axes[0], order=CLASS_ORDER, unit="lesions", title="diagnosis_1")
    plot_class_balance(lesions[C], ax=axes[1], order=lesions[C].value_counts().index.tolist(), log=True,
                       unit="lesions", title="11-class diagnosis, log scale")
    fig.tight_layout()
    save_figure(fig, FIG / "class_distribution.png")
    mapping = lb.class_to_diagnosis1(lesions)
    mapping.to_csv(OUT / "class_to_diagnosis1.csv")
    print("11-class -> diagnosis_1 (lesions):\n", mapping.to_string())

    # Session 2 findings re-checked on ALL images / lesions
    print("re-checking Session 2 findings on the full dataset (colour of all 10,480 images) ...")
    features, _ = ca.compute_color_profiles(df, bins=32)
    auc = ca.separability_auc(features, T, positive="Malignant", negative="Benign")
    auc.to_csv(OUT / "color_auc_full_dataset.csv")
    best = (auc - 0.5).abs().max(axis=1) + 0.5
    blue = features.groupby("image_type")["mean_B"].mean()
    ages = lesions.groupby(C)["age"].median()
    site = pd.crosstab(lesions["site"].fillna("(missing)"), lesions[T], normalize="index").mul(100)
    manip = pd.crosstab(df["image_manipulation"], df[T], normalize="index").mul(100)
    rows = [
        ("Age differs strongly by class (Kruskal-Wallis; nevi from young patients)",
         f"yes: median age NV {ages['NV']:.0f} vs BCC {ages['BCC']:.0f}, SCCKA {ages['SCCKA']:.0f}",
         "age is NOT an input of the image pipeline; stratified splits keep the age mix similar; "
         "check later that the model does not just learn 'young skin = benign'"),
        ("Six columns leak the label (diagnosis_2/3/4, melanocytic, diagnosis_confirm_type, concomitant_biopsy)",
         "yes (they are label-derived by construction)",
         "excluded from model inputs; the split CSVs carry only ids, image_type and the labels"),
        ("image_manipulation = 'altered' is enriched in Indeterminate lesions",
         f"yes: {manip.loc['altered', 'Indeterminate']:.1f}% vs {manip.loc['instrument only', 'Indeterminate']:.1f}% Indeterminate",
         "possible acquisition shortcut: not an input; report metrics separately for altered images"),
        ("Colour alone separates Malignant from Benign only weakly (sample AUC <= 0.66)",
         f"yes, on all 10,480 images: best single-feature AUC {best.max():.2f}",
         "a CNN must learn spatial patterns; colour augmentation stays mild (no hue jitter)"),
        ("Dermoscopic images are much bluer than clinical ones (device effect > class effect)",
         f"yes: mean B {blue['dermoscopic']:.0f} (dermoscopic) vs {blue['clinical: close-up']:.0f} (clinical)",
         "both images of a lesion go to the same split, so every split has both types in equal numbers; "
         "evaluate per image type as well"),
        ("A missing body site goes with more Benign lesions",
         f"yes: {site.loc['(missing)', 'Benign']:.1f}% Benign when missing vs {100 * (lesions[T] == 'Benign').mean():.1f}% overall; "
         "explained: 3,850 of the 3,912 'missing' images are trunk lesions in training_input.csv",
         "missing is not random: if site is used, recover 'trunk' from training_input.csv, the rest 'unknown'"),
    ]
    findings = pd.DataFrame(rows, columns=["Session 2 finding", "Still true on the full dataset?",
                                           "Consequence for the pipeline"])
    findings.to_csv(OUT / "session2_findings_recheck.csv", index=False)
    print(findings.to_string(index=False))

    # 3 x 4 gallery: one dermoscopic example per class (+ the Indeterminate side of AKIEC)
    picks = []
    for dx in sorted(lesions[C].unique()):
        want = "Malignant" if dx == "AKIEC" else None
        pool = lesions[(lesions[C] == dx) & ((lesions[T] == want) if want else True)]
        picks.append(pool.sample(1, random_state=config.SEED).iloc[0])
    picks.append(lesions[(lesions[C] == "AKIEC") & (lesions[T] == "Indeterminate")]
                 .sample(1, random_state=config.SEED).iloc[0])
    images = [np.asarray(Image.open(image_path(p["derm_id"])).convert("RGB")) for p in picks]
    titles = [f"{p[C]} · {p[T]}" for p in picks]
    save_figure(show_grid(images, labels=titles, n=12, ncols=4, cell_size=2.3,
                          title="One dermoscopic example per class"), FIG / "class_gallery.jpg")


def b3_labels(lesions: pd.DataFrame) -> dict:
    print("\n=== B3: label strategy ===")
    path = lb.save_label_map()
    label_map = lb.load_label_map(path)
    counts = lb.add_dx10(lesions)["dx10"].value_counts()
    print(f"saved {path.relative_to(config.REPO_ROOT)}; dx10 lesion counts: {counts.to_dict()}")
    return label_map


def b4_quality(meta: pd.DataFrame) -> None:
    print("\n=== B4: data-quality report ===")
    report = quality.quality_report(meta)
    (OUT / "data_quality_report.md").write_text(report + "\n")
    print(report)


def b5_splits(df: pd.DataFrame, lesions: pd.DataFrame) -> dict:
    print("\n=== B5: splits ===")
    train, val, test = sp.split_lesions(lesions, seed=config.SEED)
    splits = {"train": train, "val": val, "test": test}
    sp.check_no_overlap(splits)
    print("zero lesion overlap between every pair of splits: OK")
    config.SPLITS_DIR.mkdir(exist_ok=True)
    for name, ids in splits.items():
        images = lb.add_dx10(sp.images_of(df, ids))[["isic_id", "lesion_id", "image_type", T, C, "dx10"]]
        per_lesion = images.groupby("lesion_id")["image_type"].agg(lambda s: tuple(sorted(s)))
        assert len(images) == 2 * len(ids), f"{name}: not exactly 2 images per lesion"
        assert (per_lesion == ("clinical: close-up", "dermoscopic")).all(), f"{name}: image types"
        images.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(ids):,} lesions ({100 * len(ids) / len(lesions):.1f}%), "
              f"{len(images):,} images, exactly 2 per lesion (1 dermoscopic + 1 clinical): OK")
    for col in (C, T):
        table = sp.proportions(lesions, splits, col)
        table.to_csv(OUT / f"split_proportions_{col}.csv")
        print(f"\nclass proportions (% of lesions) by split, {col}; max deviation from global = "
              f"{table['max_abs_dev_pp'].max():.2f} pp\n", table.to_string())
    print(f"\nseed {config.SEED}, created {SPLIT_DATE}")
    return splits


def b6_b7_transforms(label_map: dict, lesions: pd.DataFrame) -> tuple:
    print("\n=== B6-B7: normalisation statistics (train only) and augmentation ===")
    lm = label_map[T]
    stats_ds = MilkDataset(config.SPLITS_DIR / "train.csv", lm, transform=tr.stats_transform())
    stats = tr.compute_norm_stats(DataLoader(stats_ds, batch_size=64, num_workers=NUM_WORKERS))
    tr.save_norm_stats(stats)
    print(f"norm stats from {len(stats_ds):,} TRAIN images: mean {stats['mean']} std {stats['std']}")
    mean, std = stats["mean"], stats["std"]
    train_tf, eval_tf = tr.train_transform(mean, std), tr.eval_transform(mean, std)
    print("train_transform:", train_tf, "\neval_transform:", eval_tf)

    sample = Image.open(image_path(lesions["derm_id"].iloc[0])).convert("RGB")
    assert torch.equal(eval_tf(sample), eval_tf(sample)), "eval_transform is not deterministic"
    print("eval_transform applied twice to the same image -> identical tensors: OK")

    torch.manual_seed(config.SEED)
    rows, titles = [], []
    for dx in ("MEL", "BCC", "VASC"):
        iid = lesions[lesions[C] == dx]["derm_id"].sample(1, random_state=config.SEED).iloc[0]
        img = Image.open(image_path(iid)).convert("RGB")
        rows.append(np.asarray(tr.stats_transform()(img).permute(1, 2, 0)))
        titles.append(f"{dx} original")
        for k in range(7):
            rows.append(tr.denormalize(train_tf(img), mean, std))
            titles.append(f"{dx} augmented {k + 1}")
    save_figure(show_grid(rows, labels=titles, n=len(rows), ncols=8, cell_size=1.6,
                          title="1 original + 7 train_transform outputs (MEL, BCC and the rare VASC)"),
                FIG / "augmentations.jpg")
    return train_tf, eval_tf, mean, std


def b8_loaders(label_map: dict, train_tf, eval_tf, mean, std) -> None:
    print("\n=== B8: datasets and loaders ===")
    seed_everything(config.SEED)
    lm = label_map[T]
    classes = sorted(lm, key=lm.get)
    datasets = {"train": MilkDataset(config.SPLITS_DIR / "train.csv", lm, transform=train_tf),
                "val": MilkDataset(config.SPLITS_DIR / "val.csv", lm, transform=eval_tf),
                "test": MilkDataset(config.SPLITS_DIR / "test.csv", lm, transform=eval_tf)}
    print({k: len(v) for k, v in datasets.items()}, "images")

    weights = class_weights(datasets["train"].labels, len(classes))
    save_class_weights(weights, classes)
    counts = np.bincount(datasets["train"].labels)
    print("train class counts:", dict(zip(classes, counts.tolist())),
          f"-> imbalance ratio {counts.max() / counts.min():.1f}:1")
    print("class weights (train):", dict(zip(classes, weights.round(3).tolist())))

    try:
        MilkDataset(pd.DataFrame({"isic_id": ["ISIC_DOES_NOT_EXIST"], T: ["Benign"]}), lm)
    except FileNotFoundError as err:
        print("missing file -> fails loudly:", err)

    loaders = make_loaders(datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    x, y, ids = next(iter(loaders["train"]))
    print(f"train batch: images {tuple(x.shape)} {x.dtype}, min {x.min():.2f} max {x.max():.2f}, "
          f"labels {tuple(y.shape)} {y.dtype}, ids e.g. {ids[:2]}")
    xv, _, _ = next(iter(loaders["val"]))
    print(f"val batch: images {tuple(xv.shape)}, min {xv.min():.2f} max {xv.max():.2f}")

    for use_sampler in (False, True):
        loader = make_loaders(datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              use_sampler=use_sampler)["train"]
        hist = Counter()
        for b, (_, labels, _) in enumerate(loader):
            hist.update(labels.tolist())
            if b == 19:
                break
        name = "WeightedRandomSampler" if use_sampler else "plain shuffle"
        print(f"label histogram of 20 train batches ({name}):",
              {classes[k]: hist[k] for k in range(len(classes))})

    start = time.perf_counter()
    n = sum(len(labels) for _, labels, _ in loaders["train"])
    print(f"one full train epoch: {n:,} images in {time.perf_counter() - start:.1f} s "
          f"(batch size {BATCH_SIZE}, {NUM_WORKERS} workers)")

    images, labels, ids = next(iter(loaders["train"]))
    shown = tr.denormalize(images[:16], mean, std)
    save_figure(show_grid(shown, labels=[classes[k] for k in labels[:16]], ids=ids[:16], n=16, ncols=8,
                          cell_size=1.6, title="A train batch after train_transform (de-normalised for display)"),
                FIG / "train_batch.jpg")


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    df = load_dataset()
    lesions = lb.lesion_table(meta, load_ground_truth())
    b1_integrity(meta)
    b2_eda(df, lesions)
    label_map = b3_labels(lesions)
    b4_quality(meta)
    b5_splits(df, lesions)
    train_tf, eval_tf, mean, std = b6_b7_transforms(label_map, lesions)
    b8_loaders(label_map, train_tf, eval_tf, mean, std)
    print("\nDone. Outputs: figures/milestone1/, outputs/milestone1/, splits/, artifacts/")


if __name__ == "__main__":
    main()
