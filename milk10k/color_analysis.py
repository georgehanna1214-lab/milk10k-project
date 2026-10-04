"""Part 2 - colour and brightness profiles compared across classes.

Method in short:
    1. pick the same number of lesions per class (fixed seed) and keep BOTH images;
    2. per image: grey and R/G/B histograms (normalised to sum to 1, so every image
       counts equally) and simple statistics (mean / std per channel, brightness);
    3. average per class, SEPARATELY for dermoscopic and clinical images - the two
       image types are taken with different devices and lighting, so mixing them
       would hide or fake class differences.

Images are decoded at reduced size (longest side 256 px) to keep this fast; a
normalised histogram barely changes with resolution.
"""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from .data import image_path
from .visualize import CLASS_COLORS, CLASS_ORDER, INK, MIXED_COLOR

GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)   # ITU-R BT.601 luma
CHANNELS = ("gray", "R", "G", "B")
IMAGE_TYPES = ("dermoscopic", "clinical: close-up")


def sample_per_class(df: pd.DataFrame, label_col: str, n_per_class: int, seed: int) -> pd.DataFrame:
    """Up to ``n_per_class`` lesions per class (all of them if a class is smaller), both images kept.

    Sampling lesions (not images) keeps the dermoscopic and clinical image of each
    lesion together, so the two image types are compared on the same lesions.
    """
    lesions = df.drop_duplicates("lesion_id").sample(frac=1.0, random_state=seed)   # shuffle once
    chosen = lesions.groupby(label_col).head(n_per_class)["lesion_id"]             # first n of each class
    return df[df["lesion_id"].isin(chosen)].reset_index(drop=True)


def load_rgb_small(path: Path, max_side: int = 256) -> np.ndarray:
    """Read a JPEG as an RGB uint8 array whose longest side is at most ``max_side`` pixels."""
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))   # let the JPEG decoder skip detail we do not need
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        return np.asarray(im)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Weighted grey level (0-255): 0.299 R + 0.587 G + 0.114 B."""
    return rgb.astype(np.float32) @ GRAY_WEIGHTS


def normalized_histogram(values: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram over 0-255 that sums to 1 (fraction of pixels per bin)."""
    counts, _ = np.histogram(values, bins=bins, range=(0, 256))
    return counts / counts.sum()


def color_statistics(rgb: np.ndarray) -> dict:
    """Mean and standard deviation of each RGB channel, plus mean brightness (grey level)."""
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    means, stds = pixels.mean(axis=0), pixels.std(axis=0)
    return {"mean_R": means[0], "mean_G": means[1], "mean_B": means[2],
            "std_R": stds[0], "std_G": stds[1], "std_B": stds[2],
            "brightness": float(to_gray(rgb).mean())}


def compute_color_profiles(sample: pd.DataFrame, img_dir: Optional[Path] = None,
                           max_side: int = 256, bins: int = 256):
    """Histograms and colour statistics for every image of ``sample``.

    Returns ``(features, histograms)``: ``features`` has one row per image (the
    sample's columns + colour statistics); ``histograms`` maps 'gray', 'R', 'G', 'B'
    to an array of shape (n_images, bins) aligned with the rows of ``features``.
    """
    stats_rows, hists = [], {c: [] for c in CHANNELS}
    for isic_id in sample["isic_id"]:
        rgb = load_rgb_small(image_path(isic_id, img_dir), max_side)
        stats_rows.append(color_statistics(rgb))
        hists["gray"].append(normalized_histogram(to_gray(rgb), bins))
        for c, name in enumerate("RGB"):
            hists[name].append(normalized_histogram(rgb[..., c], bins))
    features = pd.concat([sample.reset_index(drop=True), pd.DataFrame(stats_rows)], axis=1)
    return features, {c: np.stack(h) for c, h in hists.items()}


def summary_table(features: pd.DataFrame, label_col: str,
                  stat_cols: Sequence[str] = ("brightness", "mean_R", "mean_G", "mean_B",
                                              "std_R", "std_G", "std_B")) -> pd.DataFrame:
    """Mean and standard deviation of each colour statistic per image type and class."""
    grouped = features.groupby(["image_type", label_col])[list(stat_cols)]
    table = grouped.mean().round(1).astype(str) + " ± " + grouped.std().round(1).astype(str)
    table.insert(0, "n_images", grouped.size())
    return table


def separability_auc(features: pd.DataFrame, label_col: str, positive: str, negative: str,
                     stat_cols: Sequence[str] = ("brightness", "mean_R", "mean_G", "mean_B",
                                                 "std_R", "std_G", "std_B")) -> pd.DataFrame:
    """ROC-AUC of each single colour statistic for telling ``positive`` from ``negative``.

    0.5 = the two classes overlap completely on that feature; 1.0 (or 0.0) = perfect
    separation. Values below 0.5 mean the positive class tends to have LOWER values.
    """
    rows = []
    for image_type, part in features.groupby("image_type"):
        part = part[part[label_col].isin([positive, negative])]
        y = (part[label_col] == positive).astype(int)
        rows.append({"image_type": image_type,
                     **{col: roc_auc_score(y, part[col]) for col in stat_cols}})
    return pd.DataFrame(rows).set_index("image_type").round(3)


# --- Plots --------------------------------------------------------------------------------------
def _smooth(curve: np.ndarray, width: int = 5) -> np.ndarray:
    """Moving average, only to make the plotted lines easier to read."""
    return np.convolve(curve, np.ones(width) / width, mode="same")


def plot_mean_histograms(features: pd.DataFrame, hists: dict, label_col: str,
                         classes: Optional[Sequence[str]] = None) -> plt.Figure:
    """Average histogram per class: rows = image type, columns = grey, R, G, B."""
    classes = classes or [c for c in CLASS_ORDER if c in set(features[label_col])]
    fig, axes = plt.subplots(len(IMAGE_TYPES), len(CHANNELS), figsize=(11, 4.6), sharex=True)
    centers = np.arange(hists["gray"].shape[1]) * (256 / hists["gray"].shape[1])
    for r, image_type in enumerate(IMAGE_TYPES):
        for c, channel in enumerate(CHANNELS):
            ax = axes[r, c]
            for cls in classes:
                mask = ((features["image_type"] == image_type) & (features[label_col] == cls)).to_numpy()
                mean_hist = hists[channel][mask].mean(axis=0)
                ax.plot(centers, _smooth(mean_hist) * 100, color=CLASS_COLORS.get(cls, MIXED_COLOR),
                        linewidth=1.6, label=f"{cls} (n={mask.sum()})")
            ax.set_yticks([])
            ax.set_xlim(0, 255)
            if r == 0:
                ax.set_title("grey level" if channel == "gray" else f"{channel} channel", fontsize=9)
            if c == 0:
                ax.set_ylabel(image_type.replace(": ", ":\n"), color=INK["primary"])
            if r == len(IMAGE_TYPES) - 1:
                ax.set_xlabel("pixel value (0-255)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(classes), bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Average normalised histogram per class (share of pixels per value)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_stat_boxplots(features: pd.DataFrame, label_col: str, group_of: dict,
                       stat_cols: Sequence[str] = ("mean_R", "mean_G", "mean_B", "brightness"),
                       order: Optional[Sequence[str]] = None) -> plt.Figure:
    """Per-image colour statistics per class as boxplots: rows = image type, columns = statistic.

    ``group_of`` maps each class to its diagnosis_1 group (or 'mixed') for the box colour.
    """
    order = list(order or sorted(features[label_col].unique()))
    fig, axes = plt.subplots(len(IMAGE_TYPES), len(stat_cols), figsize=(11, 5.2), sharey="col")
    for r, image_type in enumerate(IMAGE_TYPES):
        part = features[features["image_type"] == image_type]
        for c, col in enumerate(stat_cols):
            ax = axes[r, c]
            data = [part.loc[part[label_col] == cls, col].to_numpy() for cls in order]
            box = ax.boxplot(data, patch_artist=True, widths=0.6,
                             medianprops={"color": "white", "linewidth": 1.4},
                             whiskerprops={"color": INK["muted"]}, capprops={"color": INK["muted"]},
                             flierprops={"marker": "o", "markersize": 2, "markerfacecolor": INK["muted"],
                                         "markeredgecolor": "none", "alpha": 0.6})
            for patch, cls in zip(box["boxes"], order):
                patch.set_facecolor(CLASS_COLORS.get(group_of.get(cls), MIXED_COLOR))
                patch.set_edgecolor("white")
            ax.set_xticks(range(1, len(order) + 1), order if r == len(IMAGE_TYPES) - 1 else [],
                          rotation=60, ha="right", fontsize=7)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(col.replace("mean_", "mean ") + (" (grey)" if col == "brightness" else ""),
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(image_type.replace(": ", ":\n") + "\n(0-255)", color=INK["primary"])
    fig.suptitle("Per-image colour statistics by 11-class diagnosis (box colour = diagnosis_1; "
                 "grey = AKIEC, mixed)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig
