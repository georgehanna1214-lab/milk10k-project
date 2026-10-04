"""Part 5 - reusable plots for inspecting the data and the loader output.

    show_grid(...)            grid of images with labels as titles (arrays OR a loader batch)
    plot_class_balance(...)   bar chart of how many images/lesions per class
    plot_batch_summary(...)   pixel-value distribution of a batch + raw vs processed examples

The colours and chart style used by every figure in the project live here too,
so all figures look the same and the class colours never change between plots.
"""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Shared style ---------------------------------------------------------------
# Class colours: three colour-blind-safe hues (checked with a palette validator).
# A class always keeps its colour, in every figure.
CLASS_COLORS = {"Benign": "#2a78d6", "Indeterminate": "#1baf7a", "Malignant": "#eb6834"}
CLASS_ORDER = ["Benign", "Indeterminate", "Malignant"]   # alphabetical = label index order
SINGLE_COLOR = "#2a78d6"   # used when only one series is drawn
MIXED_COLOR = "#898781"    # neutral grey, e.g. a class that mixes two diagnosis_1 values
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titlelocation": "left",
    "axes.labelsize": 8.5, "axes.labelcolor": INK["secondary"], "text.color": INK["primary"],
    "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK["axis"], "ytick.color": INK["axis"],
    "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "savefig.dpi": 200, "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Apply the project chart style to every matplotlib figure made afterwards."""
    plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a figure (creating the folder if needed) and close it.

    Use .png for charts (sharp lines) and .jpg for grids of photos (about 10x smaller).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        fig.savefig(path, pil_kwargs={"quality": 88})
    else:
        fig.savefig(path)
    plt.close(fig)
    return path


# --- Images ---------------------------------------------------------------------
def to_displayable(img: np.ndarray) -> np.ndarray:
    """Make any image array safe for ``imshow``. Used for DISPLAY only, never for training.

    * uint8 images (0-255) are divided by 255;
    * float images already in [0, 1] are shown as they are;
    * anything else (e.g. z-scored values around 0) is stretched to [0, 1].
    Single-channel images (H, W, 1) become (H, W) so they can be drawn in grey.
    PyTorch tensors (C, H, W) are converted to numpy (H, W, C) first (Milestone 1).
    """
    if hasattr(img, "detach"):                                 # torch tensor
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr / 255.0
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if lo >= 0.0 and hi <= 1.0:
        return arr
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def show_grid(images, labels: Optional[Sequence] = None, class_names: Optional[Sequence[str]] = None,
              ids: Optional[Sequence[str]] = None, n: int = 16, ncols: int = 4,
              cell_size: float = 2.0, title: Optional[str] = None) -> plt.Figure:
    """Show up to ``n`` images in a grid with their labels as titles.

    ``images`` can be a list/array of images, or directly a batch from
    ``MilkLoader`` - the tuple ``(images, labels, ids)`` - so the loader output can
    be checked with one call: ``show_grid(next(iter(loader)), class_names=loader.classes)``.
    Integer labels are turned into names with ``class_names``.
    """
    if isinstance(images, tuple) and len(images) == 3:      # a batch from the loader
        images, labels, ids = images
    images = list(images)[:n]
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * cell_size * 0.95),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, img in enumerate(images):
        shown = to_displayable(img)
        axes.flat[i].imshow(shown, cmap="gray" if shown.ndim == 2 else None, vmin=0, vmax=1)
        caption = []
        if labels is not None:
            lab = labels[i]
            caption.append(str(class_names[int(lab)]) if class_names is not None else str(lab))
        if ids is not None:
            caption.append(str(ids[i]))
        if caption:
            axes.flat[i].set_title("\n".join(caption), fontsize=7.5, loc="center",
                                   color=INK["primary"])
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


# --- Class balance ----------------------------------------------------------------
def plot_class_balance(labels, ax: Optional[plt.Axes] = None, order: Optional[Sequence[str]] = None,
                       log: bool = False, unit: str = "images", title: Optional[str] = None) -> plt.Axes:
    """Bar chart of the number of items per class, annotated with count and percentage.

    ``labels`` is any sequence of class labels (one entry per image or per lesion).
    The title reports the imbalance ratio (largest class / smallest class).
    """
    counts = pd.Series(labels).value_counts()
    counts = counts.reindex(order, fill_value=0) if order is not None else counts
    if ax is None:
        _, ax = plt.subplots(figsize=(max(3.0, 0.55 * len(counts) + 1.5), 2.8))
    x = np.arange(len(counts))
    ax.bar(x, counts.to_numpy(), width=0.62, color=SINGLE_COLOR)
    total = counts.sum()
    for xi, value in zip(x, counts.to_numpy()):
        ax.annotate(f"{value:,}\n{100 * value / total:.1f}%", (xi, value), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7,
                    color=INK["secondary"])
    ax.set_xticks(x, counts.index.astype(str), rotation=0 if len(counts) <= 4 else 45,
                  ha="center" if len(counts) <= 4 else "right")
    ax.set_ylabel(f"# {unit}")
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(top=counts.max() * 8)
    else:
        ax.set_ylim(top=counts.max() * 1.25)
    ratio = counts.max() / counts[counts > 0].min()
    ax.set_title(f"{title or 'Class balance'}  (imbalance {ratio:.0f}:1)")
    return ax


# --- Pipeline debugging -----------------------------------------------------------
def plot_batch_summary(images: np.ndarray, raw_images: Optional[Sequence[np.ndarray]] = None,
                       channel_names: Optional[Sequence[str]] = None, n_examples: int = 4,
                       bins: int = 60) -> plt.Figure:
    """Debug view of a processed batch (N, H, W, C).

    Top row: the distribution of pixel values per channel, with min / max / mean / std,
    to confirm that normalisation did what it should (e.g. [0, 1] or mean 0 / std 1).
    Bottom rows (if ``raw_images`` is given): a few raw images above their processed version.
    """
    images = np.asarray(images)
    n_channels = images.shape[-1]
    names = channel_names or (["R", "G", "B"] if n_channels == 3 else [f"ch{c}" for c in range(n_channels)])
    show_examples = raw_images is not None
    fig = plt.figure(figsize=(9, 5.4 if show_examples else 2.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.9]) if show_examples else (fig, None)

    hist_axes = np.atleast_1d(top.subplots(1, n_channels))
    for c, ax in enumerate(hist_axes):
        values = images[..., c].ravel()
        ax.hist(values, bins=bins, color=SINGLE_COLOR, histtype="stepfilled", alpha=0.85)
        ax.set_title(f"channel {names[c]}")
        ax.set_yticks([])
        ax.text(0.98, 0.95, f"min {values.min():.2f}  max {values.max():.2f}\n"
                            f"mean {values.mean():.2f}  std {values.std():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK["secondary"])
    top.suptitle(f"Pixel values of the processed batch ({len(images)} images, shape {images.shape[1:]})",
                 x=0.01, ha="left", fontsize=9.5)

    if show_examples:
        k = min(n_examples, len(images), len(raw_images))
        grid = bottom.subplots(2, k, squeeze=False)
        for i in range(k):
            grid[0, i].imshow(to_displayable(raw_images[i]))
            processed = to_displayable(images[i])
            grid[1, i].imshow(processed, cmap="gray" if processed.ndim == 2 else None, vmin=0, vmax=1)
            for row in (0, 1):
                grid[row, i].axis("off")
        grid[0, 0].set_title("raw", fontsize=8, loc="left")
        grid[1, 0].set_title("processed (values outside [0, 1] are rescaled for display only)",
                             fontsize=8, loc="left")
    return fig
