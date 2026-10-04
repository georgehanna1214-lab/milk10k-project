"""Part 3 - turning image files into clean, ready-to-use arrays.

Public interface:
    preprocess_image(source, ...) -> (array, info)   one image (file path, array or PIL image)
    preprocess_batch(items, ...)  -> BatchResult      many images; a bad file is skipped, not fatal

Pipeline: load as RGB -> resize -> colour space -> normalise -> float32 array (H, W, C).

Normalisation (default "scale"): pixel / 255, i.e. min-max scaling with the FIXED
8-bit range [0, 255] -> [0, 1]. Why this default:
    * it needs no statistics fitted on the data, so it cannot leak information
      from a validation/test set into the training pipeline;
    * every image is scaled the same way, so real brightness and colour
      differences between images (pigmentation, redness) are kept;
    * it is the same range torchvision's ToTensor() produces, which the training
      pipeline will use later.
Other options: "minmax" (per-image min/max: stretches each image's contrast, which
erases brightness differences between images), "zscore" ((x - mean) / std per
channel; pass the mean/std computed on the TRAINING split, otherwise the image's own
statistics are used), "none" (raw 0-255 values as float32).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .data import image_path

GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)   # ITU-R BT.601 luma
COLOR_MODES = ("rgb", "gray", "hsv")
NORMALIZATIONS = ("scale", "minmax", "zscore", "none")

ImageSource = Union[str, Path, np.ndarray, Image.Image]


def check_options(color_mode: str = "rgb", normalize: str = "scale", size: Optional[int] = 1) -> None:
    """Raise a clear error for an unknown option (a typo should stop the run immediately)."""
    if color_mode not in COLOR_MODES:
        raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")
    if size is not None and size < 1:
        raise ValueError(f"size must be a positive integer or None, got {size!r}")


def load_image(source: ImageSource) -> Image.Image:
    """Return an RGB PIL image from a file path, a uint8 array (H, W[, 3]) or a PIL image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise ValueError(f"image arrays must be uint8 (0-255), got {source.dtype}")
        return Image.fromarray(source).convert("RGB")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    with Image.open(path) as f:          # raises an OSError if the file is not a readable image
        return f.convert("RGB")          # convert() reads all pixels before the file is closed


def resize_image(img: Image.Image, size: int, keep_aspect: bool = True) -> Image.Image:
    """Resize to ``size`` x ``size`` pixels.

    keep_aspect=True : scale so the SHORTER side equals ``size``, then cut the centre
                       square. Shapes are not distorted (asymmetry and border shape are
                       diagnostic); lesions are usually centred, and the trimmed edges are
                       mostly background (and dark dermoscope corners).
    keep_aspect=False: stretch the whole image to a square (nothing cut, shapes distorted).
    """
    if not keep_aspect:
        return img.resize((size, size), Image.Resampling.BILINEAR)
    width, height = img.size
    scale = size / min(width, height)
    new_w, new_h = max(size, round(width * scale)), max(size, round(height * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def convert_color(img: Image.Image, color_mode: str) -> np.ndarray:
    """RGB PIL image -> float32 array (H, W, C) with values 0-255 in the requested colour space."""
    rgb = np.asarray(img, dtype=np.float32)
    if color_mode == "rgb":
        return rgb
    if color_mode == "gray":
        return (rgb @ GRAY_WEIGHTS)[..., np.newaxis]          # keep a channel axis: (H, W, 1)
    if color_mode == "hsv":
        return np.asarray(img.convert("HSV"), dtype=np.float32)
    raise ValueError(f"unknown color_mode {color_mode!r}")


def normalize_array(arr: np.ndarray, method: str, mean: Optional[Sequence[float]] = None,
                    std: Optional[Sequence[float]] = None):
    """Normalise a 0-255 float array. Returns (array, dict describing what was done)."""
    if method == "none":
        return arr, {}
    if method == "scale":
        return arr / 255.0, {"divided_by": 255}
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        out = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        return out, {"image_min": lo, "image_max": hi}
    if method == "zscore":
        x = arr / 255.0
        if mean is None or std is None:
            mean, std, origin = x.mean(axis=(0, 1)), x.std(axis=(0, 1)), "this image"
        else:
            origin = "given (should come from the training split)"
        mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        out = (x - mean) / np.maximum(std, 1e-6)
        return out, {"mean": mean.ravel().round(4).tolist(), "std": std.ravel().round(4).tolist(),
                     "stats_from": origin}
    raise ValueError(f"unknown normalize {method!r}")


def preprocess_image(source: ImageSource, size: Optional[int] = config.IMAGE_SIZE,
                     color_mode: str = "rgb", normalize: str = "scale",
                     mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None,
                     keep_aspect: bool = True):
    """Load one image and apply resize -> colour conversion -> normalisation.

    Args:
        source: file path, uint8 array or PIL image.
        size: output is ``size`` x ``size`` pixels (None = keep the original size).
        color_mode: "rgb" (3 channels), "gray" (1 channel) or "hsv" (3 channels).
        normalize: "scale" (default, /255), "minmax", "zscore" or "none" - see module docstring.
        mean, std: per-channel statistics for "zscore" (values on the 0-1 scale).
        keep_aspect: centre-crop (True) or stretch (False) when resizing.

    Returns:
        (array, info): float32 array of shape (H, W, C) and a dict saying what was done
        (original size, final shape, options, value range).
    """
    check_options(color_mode, normalize, size)
    img = load_image(source)
    original_size = img.size                                  # (width, height)
    if size is not None:
        img = resize_image(img, size, keep_aspect)
    arr = convert_color(img, color_mode)
    arr, norm_details = normalize_array(arr, normalize, mean, std)
    arr = arr.astype(np.float32)
    info = {
        "source": str(source) if isinstance(source, (str, Path)) else type(source).__name__,
        "original_size_wh": original_size,
        "resize": "none" if size is None else
                  f"{'shorter side + centre crop' if keep_aspect else 'stretch'} to {size}x{size}",
        "color_mode": color_mode,
        "normalize": normalize,
        **norm_details,
        "final_shape": arr.shape,
        "value_range": (round(float(arr.min()), 4), round(float(arr.max()), 4)),
        "dtype": str(arr.dtype),
    }
    return arr, info


@dataclass
class BatchResult:
    """Output of ``preprocess_batch``."""
    images: Union[np.ndarray, list]    # (N, H, W, C) array, or a list if the sizes differ
    ids: list                          # identifier of each processed image, same order as images
    infos: list                        # the info dict of each processed image
    skipped: list = field(default_factory=list)   # (identifier, reason) of each failed image

    def summary(self) -> str:
        """One-line report, plus the reason for every skipped image."""
        lines = [f"processed {len(self.ids)} image(s), skipped {len(self.skipped)}"]
        lines += [f"  skipped {ident}: {reason}" for ident, reason in self.skipped]
        return "\n".join(lines)


def _identifier(item, position: int) -> str:
    """Name used in reports: the file name without extension, or the position in the list."""
    return Path(item).stem if isinstance(item, (str, Path)) else f"item_{position}"


def preprocess_batch(items: Union[pd.DataFrame, Sequence[ImageSource]],
                     img_dir: Optional[Path] = None, skip_errors: bool = True, **options) -> BatchResult:
    """Apply ``preprocess_image`` to many images.

    Args:
        items: a DataFrame with an ``isic_id`` column (rows of the metadata table), or a
            list of file paths / arrays.
        img_dir: image folder for DataFrame input (default: config.IMG_DIR).
        skip_errors: True (default) = a missing, truncated or unreadable file is skipped
            and reported in ``result.skipped`` as (identifier, reason), so one bad image
            never stops the batch; False = the first bad file raises (fail loudly).
        **options: passed to ``preprocess_image`` (size, color_mode, normalize, ...).

    Invalid options are never skipped: they raise immediately, because they would
    make every image fail.
    """
    check_options(options.get("color_mode", "rgb"), options.get("normalize", "scale"),
                  options.get("size", 1))
    if isinstance(items, pd.DataFrame):
        jobs = [(iid, image_path(iid, img_dir)) for iid in items["isic_id"]]
    else:
        jobs = [(_identifier(item, i), item) for i, item in enumerate(items)]

    images, ids, infos, skipped = [], [], [], []
    for ident, source in jobs:
        try:
            arr, info = preprocess_image(source, **options)
        except (OSError, ValueError) as err:     # missing file, corrupt/unreadable image, bad array
            if not skip_errors:
                raise
            skipped.append((ident, f"{type(err).__name__}: {err}"))
            continue
        images.append(arr)
        ids.append(ident)
        infos.append(info)

    if images and len({a.shape for a in images}) == 1:
        images = np.stack(images)                # (N, H, W, C)
    return BatchResult(images=images, ids=ids, infos=infos, skipped=skipped)
