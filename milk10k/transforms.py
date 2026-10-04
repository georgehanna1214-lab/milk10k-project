"""Image transforms for training and evaluation (Milestone 1, B6-B7).

    train_transform(mean, std)  random, label-safe augmentation (TRAIN split only)
    eval_transform(mean, std)   deterministic: same input -> same tensor (val / test)
    compute_norm_stats(dataset) per-channel mean / std from the TRAIN split only

Resolution: every MILK10k image is 600 x 450 px, so 224 x 224 (the standard input of
ImageNet-pretrained CNNs) is a 2x downscale for 100% of the images and no image is
upsampled. Evaluation resizes the shorter side to 224 and crops the centre square,
the same geometry as the Session 2 ``preprocess_image``.
"""
import json
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from . import config


class RandomRotate90:
    """Rotate by 0, 90, 180 or 270 degrees at random.

    Skin lesions have no natural 'up', and right-angle rotations of a square image need
    no interpolation and create no black corners (unlike rotations by small angles).
    """

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return TF.rotate(img, 90 * k) if k else img

    def __repr__(self) -> str:
        return "RandomRotate90()"


# Augmentations and their parameters (also listed with a medical justification in the README).
AUGMENT = {
    "random_resized_crop_scale": (0.75, 1.0),   # keep at least 75% of the image: the lesion stays in view
    "color_jitter": dict(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),   # mild, no hue shift
}


def train_transform(mean: Sequence[float], std: Sequence[float], size: int = config.IMAGE_SIZE):
    """Random but label-safe augmentation, used on the TRAIN split only."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=AUGMENT["random_resized_crop_scale"], ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        T.ColorJitter(**AUGMENT["color_jitter"]),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(mean: Sequence[float], std: Sequence[float], size: int = config.IMAGE_SIZE):
    """Deterministic transform for validation and test: resize, centre crop, normalise."""
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def stats_transform(size: int = config.IMAGE_SIZE):
    """Eval geometry without normalisation: used to measure the normalisation statistics."""
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])


def compute_norm_stats(loader) -> dict:
    """Per-channel mean and std over every pixel of every image the loader yields.

    Call it with a loader over the TRAIN split only: statistics that include
    validation/test images would leak information from them (preprocessing leakage).
    """
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for images, _, _ in loader:                      # images: (B, 3, H, W) in [0, 1]
        x = images.double()
        total += x.sum(dim=(0, 2, 3))
        total_sq += (x ** 2).sum(dim=(0, 2, 3))
        n_pixels += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / n_pixels
    std = (total_sq / n_pixels - mean ** 2).sqrt()
    return {"mean": [round(v, 4) for v in mean.tolist()], "std": [round(v, 4) for v in std.tolist()],
            "computed_on": "train split only", "n_pixels": n_pixels}


def save_norm_stats(stats: dict, path: Path = config.ARTIFACTS_DIR / "norm_stats.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n")
    return path


def load_norm_stats(path: Path = config.ARTIFACTS_DIR / "norm_stats.json") -> dict:
    return json.loads(Path(path).read_text())


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo ``Normalize`` (for display only): x * std + mean, clipped to [0, 1]."""
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    return (images * s + m).clamp(0, 1)
