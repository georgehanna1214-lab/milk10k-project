"""Project configuration: the ONE place for paths, seed and image settings.

Where is the data?
    The dataset folder (the one that contains ``metadata.csv``, ``supplements/``
    and ``images/``) is read from the environment variable ``MILK10K_DIR``.
    If it is not set, the code uses ``data/`` at the repository root, which can be
    a symbolic link to wherever the dataset was downloaded:

        ln -s /path/to/milk10k data            # macOS / Linux
        export MILK10K_DIR=/path/to/milk10k    # or use the environment variable

    No absolute paths are written anywhere in the code.
"""
import os
from pathlib import Path

# Repository root = the folder that contains this package.
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Input data (never committed to git) ------------------------------------
DATA_DIR = Path(os.environ.get("MILK10K_DIR", REPO_ROOT / "data"))
IMG_DIR = Path(os.environ.get("MILK10K_IMAGES_DIR", DATA_DIR / "images"))
METADATA_PATH = DATA_DIR / "metadata.csv"                    # one row per IMAGE
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"       # one row per LESION

# --- Generated outputs (kept separate from the source code) ------------------
FIGURES_DIR = REPO_ROOT / "figures"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SPLITS_DIR = REPO_ROOT / "splits"            # train.csv / val.csv / test.csv (committed)
ARTIFACTS_DIR = REPO_ROOT / "artifacts"      # label_map.json, norm_stats.json, class_weights.json

# --- Shared settings ----------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
TARGET_COL = "diagnosis_1"   # Benign / Malignant / Indeterminate (project target)
CLASS_COL = "dx"             # 11-class diagnosis taken from training_gt.csv
VAL_SIZE = 0.15              # fraction of LESIONS in the validation split
TEST_SIZE = 0.15             # fraction of LESIONS in the test split
