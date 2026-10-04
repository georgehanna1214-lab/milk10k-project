# MILK10k skin-lesion classification

Course project for **Computer Vision & Speech Recognition (EADA)**, by George Hanna.

## 1. What this project is

- **Task:** predict `diagnosis_1` (**Benign / Malignant / Indeterminate**) for a skin lesion from its two images, one
  dermoscopic and one clinical close-up. The stretch goal is the 11-class diagnosis.
- **Dataset:** [MILK10k](https://doi.org/10.34970/648456) from the ISIC Archive.
  - 10,480 JPEG images, all 600×450 px, of 5,240 lesions (exactly 2 images per lesion).
  - `metadata.csv` has one row per image; `supplements/training_gt.csv` has the one-hot 11-class label per lesion.
  - License **CC BY-NC 4.0**, attribution **"MILK study team"**. The images are not redistributed here.
- **Goal of Milestone 1:** a leak-free, reproducible data pipeline. It checks every image, sets the label strategy,
  produces lesion-level train/val/test splits, normalisation statistics from train only, augmentation, and PyTorch
  Datasets/DataLoaders with imbalance handling. It grows out of the Session 2 code.

Reports: [Milestone 1 report](reports/milestone1/Milestone1_Report_George_Hanna.pdf) ·
[EDA notes](reports/milestone1/eda_notes.md) ·
[data-quality report](outputs/milestone1/data_quality_report.md) ·
[Homework 1 report](reports/session2/Homework1_Report_George_Hanna.pdf) ·
Part A: [notebook](homework_part_a.ipynb), [PDF](homework_part_a.pdf)

## 2. Setup and how to run

Requires Python 3.10 or newer (tested with 3.11 on macOS).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, pandas, torch, ... plus this repo's package `milk10k` (editable)
```

**Data.** Download and unzip MILK10k. The folder must contain `metadata.csv`, `supplements/` and `images/*.jpg`.
Point the code to it in **one** of two ways:

```bash
ln -s /path/to/milk10k data               # a link called data/ at the repo root (the default location)
export MILK10K_DIR=/path/to/milk10k       # or an environment variable (Windows: set MILK10K_DIR=C:\path\to\milk10k)
```

Only [`milk10k/config.py`](milk10k/config.py) reads this setting. The code contains no absolute paths.

**Reproduce everything:**

| Command | What it produces | Time |
|---|---|---|
| `python scripts/run_milestone1.py` | integrity check, EDA figures, label map, quality report, **splits**, normalisation stats, class weights, augmentation and batch figures | ~3 min |
| `pytest` | 29 tests: preprocessing, loader, split properties (10 seeds), deterministic eval transform, datasets | ~40 s |
| `jupyter nbconvert --to notebook --execute --inplace homework_part_a.ipynb` | Part A, run top to bottom | ~1 min |
| `python scripts/run_session2.py` | Homework 1 (Session 2) figures and tables | ~20 s |

## 3. Repository structure

```
milk10k-project/
├── README.md                     this file
├── SUBMISSION.md                 links to every deliverable
├── requirements.txt              packages (+ "-e ." to install the milk10k package)
├── pyproject.toml                makes milk10k/ installable; pytest settings
├── .gitignore                    keeps the dataset, the virtual environment and caches out of git
├── homework_part_a.ipynb / .pdf  Part A exercises (A1-A3) with written answers
├── milk10k/                      REUSABLE CODE: the importable package, one module per job
│   ├── config.py                 the one place for paths, seed (42), image size (224), split sizes
│   ├── data.py                   load the CSVs, add the 11-class label, link rows to image files
│   ├── labels.py                 lesion table (pivot), class -> diagnosis_1 mapping, label map
│   ├── splits.py                 split_lesions (StratifiedGroupKFold at lesion level) + checks
│   ├── quality.py                missing files, Image.verify of every image, data-quality report
│   ├── preprocessing.py          Session 2: preprocess_image / preprocess_batch (numpy)
│   ├── loader.py                 Session 2 MilkLoader, now failing loudly on missing files
│   ├── transforms.py             train_transform / eval_transform, train-only normalisation stats
│   ├── dataset.py                MilkDataset (per image), LesionDataset (per lesion), loaders, sampler, class weights
│   ├── metadata_analysis.py      Session 2 Part 1: metadata vs. diagnosis statistics and plots
│   ├── color_analysis.py         Session 2 Part 2: colour and histogram statistics
│   └── visualize.py              image grids (arrays or tensors), class balance, batch summary, chart style
├── scripts/
│   ├── run_milestone1.py         Milestone 1 end to end (B1-B8), writes all outputs
│   └── run_session2.py           Homework 1 end to end
├── tests/                        pytest: preprocessing, loader, splits, transforms, datasets
├── splits/                       GENERATED, committed: train.csv / val.csv / test.csv (one row per image)
├── artifacts/                    GENERATED, committed: label_map.json, norm_stats.json, class_weights.json
├── figures/                      GENERATED: milestone1/ (EDA, gallery, augmentation, batch), session2/
├── outputs/                      GENERATED: tables, data-quality report, run logs, Part A lesion table
├── reports/                      Milestone 1 report + EDA notes, Homework 1 report
└── data -> /path/to/milk10k      NOT COMMITTED: link to the dataset (or use MILK10K_DIR)
```

**Why this layout.** Logic used more than once lives in the `milk10k/` package, so scripts, the notebook and later
milestones import it instead of copying it. The Session 2 modules were extended, not rewritten (the loader now
fails loudly, the visualizer draws tensors). Scripts and the notebook only call the package. Everything they produce
goes to `figures/`, `outputs/`, `splits/` or `artifacts/`, never next to the source, so it can be regenerated at any
time. Splits and artifacts are committed because later milestones must load exactly the same files. `tests/` guards
the pieces every later step depends on, and `config.py` is the single switch for paths and settings.

## 4. Data handling rules

- **Never committed:** the raw images and CSVs (`data/` is git-ignored: 364 MB, licensed separately), the virtual
  environment and caches.
- **Committed:** code, tests, the split CSVs (small, needed for reproducibility), JSON artifacts, figures, tables and
  reports.
- **Where generated files go:** `figures/`, `outputs/`, `splits/`, `artifacts/`, `reports/`.
- **Seed:** 42 (`config.SEED`) for splits, sampling, shuffling and the weighted sampler.
- **Splits created:** 2026-10-04 with seed 42 by `scripts/run_milestone1.py`. They are lesion-level, 70 / 15 / 15
  (3,667 / 787 / 786 lesions; 7,334 / 1,574 / 1,572 images).

## 5. Key decisions and results so far

More detail is in the [Milestone 1 report](reports/milestone1/Milestone1_Report_George_Hanna.pdf).

- **Data integrity:** all 10,480 rows have their image, `Image.verify()` finds 0 unreadable files, and every image is
  600×450.
- **Label strategy** ([artifacts/label_map.json](artifacts/label_map.json)):
  - `diagnosis_1` keeps **Indeterminate as a 3rd class**: 123 lesions, all actinic keratoses, pre-cancerous and
    treated differently.
  - A binary "needs action" view (Malignant or Indeterminate vs. Benign) is stored for clinical reporting.
  - 11-class stretch goal: DF, INF, VASC and BEN_OTH are kept with class weights; MAL_OTH (9 lesions) is merged with
    BEN_OTH into **OTHER** (refer), giving 10 classes.
- **Splits:** StratifiedGroupKFold at lesion level, stratified on the 11 classes. There is zero lesion overlap, and
  class proportions deviate from the global ones by at most 0.21 pp (11 classes) and 0.57 pp (`diagnosis_1`).
- **Columns never used as inputs** (they leak the label): `diagnosis_2/3/4`, `melanocytic`, `diagnosis_confirm_type`,
  `concomitant_biopsy`. `image_manipulation` is not an input either (possible acquisition shortcut).
- **Preprocessing:** 224×224 (a 2× downscale for 100% of the images). Evaluation: resize the shorter side to 224, then
  centre crop. Normalisation uses the train-only mean [0.685, 0.527, 0.477] and std [0.122, 0.131, 0.152].
- **Views:** each image is a training sample with its lesion's label, and evaluation averages the two views per
  lesion.
- **Imbalance (train):** 27.6 : 1 (Malignant 5,076 vs. Indeterminate 184 images). Two corrections are implemented,
  to be used one at a time:
  - class-weighted loss: Benign 1.18, Indeterminate 13.29, Malignant 0.48 ([class_weights.json](artifacts/class_weights.json));
  - a WeightedRandomSampler: about 1/3 of each class per batch.

**Augmentation (train only)**, in [`milk10k/transforms.py`](milk10k/transforms.py):

| Augmentation | Parameters | Why it does not change the label |
|---|---|---|
| RandomResizedCrop | 224 px, scale 0.75-1.0, ratio 3/4-4/3 | keeps at least 75% of the image, so the (centred) lesion stays in view; mimics distance / zoom |
| RandomHorizontalFlip | p = 0.5 | skin lesions have no left/right; mirror images occur naturally |
| RandomVerticalFlip | p = 0.5 | no canonical "up" for a lesion on the skin |
| RandomRotate90 | 0 / 90 / 180 / 270° | the dermoscope can be held in any orientation; no interpolation, no black corners |
| ColorJitter | brightness 0.1, contrast 0.1, saturation 0.05, **hue 0** | within normal lighting variation; colour is diagnostic, and A3.5 showed any hue jitter shifts colour more than the Benign/Malignant gap |
