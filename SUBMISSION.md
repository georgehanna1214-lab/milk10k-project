# Submission: Homework Sessions 1-3 + Course Project Milestone 1

- **Name:** George Hanna
- **Repository:** REPO_URL

## Part A: notebook and PDF

- Notebook: [homework_part_a.ipynb](homework_part_a.ipynb)
- PDF export: [homework_part_a.pdf](homework_part_a.pdf)

## Part B: Milestone 1, data pipeline

| Item | Deliverable | File |
|---|---|---|
| B0 | README (task, setup, structure, data rules, decisions) | [README.md](README.md) |
| B0 | Requirements | [requirements.txt](requirements.txt) · [pyproject.toml](pyproject.toml) |
| B0 | Configuration (paths, seed, image size) | [milk10k/config.py](milk10k/config.py) |
| B1 | Integrity check (missing files, `Image.verify()` on every image) | [milk10k/quality.py](milk10k/quality.py) |
| B1 | Image size summary | [outputs/milestone1/image_size_summary.csv](outputs/milestone1/image_size_summary.csv) |
| B1 | Loader that fails loudly on missing files | [milk10k/loader.py](milk10k/loader.py) · [milk10k/dataset.py](milk10k/dataset.py) |
| B2 | Class distribution (3 classes + 11 classes on a log scale) | [figures/milestone1/class_distribution.png](figures/milestone1/class_distribution.png) |
| B2 | 11-class to `diagnosis_1` mapping | [outputs/milestone1/class_to_diagnosis1.csv](outputs/milestone1/class_to_diagnosis1.csv) |
| B2 | Session 2 findings re-checked, gallery, biopsy-enrichment analysis | [reports/milestone1/eda_notes.md](reports/milestone1/eda_notes.md) |
| B2 | 3×4 gallery, one image per class | [figures/milestone1/class_gallery.jpg](figures/milestone1/class_gallery.jpg) |
| B3 | Label strategy (code) and label map | [milk10k/labels.py](milk10k/labels.py) · [artifacts/label_map.json](artifacts/label_map.json) |
| B4 | Data-quality report | [outputs/milestone1/data_quality_report.md](outputs/milestone1/data_quality_report.md) |
| B5 | Splitting code (`split_lesions`) | [milk10k/splits.py](milk10k/splits.py) |
| B5 | Split files | [splits/train.csv](splits/train.csv) · [splits/val.csv](splits/val.csv) · [splits/test.csv](splits/test.csv) |
| B5 | Split verification (overlap, 2 images per lesion, proportions) | [outputs/milestone1/run_log.txt](outputs/milestone1/run_log.txt) · [split_proportions_dx.csv](outputs/milestone1/split_proportions_dx.csv) · [split_proportions_diagnosis_1.csv](outputs/milestone1/split_proportions_diagnosis_1.csv) |
| B6-B7 | Transforms (`train_transform`, `eval_transform`, train-only statistics) | [milk10k/transforms.py](milk10k/transforms.py) |
| B6 | Normalisation statistics (train only) | [artifacts/norm_stats.json](artifacts/norm_stats.json) |
| B7 | Augmentation figure (1 original + 7 augmented, 3 classes) | [figures/milestone1/augmentations.jpg](figures/milestone1/augmentations.jpg) |
| B8 | Dataset / DataLoaders, weighted sampler, class weights | [milk10k/dataset.py](milk10k/dataset.py) · [artifacts/class_weights.json](artifacts/class_weights.json) |
| B8 | Visual check of a batch after the transforms | [figures/milestone1/train_batch.jpg](figures/milestone1/train_batch.jpg) |
| B1-B8 | Pipeline script that reproduces everything | [scripts/run_milestone1.py](scripts/run_milestone1.py) |
| B1-B8 | Tests (splits over 10 seeds, deterministic eval transform, datasets, loader) | [tests/](tests/) |
| B9 | Milestone 1 report (2 pages) | [reports/milestone1/Milestone1_Report_George_Hanna.pdf](reports/milestone1/Milestone1_Report_George_Hanna.pdf) |

Session 2 code that Milestone 1 builds on: [milk10k/preprocessing.py](milk10k/preprocessing.py),
[milk10k/visualize.py](milk10k/visualize.py), [milk10k/metadata_analysis.py](milk10k/metadata_analysis.py),
[milk10k/color_analysis.py](milk10k/color_analysis.py), [scripts/run_session2.py](scripts/run_session2.py) and the
[Homework 1 report](reports/session2/Homework1_Report_George_Hanna.pdf).
