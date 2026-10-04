# Data-quality report (Milestone 1)

## 1. Missing values: handling decision per column

| column | missing | pct | decision |
|---|---|---|---|
| anatom_site_special | 10274 | 98.0 | drop: 98% missing, negligible association |
| diagnosis_4 | 8958 | 85.5 | leave NaN: label hierarchy, never an input (most lesions stop at level 2/3) |
| melanocytic | 8088 | 77.2 | drop: leaks the label (NaN means 'not melanocytic') |
| anatom_site_general | 3912 | 37.3 | recover: 3,850 of the 3,912 missing images are 'trunk' in supplements/training_input.csv (the column has no trunk category); the other 62 -> explicit 'unknown' |
| diagnosis_3 | 158 | 1.5 | leave NaN: label hierarchy, never an input |
| age_approx | 40 | 0.4 | impute with the TRAIN median when used as a feature (0.4% missing, 20 lesions) |

## 2. Label-consistency checks

| check | result | detail |
|---|---|---|
| every lesion has exactly 2 images | PASS | 0 lesions violate |
| one dermoscopic + one clinical image per lesion | PASS | 0 lesions violate |
| diagnosis/age/sex/site identical for both images | PASS | columns that vary: none |

## 3. Suspicious shortcuts (row % of diagnosis_1, per image)

**image_manipulation**

| image_manipulation | Benign | Indeterminate | Malignant | n_images |
|---|---|---|---|---|
| altered | 39.7 | 12.8 | 47.5 | 335.0 |
| instrument only | 27.9 | 2.0 | 70.1 | 10145.0 |

**image_type**

| image_type | Benign | Indeterminate | Malignant | n_images |
|---|---|---|---|---|
| clinical: close-up | 28.3 | 2.3 | 69.4 | 5240.0 |
| dermoscopic | 28.3 | 2.3 | 69.4 | 5240.0 |

Conclusion: `image_type` carries no label information (every lesion has one image of each type). `image_manipulation = altered` (335 images) is 12.8% Indeterminate vs 2.0% overall: it describes how the picture was taken or edited, can stand for a clinic or protocol, and is NOT used as an input. Both fields stay available for error analysis (does the model behave differently on altered images?).

## 4. Columns never used as model inputs (they leak the label)

- `diagnosis_2`: finer level of the label itself
- `diagnosis_3`: finer level of the label itself
- `diagnosis_4`: finer level of the label itself
- `melanocytic`: True exactly for NV and MEL (derived from the diagnosis)
- `diagnosis_confirm_type`: says whether the lesion was biopsied: decided because of the suspected diagnosis
- `concomitant_biopsy`: identical to diagnosis_confirm_type
- also not inputs: `isic_id`, `lesion_id` (identifiers), `attribution`, `copyright_license` (constant)

