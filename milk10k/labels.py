"""Labels: the lesion-level table, the 11-class <-> diagnosis_1 mapping, and the label map.

Label strategy (Milestone 1, see README / report):
    * Primary target = diagnosis_1 with THREE classes. "Indeterminate" (123 lesions, 2.3%,
      all actinic keratoses) is kept as its own class: dropping it would leave the model
      unable to handle actinic keratoses it will meet in practice, and merging it would
      hide a pre-cancerous group that is managed differently. Its rarity is handled with
      class weights / weighted sampling, and a secondary binary view ("needs action" =
      Malignant or Indeterminate vs. Benign) is stored for clinical reporting.
    * 11-class stretch goal: DF, INF, VASC and BEN_OTH (44-52 lesions each) are kept and
      handled with class weights; MAL_OTH (9 lesions: about 6 for training and 1-2 per
      evaluation split) cannot be learned or evaluated alone and is merged with BEN_OTH
      into OTHER ("unusual lesion -> refer"), giving 10 classes.
"""
import json
from pathlib import Path

import pandas as pd

from . import config
from .data import onehot_to_label

LABEL_MAP = {
    "target": config.TARGET_COL,
    config.TARGET_COL: {"Benign": 0, "Indeterminate": 1, "Malignant": 2},
    "needs_action": {"Benign": 0, "Indeterminate": 1, "Malignant": 1},
    "dx_merge": {"BEN_OTH": "OTHER", "MAL_OTH": "OTHER"},
    "dx10": {"AKIEC": 0, "BCC": 1, "BKL": 2, "DF": 3, "INF": 4, "MEL": 5, "NV": 6,
             "OTHER": 7, "SCCKA": 8, "VASC": 9},
    "notes": {
        "diagnosis_1": "primary target; Indeterminate kept as a 3rd class (all are actinic keratoses)",
        "needs_action": "secondary clinical view: Malignant or Indeterminate = needs treatment/biopsy",
        "dx10": "11-class stretch goal with BEN_OTH + MAL_OTH merged into OTHER (= refer)",
    },
}


def lesion_table(meta: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion: lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site.

    Built with ``pivot`` (no loop over rows): the two images of a lesion become two
    columns. Age, sex, site and diagnosis are identical for both images, so the first
    image of each lesion is used for them.
    """
    ids = (meta.pivot(index="lesion_id", columns="image_type", values="isic_id")
               .rename(columns={"dermoscopic": "derm_id", "clinical: close-up": "clinical_id"}))
    fields = (meta.drop_duplicates("lesion_id").set_index("lesion_id")
                  [[config.TARGET_COL, "age_approx", "sex", "anatom_site_general"]])
    dx = pd.Series(onehot_to_label(gt).to_numpy(), index=gt["lesion_id"], name=config.CLASS_COL)
    lesions = ids.join(fields).join(dx).reset_index()
    lesions.columns.name = None
    lesions = lesions.rename(columns={"age_approx": "age", "anatom_site_general": "site"})
    return lesions[["lesion_id", "derm_id", "clinical_id", config.TARGET_COL, config.CLASS_COL,
                    "age", "sex", "site"]]


def class_to_diagnosis1(lesions: pd.DataFrame) -> pd.DataFrame:
    """Lesion counts of every 11-class label against diagnosis_1 (+ how many diagnosis_1 values)."""
    table = pd.crosstab(lesions[config.CLASS_COL], lesions[config.TARGET_COL])
    table["n_diagnosis_1_values"] = (table > 0).sum(axis=1)
    return table


def add_dx10(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 10-class stretch label ``dx10`` (BEN_OTH and MAL_OTH merged into OTHER)."""
    out = df.copy()
    out["dx10"] = out[config.CLASS_COL].replace(LABEL_MAP["dx_merge"])
    return out


def save_label_map(path: Path = config.ARTIFACTS_DIR / "label_map.json") -> Path:
    """Write the label map to JSON (every later milestone loads it from there)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(LABEL_MAP, indent=2) + "\n")
    return path


def load_label_map(path: Path = config.ARTIFACTS_DIR / "label_map.json") -> dict:
    """Read the label map saved by ``save_label_map``."""
    return json.loads(Path(path).read_text())
