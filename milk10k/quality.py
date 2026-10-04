"""Data integrity and data-quality checks (Milestone 1, B1 and B4)."""
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from . import config
from .data import image_path

# Columns that must never be model inputs because they leak the label.
LEAKY_COLUMNS = {
    "diagnosis_2": "finer level of the label itself",
    "diagnosis_3": "finer level of the label itself",
    "diagnosis_4": "finer level of the label itself",
    "melanocytic": "True exactly for NV and MEL (derived from the diagnosis)",
    "diagnosis_confirm_type": "says whether the lesion was biopsied: decided because of the suspected diagnosis",
    "concomitant_biopsy": "identical to diagnosis_confirm_type",
}

# Handling decision for every column that has missing values (one line each).
MISSING_DECISIONS = {
    "age_approx": "impute with the TRAIN median when used as a feature (0.4% missing, 20 lesions)",
    "anatom_site_general": "recover: 3,850 of the 3,912 missing images are 'trunk' in supplements/training_input.csv "
                           "(the column has no trunk category); the other 62 -> explicit 'unknown'",
    "anatom_site_special": "drop: 98% missing, negligible association",
    "melanocytic": "drop: leaks the label (NaN means 'not melanocytic')",
    "diagnosis_3": "leave NaN: label hierarchy, never an input",
    "diagnosis_4": "leave NaN: label hierarchy, never an input (most lesions stop at level 2/3)",
}


def missing_files(meta: pd.DataFrame, img_dir: Optional[Path] = None) -> list:
    """isic_ids of the metadata rows whose .jpg file does not exist."""
    return [iid for iid in meta["isic_id"] if not image_path(iid, img_dir).is_file()]


def verify_images(meta: pd.DataFrame, img_dir: Optional[Path] = None) -> pd.DataFrame:
    """Run ``Image.verify()`` on every image; one row per image with size, mode and any error."""
    rows = []
    for iid in meta["isic_id"]:
        row = {"isic_id": iid, "readable": True, "error": "", "width": None, "height": None, "mode": None}
        try:
            with Image.open(image_path(iid, img_dir)) as im:
                row.update(width=im.size[0], height=im.size[1], mode=im.mode)
                im.verify()                  # checks the file structure without decoding all pixels
        except Exception as err:             # any failure = unreadable file, recorded, not fatal
            row.update(readable=False, error=f"{type(err).__name__}: {err}")
        rows.append(row)
    return pd.DataFrame(rows)


def size_summary(verified: pd.DataFrame) -> pd.DataFrame:
    """Min / median / max width and height, plus image and unreadable counts."""
    ok = verified[verified["readable"]]
    return pd.DataFrame({
        "statistic": ["n_images", "n_unreadable", "width_min", "width_median", "width_max",
                      "height_min", "height_median", "height_max", "pct_larger_than_224",
                      "pct_larger_than_256"],
        "value": [len(verified), int((~verified["readable"]).sum()),
                  ok["width"].min(), ok["width"].median(), ok["width"].max(),
                  ok["height"].min(), ok["height"].median(), ok["height"].max(),
                  round(100 * ((ok["width"] > 224) & (ok["height"] > 224)).mean(), 1),
                  round(100 * ((ok["width"] > 256) & (ok["height"] > 256)).mean(), 1)],
    })


def _markdown_table(df: pd.DataFrame) -> str:
    """Small DataFrame -> Markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def quality_report(meta: pd.DataFrame) -> str:
    """Build the Milestone 1 data-quality report (Markdown text)."""
    out = ["# Data-quality report (Milestone 1)", ""]

    out += ["## 1. Missing values: handling decision per column", ""]
    miss = meta.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    table = pd.DataFrame({"missing": miss, "pct": (100 * miss / len(meta)).round(1),
                          "decision": [MISSING_DECISIONS.get(c, "-") for c in miss.index]})
    table.index.name = "column"
    out += [_markdown_table(table), ""]

    out += ["## 2. Label-consistency checks", ""]
    per_lesion = meta.groupby("lesion_id")["isic_id"].count()
    types = meta.groupby("lesion_id")["image_type"].agg(lambda s: tuple(sorted(s)))
    expected = ("clinical: close-up", "dermoscopic")
    varying = [c for c in meta.columns if c not in ("isic_id", "image_type", "image_manipulation")
               and (meta.groupby("lesion_id")[c].nunique(dropna=False) > 1).any()]
    checks = [
        ("every lesion has exactly 2 images", bool((per_lesion == 2).all()), f"{int((per_lesion != 2).sum())} lesions violate"),
        ("one dermoscopic + one clinical image per lesion", bool((types == expected).all()), f"{int((types != expected).sum())} lesions violate"),
        ("diagnosis/age/sex/site identical for both images", not varying, f"columns that vary: {varying or 'none'}"),
    ]
    out += ["| check | result | detail |", "|---|---|---|"]
    out += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in checks]
    out += [""]

    out += ["## 3. Suspicious shortcuts (row % of diagnosis_1, per image)", ""]
    for col in ("image_manipulation", "image_type"):
        ct = pd.crosstab(meta[col], meta[config.TARGET_COL], normalize="index").mul(100).round(1)
        ct["n_images"] = meta[col].value_counts()
        ct.index.name = col
        out += [f"**{col}**", "", _markdown_table(ct), ""]
    out += ["Conclusion: `image_type` carries no label information (every lesion has one image of each type). "
            "`image_manipulation = altered` (335 images) is 12.8% Indeterminate vs 2.0% overall: it describes "
            "how the picture was taken or edited, can stand for a clinic or protocol, and is NOT used as an input. "
            "Both fields stay available for error analysis (does the model behave differently on altered images?).", ""]

    out += ["## 4. Columns never used as model inputs (they leak the label)", ""]
    out += [f"- `{col}`: {why}" for col, why in LEAKY_COLUMNS.items()]
    out += ["- also not inputs: `isic_id`, `lesion_id` (identifiers), `attribution`, `copyright_license` (constant)", ""]
    return "\n".join(out)
