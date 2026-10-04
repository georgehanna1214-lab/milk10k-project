"""Part 1 - Which metadata fields are related to the diagnosis label?

Unit of analysis (important):
    * Fields that are identical for both images of a lesion (age, sex, site, ...)
      are analysed with ONE ROW PER LESION. Using both images would count every
      lesion twice, double the sample size and make the tests look more
      significant than they are (the two rows are not independent).
    * Fields that can differ between the two images of a lesion
      (image_type, image_manipulation) are analysed per image.

Statistics:
    * categorical field vs. categorical label -> chi-square test of independence,
      with Cramer's V as effect size (0 = no association, 1 = perfect).
    * numeric field vs. categorical label -> Kruskal-Wallis test (rank-based
      one-way ANOVA), plus the classic one-way ANOVA for reference.
    With 5,240 lesions almost any difference is "significant", so the effect
    size - not the p-value - is what ranks the fields.
"""
import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy import stats

from . import config
from .visualize import CLASS_COLORS, CLASS_ORDER, INK, MIXED_COLOR

MISSING = "(missing)"
ID_COLUMNS = ("isic_id", "lesion_id")


# --- 1. Column audit ------------------------------------------------------------------
def infer_kind(series: pd.Series) -> str:
    """Rough type of a column: constant, boolean, numeric, categorical or free text / id."""
    values = series.dropna()
    if len(values) and set(values.unique()) <= {True, False}:
        return "boolean"          # also covers 'True or missing' columns such as melanocytic
    if values.nunique() <= 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if values.nunique() > 50:
        return "free text / id"
    return "categorical"


def is_lesion_level(df: pd.DataFrame, column: str) -> bool:
    """True if the column has the same value for every image of a lesion."""
    return bool((df.groupby("lesion_id")[column].nunique(dropna=False) <= 1).all())


def column_audit(df: pd.DataFrame, exclude: Sequence[str] = ID_COLUMNS + (config.TARGET_COL, config.CLASS_COL)) -> pd.DataFrame:
    """One row per metadata column: type, number of values, missingness and level."""
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        rows.append({
            "column": col,
            "kind": infer_kind(s),
            "n_unique": int(s.nunique()),
            "missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 1),
            "level": "lesion" if is_lesion_level(df, col) else "image",
            "example_values": "; ".join(map(str, s.dropna().unique()[:3])),
        })
    return pd.DataFrame(rows)


# --- 2. Class distribution inside each category -----------------------------------------
def as_category(series: pd.Series) -> pd.Series:
    """Values as text, with missing values kept as their own category '(missing)'."""
    return series.astype("object").where(series.notna(), MISSING).astype(str)


def analysis_rows(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """One row per lesion for lesion-level fields, one row per image otherwise."""
    return df.drop_duplicates("lesion_id") if is_lesion_level(df, field) else df


def target_distribution(df: pd.DataFrame, field: str, target: str = config.TARGET_COL) -> pd.DataFrame:
    """Percentage of each label inside each category of ``field`` (rows sum to 100) + count n."""
    rows = analysis_rows(df, field)
    counts = pd.crosstab(as_category(rows[field]), rows[target])
    table = counts.div(counts.sum(axis=1), axis=0) * 100
    table["n"] = counts.sum(axis=1)
    return table


# --- 3. Association statistics -------------------------------------------------------------
def cramers_v(chi2: float, n: int, shape: tuple) -> float:
    """Cramer's V = sqrt(chi2 / (n * (min(rows, cols) - 1)))."""
    return float(np.sqrt(chi2 / (n * (min(shape) - 1))))


def chi_square_association(df: pd.DataFrame, field: str, target: str) -> dict:
    """Chi-square test of independence between a categorical field and the label."""
    rows = analysis_rows(df, field)
    table = pd.crosstab(as_category(rows[field]), rows[target])
    if table.shape[0] < 2:                     # constant column: nothing to test
        return {"test": "none (constant)", "statistic": np.nan, "p_value": np.nan,
                "effect": 0.0, "n": int(table.to_numpy().sum()), "min_expected": np.nan}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    return {"test": "chi-square", "statistic": float(chi2), "dof": int(dof), "p_value": float(p),
            "effect": cramers_v(chi2, n, table.shape), "n": n, "min_expected": float(expected.min())}


def numeric_association(df: pd.DataFrame, field: str, target: str) -> dict:
    """Kruskal-Wallis (main) and one-way ANOVA (reference) for a numeric field vs. the label.

    Effect sizes: eta-squared from ANOVA (share of variance explained by the label)
    and eta-squared based on H for Kruskal-Wallis: (H - k + 1) / (n - k).
    """
    rows = analysis_rows(df, field).dropna(subset=[field])
    groups = [g[field].to_numpy() for _, g in rows.groupby(target)]
    k, n = len(groups), len(rows)
    h, p_kw = stats.kruskal(*groups)
    f, p_anova = stats.f_oneway(*groups)
    grand_mean = rows[field].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((rows[field] - grand_mean) ** 2).sum()
    return {"test": "Kruskal-Wallis", "statistic": float(h), "p_value": float(p_kw),
            "effect": float((h - k + 1) / (n - k)), "n": n,
            "anova_F": float(f), "anova_p": float(p_anova), "anova_eta2": float(ss_between / ss_total)}


def strength(cramers: float) -> str:
    """Verbal label for Cramer's V (rule of thumb)."""
    if cramers < 0.10:
        return "negligible"
    if cramers < 0.30:
        return "weak"
    if cramers < 0.50:
        return "moderate"
    return "strong"


def association_table(df: pd.DataFrame, categorical: Sequence[str], numeric: Sequence[str],
                      targets: Sequence[str] = (config.TARGET_COL, config.CLASS_COL)) -> pd.DataFrame:
    """Test every field against each label and collect the results in one table.

    For numeric fields a Cramer's V on the binned values (age is already in 5-year
    bins) is added so that every field can be compared on the same scale.
    """
    rows = []
    for field in list(categorical) + list(numeric):
        row = {"field": field, "kind": infer_kind(df[field]),
               "unit": "lesion" if is_lesion_level(df, field) else "image"}
        for target in targets:
            res = (numeric_association(df, field, target) if field in numeric
                   else chi_square_association(df, field, target))
            tag = "3cls" if target == config.TARGET_COL else "11cls"
            row.update({"test": res["test"], "n": res["n"],
                        f"statistic_{tag}": res["statistic"], f"p_{tag}": res["p_value"],
                        f"effect_{tag}": res["effect"]})
            if field in numeric:   # same-scale comparison: treat the bins as categories
                row[f"cramers_v_{tag}"] = chi_square_association(df, field, target)["effect"]
                row[f"anova_eta2_{tag}"] = res["anova_eta2"]
            else:
                row[f"cramers_v_{tag}"] = res["effect"]
                row[f"min_expected_{tag}"] = res["min_expected"]
        row["strength_3cls"] = strength(row["cramers_v_3cls"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cramers_v_3cls", ascending=False, ignore_index=True)


# --- 4. Plots ---------------------------------------------------------------------------------
def plot_target_by_fields(df: pd.DataFrame, fields: Sequence[str], target: str = config.TARGET_COL,
                          ncols: int = 3) -> plt.Figure:
    """Small multiples of 100% stacked bars: label mix inside every category of each field.

    The top bar of each panel ('all') is the overall mix, as a reference: a field is
    informative when its categories look different from that reference bar.
    """
    classes = [c for c in CLASS_ORDER if c in set(df[target])]
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.0 * nrows + 0.5), squeeze=False)
    for ax in axes.flat[len(fields):]:
        ax.set_visible(False)
    for ax, field in zip(axes.flat, fields):
        rows = analysis_rows(df, field)
        overall = rows[target].value_counts(normalize=True).mul(100)
        table = target_distribution(df, field, target)
        pct = pd.concat([overall.to_frame().T, table[classes]]).fillna(0)
        n = [len(rows)] + table["n"].tolist()
        names = ["all"] + [textwrap.shorten(str(i), 26, placeholder="…") for i in table.index]
        y = np.arange(len(pct))
        left = np.zeros(len(pct))
        for cls in classes:
            ax.barh(y, pct[cls].to_numpy(), left=left, height=0.64, color=CLASS_COLORS[cls],
                    edgecolor="white", linewidth=1.0)
            left += pct[cls].to_numpy()
        ax.set_yticks(y, [f"{name}  ({count:,})" for name, count in zip(names, n)], fontsize=7.5)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        unit = "lesions" if is_lesion_level(df, field) else "images"
        ax.set_title(f"{field}  ·  per {unit[:-1]}", fontsize=8.5)
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in classes]
    fig.legend(handles=handles, loc="upper center", ncol=len(classes), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_numeric_by_class(df: pd.DataFrame, field: str = "age_approx", target: str = config.TARGET_COL,
                          class_col: str = config.CLASS_COL) -> plt.Figure:
    """Left: distribution of a numeric field per label (overlaid). Right: boxplot per 11-class.

    Boxes are coloured by the diagnosis_1 group of the class (grey = class that mixes groups).
    """
    lesions = df.drop_duplicates("lesion_id").dropna(subset=[field])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.7]})

    step = 5   # age_approx is given in 5-year steps: one histogram bin per step
    bins = np.arange(lesions[field].min() - step / 2, lesions[field].max() + step, step)
    for cls in [c for c in CLASS_ORDER if c in set(lesions[target])]:
        values = lesions.loc[lesions[target] == cls, field]
        ax1.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8,
                 color=CLASS_COLORS[cls], label=f"{cls} (n={len(values):,})")
    ax1.set_xlabel(field)
    ax1.set_ylabel("density")
    ax1.legend(loc="upper left")
    ax1.set_title(f"{field} by {target}")

    order = lesions.groupby(class_col)[field].median().sort_values().index.tolist()
    group = lesions.groupby(class_col)[target].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    data = [lesions.loc[lesions[class_col] == c, field].to_numpy() for c in order]
    box = ax2.boxplot(data, patch_artist=True, widths=0.55,
                      medianprops={"color": "white", "linewidth": 1.6},
                      whiskerprops={"color": INK["muted"]}, capprops={"color": INK["muted"]},
                      flierprops={"marker": "o", "markersize": 2, "markerfacecolor": INK["muted"],
                                  "markeredgecolor": "none", "alpha": 0.6})
    for patch, cls in zip(box["boxes"], order):
        patch.set_facecolor(CLASS_COLORS.get(group[cls], MIXED_COLOR))
        patch.set_edgecolor("white")
    counts = lesions[class_col].value_counts()
    ax2.set_xticks(range(1, len(order) + 1), [f"{c}\n({counts[c]})" for c in order], fontsize=7.5)
    ax2.set_ylabel(field)
    ax2.grid(axis="x", visible=False)
    ax2.set_title(f"{field} by 11-class diagnosis (ordered by median)")
    legend = [Patch(color=CLASS_COLORS[c], label=c) for c in ("Benign", "Malignant")]
    legend.append(Patch(color=MIXED_COLOR, label="mixed (AKIEC: Malignant + Indeterminate)"))
    ax2.legend(handles=legend, title="box colour = diagnosis_1", title_fontsize=7, fontsize=7,
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    return fig
