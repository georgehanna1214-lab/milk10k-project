"""Session 2 homework - run every part and save all figures and tables.

    python scripts/run_session2.py

Figures -> figures/session2/   Tables -> outputs/session2/   (about 30 seconds)
"""
import matplotlib.pyplot as plt
import pandas as pd

from milk10k import color_analysis as ca
from milk10k import config
from milk10k import metadata_analysis as ma
from milk10k.data import image_path, load_dataset
from milk10k.loader import MilkLoader
from milk10k.preprocessing import preprocess_batch, preprocess_image
from milk10k.visualize import apply_style, plot_batch_summary, plot_class_balance, save_figure, show_grid

FIG = config.FIGURES_DIR / "session2"
OUT = config.OUTPUTS_DIR / "session2"
pd.set_option("display.width", 200, "display.max_columns", 30)


def part1_metadata(df: pd.DataFrame) -> None:
    print("\n=== PART 1: metadata vs. diagnosis ===")
    audit = ma.column_audit(df)
    audit.to_csv(OUT / "column_audit.csv", index=False)
    print(audit.to_string(index=False))

    kinds = audit.set_index("column")["kind"]
    numeric = kinds[kinds == "numeric"].index.tolist()
    categorical = kinds[kinds != "numeric"].index.tolist()
    assoc = ma.association_table(df, categorical, numeric)
    assoc.to_csv(OUT / "metadata_association.csv", index=False)
    print("\n", assoc[["field", "unit", "test", "n", "statistic_3cls", "p_3cls", "cramers_v_3cls",
                       "cramers_v_11cls", "strength_3cls"]].round(4).to_string(index=False))
    age = assoc.set_index("field").loc["age_approx"]
    print(f"\nage: Kruskal-Wallis H={age.statistic_3cls:.1f} p={age.p_3cls:.2e} eta2_H={age.effect_3cls:.3f} "
          f"| ANOVA eta2={age.anova_eta2_3cls:.3f} | 11-class eta2_H={age.effect_11cls:.3f} "
          f"ANOVA eta2={age.anova_eta2_11cls:.3f}")

    for field in categorical:
        ma.target_distribution(df, field).round(1).to_csv(OUT / f"label_mix_by_{field}.csv")
    for field in ("anatom_site_general", "image_manipulation", "diagnosis_confirm_type", "sex"):
        print(f"\n{field}:\n", ma.target_distribution(df, field).round(1).to_string())

    fields = ["anatom_site_general", "sex", "image_manipulation", "diagnosis_confirm_type",
              "melanocytic", "image_type"]
    save_figure(ma.plot_target_by_fields(df, fields), FIG / "fig1_label_mix_by_field.png")
    save_figure(ma.plot_numeric_by_class(df, "age_approx"), FIG / "fig2_age_by_class.png")
    lesions = df.drop_duplicates("lesion_id")
    print("\nmedian age per 11-class:", lesions.groupby(config.CLASS_COL)["age_approx"].median().sort_values().to_dict())


def part2_color(df: pd.DataFrame) -> None:
    print("\n=== PART 2: colour across classes ===")
    # (a) 100 lesions per diagnosis_1 class -> 200 images per class (100 per image type)
    sample = ca.sample_per_class(df, config.TARGET_COL, n_per_class=100, seed=config.SEED)
    features, hists = ca.compute_color_profiles(sample)
    save_figure(ca.plot_mean_histograms(features, hists, config.TARGET_COL), FIG / "fig3_histograms_by_class.png")
    table = ca.summary_table(features, config.TARGET_COL)
    table.to_csv(OUT / "color_stats_by_diagnosis1.csv")
    print(table.to_string())
    auc = ca.separability_auc(features, config.TARGET_COL, positive="Malignant", negative="Benign")
    auc.to_csv(OUT / "color_auc_malignant_vs_benign.csv")
    print("\nROC-AUC of single colour features, Malignant vs Benign:\n", auc.to_string())

    # (b) 30 lesions per 11-class diagnosis (MAL_OTH has only 9) -> finer view
    sample11 = ca.sample_per_class(df, config.CLASS_COL, n_per_class=30, seed=config.SEED)
    features11, _ = ca.compute_color_profiles(sample11)
    lesions = df.drop_duplicates("lesion_id")
    group_of = lesions.groupby(config.CLASS_COL)[config.TARGET_COL].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else "mixed").to_dict()
    order = sorted(group_of, key=lambda c: (["Benign", "mixed", "Malignant"].index(group_of[c]), c))
    save_figure(ca.plot_stat_boxplots(features11, config.CLASS_COL, group_of, order=order),
                FIG / "fig4_color_stats_by_11class.png")
    table11 = ca.summary_table(features11, config.CLASS_COL)
    table11.to_csv(OUT / "color_stats_by_11class.csv")
    medians = features11.groupby(["image_type", config.CLASS_COL])[["brightness", "mean_R", "mean_G", "mean_B"]].median()
    print("\nmedian per 11-class:\n", medians.round(0).to_string())


def part3_preprocessing(df: pd.DataFrame) -> None:
    print("\n=== PART 3: preprocessing functions ===")
    picks = []
    for dx, image_type in [("MEL", "dermoscopic"), ("VASC", "clinical: close-up")]:
        rows = df[(df[config.CLASS_COL] == dx) & (df["image_type"] == image_type)]
        picks.append(rows.sample(1, random_state=config.SEED).iloc[0])
    variants = [("RGB, centre crop, /255", {}),
                ("RGB, stretched, /255", {"keep_aspect": False}),
                ("grey, centre crop, /255", {"color_mode": "gray"}),
                ("RGB, per-image min-max", {"normalize": "minmax"})]
    images, titles = [], []
    for row in picks:
        raw, raw_info = preprocess_image(image_path(row["isic_id"]), size=None, normalize="none")
        images.append(raw.astype("uint8"))
        titles.append(f"raw {row[config.CLASS_COL]} ({row['image_type'].split(':')[0]})\n"
                      f"{raw_info['final_shape']} 0-255")
        for name, options in variants:
            arr, info = preprocess_image(image_path(row["isic_id"]), **options)
            images.append(arr)
            titles.append(f"{name}\n{info['final_shape']} "
                          f"[{info['value_range'][0]:.2f}, {info['value_range'][1]:.2f}]")
    fig = show_grid(images, labels=titles, ncols=5, n=len(images), cell_size=2.1)
    save_figure(fig, FIG / "fig5_preprocessing_before_after.jpg")
    print("example info dict:", preprocess_image(image_path(picks[0]["isic_id"]))[1])

    # a batch with one missing and one corrupt file: both are skipped and reported
    corrupt = OUT / "corrupt_example.jpg"
    corrupt.write_bytes(b"not a jpeg")
    sources = [image_path(i) for i in df["isic_id"].head(3)] + [image_path("ISIC_DOES_NOT_EXIST"), corrupt]
    result = preprocess_batch(sources, size=224)
    corrupt.unlink()
    print("batch images:", result.images.shape)
    print(result.summary())
    (OUT / "batch_skip_demo.txt").write_text(result.summary() + "\n")


def part4_5_loader_and_visualizer(df: pd.DataFrame) -> None:
    print("\n=== PARTS 4-5: loader + visualizer ===")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1, 2.6]})
    plot_class_balance(df[config.TARGET_COL], ax=axes[0], order=["Benign", "Indeterminate", "Malignant"],
                       title="diagnosis_1")
    order11 = df[config.CLASS_COL].value_counts().index.tolist()
    plot_class_balance(df[config.CLASS_COL], ax=axes[1], order=order11, log=True,
                       title="11-class diagnosis, log scale")
    fig.tight_layout()
    save_figure(fig, FIG / "fig6_class_balance.png")
    print("images per diagnosis_1:", df[config.TARGET_COL].value_counts().to_dict())
    print("images per 11-class:", df[config.CLASS_COL].value_counts().to_dict())

    loader = MilkLoader(df, batch_size=16, shuffle=True, size=224, normalize="scale")
    print(loader)
    print("class index:", loader.class_to_index, "| class counts:", loader.class_counts().to_dict())
    batch = next(iter(loader))
    images, labels, ids = batch
    print("first batch:", images.shape, images.dtype, labels[:8], ids[:2])
    save_figure(show_grid(batch, class_names=loader.classes, n=16, ncols=8, cell_size=1.5),
                FIG / "fig7_loader_batch.jpg")
    raw = [preprocess_image(image_path(i), size=None, normalize="none")[0].astype("uint8") for i in ids[:4]]
    save_figure(plot_batch_summary(images, raw_images=raw), FIG / "fig8_batch_summary.jpg")


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"{len(df):,} images, {df['lesion_id'].nunique():,} lesions")
    part1_metadata(df)
    part2_color(df)
    part3_preprocessing(df)
    part4_5_loader_and_visualizer(df)
    print(f"\nFigures saved in {FIG.relative_to(config.REPO_ROOT)}/, tables in {OUT.relative_to(config.REPO_ROOT)}/")


if __name__ == "__main__":
    main()
