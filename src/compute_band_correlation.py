"""Compute dataset-level spectral band correlations for EuroSAT_MS.

M2-T01 support script. The existing `reports/tables/sample_band_correlation.csv`
is a single-image, pixel-level check (one AnnualCrop image, see
notebooks/02_data_exploration_eurosat_ms.ipynb). This script computes the
dataset-level counterpart: Pearson correlation between bands of the
*per-image band means* across many images, so the values describe how bands
co-vary across the dataset rather than within one image.

No results are invented: the script refuses to run without the real
EuroSAT_MS dataset and validates shape/content of every image it reads
(via src.dataset_ms.load_multispectral_image).

By default the correlation is fit on the TRAIN split only (using the
repo's persisted data/processed/splits.csv), so the result can safely feed
band-selection or PCA decisions without test leakage. If the split file is
missing or inconsistent, the script fails with a clear error instead of
silently using all images. `--split all` is an explicit report-only mode
that uses every image and discloses that choice in the run metadata.

Usage:
    python -m src.compute_band_correlation \
        --data-dir data/raw/EuroSAT_MS \
        --max-per-class 300 --seed 42

    # report-only, all images (disclosed; do not use for band selection/PCA):
    python -m src.compute_band_correlation --split all

Outputs (repo-relative):
    reports/tables/dataset_band_correlation.csv
    reports/figures/dataset_band_correlation_heatmap.png
    reports/tables/dataset_band_correlation_summary.csv  (run metadata)
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.dataset_ms import (
    SENTINEL2_BAND_ORDER,
    discover_samples,
    load_multispectral_image,
    load_splits,
    resolve_data_dir,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"
DEFAULT_SPLITS_PATH = ROOT / "data" / "processed" / "splits.csv"


def select_samples(samples, max_per_class, seed):
    """Seeded per-class subsample, or everything when max_per_class is None."""
    if max_per_class is None:
        return list(samples)
    rng = np.random.default_rng(seed)
    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[2], []).append(sample)
    selected = []
    for class_name in sorted(by_class):
        entries = by_class[class_name]
        if len(entries) > max_per_class:
            picked = rng.choice(len(entries), max_per_class, replace=False)
            entries = [entries[i] for i in sorted(picked)]
        selected.extend(entries)
    return selected


def per_image_band_means(selected, num_bands):
    """Return (n_images x num_bands) matrix of per-image band means."""
    means = np.full((len(selected), num_bands), np.nan, dtype=np.float64)
    errors = []
    for row, (path, _label, _class_name) in enumerate(selected):
        try:
            image = load_multispectral_image(path)
            if image.shape[0] != num_bands:
                raise ValueError(
                    f"Expected {num_bands} bands, got {image.shape[0]}: {path}"
                )
            means[row] = image.reshape(num_bands, -1).mean(axis=1)
        except Exception as exc:  # recorded, not hidden
            errors.append({"path": str(path), "error": repr(exc)})
    return means, errors


def save_heatmap(corr, band_names, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 7))
    image = plt.imshow(corr.values if hasattr(corr, "values") else corr,
                       aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(len(band_names)), band_names, rotation=45, ha="right")
    plt.yticks(range(len(band_names)), band_names)
    plt.colorbar(image, label="Pearson correlation")
    plt.title("Band correlation - per-image band means across dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-level EuroSAT_MS band correlation (M2-T01)."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="EuroSAT_MS dataset directory (default: repo standard locations).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional per-class cap with seeded sampling (default: all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("train", "all"),
        default="train",
        help=(
            "train (default): fit on the persisted train split only, safe "
            "for band-selection/PCA decisions; fails clearly if the split "
            "file is missing or inconsistent. "
            "all: report-only mode over every image; disclosed in metadata, "
            "not safe as a modeling decision input."
        ),
    )
    parser.add_argument(
        "--splits-path",
        default=str(DEFAULT_SPLITS_PATH),
        help="Split CSV used when --split train (default: repo persisted split).",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    samples, classes = discover_samples(data_dir)
    if len(classes) < 2:
        raise ValueError(
            f"Need at least 2 classes for a dataset-level correlation, "
            f"found {len(classes)} in {data_dir}"
        )

    if args.split == "train":
        splits_path = Path(args.splits_path)
        if not splits_path.is_file():
            raise FileNotFoundError(
                f"Train-only mode needs a persisted split file, not found: "
                f"{splits_path}. Generate it via get_loaders(splits_path=...) "
                f"or rerun with --split all (report-only)."
            )
        # load_splits raises a clear ValueError on unknown samples,
        # duplicates, metadata mismatch, or incomplete coverage — no silent
        # fallback to all images.
        train_idx, _val_idx, _test_idx = load_splits(
            samples, data_dir, splits_path
        )
        base_samples = [samples[i] for i in train_idx]
    else:
        base_samples = list(samples)

    selected = select_samples(base_samples, args.max_per_class, args.seed)
    if len(selected) < 30:
        raise ValueError(
            f"Too few images for a meaningful correlation: {len(selected)}"
        )

    num_bands = len(SENTINEL2_BAND_ORDER)
    means, errors = per_image_band_means(selected, num_bands)

    valid = ~np.isnan(means).any(axis=1)
    if valid.sum() < 30:
        raise ValueError(
            f"Too few readable images: {int(valid.sum())} of {len(selected)}"
        )
    means = means[valid]

    band_names = list(SENTINEL2_BAND_ORDER)
    corr = np.corrcoef(means, rowvar=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corr_path = TABLES_DIR / "dataset_band_correlation.csv"
    with corr_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([""] + band_names)
        for name, row_values in zip(band_names, corr):
            writer.writerow([name] + [f"{value:.6f}" for value in row_values])

    heatmap_path = FIGURES_DIR / "dataset_band_correlation_heatmap.png"
    save_heatmap(corr, band_names, heatmap_path)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Pearson correlation between bands of per-image band means "
            "across images (dataset-level), computed with "
            "src/compute_band_correlation.py"
        ),
        "data_dir": str(data_dir),
        "classes": classes,
        "split_mode": args.split,
        "split_disclosure": (
            "train split only; safe as modeling-decision input"
            if args.split == "train"
            else "REPORT-ONLY: includes val/test images; do NOT use for "
                 "band selection or PCA fitting"
        ),
        "splits_path": str(args.splits_path) if args.split == "train" else None,
        "images_discovered": len(samples),
        "images_in_scope": len(base_samples),
        "images_selected": len(selected),
        "images_used": int(valid.sum()),
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "read_errors": errors,
    }
    summary_path = TABLES_DIR / "dataset_band_correlation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "value"])
        for key, value in summary.items():
            writer.writerow([key, json.dumps(value, default=str)])

    print(f"Split mode: {args.split} "
          f"({'train-only, decision-safe' if args.split == 'train' else 'REPORT-ONLY, includes val/test'})")
    print(f"Images used: {int(valid.sum())} / {len(base_samples)} in scope "
          f"({len(samples)} discovered, {len(errors)} read errors)")
    print(f"Wrote {corr_path}")
    print(f"Wrote {heatmap_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
