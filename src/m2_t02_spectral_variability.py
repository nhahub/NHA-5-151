from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Paths and configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT_MS"
DATA_DIR = (
    BASE_DIR / "EuroSAT_MS"
    if (BASE_DIR / "EuroSAT_MS").exists()
    else BASE_DIR
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "M2_T02_seasonal_analysis.md"

for folder in (TABLES_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A"
]

WAVELENGTH_NM = {
    "B01": 443,
    "B02": 490,
    "B03": 560,
    "B04": 665,
    "B05": 705,
    "B06": 740,
    "B07": 783,
    "B08": 842,
    "B8A": 865,
    "B09": 945,
    "B10": 1375,
    "B11": 1610,
    "B12": 2190,
}

# Storage order is preserved in CSV files.
# Plots use increasing wavelength order.
PLOT_ORDER = sorted(BAND_NAMES, key=lambda b: WAVELENGTH_NM[b])

EXPECTED_BANDS = len(BAND_NAMES)
CV_EPSILON = 1e-10

DATE_KEY_TERMS = (
    "date",
    "time",
    "acq",
    "sensing",
    "timestamp",
    "datetime",
    "start_time",
    "end_time",
)

print("Dataset:", DATA_DIR)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")


# 2. Metadata audit helper
# ---------------------------------------------------------------------

def inspect_metadata(src, image_name):
    """Inspect dataset-level and per-band TIFF metadata tags."""

    found_keys = set()
    date_candidates = []
    scopes_checked = 0

    # Dataset-level tags and namespaces.
    dataset_namespaces = [None]

    try:
        dataset_namespaces.extend(src.tag_namespaces())
    except Exception:
        pass

    for namespace in dataset_namespaces:
        try:
            tags = (
                src.tags()
                if namespace is None
                else src.tags(ns=namespace)
            )

            scopes_checked += 1

            for key, value in tags.items():
                found_keys.add(key)

                if any(term in key.lower() for term in DATE_KEY_TERMS):
                    date_candidates.append({
                        "image": image_name,
                        "scope": "dataset",
                        "namespace": namespace or "default",
                        "key": key,
                        "value": str(value),
                    })

        except Exception:
            continue

    # Per-band tags and namespaces.
    for band_index in range(1, src.count + 1):
        namespaces = [None]

        try:
            namespaces.extend(src.tag_namespaces(bidx=band_index))
        except Exception:
            pass

        for namespace in namespaces:
            try:
                tags = (
                    src.tags(bidx=band_index)
                    if namespace is None
                    else src.tags(bidx=band_index, ns=namespace)
                )

                scopes_checked += 1

                for key, value in tags.items():
                    found_keys.add(key)

                    if any(term in key.lower() for term in DATE_KEY_TERMS):
                        date_candidates.append({
                            "image": image_name,
                            "scope": f"band_{band_index}",
                            "namespace": namespace or "default",
                            "key": key,
                            "value": str(value),
                        })

            except Exception:
                continue

    return found_keys, date_candidates, scopes_checked


# 3. Read images and audit metadata
# ---------------------------------------------------------------------

records = []
errors = []
metadata_errors = []
metadata_date_candidates = []

tag_keys = set()

images_discovered = 0
images_checked = 0
metadata_scopes_checked = 0

class_dirs = sorted(
    path for path in DATA_DIR.iterdir()
    if path.is_dir()
)

for class_dir in class_dirs:

    paths = sorted(
        path for path in class_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff"}
    )

    images_discovered += len(paths)

    print(f"{class_dir.name}: {len(paths)} images")

    for path in paths:

        try:
            with rasterio.open(path) as src:

                # Audit metadata independently from image processing.
                try:
                    keys, candidates, scopes = inspect_metadata(
                        src, str(path.relative_to(DATA_DIR))
                    )

                    tag_keys.update(keys)
                    metadata_date_candidates.extend(candidates)
                    metadata_scopes_checked += scopes
                    images_checked += 1

                except Exception as metadata_exc:
                    metadata_errors.append({
                        "image": str(path.relative_to(DATA_DIR)),
                        "error": str(metadata_exc),
                    })

                image = src.read().astype(np.float64)

            if image.ndim != 3 or image.shape[0] != EXPECTED_BANDS:
                raise ValueError(
                    f"Unexpected image shape: {image.shape}"
                )

            means = np.nanmean(image, axis=(1, 2))

            if not np.all(np.isfinite(means)):
                raise ValueError("Non-finite band mean")

            records.append({
                "class": class_dir.name,
                "image": path.name,
                **dict(zip(BAND_NAMES, map(float, means))),
            })

        except Exception as exc:
            errors.append({
                "class": class_dir.name,
                "image": path.name,
                "error": str(exc),
            })


if not records:
    raise RuntimeError("No images were successfully processed.")


# 4. Save image-level means and errors
# ---------------------------------------------------------------------

image_df = pd.DataFrame(records)

image_df.to_csv(
    TABLES_DIR / "spectral_variability_image_means.csv",
    index=False,
)

pd.DataFrame(
    errors,
    columns=["class", "image", "error"],
).to_csv(
    TABLES_DIR / "spectral_variability_errors.csv",
    index=False,
)

pd.DataFrame(
    metadata_errors,
    columns=["image", "error"],
).to_csv(
    TABLES_DIR / "spectral_variability_metadata_errors.csv",
    index=False,
)

pd.DataFrame(
    metadata_date_candidates,
    columns=["image", "scope", "namespace", "key", "value"],
).to_csv(
    TABLES_DIR / "spectral_variability_metadata_date_candidates.csv",
    index=False,
)

# 5. Calculate between-image variability
# ---------------------------------------------------------------------

rows = []

for class_name, group in image_df.groupby("class"):

    for band in BAND_NAMES:

        values = group[band].dropna()

        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else np.nan

        if (
            np.isfinite(mean)
            and abs(mean) > CV_EPSILON
            and np.isfinite(std)
        ):
            cv = std / abs(mean)
        else:
            cv = np.nan

        rows.append({
            "class": class_name,
            "band": band,
            "mean_between_images": mean,
            "std_between_images": std,
            "cv_between_images": cv,
            "images_used": len(values),
        })


summary_df = pd.DataFrame(rows)

summary_df.to_csv(
    TABLES_DIR / "spectral_variability_by_class.csv",
    index=False,
)

# 6. Mean spectral signature with +/- 1 SD
# ---------------------------------------------------------------------

x = np.arange(len(PLOT_ORDER))

plt.figure(figsize=(15, 8))

for class_name in sorted(image_df["class"].unique()):

    subset = (
        summary_df[summary_df["class"] == class_name]
        .set_index("band")
        .reindex(PLOT_ORDER)
    )

    means = subset["mean_between_images"].to_numpy()
    stds = subset["std_between_images"].fillna(0).to_numpy()

    plt.plot(
        x,
        means,
        marker="o",
        linewidth=1.5,
        label=class_name,
    )

    plt.fill_between(
        x,
        means - stds,
        means + stds,
        alpha=0.10,
    )

plt.xticks(x, PLOT_ORDER, rotation=45)
plt.xlabel("Sentinel-2 bands (ordered by wavelength)")
plt.ylabel("Mean pixel value")

plt.title(
    "EuroSAT-MS: Mean spectral signature per class "
    "(shaded: +/- 1 SD between images)"
)

plt.grid(alpha=0.25)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "spectral_variability_by_class.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close()

# 7. Heatmap helper
# ---------------------------------------------------------------------

def create_heatmap(
    value_column,
    title,
    colorbar_label,
    filename,
    scale_per_band=False,
    skip_bands=(),
):

    bands = [
        band for band in PLOT_ORDER
        if band not in skip_bands
    ]

    pivot = (
        summary_df
        .pivot(index="class", columns="band", values=value_column)
        .reindex(columns=bands)
        .sort_index()
    )

    data = pivot.to_numpy(dtype=float)

    if scale_per_band:

        maxima = np.nanmax(data, axis=0, keepdims=True)
        maxima[maxima == 0] = np.nan

        data = data / maxima

    plt.figure(figsize=(15, 7))

    plt.imshow(data, aspect="auto")
    plt.colorbar(label=colorbar_label)

    plt.xticks(
        np.arange(len(bands)),
        bands,
        rotation=45,
    )

    plt.yticks(
        np.arange(len(pivot.index)),
        pivot.index,
    )

    plt.xlabel("Sentinel-2 bands (ordered by wavelength)")
    plt.ylabel("Land-cover class")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()


# SD heatmap: normalized independently within each band.
create_heatmap(
    value_column="std_between_images",
    title="Between-image SD per class (normalized within each band)",
    colorbar_label="SD / maximum SD of the same band",
    filename="spectral_variability_heatmap.png",
    scale_per_band=True,
)

# CV heatmap: B10 is omitted from this visualization only.
# B10 remains in the CSV and the SD analysis.
create_heatmap(
    value_column="cv_between_images",
    title="Between-image relative variability (CV = SD / mean)",
    colorbar_label="Coefficient of Variation",
    filename="spectral_variability_cv_heatmap.png",
    skip_bands=("B10",),
)

# 8. Generate interpretation and report
# ---------------------------------------------------------------------

class_count = image_df["class"].nunique()
processed_count = len(image_df)
error_count = len(errors)

date_like_keys = sorted({
    item["key"]
    for item in metadata_date_candidates
})

if date_like_keys:
    metadata_conclusion = (
        "Date-like metadata keys were detected. Their values require "
        "manual verification to determine whether they represent actual "
        "image acquisition dates. Their presence alone does not establish "
        "reliable temporal information."
    )
else:
    metadata_conclusion = (
        "No date-like keys were found in the dataset-level or per-band "
        "TIFF metadata tags and namespaces exposed by Rasterio."
    )

# Identify the class with the largest SD within each band.
top_sd_rows = []

for band in BAND_NAMES:

    band_data = summary_df[
        summary_df["band"] == band
    ].dropna(subset=["std_between_images"])

    if not band_data.empty:

        top_row = band_data.loc[
            band_data["std_between_images"].idxmax()
        ]

        top_sd_rows.append(
            f"| {band} | {top_row['class']} | "
            f"{top_row['std_between_images']:.3f} |"
        )

top_sd_table = "\n".join(top_sd_rows)

report = f"""# M2-T02: Seasonal Analysis and Spectral Variability

## Objective

Investigate whether reliable image acquisition dates are available
for seasonal analysis. If they are unavailable, use between-image
spectral variability per land-cover class as an alternative.

## Acquisition-date availability

The metadata audit successfully inspected {images_checked:,} TIFF images.

{metadata_conclusion}

The audit examined dataset-level and per-band tags and the metadata
namespaces exposed by Rasterio. It does not establish the absence of
dates from every possible external source.

The project-file inspection also did not identify verified
per-image acquisition dates.

Therefore, seasonal variation could not be reliably analyzed.
No seasons were inferred from image appearance.

## Alternative method: Between-image spectral variability

For each successfully processed image:

1. Read the 13 stored spectral bands.
2. Calculate the mean pixel value of each band within the image.
3. Group image-level means by land-cover class.
4. Calculate the mean and sample standard deviation across images
   within each class and band.
5. Calculate the coefficient of variation (CV) as SD divided by
   the absolute mean, where the mean is sufficiently far from zero.

The standard deviation measures variability between images,
not variability among pixels within one image.

CV provides a relative measure of variability. It is unitless and
should be interpreted carefully when mean values are small.

## Band order

The CSV tables preserve the corrected band storage order:

{", ".join(BAND_NAMES)}.

The figures display bands in increasing wavelength order:

{", ".join(PLOT_ORDER)}.

## Results

- Images discovered: {images_discovered:,}
- Images successfully processed: {processed_count:,}
- Processing errors: {error_count:,}
- Images inspected for metadata: {images_checked:,}
- Metadata scopes inspected: {metadata_scopes_checked:,}
- Date-like metadata keys detected: {len(date_like_keys)}
- Land-cover classes: {class_count}
- Bands per image: {EXPECTED_BANDS}

### Highest between-image SD within each band

The following table identifies the class with the largest SD
for each individual band. Comparisons are made within the same band,
not across bands.

| Band | Class | SD |
|---|---|---:|
{top_sd_table}

## Interpretation

Between-image spectral variability differs across land-cover classes
and spectral bands. The SD describes the spread of image-level mean
values within each class and band. CV complements SD by expressing
variability relative to the mean.

The SD heatmap is normalized independently within each band.
Its colors therefore support comparisons between classes for the
same band, not comparisons of absolute SD across different bands.

B10 is retained in the numerical tables and SD analysis but omitted
from the CV heatmap because its comparatively low value scale can
make relative variability disproportionately sensitive to the mean.

These results describe cross-image variability. They do not establish
seasonal patterns or temporal change.

## Limitations

- Reliable per-image acquisition dates were not identified in the
  inspected project files or TIFF metadata.
- Seasonal trends cannot be established without reliable dates.
- Between-image differences may reflect multiple factors and cannot
  be attributed to season.
- Band values are reported as stored pixel values; no additional
  reflectance conversion or scaling was applied.
- CV can be sensitive to small mean values.
- The analysis uses image-level band means and does not describe
  within-image pixel variability.

## Generated outputs

### Tables

- `reports/tables/spectral_variability_image_means.csv`
- `reports/tables/spectral_variability_by_class.csv`
- `reports/tables/spectral_variability_errors.csv`
- `reports/tables/spectral_variability_metadata_errors.csv`
- `reports/tables/spectral_variability_metadata_date_candidates.csv`

### Figures

- `reports/figures/spectral_variability_by_class.png`
- `reports/figures/spectral_variability_heatmap.png`
- `reports/figures/spectral_variability_cv_heatmap.png`

"""

REPORT_PATH.write_text(report, encoding="utf-8")


# 9. Final console summary
# ---------------------------------------------------------------------

print("\n" + "=" * 60)
print("M2-T02 ANALYSIS COMPLETED")
print("=" * 60)

print("Images discovered :", images_discovered)
print("Processed OK      :", processed_count)
print("Processing errors :", error_count)
print("Metadata checked  :", images_checked)
print("Metadata scopes   :", metadata_scopes_checked)
print("Date-like keys    :", date_like_keys or "none")
print("Classes           :", class_count)
print("Bands             :", ", ".join(BAND_NAMES))

print("\nGenerated report:")
print(REPORT_PATH)

print("\nGenerated tables:")
print("spectral_variability_image_means.csv")
print("spectral_variability_by_class.csv")
print("spectral_variability_errors.csv")
print("spectral_variability_metadata_errors.csv")
print("spectral_variability_metadata_date_candidates.csv")

print("\nGenerated figures:")
print("spectral_variability_by_class.png")
print("spectral_variability_heatmap.png")
print("spectral_variability_cv_heatmap.png")

print("\nM2-T02 finished successfully.")
