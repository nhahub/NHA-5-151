# M2-T01 — Spectral Band Correlation Interpretation

**Author:** Ramez Hesham
**Status:** Preliminary descriptive check — pending multi-image recomputation before use in the final report.

## Provenance and method

- **Source artifacts:** `reports/tables/sample_band_correlation.csv` and `reports/figures/band_correlation_heatmap.png`, produced by `notebooks/02_data_exploration_eurosat_ms.ipynb` (section 11, "Band correlation analysis").
- **Data:** a real EuroSAT_MS image — the recorded sample is `AnnualCrop/AnnualCrop_1.tif` (see `reports/tables/eda_summary.csv`; the same notebook audit read 300 real dataset files successfully, 0 errors).
- **Method:** Pearson correlation computed **across pixels of that single image** (13 bands x 64 x 64 = 53,248 pixels, subsampled to 50,000 for runtime). The notebook itself frames this as "a descriptive check of how band values co-vary in this image, not a model feature-importance score."
- **Consequence:** the matrix measures *within-image, pixel-level linear covariation in one AnnualCrop image*. It is not computed across images or classes, so it does not by itself establish dataset-level band redundancy or anything about classification.

## Band order and wavelengths (verified)

- EuroSAT_MS `.tif` files store the 13 bands in the order `B01..B08, B09, B10, B11, B12, B8A` — **B8A is the last band, not the 9th**. This matches evidence recorded in the torchgeo project's EuroSAT fix (torchgeo issue #2458: the 9th stored band has 60 m resolution, which is true of B09 water vapour, not B8A; the 10th band's value range matches B10 cirrus) and the issue on the original EuroSAT repository. The repo's loader (`src/dataset_ms.py`, `SENTINEL2_BAND_ORDER`) and the exploration notebooks use this order consistently.
- `outputs/sentinel2_band_information.csv` is a band-to-wavelength reference table sorted by wavelength (B8A between B08 and B09); that is the *reference sort*, not the file storage order. Its wavelength values (B01 443 nm … B12 2190 nm, B8A 865 nm) agree with the official Sentinel-2 MSI band table (ESA/Copernicus documentation).

## What the matrix shows (in this sample image)

1. **The visible bands B02, B03, B04 covary strongly**: pairwise r = 0.93-0.96 (B02-B04 0.960, B03-B04 0.944, B02-B03 0.929).
2. **The red-edge/NIR group B06, B07, B08, B8A covaries very strongly**: pairwise r = 0.98-0.998 (B06-B07 0.998).
3. **B05 sits between the two groups**: r = 0.85-0.89 with B03/B04 and 0.76-0.80 with B06/B07/B08/B8A.
4. **B09 also covaries with the NIR group** in this image (r = 0.89-0.95).
5. **B01 is weakly linearly related to the red-edge/NIR group** here (|r| <= 0.08) and more strongly to B02 (0.77) and B12 (0.84).
6. **B10 is the least linearly correlated with the other bands** in this sample (strongest |r| = 0.60, with B12; negative values against the NIR group).
7. **B12 shows negative linear covariation with the NIR group** in this image (r = -0.35 to -0.53).

## What this means (and does not mean)

- In this sample image, band values form two high-covariation groups (visible; red-edge/NIR). **Whether the same grouping holds across the dataset is not established by this artifact** — that needs correlation computed across many images (e.g. per-image band means across classes).
- These results are a **hypothesis input for M2-T03 (PCA, Rawda)**: if the grouping holds at dataset level, PCA should capture much of the variance in few components. That is a claim to test with explained-variance ratios and a baseline comparison — not a conclusion from this matrix.
- **No classification statement is made here.** Linear correlation between band pixels does not measure class separability, and no band is dropped based on correlation alone: a band that looks redundant on average can still separate specific class pairs. All 13 bands stay in the baseline input; any dimensionality reduction is adopted only after an experiment on the same split shows it works.

## Limitations and next step

- Single image (AnnualCrop), pixel-level values, one descriptive run. Absolute values will differ on other images and classes.
- **Follow-up is implemented, pending a real-data run:** `src/compute_band_correlation.py` computes the dataset-level counterpart — Pearson correlation between bands of *per-image band means* across images, with seeded per-class sampling, 13-band shape validation and per-file error recording. It refuses to run without the real EuroSAT_MS dataset (no invented results) and writes `reports/tables/dataset_band_correlation.csv`, `reports/figures/dataset_band_correlation_heatmap.png` and a run-metadata summary (`..._summary.csv`, including split mode and read errors).
- **Train-only by default (no leakage):** the script fits on the repo's persisted train split (`data/processed/splits.csv`, 18,900 images) so the result can safely feed band-selection or PCA (M2-T03) decisions. If the split file is missing or stale it fails with a clear error instead of silently using all images. `--split all` exists as an explicit report-only mode and is disclosed as such in the run metadata. Verified with planted-signal fixtures (exact reference recovery, val/test exclusion, failure paths): see `tests/test_compute_band_correlation.py`. Run it on any machine that has the dataset:

  ```bash
  python -m src.compute_band_correlation --data-dir data/raw/EuroSAT_MS --max-per-class 300 --seed 42
  # default: --split train. Report-only over every image: add --split all
  ```

  When that output exists, update the numbers in section "What the matrix shows" with the dataset-level values and state both methods. Until then, all findings above stay qualified as single-image.
- The current CSV/heatmap remain valid as a single-image descriptive check with the method documented above.

## Related artifacts

- Correlation matrix (this analysis): `reports/tables/sample_band_correlation.csv`
- Heatmap: `reports/figures/band_correlation_heatmap.png`
- Producing notebook: `notebooks/02_data_exploration_eurosat_ms.ipynb` (section 11)
- Sample/audit record: `reports/tables/eda_summary.csv`
- Band wavelength reference: `outputs/sentinel2_band_information.csv`
- Mean spectral signatures by class: `reports/tables/mean_spectral_signatures.csv`, `outputs/spectral_signatures_by_class.csv`
- NDVI by class: `outputs/ndvi_by_class.csv`, `outputs/ndvi_by_class.png`
