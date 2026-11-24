# M2-T02: Seasonal Analysis and Spectral Variability

## Objective

Investigate whether reliable image acquisition dates are available
for seasonal analysis. If they are unavailable, use between-image
spectral variability per land-cover class as an alternative.

## Acquisition-date availability

The metadata audit successfully inspected 27,000 TIFF images.

No date-like keys were found in the dataset-level or per-band TIFF metadata tags and namespaces exposed by Rasterio.

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

B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A.

The figures display bands in increasing wavelength order:

B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.

## Results

- Images discovered: 27,000
- Images successfully processed: 27,000
- Processing errors: 0
- Images inspected for metadata: 27,000
- Metadata scopes inspected: 459,000
- Date-like metadata keys detected: 0
- Land-cover classes: 10
- Bands per image: 13

### Highest between-image SD within each band

The following table identifies the class with the largest SD
for each individual band. Comparisons are made within the same band,
not across bands.

| Band | Class | SD |
|---|---|---:|
| B01 | Pasture | 263.892 |
| B02 | Industrial | 219.104 |
| B03 | SeaLake | 288.143 |
| B04 | AnnualCrop | 481.233 |
| B05 | AnnualCrop | 457.656 |
| B06 | River | 472.406 |
| B07 | River | 628.220 |
| B08 | River | 614.002 |
| B09 | Forest | 342.627 |
| B10 | HerbaceousVegetation | 5.582 |
| B11 | AnnualCrop | 664.210 |
| B12 | AnnualCrop | 559.999 |
| B8A | River | 699.429 |

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

