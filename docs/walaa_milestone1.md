# Walaa Mohamed — Milestone 1

## Dataset and class balance

The dataset is EuroSAT RGB in `data/raw/EuroSAT_RGB/EuroSAT_RGB`, loaded as
class folders of JPEG images with `torchvision.datasets.ImageFolder`. It was
downloaded with the repository's downloader because no usable raw images were
present initially. The raw images remain unchanged.

| Class | Before split |
| --- | ---: |
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |
| **Total** | **27,000** |

Pasture is the smallest class (2,000 images); Highway, Industrial,
PermanentCrop, and River each have 2,500, compared with 3,000 in the largest
classes. These are comparatively smaller, but all classes have enough examples
for the reproducible 70/15/15 split and appear in each split. Additional QGIS
labeling is **not currently required**. No QGIS work or labels are claimed.

## Split and validation

`src/dataset.py` performs a two-stage stratified split with random seed `42`.
`src/split_data.py` runs it, reports source and split class counts, checks
class-ratio deviation (maximum 1 percentage point), verifies that sample
indices and image paths do not overlap, and verifies every source sample is
assigned exactly once. It writes CSV manifests; it does not copy or alter raw
images. Manifest paths are relative to the `ImageFolder` root.

| Class | Train (70%) | Validation (15%) | Test (15%) |
| --- | ---: | ---: | ---: |
| AnnualCrop | 2,100 | 450 | 450 |
| Forest | 2,100 | 450 | 450 |
| HerbaceousVegetation | 2,100 | 450 | 450 |
| Highway | 1,750 | 375 | 375 |
| Industrial | 1,750 | 375 | 375 |
| Pasture | 1,400 | 300 | 300 |
| PermanentCrop | 1,750 | 375 | 375 |
| Residential | 2,100 | 450 | 450 |
| River | 1,750 | 375 | 375 |
| SeaLake | 2,100 | 450 | 450 |
| **Total** | **18,900** | **4,050** | **4,050** |

The generated manifest structure is
`data/processed/splits/seed_42/{train,validation,test}.csv`. Each CSV has
`image_path`, `class_name`, and `label_id` columns. The generated files are
under the repository's existing ignored processed-data directory.

## Files and remaining work

- Created `src/split_data.py` to run, validate, summarize, and manifest the split.
- Created this contribution record.
- Updated `README.md` with the script and manifest usage.
- No manual QGIS labeling remains for this milestone. Re-run the script after
  any future data/label changes to regenerate and revalidate the manifests.

Run from the repository root with:

```bash
python src/split_data.py
```
