# Land Type Classification with Sentinel-2 Satellite Imagery

Project 6 for the Digital Egypt Pioneers Initiative (DEPI), AI & Data Science
Track, Round 2.

Land-cover classification on the EuroSAT dataset (10 classes) using machine
learning and deep learning. The project has two parts:

1. **Multispectral (13-band) pipeline**: preprocessing, data loaders, a
   `SmallCNN` baseline with early stopping, evaluation, and plots.
2. **RGB architecture comparison**: CNN, ResNet-18, and VGG-16 (scratch,
   frozen, and fine-tuned) trained and compared on EuroSAT RGB.

> **Scope note:** the Sentinel-2 AOI definitions in `src/aoi_config.py` are
> configuration only. Automated Sentinel-2 scene search/download is not
> implemented.

## Project structure

| Path | Description |
| --- | --- |
| `src/download_eurosat.py` | Download and extract the EuroSAT RGB or multispectral archive |
| `src/preprocessing.py` | Multispectral preprocessing: dark-object haze removal, resize, normalization (13-band GeoTIFF to `.npy`) |
| `src/dataset_ms.py` | Multispectral loader with reproducible, saved train/val/test splits |
| `src/dataset.py` | ImageFolder-based loaders and split manifests (RGB) |
| `src/model.py` | `SmallCNN`, ResNet-18, and configurable VGG-16 builders |
| `src/train.py` | Train the multispectral `SmallCNN` (best-checkpoint selection, early stopping) |
| `src/evaluate.py` | Evaluate the saved multispectral checkpoint on the test split |
| `src/plot_metrics.py` | Plot training curves from the metrics CSV |
| `src/architecture_experiments.py` | RGB architecture training, comparison, and MLflow logging |
| `src/tracking.py` | Local MLflow experiment tracking |
| `src/aoi_config.py` | Example Egypt AOIs, Sentinel-2 band groups, land-cover labels |
| `notebooks/` | Data exploration, PCA analysis, and data collection notebooks |
| `tests/` | Tests for preprocessing, datasets, models, tracking, and training |
| `outputs/` | Baseline, evaluation, and architecture-comparison artifacts |
| `reports/` | Report figures and tables |
| `docs/` | Milestone notes and handoff documents |
| `API_DESIGN.md` | Proposed prediction API (design only, no server included) |

## Setup

Python 3.10 to 3.12 is recommended (some dependencies may not yet ship
wheels for the newest Python releases).

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download the data

RGB (default, saved to `data/raw/EuroSAT_RGB`):

```bash
python src/download_eurosat.py
```

13-band multispectral (saved to `data/raw/EuroSAT_MS`):

```bash
python src/download_eurosat.py --kind ms
```

Archives come from [Zenodo record 7711810](https://zenodo.org/records/7711810).
Raw data is not tracked by Git. The only tracked file under
`data/processed/` is `splits.csv`, so every teammate uses the same
multispectral split.

## Multispectral pipeline (13 bands)

### Expected layout

```text
data/raw/EuroSAT_MS/
  AnnualCrop/
    image_1.tif
  Forest/
    image_2.tif
  ...
```

`resolve_data_dir()` also finds the class folders if the archive extracts into
an extra nested directory. Raw Sentinel-2 values are divided by 10000 when
loaded; values already in the 0-1 range are left unchanged.

### Preprocess (optional)

The loaders can read `.tif` files directly. To create resized, normalized
`.npy` arrays instead:

```bash
python src/preprocessing.py
```

Settings (`INPUT_FOLDER`, `OUTPUT_FOLDER`, `IMAGE_SIZE`, `REMOVE_HAZE`,
`SCALE`) are constants at the top of the file. The haze offset is estimated per
band from a sample of the whole dataset (dark-object subtraction).

### Train

```bash
python -u -m src.train --data-dir data/raw/EuroSAT_MS
```

- Creates `data/processed/splits.csv` on the first run and reuses it afterward.
- Selects the best checkpoint by validation loss and stops early when it
  stops improving.
- Saves `best_model.pth`, `classes.json`, `metrics.csv`, and `metrics.json`
  to `outputs/baseline/`.
- Logs the run to local MLflow (use `--no-tracking` to skip).

### Evaluate and plot

```bash
python -u -m src.evaluate
python src/plot_metrics.py
```

Evaluation writes `evaluation_summary.json`, `classification_report.csv`, and
a confusion matrix to `outputs/evaluation/`.

> **Baseline metrics:** the files under `outputs/baseline/` and
> `outputs/evaluation/` were produced in earlier development and copied here.
> Treat the recorded numbers as historical until training is re-run locally.

## RGB architecture comparison

Presets: `cnn`, `resnet18`, `vgg16`, plus `-frozen` (ImageNet features frozen,
train the head only) and `-finetune` (ImageNet initialization, full update)
variants for ResNet-18 and VGG-16. Pretrained runs resize to 224x224 and use
ImageNet normalization; scratch runs keep the native image size.

```bash
# Scratch baselines
python -m src.architecture_experiments --models cnn resnet18 vgg16 --epochs 5 --batch-size 64 --seed 42

# Transfer learning
python -m src.architecture_experiments --models resnet18-frozen vgg16-frozen --epochs 5 --batch-size 64 --seed 42

# Quick smoke run
python -m src.architecture_experiments --models cnn --epochs 1 --batch-size 8 --seed 42
```

The split is a deterministic, class-stratified 70/15/15 split from the seed.
A saved manifest directory can be supplied with `--split-manifest-dir`.
Each invocation creates a timestamped folder under
`outputs/architecture_comparison/` with checkpoints, histories, confusion
matrices, JSON metrics, and sorted comparison tables.

### Recorded results (EuroSAT RGB, seed 42, 5 epochs)

| Configuration | Test accuracy | Test macro F1 | Parameters (total / trainable) |
| --- | ---: | ---: | ---: |
| Custom CNN, scratch | 87.70% | 87.37% | 289,194 / 289,194 |
| ResNet-18, scratch | 70.81% | 70.33% | 11,181,642 / 11,181,642 |
| VGG-16, scratch | 65.46% | 59.78% | 14,982,474 / 14,982,474 |
| ResNet-18, ImageNet frozen | 92.57% | 92.20% | 11,181,642 / 5,130 |

Frozen ResNet-18 gave the best accuracy on this split, while the custom CNN is
the compact option. These results cover one seed and five epochs and do not
establish performance over repeated splits or longer training. Full per-class
metrics are in `outputs/architecture_comparison/`.

## Experiment tracking

Runs are logged to a project-local `mlruns/` directory (no server needed):

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000` and select the
`eurosat-land-cover-classification` experiment. Multispectral runs carry
`pipeline=multispectral` as a parameter.

## Explore the data

```bash
jupyter notebook
```

Open a notebook under `notebooks/`. They cover class counts, sample images,
band metadata, spectral signatures, NDVI (from `B04` and `B08`), and PCA.
Selected results are saved under `outputs/` and `reports/`.

## Tests

Run from the repository root (do not run test files with an editor "Run"
button, since imports need the project root):

```bash
python -m pytest
```

Tests use small synthetic data and do not require the EuroSAT download.

## Data and citation

EuroSAT is a land-use and land-cover dataset derived from Sentinel-2 imagery.
See the [EuroSAT project page](https://github.com/phelber/EuroSAT) and the
[Zenodo record](https://zenodo.org/records/7711810) for citation details and
terms of use.