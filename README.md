# Land Type Classification with Sentinel-2 Satellite Imagery

Project 6 for the Digital Egypt Pioneers Initiative (DEPI), AI & Data Science
Track, Round 2.

This repository contains data exploration notebooks and utilities for
land-cover classification with the EuroSAT dataset. It includes RGB and
multispectral dataset download support, image preprocessing, class-stratified
PyTorch data loaders, CNN/ResNet-18 model definitions, and a complete
multispectral training and evaluation pipeline.

> **Project status:** EuroSAT download, exploration, RGB training utilities,
> and the multispectral preprocessing/training/evaluation pipeline are
> available. The Sentinel-2 AOI definitions are configuration only; automated
> Sentinel-2 scene search/download is not implemented here.

## Contents

| Path | Description |
| --- | --- |
| `src/aoi_config.py` | Example Egypt areas of interest, Sentinel-2 band groups, land-cover labels, and scene-query preferences |
| `src/download_eurosat.py` | Download and extract the EuroSAT RGB or multispectral archive |
| `src/preprocessing.py` | Resize and normalize RGB or multispectral images into `.npy` files |
| `src/dataset.py` | ImageFolder-based train, validation, and test data loaders (RGB) |
| `src/dataset_ms.py` | EuroSAT multispectral loader with reproducible saved train/validation/test splits |
| `src/preprocessing_ms.py` | Multispectral preprocessing: dark-object haze removal, resize, and normalization for 13-band GeoTIFFs |
| `src/split_data.py` | Run and validate the stratified split; write CSV manifests |
| `src/model.py` | `SmallCNN`, ResNet-18, and configurable VGG-16 model builders |
| `src/architecture_experiments.py` | Reproducible EuroSAT RGB architecture training, evaluation, comparison, and MLflow logging |
| `src/tracking.py` | Log training parameters, metrics, and artifacts to local MLflow |
| `train.py` | Train and evaluate the EuroSAT RGB image classifier |
| `src/train.py` | Train the multispectral `SmallCNN` baseline with checkpoint selection and early stopping |
| `src/evaluate.py` | Evaluate the saved multispectral checkpoint on the held-out test split |
| `src/test_dataset.py` | Quick sanity check for the multispectral loaders and saved splits |
| `src/plot_metrics.py` | Plot training curves from the multispectral metrics CSV |
| `API_DESIGN.md` | Proposed prediction API endpoints and payload formats |
| `notebooks/initial_inspection.ipynb` | Initial dataset inspection notebook |
| `notebooks/01_data_exploration_eurosat_ms.ipynb` | Multispectral exploration, spectral signatures, and optional NDVI analysis |
| `notebooks/02_data_exploration_eurosat_ms.ipynb` | Extended multispectral EDA: band audit, spectral signatures, NDVI, and summary tables |
| `notebooks/data_collection.ipynb` | Notebook for the EuroSAT download and project workflow |
| `data/processed/splits.csv` | Saved multispectral train/validation/test assignments (paths relative to the dataset root) |
| `outputs/` | Example exported tables and charts from dataset exploration, plus baseline training artifacts |
| `reports/` | Report figures and tables produced by the exploration and evaluation notebooks/scripts |
| `tests/` | Focused tests for the downloader and model builder |

## Requirements

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)
- Jupyter Notebook or JupyterLab to run the notebooks

## Setup

Create and activate a virtual environment, then install the project
dependencies.

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

`notebook` is for interactive notebook work and is not listed as a core
application dependency in `requirements.txt`.
The architecture shortlist uses the torchvision weight-enum API for explicit
ImageNet transfer learning (`torchvision>=0.13`); MLflow remains the project's
local experiment tracker.

## Download EuroSAT

The downloader supports the RGB and multispectral archives. By default, it
downloads and extracts the RGB archive into `data/raw/EuroSAT_RGB`:

```bash
python src/download_eurosat.py
```

Download the 13-band multispectral version instead:

```bash
python src/download_eurosat.py --kind ms
```

The older `--ms` flag is also supported:

```bash
python src/download_eurosat.py --ms
```

To choose a different extraction directory or provide a mirror URL:

```bash
python src/download_eurosat.py --kind rgb --output_dir ./data/custom/EuroSAT_RGB
python src/download_eurosat.py --kind ms --download_url https://example.com/EuroSAT_MS.zip
```

The default archives are hosted on
[Zenodo record 7711810](https://zenodo.org/records/7711810). The downloader
stores each ZIP beside its extraction directory. Downloaded data is not
included in Git; the `.gitignore` excludes raw, interim, and processed data
(the saved multispectral split manifest `data/processed/splits.csv` is kept
in Git on purpose so every run reuses the same split).

## Preprocess images

Convert the supported JPEG, PNG, and GeoTIFF files under an input directory to
resized, normalized `.npy` arrays while preserving their folder structure:

```bash
python src/preprocessing.py --input data/raw --output data/processed --size 64 --norm scale
```

Use `--norm minmax` to stretch each image band to 0-1, or `--correction dos` to
apply simple dark-object subtraction before resizing and normalization. The
default scaling uses 255 for 8-bit images and 10,000 for Sentinel-2 reflectance.

### Expected image-folder layout

`src/dataset.py` uses `torchvision.datasets.ImageFolder`. It expects the
directory passed as `data_dir` to contain one subdirectory per class, with
images inside each class directory:

```text
data/raw/EuroSAT_RGB/
  AnnualCrop/
    image_1.jpg
  Forest/
    image_2.jpg
  ...
```

Some archive versions may extract into an additional nested `EuroSAT_RGB/`
directory. If that happens, pass the nested class-folder directory as
`data_dir`.

## Multispectral pipeline (13-band)

The multispectral pipeline uses its own modules so the RGB utilities above
stay untouched: `src/preprocessing_ms.py`, `src/dataset_ms.py`, `src/train.py`,
and `src/evaluate.py`.

### Expected multispectral dataset layout

`src/dataset_ms.py` reads the 13-band GeoTIFFs using `rasterio`, or saved
NumPy `.npy`/`.npz` arrays. It accepts a dataset root or a containing
directory and searches nested directories for the class folders:

```text
data/raw/EuroSAT_MS/
  EuroSAT_MS/
    AnnualCrop/
      image_1.tif
    Forest/
      image_2.tif
    ...
```

If the archive extracts with an extra nested directory, `resolve_data_dir()`
finds the directory containing the class folders. Raw reflectance values are
divided by 10000 when loaded; values already in the 0-1 range are left
unchanged.

Preprocess the raw GeoTIFFs (optional; the loaders can also read `.tif`
directly):

```bash
python src/preprocessing_ms.py
```

### Train

```bash
python -u -m src.train --data-dir data/raw/EuroSAT_MS
```

Training creates `data/processed/splits.csv` on the first run and reuses it
afterward, so every teammate trains and evaluates on the same sample
assignments. The repository ships the `splits.csv` used for the recorded
baseline. The best checkpoint is selected by validation loss with early
stopping and saved to `outputs/baseline/best_model.pth` together with
`classes.json`, `metrics.csv`, and `metrics.json`.

Sanity-check the loaders and saved splits without training:

```bash
python src/test_dataset.py
```

### Evaluate

```bash
python -u -m src.evaluate
```

Evaluation loads the saved checkpoint and the same `splits.csv`, then writes
`evaluation_summary.json`, `classification_report.csv`, and a confusion
matrix under `outputs/evaluation/`. Plot the training curves with
`python src/plot_metrics.py`.

> **Baseline metrics:** the artifacts under `outputs/baseline/` and
> `outputs/evaluation/` were produced during earlier development on the
> previous working repository and were copied here for continuity. They were
> not re-trained or re-verified against this repository; treat the recorded
> numbers as historical until a teammate re-runs training locally.

## Explore the data

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open a notebook under `notebooks/`. The multispectral exploration covers
dataset structure and class counts, sample images, band metadata, per-class
spectral signatures, variability, and optional NDVI. The notebooks read local
dataset files and save selected results under `outputs/` and
`reports/figures/`.

EuroSAT RGB contains 27,000 labeled images across 10 classes. EuroSAT
multispectral contains the corresponding 13-band image data; the multispectral
notebook uses it to analyze class-level spectral responses. NDVI is calculated
from the red (`B04`) and near-infrared (`B08`) bands when the notebook's input
data supports that calculation.

## Python utilities

Run the reproducible 70/15/15 split summary and validation from the repository
root:

```bash
python src/split_data.py
```

The script writes train, validation, and test CSV manifests under
`data/processed/splits/seed_42/`; image paths are relative to the selected
ImageFolder root. It preserves the raw images and checks class balance and
split overlap. See [`docs/walaa_milestone1.md`](docs/walaa_milestone1.md) for
the measured class counts and contribution details.

These manifests are for the RGB ImageFolder dataset. They are separate from
the multispectral `data/processed/splits.csv` manifest used by the 13-band
pipeline.

Create train, validation, and test loaders from an ImageFolder-compatible
dataset:

```python
from src.dataset import get_loaders

train_loader, val_loader, test_loader, class_names = get_loaders(
    data_dir="data/raw/EuroSAT_RGB/EuroSAT_RGB",
    batch_size=64,
    seed=42,
)
```

The split is stratified by class (70% train, 15% validation, 15% test). Training
images receive random flips and rotation; validation and test images use only
tensor conversion and normalization. The loader returns the class names in
ImageFolder order.

Build one of the available models for the desired number of classes:

```python
from src.model import build_model

model = build_model("cnn", num_classes=len(class_names))
# Or use a ResNet-18 model:
model = build_model("resnet18", num_classes=len(class_names))
```

The legacy RGB training command retains its ResNet-18 weight-loading behavior.
The architecture comparison runner uses explicit scratch or pretrained
settings instead.

## RGB model architecture shortlist

The shortlist uses the same class mapping and held-out split for each
architecture:

- **Custom CNN (`cnn`)** reuses `SmallCNN` and supports both three-channel RGB
  and the existing 13-channel multispectral model API.
- **ResNet-18 (`resnet18`)** uses torchvision and supports scratch training or
  ImageNet initialization.
- **VGG-16 (`vgg16`)** reuses torchvision's convolutional feature layout with a
  configurable compact classifier instead of its large default fully
  connected layers.
- **Frozen transfer variants** (`resnet18-frozen`, `vgg16-frozen`) initialize
  from ImageNet and train only the replacement 10-class head.
- **Fine-tuning variants** (`resnet18-finetune`, `vgg16-finetune`) start from
  ImageNet weights and update the full model.

Pretrained runs resize inputs to 224×224 and use ImageNet normalization;
scratch runs retain the existing RGB transforms and native image size.
Scratch learning rate defaults to `0.001`; pretrained variants default to
`0.0001`. VGG classifier width and dropout can be set with
`--vgg-hidden-units` and `--vgg-dropout`.

Generate and validate the RGB split manifests first if you want comparisons
to reuse the saved sample assignments:

```bash
python src/split_data.py
```

By default, the experiment runner uses
`data/processed/splits/seed_<seed>/` when all three RGB manifests are present.
If they are absent, it creates the same deterministic, stratified 70/15/15
split from the selected seed. A custom manifest directory can be supplied
with `--split-manifest-dir`. Manifest paths and class labels are checked
against the selected ImageFolder dataset; the multispectral `splits.csv` is
never used for RGB.

Run the three scratch baselines:

```bash
python -m src.architecture_experiments --models cnn resnet18 vgg16 --epochs 5 --batch-size 64 --seed 42
```

Run transfer learning or fine-tuning:

```bash
python -m src.architecture_experiments --models resnet18-frozen vgg16-frozen --epochs 5 --batch-size 64 --seed 42
python -m src.architecture_experiments --models resnet18-finetune vgg16-finetune --epochs 5 --batch-size 64 --seed 42
```

For a quick smoke run, select one model and reduce the workload:

```bash
python -m src.architecture_experiments --models cnn --epochs 1 --batch-size 8 --seed 42
```

Each invocation creates a new timestamped directory under
`outputs/architecture_comparison/` with model checkpoints, epoch history,
confusion matrices, JSON metrics, and sorted CSV/JSON comparison tables.
Reported metrics include accuracy, macro and weighted precision/recall/F1,
training duration, parameter counts, and checkpoint size. The best
validation-loss checkpoint is evaluated on the held-out test split once.
Each architecture has a separate run in the existing
`eurosat-land-cover-classification` MLflow experiment. No model is identified
as best until full experiments have been run; smoke checks are not
experimental results.

### Verified RGB experiment results

The following runs used EuroSAT RGB, seed 42, a stratified 70/15/15 split
(18,900/4,050/4,050 samples), five epochs, and the checkpoint with the lowest
validation loss. Test metrics below are from that selected checkpoint's single
held-out evaluation. All times are CPU training-loop durations.

| Configuration | Best epoch | Validation accuracy / loss | Test accuracy / loss | Test macro F1 | Parameters (total / trainable) | Training time | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Custom CNN, scratch | 5 | 87.90% / 0.3609 | 87.70% / 0.3599 | 87.37% | 289,194 / 289,194 | 29.6 min | Complete |
| ResNet-18, scratch | 4 | 69.63% / 0.8733 | 70.81% / 0.8278 | 70.33% | 11,181,642 / 11,181,642 | 33.5 min | Complete |
| VGG-16, scratch | 5 | 64.84% / 0.9005 | 65.46% / 0.8919 | 59.78% | 14,982,474 / 14,982,474 | 128.8 min | Complete |
| ResNet-18, ImageNet frozen features | 5 | 92.30% / 0.2442 | 92.57% / 0.2357 | 92.20% | 11,181,642 / 5,130 | 82.6 min | Complete |

Validation metrics are from each run's best-validation-loss epoch; test
metrics are reported separately and were not used for checkpoint selection.
Per-class precision, recall, and F1, confusion matrices, epoch histories,
checkpoints, and full metric records are in each run directory under
`outputs/architecture_comparison/`; completed runs were also logged
individually to MLflow. The scratch outputs are in
`run_20261118T211441Z/`, and the completed frozen transfer run is in
`transfer_resnet18_frozen/run_20261119T004031Z/`.

The observed results favor ResNet-18 frozen transfer for predictive
performance on this split: it improved test accuracy by 4.86 percentage
points and macro F1 by 4.83 points over the CNN baseline, while training only
5,130 of its 11.18 million parameters. The CNN remains the compact option
(289 thousand parameters and a 1.12 MB checkpoint) and completed faster on
this CPU. These results cover one seed and five epochs; they do not establish
performance over repeated splits or longer training.

An additional ResNet-18 fine-tuning attempt saved a best-epoch-5 checkpoint at
`outputs/architecture_comparison/transfer_resnet18_finetune/run_20261119T020637Z/`,
but did not produce its history, test metrics, comparison output, or MLflow
run. It is an incomplete attempt, not a reported experiment result.

To reproduce the standard scratch comparison (with EuroSAT RGB available at
the default dataset path), first create the seed-42 RGB manifests and then run:

```bash
python src/split_data.py
python -m src.architecture_experiments --models cnn resnet18 vgg16 --epochs 5 --batch-size 64 --seed 42
```

To reproduce the completed frozen ResNet-18 configuration with a saved RGB
manifest:

```bash
python -m src.architecture_experiments --models resnet18-frozen --epochs 5 --batch-size 8 --learning-rate 0.0001 --seed 42 --split-manifest-dir data/processed/splits/seed_42
```

Pretrained runs resize images to 224×224 and use ImageNet normalization. Run
directories are timestamped, so reruns create new outputs and MLflow runs
rather than replacing the historical results above.

## Train and track a model (RGB)

Download and extract EuroSAT RGB if it is not already available:

```bash
python src/download_eurosat.py
```

Train the default CNN for five epochs using the existing stratified data
loaders. The script saves the checkpoint under `models/` and logs the final
training, validation, and test metrics to a local MLflow experiment:

```bash
python train.py
```

Optional arguments let you select another supported model or adjust the
training settings:

```bash
python train.py --model cnn --epochs 5 --batch-size 64 --learning-rate 0.001 --seed 42
```

MLflow uses the project-local `mlruns/` directory; it does not require a
remote tracking server. The tracking module opts in to MLflow's local file
store for recent MLflow versions. Install dependencies with
`pip install -r requirements.txt`, then launch the local results browser:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000` to view the `eurosat-land-cover-classification`
experiment, its recorded parameters and metrics, and the saved checkpoint
artifact. The proposed prediction API is described in
[`API_DESIGN.md`](API_DESIGN.md); no API server is included.

## Tests

Run the focused project tests from the repository root:

```bash
python -m pytest
```

Architecture tests exercise synthetic one-batch training, model outputs,
split and preprocessing behavior, evaluation metrics, checkpoint artifacts,
and MLflow calls without claiming full-dataset performance.

## Outputs

The repository includes example exploration outputs, such as spectral
signature and NDVI tables and plots, plus the recorded multispectral baseline
and evaluation artifacts described above. They are illustrative results
already stored in the repository; re-running the notebooks or scripts may
replace or regenerate them from the local dataset.

## Known limitations

- The repository root still contains two historical git links
  (`satellite_empty` and `NHA-5-151`) left over from earlier development.
  They are not submodules anyone should initialize, and they cannot be
  removed through the GitHub web interface; ignore them until a maintainer
  removes them with git locally.
- The recorded baseline metrics were copied from the previous development
  repository (see the note under "Evaluate") and are not freshly verified
  here.

## Data and citation

EuroSAT is a land-use and land-cover dataset derived from Sentinel-2 satellite
imagery. See the
[EuroSAT project page](https://github.com/phelber/EuroSAT) and the
[Zenodo dataset record](https://zenodo.org/records/7711810) for dataset
information and citation details. Review and follow the dataset's terms of use
when downloading or redistributing it.
