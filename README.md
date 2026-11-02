# Land Type Classification with Sentinel-2 Satellite Imagery

Project 6 for the Digital Egypt Pioneers Initiative (DEPI), AI & Data Science
Track, Round 2.

This repository contains data exploration notebooks and starter utilities for
land-cover classification with the EuroSAT dataset. It includes RGB and
multispectral dataset download support, image preprocessing, class-stratified
PyTorch data loaders, and CNN/ResNet-18 model definitions.

> **Project status:** The repository currently provides EuroSAT download,
> exploration, and model-building components. The Sentinel-2 AOI definitions
> are configuration only; automated Sentinel-2 scene search/download and a
> complete model training/evaluation pipeline are not implemented here.

## Contents

| Path | Description |
| --- | --- |
| `src/aoi_config.py` | Example Egypt areas of interest, Sentinel-2 band groups, land-cover labels, and scene-query preferences |
| `src/download_eurosat.py` | Download and extract the EuroSAT RGB or multispectral archive |
| `src/preprocessing.py` | Resize and normalize RGB or multispectral images into `.npy` files |
| `src/dataset.py` | ImageFolder-based train, validation, and test data loaders |
| `src/split_data.py` | Run and validate the stratified split; write CSV manifests |
| `src/model.py` | `SmallCNN` and `resnet18` model builders |
| `src/tracking.py` | Log training parameters, metrics, and artifacts to local MLflow |
| `train.py` | Train and evaluate the EuroSAT image classifier |
| `API_DESIGN.md` | Proposed prediction API endpoints and payload formats |
| `notebooks/initial_inspection.ipynb` | Initial dataset inspection notebook |
| `notebooks/01_data_exploration_eurosat_ms.ipynb` | Multispectral exploration, spectral signatures, and optional NDVI analysis |
| `notebooks/data_collection.ipynb` | Notebook for the EuroSAT download and project workflow |
| `outputs/` | Example exported tables and charts from dataset exploration |
| `reports/figures/` | Example report figures |
| `tests/` | Focused tests for the downloader and model builder |

## Requirements

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)
- Jupyter Notebook or JupyterLab to run the notebooks

The multispectral exploration notebook may also require `rasterio` to read the
GeoTIFF images. Install it in your environment if you run the notebook cells
that inspect multispectral image pixels. `rasterio` is also needed when
preprocessing GeoTIFF files; it is optional for JPEG and PNG input.

## Setup

Create and activate a virtual environment, then install the project
dependencies.

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook rasterio
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook rasterio
```

`notebook` and `rasterio` are for interactive notebook work and are not listed
as core application dependencies in `requirements.txt`.

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
included in Git; the `.gitignore` excludes raw, interim, and processed data.

## Preprocess images

Convert the supported JPEG, PNG, and GeoTIFF files under an input directory to
resized, normalized `.npy` arrays while preserving their folder structure:

```bash
python src/preprocessing.py --input data/raw --output data/processed --size 64 --norm scale
```

Use `--norm minmax` to stretch each image band to 0–1, or `--correction dos` to
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

The ResNet-18 builder attempts to load ImageNet weights and falls back to
random initialization if the weights cannot be loaded. These utilities do not
train a model on their own; use the training command below for a simple CNN
training and evaluation run.

## Train and track a model

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

## Outputs

The repository includes example exploration outputs, such as spectral
signature and NDVI tables and plots. They are illustrative results already
stored in the repository; re-running the notebooks may replace or regenerate
them from the local dataset.

## Data and citation

EuroSAT is a land-use and land-cover dataset derived from Sentinel-2 satellite
imagery. See the
[EuroSAT project page](https://github.com/phelber/EuroSAT) and the
[Zenodo dataset record](https://zenodo.org/records/7711810) for dataset
information and citation details. Review and follow the dataset's terms of use
when downloading or redistributing it.
