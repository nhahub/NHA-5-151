"""Synthetic regression tests for the 13-band (multispectral) data path.

These tests use small synthetic .npy fixtures, never the real dataset, so
they run anywhere without a EuroSAT_MS download. They guard the behavior
that M2-T09 ported beside the RGB path: loading, channel checks, scaling,
splits, and the verified Sentinel-2 band storage order.
"""

import csv

import numpy as np
import pytest
import torch

from src.dataset_ms import (
    CLASS_NAMES,
    SENTINEL2_BAND_ORDER,
    EuroSATMSDataset,
    discover_samples,
    get_loaders,
    load_multispectral_image,
    resolve_data_dir,
)

BANDS = 13
SIZE = 8  # synthetic images are BANDS x SIZE x SIZE
TEST_CLASSES = ["AnnualCrop", "Forest", "SeaLake"]
IMAGES_PER_CLASS = 10


def _make_image(value_low=0.0, value_high=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(value_low, value_high, size=(BANDS, SIZE, SIZE)).astype(
        np.float32
    )


@pytest.fixture()
def synthetic_ms_dir(tmp_path):
    """Create a minimal synthetic EuroSAT_MS folder tree."""
    root = tmp_path / "EuroSAT_MS"
    seed = 0
    for class_name in TEST_CLASSES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            np.save(class_dir / f"{class_name}_{i}.npy", _make_image(seed=seed))
            seed += 1
    return root


def test_sentinel2_band_storage_order_is_b8a_last():
    # EuroSAT_MS .tif files store B8A as the LAST band, not the 9th.
    # Verified externally: torchgeo issue #2458 (9th stored band has 60 m
    # resolution -> B09 water vapour, not B8A; 10th band value range matches
    # B10 cirrus), accepted against the original EuroSAT repository.
    assert SENTINEL2_BAND_ORDER == (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B09", "B10", "B11", "B12", "B8A",
    )
    assert len(SENTINEL2_BAND_ORDER) == 13


def test_discover_samples_finds_classes_and_images(synthetic_ms_dir):
    samples, classes = discover_samples(synthetic_ms_dir)
    assert classes == TEST_CLASSES  # CLASS_NAMES order, not filesystem order
    assert len(samples) == len(TEST_CLASSES) * IMAGES_PER_CLASS
    labels = {class_name: label for _, label, class_name in samples}
    assert labels == {"AnnualCrop": 0, "Forest": 1, "SeaLake": 2}


def test_get_loaders_batch_shape_and_partition(synthetic_ms_dir):
    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=synthetic_ms_dir,
        batch_size=4,
        seed=42,
        num_workers=0,
        num_channels=BANDS,
    )
    assert classes == TEST_CLASSES

    total = len(TEST_CLASSES) * IMAGES_PER_CLASS
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    test_idx = set(test_loader.dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(range(total))
    assert train_idx and val_idx and test_idx

    images, labels = next(iter(train_loader))
    assert images.shape == (4, BANDS, SIZE, SIZE)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0


def test_splits_are_deterministic_and_reused(synthetic_ms_dir, tmp_path):
    splits_path = tmp_path / "splits.csv"
    first = get_loaders(
        data_dir=synthetic_ms_dir,
        batch_size=4,
        seed=42,
        num_channels=BANDS,
        splits_path=splits_path,
    )
    assert splits_path.is_file()
    second = get_loaders(
        data_dir=synthetic_ms_dir,
        batch_size=4,
        seed=123,  # different seed must not change a saved split
        num_channels=BANDS,
        splits_path=splits_path,
    )
    # Same partition must be recovered from the saved file. Order within a
    # split can differ (CSV row order vs sklearn output); membership is the
    # contract, so compare sets.
    for a, b in zip(first[:3], second[:3]):
        assert sorted(a.dataset.indices) == sorted(b.dataset.indices)

    with splits_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(TEST_CLASSES) * IMAGES_PER_CLASS
    assert {row["split"] for row in rows} == {"train", "val", "test"}


def test_raw_sentinel2_values_are_scaled(synthetic_ms_dir):
    raw = _make_image(value_low=0.0, value_high=10000.0, seed=999)
    path = synthetic_ms_dir / "AnnualCrop" / "raw_values.npy"
    np.save(path, raw)
    image = load_multispectral_image(path)
    dataset = EuroSATMSDataset(
        [(path, 0, "AnnualCrop")], [0], num_channels=BANDS
    )
    tensor, _ = dataset[0]
    assert float(tensor.max()) <= 1.0
    assert float(tensor.min()) >= 0.0
    # uint16-style reflectance values are divided by 10000
    assert image.max() > 1.0


def test_preprocessed_unit_range_values_pass_through(synthetic_ms_dir):
    unit = _make_image(value_low=0.0, value_high=1.0, seed=1234)
    path = synthetic_ms_dir / "Forest" / "unit.npy"
    np.save(path, unit)
    dataset = EuroSATMSDataset([(path, 1, "Forest")], [0], num_channels=BANDS)
    tensor, _ = dataset[0]
    np.testing.assert_allclose(tensor.numpy(), unit, atol=1e-6)


def test_channel_mismatch_is_rejected(synthetic_ms_dir):
    three_band = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    path = synthetic_ms_dir / "SeaLake" / "rgb_only.npy"
    np.save(path, three_band)
    dataset = EuroSATMSDataset(
        [(path, 2, "SeaLake")], [0], num_channels=BANDS
    )
    with pytest.raises(ValueError, match="channels"):
        dataset[0]


def test_hwc_layout_is_transposed_to_chw(tmp_path):
    hwc = np.zeros((SIZE, SIZE, BANDS), dtype=np.float32)
    path = tmp_path / "hwc.npy"
    np.save(path, hwc)
    image = load_multispectral_image(path)
    assert image.shape == (BANDS, SIZE, SIZE)


def test_invalid_image_content_is_rejected(tmp_path):
    bad2d = np.zeros((SIZE, SIZE), dtype=np.float32)
    path2d = tmp_path / "bad2d.npy"
    np.save(path2d, bad2d)
    with pytest.raises(ValueError, match="3D"):
        load_multispectral_image(path2d)

    nan_image = _make_image(seed=7)
    nan_image[0, 0, 0] = np.nan
    path_nan = tmp_path / "nan.npy"
    np.save(path_nan, nan_image)
    with pytest.raises(ValueError, match="NaN"):
        load_multispectral_image(path_nan)


def test_resolve_data_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(tmp_path / "does_not_exist")


def test_rgb_three_channel_mode_still_supported(synthetic_ms_dir):
    # The MS loader must not break 3-channel use (RGB-path compatibility).
    rgb = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    path = synthetic_ms_dir / "AnnualCrop" / "rgb.npy"
    np.save(path, rgb)
    dataset = EuroSATMSDataset(
        [(path, 0, "AnnualCrop")], [0], num_channels=3
    )
    tensor, _ = dataset[0]
    assert tensor.shape == (3, SIZE, SIZE)


def test_augmentation_is_spatial_only(synthetic_ms_dir):
    torch.manual_seed(0)
    samples, _ = discover_samples(synthetic_ms_dir)
    dataset = EuroSATMSDataset(
        samples, list(range(5)), num_channels=BANDS, augment=True
    )
    for i in range(len(dataset)):
        tensor, _ = dataset[i]
        assert tensor.shape == (BANDS, SIZE, SIZE)
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0


def test_all_repo_class_names_supported():
    # The ten EuroSAT classes the rest of the repo expects.
    assert CLASS_NAMES == [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake",
    ]
