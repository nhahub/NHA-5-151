import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from src import preprocessing  # noqa: E402


def write_tif(path, array):
    """Write a (bands, height, width) array as a GeoTIFF."""
    bands, height, width = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
    ) as dst:
        dst.write(array.astype(np.float32))


def test_load_image_returns_height_width_channels(tmp_path):
    path = tmp_path / "sample.tif"
    write_tif(path, np.full((13, 8, 10), 500, dtype=np.float32))

    image = preprocessing.load_image(path)

    assert image.shape == (8, 10, 13)
    assert image.dtype == np.float32


def test_resize_changes_size_and_keeps_channels():
    image = np.ones((8, 10, 13), dtype=np.float32)

    resized = preprocessing.resize(image, 4)

    assert resized.shape == (4, 4, 13)


def test_resize_returns_same_image_when_size_matches():
    image = np.ones((4, 4, 3), dtype=np.float32)

    resized = preprocessing.resize(image, 4)

    assert resized.shape == (4, 4, 3)
    np.testing.assert_array_equal(resized, image)


def test_normalize_scales_and_clips():
    image = np.array([[[5000.0, 20000.0, -10.0]]], dtype=np.float32)

    normalized = preprocessing.normalize(image)

    np.testing.assert_allclose(normalized[0, 0], [0.5, 1.0, 0.0])


def test_remove_haze_subtracts_and_never_goes_negative():
    image = np.array([[[10.0, 3.0]]], dtype=np.float32)
    haze = np.array([4.0, 5.0], dtype=np.float32)

    result = preprocessing.remove_haze(image, haze)

    np.testing.assert_allclose(result[0, 0], [6.0, 0.0])


def test_find_haze_returns_one_value_per_band(tmp_path):
    path = tmp_path / "a.tif"
    write_tif(path, np.full((13, 8, 8), 100, dtype=np.float32))

    haze = preprocessing.find_haze([path])

    assert haze.shape == (13,)
    np.testing.assert_allclose(haze, 100)


def test_preprocess_output_shape_dtype_and_range(monkeypatch):
    monkeypatch.setattr(preprocessing, "IMAGE_SIZE", 4)
    image = np.full((8, 8, 13), 5000, dtype=np.float32)

    result = preprocessing.preprocess(image)

    assert result.shape == (4, 4, 13)
    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_main_saves_npy_in_matching_class_folder(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    write_tif(
        input_dir / "Forest" / "sample.tif",
        np.full((13, 8, 8), 5000, dtype=np.float32),
    )
    monkeypatch.setattr(preprocessing, "INPUT_FOLDER", str(input_dir))
    monkeypatch.setattr(preprocessing, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(preprocessing, "IMAGE_SIZE", 4)
    monkeypatch.setattr(preprocessing, "REMOVE_HAZE", False)

    preprocessing.main()

    output_path = output_dir / "Forest" / "sample.npy"
    assert output_path.exists()
    processed = np.load(output_path)
    assert processed.shape == (4, 4, 13)
    assert processed.dtype == np.float32
    np.testing.assert_allclose(processed, 0.5)