import numpy as np
from PIL import Image

from src.preprocessing import (
    dark_object_subtraction,
    load_image,
    main,
    normalize,
    resize_image,
)


def test_cli_saves_preprocessed_image_in_matching_folder(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    image_path = input_dir / "Forest" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((8, 10, 3), 128, dtype=np.uint8)).save(image_path)
    output_dir = tmp_path / "processed"
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocessing.py",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--size",
            "4",
        ],
    )

    main()

    output_path = output_dir / "Forest" / "sample.npy"
    assert output_path.exists()
    processed = np.load(output_path)
    assert processed.shape == (4, 4, 3)
    assert processed.dtype == np.float32


def test_load_resize_and_scale_rgb_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray(np.full((8, 10, 3), 128, dtype=np.uint8)).save(path)

    image, scale = load_image(path)
    resized = resize_image(image, 4)
    normalized = normalize(resized, scale, "scale")

    assert resized.shape == (4, 4, 3)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, 128 / 255)


def test_minmax_normalization_and_dark_object_subtraction():
    image = np.array(
        [[[2.0, 4.0], [4.0, 8.0]], [[6.0, 12.0], [8.0, 16.0]]],
        dtype=np.float32,
    )

    corrected = dark_object_subtraction(image, percentile=0)
    normalized = normalize(corrected, scale=1, method="minmax")

    np.testing.assert_allclose(corrected.min(axis=(0, 1)), 0)
    np.testing.assert_allclose(normalized.min(axis=(0, 1)), 0)
    np.testing.assert_allclose(normalized.max(axis=(0, 1)), 1)
