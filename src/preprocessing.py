"""
Preprocess RGB and multispectral imagery into resized, normalized NumPy arrays.

GeoTIFF input requires the optional ``rasterio`` package.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:  # Only needed for GeoTIFF input.
    rasterio = None

EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def load_image(path: Path):
    """Return an image as an H x W x C float32 array and its scaling maximum."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError("Install rasterio first: pip install rasterio")
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)
        return np.transpose(image, (1, 2, 0)), 10000.0

    with Image.open(path) as source:
        image = np.array(source.convert("RGB"), dtype=np.float32)
    return image, 255.0


def dark_object_subtraction(img: np.ndarray, percentile: float = 1.0):
    """Subtract the darkest values in each band as a simple haze correction."""
    dark = np.percentile(img.reshape(-1, img.shape[-1]), percentile, axis=0)
    return np.clip(img - dark, 0, None)


def resize_image(img: np.ndarray, size: int):
    """Resize each band to a square image of the requested size."""
    height, width = img.shape[:2]
    interpolation = cv2.INTER_AREA if size < min(height, width) else cv2.INTER_LINEAR
    bands = [
        cv2.resize(img[:, :, band], (size, size), interpolation=interpolation)
        for band in range(img.shape[2])
    ]
    return np.stack(bands, axis=-1)


def normalize(img: np.ndarray, scale: float, method: str):
    """Normalize pixel values using source scaling or per-image min-max stretch."""
    if method == "scale":
        return np.clip(img / scale, 0.0, 1.0)
    if method == "minmax":
        minimum = img.min(axis=(0, 1), keepdims=True)
        maximum = img.max(axis=(0, 1), keepdims=True)
        return (img - minimum) / (maximum - minimum + 1e-8)
    raise ValueError(f"unknown normalization method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--correction", choices=["none", "dos"], default="none")
    parser.add_argument("--norm", choices=["scale", "minmax"], default="scale")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    files = [path for path in input_dir.rglob("*") if path.suffix.lower() in EXTS]
    if not files:
        print(f"No images found in {input_dir}")
        return

    for index, path in enumerate(files, 1):
        image, scale = load_image(path)
        if args.correction == "dos":
            image = dark_object_subtraction(image)
        image = resize_image(image, args.size)
        image = normalize(image, scale, args.norm).astype(np.float32)

        output_path = output_dir / path.relative_to(input_dir).with_suffix(".npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, image)
        if index % 500 == 0:
            print(f"{index}/{len(files)} done")

    print(f"Finished: {len(files)} images -> {output_dir}")


if __name__ == "__main__":
    main()
