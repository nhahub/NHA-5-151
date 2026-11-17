"""Tests for src/compute_band_correlation.py with planted-signal fixtures.

The fixtures build constant-band images so each image's per-band mean is an
exact planted value. The script's aggregate output is then compared against
an INDEPENDENT reference (numpy.corrcoef over the planted means), and the
train/all split modes are checked for exclusion and disjointness — not just
output existence.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import src.compute_band_correlation as cbc
from src.dataset_ms import discover_samples, save_splits

BANDS = 13
SIZE = 4
TEST_CLASSES = ["AnnualCrop", "Forest"]
N_TRAIN, N_VAL, N_TEST = 20, 6, 6


def _constant_image(band_values):
    return np.stack(
        [np.full((SIZE, SIZE), v, dtype=np.float32) for v in band_values]
    )


def _plant_means(rng, n, anticorrelate=False):
    """Return (n, BANDS) planted per-image band-mean values."""
    latent = rng.normal(size=n)
    means = rng.normal(loc=2000.0, scale=300.0, size=(n, BANDS))
    means[:, 0] = 1000.0 + 100.0 * latent
    sign = -1.0 if anticorrelate else 1.0
    means[:, 1] = (
        1500.0 + sign * 50.0 * latent + rng.normal(scale=0.001, size=n)
    )
    return means


@pytest.fixture()
def planted_dataset(tmp_path):
    """Synthetic dataset + persisted split; train and eval splits carry
    OPPOSITE planted correlations between B01 and B02."""
    rng = np.random.default_rng(0)
    root = tmp_path / "EuroSAT_MS"
    planted = {}
    for class_name in TEST_CLASSES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        parts = {
            "train": _plant_means(rng, N_TRAIN),
            "val": _plant_means(rng, N_VAL, anticorrelate=True),
            "test": _plant_means(rng, N_TEST, anticorrelate=True),
        }
        planted[class_name] = parts
        for split, rows in parts.items():
            for i, row in enumerate(rows):
                np.save(
                    class_dir / f"{class_name}_{split}_{i}.npy",
                    _constant_image(row),
                )

    samples, _ = discover_samples(root)
    indices = {"train": [], "val": [], "test": []}
    for idx, (path, _label, _cls) in enumerate(samples):
        stem = path.stem  # e.g. AnnualCrop_train_3
        indices[stem.rsplit("_", 2)[1]].append(idx)
    splits_path = tmp_path / "splits.csv"
    save_splits(
        samples,
        indices["train"],
        indices["val"],
        indices["test"],
        splits_path,
        data_dir=root,
    )
    return root, splits_path, planted


def _run_script(monkeypatch, tmp_path, *argv):
    tables = tmp_path / "out_tables"
    figures = tmp_path / "out_figures"
    monkeypatch.setattr(cbc, "TABLES_DIR", tables)
    monkeypatch.setattr(cbc, "FIGURES_DIR", figures)
    monkeypatch.setattr(
        sys, "argv", ["compute_band_correlation.py", *argv]
    )
    cbc.main()
    with (tables / "dataset_band_correlation.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.reader(file))
    band_names = rows[0][1:]
    corr = np.array(
        [[float(v) for v in row[1:]] for row in rows[1:]]
    )
    summary_rows = list(
        csv.reader(
            (tables / "dataset_band_correlation_summary.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )[1:]  # skip header
    summary = {key: json.loads(value) for key, value in summary_rows}
    return band_names, corr, summary


def _reference_corr(planted, classes, splits):
    rows = np.concatenate(
        [planted[c][s] for c in classes for s in splits], axis=0
    )
    return np.corrcoef(rows, rowvar=False)


def test_train_mode_recovers_planted_reference_exactly(
    planted_dataset, monkeypatch, tmp_path
):
    root, splits_path, planted = planted_dataset
    band_names, corr, summary = _run_script(
        monkeypatch,
        tmp_path,
        "--data-dir", str(root),
        "--splits-path", str(splits_path),
    )
    reference = _reference_corr(planted, TEST_CLASSES, ["train"])
    assert band_names == list(cbc.SENTINEL2_BAND_ORDER)
    # CSV is written with 6 decimal places; compare at that precision.
    np.testing.assert_allclose(corr, reference, atol=1e-5)
    # planted train signal: B01 and B02 are strongly positively correlated
    assert corr[0, 1] > 0.99
    assert summary["split_mode"] == "train"
    assert int(summary["images_used"]) == len(TEST_CLASSES) * N_TRAIN


def test_train_mode_excludes_val_and_test(planted_dataset, monkeypatch, tmp_path):
    root, splits_path, planted = planted_dataset
    _, corr_train, _ = _run_script(
        monkeypatch, tmp_path, "--data-dir", str(root),
        "--splits-path", str(splits_path),
    )
    _, corr_all, summary_all = _run_script(
        monkeypatch, tmp_path, "--data-dir", str(root), "--split", "all",
    )
    # eval splits carry the OPPOSITE sign; mixing them must measurably
    # weaken the B01-B02 correlation, proving train mode excluded them.
    assert corr_all[0, 1] < corr_train[0, 1] - 0.2
    assert "REPORT-ONLY" in summary_all["split_disclosure"]
    assert int(summary_all["images_used"]) == len(TEST_CLASSES) * (
        N_TRAIN + N_VAL + N_TEST
    )


def test_train_mode_fails_clearly_without_split_file(
    planted_dataset, monkeypatch, tmp_path
):
    root, _splits_path, _ = planted_dataset
    tables = tmp_path / "t"
    figures = tmp_path / "f"
    monkeypatch.setattr(cbc, "TABLES_DIR", tables)
    monkeypatch.setattr(cbc, "FIGURES_DIR", figures)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_band_correlation.py",
            "--data-dir", str(root),
            "--splits-path", str(tmp_path / "missing.csv"),
        ],
    )
    with pytest.raises(FileNotFoundError, match="split file"):
        cbc.main()


def test_train_mode_fails_clearly_on_stale_split(
    planted_dataset, monkeypatch, tmp_path
):
    root, splits_path, _ = planted_dataset
    # remove one image so the persisted split no longer covers the dataset
    victim = next((root / "AnnualCrop").glob("*.npy"))
    victim.unlink()
    monkeypatch.setattr(cbc, "TABLES_DIR", tmp_path / "t")
    monkeypatch.setattr(cbc, "FIGURES_DIR", tmp_path / "f")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_band_correlation.py",
            "--data-dir", str(root),
            "--splits-path", str(splits_path),
        ],
    )
    with pytest.raises(ValueError, match="Split file"):
        cbc.main()
