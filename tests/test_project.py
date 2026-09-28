from pathlib import Path
import zipfile

import src.download_eurosat as download_eurosat
from src.model import build_model


def test_download_supports_output_dir_and_download_url(tmp_path, monkeypatch):
    calls = {}

    def fake_urlretrieve(url, destination):
        calls["url"] = url
        calls["destination"] = destination
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

    class DummyZip:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, target_dir):
            calls["target_dir"] = target_dir
            Path(target_dir, "dummy.txt").write_text("ok")

    monkeypatch.setattr(download_eurosat, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(zipfile, "ZipFile", DummyZip)

    output_dir = tmp_path / "custom" / "EuroSAT_RGB"
    url = "https://example.com/EuroSAT_RGB.zip"
    result = download_eurosat.download("rgb", output_dir=output_dir, url=url)

    assert result == output_dir
    assert calls["url"] == url
    assert calls["destination"] == str(output_dir.parent / "EuroSAT_RGB.zip")
    assert Path(calls["target_dir"], "dummy.txt").exists()


def test_build_model_cnn_works():
    model = build_model("cnn", 7)
    assert model.__class__.__name__ == "SmallCNN"
    assert model.head[-1].out_features == 7
