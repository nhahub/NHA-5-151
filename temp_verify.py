import importlib
from pathlib import Path

mods = ['src.download_eurosat', 'src.model', 'src.aoi_config']
for name in mods:
    mod = importlib.import_module(name)
    print(name, 'OK')

from src.model import build_model
m = build_model('cnn', 7)
print(type(m).__name__, m.head[-1].out_features)

from src.download_eurosat import download
root = Path('D:/My Projects/satellite_empty')
res = download('rgb', output_dir=root / 'data' / 'raw' / 'EuroSAT_RGB_test', url='https://example.com/EuroSAT_RGB.zip')
print(res)
