"""
Area of Interest (AOI) definitions for the Sentinel-2 Land Type Classification project.

Each AOI is a small bounding box over a region of Egypt chosen because it contains
a mix of the target land types (agriculture, urban, water, desert, roads, trees).
Bounding boxes are (min_lon, min_lat, max_lon, max_lat) in WGS84.

Mix rationale:
- Nile Delta (Mansoura/Damietta area): dense agriculture + water + urban + trees
- Cairo metro: urban + roads + some desert fringe
- Aswan / Nile at Aswan: water (High Dam lake), desert, some agriculture
- Fayoum: agriculture, desert, water (Qarun Lake) — classic mixed-class oasis
"""

AOIS = {
    "nile_delta_mansoura": {
        "bbox": (31.35, 31.03, 31.45, 31.11),
        "description": "Nile Delta near Mansoura — dense agriculture, canals, urban, trees",
    },
    "cairo_metro": {
        "bbox": (31.20, 29.98, 31.32, 30.08),
        "description": "Greater Cairo — urban core, roads, desert fringe to the east",
    },
    "aswan_nile": {
        "bbox": (32.85, 24.02, 32.95, 24.10),
        "description": "Aswan / Nile corridor — water, narrow agriculture strip, desert",
    },
    "fayoum_oasis": {
        "bbox": (30.75, 29.35, 30.85, 29.43),
        "description": "Fayoum oasis — agriculture, desert, Qarun Lake water",
    },
}

# Sentinel-2 bands relevant for land classification (10m/20m resampled to 10m)
BANDS_10M = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]  # Red edge, NIR narrow, SWIR

# Land type classes for this project (adjust to match what you can label in your AOIs)
LAND_CLASSES = ["agriculture", "water", "urban", "desert", "road", "trees"]

# Cloud cover threshold when querying the catalog
MAX_CLOUD_COVER = 15  # percent

# Preferred date range (choose a dry/clear season for Egypt — winter has less haze)
DATE_RANGE = ("2024-11-01", "2025-02-28")
