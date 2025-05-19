"""
from PlanetConstants import (
    # Core Dependencies
    TEXCONV_PATH,
    # Core directories
    BASE_DIR,
    CONFIG_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    ASSETS_DIR,
    CSV_DIR,
    IMAGE_DIR,
    PNG_OUTPUT_DIR,

    # Config and data files
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    CSV_PATH,
    PREVIEW_PATH,

    # Script and template paths
    SCRIPT_PATH,
    TEMPLATE_PATH,
    PREVIEW_BIOME_PATH,

    # UI and static assets
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    IMAGE_FILES,

    # Logic/data maps
    BOOLEAN_KEYS,
    PROCESSING_MAP,
)
"""

import sys
import json
from pathlib import Path

# --- Core directories ---
bundle_dir = getattr(sys, "_MEIPASS", None)
if bundle_dir:
    BASE_DIR = Path(bundle_dir).resolve()
else:
    BASE_DIR = Path(__file__).resolve().parent.parent


SCRIPT_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
CSV_DIR = BASE_DIR / "csv"
IMAGE_DIR = ASSETS_DIR / "images"
TEXTURE_OUTPUT_DIR = OUTPUT_DIR / "textures"
PNG_OUTPUT_DIR = TEXTURE_OUTPUT_DIR / "PNGs"

# --- Config and data files ---
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"

THEME_PATH = CONFIG_DIR / "custom_themes.json"
DEFAULT_THEME_PATH = CONFIG_DIR / "default_themes.json"

CSV_PATH = CSV_DIR / "Biomes.csv"
PREVIEW_PATH = CSV_DIR / "preview.csv"

# --- Script and template paths ---
SCRIPT_PATH = SCRIPT_DIR / "PlanetBiomes.py"
TEMPLATE_PATH = ASSETS_DIR / "PlanetBiomes.biom"
PREVIEW_BIOME_PATH = ASSETS_DIR / "PlanetBiomes.biom"

# --- UI and static assets ---
UI_PATH = SCRIPT_DIR / "mainwindow.ui"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "default.png"
TEXCONV_PATH = BASE_DIR / "textconv" / "texconv.exe"

GIF_PATHS = {
    1: IMAGE_DIR / "progress_1.gif",
    2: IMAGE_DIR / "progress_2.gif",
    3: IMAGE_DIR / "progress_3.gif",
}

IMAGE_FILES = [
    "preview_North_albedo.png",
    "preview_North_normal.png",
    "preview_North_rough.png",
    "preview_North_alpha.png",
]

# --- Configuration flags ---
BOOLEAN_KEYS = {
    "enable_equator_drag",
    "enable_pole_drag",
    "enable_equator_intrusion",
    "enable_pole_intrusion",
    "apply_distortion",
    "apply_resource_gradient",
    "apply_latitude_blending",
    "keep_pngs_after_conversion",
    "enable_noise",
    "enable_anomalies",
    "enable_biases",
    "use_random",
    "enable_texture_light",
    "enable_texture_edges",
    "enable_basic_filters",
    "enable_texture_anomalies",
    "process_images",
    "enable_texture_noise",
    "upscale_image",
    "enable_texture_preview",
    "output_csv_files",
    "output_dds_files",
    "output_mat_files",
    "output_biom_files",
    "enable_random_drag",
    "random_distortion",
}

# --- Textual progress mapping ---
PROCESSING_MAP = {
    "Permits closed, loan secured.": [0],
    "Terraforming complete.": [1],
    "Ore distributed.": [2],
    "Landscaping complete.": [3],
    "don't panic!": [0, 1, 2, 3],
}

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)
