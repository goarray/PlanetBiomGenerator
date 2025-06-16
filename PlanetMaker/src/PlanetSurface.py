from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import Dict, List, Set, Tuple, NamedTuple, cast
import colorsys
import argparse
import subprocess
import json
import csv
import random
import sys
import shutil
from PIL import Image, ImageEnhance
from PlanetNewsfeed import handle_news
from PlanetUtils import biome_db
from PlanetConstants import (
    get_config,
    # Core Dependencies
    TEXCONV_PATH,
    # Core directories
    BASE_DIR,
    CONFIG_DIR,
    INPUT_DIR,
    BIOM_DIR,  # BIOM_DIR = "planetdata/biomemaps"
    # plugin_name in _congif.json > "plugin_name": "preview.esm"
    OUTPUT_DIR,
    TEMP_DIR,
    ASSETS_DIR,
    SCRIPT_DIR,
    PLUGINS_DIR,  # PLUGINS_DIR = BASE_DIR / "Plugins"
    CSV_DIR,
    IMAGE_DIR,
    DDS_OUTPUT_DIR,
    PNG_OUTPUT_DIR,
    # Config and data files
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    CSV_PATH,
    PREVIEW_PATH,
    BLOCK_PATTERN_PATH,
    # Script and template paths
    SCRIPT_PATH,
    TEMPLATE_PATH,
    # UI and static assets
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    IMAGE_FILES,
    # Logic/data maps
    BOOLEAN_KEYS,
    PROCESSING_MAP,
)


# Grid constants
GRID_SIZE = [256, 256]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

# Global configuration
config = get_config()
plugin_name = config.get("plugin_name", "default_plugin")
planet_name = config.get("planet_name", "default_planet")


def load_biome_colors(csv_path, used_biome_ids, saturate_factor=None):
    """Load RGB colors for used biome IDs from CSV."""
    if saturate_factor is None:
        saturate_factor = config.get("texture_saturation", 0.29)

    if not isinstance(saturate_factor, float):
        raise TypeError(f"saturate_factor must be a float, got {type(saturate_factor)}")

    biome_colors = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                form_id = int(row[0], 16)
                r, g, b = int(row[2]), int(row[3]), int(row[4])
            except (ValueError, IndexError):
                print(f"Warning: Invalid row in Biomes.csv: {row}. Skipping.")

    return biome_colors


def load_block_patterns_by_category(csv_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Load and group block patterns by biome category."""
    patterns_by_category = {}

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            try:
                form_id = row[0].strip()
                editor_id = row[1].strip()
                category = row[2].strip().lower()

                if category not in patterns_by_category:
                    patterns_by_category[category] = []

                patterns_by_category[category].append((form_id, editor_id))

            except IndexError:
                print(f"[Warning] Skipping malformed row in BlockPatterns.csv: {row}")

    return patterns_by_category


def load_biom_file(png_output_dir: Path, planet_name: str) -> np.ndarray:
    biome_path = png_output_dir / plugin_name / planet_name / f"{planet_name}_biome.png"
    print(f"Loading biome PNG: {biome_path}")
    if not biome_path.exists():
        handle_news(None, "error", f"Biome PNG not found: {biome_path}")
        raise FileNotFoundError(f"Biome PNG not found: {biome_path}")

    try:
        biome_img = Image.open(biome_path).convert("RGB")
        biome_array = np.array(biome_img, dtype=np.uint8)
        print(f"Biome image shape: {biome_array.shape}")
        return biome_array
    except Exception as e:
        handle_news(None, "error", f"Failed to load biome PNG {biome_path}: {e}")
        raise


def generate_surface_tree(
    biome_array: np.ndarray,
    rgb_to_biome: dict,
    output_path: Path,
):
    height, width, _ = biome_array.shape
    rows = []

    for y in range(height):
        for x in range(width):
            rgb = tuple(int(c) for c in biome_array[y, x])
            form_id, editor_id, category = rgb_to_biome.get(
                rgb, ("00000000", "Unknown", "Unknown")
            )
            rows.append((form_id, editor_id, category))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FormID", "EditorID", "Category"])
        writer.writerows(rows)

    print(f"[Info] Wrote surface tree to: {output_path}")


def main():
    # Load biome PNG
    biome_array = load_biom_file(PNG_OUTPUT_DIR, planet_name)

    # Build RGB → (FormID, EditorID) lookup 
    rgb_to_biome = {
        biome.color: (f"{biome.form_id:08X}", biome.editor_id, biome.category)
        for biome in biome_db.all_biomes()
    }

    # Generate TSV SurfaceTree
    surface_tree_path = OUTPUT_DIR / "CSVs" / plugin_name / planet_name / f"{planet_name}_SurfaceTree.csv"
    surface_tree_path.parent.mkdir(parents=True, exist_ok=True)
    generate_surface_tree(biome_array, rgb_to_biome, surface_tree_path)


if __name__ == "__main__":
    main()
