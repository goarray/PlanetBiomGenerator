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
    PATTERN_PATH,
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

def load_biome_pattern_map(csv_path: Path) -> dict[str, tuple[str, str]]:
    pattern_map = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["BiomeCategory"].strip().lower()
            pattern_id = row["BlockPatternID"].strip()
            pattern_editor = row["BlockPatternEditorID"].strip()
            if category and pattern_id and pattern_editor:
                pattern_map[category] = (pattern_id, pattern_editor)
    return pattern_map

def build_category_to_pattern_map(biome_csv: Path, special_csv: Path) -> dict[str, tuple[str, str]]:
    pattern_map = load_biome_pattern_map(biome_csv)
    pattern_map.update(load_special_pattern_map(special_csv))  # override/add roads/rivers
    return pattern_map

def load_special_pattern_map(csv_path: Path) -> dict[str, tuple[str, str]]:
    pattern_map = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["Category"].strip().lower()
            pattern_id = row["FormID"].strip()
            pattern_editor = row["EditorID"].strip()
            if category and pattern_id and pattern_editor:
                pattern_map[category] = (pattern_id, pattern_editor)
    return pattern_map

def load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    img = Image.open(mask_path).convert("L")  # grayscale mask
    return np.array(img, dtype=np.uint8)

def generate_surface_tree(
    biome_array: np.ndarray,
    river_mask: np.ndarray,
    road_mask: np.ndarray,
    rgb_to_biome: dict,
    output_path: Path,
):
    height, width, _ = biome_array.shape
    rows = []

    category_to_pattern = build_category_to_pattern_map(CSV_PATH, PATTERN_PATH)

    for y in range(height):
        for x in range(width):
            if river_mask[y, x] > 0:
                category = "rivers"
            elif road_mask[y, x] > 0:
                category = "roads"
            else:
                rgb = tuple(int(c) for c in biome_array[y, x])
                _, _, category = rgb_to_biome.get(rgb, ("00000000", "Unknown", "unknown"))

            form_id, editor_id = category_to_pattern.get(category.lower(), ("00000000", "UnknownPattern"))
            rows.append((form_id, editor_id, category))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FormID", "EditorID", "Category"])
        writer.writerows(rows)

    print(f"[Info] Wrote surface tree to: {output_path}")


def export_colony_overlay_coords(
    colony_mask: np.ndarray, overlays_csv: Path, output_path: Path
):
    """Export colony locations in Bethesda normalized coordinates, with overlay info."""
    height, width = colony_mask.shape
    rows = [("Latitude", "Longitude", "FormID", "EditorID", "Name")]

    # Load overlay info
    overlays = []
    with open(overlays_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            overlays.append((row["FormID"], row["EditorID"], row["Name"]))

    if not overlays:
        raise ValueError("No overlays found in WorldOverlays.csv!")

    ys, xs = np.where(colony_mask == 255)
    for i, (y, x) in enumerate(zip(ys, xs)):
        lat = 1.0 - (y / height) * 2.0
        lon = (x / width) * 2.0 - 1.0

        # Round-robin assignment
        form_id, editor_id, name = overlays[i % len(overlays)]
        safe_name = name.replace(" ", "_").replace(",", "_")
        rows.append((lat, lon, form_id, editor_id, safe_name))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[Info] Exported {len(rows)-1} colony overlays to: {output_path}")


def main():
    # Load biome PNG
    biome_array = load_biom_file(PNG_OUTPUT_DIR, planet_name)

    # Build RGB → (FormID, EditorID) lookup
    rgb_to_biome = {
        biome.color: (f"{biome.form_id:08X}", biome.editor_id, biome.category)
        for biome in biome_db.all_biomes()
    }

    river_mask = load_mask(PNG_OUTPUT_DIR / plugin_name / planet_name / f"{planet_name}_river_mask.png")
    road_mask  = load_mask(PNG_OUTPUT_DIR / plugin_name / planet_name / f"{planet_name}_road_mask.png")
    colony_mask = load_mask(PNG_OUTPUT_DIR / plugin_name / planet_name / f"{planet_name}_colony_mask.png")

    # Generate TSV SurfaceTree
    surface_tree_path = OUTPUT_DIR / "CSVs" / plugin_name / planet_name / f"{planet_name}_SurfaceTree.csv"
    surface_tree_path.parent.mkdir(parents=True, exist_ok=True)
    generate_surface_tree(
        biome_array, river_mask, road_mask, rgb_to_biome, surface_tree_path
    )

    overlays_csv = CSV_DIR / "WorldOverlays.csv"
    overlay_output = OUTPUT_DIR / "CSVs" / plugin_name / planet_name / f"{planet_name}_WorldOverlays.csv"
    overlay_output.parent.mkdir(parents=True, exist_ok=True)
    export_colony_overlay_coords(colony_mask, overlays_csv, overlay_output)


if __name__ == "__main__":
    main()
