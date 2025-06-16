#!/usr/bin/env python3
"""
PlanetBiomes Simplified

Creates a .biom file from two 256x256 input images (north and south hemispheres)
stacked vertically into a 256x512 grid, using a .biom template.
Maps RGB colors to biome IDs from Biomes.csv and assigns resources.
"""

# Standard Libraries
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
from PIL import Image
import subprocess
from scipy.ndimage import distance_transform_edt

# Project Modules
from PlanetTextures import load_biome_data
from PlanetNewsfeed import handle_news
from PlanetConstants import (
    TEMP_DIR,
    CSV_PATH,
    BIOM_DIR,
    INPUT_DIR,
    PLUGINS_DIR,
    SCRIPT_DIR,
    OUTPUT_DIR,
    CONFIG_PATH,
    TEMPLATE_PATH,
    PNG_OUTPUT_DIR,
)

# Third Party Libraries
from construct import Struct, Const, Rebuild, Container, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array

# Constants
GRID_SIZE = (256, 256)
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

# .biom Structure
CsSF_Biom = Struct(
    "magic" / Const(0x105, UInt16),
    "numBiomes" / Rebuild(UInt32, len_(this.biomeIds)),
    "biomeIds" / Array(this.numBiomes, UInt32),
    Const(2, UInt32),
    Const(GRID_SIZE[0], UInt32),
    Const(GRID_SIZE[1], UInt32),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridN" / Array(GRID_FLATSIZE, UInt32),
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridN" / Array(GRID_FLATSIZE, UInt8),
    Const(GRID_SIZE[0], UInt32),
    Const(GRID_SIZE[1], UInt32),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridS" / Array(GRID_FLATSIZE, UInt32),
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridS" / Array(GRID_FLATSIZE, UInt8),
)

def load_json(path: Path) -> Dict:
    """Load config.json."""
    try:
        with open(path, "r") as f:
            import json
            return json.load(f)
    except FileNotFoundError:
        handle_news(None, "error", f"Missing config: {path}")
        return {}

def load_biomes(csv_path: Path) -> Tuple[str, Dict[str, List[int]], Set[int], Set[int], Set[int]]:
    """Load planet and biome data from CSV."""
    import csv
    if not csv_path.exists():
        handle_news(None, "error", f"CSV not found: {csv_path}")
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        plugin = f.readline().strip().rstrip(",")
        reader = csv.DictReader(f, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"])
        next(reader, None)  # Skip header
        planet_data, life, nolife, ocean = {}, set(), set(), set()
        for row in reader:
            name = row["PlanetName"].strip()
            if not name:
                continue
            try:
                fid = int(row["BIOM_FormID"], 16)
                planet_data.setdefault(name, []).append(fid)
                eid = row["BIOM_EditorID"].strip().lower()
                if "ocean" in eid:
                    ocean.add(fid)
                elif "nolife" in eid:
                    nolife.add(fid)
                elif "life" in eid:
                    life.add(fid)
            except ValueError:
                handle_news(None, "warning", f"Invalid FormID: {row['BIOM_FormID']}")
        return plugin, planet_data, life, nolife, ocean

def load_combined_images(planet: str, input_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 256x512 biome and resource PNGs and split into hemispheres."""
    biome_path = input_dir / f"{planet}_biome.png"
    resource_path = input_dir / f"{planet}_resource.png"

    if not biome_path.exists():
        raise FileNotFoundError(f"Missing biome image: {biome_path}")
    if not resource_path.exists():
        raise FileNotFoundError(f"Missing resource image: {resource_path}")

    biome_img = np.array(Image.open(biome_path).convert("RGB"))
    resrc_img = np.array(Image.open(resource_path).convert("L"))

    if biome_img.shape != (512, 256, 3) or resrc_img.shape != (512, 256):
        raise ValueError("Biome image must be 256x512 RGB, resource must be 256x512 L")

    # Split into North and South hemispheres
    north_biome = biome_img[:256, :, :]
    south_biome = biome_img[256:, :, :]
    north_resrc = resrc_img[:256, :]
    south_resrc = resrc_img[256:, :]

    return (north_biome, south_biome, north_resrc, south_resrc)

def map_images_to_biomes(
    north_img: np.ndarray, south_img: np.ndarray, biome_data: Dict[int, Dict]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Map RGB colors in images to biome IDs."""
    rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}
    north_grid = np.zeros(GRID_SIZE, dtype=np.uint32)
    south_grid = np.zeros(GRID_SIZE, dtype=np.uint32)
    used_biome_ids = set()

    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            north_rgb = tuple(north_img[y, x])
            south_rgb = tuple(south_img[y, x])
            north_form_id = rgb_to_form_id.get(north_rgb, 0)  # Default to 0 if not found
            south_form_id = rgb_to_form_id.get(south_rgb, 0)
            north_grid[y, x] = north_form_id
            south_grid[y, x] = south_form_id
            used_biome_ids.add(north_form_id)
            used_biome_ids.add(south_form_id)

    if 0 in used_biome_ids:
        handle_news(None, "warning", "Some pixels mapped to default biome ID 0 (color not found in Biomes.csv)")

    return north_grid.flatten(), south_grid.flatten(), list(used_biome_ids)

def assign_resources_from_image(resrc_img: np.ndarray) -> np.ndarray:
    """Convert 256x256 grayscale image to resource ID array."""
    if resrc_img.shape != GRID_SIZE:
        raise ValueError("Resource grid shape mismatch")
    return resrc_img.flatten().astype(np.uint8)

def save_biome_grid_image(
    gridN: np.ndarray, gridS: np.ndarray, biome_colors: Dict[int, Tuple[int, int, int]], path_out: str, planet: str
):
    """Save combined biome grid as PNG."""
    os.makedirs(path_out, exist_ok=True)
    combined_grid = np.vstack((gridN.reshape(GRID_SIZE), gridS.reshape(GRID_SIZE)))
    h, w = combined_grid.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            form_id = int(combined_grid[y, x])
            color = biome_colors.get(form_id, (128, 128, 128))
            color_image[y, x] = color
    image = Image.fromarray(color_image, mode="RGB")
    path = os.path.join(path_out, f"{planet}_biome.png")
    image.save(path)
    handle_news(None, "info", f"Biome grid saved to: {path}")

class BiomFile:
    def __init__(self):
        self.biomeIds = []
        self.biomeGridN = np.array([], dtype=np.uint32)
        self.resrcGridN = np.array([], dtype=np.uint8)
        self.biomeGridS = np.array([], dtype=np.uint32)
        self.resrcGridS = np.array([], dtype=np.uint8)

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = cast(Container, CsSF_Biom.parse_stream(f))
        self.biomeIds = list(d.biomeIds)
        self.biomeGridN = np.array(d.biomeGridN, dtype=np.uint32)
        self.resrcGridN = np.array(d.resrcGridN, dtype=np.uint8)
        self.biomeGridS = np.array(d.biomeGridS, dtype=np.uint32)
        self.resrcGridS = np.array(d.resrcGridS, dtype=np.uint8)

    def save(self, path: Path):
        obj = {
            "magic": 0x105,
            "numBiomes": len(self.biomeIds),
            "biomeIds": self.biomeIds,
            "biomeGridN": self.biomeGridN.tolist(),
            "resrcGridN": [int(x) for x in self.resrcGridN.flatten()],
            "biomeGridS": self.biomeGridS.tolist(),
            "resrcGridS": [int(x) for x in self.resrcGridS.flatten()],
        }
        with open(path, "wb") as f:
            CsSF_Biom.build_stream(obj, f)

    def overwrite(self, biome_ids, grid_n, grid_s, resrc_n, resrc_s):
        self.biomeIds = biome_ids
        self.biomeGridN = grid_n
        self.biomeGridS = grid_s
        self.resrcGridN = resrc_n
        self.resrcGridS = resrc_s

def main():
    handle_news(None, "success", "=== Starting PlanetBiomes ===", flush=True)
    config = load_json(CONFIG_PATH)
    plugin_name = config.get("plugin_name", "default_plugin")
    planet_name = config.get("planet_name", "default_planet")
    out_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name

    # Load biome data and CSV
    plugin, planets, life, nolife, ocean = load_biomes(CSV_PATH)
    if planet_name not in planets:
        handle_news(None, "error", f"Planet {planet_name} not found in CSV")
        sys.exit(1)
    biome_ids = planets[planet_name]

    # Load input images
    north_biome, south_biome, north_resrc, south_resrc = load_combined_images(planet_name, INPUT_DIR)
    biome_data = load_biome_data(CSV_PATH, set(biome_ids))
    north_grid, south_grid, image_biome_ids = map_images_to_biomes(north_biome, south_biome, biome_data)
    north_resrc_ids = assign_resources_from_image(north_resrc)
    south_resrc_ids = assign_resources_from_image(south_resrc)

    biom = BiomFile()
    biom.load(TEMPLATE_PATH)
    biom.overwrite(image_biome_ids, north_grid, south_grid, north_resrc_ids, south_resrc_ids)

    # Save biome grid as PNG for verification
    biome_colors = {k: v["color"] for k, v in biome_data.items()}
    save_biome_grid_image(north_grid, south_grid, biome_colors, str(PNG_OUTPUT_DIR / plugin_name / planet_name), planet_name)

    # Save .biom file
    biom.save(out_dir / f"{planet_name}.biom")
    handle_news(None, "info", f"Biom file saved to: {out_dir / f'{planet_name}.biom'}")

    # Run next script
    #subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetSurface.py")], check=True)

if __name__ == "__main__":
    main()