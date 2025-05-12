#!/usr/bin/env python3
"""
Planet Biomes Generator

Generates biome and resource grids for planets based on configuration and CSV data.
Outputs .biom files with biome assignments and resource distributions.
Supports procedural generation with noise, distortion, and drag effects.

Dependencies:
- Python 3.8+
- construct
- scipy
- numpy
- json
- csv
- pathlib
- subprocess
"""

from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter, distance_transform_edt
import numpy as np
import subprocess
import json
import csv
import sys

# Directory paths
BASE_DIR = Path(__file__).parent.parent
SCRIPT_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
TEMPLATE_PATH = BASE_DIR / "assets" / "PlanetBiomes.biom"
CSV_PATH = BASE_DIR / "assets" / "PlanetBiomes.csv"
OUTPUT_DIR = BASE_DIR / "Output"

# Grid constants
GRID_SIZE = [0x100, 0x100]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

# Global configuration
config = {}

def load_config():
    """Load configuration from JSON file."""
    global config
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {CONFIG_PATH} not found.")
        config = {}

# Initialize configuration
load_config()

# Flatten nested config dictionary
biome_config = {key: int(value) if isinstance(value, float) and value.is_integer() else value
                for category in config.values() for key, value in category.items()}

# Define .biom file structure
CsSF_Biom = Struct(
    "magic" / Const(0x105, UInt16),
    "_numBiomes" / Rebuild(UInt32, len_(this.biomeIds)),
    "biomeIds" / UInt32[this._numBiomes],
    Const(2, UInt32),
    Const(GRID_SIZE, UInt32[2]),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridN" / UInt32[GRID_FLATSIZE],
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridN" / UInt8[GRID_FLATSIZE],
    Const(GRID_SIZE, UInt32[2]),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridS" / UInt32[GRID_FLATSIZE],
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridS" / UInt8[GRID_FLATSIZE],
)

def load_planet_biomes(csv_path):
    """Load biome data from CSV and categorize biomes."""
    planet_biomes = {}
    life_biomes = set()
    no_life_biomes = set()
    ocean_biomes = set()

    with open(csv_path, newline="") as csvfile:
        first_line = csvfile.readline().strip()
        plugin_name = first_line.rstrip(",").strip()
        if not plugin_name:
            raise ValueError("Plugin name is missing or empty.")

        reader = csv.DictReader(csvfile, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"])
        next(reader, None)  # Skip header row

        for row in reader:
            planet = row["PlanetName"].strip()
            if not planet:
                continue

            try:
                form_id = int(row["BIOM_FormID"], 16)
                planet_biomes.setdefault(planet, []).append(form_id)
                editor_id = row["BIOM_EditorID"].strip().lower()
                if "ocean" in editor_id:
                    ocean_biomes.add(form_id)
                elif "life" in editor_id and "nolife" not in editor_id:
                    life_biomes.add(form_id)
                elif "nolife" in editor_id:
                    no_life_biomes.add(form_id)
            except ValueError:
                print(f"Warning: Invalid FormID '{row['BIOM_FormID']}' for planet '{planet}'. Skipping.")

    return plugin_name, planet_biomes, list(life_biomes), list(no_life_biomes), list(ocean_biomes)

class BiomFile:
    """Manages .biom file data for biome and resource grids."""
    def __init__(self):
        self.biomeIds = []
        self.biomeGridN = []
        self.resrcGridN = []
        self.biomeGridS = []
        self.resrcGridS = []

    def load(self, filename):
        """Load .biom file data into instance."""
        with open(filename, "rb") as f:
            data = CsSF_Biom.parse_stream(f)
        self.biomeIds = list(data.biomeIds)
        self.biomeGridN = np.array(data.biomeGridN)
        self.resrcGridN = np.array(data.resrcGridN)
        self.biomeGridS = np.array(data.biomeGridS)
        self.resrcGridS = np.array(data.resrcGridS)

    def overwrite_biome_ids(self, new_biome_ids):
        """Update biome grids with new biome IDs."""
        if not self.biomeIds:
            raise ValueError("No biome IDs found in file.")
        if not new_biome_ids or len(new_biome_ids) < 1:
            raise ValueError("At least one biome ID is required.")

        if len(new_biome_ids) == 1:
            print(f"Warning: Only one biome ID provided. Filling entire grid with {new_biome_ids[0]}.")
            self.biomeIds = [new_biome_ids[0]]
            self.biomeGridN = np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
            self.biomeGridS = np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
            return

        if len(new_biome_ids) > 7:
            print(f"Warning: {len(new_biome_ids)} biomes provided, but max is 7. Truncating to first 7.")
            new_biome_ids = new_biome_ids[:7]

        num_biomes = len(new_biome_ids)
        zones = np.linspace(0, GRID_SIZE[1], num_biomes + 1, dtype=int)

        def generate_zone_map(shape, seed=biome_config["zone_seed"]):
            """Generate smoothed noise map for biome zones."""
            np.random.seed(seed)
            base = np.random.rand(*shape)
            large = gaussian_filter(base, sigma=16)
            medium = gaussian_filter(np.random.rand(*shape), sigma=6)
            small = gaussian_filter(np.random.rand(*shape), sigma=2)
            combined = 0.6 * large + 0.3 * medium + 0.1 * small
            combined = np.power(combined, 1.5)
            return (combined - combined.min()) / (combined.max() - combined.min())

        def assign_biomes(smoothed_noise, hemisphere, biome_config):
            """Assign biome IDs to grid based on noise and config."""
            if len(new_biome_ids) == 1:
                print(f"Single-biome planet detected. Filling grid with biome ID {new_biome_ids[0]}")
                return np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)

            grid = np.zeros(GRID_FLATSIZE, dtype=np.uint32)
            center_y, center_x = GRID_SIZE[1] // 2, GRID_SIZE[0] // 2
            n = biome_config["squircle_exponent"]

            if biome_config["apply_distortion"]:
                distortion = gaussian_filter(np.random.rand(GRID_SIZE[1], GRID_SIZE[0]), sigma=(1 / biome_config["distortion_sigma"]))
                distortion = (distortion - distortion.min()) / (distortion.max() - distortion.min())
            else:
                distortion = np.zeros((GRID_SIZE[1], GRID_SIZE[0]))

            num_equator_drags = int(biome_config["num_equator_drags"])
            equator_drag_centers = []
            num_pole_drags = int(biome_config["num_pole_drags"])
            pole_drag_centers = []

            for _ in range(num_equator_drags):
                x_min, x_max = biome_config["equator_drag_x_min"], biome_config["equator_drag_x_max"]
                y_min, y_max = biome_config["equator_drag_y_min"], biome_config["equator_drag_y_max"]
                if x_min >= x_max or y_min >= y_max:
                    raise ValueError(f"Invalid equator_drag range: X=({x_min}, {x_max}), Y=({y_min}, {y_max})")
                equator_drag_centers.append(
                    (center_x + np.random.randint(x_min, x_max), center_y + np.random.randint(y_min, y_max))
                )

            for _ in range(num_pole_drags):
                x_min, x_max = biome_config["pole_drag_x_min"], biome_config["pole_drag_x_max"]
                y_min, y_max = biome_config["pole_drag_y_min"], biome_config["pole_drag_y_max"]
                if x_min >= x_max or y_min >= y_max:
                    raise ValueError(f"Invalid pole_drag range: X=({x_min}, {x_max}), Y=({y_min}, {y_max})")
                pole_drag_centers.append(
                    (center_x + np.random.randint(x_min, x_max), center_y + np.random.randint(y_min, y_max))
                )

            drag_radius = biome_config["drag_radius"]

            for y in range(GRID_SIZE[1]):
                for x in range(GRID_SIZE[0]):
                    i = y * GRID_SIZE[0] + x
                    dx = (x - center_x) / (GRID_SIZE[0] / 2)
                    dy = (y - center_y) / (GRID_SIZE[1] / 2)
                    r = (abs(dx) ** n + abs(dy) ** n) ** (1 / n)
                    r = min(r, 1.0)

                    lat_factor = r + biome_config["lat_distortion_factor"] * (distortion[y, x] - 0.5) if biome_config["apply_latitude_blending"] else r
                    lat_factor = np.clip(lat_factor, 0, 1)

                    noise = smoothed_noise[y, x]
                    combined = biome_config["noise_factor"] * noise + (1 / biome_config["lat_weight_factor"]) * lat_factor

                    reversed_biome_ids = list(reversed(new_biome_ids))
                    biome_index = int(combined * len(reversed_biome_ids))
                    biome_index = min(biome_index, len(reversed_biome_ids) - 1)

                    if biome_index == 0 and biome_config["enable_equator_drag"]:
                        for cx, cy in equator_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2
                                combined -= weight * biome_config["equator_drag_strength"]
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    elif biome_index == len(reversed_biome_ids) - 1 and biome_config["enable_pole_drag"]:
                        for cx, cy in pole_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2
                                combined += weight * biome_config["pole_drag_strength"]
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    elif biome_index == 1 and len(reversed_biome_ids) > 1 and biome_config["enable_equator_intrusion"]:
                        for cx, cy in equator_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2
                                combined -= weight * biome_config["equator_intrusion_strength"]
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    elif biome_index == len(reversed_biome_ids) - 2 and len(reversed_biome_ids) > 2 and biome_config["enable_pole_intrusion"]:
                        for cx, cy in pole_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2
                                combined += weight * biome_config["pole_intrusion_strength"]
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    grid[i] = reversed_biome_ids[biome_index]

            return grid

        shape = (GRID_SIZE[1], GRID_SIZE[0])
        noise_n = generate_zone_map(shape)
        noise_s = generate_zone_map(shape)
        self.biomeGridN = assign_biomes(noise_n, "N", biome_config)
        self.biomeGridS = assign_biomes(noise_s, "S", biome_config)
        self.biomeIds = list(set(new_biome_ids))

    def save(self, filename):
        """Save biome data to .biom file."""
        obj = {
            "biomeIds": self.biomeIds,
            "biomeGridN": self.biomeGridN,
            "biomeGridS": self.biomeGridS,
            "resrcGridN": self.resrcGridN,
            "resrcGridS": self.resrcGridS,
        }
        with open(filename, "wb") as f:
            CsSF_Biom.build_stream(obj, f)
        print(f"Saved to: {filename}")

def assign_resources(biome_grid, life_biome_ids, nolife_biome_ids, ocean_ids):
    """Distribute resources across biome grid."""
    height, width = biome_grid.shape
    resource_grid = np.zeros((height, width), dtype=np.uint8)
    resource_rings_life = [0, 1, 2, 3, 4]
    resource_rings_nolife = [80, 81, 82, 83, 84]

    for biome_id in np.unique(biome_grid):
        if biome_id == 0:
            continue

        mask = (biome_grid == biome_id)
        if biome_id in ocean_ids:
            resource_grid[mask] = 8
            continue

        if biome_id in life_biome_ids:
            ring_codes = resource_rings_life
        elif biome_id in nolife_biome_ids:
            ring_codes = resource_rings_nolife
        else:
            continue

        dist_outer = distance_transform_edt(~mask)
        dist_inner = distance_transform_edt(mask)
        max_outer = dist_outer[mask].max() or 1.0
        max_inner = dist_inner[mask].max() or 1.0
        normalized_outer = np.clip(dist_outer / max_outer, 0, 1)
        normalized_inner = np.clip(dist_inner / max_inner, 0, 1)
        final_gradient = (0.5 * normalized_outer) + (0.5 * (1 - normalized_inner))

        for i, code in enumerate(ring_codes):
            lower = i / (len(ring_codes) * 1.3)
            upper = (i + 1) / (len(ring_codes) * 1.2)
            ring_mask = mask & (final_gradient >= lower) & (final_gradient < upper)
            resource_grid[ring_mask] = code

    return resource_grid

def clone_biom(biom):
    """Create a copy of a BiomFile instance."""
    new = BiomFile()
    new.biomeIds = biom.biomeIds.copy()
    new.biomeGridN = biom.biomeGridN.copy()
    new.resrcGridN = biom.resrcGridN.copy()
    new.biomeGridS = biom.biomeGridS.copy()
    new.resrcGridS = biom.resrcGridS.copy()
    return new

def main():
    """Process planet biomes and generate output files."""
    plugin_name, planet_biomes, life_biomes, nolife_biomes, ocean_biomes = load_planet_biomes(CSV_PATH)
    output_subdir = OUTPUT_DIR / plugin_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    template = BiomFile()
    template.load(TEMPLATE_PATH)

    for planet, new_ids in planet_biomes.items():
        print(f"Processing {planet} with {len(new_ids)} biome(s)")
        new_biom = clone_biom(template)
        new_biom.overwrite_biome_ids(new_ids)
        new_biom.resrcGridN = assign_resources(
            new_biom.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]),
            life_biomes, nolife_biomes, ocean_biomes
        ).flatten()
        new_biom.resrcGridS = assign_resources(
            new_biom.biomeGridS.reshape(GRID_SIZE[1], GRID_SIZE[0]),
            life_biomes, nolife_biomes, ocean_biomes
        ).flatten()
        out_path = output_subdir / f"{planet}.biom"
        new_biom.save(out_path)

    subprocess.run(["python", str(Path(__file__).parent / "PlanetTextures.py")], check=True)
    sys.exit()

if __name__ == "__main__":
    main()