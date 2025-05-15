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
import noise
import json
import csv
import sys
import os

# Directory paths
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS).resolve()
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()

BASE_DIR = Path(__file__).parent.parent
SCRIPT_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
OUTPUT_DIR = BASE_DIR / "Output" / "planetdata" / "biomemaps"

# File Paths
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
TEMPLATE_PATH = BASE_DIR / "assets" / "PlanetBiomes.biom"
CSV_PATH = BASE_DIR / "csv" / "PlanetBiomes.csv"
PREVIEW_BIOM_PATH = BASE_DIR / "csv" / "preview.csv"

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


def save_config(config):
    """Save config to JSON file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


PREVIEW_MODE = "--preview" in sys.argv
if PREVIEW_MODE:
    print("Running in preview mode!")
biome_path = PREVIEW_BIOM_PATH if PREVIEW_MODE else CSV_PATH

# Flatten nested config dictionary
biome_config = {
    key: int(value) if isinstance(value, float) and value.is_integer() else value
    for category in config.values()
    for key, value in category.items()
}

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
    print(f"Attempting to load: {csv_path}")
    planet_biomes = {}
    life_biomes = set()
    no_life_biomes = set()
    ocean_biomes = set()
    with open(csv_path, newline="") as csvfile:
        first_line = csvfile.readline().strip()
        plugin_name = first_line.rstrip(",").strip()
        if not plugin_name:
            raise ValueError("Plugin name is missing or empty.")
        reader = csv.DictReader(
            csvfile, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"]
        )
        next(reader, None)
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
                print(
                    f"Warning: Invalid FormID '{row['BIOM_FormID']}' for planet '{planet}'. Skipping."
                )
        return plugin_name, planet_biomes, life_biomes, no_life_biomes, ocean_biomes


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
        """Update biome grids with new biome IDs, applying distortion effects before noise."""
        if not self.biomeIds:
            raise ValueError("No biome IDs found in file.")
        if not new_biome_ids or len(new_biome_ids) < 1:
            raise ValueError("At least one biome ID is required.")
        if len(new_biome_ids) == 1:
            print(
                f"Warning: Only one biome ID provided. Filling entire grid with {new_biome_ids[0]}."
            )
            self.biomeIds = [new_biome_ids[0]]
            self.biomeGridN = np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
            self.biomeGridS = np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
            return
        if len(new_biome_ids) > 7:
            print(
                f"Warning: {len(new_biome_ids)} biomes provided, but max is 7. Truncating to first 7."
            )
            new_biome_ids = new_biome_ids[:7]

        def generate_noise_map(shape, seed=None, use_random_seed=False):
            """Generate smoothed noise map for biome zones with optional config-based seed."""
            if use_random_seed:
                seed = np.random.randint(0, 10000)
            elif seed is None:
                seed = config["global_seed"].get("zone_seed", 0)
            np.random.seed(seed)
            base = np.random.rand(*shape)
            large = gaussian_filter(base, sigma=16)
            medium = gaussian_filter(np.random.rand(*shape), sigma=6)
            small = gaussian_filter(np.random.rand(*shape), sigma=2)
            combined = 0.6 * large + 0.3 * medium + 0.1 * small
            combined = np.power(combined, 1.5)
            return (combined - combined.min()) / (combined.max() - combined.min())

        def assign_biomes(base_grid, hemisphere, biome_config):
            """Assign biome IDs to grid based on distortion and noise."""
            if len(new_biome_ids) == 1:
                print(
                    f"Single-biome planet detected. Filling grid with biome ID {new_biome_ids[0]}"
                )
                return np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)

            # Generate noise to layer on top of distortion
            noise_map = generate_noise_map((GRID_SIZE[1], GRID_SIZE[0]))

            # Combine distortion (base_grid) with noise
            combined = (biome_config["noise_factor"] * noise_map) + (
                (1 - biome_config["noise_factor"]) * base_grid
            )
            combined = np.clip(combined, 0, 1)

            # Assign biomes based on combined grid
            reversed_biome_ids = list(reversed(new_biome_ids))
            grid = np.zeros(GRID_FLATSIZE, dtype=np.uint32)
            for y in range(GRID_SIZE[1]):
                for x in range(GRID_SIZE[0]):
                    i = y * GRID_SIZE[0] + x
                    biome_index = min(
                        int(combined[y, x] * len(reversed_biome_ids)),
                        len(reversed_biome_ids) - 1,
                    )
                    grid[i] = reversed_biome_ids[biome_index]
            return grid

        def generate_distortion_map(shape, biome_config):
            """Generate base distortion map with equator/pole drag and intrusion effects."""
            distortion_grid = np.zeros(shape)
            center_y, center_x = GRID_SIZE[1] // 2, GRID_SIZE[0] // 2
            n = biome_config["squircle_exponent"]

            # Base latitude factor
            for y in range(GRID_SIZE[1]):
                for x in range(GRID_SIZE[0]):
                    dx = (x - center_x) / (GRID_SIZE[0] / 2)
                    dy = (y - center_y) / (GRID_SIZE[1] / 2)
                    r = (abs(dx) ** n + abs(dy) ** n) ** (1 / n)
                    r = min(r, 1.0)
                    distortion_grid[y, x] = r

            # Apply latitude-based distortion using Perlin noise
            if biome_config["apply_distortion"]:
                distortion_sigma = biome_config.get("distortion_sigma", 0.05)
                distortion_sigma = 0.999 * (distortion_sigma ** 0.95)

                noise_amplitude = biome_config.get("distortion_amplitude", 0.2)

                for y in range(GRID_SIZE[1]):
                    for x in range(GRID_SIZE[0]):
                        noise_offset = noise.pnoise2(
                            x * distortion_sigma, y * distortion_sigma, octaves=3
                        )
                        lat_factor = distortion_grid[y, x] + biome_config[
                            "lat_distortion_factor"
                        ] * (noise_offset * noise_amplitude)
                        distortion_grid[y, x] = np.clip(lat_factor, 0, 1)

            # Apply equator and pole drag/intrusion effects
            num_equator_drags = int(biome_config["num_equator_drags"])
            equator_drag_centers = []
            num_pole_drags = int(biome_config["num_pole_drags"])
            pole_drag_centers = []

            for _ in range(num_equator_drags):
                x_min, x_max = (
                    biome_config["equator_drag_x_min"],
                    biome_config["equator_drag_x_max"],
                )
                y_min, y_max = (
                    biome_config["equator_drag_y_min"],
                    biome_config["equator_drag_y_max"],
                )
                if x_min >= x_max or y_min >= y_max:
                    raise ValueError(
                        f"Invalid equator_drag range: X=({x_min}, {x_max}), Y=({y_min}, {y_max})"
                    )
                equator_drag_centers.append(
                    (
                        center_x + np.random.randint(x_min, x_max),
                        center_y + np.random.randint(y_min, y_max),
                    )
                )

            for _ in range(num_pole_drags):
                x_min, x_max = (
                    biome_config["pole_drag_x_min"],
                    biome_config["pole_drag_x_max"],
                )
                y_min, y_max = (
                    biome_config["pole_drag_y_min"],
                    biome_config["pole_drag_y_max"],
                )
                if x_min >= x_max or y_min >= y_max:
                    raise ValueError(
                        f"Invalid pole_drag range: X=({x_min}, {x_max}), Y=({y_min}, {y_max})"
                    )
                pole_drag_centers.append(
                    (
                        center_x + np.random.randint(x_min, x_max),
                        center_y + np.random.randint(y_min, y_max),
                    )
                )

            drag_radius = biome_config["drag_radius"]
            radius_scale = biome_config.get("drag_radius_scale", 2.0)
            noise_amplitude = biome_config.get("noise_amplitude", 0.3)
            equator_influence_zones = biome_config.get("equator_influence_zones", 2)
            pole_influence_zones = biome_config.get("pole_influence_zones", 2)
            # New config parameter for noise scale factor (0 to 1, where 1 produces ~50x50 pixel globs)
            noise_scale_factor = biome_config.get("noise_scale_factor", 0.1)
            # Exponentially scale the noise frequency to control blob size
            # Base scale is divided by 2^factor to reduce frequency (increase blob size)
            noise_scale = 0.1 / (
                2 ** (noise_scale_factor * 4)
            )  # Exponential scaling: 0.1 to 0.00625

            for y in range(GRID_SIZE[1]):
                for x in range(GRID_SIZE[0]):
                    if biome_config["enable_equator_drag"]:
                        for cx, cy in equator_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            noise_offset = (
                                noise.pnoise2(
                                    x * noise_scale,
                                    y * noise_scale,
                                    octaves=1,
                                    persistence=0.5,
                                )
                                * noise_amplitude
                            )
                            effective_radius = (
                                drag_radius * radius_scale * (1 + noise_offset)
                            )
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < effective_radius:
                                weight = (1 - dist / effective_radius) ** 2
                                zone_shift = (
                                    weight
                                    * biome_config["equator_drag_strength"]
                                    * equator_influence_zones
                                )
                                distortion_grid[y, x] -= zone_shift
                                distortion_grid[y, x] = np.clip(
                                    distortion_grid[y, x], 0, 1
                                )

                    if biome_config["enable_pole_drag"]:
                        for cx, cy in pole_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            noise_offset = (
                                noise.pnoise2(
                                    x * noise_scale,
                                    y * noise_scale,
                                    octaves=1,
                                    persistence=0.5,
                                )
                                * noise_amplitude
                            )
                            effective_radius = (
                                drag_radius * radius_scale * (1 + noise_offset)
                            )
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < effective_radius:
                                weight = (1 - dist / effective_radius) ** 2
                                zone_shift = (
                                    weight
                                    * biome_config["pole_drag_strength"]
                                    * pole_influence_zones
                                )
                                distortion_grid[y, x] += zone_shift
                                distortion_grid[y, x] = np.clip(
                                    distortion_grid[y, x], 0, 1
                                )

                    if biome_config["enable_equator_intrusion"]:
                        for cx, cy in equator_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            noise_offset = (
                                noise.pnoise2(
                                    x * noise_scale,
                                    y * noise_scale,
                                    octaves=1,
                                    persistence=0.5,
                                )
                                * noise_amplitude
                            )
                            effective_radius = (
                                drag_radius * radius_scale * (1 + noise_offset)
                            )
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < effective_radius:
                                weight = (1 - dist / effective_radius) ** 2
                                zone_shift = (
                                    weight
                                    * biome_config["equator_intrusion_strength"]
                                    * equator_influence_zones
                                )
                                distortion_grid[y, x] -= zone_shift
                                distortion_grid[y, x] = np.clip(
                                    distortion_grid[y, x], 0, 1
                                )

                    if biome_config["enable_pole_intrusion"]:
                        for cx, cy in pole_drag_centers:
                            ddx, ddy = x - cx, y - cy
                            noise_offset = (
                                noise.pnoise2(
                                    x * noise_scale,
                                    y * noise_scale,
                                    octaves=1,
                                    persistence=0.5,
                                )
                                * noise_amplitude
                            )
                            effective_radius = (
                                drag_radius * radius_scale * (1 + noise_offset)
                            )
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < effective_radius:
                                weight = (1 - dist / effective_radius) ** 2
                                zone_shift = (
                                    weight
                                    * biome_config["pole_intrusion_strength"]
                                    * pole_influence_zones
                                )
                                distortion_grid[y, x] += zone_shift
                                distortion_grid[y, x] = np.clip(
                                    distortion_grid[y, x], 0, 1
                                )

            return (distortion_grid - distortion_grid.min()) / (
                distortion_grid.max() - distortion_grid.min()
            )

        shape = (GRID_SIZE[1], GRID_SIZE[0])
        distortion_n = generate_distortion_map(shape, biome_config)
        distortion_s = generate_distortion_map(shape, biome_config)
        self.biomeGridN = assign_biomes(distortion_n, "N", biome_config)
        self.biomeGridS = assign_biomes(distortion_s, "S", biome_config)
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
        mask = biome_grid == biome_id
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


def start_processing_widget(title):
    """Launch processing indicator and store process handle."""
    global processing_widget_process
    script_path = os.path.join(os.path.dirname(__file__), "processing_widget.py")
    if not os.path.exists(script_path):
        print(f"Error: {script_path} does not exist!")
        return
    processing_widget_process = subprocess.Popen(["python", script_path, title])


def stop_processing_widget():
    """Terminate processing widget when script completes."""
    global processing_widget_process
    if processing_widget_process:
        processing_widget_process.terminate()


_progress_started = False


def main():
    global _progress_started
    if not _progress_started:
        _progress_started = True
        start_processing_widget("Processing Planet Biomes")
    plugin_name, planet_biomes, life_biomes, no_life_biomes, ocean_biomes = (
        load_planet_biomes(biome_path)
    )
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
            life_biomes,
            no_life_biomes,
            ocean_biomes,
        ).flatten()
        new_biom.resrcGridS = assign_resources(
            new_biom.biomeGridS.reshape(GRID_SIZE[1], GRID_SIZE[0]),
            life_biomes,
            no_life_biomes,
            ocean_biomes,
        ).flatten()
        out_path = output_subdir / f"{planet}.biom"
        new_biom.save(out_path)
    subprocess.run(
        ["python", str(Path(__file__).parent / "PlanetTextures.py")], check=True
    )
    stop_processing_widget()
    sys.exit()


if __name__ == "__main__":
    main()
