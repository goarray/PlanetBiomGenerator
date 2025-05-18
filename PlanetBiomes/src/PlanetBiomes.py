#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Set, Tuple
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter, distance_transform_edt
import numpy as np
import subprocess
import json
import csv
import sys
import random

# Constants
GRID_SIZE = [256, 256]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

# Directories
BASE_DIR = (
    Path(sys._MEIPASS).resolve()
    if hasattr(sys, "_MEIPASS")
    else Path(__file__).parent.parent.resolve()
)

# File paths
CONFIG_PATH = BASE_DIR / "config" / "custom_config.json"
TEMPLATE_PATH = BASE_DIR / "assets" / "PlanetBiomes.biom"
CSV_PATH = BASE_DIR / "csv" / "PlanetBiomes.csv"
PREVIEW_PATH = BASE_DIR / "csv" / "preview.csv"

# Custom user input path
INPUT_DIR = BASE_DIR / "input"

# Outputs
OUTPUT_DIR = BASE_DIR / "output" / "planetdata" / "biomemaps"

# .biom structure
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


def load_json(path: Path) -> Dict:
    """Load config.json."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Missing config: {path}")
        return {}


# Load and use config
config = load_json(CONFIG_PATH)


def load_biomes(
    csv_path: Path,
) -> Tuple[str, Dict[str, List[int]], Set[int], Set[int], Set[int]]:
    with open(csv_path, newline="") as f:
        plugin = f.readline().strip().rstrip(",")
        reader = csv.DictReader(
            f, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"]
        )
        next(reader, None)
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
                print(f"Invalid FormID: {row['BIOM_FormID']}")
        return plugin, planet_data, life, nolife, ocean


def get_seed(config):
    """Return either a random seed or the user-defined seed from config."""
    use_random = config.get("use_random", False)
    if use_random:
        seed = random.randint(0, 99999)
        config["user_seed"] = seed
        return seed
    return int(config.get("user_seed", 0))


def generate_base_pattern(shape: Tuple[int, int]) -> np.ndarray:
    """Generate a square target pattern with values from 0 (perimeter) to 1 (center)."""
    h, w = shape
    # Create grid of coordinates normalized to [0,1]
    y = np.linspace(0, 1, h)[:, None]  # column vector
    x = np.linspace(0, 1, w)[None, :]  # row vector

    # Distance to nearest edge along x and y
    dist_x = np.minimum(x, 1 - x)
    dist_y = np.minimum(y, 1 - y)

    # Use the smaller distance to edges (like a square ring)
    dist_to_edge = np.minimum(dist_x, dist_y)

    # Normalize so perimeter=0, center=1
    base_pattern = dist_to_edge / dist_to_edge.max()

    return base_pattern


def remap_biome_weights(grid: np.ndarray, weights: List[float]) -> np.ndarray:
    """
    Remap grid values [0,1] using cumulative weights so biome bands vary in size.
    """
    import numpy as np

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()  # Normalize
    cdf = np.cumsum(weights)
    cdf = np.insert(cdf, 0, 0.0)  # start at 0

    # Allocate output grid
    remapped = np.zeros_like(grid)

    # Define ranges and remap each slice of the input
    for i in range(len(weights)):
        # The input range this slice originally covered
        input_lower = i / len(weights)
        input_upper = (i + 1) / len(weights)

        # The output range to map it to
        output_lower = cdf[i]
        output_upper = cdf[i + 1]

        # Mask of values in this input range
        mask = (grid >= input_lower) & (grid < input_upper)

        # Linearly scale within that range
        slice_grid = (grid[mask] - input_lower) / (input_upper - input_lower)
        remapped[mask] = output_lower + slice_grid * (output_upper - output_lower)

    return remapped


def generate_noise(shape: Tuple[int, int], config: Dict) -> np.ndarray:
    """Generate smooth noise normalized to 0..1."""
    seed = get_seed(config)
    np.random.seed(seed)
    noise = np.random.rand(*shape)
    noise = gaussian_filter(noise, sigma=16)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_combined_pattern(shape: Tuple[int, int], config: Dict) -> np.ndarray:
    """Generate combined base pattern plus noise (if enabled)."""
    base_pattern = generate_base_pattern(shape)

    if not config.get("enable_noise", True):
        return base_pattern

    noise = generate_noise(shape, config)
    combined = base_pattern + noise
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    return combined


def assign_biomes(grid: np.ndarray, biome_ids: List[int]) -> np.ndarray:
    if len(biome_ids) == 1:
        return np.full(GRID_FLATSIZE, biome_ids[0], dtype=np.uint32)
    epsilon = 1e-6
    grid = np.clip(grid, 0, 1 - epsilon)
    mapped = np.zeros(GRID_FLATSIZE, dtype=np.uint32)
    n_biomes = len(biome_ids)
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            i = y * GRID_SIZE[0] + x
            idx = int(grid[y, x] * n_biomes)
            mapped[i] = biome_ids[idx]
    return mapped


def assign_resources(
    grid: np.ndarray, life: Set[int], nolife: Set[int], ocean: Set[int]
) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h, w), dtype=np.uint8)
    rings = {True: [0, 1, 2, 3, 4], False: [80, 81, 82, 83, 84]}
    for b in np.unique(grid):
        if b == 0:
            continue
        mask = grid == b
        if b in ocean:
            out[mask] = 8
            continue
        ring = rings[b in life]
        dist_out = distance_transform_edt(~mask)
        dist_in = distance_transform_edt(mask)
        grad = 0.5 * (dist_out / max(1, dist_out[mask].max())) + 0.5 * (
            1 - dist_in / max(1, dist_in[mask].max())
        )
        for i, val in enumerate(ring):
            band = (grad >= i / 6.5) & (grad < (i + 1) / 6.0)
            out[band & mask] = val
    return out


class BiomFile:
    def __init__(self):
        (
            self.biomeIds,
            self.biomeGridN,
            self.resrcGridN,
            self.biomeGridS,
            self.resrcGridS,
        ) = ([], [], [], [], [])

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = CsSF_Biom.parse_stream(f)
        self.biomeIds, self.biomeGridN, self.resrcGridN = (
            list(d.biomeIds),
            np.array(d.biomeGridN),
            np.array(d.resrcGridN),
        )
        self.biomeGridS, self.resrcGridS = np.array(d.biomeGridS), np.array(
            d.resrcGridS
        )

    def save(self, path: Path):
        obj = {
            "biomeIds": self.biomeIds,
            "biomeGridN": self.biomeGridN,
            "biomeGridS": self.biomeGridS,
            "resrcGridN": self.resrcGridN,
            "resrcGridS": self.resrcGridS,
        }
        with open(path, "wb") as f:
            CsSF_Biom.build_stream(obj, f)

    def overwrite(self, biome_ids: List[int], grid: np.ndarray):
        """Replace biomes using either noise or structured grid."""
        self.biomeGridN = assign_biomes(grid, biome_ids)
        self.biomeGridS = assign_biomes(grid, biome_ids)
        self.biomeIds = list(set(biome_ids))


def main():
    preview = "--preview" in sys.argv
    print("Running in preview mode" if preview else "Generating full biome set...")
    config = load_json(CONFIG_PATH)
    biome_cfg = config
    biome_csv = PREVIEW_PATH if preview else CSV_PATH

    plugin, planets, life, nolife, ocean = load_biomes(biome_csv)
    out_dir = OUTPUT_DIR / plugin
    out_dir.mkdir(parents=True, exist_ok=True)
    template = BiomFile()
    template.load(TEMPLATE_PATH)

    for planet, biomes in planets.items():
        print(f"Processing: {planet} ({len(biomes)} biomes)")
        inst = BiomFile()
        inst.load(TEMPLATE_PATH)
        zone_weights = [config.get(f"zone_0{i}", 1.0) for i in range(7)]
        pattern = generate_combined_pattern((GRID_SIZE[1], GRID_SIZE[0]), biome_cfg)
        remapped_pattern = remap_biome_weights(pattern, zone_weights)
        inst.overwrite(biomes, remapped_pattern)
        inst.resrcGridN = assign_resources(
            inst.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()
        inst.resrcGridS = assign_resources(
            inst.biomeGridS.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()
        inst.save(out_dir / f"{planet}.biom")

    subprocess.run(["python", str(BASE_DIR / "src" / "PlanetTextures.py")], check=True)
    print("Biome processing complete.", flush=True)


if __name__ == "__main__":
    main()
