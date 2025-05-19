#!/usr/bin/env python3

# Standard Libraries
import sys
import os
import json
import csv
import subprocess
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, NamedTuple, cast
from PlanetConstants import (
    BASE_DIR,
    SCRIPT_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    CONFIG_PATH,
    TEMPLATE_PATH,
    PREVIEW_PATH,
)

# Third Party Libraries
import scipy.ndimage
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array


# Constants
GRID_SIZE = [256, 256]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]


class CsSF_BiomContainer(NamedTuple):
    magic: int
    numBiomes: int
    biomeIds: List[int]
    biomeGridN: List[int]
    resrcGridN: List[int]
    biomeGridS: List[int]
    resrcGridS: List[int]


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
            return json.load(f)
    except FileNotFoundError:
        print(f"Missing config: {path}")
        return {}


# Load and use config
config = load_json(CONFIG_PATH)
theme = config.get("theme", "Starfield")


def load_biomes(
    input_path: Path,
) -> Tuple[str, Dict[str, List[int]], Set[int], Set[int], Set[int]]:
    with open(input_path, newline="") as f:
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


def zoom_and_fill(
    grid: np.ndarray, zoom_factor: float, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Scales the grid and fills or crops to match target_shape, centering the result."""
    target_h, target_w = target_shape[1], target_shape[0]

    # Scale grid using interpolation
    zoomed_grid = scipy.ndimage.zoom(grid, zoom_factor, order=1)  # Linear interpolation

    # Initialize output grid
    filled_grid = np.zeros((target_h, target_w), dtype=np.float32)

    # Get shapes
    zoomed_h, zoomed_w = zoomed_grid.shape

    if zoom_factor < 1:
        # Zoomed grid is smaller; center it in the output grid
        h_offset = (target_h - zoomed_h) // 2
        w_offset = (target_w - zoomed_w) // 2

        # Ensure offsets are non-negative
        h_offset = max(0, h_offset)
        w_offset = max(0, w_offset)

        # Calculate the region to copy
        h_end = min(h_offset + zoomed_h, target_h)
        w_end = min(w_offset + zoomed_w, target_w)
        copy_h = min(zoomed_h, target_h - h_offset)
        copy_w = min(zoomed_w, target_w - w_offset)

        if copy_h > 0 and copy_w > 0:
            filled_grid[h_offset:h_end, w_offset:w_end] = zoomed_grid[:copy_h, :copy_w]

        # For zoom_factor < 1, extrapolate the outer regions using the edge values
        # to simulate the distortion dying off naturally
        if copy_h > 0 and copy_w > 0:
            # Get the edge values of the zoomed grid
            edge_value = np.mean(
                [
                    zoomed_grid[0, :].mean(),  # Top edge
                    zoomed_grid[-1, :].mean(),  # Bottom edge
                    zoomed_grid[:, 0].mean(),  # Left edge
                    zoomed_grid[:, -1].mean(),  # Right edge
                ]
            )
            # Fill the outer regions with a smooth transition to the edge value
            y = np.linspace(-1, 1, target_h)[:, None]
            x = np.linspace(-1, 1, target_w)[None, :]
            dist = np.sqrt(x**2 + y**2)
            fade = np.clip(dist / dist.max(), 0, 1)
            mask = filled_grid == 0  # Where the grid hasn't been filled
            filled_grid[mask] = edge_value * (
                1 - fade[mask]
            )  # Smooth fade to edge value
    else:
        # Zoomed grid is larger; crop to center it
        h_offset = (zoomed_h - target_h) // 2
        w_offset = (zoomed_w - target_w) // 2

        # Ensure offsets are non-negative
        h_offset = max(0, h_offset)
        w_offset = max(0, w_offset)

        # Crop the zoomed grid to fit target size
        h_end = min(h_offset + target_h, zoomed_h)
        w_end = min(w_offset + target_w, zoomed_w)
        copy_h = min(target_h, h_end - h_offset)
        copy_w = min(target_w, w_end - w_offset)

        if copy_h > 0 and copy_w > 0:
            filled_grid[:copy_h, :copy_w] = zoomed_grid[
                h_offset : h_offset + copy_h, w_offset : w_offset + copy_w
            ]

    return filled_grid


def remap_biome_weights(grid: np.ndarray, weights: List[float]) -> np.ndarray:
    grid = np.clip(grid, 0.0, 1.0)

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = (1.0 / (weights_arr + 1e-6)) ** 0.5
    weights_arr = weights_arr / weights_arr.sum()

    cdf = np.cumsum(weights_arr)
    cdf = np.insert(cdf, 0, 0.0)

    remapped = np.zeros_like(grid, dtype=np.float32)

    n_biomes = len(weights)

    for i in range(n_biomes):
        input_lower = i / n_biomes
        input_upper = (
            (i + 1) / n_biomes if i < n_biomes - 1 else 1.0
        )
        output_lower = cdf[i]
        output_upper = cdf[i + 1]

        mask = (grid >= input_lower) & (grid <= input_upper)

        if mask.any():
            normalized = (grid[mask] - input_lower) / (input_upper - input_lower)
            remapped[mask] = output_lower + normalized * (output_upper - output_lower)

    center_mask = grid == 0.0
    remapped[center_mask] = 0.0

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
            idx = np.clip(idx, 0, n_biomes - 1)
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
        mask: np.ndarray = grid == b  # Boolean mask
        if b in ocean:
            out[mask] = 8
            continue
        ring = rings[b in life]

        dist_out = cast(np.ndarray, distance_transform_edt(~mask))
        dist_in = cast(np.ndarray, distance_transform_edt(mask))

        grad = 0.5 * (dist_out / max(1, dist_out[mask].max())) + 0.5 * (
            1 - dist_in / max(1, dist_in[mask].max())
        )
        for i, val in enumerate(ring):
            band = (grad >= i / 6.5) & (grad < (i + 1) / 6.0)
            out[band & mask] = val

    return out


class BiomFile:
    biomeIds: List[int]
    biomeGridN: np.ndarray
    resrcGridN: np.ndarray
    biomeGridS: np.ndarray
    resrcGridS: np.ndarray

    def __init__(self):
        self.biomeIds = []
        self.biomeGridN = np.array([], dtype=np.uint32)
        self.resrcGridN = np.array([], dtype=np.uint8)
        self.biomeGridS = np.array([], dtype=np.uint32)
        self.resrcGridS = np.array([], dtype=np.uint8)

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = cast(CsSF_BiomContainer, CsSF_Biom.parse_stream(f))
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
            "resrcGridN": self.resrcGridN.tolist(),
            "biomeGridS": self.biomeGridS.tolist(),
            "resrcGridS": self.resrcGridS.tolist(),
        }
        with open(path, "wb") as f:
            CsSF_Biom.build_stream(obj, f)

    def overwrite(self, biome_ids: List[int], grid: np.ndarray):
        """Replace biomes using either noise or structured grid."""
        self.biomeGridN = assign_biomes(grid, biome_ids)
        self.biomeGridS = assign_biomes(grid, biome_ids)
        self.biomeIds = list(set(biome_ids))


def main():
    print("=== Starting PlanetBiomes ===", flush=True)
    preview = "--preview" in sys.argv
    config = load_json(CONFIG_PATH)
    biome_cfg = config
    biome_csv = PREVIEW_PATH if preview else INPUT_DIR

    plugin, planets, life, nolife, ocean = load_biomes(biome_csv)
    out_dir = OUTPUT_DIR / plugin
    out_dir.mkdir(parents=True, exist_ok=True)
    template = BiomFile()
    template.load(TEMPLATE_PATH)

    for planet, biomes in planets.items():
        print(f"Location: {planet}. approved for ({len(biomes)}) biomes.")
        print(
            f"Biom file '{planet}.biom' with {len(biomes)} biomes created in '{out_dir / (planet + '.esm')}'",
            file=sys.stderr,
            flush=True,
        )

        inst = BiomFile()
        inst.load(TEMPLATE_PATH)

        # Step 1: Determine the initial grid size based on zoom_factor
        zoom_factor = biome_cfg["zoom"]
        if zoom_factor < 1:
            # Use a larger grid to capture the full distortion
            scale_factor = 1 / zoom_factor  # e.g., 2.0 for zoom_factor=0.5
            grid_h = int(GRID_SIZE[1] * scale_factor)
            grid_w = int(GRID_SIZE[0] * scale_factor)
        else:
            grid_h, grid_w = GRID_SIZE[1], GRID_SIZE[0]

        # Step 2: Generate Combined Biome Pattern on the larger grid
        pattern = generate_combined_pattern((grid_h, grid_w), biome_cfg)

        # Step 3: Apply Weights (Distortion) on the larger grid
        enable_biases = config.get("enable_biases", False)
        zone_weights = [config.get(f"zone_0{i}", 1.0) for i in range(7)]

        if enable_biases:
            remapped_pattern = remap_biome_weights(
                pattern, zone_weights
            )  # Apply weights on larger grid
        else:
            remapped_pattern = pattern  # Keep pattern unchanged if biases are off

        # Step 4: Zoom to target GRID_SIZE
        zoomed_pattern = zoom_and_fill(
            remapped_pattern, zoom_factor, (GRID_SIZE[1], GRID_SIZE[0])
        )

        # Step 5: Assign Biomes
        inst.overwrite(biomes, zoomed_pattern)

        # Step 6: Assign Resources
        inst.resrcGridN = assign_resources(
            inst.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()
        inst.resrcGridS = assign_resources(
            inst.biomeGridS.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()

        inst.save(out_dir / f"{planet}.biom")

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetTextures.py")], check=True)


if __name__ == "__main__":
    main()
