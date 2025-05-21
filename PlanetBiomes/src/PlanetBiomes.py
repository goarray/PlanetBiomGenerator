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
    BIOM_DIR,
    SCRIPT_DIR,
    INPUT_DIR,
    PLUGINS_DIR,
    OUTPUT_DIR,
    CONFIG_PATH,
    TEMPLATE_PATH,
    PREVIEW_PATH,
)

# Third Party Libraries
import scipy.ndimage
import numpy as np
from PIL import Image
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


def save_json(path: Path, data: dict):
    """Save dictionary data to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Config saved successfully to {path}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving JSON: {e}")


# Load and use config
config = load_json(CONFIG_PATH)
theme = config.get("theme", "Starfield")
plugin_name = config.get("plugin_name", "default_plugin")


def load_biomes(
    input_path: Path,
) -> Tuple[str, Dict[str, List[int]], Set[int], Set[int], Set[int]]:
    # Respect preview mode immediately
    if config.get("enable_preview_mode", False):
        input_path = PREVIEW_PATH
        csv_files = [PREVIEW_PATH]
        config["plugin_index"] = ["preview.csv"]
    else:
        csv_files = list(INPUT_DIR.glob("*.csv"))
        if not csv_files:
            csv_files.append(PREVIEW_PATH)
            config["enable_preview_mode"] = True
        else:
            config["plugin_index"] = [f.name for f in csv_files]
            selected_index = min(
                config.get("plugin_selected", 0), max(len(csv_files) - 1, 0)
            )
            selected_index = max(0, selected_index)
            input_path = csv_files[selected_index]
            config["enable_preview_mode"] = input_path == PREVIEW_PATH

    print(f"DEBUG: input_path = {input_path}")
    print(f"DEBUG: enable_preview_mode = {config['enable_preview_mode']}")

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
    if dist_to_edge.size == 0:
        raise ValueError("dist_to_edge array is empty! Check input shape.")

    base_pattern = dist_to_edge / (dist_to_edge.max() + 1e-6)

    return base_pattern


def add_distortion(
    grid: np.ndarray, distortion_factor: float, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Scales the grid and fills or crops to match target_shape, centering the result."""
    target_h, target_w = target_shape[1], target_shape[0]

    # Scale grid using interpolation
    distorted_grid = scipy.ndimage.zoom(
        grid, distortion_factor, order=1
    )  # Linear interpolation

    # Initialize output grid
    filled_grid = np.zeros((target_h, target_w), dtype=np.float32)

    # Get shapes
    distorted_h, distorted_w = distorted_grid.shape

    if distortion_factor < 1:
        # distortion grid is finer; center it in the output grid
        h_offset = (target_h - distorted_h) // 2
        w_offset = (target_w - distorted_w) // 2

        # Ensure offsets are non-negative
        h_offset = max(0, h_offset)
        w_offset = max(0, w_offset)

        # Calculate the region to copy
        h_end = min(h_offset + distorted_h, target_h)
        w_end = min(w_offset + distorted_w, target_w)
        copy_h = min(distorted_h, target_h - h_offset)
        copy_w = min(distorted_w, target_w - w_offset)

        if copy_h > 0 and copy_w > 0:
            filled_grid[h_offset:h_end, w_offset:w_end] = distorted_grid[:copy_h, :copy_w]

        # For distortion_factor < 1, extrapolate the outer regions using the edge values
        # to simulate the distortion dying off naturally
        if copy_h > 0 and copy_w > 0:
            # Get the edge values of the distorted grid
            edge_value = np.mean(
                [
                    distorted_grid[0, :].mean(),  # Top edge
                    distorted_grid[-1, :].mean(),  # Bottom edge
                    distorted_grid[:, 0].mean(),  # Left edge
                    distorted_grid[:, -1].mean(),  # Right edge
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
        # distorted grid is larger; crop to center it
        h_offset = (distorted_h - target_h) // 2
        w_offset = (distorted_w - target_w) // 2

        # Ensure offsets are non-negative
        h_offset = max(0, h_offset)
        w_offset = max(0, w_offset)

        # Crop the distorted grid to fit target size
        h_end = min(h_offset + target_h, distorted_h)
        w_end = min(w_offset + target_w, distorted_w)
        copy_h = min(target_h, h_end - h_offset)
        copy_w = min(target_w, w_end - w_offset)

        if copy_h > 0 and copy_w > 0:
            filled_grid[:copy_h, :copy_w] = distorted_grid[
                h_offset : h_offset + copy_h, w_offset : w_offset + copy_w
            ]

    return filled_grid


def remap_biome_weights(grid: np.ndarray, weights: List[float]) -> np.ndarray:
    # Normalize grid
    grid = np.clip(grid, 0.0, 1.0)

    # Normalize weights (inverted and softened)
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = (1.0 / (weights_arr + 1e-6)) ** 0.5
    weights_arr = weights_arr / weights_arr.sum()

    # Build cumulative distribution function
    cdf = np.cumsum(weights_arr)
    cdf = np.insert(cdf, 0, 0.0)  # Add 0.0 at start

    # Remap each grid value through the CDF
    remapped = np.interp(grid, np.linspace(0.0, 1.0, len(cdf)), cdf)

    # Optional: Lock pole center to 0 (if needed)
    remapped[grid == 0.0] = 0.0

    return remapped


def generate_distortion(shape: Tuple[int, int], config: Dict) -> np.ndarray:
    """Generate smooth distortion normalized to 0..1."""
    seed = get_seed(config)
    np.random.seed(seed)
    distortion = np.random.rand(*shape)
    distortion = gaussian_filter(distortion, sigma=16)
    distortion = (distortion - distortion.min()) / (distortion.max() - distortion.min())
    return distortion


def generate_noise(shape: Tuple[int, int], config: Dict) -> np.ndarray:
    """Generate smooth noise with configurable parameters, normalized to 0..1."""
    if not config.get("enable_noise", True):
        return np.zeros(shape, dtype=np.float32)

    seed = get_seed(config)
    np.random.seed(seed)

    # Retrieve noise parameters from config with defaults
    noise_scale = config.get("noise_scale", 0.5)
    noise_amplitude = config.get("noise_amplitude", 0.5)
    biome_perlin = config.get("biome_perlin", 0.5)
    noise_scatter = config.get("noise_scatter", 0.5)
    biome_swap = config.get("biome_swap", 0.5)
    biome_fractal = config.get("biome_fractal", 0.5)

    # Initialize noise array
    noise = np.random.rand(*shape)

    # Apply Perlin-like smoothing with configurable scale
    noise = gaussian_filter(noise, sigma=16 * noise_scale)

    # Apply fractal noise by layering multiple scales
    for i in range(1, int(3 * biome_fractal) + 1):
        scale = noise_scale * (2**i)
        amplitude = noise_amplitude / (2**i)
        layer = gaussian_filter(np.random.rand(*shape), sigma=16 * scale)
        noise += amplitude * layer

    # Apply scatter effect
    if noise_scatter > 0:
        scatter = np.random.rand(*shape) * noise_scatter
        noise = noise + scatter
        noise = np.clip(noise, 0, 1)

    # Apply biome swap effect (randomly swap small patches)
    if biome_swap > 0:
        swap_mask = np.random.rand(*shape) < biome_swap * 0.1
        swap_noise = np.random.rand(*shape)
        noise[swap_mask] = swap_noise[swap_mask]

    # Normalize to 0..1
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    # Apply Perlin weight
    noise = noise * biome_perlin + (1 - biome_perlin) * np.random.rand(*shape)

    # Final amplitude adjustment
    noise = noise * noise_amplitude

    return noise


def stitch_hemispheres(
    grid_north: np.ndarray, grid_south: np.ndarray, blend_strength: float = 1.0
) -> np.ndarray:
    """Blend the hemispheres smoothly at the equator with adjustable transition strength."""
    height, width = grid_north.shape

    # Create a vertical transition mask with adjustable blend_strength
    y = np.linspace(0, 1, height) ** blend_strength  # ✅ Adjust blending curve
    mask = np.tile(y[:, np.newaxis], (1, width))  # Expand mask across width

    # Blend the grids using the mask
    stitched_grid = grid_north * (1 - mask) + grid_south * mask

    return stitched_grid


def generate_combined_pattern(shape: Tuple[int, int], config: Dict) -> np.ndarray:
    """Generate combined base pattern plus optional transformations based on config."""
    base_pattern = generate_base_pattern(shape)

    # Apply distortion if enabled
    if config.get("enable_distortion", False):
        distortion = generate_distortion(shape, config)
        base_pattern += distortion

    # Apply additional effects based on config
    if config.get("enable_noise", False):
        base_pattern += generate_noise(shape, config)

    if config.get("enable_smoothing", False):
        base_pattern = gaussian_filter(base_pattern, sigma=8)

    grid_north = generate_base_pattern((shape[0] // 2, shape[1]))
    grid_south = generate_base_pattern((shape[0] // 2, shape[1]))

    print(f"DEBUG: grid_north range = ({grid_north.min()}, {grid_north.max()})")
    print(f"DEBUG: grid_south range = ({grid_south.min()}, {grid_south.max()})")

    if config.get("stitch_hemispheres", False):
        base_pattern = stitch_hemispheres(grid_north, grid_south)

    # Normalize after applying transformations
    combined = (base_pattern - base_pattern.min()) / (
        base_pattern.max() - base_pattern.min()
    )

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
    global plugin_name

    # Debug: Check if "--preview" is being detected in sys.argv
    print(f"Command-line args: {sys.argv}")
    preview = "--preview" in sys.argv
    print(f"Preview mode: {preview}")

    config = load_json(CONFIG_PATH)
    print(
        f"Loaded config: {config}"
    )  # Debug: Print config to confirm it loads properly

    biome_cfg = config
    biome_csv = PREVIEW_PATH if preview else INPUT_DIR
    print(f"Biome CSV path: {biome_csv}")  # Debug: Confirm correct path selection

    plugin, planets, life, nolife, ocean = load_biomes(biome_csv)
    print(f"Plugin Name from CSV: {plugin}")  # Debug: Check the extracted plugin name

    config["plugin_name"] = plugin  # Set new active plugin
    save_json(CONFIG_PATH, config)
    print(f"Updated plugin name in config: {config['plugin_name']}")

    out_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    print(
        f"Output directory path: {out_dir}"
    )  # Debug: Check if correct output path is set
    out_dir.mkdir(parents=True, exist_ok=True)

    template = BiomFile()
    template.load(TEMPLATE_PATH)
    print(f"Template loaded successfully from: {TEMPLATE_PATH}")

    for planet, biomes in planets.items():
        print(f"Location: {planet}. approved for ({len(biomes)}) biomes.")
        print(
            f"Biom file '{planet}.biom' with {len(biomes)} biomes created in '{out_dir / (planet + '.esm')}'",
            file=sys.stderr,
            flush=True,
        )

        inst = BiomFile()
        inst.load(TEMPLATE_PATH)

        # Step 1: Determine the initial grid size based on distortion_scale
        distortion_factor = biome_cfg["distortion_scale"]
        if distortion_factor < 1:
            # Use a larger grid to capture the full distortion
            scale_factor = 1 / distortion_factor # e.g., 2.0 for distortion_scale=0.5
            grid_h = int(GRID_SIZE[1] * scale_factor)
            grid_w = int(GRID_SIZE[0] * scale_factor)
        else:
            grid_h, grid_w = GRID_SIZE[1], GRID_SIZE[0]

        # Step 2: Generate Combined Biome Pattern on the larger grid
        pattern = generate_combined_pattern((grid_h, grid_w), biome_cfg)

        # Step 3: Apply Distortion first
        distorted_pattern = add_distortion(
            pattern, distortion_factor, (GRID_SIZE[1], GRID_SIZE[0])
        )

        # Step 3: Apply Weights (Distortion) on the larger grid
        enable_biases = config.get("enable_biases", False)
        zone_weights = [config.get(f"zone_0{i}", 1.0) for i in range(7)]

        if enable_biases:
            remapped_pattern = remap_biome_weights(
                distorted_pattern, zone_weights
            )
        else:
            remapped_pattern = distorted_pattern
        
        Image.fromarray((pattern * 255).astype(np.uint8)).save("raw_pattern.png")
        Image.fromarray((distorted_pattern * 255).astype(np.uint8)).save("distorted.png")
        Image.fromarray((remapped_pattern * 255).astype(np.uint8)).save("remapped.png")

        # Step 5: Assign Biomes
        inst.overwrite(biomes, remapped_pattern)

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
