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
from PlanetTextures import load_biome_data
from PlanetNewsfeed import handle_news
from PlanetConstants import (
    TEMP_DIR,
    CSV_PATH,
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
import noise
from noise import snoise2
import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    gaussian_laplace,
    zoom,
)
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array


# Constants
GRID_SIZE = (256, 256)
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
        handle_news(None, "error", f"Missing config: {path}")
        return {}


def save_json(path: Path, data: dict):
    """Save dictionary data to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        handle_news(None, "warn", f"Config saved successfully to {path}")
    except Exception as e:
        handle_news(None, "error", f"Error saving JSON: {e}")


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

    handle_news(None, "into", f"PlanetBiomes: Plugin's csv input_path = {input_path}")
    handle_news(None, "into", f"DEBUG: enable_preview_mode = {config['enable_preview_mode']}")

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
        seed = random.randint(0, 9999999)
        config["user_seed"] = seed
        return seed
    return int(config.get("user_seed", 0))


#############################################################################


def generate_hemisphere_patterns(
    shape: Tuple[int, int], config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    def single_hemi(seed_offset: int) -> np.ndarray:
        np.random.seed(get_seed(config) + seed_offset)
        pattern = generate_squircle_pattern(shape, config.get("squircle_factor", 0.5))

        zone_keys = sorted(k for k in config if k.startswith("zone_"))
        weights = [config[k] for k in zone_keys]

        pattern = tilt_zone_weights(pattern, tilt_factor=config.get("zoom_factor", 1.0))

        if config.get("enable_biases", False):        
            pattern = remap_biome_weights(pattern, weights)

        if config.get("enable_noise", False):
            pattern += generate_noise(shape, config)

        if config.get("enable_distortion", False):
            pattern += generate_distortion(shape, config)

        if config.get("enable_smoothing", False):
            pattern = gaussian_filter(pattern, sigma=8)

        if config.get("enable_anomalies", False):
            pattern = apply_anomalies(pattern, config)

        # Normalize before return
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        return pattern

    north = single_hemi(0)
    south = single_hemi(1)

    north, south = stitch_hemispheres(north, blend_px=50)

    return north, south


def generate_squircle_pattern(
    shape: Tuple[int, int], squircle_factor: float
) -> np.ndarray:
    h, w = shape
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]

    # Map squircle_factor from [0, 1] to n:
    # n = 10 for square, n = 2 for circle, n = 1 for diamond
    if squircle_factor <= 0.5:
        n = 10 - squircle_factor * 16  # 10 → 2 as squircle_factor goes 0 → 0.5
    else:
        n = 2 - (squircle_factor - 0.5) * 2  # 2 → 1 as squircle_factor goes 0.5 → 1

    dist = (np.abs(x) ** n + np.abs(y) ** n) ** (1 / n)
    pattern = 1 - dist
    pattern = np.clip(pattern, 0, 1)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)

    return pattern


def remap_biome_weights(grid: np.ndarray, weights: List[float]) -> np.ndarray:
    # Normalize grid
    grid = np.clip(grid, 0.0, 1.0)

    # Convert weights to numpy array
    weights_arr = np.array(weights, dtype=np.float32)

    # Invert weights so that lower weights get more area
    inv_weights = 1.0 / (weights_arr + 1e-6)
    inv_weights = inv_weights / inv_weights.sum()

    # Build cumulative distribution function (CDF)
    cdf = np.cumsum(inv_weights)
    cdf = np.insert(cdf, 0, 0.0)

    # Remap grid using the inverted CDF
    remapped = np.interp(grid, np.linspace(0.0, 1.0, len(cdf)), cdf)

    # Lock pole center if needed
    remapped[grid == 0.0] = 0.0

    return remapped


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
    noise = gaussian_filter(noise, sigma=64 * noise_scale)

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


def polar_projection_correction(grid_shape=(256, 256)) -> np.ndarray:
    h, w = grid_shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / r.max()

    # Inverse projection correction: outer edges (equator) need stronger weight
    correction = 1 / (r_norm + 0.01)  # Add epsilon to avoid divide by zero
    correction /= correction.max()  # Normalize to 0–1

    return correction


def generate_distortion(shape: tuple[int, int], config: dict) -> np.ndarray:
    """Generate contrast-preserving distortion with large-scale disturbances."""
    h, w = shape
    scale = np.clip(config.get("distortion_scale", 0.5), 0.0, 1.0)

    # Initialize random number generator
    seed = get_seed(config)
    rng = np.random.default_rng(seed)

    # Generate low-frequency simplex noise for large-scale distortions
    freq = 0.02 * (1.0 + scale)  # Lower frequency for larger patterns
    noise_grid = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            noise_grid[i, j] = snoise2(
                i * freq,
                j * freq,
                octaves=2,
                persistence=0.5,
                lacunarity=2.0,
                base=seed,
            )

    # Add a layer of higher-frequency noise for detail
    high_freq_noise = rng.normal(loc=0.0, scale=0.3, size=(h, w))
    combined_noise = noise_grid + 0.3 * high_freq_noise  # Blend low and high frequency

    # Apply light smoothing to reduce jaggedness but preserve large features
    sigma = max(0.5, 3.0 * (1.0 - scale))  # Lower sigma for sharper, larger distortions
    smoothed_noise = gaussian_filter(combined_noise, sigma=sigma)

    # Normalize to [-1, 1] range
    smoothed_noise -= smoothed_noise.mean()
    smoothed_noise /= np.abs(smoothed_noise).max()

    # Scale to desired distortion strength
    distortion = smoothed_noise * scale

    # Ensure zero-centered output
    return distortion


def apply_anomalies(grid: np.ndarray, config: Dict) -> np.ndarray:
    h, w = grid.shape
    modified_grid = grid.copy()

    # Settings
    enable_equator_anomalies = config.get("enable_equator_anomalies", False)
    enable_seed_anomalies = config.get("enable_seed_anomalies", False)
    enable_polar_anomalies = config.get("enable_polar_anomalies", False)
    equator_anomaly_count = config.get("equator_anomaly_count", 0.5)
    equator_anomaly_scale = config.get("equator_anomaly_scale", 0.5)
    polar_anomaly_count = config.get("polar_anomaly_count", 0.5)
    polar_anomaly_scale = config.get("polar_anomaly_scale", 0.5)

    seed = get_seed(config)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Utility: Get radial distance (0 = center, 1 = edge)
    def radial_mask(h, w):
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return dist / dist.max()

    radial = radial_mask(h, w)

    def make_mask(center: float, width: float):
        """Smooth bump centered at radius with adjustable width (0.01 to 1.0)."""
        d = np.abs(radial - center)
        falloff = np.clip(1.0 - (d / width) ** 2, 0.0, 1.0)
        return falloff

    # 1. Equator anomaly (centered at 0.5 radius)
    if enable_equator_anomalies:
        equator_mask = make_mask(center=0.5, width=equator_anomaly_scale)
        equator_noise = gaussian_filter(np.random.randn(h, w), sigma=8)
        boosted_equator_strength = equator_anomaly_count
        modified_grid += equator_mask * equator_noise * boosted_equator_strength

    # 2. Polar anomaly (centered at 0.0)
    if enable_polar_anomalies:
        polar_mask = make_mask(center=0.0, width=polar_anomaly_scale)
        polar_noise = gaussian_filter(np.random.randn(h, w), sigma=8)
        boosted_polar_strength = polar_anomaly_count
        modified_grid += polar_mask * polar_noise * boosted_polar_strength

    # 3. Seed-based variant (randomized masks + noise scales)
    if enable_seed_anomalies:

        def normalize_seed(seed: int, min_val=0.1, max_val=1.0) -> float:
            return min_val + (seed % 1000) / 1000 * (max_val - min_val)

        equator_strength = normalize_seed(seed, 0.2, 1.0)
        equator_width = normalize_seed(seed + 1, 0.2, 0.7)
        polar_strength = normalize_seed(seed + 2, 0.2, 1.0)
        polar_width = normalize_seed(seed + 3, 0.2, 0.7)

        equator_mask = make_mask(0.5, equator_width)
        equator_noise = gaussian_filter(rng.normal(size=(h, w)), sigma=[3, 3])
        modified_grid += equator_mask * equator_noise * equator_strength

        polar_mask = make_mask(0.0, polar_width)
        polar_noise = gaussian_filter(rng.normal(size=(h, w)), sigma=3)
        modified_grid += polar_mask * polar_noise * polar_strength

    return np.clip(modified_grid, 0, 1)


def tilt_zone_weights(grid: np.ndarray, tilt_factor: float) -> np.ndarray:
    center = np.array(grid.shape) / 2
    y_indices, x_indices = np.indices(grid.shape)
    distances = np.sqrt((x_indices - center[1]) ** 2 + (y_indices - center[0]) ** 2)
    distances /= distances.max()

    middle_ring = 0.5
    distance_from_middle = np.abs(distances - middle_ring)

    # Invert and scale the bias effect
    bias_strength = 1.0 - distance_from_middle * 2.0
    bias_strength = np.clip(bias_strength, 0.0, 1.0)

    if tilt_factor < 0.5:
        # Favor edges: invert and scale
        weight_map = 1.0 - bias_strength * (1.0 - 2 * tilt_factor)
    elif tilt_factor > 0.5:
        # Favor center: scale bias toward center
        weight_map = 1.0 + bias_strength * (2 * (tilt_factor - 0.5))
    else:
        weight_map = np.ones_like(grid)  # Neutral

    skewed = grid * weight_map
    return np.clip(skewed, 0.0, 1.0)


import numpy as np


def stitch_hemispheres(
    north: np.ndarray, blend_px: int = 8, blend_strength: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blend edges of the north hemisphere with its horizontal mirror,
    and force seamless left-right wrapping.

    Returns north and south (mirror of north).
    """
    h, w = north.shape
    blend_strength = np.clip(blend_strength, 0.01, 1.0)

    # Mirror for equator blending
    mirror = np.fliplr(north)

    # Distance from edge (top, bottom, left, right)
    y = np.minimum(np.arange(h), np.arange(h)[::-1]).reshape(-1, 1)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]).reshape(1, -1)
    dist_to_edge = np.minimum(y, x)

    # Weight map: 1 at edge, 0 in center
    weight = np.clip(1.0 - dist_to_edge / blend_px, 0, 1) ** (1.0 / blend_strength)

    # Equator-blend with mirror
    north_blended = (1 - weight) * north + weight * mirror

    # --- Force left-right wraparound ---
    # Average left and right edge columns
    left_col = north_blended[:, 0]
    right_col = north_blended[:, -1]
    avg_col = (left_col + right_col) / 2

    north_blended[:, 0] = avg_col
    north_blended[:, -1] = avg_col

    # Optionally smooth inward by one pixel
    if blend_px > 1:
        north_blended[:, 1] = (north_blended[:, 1] + avg_col) / 2
        north_blended[:, -2] = (north_blended[:, -2] + avg_col) / 2

    # South is flipped version of north
    south = np.fliplr(north_blended)

    return north_blended, south


######################################################################################


def assign_biomes(grid: np.ndarray, biome_ids: List[int]) -> np.ndarray:
    if len(biome_ids) == 1:
        return np.full(GRID_FLATSIZE, biome_ids[0], dtype=np.uint32)
    epsilon = 1e-6
    grid = np.clip(grid, 0, 1 - epsilon)
    grid = np.power(grid, 1.5)
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


def save_biome_grid(
    grid: np.ndarray,
    biome_colors: dict[int, tuple[int, int, int]],
    path_out: str,
    suffix: str = "",
):
    """Save biome grid as a color PNG image based on biome_colors mapping."""
    biome_path = os.path.join(path_out, "temp_biome.png")
    os.makedirs(path_out, exist_ok=True)

    h, w = grid.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            form_id = int(grid[y, x])
            color = biome_colors.get(form_id, (128, 128, 128))  # fallback: neutral gray
            color_image[y, x] = color

    image = Image.fromarray(color_image, mode="RGB")
    image.save(biome_path)
    handle_news(
        None, "info", f"Biome color grid saved to: {biome_path}"
    )


def save_resource_grid(resource_grid: np.ndarray, path_out: str):
    """Save resource grid as a color PNG image with distinct resource band colors."""
    resource_path = os.path.join(path_out, "temp_resource.png")
    os.makedirs(path_out, exist_ok=True)

    # Color mapping: 0–4 (life group), 80–84 (nolife group), 8 (ocean), 88 (mixed)
    color_map = {
        0: (128, 0, 0),  # Maroon
        1: (255, 0, 0),  # Red
        2: (255, 128, 0),  # Orange
        3: (255, 204, 0),  # Mustard
        4: (255, 255, 0),  # Bright Yellow
        8: (0, 0, 0),  # Ocean (Black)
        80: (153, 204, 255),  # Light Blue
        81: (0, 255, 255),  # Cyan
        82: (0, 128, 255),  # Blue
        83: (0, 200, 0),  # Green
        84: (0, 255, 0),  # Bright Green
        88: (255, 255, 255),  # None (White)
    }

    h, w = resource_grid.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for key, rgb in color_map.items():
        color_image[resource_grid == key] = rgb

    resource_image = Image.fromarray(color_image, mode="RGB")
    resource_image.save(resource_path)
    handle_news(
        None,
        "header",
        f"PlanetBiomes: Resource grid saved to: {resource_path}"
    )


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

    def overwrite(self, biome_ids: List[int], grid_n: np.ndarray, grid_s: np.ndarray):
        """Replace biomes using separate grids for north and south hemispheres."""
        self.biomeGridN = assign_biomes(grid_n, biome_ids)
        self.biomeGridS = assign_biomes(grid_s, biome_ids)
        self.biomeIds = list(set(biome_ids))


def main():
    handle_news(None, "success", f"=== Starting PlanetBiomes ===", flush=True)
    global plugin_name

    # Debug: Check if "--preview" is being detected in sys.argv
    handle_news(
        None, "debug", f"PlanetBiomes: Command-line args: {sys.argv}"
    )
    preview = "--preview" in sys.argv
    handle_news(
        None, "debug", f"PlanetBiomes: Preview mode: {preview}"
    )

    config = load_json(CONFIG_PATH)
    handle_news(
        None, "debug", f"PlanetBiomes: Loaded config: {config}"
    )  # Debug: Print config to confirm it loads properly

    biome_cfg = config
    biome_csv = PREVIEW_PATH if preview else INPUT_DIR
    handle_news(
        None, "debug", f"PlanetBiomes: Biome CSV path: {biome_csv}"
    )  # Debug: Confirm correct path selection

    plugin, planets, life, nolife, ocean = load_biomes(biome_csv)
    handle_news(
        None, "debug", f"PlanetBiomes: Plugin Name from CSV: {plugin}"
    )  # Debug: Check the extracted plugin name

    config["plugin_name"] = plugin  # Set new active plugin
    save_json(CONFIG_PATH, config)
    handle_news(
        None,
        "debug",
        f"PlanetBiomes: Updated plugin name in config: {config['plugin_name']}"
    )

    out_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    handle_news(
        None,
        "debug",
        f"PlanetBiomes: Output directory path: {out_dir}"
    )  # Debug: Check if correct output path is set
    out_dir.mkdir(parents=True, exist_ok=True)

    template = BiomFile()
    template.load(TEMPLATE_PATH)
    handle_news(
        None,
        "warn",
        f"Template loaded successfully from: {TEMPLATE_PATH}"
    )

    for planet, biomes in planets.items():
        print(f"Location: {planet}. approved for ({len(biomes)}) biomes.")
        handle_news(
            None,
            "info",
            f"PlanetBiomes: Biom file '{planet}.biom' with {len(biomes)} biomes created in '{out_dir / (planet + '.esm')}'"
        )

        inst = BiomFile()
        inst.load(TEMPLATE_PATH)

        north_pattern, south_pattern = generate_hemisphere_patterns(
            (GRID_SIZE), biome_cfg
        )

        # Step 5: Assign Biomes
        inst.overwrite(biomes, north_pattern, south_pattern)

        # Step 6: Assign Resources
        inst.resrcGridN = assign_resources(
            inst.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()
        inst.resrcGridS = assign_resources(
            inst.biomeGridS.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()

        save_resource_grid(inst.resrcGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), str(TEMP_DIR))

        used_biome_ids = set(inst.biomeGridN.flatten()) | set(inst.biomeGridS.flatten())
        biome_data = load_biome_data(str(CSV_PATH), used_biome_ids)
        biome_colors = {k: v["color"] for k, v in biome_data.items()}
        save_biome_grid(inst.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), biome_colors, str(TEMP_DIR))

        inst.save(out_dir / f"{planet}.biom")

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetTextures.py")], check=True)


if __name__ == "__main__":
    main()
