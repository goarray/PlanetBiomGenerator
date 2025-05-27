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
from noise import snoise2, pnoise2
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

        # Generate tectonic data within the function
        plate_map, boundary_map, elevation_map = generate_plate_map(shape, config)

        # Apply elevation influence to pattern
        pattern += config.get("elevation_influence", 0.3) * elevation_map

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

        # Apply boundary effects (e.g., volcanic biomes near subduction zones)
        if config.get("enable_tectonic_biomes", False):
            subduction_mask = boundary_map == 3
            pattern[subduction_mask] += config.get("subduction_biome_boost", 0.2)
            divergent_mask = boundary_map == 2
            pattern[divergent_mask] -= config.get("divergent_biome_depress", 0.1)

        # Normalize before return
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        return pattern

    north = single_hemi(0)
    south = single_hemi(1)

    north, south = stitch_hemispheres(north, blend_px=50)

    # Save tectonic maps for debugging (only for north hemisphere to avoid duplication)
    if config.get("save_tectonic_maps", False):
        plate_map, boundary_map, elevation_map = generate_plate_map(shape, config)
        save_plate_map_png(plate_map, str(TEMP_DIR), "plate_map")
        save_boundary_map_png(boundary_map, str(TEMP_DIR), "boundary_map")
        save_elevation_map_png(elevation_map, str(TEMP_DIR), "elevation_map")

    return north, south


def save_plate_map_png(plate_map: np.ndarray, path_out: str, suffix: str = ""):
    """Save plate map as a color PNG image."""
    path = os.path.join(path_out, f"temp_plate_map{suffix}.png")
    os.makedirs(path_out, exist_ok=True)
    h, w = plate_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    unique_plates = np.unique(plate_map)
    colors = [(np.random.randint(0, 255, 3)).tolist() for _ in unique_plates]
    for plate_id, color in zip(unique_plates, colors):
        color_image[plate_map == plate_id] = color
    image = Image.fromarray(color_image, mode="RGB")
    image.save(path)
    handle_news(None, "info", f"Plate map saved to: {path}")


def save_boundary_map_png(boundary_map: np.ndarray, path_out: str, suffix: str = ""):
    """Save boundary map as a color PNG image."""
    path = os.path.join(path_out, f"temp_boundary_map{suffix}.png")
    os.makedirs(path_out, exist_ok=True)
    color_map = {
        0: (0, 0, 0),  # No boundary
        1: (255, 0, 0),  # Convergent
        2: (0, 255, 0),  # Divergent
        3: (0, 0, 255),  # Subduction
    }
    h, w = boundary_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for key, rgb in color_map.items():
        color_image[boundary_map == key] = rgb
    image = Image.fromarray(color_image, mode="RGB")
    image.save(path)
    handle_news(None, "info", f"Boundary map saved to: {path}")


def save_elevation_map_png(elevation_map: np.ndarray, path_out: str, suffix: str = ""):
    """Save elevation map as a grayscale PNG image."""
    path = os.path.join(path_out, f"temp_elevation_map{suffix}.png")
    os.makedirs(path_out, exist_ok=True)
    norm_elevation = (elevation_map - elevation_map.min()) / (
        elevation_map.max() - elevation_map.min() + 1e-6
    )
    color_image = (norm_elevation * 255).astype(np.uint8)
    image = Image.fromarray(color_image, mode="L")
    image.save(path)
    handle_news(None, "info", f"Elevation map saved to: {path}")


def generate_squircle_pattern(
    shape: Tuple[int, int], squircle_factor: float
) -> np.ndarray:
    h, w = shape
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]

    squircle_factor = np.clip(squircle_factor, 0.0, 1.0)

    if squircle_factor <= 0.3:
        t = squircle_factor / 0.3
        n = 10 - 6 * t  # 10 → 4

        dist = (np.abs(x) ** n + np.abs(y) ** n) ** (1 / n)

        # Inverted warp to push edges inward (concave)
        # cos(pi*x)*cos(pi*y) ranges [-1,1], so square it to [0,1]
        edge_warp = (np.cos(np.pi * x) * np.cos(np.pi * y)) ** 2

        # Add inward warp: stronger when t is near 0, zero at 0.3
        dist += 0.15 * (1 - t) * edge_warp

    # Zone 2: Smooth squircle to circle (0.3–0.7)
    elif squircle_factor <= 0.7:
        t = (squircle_factor - 0.3) / 0.4
        n = 4 - 2 * t  # 4 → 2
        dist = (np.abs(x) ** n + np.abs(y) ** n) ** (1 / n)

    # Zone 3: Circle to diamond (0.7–1.0)
    else:
        t = (squircle_factor - 0.7) / 0.3
        n = 2 - 1 * t  # 2 → 1
        dist = (np.abs(x) ** n + np.abs(y) ** n) ** (1 / n)

    pattern = 1 - dist
    pattern = np.clip(pattern, 0, 1)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)

    return pattern


def distort_coords(height, width, scale=10.0, magnitude=5.0, seed=0):
    np.random.seed(seed)
    noise_x = gaussian_filter(np.random.rand(height, width), sigma=scale)
    noise_y = gaussian_filter(np.random.rand(height, width), sigma=scale)

    dx = (noise_x - 0.5) * magnitude
    dy = (noise_y - 0.5) * magnitude

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    distorted_yy = np.clip(yy + dy, 0, height - 1)
    distorted_xx = np.clip(xx + dx, 0, width - 1)

    return distorted_yy, distorted_xx


def generate_plate_map(
    shape: Tuple[int, int],
    config: Dict,
    num_plates: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a tectonic plate map with boundary types and elevation influences."""
    height, width = shape
    plate_map = np.zeros((height, width), dtype=np.uint8)
    boundary_map = np.zeros(
        (height, width), dtype=np.uint8
    )  # 0: none, 1: convergent, 2: divergent, 3: subduction
    elevation_map = np.zeros((height, width), dtype=np.float32)  # Elevation influence

    np.random.seed(get_seed(config) + 999)
    plate_centers = np.random.randint(0, [height, width], size=(num_plates, 2))
    # Apply distortion directly to plate centers
    jitter = (np.random.rand(num_plates, 2) - 0.5) * 2 * 8.0  # magnitude
    plate_centers = plate_centers + jitter.astype(int)
    plate_centers = np.clip(plate_centers, [0, 0], [height - 1, width - 1])
    plate_vectors = np.random.uniform(
        -1, 1, (num_plates, 2)
    )  # Random motion vectors for plates

    # yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Initialize distances and labels with defaults
    distances = np.full((height, width, 1), np.inf, dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.int32)

    # Handle case where num_plates is 0
    if num_plates == 0:
        return plate_map, boundary_map, elevation_map
    distorted_yy, distorted_xx = distort_coords(
        height, width, scale=40.0, magnitude=115.0, seed=get_seed(config) + 123
    )
    # Assign plate regions using Voronoi
    for i, (cy, cx) in enumerate(plate_centers):
        dist = (distorted_yy - cy) ** 2 + (distorted_xx - cx) ** 2
        if i == 0:
            distances = np.expand_dims(dist, axis=2)
            labels = np.full_like(dist, i)
        else:
            closer = dist < distances[..., 0]
            labels = np.where(closer, i, labels)
            distances = np.where(closer[..., None], dist[..., None], distances)

    plate_map = labels.astype(np.uint8)

    # Detect boundaries and classify them
    boundary_mask = np.zeros_like(plate_map, dtype=bool)
    for i in range(height):
        for j in range(width):
            if i < height - 1 and j < width - 1:
                if (
                    plate_map[i, j] != plate_map[i + 1, j]
                    or plate_map[i, j] != plate_map[i, j + 1]
                ):
                    boundary_mask[i, j] = True

    # Classify boundaries based on plate motion vectors
    for i in range(height):
        for j in range(width):
            if boundary_mask[i, j]:
                current_plate = plate_map[i, j]
                neighbors = []
                if i > 0:
                    neighbors.append(plate_map[i - 1, j])
                if i < height - 1:
                    neighbors.append(plate_map[i + 1, j])
                if j > 0:
                    neighbors.append(plate_map[i, j - 1])
                if j < width - 1:
                    neighbors.append(plate_map[i, j + 1])

                for neighbor in set(neighbors):
                    if neighbor != current_plate:
                        vec1 = plate_vectors[current_plate]
                        vec2 = plate_vectors[neighbor]
                        relative_motion = np.dot(
                            vec1, vec2
                        )  # Dot product to determine convergence/divergence
                        if (
                            relative_motion < -0.2
                        ):  # Convergent (plates moving toward each other)
                            boundary_map[i, j] = 1
                            elevation_map[i, j] += config.get(
                                "convergent_elevation", 0.5
                            )  # Mountain formation
                        elif relative_motion > 0.2:  # Divergent (plates moving apart)
                            boundary_map[i, j] = 2
                            elevation_map[i, j] -= config.get(
                                "divergent_depression", 0.3
                            )  # Rift valleys
                        else:  # Subduction (one plate slides under another)
                            boundary_map[i, j] = 3
                            elevation_map[i, j] += config.get(
                                "subduction_elevation", 0.4
                            )  # Volcanic arcs

    # Smooth elevation map
    elevation_map = gaussian_filter(
        elevation_map, sigma=config.get("elevation_smoothing", 2.0)
    )
    elevation_map = np.clip(elevation_map, -1.0, 1.0)  # Normalize to [-1, 1]

    return plate_map, boundary_map, elevation_map


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
    raw_scale = config.get("distortion_scale", 0.5)
    # Map [0.1, 1.0] → [0.1, 0.5]
    scale = 0.1 + ((np.clip(raw_scale, 0.1, 1.0) - 0.1) / 0.9) * 0.4

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
        equator_mask = make_mask(center=1.0, width=equator_anomaly_scale)
        equator_noise = gaussian_filter(np.random.randn(h, w), sigma=4)
        boosted_equator_strength = 2 * equator_anomaly_count
        modified_grid += equator_mask * equator_noise * boosted_equator_strength

    # 2. Polar anomaly (centered at 0.0)
    if enable_polar_anomalies:
        polar_mask = make_mask(center=0.0, width=polar_anomaly_scale)
        polar_noise = gaussian_filter(np.random.randn(h, w), sigma=4)
        boosted_polar_strength = 2 * polar_anomaly_count
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


def smoothstep(t):
    """Smoothstep function for smoother interpolation between 0 and 1."""
    return t * t * (3 - 2 * t)


def stitch_hemispheres(
    north: np.ndarray, blend_px: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    h, w = north.shape
    original = north.copy()

    def smoothstep(t):
        return t * t * (3 - 2 * t)

    ramp = np.array([smoothstep((blend_px - i) / blend_px) for i in range(blend_px)])

    # --- Left-right blending ---
    lr_blend = original.copy()
    left_edge = original[:, :blend_px].copy()
    right_edge = original[:, -blend_px:].copy()

    for i in range(blend_px):
        wgt = ramp[i]
        blended = (1 - wgt) * left_edge[:, i] + wgt * right_edge[:, i]
        lr_blend[:, i] = blended
        lr_blend[:, w - blend_px + i] = blended

    # --- Top-bottom blending ---
    tb_blend = original.copy()
    top_edge = original[:blend_px, :].copy()
    bottom_edge = original[-blend_px:, :].copy()

    for j in range(blend_px):
        wgt = ramp[j]
        blended = (1 - wgt) * top_edge[j, :] + wgt * bottom_edge[j, :]
        tb_blend[j, :] = blended
        tb_blend[h - blend_px + j, :] = blended

    # --- Combine both blends ---
    combined = original.copy()

    # Blend interior (non-edge) pixels directly from original
    combined[blend_px : h - blend_px, blend_px : w - blend_px] = original[
        blend_px : h - blend_px, blend_px : w - blend_px
    ]

    # Combine left-right edges from lr_blend
    combined[:, :blend_px] = lr_blend[:, :blend_px]
    combined[:, w - blend_px :] = lr_blend[:, w - blend_px :]

    # Combine top-bottom edges from tb_blend
    combined[:blend_px, :] = tb_blend[:blend_px, :]
    combined[h - blend_px :, :] = tb_blend[h - blend_px :, :]

    # For corners where both blends overlap, average both
    for i in range(blend_px):
        for j in range(blend_px):
            # top-left corner
            combined[i, j] = (lr_blend[i, j] + tb_blend[i, j]) / 2
            # top-right corner
            combined[i, w - blend_px + j] = (
                lr_blend[i, w - blend_px + j] + tb_blend[i, w - blend_px + j]
            ) / 2
            # bottom-left corner
            combined[h - blend_px + i, j] = (
                lr_blend[h - blend_px + i, j] + tb_blend[h - blend_px + i, j]
            ) / 2
            # bottom-right corner
            combined[h - blend_px + i, w - blend_px + j] = (
                lr_blend[h - blend_px + i, w - blend_px + j]
                + tb_blend[h - blend_px + i, w - blend_px + j]
            ) / 2

    # South hemisphere vertical flip
    south = np.flipud(combined)

    return combined, south


def save_biome_grid_png_img(
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
    handle_news(None, "info", f"Biome color grid saved to: {biome_path}")


def save_resource_grid_png_img(resource_grid: np.ndarray, path_out: str):
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
        None, "header", f"PlanetBiomes: Resource grid saved to: {resource_path}"
    )


######################################################################################


def assign_biomes(grid: np.ndarray, biome_ids: List[int]) -> np.ndarray:
    if len(biome_ids) == 1:
        return np.full(GRID_FLATSIZE, biome_ids[0], dtype=np.uint32)
    epsilon = 1e-6
    grid = np.clip(grid, 0, None)  # remove negative values
    grid = grid / np.max(grid)     # normalize to [0,1]
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
            "resrcGridN": [int(x) for x in self.resrcGridN.flatten()],
            "biomeGridS": self.biomeGridS.tolist(),
            "resrcGridS": [int(x) for x in self.resrcGridS.flatten()],
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

    preview = "--preview" in sys.argv
    config = load_json(CONFIG_PATH)
    biome_csv = PREVIEW_PATH if preview else INPUT_DIR
    plugin, planets, life, nolife, ocean = load_biomes(biome_csv)
    config["plugin_name"] = plugin
    save_json(CONFIG_PATH, config)

    out_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    out_dir.mkdir(parents=True, exist_ok=True)

    template = BiomFile()
    template.load(TEMPLATE_PATH)

    for planet, biomes in planets.items():
        handle_news(
            None,
            "info",
            f"PlanetBiomes: Biom file '{planet}.biom' with {len(biomes)} biomes created in '{out_dir / (planet + '.esm')}'",
        )

        inst = BiomFile()
        inst.load(TEMPLATE_PATH)

        grid_dim = GRID_SIZE
        north_pattern, south_pattern = generate_hemisphere_patterns(grid_dim, config)

        inst.overwrite(biomes, north_pattern, south_pattern)
        inst.resrcGridN = assign_resources(
            inst.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()
        inst.resrcGridS = assign_resources(
            inst.biomeGridS.reshape(GRID_SIZE[1], GRID_SIZE[0]), life, nolife, ocean
        ).flatten()

        save_resource_grid_png_img(
            inst.resrcGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]), str(TEMP_DIR)
        )

        used_biome_ids = set(inst.biomeGridN.flatten()) | set(inst.biomeGridS.flatten())
        biome_data = load_biome_data(str(CSV_PATH), used_biome_ids)
        biome_colors = {k: v["color"] for k, v in biome_data.items()}
        save_biome_grid_png_img(
            inst.biomeGridN.reshape(GRID_SIZE[1], GRID_SIZE[0]),
            biome_colors,
            str(TEMP_DIR),
        )

        inst.save(out_dir / f"{planet}.biom")

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetTextures.py")], check=True)


if __name__ == "__main__":
    main()
