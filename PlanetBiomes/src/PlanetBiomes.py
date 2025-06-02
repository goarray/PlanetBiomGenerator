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
    binary_dilation,
    zoom,
)
from itertools import cycle
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


CsSF_Biom = Struct(
    "magic" / Const(0x105, UInt16),
    "numBiomes" / Rebuild(UInt32, len_(this.biomeIds)),
    "biomeIds" / Array(this.numBiomes, UInt32),
    Const(2, UInt32),
    Const([GRID_SIZE[0], GRID_SIZE[1]], Array(2, UInt32)),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridN" / Array(GRID_FLATSIZE, UInt32),
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridN" / Array(GRID_FLATSIZE, UInt8),
    Const([GRID_SIZE[0], GRID_SIZE[1]], Array(2, UInt32)),
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


def process_biomes(shape, config):
    handle_news(None)

    # Initialize random generators
    seed = get_seed(config)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Biome processing begins here
    pattern = generate_squircle_pattern(shape, config.get("squircle_factor", 0.5))
    elevation_weight = config.get("elevation_influence", 0.9)

    north_pattern = (1 - elevation_weight) * pattern
    south_pattern = (1 - elevation_weight) * pattern

    zone_keys = sorted(k for k in config if k.startswith("zone_"))
    weights = [config[k] for k in zone_keys]

    north_pattern = tilt_zone_weights(
        north_pattern, tilt_factor=config.get("zoom_factor", 0.5)
    )
    south_pattern = tilt_zone_weights(
        south_pattern, tilt_factor=config.get("zoom_factor", 0.5)
    )

    # If biome processing is disabled, return early
    if not config.get("process_biomes", True):
        # Normalize patterns
        north_pattern = (north_pattern - north_pattern.min()) / (
            north_pattern.max() - north_pattern.min() + 1e-6
        )
        south_pattern = (south_pattern - south_pattern.min()) / (
            south_pattern.max() - south_pattern.min() + 1e-6
        )
        
        return north_pattern, south_pattern

    if config.get("enable_biases", False):
        north_pattern = remap_biome_weights(north_pattern, weights)
        south_pattern = remap_biome_weights(south_pattern, weights)

    if config.get("enable_tectonic_plates", False):
        north_elevation, south_elevation = generate_faults(
            shape,
            number_faults=config.get("number_faults", 4),
            seed=seed,
            temp_dir=TEMP_DIR,
            fault_width=config.get("fault_width", 4),
            rng=rng,
            py_rng=py_rng,
        )
        north_pattern += elevation_weight * (north_elevation * 0.25)
        north_pattern = np.clip(north_pattern, 0.0, 1.0)

        south_pattern += elevation_weight * (south_elevation * 0.25)
        south_pattern = np.clip(south_pattern, 0.0, 1.0)

    if config.get("enable_noise", False):
        north_pattern += generate_noise(shape, config)
        south_pattern += generate_noise(shape, config)

    if config.get("enable_distortion", False):
        north_pattern += generate_distortion(shape, config)
        south_pattern += generate_distortion(shape, config)

    if config.get("enable_smoothing", False):
        north_pattern = gaussian_filter(north_pattern, sigma=8)
        south_pattern = gaussian_filter(south_pattern, sigma=8)

    if config.get("enable_anomalies", False):
        north_pattern = apply_anomalies(north_pattern, config)
        south_pattern = apply_anomalies(south_pattern, config)

    # Normalize patterns
    north_pattern = (north_pattern - north_pattern.min()) / (
        north_pattern.max() - north_pattern.min() + 1e-6
    )
    south_pattern = (south_pattern - south_pattern.min()) / (
        south_pattern.max() - south_pattern.min() + 1e-6
    )

    south_pattern = np.flipud(south_pattern)
    return north_pattern, south_pattern


def generate_faults(shape, number_faults, seed, temp_dir, fault_width, rng, py_rng):
    """
    Generate a single fault map and elevation maps for both hemispheres.
    """
    handle_news(None)
    # Generate edge faults and inward fault lines
    fault_map, edge_points = generate_edge_faults(shape, number_faults, py_rng)
    fault_lines = generate_inward_faults(
        shape, edge_points, seed_offset=seed, rng=rng, py_rng=py_rng
    )

    # Dilate fault lines to widen them
    fault_lines = dilate_fault_lines(
        fault_lines, fault_width=fault_width, py_rng=py_rng
    )

    # Generate plate assignments
    plate_map = rng.integers(0, number_faults, shape, dtype=int)

    # Generate elevation maps for both hemispheres
    north_elevation = generate_plate_elevation(
        shape, plate_map, fault_map, fault_lines, seed, rng
    )
    south_elevation = generate_plate_elevation(
        shape, plate_map, fault_map, fault_lines, seed, rng
    )

    # Flip south elevation for correct orientation
    #south_elevation = np.flipud(south_elevation)

    # Save elevation maps for debugging
    save_elevation_map_png(north_elevation, str(temp_dir), "north")
    save_elevation_map_png(south_elevation, str(temp_dir), "south")

    # Save fault map for debugging
    save_boundary_map_png(fault_lines, str(temp_dir), "unified")

    return north_elevation, south_elevation


def generate_edge_faults(grid_size, number_faults, py_rng):
    """
    Generate fault points on the edges of a single grid, assigning initial directions
    perpendicular to the edge for fault lines.
    """
    h, w = grid_size
    faults_per_edge = max(1, (number_faults * 2) // 4)

    fault_map = np.full(grid_size, -1, dtype=int)
    edges = [
        ("top", (0, 0, 0, w), (1, 0)),  # Downward direction
        ("bottom", (0, h - 1, 0, w), (-1, 0)),  # Upward
        ("left", (1, 0, 0, h), (0, 1)),  # Rightward
        ("right", (1, w - 1, 0, h), (0, -1)),  # Leftward
    ]

    fault_type_cycle = cycle([0, 1])  # Convergent, Divergent
    edge_points = {name: [] for name, *_ in edges}

    for name, (axis, fixed, start, end), direction in edges:
        step = (end - start) / faults_per_edge if faults_per_edge > 0 else end - start
        positions = sorted(
            py_rng.randint(int(start + i * step), int(start + (i + 1) * step))
            for i in range(faults_per_edge)
        )
        for pos in positions:
            y, x = (fixed, pos) if axis == 0 else (pos, fixed)
            fault_type = next(fault_type_cycle)
            fault_map[y, x] = fault_type
            edge_points[name].append(((y, x), fault_type, direction))

    convergent = np.sum(fault_map == 0)
    divergent = np.sum(fault_map == 1)
    handle_news(
        None,
        "debug",
        f"Faults: {convergent} convergent, {divergent} divergent",
    )

    return fault_map, edge_points


def draw_noise_driven_squiggle_line(
    p0, p1, direction, steps=800, distort_scale=5, fault_jitter=0.3, seed=0, rng=None
):
    """
    Draw a fault line from p0 to p1, starting perpendicular to the edge for a short
    distance before becoming squiggly.
    """
    height, width = GRID_SIZE
    fault_jitter = config.get("fault_jitter", 0.5)
    perpendicular_distance = config.get("perpendicular_distance", 0.1) * min(
        height, width
    )

    if rng is None:
        rng = np.random.default_rng(seed)

    noise_x = gaussian_filter(rng.random((height, width)), sigma=distort_scale)
    noise_y = gaussian_filter(rng.random((height, width)), sigma=distort_scale)

    y0, x0 = p0
    y1, x1 = p1
    cx, cy = float(x0), float(y0)
    path = [(int(y0), int(x0))]

    # Calculate initial perpendicular segment
    dx_perp, dy_perp = direction
    perp_steps = int(steps * perpendicular_distance / np.hypot(x1 - x0, y1 - y0))
    perp_steps = max(1, min(perp_steps, steps // 4))

    # Draw straight perpendicular segment
    step_size = perpendicular_distance / perp_steps
    for step in range(1, perp_steps + 1):
        cx += dx_perp * step_size
        cy += dy_perp * step_size
        gx = int(np.clip(round(cx), 0, width - 1))
        gy = int(np.clip(round(cy), 0, height - 1))
        path.append((gy, gx))

    # Continue with noise-driven squiggly path
    remaining_steps = steps - perp_steps
    for step in range(remaining_steps):
        progress = step / (remaining_steps - 1) if remaining_steps > 1 else 1
        curr_dx = x1 - cx
        curr_dy = y1 - cy
        curr_dist = np.hypot(curr_dx, curr_dy)
        curr_angle = np.arctan2(curr_dy, curr_dx)

        gx = int(np.clip(round(cx), 0, width - 1))
        gy = int(np.clip(round(cy), 0, height - 1))
        local_dx = (noise_x[gy, gx] - 0.5) * 2
        local_dy = (noise_y[gy, gx] - 0.5) * 2
        local_angle = np.arctan2(local_dy, local_dx)

        final_angle = (curr_angle + (local_angle * fault_jitter)) * 0.5
        step_size = (
            curr_dist / (remaining_steps - step)
            if step < remaining_steps - 1
            else curr_dist
        )
        step_size = min(step_size, np.hypot(x1 - x0, y1 - y0) / steps * 2)

        cx += np.cos(final_angle) * step_size
        cy += np.sin(final_angle) * step_size

        gx = int(np.clip(round(cx), 0, width - 1))
        gy = int(np.clip(round(cy), 0, height - 1))
        path.append((gy, gx))

        if curr_dist < step_size * 1.5:
            path.append((y1, x1))
            break

    if path[-1] != (y1, x1):
        path.append((y1, x1))

    return list(dict.fromkeys(path))


def generate_inward_faults(
    grid_size, edge_points, seed_offset=0, rng=None, py_rng=None
):
    """
    Generate fault lines connecting edge points, starting perpendicular to edges.
    """
    fault_line_type_map = np.full(grid_size, -1, dtype=int)
    base_seed = 42 + seed_offset

    edge_pairs = [("top", "bottom"), ("left", "right")]

    for edge1, edge2 in edge_pairs:
        starters1 = edge_points[edge1]
        starters2 = edge_points[edge2]

        used_indices = set()
        for i, (p0, t0, direction) in enumerate(starters1):
            candidates = [
                (j, p1, d1)
                for j, (p1, t1, d1) in enumerate(starters2)
                if t1 == t0 and j not in used_indices
            ]
            if not candidates:
                continue

            j_min, p1_closest, _ = min(
                candidates,
                key=lambda item: float(np.linalg.norm(np.subtract(p0, item[1]))),
            )
            used_indices.add(j_min)

            path = draw_noise_driven_squiggle_line(
                p0, p1_closest, direction, base_seed + i, rng=rng
            )
            for y, x in path:
                fault_line_type_map[y, x] = t0

    convergent_lines = np.sum(fault_line_type_map == 0)
    divergent_lines = np.sum(fault_line_type_map == 1)
    handle_news(
        None,
        "debug",
        f"Inward faults: {convergent_lines} convergent, {divergent_lines} divergent",
    )

    return fault_line_type_map


def dilate_fault_lines(fault_map, fault_width=3, fault_smooth=0.5, seed=0, py_rng=None):
    handle_news(None)
    h, w = fault_map.shape
    fault_mask = fault_map >= 0
    fault_width = (config.get("fault_width", 3) * 3)
    fault_smooth = config.get("fault_smooth", 0.5)
    dilated_mask = binary_dilation(fault_mask, iterations=fault_width)

    # Initialize py_rng if not provided
    if py_rng is None:
        py_rng = random.Random(seed)

    dilated_fault_map = np.full((h, w), -1, dtype=int)
    dilated_fault_map[fault_map >= 0] = fault_map[fault_map >= 0]

    dilated_area = (dilated_mask) & (fault_map == -1)
    for y, x in np.argwhere(dilated_area):
        neighbors = [
            fault_map[ny, nx]
            for ny in range(max(0, y - 1), min(h, y + 2))
            for nx in range(max(0, x - 1), min(w, x + 2))
            if fault_map[ny, nx] >= 0
        ]
        if neighbors:
            dilated_fault_map[y, x] = py_rng.choice(neighbors)
        else:
            dilated_fault_map[y, x] = py_rng.randint(0, 2)

    # Log dilated fault coverage
    convergent_dilated = np.sum(dilated_fault_map == 0)
    divergent_dilated = np.sum(dilated_fault_map == 1)
    handle_news(
        None,
        "debug",
        f"Dilated faults: {convergent_dilated} convergent, {divergent_dilated} divergent",
    )

    if fault_smooth > 0:
        smoothed_map = dilated_fault_map.copy()
        for y, x in np.argwhere(dilated_area):
            neighbors = [
                dilated_fault_map[ny, nx]
                for ny in range(max(0, y - 1), min(h, y + 2))
                for nx in range(max(0, x - 1), min(w, x + 2))
                if dilated_fault_map[ny, nx] >= 0
            ]
            if neighbors:
                avg = sum(neighbors) / len(neighbors)
                blended = (
                    fault_smooth * avg + (1 - fault_smooth) * dilated_fault_map[y, x]
                )
                smoothed_map[y, x] = int(round(blended))
        dilated_fault_map = smoothed_map

    return dilated_fault_map


def generate_plate_elevation(grid_size, plate_map, fault_map, fault_lines, seed, rng):
    handle_news(None)
    h, w = grid_size
    elevation_map = np.zeros((h, w), dtype=np.float32)

    for plate_id in np.unique(plate_map):
        plate_mask = plate_map == plate_id
        base_elevation = 0
        elevation_map[plate_mask] = base_elevation

        convergent_mask = (fault_map == 0) | (fault_lines == 0)
        divergent_mask = (fault_map == 1) | (fault_lines == 1)

        convergent_bump = np.zeros((h, w), dtype=np.float32)
        divergent_dip = np.zeros((h, w), dtype=np.float32)

        # Use variable elevation with seeded RNG
        convergent_bump[convergent_mask] = rng.uniform(
            0.1, 0.3, size=np.sum(convergent_mask)
        )
        divergent_dip[divergent_mask] = rng.uniform(
            0.1, 0.3, size=np.sum(divergent_mask)
        )

        elevation_map += convergent_bump
        elevation_map -= divergent_dip

    # Normalize to [0, 1]
    elevation_map = (elevation_map - elevation_map.min()) / (
        elevation_map.max() - elevation_map.min() + 1e-6
    )

    # Log elevation stats
    handle_news(
        None,
        "debug",
        f"Elevation range: {elevation_map.min():.4f} to {elevation_map.max():.4f}",
    )

    return elevation_map


def generate_squircle_pattern(
    shape: Tuple[int, int], squircle_factor: float
) -> np.ndarray:
    handle_news(None)
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


def remap_biome_weights(grid: np.ndarray, weights: List[float]) -> np.ndarray:
    handle_news(None)
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
    handle_news(None)
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


def generate_distortion(shape: tuple[int, int], config: dict) -> np.ndarray:
    """Generate contrast-preserving distortion with large-scale disturbances."""
    handle_news(None)
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
    handle_news(None)
    h, w = grid.shape
    modified_grid = grid.copy()

    # Settings
    enable_equator_anomalies = config.get("enable_equator_anomalies", False)
    enable_seed_anomalies = config.get("enable_seed_anomalies", False)
    enable_polar_anomalies = config.get("enable_polar_anomalies", False)
    equator_anomaly_count = config.get("equator_anomaly_count", 0.5)
    equator_anomaly_spray = config.get("equator_anomaly_spray", 0.5)
    polar_anomaly_count = config.get("polar_anomaly_count", 0.5)
    polar_anomaly_spray = config.get("polar_anomaly_spray", 0.5)

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

    # 1. Equator anomaly (centered at 1.0)
    if enable_equator_anomalies:
        equator_mask = make_mask(center=1.0, width=equator_anomaly_count)
        equator_noise = gaussian_filter(np.random.randn(h, w), sigma=4)
        boosted_equator_strength = 2 * equator_anomaly_spray
        modified_grid += equator_mask * equator_noise * boosted_equator_strength

    # 2. Polar anomaly (centered at 1.0)
    if enable_polar_anomalies:
        polar_mask = make_mask(center=0.0, width=polar_anomaly_count)
        polar_noise = gaussian_filter(np.random.randn(h, w), sigma=4)
        boosted_polar_strength = 2 * polar_anomaly_spray
        modified_grid += polar_mask * polar_noise * boosted_polar_strength

    # 3. Seed-based variant (scattered mid-biome influences)
    if enable_seed_anomalies:
        strength = 0.25 * (equator_anomaly_count * polar_anomaly_count)

        anomaly_noise = gaussian_filter(rng.normal(size=(h, w)), sigma=4)

        # Normalize noise to 0–1 range manually
        noise_min = anomaly_noise.min()
        noise_max = anomaly_noise.max()
        normalized_noise = (anomaly_noise - noise_min) / (noise_max - noise_min + 1e-6)

        # Compress values toward 0.5 to favor mid-biomes
        centered_noise = 0.5 + (normalized_noise - 0.5) * 0.5  # result in [0.25–0.75]

        modified_grid += centered_noise * strength

    return np.clip(modified_grid, 0, 1)


def tilt_zone_weights(grid: np.ndarray, tilt_factor: float) -> np.ndarray:
    handle_news(None)
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


def save_plate_map_png(
    plate_map: np.ndarray, path_out: str, hemisphere: str, suffix: str = ""
):
    """Save plate map as a color PNG image."""
    path = os.path.join(
        path_out, f"temp_{hemisphere}_plate_map{suffix}.png"
    )
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


def save_boundary_map_png(
    boundary_map: np.ndarray, path_out: str, hemisphere: str, suffix: str = ""
):
    path = os.path.join(
        path_out, f"temp_{hemisphere}_boundary_map{suffix}.png"
    )
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


def save_elevation_map_png(
    elevation_map: np.ndarray, path_out: str, hemisphere: str, suffix: str = ""
):
    path = os.path.join(path_out, f"temp_{hemisphere}_elevation_map{suffix}.png")
    os.makedirs(path_out, exist_ok=True)
    norm_elevation = (elevation_map - elevation_map.min()) / (
        elevation_map.max() - elevation_map.min() + 1e-6
    )
    color_image = (norm_elevation * 255).astype(np.uint8)
    image = Image.fromarray(color_image, mode="L")
    image.save(path)
    handle_news(None, "info", f"Elevation map saved to: {path}")

######################################################################################


def assign_biomes(grid: np.ndarray, biome_ids: List[int]) -> np.ndarray:
    handle_news(None)
    if len(biome_ids) == 1:
        return np.full(GRID_FLATSIZE, biome_ids[0], dtype=np.uint32)
    grid = np.clip(grid, 0, 1)  # Ensure grid is in [0, 1]
    mapped = np.zeros(GRID_FLATSIZE, dtype=np.uint32)
    n_biomes = len(biome_ids)
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            i = y * GRID_SIZE[0] + x
            # Non-linear mapping to emphasize zone 0
            value = grid[y, x] ** 1.5  # Increase weight toward lower values (zone 0)
            idx = int(value * n_biomes)
            idx = np.clip(idx, 0, n_biomes - 1)
            mapped[i] = biome_ids[idx]
    return mapped


def assign_resources(
    grid: np.ndarray, life: Set[int], nolife: Set[int], ocean: Set[int]
) -> np.ndarray:
    handle_news(None)
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
    # Add default elevation influence
    config.setdefault("elevation_influence", 0.4)  # Weight of elevation in pattern
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
        north_pattern, south_pattern = process_biomes(grid_dim, config)

        # Sort biomes to ensure low-to-high elevation mapping (optional, if needed)
        # biomes = sorted(biomes)  # Ensure biomes are ordered (e.g., ocean to mountain)
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
