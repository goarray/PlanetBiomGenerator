import numpy as np
import pyvista as pv
from typing import cast
from noise import pnoise3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from PlanetNewsfeed import handle_news
from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from typing import Dict, List, Set, Tuple, NamedTuple, cast
from multiprocessing import Pool, cpu_count
import colorsys
import argparse
import subprocess
import cProfile
import pstats
import random
import json
import csv
import sys
import os
import shutil
from PIL import Image, ImageEnhance
from PlanetUtils import get_biome_colormaps, get_average_biome_humidity, generate_and_save_road_mask, biome_db
from PlanetConstants import (
    save_config,
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


########################################Debug#####################################

# discrete_cmap, gradient_cmap = get_biome_colormaps(config)

def generate_view(mesh, config):
    discrete_cmap, gradient_cmap = get_biome_colormaps(config)

    # ZoneID
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="ZoneID",
        cmap="jet",
        clim=[mesh.point_data["ZoneID"].min(), mesh.point_data["ZoneID"].max()],
        show_scalar_bar=True,
    )
    plotter.show()

    # PolarWeight
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="PolarWeight",
        cmap="coolwarm",
        clim=[0.0, 1.0],
        show_scalar_bar=True,
    )
    plotter.show()

    # BaseHeight
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="BaseHeight",
        cmap="grey",
        clim=[mesh.point_data["BaseHeight"].min(), mesh.point_data["BaseHeight"].max()],
        show_scalar_bar=True,
    )
    plotter.show()

    # Height (with biome_db height context)
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="Height",
        cmap="terrain",
        clim=[mesh.point_data["Height"].min(), mesh.point_data["Height"].max()],
        show_scalar_bar=True,
    )
    plotter.show()

    # BiomeID
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="BiomeID",
        cmap=discrete_cmap,
        clim=[0, 6],
        show_scalar_bar=True,
    )
    plotter.show()

    # ResID
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="ResID",
        cmap="Paired",
        clim=[mesh.point_data["ResID"].min(), mesh.point_data["ResID"].max()],
        show_scalar_bar=True,
    )
    plotter.show()

    # ColorID
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="ColorID",
        cmap=gradient_cmap,
        clim=[0.0, 1.0],
        opacity=0.8,
        show_scalar_bar=True,
    )
    plotter.show()

    # RockID
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="RockID",
        cmap="gist_ncar",
        clim=[mesh.point_data["RockID"].min(), mesh.point_data["RockID"].max()],
        show_scalar_bar=True,
    )
    plotter.show()

    # RiverFlow
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="RiverFlow",
        cmap="grey",
        clim=[0.0, 1.0],
        show_scalar_bar=True,
    )
    plotter.show()

    # MountainMask
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="MountainMask",
        cmap="grey",
        clim=[0.0, 1.0],
        show_scalar_bar=True,
    )
    plotter.show()


#####################################Load########################################


def get_output_paths(plugin_name, planet_name):
    base = PNG_OUTPUT_DIR / plugin_name / planet_name
    return {
        "color": base / f"{planet_name}_color.png",
        "height": base / f"{planet_name}_height.png",
        "ocean_mask": base / f"{planet_name}_ocean_mask.png",
        "biome": base / f"{planet_name}_biome.png",
        "resource": base / f"{planet_name}_resource.png",
        "zone": base / f"{planet_name}_zone.png",
        "terrain": base / f"{planet_name}_terrain.png",
        "river_mask": base / f"{planet_name}_river_mask.png",
        "colony_mask": base / f"{planet_name}_colony_mask.png",
        "humidity": base / f"{planet_name}_humidity.png",
        "mountain_mask": base / f"{planet_name}_mountain_mask.png",
    }

# Helper to retreive settings
def param(cfg, key, default):
    return cfg.get(key, default)


######################################Process#######################################


def process_planet_maps(config):
    mesh = generate_sphere(config)
    assign_latitude_zones(mesh, config)
    apply_base_noise(mesh, config)
    apply_micro_noise(mesh, config)
    assign_uv_from_normal(mesh)
    return mesh  # Return only mesh


#########################################Modules####################################


def generate_sphere(config):
    resolution = config.get("texture_resolution", 512)
    mesh = pv.Sphere(theta_resolution=2 * resolution, phi_resolution=resolution)

    # Save the original normal for use in polar weighting
    original_normals = mesh.points / np.linalg.norm(mesh.points, axis=1, keepdims=True)
    mesh.point_data["OriginalNormal"] = original_normals

    return mesh


def apply_base_noise(mesh, config):
    scale = config.get("base_scale", 0.005)
    freq = config.get("base_frequency", 1.5)
    octaves = config.get("base_octaves", 4)
    biome_order = config.get("biome_order", 0.5)
    noise_shaping = config.get("biome_chaos", 0.5)
    use_random = config.get("use_random", False)

    if use_random:
        seed = random.randint(0, 99999)
        config["user_seed"] = seed
        print(f"[CONFIG] get_seed, ttl_river={config.get("ttl_river")}")
        save_config()
    else:
        seed = int(config.get("user_seed", 0))
    rng = random.Random(seed)

    # Generate random offsets in range [0, 1000)
    offset_x = rng.uniform(0, 1000)
    offset_y = rng.uniform(0, 1000)
    offset_z = rng.uniform(0, 1000)

    polar_weight = tilt_sphere_weights(mesh, biome_order)
    mesh.point_data["PolarWeight"] = polar_weight

    base_equator, base_pole = 0.497, 0.503
    base_heights = np.zeros(mesh.n_points)
    new_points = np.zeros_like(mesh.points)

    for i, p in enumerate(mesh.points):
        normal = p / np.linalg.norm(p)
        nx, ny, nz = normal * freq + np.array([offset_x, offset_y, offset_z])
        base_noise = pnoise3(nx, ny, nz, octaves=octaves)
        base_noise = np.sign(base_noise) * (abs(base_noise) ** (0.5 / noise_shaping))

        base_height = (1 - polar_weight[i]) * base_equator + polar_weight[i] * base_pole
        displacement = base_height + scale * base_noise
        base_heights[i] = displacement
        new_points[i] = displacement * normal

    mesh.points = new_points
    mesh.point_data["BaseHeight"] = base_heights


def tilt_sphere_weights(mesh: pv.PolyData, biome_order: float) -> np.ndarray:
    """
    Tilt bias for a sphere, favoring equator or poles.
    biome_order = 0.0 --> polar bias
    biome_order = 1.0 --> equator bias
    biome_order = 0.5 --> neutral (no bias)
    """
    normals = mesh.point_data["OriginalNormal"]  # unit vectors

    # Latitude weight: abs(dot(normal, Z)) gives 1 at poles, 0 at equator
    pole_weight = np.abs(normals @ np.array([0, 0, 1]))  # shape: (n_points,)

    # Convert to an equator-centric weight
    equator_weight = 1.0 - pole_weight  # 1.0 at equator, 0.0 at poles

    # Blend between pole_weight and equator_weight based on biome_order
    bias_weight = (1 - biome_order) * pole_weight + biome_order * equator_weight

    return np.clip(bias_weight, 0.0, 1.0)


def fbm_noise(normal, base_freq, octaves=6, gain=0.25, lacunarity=2.0):
    value = 0.0
    micro_scale = 0.5
    micro_freq = base_freq
    for _ in range(octaves):
        value += micro_scale * pnoise3(*(normal * micro_freq))
        micro_freq *= lacunarity
        micro_scale *= gain
    return value


def compute_displacement(args):
    p, base_freq, scale, erosion_effect, river_val = args
    micro = fbm_noise(p, base_freq)
    erosion_factor = np.clip(1.0 - erosion_effect * river_val, 0.2, 1.0)
    displacement = 1.0 + scale * micro * erosion_factor
    return p * displacement


def apply_micro_noise(mesh, config):
    scale = config.get("micro_scale", 0.005)
    freq = config.get("micro_freq", 1.5)
    octaves = config.get("micro_octaves", 4)
    erosion_effect = config.get("river_micro_strength", 0.6)

    base_heights = mesh.point_data["BaseHeight"]
    river_flow = mesh.point_data.get("RiverFlow", None)

    river_flow_norm = (
        river_flow.astype(np.float32) / 255.0
        if river_flow is not None
        else np.zeros_like(base_heights)
    )

    args = [
        (p, freq, scale, erosion_effect, river_flow_norm[i])
        for i, p in enumerate(mesh.points)
    ]

    with Pool(processes=cpu_count()) as pool:
        new_points = pool.map(compute_displacement, args, chunksize=512)

    mesh.points = new_points
    final_heights = np.linalg.norm(new_points, axis=1)
    mesh.point_data["MicroDisplacement"] = final_heights - base_heights
    mesh.point_data["Height"] = final_heights

import numpy as np
import pyvista as pv


def assign_humidity_mask(mesh: pv.PolyData, config, biome_db):
    """
    Compute cloud cover based on biome humidity and store as Humidity on the mesh.
    Returns cloud cover array to influence river flow.
    """
    humidity_bias = config.get("humidity_bias", 0.5)

    # Get active biome humidity
    avg_humidity = get_average_biome_humidity(config, biome_db)
    print(f"[Debug] Average biome humidity: {avg_humidity:.3f}")

    # Build biome index -> BiomeEntry map
    editor_ids = list(biome_db.biomes_by_name.keys())
    biome_indices = [config.get(f"biome{i:02}_qcombobox", 0) for i in range(7)]
    active_biome_entries = []

    for idx in biome_indices:
        if idx < len(editor_ids):
            editor_id = editor_ids[idx]
            biome = biome_db.biomes_by_name.get(editor_id)
            if biome:
                active_biome_entries.append(biome)

    # Map index (0-6) to humidity
    index_to_humidity = {i: b.humidity() for i, b in enumerate(active_biome_entries)}

    # Per-point humidity-based cloud cover
    biome_ids = mesh.point_data["BiomeID"]
    cloud_cover = np.array([index_to_humidity.get(bid, 0.0) for bid in biome_ids])

    # Apply humidity bias to cloud cover
    if humidity_bias < 0.5:
        cloud_cover *= humidity_bias * 2
    elif humidity_bias > 0.5:
        cloud_cover += (1.0 - cloud_cover) * ((humidity_bias - 0.5) * 2)

    # Simple blending effect: smooth cloud cover using neighbor averaging
    n_points = mesh.n_points
    neighbors_map = {i: set() for i in range(n_points)}
    for cell in mesh.faces.reshape((-1, 4))[:, 1:]:
        for i in range(3):
            neighbors_map[cell[i]].update(cell[j] for j in range(3) if j != i)

    smoothed_cloud_cover = np.copy(cloud_cover)
    for i in range(n_points):
        neighbors = neighbors_map[i]
        if neighbors:
            neighbor_clouds = [cloud_cover[j] for j in neighbors]
            smoothed_cloud_cover[i] = (cloud_cover[i] + np.mean(neighbor_clouds)) / 2

    # Store and return
    mesh.point_data["Humidity"] = smoothed_cloud_cover
    return smoothed_cloud_cover, avg_humidity


def assign_river_flow(mesh: pv.PolyData, config, biome_db):
    """
    Compute flow accumulation and store as RiverFlow on the mesh, influenced by cloud cover.
    """
    raw_bias = config.get("river_bias", 0.5)
    river_bias = 1 + raw_bias
    height = mesh.point_data["Height"]
    scaled_height = (height - height.min()) / (height.max() - height.min() + 1e-8)
    n_points = mesh.n_points

    # Compute cloud cover
    cloud_cover, avg_cloud_density = assign_humidity_mask(mesh, config, biome_db)
    print(f"[Debug] Average cloud density: {avg_cloud_density:.3f}")
    avg_cloud_density = 1.0 - avg_cloud_density

    # Adjust river sharpness dynamically based on cloud density
    base_power = config.get(river_bias, 1.45)
    adjusted_power = base_power * (
        1.5 - avg_cloud_density * 0.5
    )  # Dampens river cutoff in low-cloud areas
    adjusted_power = max(0.8, min(3.0, adjusted_power))  # Clamp if needed

    # Debug: Check height variation
    print(f"[Debug] Height: min={height.min()}, max={height.max()}, std={height.std()}")

    # Build point-to-neighbors map
    neighbors_map = {i: set() for i in range(n_points)}
    for cell in mesh.faces.reshape((-1, 4))[:, 1:]:
        for i in range(3):
            neighbors_map[cell[i]].update(cell[j] for j in range(3) if j != i)

    # Flow accumulation
    flow_accum = np.zeros(n_points)
    for i in range(n_points):
        my_height = scaled_height[i]
        if cloud_cover[i] < 0.01:
            continue
        # if my_height > max_river_elevation:
        #    continue  # Skip high points

        epsilon = 1e-4
        downhill = [
            j for j in neighbors_map[i] if scaled_height[j] < my_height - epsilon
        ]
        if downhill:
            share = 1.0 / len(downhill)
            for j in downhill:
                flow_accum[j] += share

    # Debug
    print(
        f"[Debug] Flow accumulation: min={flow_accum.min()}, max={flow_accum.max()}, mean={flow_accum.mean()}"
    )

    # Normalize
    flow_accum = flow_accum / (flow_accum.max() + 1e-8)

    # Emphasize strong rivers (power law shaping)
    river_strength = np.power(flow_accum, config.get(raw_bias, 1.45))

    # Cloud cover-based amplification
    cloud_boost = (4500 * river_bias) * (0.5 + avg_cloud_density)
    amplified_flow = river_strength * cloud_boost

    # Cloud density-aware thresholding (cloudier areas allow more rivers)
    threshold = 1.3 + (0.5 - avg_cloud_density) * 4
    river_mask = (amplified_flow > threshold).astype(np.uint8) * 255

    # Optional debug
    print(
        f"[Debug] River value stats: threshold={threshold:.4f}, amplified_flow={amplified_flow}, river_strength={river_strength}"
    )
    print(
        f"[Debug] River image stats: min={river_mask.max():.4f}, max={river_mask.max():.4f}, mean={river_mask.mean():.4f}"
    )

    mesh.point_data["RiverFlow"] = river_mask


def assign_mountain_mask(mesh: pv.PolyData, config: dict):
    height = mesh.point_data["Height"]
    river_flow = mesh.point_data.get("RiverFlow", None)
    if river_flow is None:
        raise ValueError("RiverFlow missing. Run assign_river_flow first.")

    # Normalize height
    scaled_height = (height - height.min()) / (height.max() - height.min() + 1e-8)
    n_points = mesh.n_points

    mountain_bias = config.get("mountain_bias", 0.5)  # [0–1], 1 = lots of mountains

    # Build neighbor map
    neighbors_map = {i: set() for i in range(n_points)}
    for cell in mesh.faces.reshape((-1, 4))[:, 1:]:
        for i in range(3):
            neighbors_map[cell[i]].update(cell[j] for j in range(3) if j != i)

    # Compute gradient (steepness)
    gradient = np.zeros(n_points)
    for i in range(n_points):
        diffs = [abs(scaled_height[j] - scaled_height[i]) for j in neighbors_map[i]]
        gradient[i] = np.mean(diffs) if diffs else 0
    gradient /= gradient.max() + 1e-8

    # Thresholds
    height_cutoff = np.percentile(scaled_height, 50 + mountain_bias * 40)  # 70–90%
    steepness_cutoff = np.percentile(gradient, 50 + mountain_bias * 40)

    height_mask = scaled_height >= height_cutoff
    steepness_mask = gradient >= steepness_cutoff

    mountain_mask = height_mask & steepness_mask

    mountain_biome_indices = {
        i
        for i, b in enumerate(biome_db.active_biomes(config))
        if b.category and "mountain" in b.category.lower()
    }
    for b in biome_db.active_biomes(config):
        print(f"[Check] Active biome: {b.editor_id}, form_id={b.form_id}, category={b.category}")

    # Optional: include biomes tagged as "mountain"
    biome_mask = np.zeros(n_points, dtype=bool)
    if "BiomeID" in mesh.point_data:
        biome_ids = mesh.point_data.get("BiomeID")
        if biome_ids is not None:
            biome_mask = np.isin(biome_ids, list(mountain_biome_indices))
            mountain_mask = mountain_mask | biome_mask

    # mountain_mask = mountain_mask | biome_mask

    # Optional growth
    def grow_mask(mask: np.ndarray, iters=2, bias=0.6):
        new_mask = mask.copy()
        indices = np.arange(len(mask))
        for _ in range(iters):
            np.random.shuffle(indices)
            for i in indices:
                if not new_mask[i]:
                    neighbors = neighbors_map[int(i)]
                    count = sum(new_mask[int(j)] for j in neighbors)
                    if count >= 2 and np.random.rand() < bias:
                        new_mask[i] = True
        return new_mask

    mountain_mask = grow_mask(mountain_mask, iters=2, bias=mountain_bias)

    # Remove river overlap
    river_mask = river_flow < 128
    mountain_mask[river_mask] = False
    print(f"River mask count: {np.sum(river_mask)} points")
    print(f"Mountain points after river removal: {np.sum(mountain_mask)}")

    mesh.point_data["MountainMask"] = mountain_mask.astype(np.uint8) * 255
    print(f"[Mountain] Assigned {mountain_mask.sum()} mountain points.")


def assign_latitude_zones(mesh, config):
    cap_blend = config.get("cap_blend", 0.7)
    cap_curve = config.get("cap_curve", 2.5)

    if "OriginalNormal" not in mesh.point_data:
        raise ValueError("OriginalNormal not found. Must compute before displacement.")

    normals = mesh.point_data["OriginalNormal"]
    z = np.abs(normals[:, 2])  # absolute Z = latitude

    polar_weight = np.clip(z / cap_blend, 0.0, 1.0) ** cap_curve
    mesh.point_data["PolarWeight"] = polar_weight


def assign_zone_ids(mesh: pv.PolyData, config):
    heights = mesh.point_data["Height"]
    # Normalize heights to 0..1 range for colormap mapping
    h_min, h_max = heights.min(), heights.max()
    zone_id = (heights - h_min) / (h_max - h_min + 1e-8)
    mesh.point_data["ZoneID"] = zone_id
    # return zone_id


def assign_biome_ids(mesh, config):
    heights = mesh.point_data["BaseHeight"]
    h_min, h_max = heights.min(), heights.max()
    normalized_heights = (heights - h_min) / (h_max - h_min + 1e-8)

    # Get biome indices and heights from config
    biome_indices = [config.get(f"biome{i:02}_qcombobox", 0) for i in range(7)]
    editor_ids = list(biome_db.biomes_by_name.keys())

    # Map biome indices to biome_db entries
    biome_heights = []
    for idx in biome_indices:
        if idx < len(editor_ids):
            editor_id = editor_ids[idx]
            biome = biome_db.biomes_by_name.get(editor_id)
            if biome:
                biome_heights.append(biome.height)
            else:
                biome_heights.append(0)  # Fallback
        else:
            biome_heights.append(0)

    # Assign biome IDs based on height thresholds
    biome_ids = np.zeros_like(heights, dtype=int)
    height_thresholds = np.linspace(0, 1, 8)  # 7 biomes, 8 boundaries
    for i, height in enumerate(normalized_heights):
        for j in range(7):
            if height >= height_thresholds[j] and height < height_thresholds[j + 1]:
                biome_ids[i] = j
                break
        else:
            biome_ids[i] = 6  # Topmost biome for highest heights

    selected_biomes = []
    for idx in biome_indices:
        if idx < len(editor_ids):
            editor_id = editor_ids[idx]
            biome = biome_db.biomes_by_name.get(editor_id)
            if biome:
                selected_biomes.append(biome)
            else:
                selected_biomes.append(None)
        else:
            selected_biomes.append(None)

    mesh.point_data["BiomeID"] = biome_ids
    mesh.field_data["SelectedBiomes"] = np.array(
        [b.height if b else -1 for b in selected_biomes]
    )
    print(
        f"[Debug] Assigned BiomeIDs: min={biome_ids.min()}, max={biome_ids.max()}, unique={np.unique(biome_ids)}"
    )


def assign_resource_ids(mesh, config):
    heights = mesh.point_data["BaseHeight"]
    # Normalize heights to 0..1 range for colormap mapping
    h_min, h_max = heights.min(), heights.max()
    res_id = (heights - h_min) / (h_max - h_min + 1e-8)
    mesh.point_data["ResID"] = res_id


def assign_color_ids(mesh, config):
    heights = mesh.point_data["BaseHeight"]
    h_min, h_max = heights.min(), heights.max()
    normalized_heights = (heights - h_min) / (h_max - h_min + 1e-8)
    mesh.point_data["ColorID"] = normalized_heights
    print(
        f"[Debug] Assigned ColorIDs: min={normalized_heights.min()}, max={normalized_heights.max()}"
    )


def assign_terrain_ids(mesh, config):
    heights = mesh.point_data["MicroDisplacement"]
    # Normalize heights to 0..1 range for colormap mapping
    h_min, h_max = heights.min(), heights.max()
    normalized_heights = (heights - h_min) / (h_max - h_min + 1e-8)

    mesh.point_data["RockID"] = normalized_heights


#########################################Exports####################################


def export_maps(mesh, output_paths, config):
    num_biomes = config.get("num_biomes", 5)
    num_res_bands = config.get("num_res_bands", 5)
    resolution = config.get("texture_resolution", 256)
    map_shape = (2 * resolution, resolution)  # (height, width)
    biom_shape = (512, 256)  # (height, width)

    if mesh.active_texture_coordinates is None:
        assign_uv_from_normal(mesh)

    colormaps = {
        "color": "gist_earth",
        "ocean_mask": "grey",
        "biome": "tab20",
        "resource": "Paired",
        "zone": "jet",
        "height": "grey",
        "terrain": "terrain",
        "river_mask": "grey",
        "colony_mask": "grey",
        "humidity": "grey",
        "mountain_mask": "grey",
    }

    # Cache common maps
    ocean_mask = export_ocean_mask_map(mesh, biom_shape, config)
    print(
        f"ocean_mask: min={ocean_mask.min()}, max={ocean_mask.max()}, mean={ocean_mask.mean():.4f}, shape={ocean_mask.shape}"
    )

    humidity = export_humidity_map(mesh, biom_shape, config)
    print(
        f"humidity_mask: min={humidity.min():.4f}, max={humidity.max():.4f}, mean={humidity.mean():.4f}, shape={humidity.shape}"
    )

    elevation = export_height_map(mesh, biom_shape)
    print("elevation_map shape:", elevation.shape)

    # Convert to float for safe math operations
    elevation = elevation.astype(np.float64)

    elevation -= elevation.min()
    elevation /= elevation.max() + 1e-8
    elevation = elevation**1.2  # sharpen valleys
    print(
        f"elevation_map boosted: min={elevation.min():.4f}, max={elevation.max():.4f}, mean={elevation.mean():.4f}, shape={elevation.shape}"
    )

    river = export_river_mask_map(mesh, biom_shape, config, ocean_mask)
    print(
        f"river_mask: min={river.min()}, max={river.max()}, mean={river.mean():.4f}, shape={river.shape}"
    )

    export_layers = {
        "color": lambda mesh, res: export_color_map(mesh, res, config),
        "height": lambda mesh, res: export_height_map(mesh, res),
        "zone": lambda mesh, _: export_zone_map(mesh, biom_shape, config),
        "biome": lambda mesh, _: export_biome_map(mesh, biom_shape, config),
        "ocean_mask": lambda mesh, res: ocean_mask,  # Use cached
        "humidity": lambda mesh, res: humidity,  # Use cached
        "river_mask": lambda mesh, res: river,  # Use cached
        "terrain": lambda mesh, res: export_terrain_map(mesh, res),
        "resource": lambda mesh, _: export_resource_map(
            mesh, biom_shape, num_biomes, num_res_bands
        ),
        "mountain_mask": lambda mesh, _: export_mountain_mask_map(
            mesh, biom_shape, config, num_biomes, num_res_bands
        ),
        "colony_mask": lambda mesh, _: export_colony_map(
            config,
            river_mask=river,
            ocean_mask=ocean_mask,
            humidity_map=humidity,
            elevation_map=elevation,
            colony_ratio=config.get("inland_population_count", 0.1),
            coastal_ratio=config.get("coastal_population_count", 0.15),
            humidity_threshold=config.get("humidity_threshold", 0.3),
            coastal_elevation_max=config.get("coastal_population_density", 0.2),
            elevation_max=config.get("inland_population_density", 0.7),
        ),
    }

    for name, path in output_paths.items():
        export_fn = export_layers.get(name)
        if export_fn is None:
            print(f"Warning: No export function for {name}. Skipping.")
            continue

        try:
            res = (
                biom_shape
                if name
                in (
                    "river_mask",
                    "colony_mask",
                    "ocean_mask",
                    "humidity",
                    "biome",
                    "resource",
                    "zone",
                    "mountain_mask",
                )
                else map_shape
            )
            data = export_fn(mesh, res)
            if data is None:
                print(f"Warning: No data returned for {name} map. Skipping.")
                continue

            print(
                f"{name} map: min={np.min(data)}, max={np.max(data)}, shape={data.shape}"
            )
            cmap = colormaps.get(name, "viridis")
            if name == "color" or name == "biome":
                Image.fromarray(data, mode="RGB").save(path)
            elif (
                name == "ocean_mask" or name == "river_mask" or name == "mountain_mask"
            ):
                Image.fromarray(data, mode="L").save(path)
            elif name in ["zone", "resource"]:
                data = np.round(data).astype(np.uint8)
                plt.imsave(path, data, cmap=cmap, vmin=np.min(data), vmax=np.max(data))
            else:
                norm_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                plt.imsave(path, norm_data, cmap=cmap)
            print(f"Saved {name} map to: {path}")

        except Exception as e:
            print(f"Error exporting {name} map: {e}")


def export_ocean_mask_map(mesh: pv.PolyData, map_shape, config) -> np.ndarray:
    height, width = map_shape
    num_biomes = config.get("num_biomes", 7)

    if "BiomeID" not in mesh.point_data:
        print("Warning: BiomeID not found. Returning black ocean mask.")
        return np.ones((height, width), dtype=np.uint8) * 255

    biome_ids = mesh.point_data["BiomeID"]
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UVs missing. Assign UVs before export.")
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)
    x_coords = uvs[:, 0] * (width - 1)  # U -> width
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)  # V -> height, flipped for image

    # Create grid with xy indexing (Cartesian)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))  # (x, y) order

    # Interpolate using NearestNDInterpolator
    points = np.column_stack((x_coords, y_coords))  # (x, y) order
    interpolator = NearestNDInterpolator(points, biome_ids)
    uv_map = interpolator(grid_points).reshape(height, width).astype(int)
    uv_map = np.clip(uv_map, 0, num_biomes - 1)

    # Map BiomeID to ocean slots
    editor_ids = list(biome_db.biomes_by_name.keys())
    ocean_slots = [
        i
        for i in range(num_biomes)
        if config.get(f"biome{i:02}_qcombobox", 0) < len(editor_ids)
        and biome_db.biomes_by_name.get(
            editor_ids[config.get(f"biome{i:02}_qcombobox", 0)], None
        )
        and biome_db.biomes_by_name[
            editor_ids[config.get(f"biome{i:02}_qcombobox", 0)]
        ].category.lower()
        == "ocean"
    ]

    # Create ocean mask: 0 for ocean, 255 for land
    ocean_mask = np.ones((height, width), dtype=np.uint8) * 255
    for slot in ocean_slots:
        ocean_mask[uv_map == slot] = 0

    print(
        f"[Debug] Ocean slots: {ocean_slots}, Ocean pixels: {np.sum(ocean_mask == 0)}"
    )
    return ocean_mask


def export_humidity_map(mesh: pv.PolyData, map_shape, config):
    if "Humidity" not in mesh.point_data:
        raise ValueError("Humidity data not found on mesh.")

    humidity = mesh.point_data["Humidity"]

    # Interpolate and fill using UV logic
    image = remap_to_uv_grid(mesh, humidity, map_shape)

    print(
        f"[Debug] Humidity map: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}"
    )
    return image


def export_river_mask_map(
    mesh: pv.PolyData, map_shape, config, ocean_mask: np.ndarray
) -> np.ndarray:
    if "RiverFlow" not in mesh.point_data:
        print("[Warning] RiverFlow not found. Returning black mask.")
        return np.zeros(map_shape, dtype=np.uint8)

    height, width = map_shape
    river_flow = mesh.point_data["RiverFlow"]

    # Use remap_to_uv_grid for consistency and speed
    river_image = remap_to_uv_grid(mesh, river_flow, map_shape)

    # Minimal smoothing to reduce artifacts
    sigma = config.get("river_blur_sigma", 0.05)  # Reduced sigma
    if sigma > 0:
        river_image = gaussian_filter(river_image, sigma=sigma)

    # Normalize to [0, 1]
    max_val = river_image.max()
    print(
        f"[Debug] River image stats: min={river_image.min():.4f}, max={max_val:.4f}, mean={river_image.mean():.4f}"
    )
    if max_val > 0:
        river_image /= max_val

    river_image = 1.0 - river_image

    # Threshold
    threshold = config.get("river_threshold", 0.05)
    mask = (river_image > threshold).astype(np.uint8) * 255

    # Apply ocean mask
    mask[ocean_mask == 0] = 0

    # Compute ttl_mountain
    ttl_river = config["ttl_river"] = int(np.sum(mask == 255))
    print(f"[Debug] ttl_river: {ttl_river}")

    return mask


def export_mountain_mask_map(
    mesh: pv.PolyData, map_shape, config, *args, **kwargs
) -> np.ndarray:
    height, width = map_shape

    if "MountainMask" not in mesh.point_data:
        print("Warning: MountainMask not found. Returning blank mask.")
        return np.zeros((height, width), dtype=np.uint8)

    mountain_data = mesh.point_data["MountainMask"]

    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UVs missing. Assign UVs before export.")

    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)

    # Correct coordinate transform
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    # Create UV grid (same indexing as image)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Interpolate MountainMask using nearest-neighbor
    points = np.column_stack((x_coords, y_coords))
    interpolator = NearestNDInterpolator(points, mountain_data)
    mountain_mask = interpolator(grid_points).reshape(height, width).astype(np.uint8)

    # Compute ttl_mountain
    ttl_mountain = config["ttl_mountain"] = int(np.sum(mountain_mask == 255))
    print(f"[Debug] ttl_mountain: {ttl_mountain}")

    print(f"[Debug] Mountain pixels: {np.sum(mountain_mask == 255)}")
    return mountain_mask


def export_colony_map(
    config,
    river_mask: np.ndarray,
    ocean_mask: np.ndarray,
    humidity_map: np.ndarray,
    elevation_map: np.ndarray,
    colony_ratio: float = 0.25,
    coastal_ratio: float = 0.25,
    humidity_threshold: float = 0.3,
    coastal_elevation_max: float = 0.2,
    elevation_max: float = 0.7,
    seed: int = 1234,
) -> np.ndarray:
    np.random.seed(seed)

    # Validate input
    assert (
        river_mask.shape
        == ocean_mask.shape
        == humidity_map.shape
        == elevation_map.shape
    ), "Map shapes must match"
    assert river_mask.dtype == np.uint8
    assert ocean_mask.dtype == np.uint8

    # Normalize elevation to [0, 1]
    elev_min, elev_max_actual = elevation_map.min(), elevation_map.max()
    normalized_elevation = (elevation_map - elev_min) / (
        elev_max_actual - elev_min + 1e-8
    )

    ocean_ratio: float = 0.05
    enable_coastal_population = config.get("enable_coastal_population", True)
    enable_inland_population = config.get("enable_inland_population", True)
    enable_ocean_population = config.get("enable_ocean_population", True)

    # Masks
    land_mask = ocean_mask != 0
    land_mask = land_mask.astype(bool)

    distance_to_ocean: np.ndarray = np.asarray(
        distance_transform_edt(~land_mask, return_distances=True)
    )

    coastal_threshold = 2
    coastal_zone = land_mask & (distance_to_ocean <= coastal_threshold)
    coastal_elevation_max = coastal_elevation_max

    coastal_candidates = (
        coastal_zone
        & (humidity_map >= humidity_threshold)
        & (normalized_elevation <= coastal_elevation_max)
    )

    elevation_max = elevation_max
    river_pixels = (river_mask > 0)
    inland_candidates = (
        river_pixels
        & (humidity_map >= humidity_threshold)
        & (normalized_elevation >= 0.0)
        & (normalized_elevation <= 1.0)
    )

    ocean_candidates = ocean_mask == 0
    ocean_indices = np.flatnonzero(ocean_candidates)

    # Create colony mask
    colonies = np.zeros_like(river_mask, dtype=np.uint8)

    # Place coastal colonies
    coastal_ratio = coastal_ratio * 0.01
    coastal_indices = np.flatnonzero(coastal_candidates)
    n_coastal = min(int(len(coastal_indices) * coastal_ratio), len(coastal_indices))
    if enable_coastal_population and n_coastal > 0:
        selected_coastal = np.random.choice(
            coastal_indices, size=n_coastal, replace=False
        )
        rows, cols = np.unravel_index(selected_coastal, colonies.shape)
        colonies[rows, cols] = 255

    # Place inland colonies
    colony_ratio = colony_ratio * 0.05
    inland_indices = np.flatnonzero(inland_candidates)
    n_inland = min(int(len(inland_indices) * colony_ratio), len(inland_indices))
    if enable_inland_population and n_inland > 0:
        selected_inland = np.random.choice(inland_indices, size=n_inland, replace=False)
        rows, cols = np.unravel_index(selected_inland, colonies.shape)
        colonies[rows, cols] = 255

    # Place ocean colonies
    n_total_so_far = n_coastal + n_inland
    n_ocean = max(1, int(n_total_so_far * ocean_ratio))
    if enable_ocean_population and n_ocean > 0 and len(ocean_indices) >= n_ocean:
        selected_ocean = np.random.choice(ocean_indices, size=n_ocean, replace=False)
        rows, cols = np.unravel_index(selected_ocean, colonies.shape)
        colonies[rows, cols] = 255

    ttl_coastal = int(np.sum(colonies[coastal_candidates] > 0))
    ttl_inland = int(np.sum(colonies[inland_candidates] > 0))
    ttl_placed = int(np.sum(colonies > 0))
    ttl_elegible = int(np.sum(coastal_candidates) + np.sum(inland_candidates))

    config["coastal_census_total"] = ttl_coastal
    config["inland_census_total"] = ttl_inland

    print(
        f"[Colonies] Placed: {ttl_placed}, Coastal: {ttl_coastal}, Inland: {ttl_inland}, Total elegible sites: {ttl_elegible}"
    )

    if ttl_placed == 0:
        print("[Warning] No colony sites found. Returning empty colony mask.")

    return colonies


def export_color_map(mesh: pv.PolyData, map_shape, config):
    height, width = map_shape

    if "ColorID" not in mesh.point_data:
        print("Warning: ColorID not found. Using zeros for color map.")
        return np.zeros((height, width, 3), dtype=np.uint8)

    scalar_map = mesh.point_data["ColorID"]
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UVs missing. Assign UVs before export.")
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    points = np.column_stack((x_coords, y_coords))
    values = scalar_map

    interpolator = NearestNDInterpolator(points, values)
    uv_map = (
        interpolator(grid_points)
        .reshape(height, width)
        .astype(np.float32)
    )

    # Normalize to [0, 1]
    uv_map = _fill_nan_safe(uv_map)

    uv_map -= uv_map.min()
    uv_map /= (uv_map.max() + 1e-8)

    _, gradient_cmap = get_biome_colormaps(config)
    colored_image = gradient_cmap(uv_map)[:, :, :3]
    rgb_image = (colored_image * 255).astype(np.uint8)

    print(f"[Debug] Color map stats: min={uv_map.min()}, max={uv_map.max()}")
    return rgb_image


def export_biome_map(mesh: pv.PolyData, biom_shape, config: dict):
    num_biomes = config.get("num_biomes", 7)
    height, width = biom_shape

    if "BiomeID" not in mesh.point_data:
        print("Warning: BiomeID not found. Using zeros for biome map.")
        return np.zeros((height, width, 3), dtype=np.uint8)

    biome_ids = mesh.point_data["BiomeID"]
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UVs missing. Assign UVs before export.")
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    # Create grid with xy indexing
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))  # (x, y) order

    # Interpolate using NearestNDInterpolator
    points = np.column_stack((x_coords, y_coords))  # shape: (N, 2)
    interpolator = NearestNDInterpolator(points, biome_ids)
    uv_map = interpolator(grid_points).reshape(height, width).astype(int)
    uv_map = np.clip(uv_map, 0, num_biomes - 1)

    # Get biome colors from config
    biome_colors_hex = [
        config.get(f"biome{i:02}_color", "#000000") for i in range(num_biomes)
    ]
    biome_colors_rgb = np.array(
        [tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) for c in biome_colors_hex],
        dtype=np.uint8,
    )

    # Create RGB image
    biome_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(num_biomes):
        mask = uv_map == i
        biome_image[mask] = biome_colors_rgb[i]

    print(f"[Debug] Biome map unique IDs: {np.unique(uv_map)}")
    return biome_image


def export_resource_map(mesh: pv.PolyData, map_shape, num_biomes, num_res_bands=5):
    height, width = map_shape

    if "ResID" not in mesh.point_data:
        print("Warning: ResID not found. Using zeros for resource map.")
        return np.zeros((height, width), dtype=np.uint8)

    # Get biome map (quantized BiomeID)
    biome_data = mesh.point_data["BiomeID"]  # Use BiomeID directly
    min_val, max_val = biome_data.min(), biome_data.max()
    range_val = max(max_val - min_val, 1e-8)
    normalized = (biome_data - min_val) / range_val

    # Quantize into discrete biome bins
    biomes = np.floor(normalized * num_biomes).astype(int)
    biomes = np.clip(biomes, 0, num_biomes - 1)

    # Remap to 2D UV grid
    biome_map = remap_to_uv_grid(mesh, biomes, map_shape)
    biome_map = np.round(biome_map).astype(int)  # Ensure integer biome IDs
    biome_map = np.clip(biome_map, 0, num_biomes - 1)

    # Initialize resource map
    resource_image_uint8 = np.zeros((height, width), dtype=np.uint8)

    # Process each biome
    for b in range(num_biomes):
        mask = biome_map == b
        if not np.any(mask):
            continue

        # Compute inward distance transform (distance from edge within biome)
        dist_in = cast(np.ndarray, distance_transform_edt(mask))

        # Normalize distance to [0, 1] within this biome
        max_dist = dist_in.max() if dist_in.max() > 0 else 1.0
        normalized_dist = dist_in / max_dist

        # Create bands based on distance
        bands = np.floor(normalized_dist * num_res_bands).astype(np.uint8)
        bands = np.clip(bands, 0, num_res_bands - 1)

        # Assign unique values for each band in this biome
        # Use b * num_res_bands + band to ensure unique IDs across biomes
        resource_image_uint8[mask] = (b * num_res_bands + bands[mask]).astype(np.uint8)

    return resource_image_uint8


def export_height_map(mesh: pv.PolyData, map_shape):
    """Export height map based on Height data."""
    if "Height" not in mesh.point_data:
        raise ValueError("Height data not found on mesh.")

    heights = mesh.point_data["Height"]
    height, width = map_shape
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UV coordinates not found. Call assign_uv_from_normal first.")

    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    # Create grid with xy indexing
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Interpolate using NearestNDInterpolator
    points = np.column_stack((x_coords, y_coords))
    interpolator = NearestNDInterpolator(points, heights)
    image = interpolator(grid_points).reshape(height, width)

    # Fill NaNs if any
    image = _fill_nan_safe(image)

    return image


def export_terrain_map(mesh: pv.PolyData, map_shape):
    """Export height map based on Height data."""
    if "RockID" not in mesh.point_data:
        raise ValueError("Height data not found on mesh.")

    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UV coordinates not found. Call assign_uv_from_normal first.")

    heights = mesh.point_data["RockID"]
    height, width = map_shape

    # Clip UVs to avoid edge artifacts
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)

    # Convert UVs to pixel coordinates
    x_coords = uvs[:, 0] * (width - 0)  # 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    # Interpolate on the UV grid
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # x_coords and y_coords are derived from UVs
    points = np.column_stack((x_coords, y_coords))  # shape: (N, 2)
    values = heights  # shape: (N,)

    # Create interpolator
    interpolator = NearestNDInterpolator(points, values)

    image = interpolator(grid_points).reshape(height, width)#.astype(np.uint8)

    # Fill remaining NaNs using mean of neighbors (inpainting)
    image = _fill_nan_safe(image)

    return image


def export_zone_map(mesh: pv.PolyData, map_shape, config):
    num_zones = config.get("num_zones", 3)
    height, width = map_shape

    if "BiomeID" not in mesh.point_data:
        print("Warning: BiomeID not found. Using zeros for color map.")
        return np.zeros((height, width, 3), dtype=np.uint8)

    scalar_map = mesh.point_data["BiomeID"]

    # Dynamically get min and max from biome data
    min_val = scalar_map.min()
    max_val = scalar_map.max()
    range_val = max_val - min_val
    if range_val < 1e-8:
        # Avoid division by zero if range is tiny
        range_val = 1e-8

    # Normalize to 0..1 based on actual range
    norm = (scalar_map - min_val) / range_val

    # Quantize into discrete biome bins
    quantized = np.floor(norm * num_zones).astype(int)
    quantized = np.clip(quantized, 0, num_zones - 1)

    # Remap quantized biome indices to 2D UV grid
    uv_map = remap_to_uv_grid(mesh, quantized, map_shape)

    # Use a discrete colormap with num_biomes colors
    colormap = colormaps["jet"]

    # Normalize uv_map again for safe colormap mapping
    uv_min, uv_max = uv_map.min(), uv_map.max()
    uv_range = uv_max - uv_min
    if uv_range < 1e-8:
        uv_range = 1e-8
    norm_uv = (uv_map - uv_min) / uv_range

    colored_image = colormap(norm_uv)

    # Convert RGBA to RGB uint8
    zone_image_uint8 = (colored_image[:, :, :3] * 255).astype(np.uint8)

    return zone_image_uint8


def remap_to_uv_grid(mesh, scalar_data, map_shape):
    """Remap scalar data to 2D UV grid using nearest-neighbor interpolation."""
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UV coordinates not found. Call assign_uv_from_normal first.")

    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)
    height, width = map_shape

    # Corrected this line
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)  # Corrected from `(1.0, - uvs[:, 1])`

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    scalar_data = np.asarray(scalar_data)

    # Handle multi-channel scalar data (e.g., RGB)
    if scalar_data.ndim == 2 and scalar_data.shape[1] > 1:
        channels = []
        for i in range(scalar_data.shape[1]):
            interpolator = NearestNDInterpolator(
                np.column_stack((x_coords, y_coords)), scalar_data[:, i]
            )
            interp = interpolator(grid_points).reshape(height, width)
            interp = _fill_nan_safe(interp)
            channels.append(interp)
        return np.stack(channels, axis=-1)

    # 1D scalar data
    if scalar_data.ndim > 1:
        scalar_data = scalar_data.ravel()

    interpolator = NearestNDInterpolator(
        np.column_stack((x_coords, y_coords)), scalar_data
    )
    image = interpolator(grid_points).reshape(height, width)

    return _fill_nan_safe(image)


def _fill_nan_safe(image):
    """Safely fill NaNs using blurred fallback."""
    nan_mask = np.isnan(image)
    if not np.any(nan_mask):
        return image
    filled = np.copy(image)
    filled[nan_mask] = 0.0
    blurred = gaussian_filter(filled, sigma=1)
    weight = ~nan_mask
    blurred_weight = gaussian_filter(weight.astype(float), sigma=1)
    image[nan_mask] = blurred[nan_mask] / np.maximum(blurred_weight[nan_mask], 1e-6)
    return image


#########################################CoreModules####################################


def assign_original_normals(mesh):
    normals = mesh.points / np.linalg.norm(mesh.points, axis=1, keepdims=True)
    mesh.point_data["OriginalNormal"] = normals


def assign_uv_from_normal(mesh: pv.PolyData, return_uv=False):
    normals = mesh.points / np.linalg.norm(mesh.points, axis=1, keepdims=True)
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    r = np.linalg.norm(normals, axis=1)
    theta = np.arctan2(y, x)  # longitude [-π, π]
    phi = np.arccos(z / r)  # latitude [0, π]

    # Apply hemisphere masks
    north_mask = phi <= np.pi / 2
    south_mask = phi > np.pi / 2

    def normalize_radius(theta):
        return 1.0 / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

    # Compute UVs for each hemisphere
    r_n = (phi[north_mask] / (np.pi / 2)) * normalize_radius(theta[north_mask])
    u_n = 0.5 + 0.5 * r_n * np.cos(theta[north_mask])
    v_n = 0.5 + 0.5 * r_n * np.sin(theta[north_mask])
    v_n = v_n * 0.5 + 0.5  # Top half

    r_s = ((np.pi - phi[south_mask]) / (np.pi / 2)) * normalize_radius(
        theta[south_mask]
    )
    u_s = 0.5 + 0.5 * r_s * np.cos(theta[south_mask])
    v_s = 0.5 + 0.5 * r_s * np.sin(-theta[south_mask])
    v_s = v_s * 0.5  # Bottom half

    # Assemble full texture coordinates
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    u[north_mask] = u_n
    v[north_mask] = v_n
    u[south_mask] = u_s
    v[south_mask] = v_s

    mesh.active_texture_coordinates = np.column_stack((u, v))

    if return_uv:
        return np.column_stack((u, v))


########################################Main#####################################


def main():
    # --- Setup ---

    # Global configuration
    config = get_config()
    print("Config ID:", id(config))
    plugin_name = config.get("plugin_name", "default_plugin")
    planet_name = config.get("planet_name", "default_planet")

    output_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = get_output_paths(plugin_name, planet_name)

    # --- Planet Generation ---
    mesh = process_planet_maps(config)  # Only mesh returned

    # --- Assign IDs ---
    assign_zone_ids(mesh, config)
    assign_biome_ids(mesh, config)
    assign_river_flow(mesh, config, biome_db)
    assign_resource_ids(mesh, config)
    assign_color_ids(mesh, config)
    assign_terrain_ids(mesh, config)
    assign_mountain_mask(mesh, config)

    # --- Export Maps ---
    export_maps(mesh, output_paths, config)
    # generate_view(mesh, config) # Remove '#' for plotter debug

    generate_and_save_road_mask(config)

    print(f"[CONFIG] PlanetMaker main(), ttl_river={config.get("ttl_river")}")
    save_config()
    print(f"[CONFIG] config id={id(config)} from {__file__}")

    if config.get("run_planet_textures", True):
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "PlanetTextures.py")], check=True
        )
    else:
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    cProfile.run("main()", "profile_output.prof")  # Optional: save to file
    # Optional: display sorted stats
    stats = pstats.Stats("profile_output.prof")
    stats.strip_dirs().sort_stats("cumulative").print_stats(25)  # Show top 25
