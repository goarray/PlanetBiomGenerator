import numpy as np
import pyvista as pv
from typing import cast
from noise import pnoise3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from scipy.interpolate import griddata
from PlanetNewsfeed import handle_news
from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from typing import Dict, List, Set, Tuple, NamedTuple, cast
import colorsys
import argparse
import subprocess
import json
import csv
import sys
import os
import shutil
from PIL import Image, ImageEnhance
from PlanetUtils import get_biome_colormaps, biome_db
from PlanetConstants import (
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

# Global configuration
config = get_config()

########################################Debug#####################################

discrete_cmap, gradient_cmap = get_biome_colormaps(config)


def generate_view(mesh):
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


#####################################Load########################################


def get_config():
    """Load plugin_name from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


config = get_config()


def load_biome_colors_from_config_or_data(
    selected_biome_ids: list[int],
) -> dict[int, tuple[float, float, float]]:
    biome_colors = {}
    editor_ids = list(biome_db.biomes_by_name.keys())

    for idx in selected_biome_ids:
        key = f"biome{idx:02}_color"
        hex_color = config.get(key)
        config_idx = config.get(f"biome{idx:02}_qcombobox", 0)

        if hex_color:
            rgb = mcolors.to_rgb(hex_color)
        else:
            if config_idx < len(editor_ids):
                editor_id = editor_ids[config_idx]
                biome = biome_db.biomes_by_name.get(editor_id)
                if biome:
                    r, g, b = biome.color
                    rgb = (r / 255.0, g / 255.0, b / 255.0)
                else:
                    rgb = (0.0, 0.0, 0.0)
            else:
                rgb = (0.0, 0.0, 0.0)

        biome_colors[idx] = rgb

    print(f"[Debug] Loaded biome colors: {biome_colors}")
    return biome_colors


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
    resolution = config.get("resolution", 512)
    mesh = pv.Sphere(theta_resolution=2 * resolution, phi_resolution=resolution)

    # Save the original normal for use in polar weighting
    original_normals = mesh.points / np.linalg.norm(mesh.points, axis=1, keepdims=True)
    mesh.point_data["OriginalNormal"] = original_normals

    return mesh


def apply_base_noise(mesh, config):
    scale = config.get("base_scale", 0.005)
    freq = config.get("base_frequency", 1.5)
    octaves = config.get("base_octaves", 4)
    tilt_factor = config.get("tilt_factor", 0.5)

    polar_weight = tilt_sphere_weights(mesh, tilt_factor)
    mesh.point_data["PolarWeight"] = polar_weight

    base_equator, base_pole = 0.498, 0.502

    base_heights = np.zeros(mesh.n_points)
    new_points = np.zeros_like(mesh.points)

    for i, p in enumerate(mesh.points):
        normal = p / np.linalg.norm(p)
        base_noise = pnoise3(*(normal * freq), octaves=octaves)
        base_height = (1 - polar_weight[i]) * base_equator + polar_weight[i] * base_pole
        displacement = base_height + scale * base_noise
        base_heights[i] = displacement
        new_points[i] = displacement * normal

    mesh.points = new_points
    mesh.point_data["BaseHeight"] = base_heights


def tilt_sphere_weights(mesh: pv.PolyData, tilt_factor: float) -> np.ndarray:
    """
    Tilt bias for a sphere, favoring equator or poles.
    tilt_factor = 0.0 --> polar bias
    tilt_factor = 1.0 --> equator bias
    tilt_factor = 0.5 --> neutral (no bias)
    """
    normals = mesh.point_data["OriginalNormal"]  # unit vectors

    # Latitude weight: abs(dot(normal, Z)) gives 1 at poles, 0 at equator
    pole_weight = np.abs(normals @ np.array([0, 0, 1]))  # shape: (n_points,)

    # Convert to an equator-centric weight
    equator_weight = 1.0 - pole_weight  # 1.0 at equator, 0.0 at poles

    # Blend between pole_weight and equator_weight based on tilt_factor
    bias_weight = (1 - tilt_factor) * pole_weight + tilt_factor * equator_weight

    return np.clip(bias_weight, 0.0, 1.0)


def apply_micro_noise(mesh, config):
    scale = config.get("micro_scale", 0.005)
    freq = config.get("micro_freq", 1.5)
    octaves = config.get("micro_octaves", 4)
    erosion_effect = config.get("river_micro_strength", 0.6)

    def fbm_noise(normal, base_freq, octaves=6, gain=0.25, lacunarity=2.0):
        value = 0.0
        micro_scale = 0.5
        micro_freq = base_freq
        for _ in range(octaves):
            value += micro_scale * pnoise3(*(normal * micro_freq))
            micro_freq *= lacunarity
            micro_scale *= gain
        return value

    base_heights = mesh.point_data["BaseHeight"]
    river_flow = mesh.point_data.get("RiverFlow", None)

    # Normalize RiverFlow if available
    if river_flow is not None:
        river_flow_norm = river_flow.astype(np.float32) / 255.0
    else:
        river_flow_norm = np.zeros_like(base_heights)

    new_points = np.zeros_like(mesh.points)

    for i, p in enumerate(mesh.points):
        normal = p / np.linalg.norm(p)
        micro_noise = fbm_noise(normal, freq, octaves=octaves)
        erosion_factor = 1.0 - erosion_effect * river_flow_norm[i]
        displacement = 1.0 + scale * micro_noise * erosion_factor
        new_points[i] = p * displacement

    mesh.points = new_points
    final_heights = np.linalg.norm(new_points, axis=1)

    mesh.point_data["MicroDisplacement"] = final_heights - base_heights
    mesh.point_data["Height"] = final_heights


def assign_river_flow(mesh: pv.PolyData, config):
    """
    Compute flow accumulation and store as RiverFlow on the mesh.
    """
    height = mesh.point_data["Height"]
    scaled_height = (height - height.min()) / (height.max() - height.min() + 1e-8)
    n_points = mesh.n_points

    # Get active biome humidity
    active_biomes = biome_db.active_biomes(config)
    avg_humidity = np.mean([b.humidity() for b in active_biomes])
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

    # Per-point humidity
    biome_ids = mesh.point_data["BiomeID"]
    humidity_mask = np.array([
        index_to_humidity.get(bid, 0.0) for bid in biome_ids
    ])

    # Adjust river sharpness dynamically
    base_power = config.get("river_power", 1.45)
    adjusted_power = base_power * (1.0 - avg_humidity * 0.5)  # Dampens river cutoff on dry planets
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
        if humidity_mask[i] < 0.01:
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
    river_strength = np.power(flow_accum, config.get("river_power", 1.45))

    # Humidity-based amplification
    humidity_boost = 1600 * (0.5 + avg_humidity)
    amplified_flow = river_strength * humidity_boost

    # Humidity-aware thresholding (wetter planets allow more rivers)
    threshold = 5 + (0.5 - avg_humidity) * 4
    river_mask = (amplified_flow > threshold).astype(np.uint8) * 255

    # Optional debug
    print(
        f"[Debug] River image stats: min={river_mask.min():.4f}, max={river_mask.max():.4f}, mean={river_mask.mean():.4f}"
    )

    mesh.point_data["RiverFlow"] = river_mask
    mesh.point_data["Humidity"] = humidity_mask


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

    mesh.point_data["BiomeID"] = biome_ids
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
    """Export various maps from the mesh to image files."""
    num_biomes = config.get("num_biomes", 5)
    num_res_bands = config.get("num_res_bands", 5)
    resolution = config.get("resolution", 256)
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
    }

    river = export_river_mask_map(mesh, biom_shape, config)
    print("river_mask shape:", river.shape)

    ocean = export_ocean_mask_map(mesh, biom_shape)
    print("ocean_mask shape:", ocean.shape)

    humidity = export_humidity_map(mesh, biom_shape, config)
    print("humidity_map shape:", humidity.shape)

    elevation = export_height_map(mesh, biom_shape)
    print("elevation_map shape:", elevation.shape)

    export_layers = {
        "color": lambda mesh, res: export_color_map(mesh, res),
        "height": lambda mesh, res: export_height_map(mesh, res),
        "zone": lambda mesh, _: export_zone_map(mesh, biom_shape),
        "biome": lambda mesh, _: export_biome_map(mesh, biom_shape, config),
        "ocean_mask": lambda mesh, res: export_ocean_mask_map(mesh, res),
        "humidity": lambda mesh, res: export_humidity_map(mesh, res, config),
        "river_mask": lambda mesh, res: export_river_mask_map(mesh, res, config),
        "terrain": lambda mesh, res: export_terrain_map(mesh, res),
        "resource": lambda mesh, _: export_resource_map(
            mesh, biom_shape, num_biomes, num_res_bands
        ),
        "colony_mask": lambda mesh, _: export_colony_map(
            river_mask=river,
            ocean_mask=ocean,
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
            data = export_fn(mesh, map_shape)
            if data is None:
                print(f"Warning: No data returned for {name} map. Skipping.")
                continue

            print(
                f"{name} map: min={np.min(data)}, max={np.max(data)}, shape={data.shape}"
            )
            cmap = colormaps.get(name, "viridis")
            if name == "color" or name == "biome":
                # RGB data
                Image.fromarray(data, mode="RGB").save(path)
            elif name == "ocean_mask" or name == "river_mask":
                # Grayscale binary mask
                Image.fromarray(data, mode="L").save(path)
            elif name in ["zone", "resource"]:
                # Categorical data
                data = np.round(data).astype(np.uint8)
                plt.imsave(path, data, cmap=cmap, vmin=np.min(data), vmax=np.max(data))
            else:
                # Continuous grayscale data
                norm_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                plt.imsave(path, norm_data, cmap=cmap)
            print(f"Saved {name} map to: {path}")
        except Exception as e:
            print(f"Error exporting {name} map: {e}")


def export_ocean_mask_map(mesh: pv.PolyData, map_shape) -> np.ndarray:
    height, width = map_shape
    num_biomes = config.get("num_biomes", 7)

    if "BiomeID" not in mesh.point_data:
        print("Warning: BiomeID not found. Returning black ocean mask.")
        return np.ones((height, width), dtype=np.uint8) * 255  # Default to all land (white)

    biome_ids = mesh.point_data["BiomeID"]
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UVs missing. Assign UVs before export.")
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Interpolate BiomeID to grid
    uv_map = griddata(
        points=np.column_stack((x_coords, y_coords)),
        values=biome_ids,
        xi=(grid_x, grid_y),
        method="nearest",
        fill_value=0,
    ).astype(int)
    uv_map = np.clip(uv_map, 0, num_biomes - 1)

    # Map BiomeID to editor_id via config
    editor_ids = list(biome_db.biomes_by_name.keys())
    ocean_slots = []
    for i in range(num_biomes):
        config_idx = config.get(f"biome{i:02}_qcombobox", 0)
        if config_idx < len(editor_ids):
            editor_id = editor_ids[config_idx]
            biome = biome_db.biomes_by_name.get(editor_id)
            if biome and biome.category.lower() == "ocean":
                ocean_slots.append(i)

    # Create ocean mask: 0 for ocean, 255 for land
    ocean_mask = np.ones((height, width), dtype=np.uint8) * 255  # Default to land (white)
    for slot in ocean_slots:
        ocean_mask[uv_map == slot] = 0  # Set ocean to black

    print(f"[Debug] Ocean slots: {ocean_slots}, Ocean pixels: {np.sum(ocean_mask == 0)}")
    return ocean_mask


def export_humidity_map(mesh: pv.PolyData, map_shape, config):
    if "Humidity" not in mesh.point_data:
        raise ValueError(
            "Humidity data not found on mesh. Make sure assign_river_flow() ran first."
        )

    biome_ids = mesh.point_data.get("BiomeID")
    if biome_ids is None:
        raise ValueError("BiomeID not found on mesh")

    # Build biome index → humidity from config
    editor_ids = list(biome_db.biomes_by_name.keys())
    biome_indices = [config.get(f"biome{i:02}_qcombobox", 0) for i in range(7)]
    active_biome_entries = []

    for idx in biome_indices:
        if idx < len(editor_ids):
            editor_id = editor_ids[idx]
            biome = biome_db.biomes_by_name.get(editor_id)
            if biome:
                active_biome_entries.append(biome)

    index_to_humidity = {i: b.humidity() for i, b in enumerate(active_biome_entries)}

    # Assign per-point humidity
    humidity = np.array(
        [index_to_humidity.get(bid, 0.0) for bid in biome_ids], dtype=np.float32
    )

    # Interpolate and fill using UV logic
    image = remap_to_uv_grid(mesh, humidity, map_shape)

    print(
        f"[Debug] Humidity map: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}"
    )
    return image


def export_river_mask_map(mesh: pv.PolyData, map_shape, config) -> np.ndarray:
    """
    Export binary river mask as 256x512 grayscale image using RiverFlow data.
    """
    if "RiverFlow" not in mesh.point_data:
        print("[Warning] RiverFlow not found. Returning black mask.")
        return np.zeros(map_shape, dtype=np.uint8)

    uv = mesh.active_texture_coordinates
    if uv is None:
        raise ValueError("UV coordinates missing on mesh. Assign UVs before export.")

    height, width = map_shape
    river_flow = mesh.point_data["RiverFlow"]

    # Flip UV V to match vertical image axis
    u = (uv[:, 0] * (width - 1)).astype(int)
    v = ((1.0 - uv[:, 1]) * (height - 1)).astype(int)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    river_image = np.zeros(map_shape, dtype=np.float32)
    counts = np.zeros(map_shape, dtype=np.float32)

    for i in range(len(river_flow)):
        river_image[v[i], u[i]] += river_flow[i]
        counts[v[i], u[i]] += 1

    counts[counts == 0] = 1
    river_image /= counts

    # Smooth to reduce artifacts
    river_image = gaussian_filter(
        river_image, sigma=config.get("river_blur_sigma", 0.1)
    )

    # Normalize to [0, 1]
    max_val = river_image.max()
    print(
        f"[Debug] River image stats: min={river_image.min():.4f}, max={max_val:.4f}, mean={river_image.mean():.4f}"
    )
    if max_val > 0:
        river_image /= max_val

    # Threshold
    threshold = config.get("river_threshold", 0.05)  # default 5% of max
    mask = (river_image > threshold).astype(np.uint8) * 255

    # Apply ocean mask
    ocean_mask = export_ocean_mask_map(mesh, map_shape)
    mask[ocean_mask == 0] = 0

    return mask


def export_colony_map(
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

    # Masks
    # land_mask: True on land, False on ocean
    land_mask = ocean_mask != 0

    # distance_to_ocean: distance of each pixel (land or ocean) to nearest ocean pixel
    distance_to_ocean: np.ndarray = np.asarray(
        distance_transform_edt(~land_mask, return_distances=True)
    )  # distance from ocean (False) to land (True)

    coastal_threshold = 5
    coastal_zone = land_mask & (distance_to_ocean <= coastal_threshold)

    inland_zone = land_mask & (~coastal_zone)

    river_inland = (river_mask > 0) & inland_zone

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
        & (normalized_elevation > coastal_elevation_max)
        & (normalized_elevation <= elevation_max)
    )

    # Create colony mask
    colonies = np.zeros_like(river_mask, dtype=np.uint8)

    # Place coastal colonies
    coastal_ratio = coastal_ratio * 0.25
    coastal_indices = np.flatnonzero(coastal_candidates)
    n_coastal = int(len(coastal_indices) * coastal_ratio)
    if n_coastal > 0 and len(coastal_indices) > 0:
        selected_coastal = np.random.choice(
            coastal_indices, size=n_coastal, replace=False
        )
        rows, cols = np.unravel_index(selected_coastal, colonies.shape)
        colonies[rows, cols] = 255

    # Place inland colonies
    colony_ratio = colony_ratio * 0.25
    inland_indices = np.flatnonzero(inland_candidates)
    n_inland = int(len(inland_indices) * colony_ratio)
    if n_inland > 0 and len(inland_indices) > 0:
        selected_inland = np.random.choice(inland_indices, size=n_inland, replace=False)
        rows, cols = np.unravel_index(selected_inland, colonies.shape)
        colonies[rows, cols] = 255

    ttl_elegible = np.sum(coastal_candidates) + np.sum(inland_candidates)
    ttl_placed = np.sum(colonies > 0)
    ttl_coastal = np.sum(colonies[coastal_candidates] > 0)
    ttl_inland = np.sum(colonies[inland_candidates] > 0)

    print(
        f"[Colonies] Placed: {ttl_placed}, Coastal: {ttl_coastal}, Inland: {ttl_inland}, Total elegible sites: {ttl_elegible}"
    )

    if np.sum(colonies > 0) == 0:
        print("[Warning] No colony sites found. Returning empty colony mask.")

    return colonies


def export_color_map(mesh: pv.PolyData, map_shape):
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

    pad = 2
    grid_x, grid_y = np.meshgrid(np.arange(width + 2 * pad), np.arange(height + 2 * pad))
    x_coords_padded = x_coords + pad
    y_coords_padded = y_coords + pad

    uv_map = griddata(
        points=np.column_stack((x_coords_padded, y_coords_padded)),
        values=scalar_map,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

    # Fill and crop
    uv_map = _fill_nan_safe(uv_map)
    uv_map = uv_map[pad:-pad, pad:-pad] 

    # Apply gradient colormap
    _, gradient_cmap = get_biome_colormaps(config)
    colored_image = gradient_cmap(uv_map)[:, :, :3]  # RGB only
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

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    uv_map = griddata(
        points=np.column_stack((x_coords, y_coords)),
        values=biome_ids,
        xi=(grid_x, grid_y),
        method="nearest",
        fill_value=0,
    ).astype(int)

    # Clip to valid biome indices
    uv_map = np.clip(uv_map, 0, num_biomes - 1)

    # Get biome colors from config
    biome_colors_hex = [
        config.get(f"biome{i:02}_color", "#000000") for i in range(num_biomes)
    ]
    biome_colors_rgb = np.array(
        [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in biome_colors_hex],
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

    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UV coordinates not found. Call assign_uv_from_normal first.")

    heights = mesh.point_data["Height"]
    height, width = map_shape

    # Clip UVs to avoid edge artifacts
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)

    # Convert UVs to pixel coordinates
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    # Create interpolation grid
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Interpolate using linear method
    image = griddata(
        points=np.column_stack((x_coords, y_coords)),
        values=heights,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

    # Fill remaining NaNs using mean of neighbors (inpainting)
    nan_mask = np.isnan(image)
    if np.any(nan_mask):
        from scipy.ndimage import gaussian_filter

        # Use Gaussian blur as a smooth fallback
        filled = np.copy(image)
        filled[nan_mask] = 0.0
        blurred = gaussian_filter(filled, sigma=1)
        weight = ~nan_mask
        blurred_weight = gaussian_filter(weight.astype(float), sigma=1)
        image[nan_mask] = blurred[nan_mask] / np.maximum(blurred_weight[nan_mask], 1e-6)

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
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    # Create interpolation grid
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Interpolate using linear method
    image = griddata(
        points=np.column_stack((x_coords, y_coords)),
        values=heights,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

    # Fill remaining NaNs using mean of neighbors (inpainting)
    nan_mask = np.isnan(image)
    if np.any(nan_mask):
        from scipy.ndimage import gaussian_filter

        # Use Gaussian blur as a smooth fallback
        filled = np.copy(image)
        filled[nan_mask] = 0.0
        blurred = gaussian_filter(filled, sigma=1)
        weight = ~nan_mask
        blurred_weight = gaussian_filter(weight.astype(float), sigma=1)
        image[nan_mask] = blurred[nan_mask] / np.maximum(blurred_weight[nan_mask], 1e-6)

    return image


def export_zone_map(mesh: pv.PolyData, map_shape):
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
    """Remap scalar data (1D or multi-channel) to a UV grid using interpolation."""
    uvs = mesh.active_texture_coordinates
    if uvs is None:
        raise ValueError("UV coordinates not found. Call assign_uv_from_normal first.")

    uvs = np.minimum(uvs, 1.0)
    uvs = np.maximum(uvs, 0.0)
    height, width = map_shape
    x_coords = uvs[:, 0] * (width - 1)
    y_coords = (1.0 - uvs[:, 1]) * (height - 1)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    scalar_data = np.asarray(scalar_data)

    # Handle multi-channel scalar data (e.g., RGB)
    if scalar_data.ndim == 2 and scalar_data.shape[1] > 1:
        channels = []
        for i in range(scalar_data.shape[1]):
            interp = griddata(
                points=np.column_stack((x_coords, y_coords)),
                values=scalar_data[:, i],
                xi=(grid_x, grid_y),
                method="linear",
                fill_value=np.nan,
            )
            interp = _fill_nan_safe(interp)
            channels.append(interp)
        return np.stack(channels, axis=-1)

    # 1D scalar data
    if scalar_data.ndim > 1:
        scalar_data = scalar_data.ravel()

    image = griddata(
        points=np.column_stack((x_coords, y_coords)),
        values=scalar_data,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

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
    config = get_config()
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
    assign_river_flow(mesh, config)
    assign_resource_ids(mesh, config)
    assign_color_ids(mesh, config)
    assign_terrain_ids(mesh, config)

    # --- Export Maps ---
    export_maps(mesh, output_paths, config)
    # generate_view(mesh)

    if config.get("run_planet_textures", True):
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "PlanetTextures.py")], check=True
        )
    else:
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
