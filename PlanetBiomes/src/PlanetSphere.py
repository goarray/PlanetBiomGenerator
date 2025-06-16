import numpy as np
import pyvista as pv
from noise import pnoise3
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PlanetNewsfeed import handle_news
from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array
from scipy.ndimage import gaussian_filter
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
from PlanetConstants import (
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
config = {}


def load_config():
    """Load plugin_name from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


config = load_config()
plugin_name = config.get("plugin_name", "default_plugin")
planet_name = config.get("planet_name", "default_planet")


def load_biome_colors(csv_path, used_biome_ids, saturate_factor=None):
    """Load RGB colors for used biome IDs from CSV."""
    if saturate_factor is None:
        saturate_factor = config.get("texture_saturation", 0.29)

    if not isinstance(saturate_factor, float):
        raise TypeError(f"saturate_factor must be a float, got {type(saturate_factor)}")

    biome_colors = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                form_id = int(row[0], 16)
                r, g, b = int(row[2]), int(row[3]), int(row[4])
            except (ValueError, IndexError):
                print(f"Warning: Invalid row in Biomes.csv: {row}. Skipping.")

    return biome_colors


def generate_sphere(resolution=256):
    handle_news(None, "info", "Sphere created!")
    return pv.Sphere(theta_resolution=2 * resolution, phi_resolution=resolution)


def apply_perlin_noise_with_microdetail(
    mesh: pv.PolyData,
    scale=0.1,
    frequency=5.0,
    octaves=4,
    pole_scale=0.01,
    micro_scale=0.01,
    micro_freq=2.0,
    micro_octaves=3,
):
    points = mesh.points.copy()
    new_points = np.zeros_like(points)

    for i, p in enumerate(points):
        normal = p / np.linalg.norm(p)

        # Main base noise
        base_noise = pnoise3(*(normal * frequency), octaves=octaves)
        pole_bias = pole_scale * (0.1 + 0.9 * abs(normal[2]))

        # Micro Perlin noise for fine detail
        micro_noise = pnoise3(*(normal * micro_freq), octaves=micro_octaves)

        # Combine both
        displacement = 0.5 + scale * base_noise + pole_bias + micro_scale * micro_noise
        new_points[i] = displacement * normal

    mesh.points = new_points
    return np.linalg.norm(new_points, axis=1)


def export_height_map(mesh: pv.PolyData, resolution_w=256, resolution_h=512
):
    uvs = mesh.active_texture_coordinates
    heights = mesh.point_data["Height"]

    # Clip UVs just under 1.0 to avoid overflow into outer pixels
    uvs = np.clip(uvs, 0.0, 1.0 - 1e-6)

    # Convert UVs to floating pixel coordinates
    x_coords = uvs[:, 0] * (resolution_w - 1)
    y_coords = (1.0 - uvs[:, 1]) * (resolution_h - 1)

    # Create interpolation grid
    grid_x, grid_y = np.meshgrid(np.arange(resolution_w), np.arange(resolution_h))

    # Interpolate using linear scattered method
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


def assign_uv_from_normal(mesh: pv.PolyData):
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


def main():

    output_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fault_path = output_dir / f"{planet_name}_fault.png"
    biome_path = output_dir / f"{planet_name}_biome.png"
    resource_path = output_dir / f"{planet_name}_resource.png"

    mesh = generate_sphere(resolution=256)
    heights = apply_perlin_noise_with_microdetail(mesh, scale=0.01, frequency=1.0)
    mesh.point_data["Height"] = heights

    assign_uv_from_normal(mesh)  # Use new mapping function

    image = export_height_map(mesh)

    # Save image

    plt.imsave(fault_path, image, cmap="grey")
    print(f"Saved height map to: {fault_path}")
    plt.imsave(biome_path, image, cmap="terrain")
    print(f"Saved height map to: {biome_path}")
    plt.imsave(resource_path, image, cmap="gnuplot2")
    print(f"Saved height map to: {resource_path}")

    """plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="Height",
        cmap="terrain",
        clim=[heights.min(), heights.max()],
        show_scalar_bar=True,
    )
    plotter.show()"""

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetTextures.py")], check=True)


if __name__ == "__main__":
    main()
