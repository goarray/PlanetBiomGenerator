#!/usr/bin/env python3
"""
Planet Textures Generator

Generates PNG and DDS texture images for planet biomes based on .biom files.
Outputs four maps per hemisphere: color, surface and ocean.
Applies effects like noise, elevation, shading, craters, and edge blending
to create realistic planetary visuals when process_images is True.
When process_images is False, generates simple texture maps directly from biome data.
Uses configuration from JSON and biome colors from CSV.
Converts PNGs to DDS format using texconv.exe for Starfield compatibility.

Dependencies:
- Python 3.8+
- construct
- scipy
- numpy
- PIL (Pillow)
- colorsys
- json
- csv
- pathlib
- texconv.exe (in textconv/ subdirectory)
"""

from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8, Array
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import Dict, List, Set, Tuple, NamedTuple, cast
import colorsys
import argparse
import subprocess
import json
import csv
import os
import sys
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
    TEXTURE_OUTPUT_DIR,
    PNG_OUTPUT_DIR,
    # Config and data files
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    CSV_PATH,
    PREVIEW_PATH,
    # Script and template paths
    SCRIPT_PATH,
    TEMPLATE_PATH,
    PREVIEW_BIOME_PATH,
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


class CsSF_BiomContainer(NamedTuple):
    magic: int
    numBiomes: int
    biomeIds: List[int]
    biomeGridN: List[int]
    resrcGridN: List[int]
    biomeGridS: List[int]
    resrcGridS: List[int]


# Define .biom file structure
CsSF_Biom = Struct(
    "magic" / Const(0x105, UInt16),
    "_numBiomes" / Rebuild(UInt32, len_(this.biomeIds)),
    "biomeIds" / Array(this._numBiomes, UInt32),
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


def load_config():
    """Load plugin_name from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


config = load_config()
plugin_name = config.get("plugin_name", "default_plugin")


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
                if form_id in used_biome_ids:
                    biome_colors[form_id] = desaturate_color((r, g, b), saturate_factor)
            except (ValueError, IndexError):
                print(f"Warning: Invalid row in Biomes.csv: {row}. Skipping.")

    return biome_colors


def load_biome_heights(csv_path, used_biome_ids):
    biome_heights = {}
    ocean_formids = set()
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                form_id = int(row[0], 16)
                height_value = int(row[5])
                is_ocean = row[6].lower() == "true" if len(row) > 6 else False
                if form_id in used_biome_ids:
                    biome_heights[form_id] = height_value
                    if is_ocean:
                        ocean_formids.add(form_id)
            except (ValueError, IndexError):
                print(f"Warning: Invalid row in Biomes.csv: {row}. Skipping.")
    return biome_heights, ocean_formids


def load_biom_file(biom_path):
    """Load .biom file from the provided path."""
    if not biom_path.exists():
        raise FileNotFoundError(f"Biom file not found at: {biom_path}")

    with open(biom_path, "rb") as f:
        data = cast(CsSF_BiomContainer, CsSF_Biom.parse_stream(f))

    biome_grid_n = np.array(data.biomeGridN, dtype=np.uint32).reshape(
        GRID_SIZE[1], GRID_SIZE[0]
    )
    biome_grid_s = np.array(data.biomeGridS, dtype=np.uint32).reshape(
        GRID_SIZE[1], GRID_SIZE[0]
    )

    return biome_grid_n, biome_grid_s


def upscale_image(image, target_size=(1024, 1024)):
    """Upscale image to target size if enabled in config."""
    if config.get("upscale_image", False):
        return image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def generate_noise(shape, scale=None):
    """Generate larger-patch high-contrast salt-and-pepper noise."""
    if scale is None:
        scale = config.get("noise_scale", 4.17)

    noise = np.random.rand(*shape)
    pepper_mask = noise < (3 * (scale / 100))
    salt_mask = noise > (1 - (3 * (scale / 100)))

    noise[pepper_mask] = 0
    noise[salt_mask] = 1

    patch_size = int(max(1, scale / 10))
    for y in range(0, shape[0], patch_size):
        for x in range(0, shape[1], patch_size):
            if np.random.rand() < 0.5:
                noise[y : y + patch_size, x : x + patch_size] = 0
            elif np.random.rand() > 0.5:
                noise[y : y + patch_size, x : x + patch_size] = 1

    swap_count = int(scale * 5)
    for _ in range(swap_count):
        x1, y1, x2, y2 = (
            np.random.randint(0, shape[1] - patch_size),
            np.random.randint(0, shape[0] - patch_size),
            np.random.randint(0, shape[1] - patch_size),
            np.random.randint(0, shape[0] - patch_size),
        )
        (
            noise[y1 : y1 + patch_size, x1 : x1 + patch_size],
            noise[y2 : y2 + patch_size, x2 : x2 + patch_size],
        ) = (
            noise[y2 : y2 + patch_size, x2 : x2 + patch_size],
            noise[y1 : y1 + patch_size, x1 : x1 + patch_size],
        )

    return noise


def generate_elevation(grid, biome_heights):
    """Generate elevation from a biome grid using height values."""
    elevation = np.zeros_like(grid, dtype=np.uint8)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            biome_id = grid[y, x]
            elevation[y, x] = biome_heights.get(biome_id, 127)  # Fallback to mid-height

    # Optional smoothing
    sigma_map = (2.0, 2.0)
    smoothed_elevation = gaussian_filter(elevation, sigma=sigma_map)

    return smoothed_elevation.astype(np.uint8)


def generate_atmospheric_fade(shape, intensity=None, spread=None):
    """Generate atmospheric fade effect from planet center."""
    if intensity is None:
        intensity = config.get("fade_intensity", 0.27)
    if spread is None:
        spread = config.get("fade_spread", 0.81)
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y_grid, x_grid = np.indices(shape)
    distance_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    return np.exp(-spread * (distance_from_center / max_distance)) * intensity


def generate_shading(grid, light_source_x=None, light_source_y=None):
    """Generate anisotropic shading based on terrain gradients."""
    if light_source_x is None:
        light_source_x = config.get("light_source_x", 0.5)
    if light_source_y is None:
        light_source_y = config.get("light_source_y", 0.5)

    grad_x = np.gradient(grid, axis=1)
    grad_y = np.gradient(grid, axis=0)
    shading = grad_x * light_source_x + grad_y * light_source_y

    # Normalize to [0, 1]
    shading_min = np.nanmin(shading)
    shading_max = np.nanmax(shading)
    range_val = shading_max - shading_min if shading_max != shading_min else 1.0
    shading = (shading - shading_min) / range_val

    # Sanitize the result
    shading = np.nan_to_num(shading, nan=0.0, posinf=1.0, neginf=0.0)
    shading = np.clip(shading, 0.0, 1.0)

    return shading


def generate_fractal_noise(
    shape, octaves=None, detail_smoothness=None, texture_contrast=None
):
    """Generate fractal noise for terrain complexity."""
    if octaves is None:
        octaves = config.get("texture_fractal", 4.23)
    if detail_smoothness is None:
        detail_smoothness = config.get("detail_smoothness", 0.41)
    if texture_contrast is None:
        texture_contrast = config.get("texture_contrast", 0.67)
    base = np.random.rand(*shape)
    combined = np.zeros_like(base)
    for i in range(int(octaves)):
        sigma = max(1, detail_smoothness ** (i * 0.3))
        weight = texture_contrast ** (i * 1.5)
        combined += gaussian_filter(base, sigma=sigma) * weight

    combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = np.power(combined, 2)
    return (combined - combined.min()) / (combined.max() - combined.min())


def add_craters(
    grid, num_craters=100, max_radius=None, crater_depth_min=None, crater_depth_max=None
):
    """Add small impact craters to terrain grid."""
    if max_radius is None:
        max_radius = config.get("crater_max_radius", 20)
    if crater_depth_min is None:
        crater_depth_min = config.get("crater_depth_min", 0.2)
    if crater_depth_max is None:
        crater_depth_max = config.get("crater_depth_max", 0.8)

    if crater_depth_min >= crater_depth_max:
        crater_depth_max = crater_depth_min + 0.01
        print(
            f"WARNING: Adjusted crater_depth_max to {crater_depth_max} to ensure min < max"
        )

    crater_map = np.zeros_like(grid, dtype=np.float32)

    for _ in range(num_craters):
        cx, cy = np.random.randint(0, GRID_SIZE[0]), np.random.randint(0, GRID_SIZE[1])
        radius = np.random.randint(1, max_radius // 3)
        y_grid, x_grid = np.indices(grid.shape)
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        crater_depth = (
            np.exp(-dist / radius)
            * np.random.uniform(crater_depth_min, crater_depth_max)
            * 1.2
        )
        crater_map -= crater_depth

    return np.clip(grid + crater_map, 0, 1)


def generate_crater_shading(crater_map):
    """Generate shading for crater rims."""
    if config.get("enable_crater_shading", False):
        shading = np.gradient(crater_map, axis=0) + np.gradient(crater_map, axis=1)
        shading *= 2
        min_shading, max_shading = shading.min(), shading.max()
        if max_shading == min_shading:
            return np.zeros_like(crater_map)
        normalized_shading = (shading - min_shading) / (max_shading - min_shading)
        normalized_shading = np.power(normalized_shading, 1.5)
        normalized_shading[np.isnan(normalized_shading)] = 0
        return normalized_shading
    return np.zeros_like(crater_map)


def generate_edge_blend(grid, blend_radius=None):
    """Generate edge blending map for biome transitions."""
    if not config.get("enable_texture_edges", False):
        return np.zeros_like(grid, dtype=np.float32)

    if blend_radius is None:
        blend_radius = config.get("texture_edges", 0.53)

    edge_map = np.zeros_like(grid, dtype=np.float32)

    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            current_biome = grid[y, x]
            neighbors = [
                grid[max(y - 1, 0), x],
                grid[min(y + 1, GRID_SIZE[1] - 1), x],
                grid[y, max(x - 1, 0)],
                grid[y, min(x + 1, GRID_SIZE[0] - 1)],
            ]
            if any(neighbor != current_biome for neighbor in neighbors):
                edge_map[y, x] = 1.0

    blurred_map = gaussian_filter(edge_map, sigma=max(0.1, 3 - blend_radius))
    return blurred_map


def desaturate_color(rgb, saturate_factor=None):
    """Adjust color saturation in HSV space."""
    if saturate_factor is None:
        saturate_factor = config.get("texture_saturation", 0.29)
    h, s, v = colorsys.rgb_to_hsv(*[c / 255.0 for c in rgb])
    s *= saturate_factor
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in (r, g, b))


def enhance_brightness(image, bright_factor=None):
    """Enhance image brightness."""
    if bright_factor is None:
        bright_factor = config.get("texture_brightness", 0.74)
    scaled_factor = bright_factor * 4
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(scaled_factor)


def generate_heightmap(grid, biome_heights):  # Change parameter to grid
    elevation = np.zeros((GRID_SIZE[1], GRID_SIZE[0]), dtype=np.uint8)
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            form_id = int(grid[y, x])
            elevation[y, x] = biome_heights.get(form_id, 127)  # Fallback to mid-height
    smoothed_elevation = gaussian_filter(elevation, sigma=2.0)
    return Image.fromarray(smoothed_elevation, mode="L")


def generate_roughness_map(
    elevation_map, fractal_map, base_value=None, noise_scale=None
):
    """Generate roughness map based on elevation and fractal noise."""
    if base_value is None:
        base_value = 1.0 - config.get("texture_roughness_base", 0.36)
    if noise_scale is None:
        noise_scale = config.get("texture_noise", 0.95)
    roughness = base_value * np.ones_like(elevation_map)
    roughness += noise_scale * fractal_map
    roughness = np.clip(roughness, 0, 1)
    roughness_map = (roughness * 255).astype(np.uint8)
    return Image.fromarray(roughness_map, mode="L")


def generate_ocean_mask(grid: np.ndarray, biome_heights: Dict[int, int]) -> Image.Image:
    """Generate ocean mask where height == 0 is black (ocean), else white (land)."""
    h, w = grid.shape
    ocean_mask = np.full((h, w), 255, dtype=np.uint8)  # Default to land (white)

    for y in range(h):
        for x in range(w):
            form_id = int(grid[y, x])
            height = biome_heights.get(form_id, 255)
            if height == 0:
                ocean_mask[y, x] = 0  # Ocean

    return Image.fromarray(ocean_mask, mode="L")


def create_biome_image(grid, biome_colors, default_color=(128, 128, 128)):
    process_images = config.get("process_images", False)
    bright_factor = config.get("texture_brightness", 0.05)
    if not biome_colors:
        print("Error: biome_colors is empty, using default color")
        color = np.full((GRID_SIZE[1], GRID_SIZE[0], 3), default_color, dtype=np.uint8)
        return {
            "color": Image.fromarray(color),
            "surface": Image.fromarray(
                np.zeros((GRID_SIZE[1], GRID_SIZE[0]), dtype=np.uint8), mode="L"
            ),
            "ocean": Image.fromarray(
                np.zeros((GRID_SIZE[1], GRID_SIZE[0]), dtype=np.uint8), mode="L"
            ),
        }

    used_biome_ids = set(grid.flatten())
    biome_heights, ocean_formids = load_biome_heights(str(CSV_PATH), used_biome_ids)

    color = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
    if process_images:
        noise_map = generate_noise(
            (GRID_SIZE[1], GRID_SIZE[0]), scale=config["noise_scale"]
        )
        elevation_map = generate_elevation(grid, biome_heights)
        edge_blend_map = generate_edge_blend(grid)
        shading_map = generate_shading(elevation_map)
        fractal_map = generate_fractal_noise((GRID_SIZE[1], GRID_SIZE[0]))
        crater_map = add_craters(elevation_map)
        crater_shading = generate_crater_shading(crater_map)
        bright_factor = config.get("texture_brightness", 0.74)

        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                form_id = int(grid[y, x])
                biome_color = biome_colors.get(form_id, default_color)
                lat_factor = abs((y / GRID_SIZE[1]) - 0.5) * 0.4
                elevation_factor = elevation_map[y, x] / 255.0  # Normalize to [0, 1]
                shaded_color = tuple(
                    int(c * (0.8 + 0.2 * elevation_factor)) for c in biome_color
                )
                light_adjusted_color = tuple(
                    int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color
                )
                fractal_adjusted_color = tuple(
                    int(c * (0.85 + 0.15 * fractal_map[y, x]))
                    for c in light_adjusted_color
                )
                crater_adjusted_color = tuple(
                    int(c * (0.7 + 0.3 * crater_shading[y, x]))
                    for c in fractal_adjusted_color
                )
                lat_adjusted_color = tuple(
                    int(c * (1 - lat_factor)) for c in crater_adjusted_color
                )
                blended_color = tuple(
                    int(c * (1 - 0.5 * edge_blend_map[y, x]))
                    for c in lat_adjusted_color
                )
                final_color = tuple(
                    np.clip(int(c * (0.91 + 0.09 * noise_map[y, x])), 0, 255)
                    for c in blended_color
                )
                color[y, x] = final_color
    else:
        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                form_id = int(grid[y, x])
                color[y, x] = biome_colors.get(form_id, default_color)

    color_image = Image.fromarray(color)
    if process_images:
        color_image = enhance_brightness(color_image, bright_factor)

    surface_image = generate_heightmap(grid, biome_heights)
    ocean_image = generate_ocean_mask(grid, biome_heights)

    return {
        "color": color_image,
        "surface": surface_image,
        "ocean": ocean_image,
    }


def convert_png_to_dds(
    png_path, texture_output_dir, plugin_name, texture_type, dds_name=None
):
    """Convert a PNG file to DDS using texconv.exe with Starfield-compatible formats."""
    if not TEXCONV_PATH.exists():
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}")

    # Ensure output directory exists
    texture_output_dir = texture_output_dir
    texture_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine DDS format based on texture type
    format_map = {
        "color": config.get("color_format", "BC7_UNORM"),
        "surface": config.get("surface_format", "BC7_UNORM"),
        "ocean": config.get("ocean_format", "BC4_UNORM"),
    }
    dds_format = format_map.get(texture_type, "BC7_UNORM")

    # Construct output DDS path
    texture_filename = dds_name if dds_name else png_path.stem + ".dds"
    texture_path = texture_output_dir / texture_filename

    # Build texconv command
    cmd = [
        str(TEXCONV_PATH),
        "-f",
        dds_format,
        "-m",
        "0",  # Generate all mipmaps
        "-y",  # Overwrite output
        "-o",
        str(texture_output_dir),
        str(png_path),
    ]

    # Execute texconv
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Converted {png_path.name} to {texture_path.name} ({dds_format})", file=sys.stderr)
        return texture_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {png_path.name} to DDS: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: texconv.exe not found at {TEXCONV_PATH}")
        raise


def main():
    global plugin_name
    print(f"=== Landscaping permit approved for: {plugin_name} ===", flush=True)
    print("=== Starting PlanetTextures ===", flush=True)

    parser = argparse.ArgumentParser(
        description="Generate PNG and DDS textures from .biom files"
    )
    parser.add_argument(
        "biom_file", nargs="?", help="Path to the .biom file (for preview mode)"
    )
    parser.add_argument("--preview", action="store_true", help="Run in preview mode")
    args = parser.parse_args()

    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.preview and args.biom_file:
        biom_files = [Path(args.biom_file)]
        print(f"Preview mode: Processing {biom_files[0]}")
        if not biom_files[0].exists():
            print(f"Error: Provided .biom file not found: {args.biom_file}")
            sys.exit(1)
    else:
        biom_files = [
            f
            for f in (PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name).rglob(
                "*.biom"
            )
        ]
        if not biom_files:
            print("No .biom files found in the output directory.")
            sys.exit(1)

    used_biome_ids = set()
    for biom_path in biom_files:
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)
            used_biome_ids.update(biome_grid_n.flatten())
            used_biome_ids.update(biome_grid_s.flatten())
        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")

    biome_colors = load_biome_colors(CSV_PATH, used_biome_ids)
    if not biome_colors:
        raise ValueError("No valid biome colors loaded from Biomes.csv")

    keep_pngs = config.get("keep_pngs_after_conversion", True)

    for biom_path in biom_files:
        planet_name = biom_path.stem
        print(f"Distributing labor force {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)
            maps_n = create_biome_image(biome_grid_n, biome_colors)
            maps_s = create_biome_image(biome_grid_s, biome_colors)

            maps_n = {k: upscale_image(v) for k, v in maps_n.items()}
            maps_s = {k: upscale_image(v) for k, v in maps_s.items()}

            once_per_run = False
            copied_textures = set()
            for hemisphere, maps in [("North", maps_n), ("South", maps_s)]:
                preview_dir = TEMP_DIR
                preview_dir.mkdir(parents=True, exist_ok=True)

                for texture_type in ["color", "surface", "ocean"]:
                    temp_filename = f"temp_{texture_type}.png"
                    png_filename = f"{planet_name}_{hemisphere}_{texture_type}.png"
                    planet_png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
                    planet_png_dir.mkdir(parents=True, exist_ok=True)

                    png_path = planet_png_dir / png_filename
                    maps[texture_type].save(png_path)

                    if texture_type not in copied_textures:
                        shutil.copy(png_path, preview_dir / temp_filename)
                        copied_textures.add(texture_type)

                    if not once_per_run:
                        print("Review documentation submitted.")
                        once_per_run = True

                    print(f"Saved PNG: {png_path}", file=sys.stderr)

                    if not args.preview:
                        texture_output_dir = (
                            PLUGINS_DIR / plugin_name / "textures" / plugin_name
                        )
                        texture_output_dir.mkdir(parents=True, exist_ok=True)
                        #texture_path = convert_png_to_dds(
                        #    png_path, texture_output_dir, plugin_name, texture_type
                        #)
                        #print(
                        #    f"Converted DDS saved to: {texture_path}", file=sys.stderr
                        #)

                        if not keep_pngs:
                            try:
                                png_path.unlink()
                                print(
                                    f"Deleted intermediate PNG: {png_path}",
                                    file=sys.stderr,
                                )
                            except OSError as e:
                                print(
                                    f"Error deleting {png_path}: {e}", file=sys.stderr
                                )

            print(
                f"Generated textures for {planet_name} (North and South: color, surface, ocean)",
                file=sys.stderr,
            )
            print(f"Visual inspection of {planet_name} complete.")
        except Exception as e:
            import traceback

            print(f"Error processing {biom_path.name}: {e}")
            traceback.print_exc()

        for texture_type in ["color", "surface", "ocean"]:
            planet_png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
            north_path = planet_png_dir / f"{planet_name}_North_{texture_type}.png"
            south_path = planet_png_dir / f"{planet_name}_South_{texture_type}.png"
            combined_path = (
                planet_png_dir / f"{planet_name}_{texture_type}_combined.png"
            )

            if north_path.exists() and south_path.exists():
                north_img = Image.open(north_path)
                south_img = Image.open(south_path)
                combined_img = Image.new(
                    "RGB", (north_img.width, north_img.height + south_img.height)
                )
                combined_img.paste(north_img, (0, 0))
                combined_img.paste(south_img, (0, north_img.height))
                combined_img.save(combined_path)
                print(f"Combined image saved: {combined_path}", file=sys.stderr)

                # Optional DDS conversion
                if not args.preview:
                    texture_output_dir = (
                        PLUGINS_DIR / plugin_name / "textures" / plugin_name
                    )
                    texture_output_dir.mkdir(parents=True, exist_ok=True)

                    dds_name_map = {
                        "color": f"{planet_name}_color.dds",
                        "surface": f"{planet_name}_surface_a_mask.dds",
                        "ocean": f"{planet_name}_ocean_mask.dds",
                    }
                    dds_filename = dds_name_map[texture_type]

                    dds_path = convert_png_to_dds(
                        combined_path, texture_output_dir, plugin_name, texture_type, dds_filename
                    )
                    print(f"Combined DDS saved: {dds_path}", file=sys.stderr)

                # Optionally clean up individual hemisphere PNGs
                if not keep_pngs:
                    try:
                        north_path.unlink()
                        south_path.unlink()
                        print(f"Deleted {north_path} and {south_path}", file=sys.stderr)
                    except OSError as e:
                        print(f"Error deleting hemisphere PNGs: {e}", file=sys.stderr)

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetMaterials.py")], check=True)
    sys.stdout.flush()
    sys.exit()


if __name__ == "__main__":
    main()
