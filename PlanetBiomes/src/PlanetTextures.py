#!/usr/bin/env python3
"""
Planet Textures Generator

Generates PNG and DDS texture images for planet biomes based on .biom files.
Outputs four maps per hemisphere: color, surface, ocean, ambient, normal, rough.
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
from scipy.ndimage import gaussian_filter, sobel
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
from PlanetNewsfeed import handle_news
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


def load_biome_data(
    csv_path: Path | str, used_biome_ids: set[int] | None = None
) -> dict[int, dict]:
    """Load biome data (colors and heights) from Biomes.csv."""
    biome_data = {}
    handle_news(None, "info", f"Loading biome data from {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_columns = {
            "FormID",
            "Red",
            "Green",
            "Blue",
            "HeightIndex",
            "BiomeCategory",
        }

        if reader.fieldnames is None:
            handle_news(
                None,
                "error",
                "Biomes.csv is empty or malformed: no column headers found",
            )
            raise ValueError(
                "Biomes.csv is empty or malformed: no column headers found"
            )

        if not expected_columns.issubset(reader.fieldnames):
            missing = expected_columns - set(reader.fieldnames)
            handle_news(None, "error", f"Missing columns in Biomes.csv: {missing}")
            raise ValueError(f"Biomes.csv missing required columns: {missing}")

        for i, row in enumerate(reader):
            handle_news(None, "info", f"Processing row {i + 1}: {row}")
            try:
                form_id_str = row.get("FormID")
                if not form_id_str or not isinstance(form_id_str, str):
                    print(
                        f"Row {i + 1} skipped: invalid or missing FormID ({form_id_str})"
                    )
                    continue

                form_id_str = form_id_str.strip().replace("0x", "")
                if not form_id_str or not all(
                    c in "0123456789ABCDEFabcdef" for c in form_id_str
                ):
                    print(f"Row {i + 1} skipped: malformed hex FormID ({form_id_str})")
                    continue

                form_id = int(form_id_str, 16)

                if used_biome_ids is not None and form_id not in used_biome_ids:
                    continue

                red = int(row.get("Red", 128))
                green = int(row.get("Green", 128))
                blue = int(row.get("Blue", 128))
                height = int(row.get("HeightIndex", 127))
                category = row.get("BiomeCategory", "").lower()

                biome_data[form_id] = {
                    "color": (red, green, blue),
                    "height": height,
                    "category": category,
                }

                handle_news(None, "info",
                    f"Row {i + 1} accepted: FormID {form_id} -> color={red, green, blue}, height={height}, category={category}"
                )

            except (ValueError, KeyError, TypeError) as e:
                print(f"Row {i + 1} error: {e}. Full row: {row}")
                handle_news(
                    None,
                    "error",
                    f"Warning: Invalid row in Biomes.csv: {row}. Skipping. Error: {e}",
                )

    if not biome_data:
        handle_news(None, "error", "No valid biome data loaded from Biomes.csv")
    else:
        handle_news(None, "info", f"Finished loading {len(biome_data)} biomes.")

    return biome_data


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


def generate_elevation(grid, biome_data):
    """Generate elevation from a biome grid using height values from biome_data."""
    elevation = np.zeros_like(grid, dtype=np.uint8)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            biome_id = int(grid[y, x])
            height = biome_data.get(biome_id, {}).get("height", 127)
            elevation[y, x] = height

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


def generate_craters(elevation_map, crater_depth_min=0.2, crater_depth_max=0.8):
    if not config.get("texture_craters", False):
        return elevation_map  # craters disabled

    crater_scale = config.get("texture_craters_scale", 1.0)
    max_radius = int(20 * crater_scale)
    num_craters = int(100 * crater_scale)
    print(f"Injecting craters: {num_craters} @ radius ~{max_radius}")

    elevation_map_f = elevation_map.astype(np.float32) / 255.0
    crater_map = np.zeros_like(elevation_map_f)

    for _ in range(num_craters):
        cx = np.random.randint(0, elevation_map.shape[1])
        cy = np.random.randint(0, elevation_map.shape[0])
        radius = np.random.randint(1, max(2, max_radius))
        y_grid, x_grid = np.indices(elevation_map.shape)
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        depth = np.exp(-dist / radius) * np.random.uniform(
            crater_depth_min, crater_depth_max
        )
        crater_map -= depth

    elevation_map_f = np.clip(elevation_map_f + crater_map, 0, 1)
    return (elevation_map_f * 255).astype(np.uint8)


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


def generate_heightmap(grid, biome_data):
    """Generate heightmap from biome grid using height values from biome_data."""
    elevation = np.zeros((GRID_SIZE[1], GRID_SIZE[0]), dtype=np.uint8)
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            form_id = int(grid[y, x])
            elevation[y, x] = biome_data.get(form_id, {}).get(
                "height", 127
            )  # Fallback to mid-height
    return Image.fromarray(elevation, mode="L")


from PIL import Image
import numpy as np


def generate_rough_map(
    height_img,
    biome_grid,
    biome_data,
    ocean_img=None,
    fractal_map=None,
    base_value=None,
    noise_scale=None,
    slope_strength=0.5,
):
    # Convert height image to normalized array
    height = np.asarray(height_img).astype(np.float32) / 255.0
    H, W = height.shape
    roughness = np.zeros((H, W), dtype=np.float32)

    # Config fallbacks
    if base_value is None:
        base_value = 1.0 - config.get("texture_roughness_base", 0.36)  # e.g., 0.64
    if noise_scale is None:
        noise_scale = config.get("texture_noise", 0.95) * 0.2  # Reduce noise impact

    # 1. Slope roughness (from gradient)
    dy, dx = np.gradient(height)
    slope = np.sqrt(dx**2 + dy**2)
    slope_roughness = np.clip(slope * slope_strength, 0, 1) * 0.3  # Weight slope at 30%
    roughness += slope_roughness
    handle_news(None, "info",
        f"Slope roughness range: {slope_roughness.min():.3f} - {slope_roughness.max():.3f}"
    )

    # 2. Biome category influence with height modulation
    biome_rough_lookup = {
        "canyon": 0.35,
        "mountain": 0.45,
        "hills": 0.3,
        "archipelago": 0.4,
        "fields": 0.2,
        "ocean": 0.01,
        "desert": 0.25,
        "flat": 0.15,
    }
    biome_rough_map = np.full_like(roughness, base_value)
    for form_id, info in biome_data.items():
        category = info.get("BiomeCategory", "").lower()
        base_roughness = biome_rough_lookup.get(category, base_value)
        if base_roughness is None:
            base_roughness = base_value or 0.5
        # Modulate by height to emphasize elevation within biomes
        mask = biome_grid == form_id
        biome_rough_map[mask] = base_roughness * (
            1 + height[mask] * 5
        )  # Height boosts roughness
        handle_news(None, "info",
            f"Biome {form_id} ({category}): roughness {base_roughness}, mask size {np.sum(mask)}"
        )

    roughness += biome_rough_map * 0.5  # Weight biome contribution at 50%
    handle_news(None, "info",
        f"Biome roughness range: {biome_rough_map.min():.3f} - {biome_rough_map.max():.3f}"
    )

    # 3. Ocean suppression
    if ocean_img is not None:
        ocean_mask = np.asarray(ocean_img).astype(np.float32) / 255.0  # 1.0 = ocean
        roughness *= np.clip(1.0 - ocean_mask * 0.9, 0, 1)  # Stronger suppression
        handle_news(None, "info", f"Ocean mask range: {ocean_mask.min():.3f} - {ocean_mask.max():.3f}")

    # 4. Fractal noise contribution
    if fractal_map is not None:
        # Normalize fractal_map to [0, 1] if it's not already
        fractal_map = (fractal_map - fractal_map.min()) / (
            fractal_map.max() - fractal_map.min() + 1e-6
        )
        roughness += noise_scale * fractal_map * 0.2  # Weight noise at 20%
        handle_news(None, "info",
            f"Fractal map range (after norm): {fractal_map.min():.3f} - {fractal_map.max():.3f}"
        )

    # Normalize and clamp
    roughness = np.clip(
        roughness / (0.3 + 0.5 + 0.2), 0, 1
    )  # Normalize by total weights
    handle_news(None, "info", f"Final roughness range: {roughness.min():.3f} - {roughness.max():.3f}")

    return Image.fromarray((roughness * 255).astype(np.uint8), mode="L")


def generate_ocean_mask(grid: np.ndarray, biome_data: Dict[int, Dict]) -> Image.Image:
    """Generate ocean mask where height == 0 is black (ocean), else white (land)."""
    h, w = grid.shape
    ocean_mask = np.full((h, w), 255, dtype=np.uint8)  # Default to land (white)

    for y in range(h):
        for x in range(w):
            form_id = int(grid[y, x])
            height = biome_data.get(form_id, {}).get("height", 255)
            if height == 0:
                ocean_mask[y, x] = 0  # Ocean

    return Image.fromarray(ocean_mask, mode="L")


def create_biome_image(grid, biome_data, default_color=(128, 128, 128)):
    """Generate biome texture images from grid and biome data."""
    process_images = config.get("process_images", False)
    bright_factor = config.get("texture_brightness", 0.05)

    # Extract colors from biome_data
    biome_colors = {k: v["color"] for k, v in biome_data.items()}

    if not biome_colors:
        print("Error: biome_colors is empty, using default color")
        color = np.full((GRID_SIZE[1], GRID_SIZE[0], 3), default_color, dtype=np.uint8)
        dummy_grayscale = np.zeros((GRID_SIZE[1], GRID_SIZE[0]), dtype=np.uint8)
        return {
            "color": Image.fromarray(color),
            "surface": Image.fromarray(dummy_grayscale, mode="L"),
            "ocean": Image.fromarray(dummy_grayscale.copy(), mode="L"),
            "normal": Image.fromarray(color, mode="L"),
            "rough": Image.fromarray(dummy_grayscale.copy(), mode="L"),
            "ao": Image.fromarray(dummy_grayscale.copy(), mode="L"),
        }

    fractal_map = None

    color = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
    if process_images:
        noise_map = generate_noise(
            (GRID_SIZE[1], GRID_SIZE[0]), scale=config.get("noise_scale", 4.17)
        )
        elevation_map = generate_elevation(grid, biome_data)
        elevation_map = generate_craters(elevation_map)
        edge_blend_map = generate_edge_blend(grid)
        shading_map = generate_shading(elevation_map)
        fractal_map = generate_fractal_noise((GRID_SIZE[1], GRID_SIZE[0]))
        bright_factor = config.get("texture_brightness", 0.74)

        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                form_id = int(grid[y, x])
                biome_color = biome_data.get(form_id, {}).get("color", default_color)
                biome_color = tuple(int(v) for v in biome_color)
                lat_factor = abs((y / GRID_SIZE[1]) - 0.5) * 0.4
                elevation_factor = elevation_map[y, x] / 255.0
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
                lat_adjusted_color = tuple(
                    int(c * (1 - lat_factor)) for c in fractal_adjusted_color
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
                color[y, x] = biome_data.get(form_id, {}).get("color", default_color)

    color_image = Image.fromarray(color)
    if process_images:
        color_image = enhance_brightness(color_image, bright_factor)

    surface_image = generate_heightmap(grid, biome_data)
    ocean_image = generate_ocean_mask(grid, biome_data)
    normal_image = generate_normal_map(surface_image)
    ao_image = generate_ao_map(grid, biome_data)
    rough_image = generate_rough_map(
        height_img=surface_image,
        biome_grid=grid,
        biome_data=biome_data,
        ocean_img=ocean_image,
        fractal_map=fractal_map,
        base_value=config.get("texture_roughness_base", 0.2),
        noise_scale=config.get("texture_roughness", 0.15),
        slope_strength=0.5,
    )

    return {
        "color": color_image,
        "surface": surface_image,
        "ocean": ocean_image,
        "normal": normal_image,
        "rough": rough_image,
        "ao": ao_image,
    }


import numpy as np
from scipy.ndimage import sobel
from PIL import Image


def generate_normal_map(height_img, strength=0.5):
    # Convert height map to float32 and normalize to [0, 1]
    height = np.asarray(height_img).astype(np.float32) / 255.0

    # Compute gradients using Sobel filter
    dx = sobel(height, axis=1) * strength
    dy = sobel(height, axis=0) * strength
    dz = np.ones_like(height)  # Z-component for flat surfaces

    # Normalize the normal vector
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / (length + 1e-8)
    ny = dy / (length + 1e-8)
    nz = dz / (length + 1e-8)

    # Convert [-1, 1] to [0, 255] for RGB
    r = ((nx + 1) * 0.5 * 255).astype(np.uint8)  # Red = X
    g = ((ny + 1) * 0.5 * 255).astype(np.uint8)  # Green = Y
    b = ((nz + 1) * 0.5 * 127).astype(np.uint8)  # Blue = Z

    # Stack channels into RGB image
    normal_map = np.stack([r, g, b], axis=-1)
    return Image.fromarray(normal_map, mode="RGB")


def generate_ao_map(grid, biome_data):
    elevation = generate_elevation(grid, biome_data)
    blurred = gaussian_filter(elevation.astype(np.float32), sigma=2.0)
    ao = np.clip((blurred - elevation), 0, 255)
    ao = 255 - (ao / ao.max() * 255)  # Normalize and invert
    ao_image = Image.fromarray(ao.astype(np.uint8), mode="L")
    handle_news(None, "info", f"AO map generated: min={ao.min()}, max={ao.max()}, shape={ao.shape}")
    return ao_image


def convert_png_to_dds(
    png_path, DDS_OUTPUT_DIR, plugin_name, texture_type, dds_name=None
):
    """Convert a PNG file to DDS using texconv.exe with Starfield-compatible formats."""
    if not TEXCONV_PATH.exists():
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}")

    # Ensure output directory exists
    DDS_OUTPUT_DIR = DDS_OUTPUT_DIR
    DDS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine DDS format based on texture type
    format_map = {
        "color": config.get("color_format", "BC7_UNORM"),
        "surface": config.get("surface_format", "BC7_UNORM"),
        "ocean": config.get("ocean_format", "BC4_UNORM"),
        "normal": config.get("normal_format", "BC5_SNORM"),
        "rough": config.get("rough_format", "BC4_UNORM"),
        "ao": config.get("ao_format", "BC4_UNORM"),
    }
    dds_format = format_map.get(texture_type, "BC7_UNORM")

    # Construct output DDS path
    texture_filename = dds_name if dds_name else png_path.stem + ".dds"
    texture_path = DDS_OUTPUT_DIR / texture_filename

    # Build texconv command
    cmd = [
        str(TEXCONV_PATH),
        "-f",
        dds_format,
        "-m",
        "0",  # Generate all mipmaps
        "-y",  # Overwrite output
        "-o",
        str(DDS_OUTPUT_DIR),
        str(png_path),
    ]

    # Execute texconv
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        handle_news(
            None,
            "info",
            f"Converted {png_path.name} to {texture_path.name} ({dds_format})"
        )
        return texture_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {png_path.name} to DDS: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: texconv.exe not found at {TEXCONV_PATH}")
        raise


def main():
    global plugin_name
    print(f"Landscaping permit approved for: {plugin_name}", flush=True)
    print("=== Starting PlanetTextures ===", flush=True)
    parser = argparse.ArgumentParser(description="Generate PNG and DDS textures from .biom files")
    parser.add_argument("biom_file", nargs="?", help="Path to the .biom file (for preview mode)")
    parser.add_argument("--preview", action="store_true", help="Run in preview mode")
    args = parser.parse_args()
    handle_news(None, "info", f"Arguments: biom_file={args.biom_file}, preview={args.preview}")

    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handle_news(None, "info", f"PNG output directory: {PNG_OUTPUT_DIR}")

    if args.preview and args.biom_file:
        biom_files = [Path(args.biom_file)]
        handle_news(None, "info", f"Preview mode: Processing {biom_files[0]}")
        if not biom_files[0].exists():
            print(f"Error: Provided .biom file not found: {args.biom_file}")
            sys.exit(1)
    else:
        biom_files = [f for f in (PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name).rglob("*.biom")]
        handle_news(None, "info", f"Found .biom files: {biom_files}")
        if not biom_files:
            print("No .biom files found in the output directory.")
            sys.exit(1)

    used_biome_ids = set()
    for biom_path in biom_files:
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)
            used_biome_ids.update(biome_grid_n.flatten())
            used_biome_ids.update(biome_grid_s.flatten())
            handle_news(None, "info", f"Loaded {biom_path.name}: biome IDs = {used_biome_ids}")
        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")
            continue

    handle_news(None, "info", f"Used biome IDs: {used_biome_ids}")
    biome_data = load_biome_data(CSV_PATH, used_biome_ids)
    handle_news(None, "info", f"Loaded biome data: {biome_data}")
    if not biome_data:
        raise ValueError("No valid biome data loaded from Biomes.csv")

    keep_pngs = config.get("keep_pngs_after_conversion", True)
    for biom_path in biom_files:
        planet_name = biom_path.stem
        handle_news(None, "info", f"Processing blueprint for planet {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)
            handle_news(None, "info", f"Generating textures for {planet_name} North...")
            maps_n = create_biome_image(biome_grid_n, biome_data)
            handle_news(None, "info", f"Generating textures for {planet_name} South...")
            maps_s = create_biome_image(biome_grid_s, biome_data)
            maps_n = {k: upscale_image(v) for k, v in maps_n.items()}
            maps_s = {k: upscale_image(v) for k, v in maps_s.items()}

            once_per_run = False
            copied_textures = set()
            for hemisphere, maps in [("North", maps_n), ("South", maps_s)]:
                preview_dir = TEMP_DIR
                preview_dir.mkdir(parents=True, exist_ok=True)

                for texture_type in [
                    "color",
                    "surface",
                    "ocean",
                    "ao",
                    "normal",
                    "rough",
                ]:
                    temp_filename = f"temp_{texture_type}.png"
                    png_filename = f"{planet_name}_{hemisphere}_{texture_type}.png"
                    planet_png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
                    planet_png_dir.mkdir(parents=True, exist_ok=True)

                    png_path = planet_png_dir / png_filename
                    try:
                        maps[texture_type].save(png_path)
                        handle_news(None, "info", f"Saved PNG: {png_path}")
                    except KeyError:
                        handle_news(
                            None,
                            "error",
                            f"Texture type '{texture_type}' not found in maps",
                        )
                        continue

                    if texture_type not in copied_textures:
                        shutil.copy(png_path, preview_dir / temp_filename)
                        copied_textures.add(texture_type)

                    if not once_per_run:
                        print("Review documentation submitted.")
                        once_per_run = True

            print(
                f"Generated textures for {planet_name} (North and South: color, surface, ocean, normal, ao)",
                file=sys.stderr,
            )

            # Process combined textures for this planet
            for texture_type in [
                "color",
                "surface",
                "ocean",
                "normal",
                "rough",
                "ao",
            ]:
                planet_png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
                north_path = planet_png_dir / f"{planet_name}_North_{texture_type}.png"
                south_path = planet_png_dir / f"{planet_name}_South_{texture_type}.png"
                combined_path = planet_png_dir / f"{planet_name}_{texture_type}.png"

                if north_path.exists() and south_path.exists():
                    try:
                        north_img = Image.open(north_path)
                        south_img = Image.open(south_path)
                        combined_img = Image.new(
                            "RGB" if texture_type != "ao" else "L",
                            (north_img.width, north_img.height + south_img.height),
                        )
                        combined_img.paste(north_img, (0, 0))
                        combined_img.paste(south_img, (0, north_img.height))
                        combined_img.save(combined_path)
                        handle_news(
                            None, "info", f"Combined image saved: {combined_path}"
                        )

                        DDS_OUTPUT_DIR = (
                            PLUGINS_DIR
                            / plugin_name
                            / "textures"
                            / plugin_name
                            / planet_name
                        )
                        DDS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                        dds_name_map = {
                            "color": f"{planet_name}_color.dds",
                            "surface": f"{planet_name}_surface_a_mask.dds",
                            "ocean": f"{planet_name}_ocean_mask.dds",
                            "normal": f"{planet_name}_normal.dds",
                            "rough": f"{planet_name}_rough.dds",
                            "ao": f"{planet_name}_ao.dds",
                        }
                        dds_filename = dds_name_map[texture_type]

                        try:
                            dds_path = convert_png_to_dds(
                                combined_path,
                                DDS_OUTPUT_DIR,
                                plugin_name,
                                texture_type,
                                dds_filename,
                            )
                            handle_news(None, "info", f"Combined DDS saved: {dds_path}")
                        except Exception as e:
                            handle_news(
                                None,
                                "error",
                                f"Failed to convert {combined_path} to DDS: {e}",
                            )

                        if not keep_pngs:
                            try:
                                north_path.unlink()
                                south_path.unlink()
                                combined_path.unlink()
                                print(
                                    f"Deleted {north_path}, {south_path}, and {combined_path}",
                                    file=sys.stderr,
                                )
                            except OSError as e:
                                print(
                                    f"Error deleting PNGs for {planet_name}: {e}",
                                    file=sys.stderr,
                                )
                    except Exception as e:
                        handle_news(
                            None,
                            "error",
                            f"Error combining textures for {planet_name} ({texture_type}): {e}",
                        )

            print(f"Visual inspection of {planet_name} complete.")

        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetMaterials.py")], check=True)
    sys.stdout.flush()
    sys.exit()


if __name__ == "__main__":
    main()
