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
import math
from PIL import Image, ImageEnhance
from PlanetNewsfeed import handle_news
from PlanetConstants import (
    TEXCONV_PATH,
    BASE_DIR,
    CONFIG_DIR,
    INPUT_DIR,
    BIOM_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
    ASSETS_DIR,
    SCRIPT_DIR,
    PLUGINS_DIR,
    CSV_DIR,
    IMAGE_DIR,
    PNG_OUTPUT_DIR,
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    CSV_PATH,
    PREVIEW_PATH,
    SCRIPT_PATH,
    TEMPLATE_PATH,
    PREVIEW_BIOME_PATH,
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    IMAGE_FILES,
    BOOLEAN_KEYS,
    PROCESSING_MAP,
)


# Global configuration
def load_config():
    """Load plugin_name from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


config = load_config()
plugin_name = config.get("plugin_name", "default_plugin")

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


def load_biome_data(
    csv_path: Path | str, used_biome_ids: set[int] | None = None
) -> dict[int, dict]:
    """Load biome data (colors and heights) from Biomes.csv."""
    biome_data = {}
    csv_path = Path(csv_path)
    
    # Verify file existence
    if not csv_path.exists():
        handle_news(None, "error", f"Biomes.csv not found at: {csv_path}")
        raise FileNotFoundError(f"Biomes.csv not found at: {csv_path}")
    
    if not csv_path.is_file():
        handle_news(None, "error", f"Path is not a file: {csv_path}")
        raise ValueError(f"Path is not a file: {csv_path}")
    
    handle_news(None, "info", f"Loading biome data from {csv_path}")
    
    try:
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

            # Log the headers for debugging
            handle_news(None, "info", f"CSV headers: {reader.fieldnames}")

            row_count = 0
            valid_row_count = 0
            for i, row in enumerate(reader):
                row_count += 1
                try:
                    form_id_str = row.get("FormID")
                    if not form_id_str or not isinstance(form_id_str, str):
                        handle_news(
                            None,
                            "warning",
                            f"Row {i + 1} skipped: invalid or missing FormID ({form_id_str})",
                        )
                        continue

                    form_id_str = form_id_str.strip().replace("0x", "")
                    if not form_id_str or not all(
                        c in "0123456789ABCDEFabcdef" for c in form_id_str
                    ):
                        handle_news(
                            None,
                            "warning",
                            f"Row {i + 1} skipped: malformed hex FormID ({form_id_str})",
                        )
                        continue

                    form_id = int(form_id_str, 16)

                    # Skip if used_biome_ids is provided and FormID not in it
                    if used_biome_ids is not None and form_id not in used_biome_ids:
                        continue

                    # Validate color values
                    try:
                        red = int(row.get("Red", 128))
                        green = int(row.get("Green", 128))
                        blue = int(row.get("Blue", 128))
                        if not (0 <= red <= 255 and 0 <= green <= 255 and 0 <= blue <= 255):
                            handle_news(
                                None,
                                "warning",
                                f"Row {i + 1} skipped: RGB values out of range ({red}, {green}, {blue})",
                            )
                            continue
                    except ValueError as e:
                        handle_news(
                            None,
                            "warning",
                            f"Row {i + 1} skipped: Invalid RGB values ({row.get('Red')}, {row.get('Green')}, {row.get('Blue')}): {e}",
                        )
                        continue

                    # Validate height
                    try:
                        height = int(row.get("HeightIndex", 127))
                        if not (0 <= height <= 255):
                            handle_news(
                                None,
                                "warning",
                                f"Row {i + 1} skipped: HeightIndex out of range ({height})",
                            )
                            continue
                    except ValueError as e:
                        handle_news(
                            None,
                            "warning",
                            f"Row {i + 1} skipped: Invalid HeightIndex ({row.get('HeightIndex')}): {e}",
                        )
                        continue

                    category = row.get("BiomeCategory", "").lower()

                    biome_data[form_id] = {
                        "color": (red, green, blue),
                        "height": height,
                        "category": category,
                    }
                    valid_row_count += 1

                except Exception as e:
                    handle_news(
                        None,
                        "warning",
                        f"Row {i + 1} error: {e}. Full row: {row}",
                    )

            handle_news(None, "info", f"Processed {row_count} rows, {valid_row_count} valid biomes loaded")

    except UnicodeDecodeError as e:
        handle_news(None, "error", f"Encoding error reading Biomes.csv: {e}")
        raise

    if not biome_data:
        handle_news(None, "error", f"No valid biome data loaded from Biomes.csv. Rows processed: {row_count}")
        raise ValueError(f"No valid biome data loaded from Biomes.csv. Rows processed: {row_count}")

    handle_news(None, "info", f"Finished loading {len(biome_data)} biomes.")
    return biome_data


def load_biom_file(biom_path, used_biome_ids, biome_data):
    """Load .biom file and convert to RGB grids using biome_data."""
    if not biom_path.exists():
        raise FileNotFoundError(f"Biom file not found at: {biom_path}")

    with open(biom_path, "rb") as f:
        data = cast(CsSF_BiomContainer, CsSF_Biom.parse_stream(f))

    biome_grid_n = np.array(data.biomeGridN, dtype=np.uint32).reshape(256, 256)
    biome_grid_s = np.array(data.biomeGridS, dtype=np.uint32).reshape(256, 256)

    used_biome_ids.update(biome_grid_n.flatten())
    used_biome_ids.update(biome_grid_s.flatten())

    upscale_factor = config["texture_resolution"] // 256
    biome_grid_n = upscale_grid(biome_grid_n, upscale_factor, biome_data=biome_data)
    biome_grid_s = upscale_grid(biome_grid_s, upscale_factor, biome_data=biome_data)

    global GRID_SIZE, GRID_FLATSIZE
    GRID_SIZE = biome_grid_n.shape[:2]
    GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

    return biome_grid_n, biome_grid_s


def upscale_grid(
    grid: np.ndarray, factor: int, biome_data: Dict[int, Dict]
) -> np.ndarray:
    """Upscale biome grid and convert to RGB using biome_data with smoothing."""
    handle_news(None)
    factor = 2 ** int(math.ceil(math.log2(factor)))
    h, w = grid.shape
    new_h, new_w = h * factor, w * factor

    # Convert grid to RGB
    color_grid = np.zeros((h, w, 3), dtype=np.uint8)
    default_color = (128, 128, 128)
    for form_id in np.unique(grid):
        color = biome_data.get(form_id, {}).get("color", default_color)
        color_grid[grid == form_id] = color

    # Upscale with bilinear interpolation
    image = Image.fromarray(color_grid, mode="RGB")
    upscaled = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Apply Gaussian blur to smooth transitions
    upscaled_array = np.array(upscaled, dtype=np.float32)
    #sigma = max(1.0, factor / 2.0)  # Scale blur with upscale factor
    #for channel in range(3):
    #    upscaled_array[:, :, channel] = gaussian_filter(
    #       upscaled_array[:, :, channel], sigma=sigma
    #    )

    return np.clip(upscaled_array, 0, 255).astype(np.uint8)


def generate_noise(shape, scale=None):
    """Generate larger-patch high-contrast salt-and-pepper noise."""
    handle_news(None)
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


def generate_elevation(rgb_grid: np.ndarray, biome_data: Dict[int, Dict]) -> np.ndarray:
    """Generate elevation from an RGB biome grid using heights from biome_data."""
    handle_news(None)
    height, width, _ = rgb_grid.shape
    elevation = np.zeros((height, width), dtype=np.uint8)

    rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}
    for y in range(height):
        for x in range(width):
            rgb = tuple(rgb_grid[y, x])
            form_id = rgb_to_form_id.get(rgb, None)
            height_value = biome_data.get(form_id, {}).get("height", 127) if form_id is not None else 127
            elevation[y, x] = height_value

    sigma_map = (2.0, 2.0)
    smoothed_elevation = gaussian_filter(elevation, sigma=sigma_map)

    return smoothed_elevation.astype(np.uint8)


def generate_atmospheric_fade(shape, intensity=None, spread=None):
    """Generate atmospheric fade effect from planet center."""
    handle_news(None)
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
    handle_news(None)
    if light_source_x is None:
        light_source_x = 0.5
    if light_source_y is None:
        light_source_y = 0.5

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


def safe_normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(arr)

    
    return (arr - min_val) / range_val


# Used by generate_fractal_noise
def generate_perlin_noise(elevation_norm, scale=10):
    """Generate Perlin-like noise seeded by elevation."""
    handle_news(None)
    # Optional slight perturbation to avoid flat noise
    noise_seed = elevation_norm + np.random.normal(0, 0.03, elevation_norm.shape)
    smooth_noise = gaussian_filter(noise_seed, sigma=scale)

    return (smooth_noise - smooth_noise.min()) / (
        smooth_noise.max() - smooth_noise.min() + 1e-6
    )


def generate_fractal_noise(
    elevation_norm, octaves=None, detail_smoothness=None, texture_contrast=None
):
    """Generate structured fractal noise for terrain elevation growth."""
    handle_news(None)
    if octaves is None:
        octaves = config.get("texture_fractal", 4.23)
    if detail_smoothness is None:
        detail_smoothness = config.get("detail_smoothness", 0.1)
    if config.get("enable_basic_filters", True):
        if texture_contrast is None:
            texture_contrast = config.get("texture_contrast", 0.67)
    else:
        texture_contrast = 0.5

    # Generate structured Perlin-style base noise
    base = generate_perlin_noise(elevation_norm)  # ✅ Structured ridge-based seed

    combined = np.zeros_like(base)
    for i in range(int(octaves)):
        sigma = max(1, detail_smoothness ** (i * 1.2))  # Adjust smoothness scaling
        weight = max(
            0.1, texture_contrast ** (i * 2.5)
        )  # More controlled contrast influence

        # **Directional Diffusion Propagation**
        gradient_x = np.gradient(base, axis=1)
        gradient_y = np.gradient(base, axis=0)
        combined += (
            gaussian_filter(base, sigma=sigma) + gradient_x * 0.3 + gradient_y * 0.3
        ) * weight  # ✅ Mountain ridges grow naturally

    # **Apply Erosion Effects for Valleys**
    grad_x, grad_y = np.gradient(combined)
    erosion_map = np.abs(grad_x) + np.abs(grad_y)
    combined -= erosion_map * 0.5  # ✅ Valleys deepen while mountains stabilize

    # **Normalize and Strengthen Terrain Growth**
    combined = safe_normalize(combined)
    combined = np.power(combined, 2.3)  # ✅ Enhances ridge prominence

    return safe_normalize(combined)


def generate_craters(elevation_map, crater_depth_min=0.2, crater_depth_max=0.8):
    handle_news(None)

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


def generate_edge_blend(rgb_grid: np.ndarray, biome_data: Dict[int, Dict], blend_radius=None) -> np.ndarray:
    """Generate edge blending map for biome transitions based on RGB grid."""
    handle_news(None)
    if not config.get("enable_texture_edges", False):
        return np.zeros(rgb_grid.shape[:2], dtype=np.float32)

    if blend_radius is None:
        blend_radius = config.get("texture_edges", 0.53)

    edge_map = np.zeros(rgb_grid.shape[:2], dtype=np.float32)
    rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}

    for y in range(rgb_grid.shape[0]):
        for x in range(rgb_grid.shape[1]):
            current_rgb = tuple(rgb_grid[y, x])
            current_form_id = rgb_to_form_id.get(current_rgb, None)
            if current_form_id is None:
                continue

            neighbors = [
                tuple(rgb_grid[max(y - 1, 0), x]),
                tuple(rgb_grid[min(y + 1, rgb_grid.shape[0] - 1), x]),
                tuple(rgb_grid[y, max(x - 1, 0)]),
                tuple(rgb_grid[y, min(x + 1, rgb_grid.shape[1] - 1)]),
            ]
            neighbor_form_ids = [rgb_to_form_id.get(rgb, None) for rgb in neighbors]
            if any(nid != current_form_id and nid is not None for nid in neighbor_form_ids):
                edge_map[y, x] = 1.0

    blurred_map = gaussian_filter(edge_map, sigma=max(0.1, 1 - blend_radius))
    return blurred_map


def enhance_brightness(image, bright_factor=None):
    """Enhance image brightness."""
    handle_news(None)
    if bright_factor is None:
        bright_factor = config.get("texture_brightness", 0.74)
    scaled_factor = bright_factor * 4
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(scaled_factor)


def adjust_tint(
    faded_color,
    texture_saturation,
    texture_tint,
    biome_category="",
    elevation_factor=0.5,
):
    """Adjust color with dynamic tinting based on biome category and elevation."""

    r, g, b = [c / 255.0 for c in faded_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    biome_tint_shifts = {
        "ocean": -0.1,
        "forest": -0.05,
        "desert": 0.05,
        "lava": 0.15,
        "tundra": 0.0,
        "soil": 0.02,
    }
    hue_shift = biome_tint_shifts.get(biome_category, 0.0)
    hue_shift += (texture_tint - 0.5) * 0.25
    hue_shift += elevation_factor * 0.05
    h = (h + hue_shift) % 1.0

    s = s * (0.8 + 0.4 * elevation_factor)
    s = min(s, 1.0)

    v = v * (0.9 + 0.2 * texture_saturation)
    v = min(v, 1.0)

    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    return tuple(int(np.clip(c * 255, 0, 255)) for c in (r, g, b))


def desaturate_color(rgb, texture_saturation):
    """Adjust color saturation in HSV space with support for boosting."""

    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if texture_saturation < 1.0:
        s = s * texture_saturation
    else:
        s = s + (1.0 - s) * (texture_saturation - 1.0)
        s = min(s, 1.0)

    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    return tuple(int(np.clip(c * 255, 0, 255)) for c in (r, g, b))


def generate_heightmap(rgb_grid: np.ndarray, biome_data: Dict[int, Dict], min_out=80, max_out=175) -> Image.Image:
    """Generate heightmap from RGB grid using biome_data heights."""
    handle_news(None)
    height, width, _ = rgb_grid.shape
    elevation = np.zeros((height, width), dtype=np.uint8)
    rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}

    for y in range(height):
        for x in range(width):
            rgb = tuple(rgb_grid[y, x])
            form_id = rgb_to_form_id.get(rgb, None)
            raw_height = biome_data.get(form_id, {}).get("height", 127) if form_id is not None else 127
            norm_height = np.clip(raw_height, 0, 255)
            scaled = min_out + (norm_height / 255) * (max_out - min_out)
            elevation[y, x] = int(scaled)

    return Image.fromarray(elevation, mode="L")


def generate_rough_map(
    height_img,
    rgb_grid,
    biome_data,
    ocean_img=None,
    fractal_map=None,
    base_value=None,
    noise_scale=None,
    slope_strength=0.5,
):
    handle_news(None)
    height = np.asarray(height_img).astype(np.float32) / 255.0
    H, W = height.shape
    if (H, W) != rgb_grid.shape[:2]:
        height_img = height_img.resize(
            rgb_grid.shape[:2][::-1], resample=Image.Resampling.NEAREST
        )
        height = np.asarray(height_img).astype(np.float32) / 255.0
        H, W = height.shape
    roughness = np.zeros((H, W), dtype=np.float32)

    if base_value is None:
        base_value = 1.0 - config.get("texture_roughness_base", 0.36)
    if noise_scale is None:
        noise_scale = config.get("texture_roughness", 0.15)

    # --- Slope influence ---
    dy, dx = np.gradient(height)
    slope = np.sqrt(dx**2 + dy**2)
    slope_roughness = np.clip(slope * slope_strength, 0, 1)
    roughness += slope_roughness
    handle_news(
        None,
        "info",
        f"Slope roughness range: {slope_roughness.min():.3f} - {slope_roughness.max():.3f}",
    )

    # --- Biome-based influence ---
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
            base_roughness = (base_value) or 0.5
        mask = np.all(rgb_grid == np.array(info["color"]), axis=2)
        biome_rough_map[mask] = base_roughness * (1 + height[mask] * 0.5)

    roughness += biome_rough_map

    # --- Fractal influence ---
    if fractal_map is not None:
        fractal_map = (fractal_map - fractal_map.min()) / (
            fractal_map.max() - fractal_map.min() + 1e-6
        )
        roughness += noise_scale * fractal_map

    # --- Ocean darkening ---
    if ocean_img is not None:
        ocean_mask = np.asarray(ocean_img).astype(np.float32) / 255
        ocean_mask = ocean_mask * 0.7 + 0.3 
        roughness -= 0.3 * ocean_mask  # Darken ocean areas

    # --- Normalize and tone down brightness ---
    roughness = np.clip(roughness / 0.5, 0, 1)

    # Apply gamma compression to tone down bright areas
    gamma = .75  # < 1.0 darkens the map
    adjusted = np.power(roughness, gamma)

    # Further compress extremely light values
    adjusted = np.where(adjusted > 0.6, 0.6 + (adjusted - 0.6) * 0.4, adjusted)

    return Image.fromarray(
        ((1.0 - np.clip(adjusted, 0, 1)) * 255).astype(np.uint8), mode="L"
    )


def generate_ocean_mask(
    rgb_grid: np.ndarray, biome_data: Dict[int, Dict]
) -> Image.Image:
    """Generate ocean mask where height <= 4 is black (ocean), else remains white (land)."""
    handle_news(None)
    h, w, _ = rgb_grid.shape
    ocean_mask = np.full((h, w), 255, dtype=np.uint8)  # Start with all white (land)

    rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}

    for y in range(h):
        for x in range(w):
            rgb = tuple(rgb_grid[y, x])
            form_id = rgb_to_form_id.get(rgb, None)
            height = (
                biome_data.get(form_id, {}).get("height", 255)
                if form_id is not None
                else 255
            )

            if height <= 4:
                ocean_mask[y, x] = 0

    return Image.fromarray(ocean_mask, mode="L")


def create_biome_image(
    rgb_grid: np.ndarray, biome_data: Dict[int, Dict]
) -> Dict[str, Image.Image]:
    """Generate biome texture images from RGB grid and biome data."""
    enable_basic_filters = config.get("enable_basic_filters", True)
    enable_texture_noise = config.get("enable_texture_noise", True)
    enable_texture_edges = config.get("enable_texture_edges", True)
    enable_texture_light = config.get("enable_texture_light", True)
    enable_texture_craters = config.get("enable_texture_craters", True)
    process_images = config.get("process_images", False)
    bright_factor = config.get("texture_brightness", 0.05)
    texture_saturation = config.get("texture_saturation", 0.5)
    texture_tint = config.get("texture_tint", 0.5)
    fractal_map = None

    height, width, _ = rgb_grid.shape
    color = rgb_grid.copy()

    handle_news(None)

    if process_images:
        noise_map = generate_noise(
            (height, width), scale=config.get("noise_scale", 4.17)
        )
        elevation_map = generate_elevation(rgb_grid, biome_data)
        if enable_texture_craters:
            elevation_map = generate_craters(elevation_map)
        edge_blend_map = generate_edge_blend(rgb_grid, biome_data)
        shading_map = generate_shading(elevation_map)
        elevation_norm = elevation_map / 255.0
        fractal_map = generate_fractal_noise(elevation_norm)
        atmospheric_fade_map = generate_atmospheric_fade((height, width))
        if enable_basic_filters:
            bright_factor = config.get("texture_brightness", 0.74)

        rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}
        for y in range(height):
            for x in range(width):
                rgb = tuple(rgb_grid[y, x])
                form_id = rgb_to_form_id.get(rgb, None)
                category = biome_data.get(form_id, {}).get("category", "") if form_id is not None else ""
                lat_factor = abs((y / height) - 0.5) * 0.4
                elevation_factor = elevation_map[y, x] / 255.0
                # Apply elevation-based shading
                # under noise
                shaded_color = tuple(
                    int(c * (0.8 + 0.2 * elevation_factor)) for c in rgb
                )
                # Apply light-based shading
                if enable_texture_light:
                    light_adjusted_color = tuple(
                        int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color
                    )
                else:
                    light_adjusted_color = shaded_color

                # Apply fractal noise # under noise
                fractal_adjusted_color = tuple(
                    int(c * (0.85 + 0.15 * fractal_map[y, x]))
                    for c in light_adjusted_color
                )
                # Apply latitude-based darkening # Under bacis
                lat_adjusted_color = tuple(
                    int(c * (1 - lat_factor)) for c in fractal_adjusted_color
                )
                # Apply edge blending
                if enable_texture_edges:
                    blended_color = tuple(
                        int(c * (1 - 0.5 * edge_blend_map[y, x])) for c in lat_adjusted_color
                    )
                else:
                    blended_color = lat_adjusted_color

                # Apply noise
                if enable_texture_noise:
                    noisy_color = tuple(
                        np.clip(int(c * (0.91 + 0.09 * noise_map[y, x])), 0, 255)
                        for c in blended_color
                    )
                else:
                    noisy_color = blended_color

                # Apply atmospheric fade (reduces intensity toward edges)
                if enable_texture_light:
                    fade_factor = 1.0 - atmospheric_fade_map[y, x]
                    faded_color = tuple(
                        int(c * (1.0 - 0.3 * fade_factor)) for c in noisy_color
                    )
                else:
                    faded_color = noisy_color
                    
                if enable_basic_filters:
                    # Apply biome-specific tint
                    tinted_color = adjust_tint(
                        faded_color,
                        texture_saturation,
                        texture_tint,
                        category,
                        elevation_factor,
                    )
                    # Apply desaturation
                
                    desaturated_color = desaturate_color(tinted_color, texture_saturation)
                    color[y, x] = desaturated_color
    else:
        color = rgb_grid

    color_image = Image.fromarray(color, mode="RGB")
    if process_images:
        if enable_basic_filters:
            color_image = enhance_brightness(color_image, bright_factor)

    surface_image = generate_heightmap(rgb_grid, biome_data)
    ocean_image = generate_ocean_mask(rgb_grid, biome_data)
    ao_image = generate_ao_map(rgb_grid, biome_data)
    rough_image = generate_rough_map(
        height_img=surface_image,
        rgb_grid=rgb_grid,
        biome_data=biome_data,
        ocean_img=ocean_image,
        fractal_map=fractal_map if process_images else None,
        base_value=config.get("texture_roughness_base", 0.2),
        noise_scale=config.get("texture_roughness", 0.15),
        slope_strength=0.5,
    )

    color_image_grayscale = color_image.convert("L")
    normal_image = generate_normal_map(color_image_grayscale)

    return {
        "color": color_image,
        "surface": surface_image,
        "ocean": ocean_image,
        "normal": normal_image,
        "rough": rough_image,
        "ao": ao_image,
    }


def generate_normal_map(height_img, invert_height=True):
    handle_news(None)
    strength = config.get("texture_roughness", 0.5)
    height = np.asarray(height_img).astype(np.float32) / 255.0
    if invert_height:
        height = 1.0 - height
    dx = sobel(height, axis=1) * strength
    dy = -sobel(height, axis=0) * strength
    dz = np.ones_like(height)
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    nx = dx / (length + 1e-8)
    ny = dy / (length + 1e-8)
    nz = dz / (length + 1e-8)
    r = ((nx + 1) * 0.5 * 255).astype(np.uint8)
    g = ((ny + 1) * 0.5 * 255).astype(np.uint8)
    b = ((nz + 1) * 0.5 * 255).astype(np.uint8)
    normal_map = np.stack([r, g, b], axis=-1)

    return Image.fromarray(normal_map, mode="RGB")


def generate_ao_map(
    rgb_grid: np.ndarray,
    biome_data: Dict[int, Dict],
    fade_intensity: float = 1.0,  # Range: 0.1–1.0 (darkness strength)
    fade_spread: float = 1.0,  # Range: 0.1–1.0 (contrast shaping)
) -> Image.Image:
    fade_intensity = config.get("fade_intensity", 0.5)
    fade_spread = config.get("fade_spread", 0.5)
    elevation = generate_elevation(rgb_grid, biome_data)
    blurred = gaussian_filter(elevation.astype(np.float32), sigma=1.0)
    ao = np.clip((blurred - elevation), 0, 255)

    # Normalize and apply fade shaping
    if ao.max() != 0:
        ao = ao / ao.max()  # Normalize to 0–1
    else:
        ao.fill(0.0)

    # Shape and scale using fade_spread
    ao = ao**fade_spread

    # Apply fade_intensity (scales AO darkness)
    ao = 1 - fade_intensity * ao

    # Scale to 128–255 grayscale (light ambient occlusion)
    ao = ao * 127 + 128
    ao = np.clip(ao, 0, 255)

    ao_image = Image.fromarray(ao.astype(np.uint8), mode="L")
    handle_news(
        None,
        "info",
        f"AO map generated: min={ao.min()}, max={ao.max()}, shape={ao.shape}",
    )
    return ao_image


def convert_png_to_dds(
    png_path, dds_output_dir, plugin_name, texture_type, dds_name=None
):
    """Convert a PNG file to DDS using texconv.exe with Starfield-compatible formats."""
    if not TEXCONV_PATH.exists():
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}")

    dds_output_dir.mkdir(parents=True, exist_ok=True)

    format_map = {
        "color": config.get("color_format", "BC7_UNORM_SRGB"),
        "surface": config.get("surface_format", "BC7_UNORM"),
        "surface_metal": config.get("surface_format", "BC7_UNORM"),
        "ocean": config.get("ocean_format", "BC4_UNORM"),
        "ocean_mask": config.get("ocean_format", "BC4_UNORM"),
        "normal": config.get("normal_format", "BC5_SNORM"),
        "rough": config.get("rough_format", "BC4_UNORM"),
        "ao": config.get("ao_format", "BC4_UNORM"),
    }
    dds_format = format_map.get(texture_type, "BC7_UNORM")

    texture_filename = dds_name if dds_name else png_path.stem + ".dds"
    texture_path = dds_output_dir / texture_filename
    texconv_output_name = png_path.stem + ".DDS"
    texconv_output_path = dds_output_dir / texconv_output_name

    cmd = [
        str(TEXCONV_PATH),
        "-f",
        dds_format,
        "-m",
        "0",
        "-y",
        "-o",
        str(dds_output_dir),
        str(png_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if texconv_output_path.exists() and texconv_output_path != texture_path:
            texconv_output_path.rename(texture_path)
        handle_news(
            None,
            "info",
            f"Converted {png_path.name} to {texture_path.name} ({dds_format})",
        )
        return texture_path
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
    # Load biome data without filtering by used_biome_ids initially
    biome_data = load_biome_data(CSV_PATH)
    if not biome_data:
        raise ValueError("No valid biome data loaded from Biomes.csv")

    for biom_path in biom_files:
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path, used_biome_ids, biome_data)
            handle_news(None, "info", f"Loaded {biom_path.name}: biome IDs = {used_biome_ids}")
        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")
            continue

    handle_news(None, "info", f"Used biome IDs: {used_biome_ids}")

    keep_pngs = config.get("keep_pngs_after_conversion", True)
    for biom_path in biom_files:
        planet_name = biom_path.stem
        handle_news(None, "info", f"Processing blueprint for planet {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path, used_biome_ids, biome_data)
            handle_news(None, "info", f"Generating textures for {planet_name} North...")
            maps_n = create_biome_image(biome_grid_n, biome_data)
            handle_news(None, "info", f"Generating textures for {planet_name} South...")
            maps_s = create_biome_image(biome_grid_s, biome_data)

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
                    suffix_map = {
                        "surface": "surface_metal",
                        "ocean": "ocean_mask",
                    }
                    suffix = suffix_map.get(texture_type, texture_type)
                    png_filename = f"{planet_name}_{hemisphere}_{suffix}.png"
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

            for texture_type in [
                "color",
                "surface",
                "ocean",
                "normal",
                "rough",
                "ao",
            ]:
                planet_png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
                suffix_map = {
                    "surface": "surface_metal",
                    "ocean": "ocean_mask",
                }
                suffix = suffix_map.get(texture_type, texture_type)
                north_path = planet_png_dir / f"{planet_name}_North_{suffix}.png"
                south_path = planet_png_dir / f"{planet_name}_South_{suffix}.png"
                combined_path = planet_png_dir / f"{planet_name}_{suffix}.png"

                if north_path.exists() and south_path.exists():
                    try:
                        north_img = Image.open(north_path)
                        south_img = Image.open(south_path)
                        combined_img = Image.new(
                            (
                                "RGB"
                                if texture_type not in ["ao", "ocean", "rough"]
                                else "L"
                            ),
                            (north_img.width, north_img.height + south_img.height),
                        )
                        combined_img.paste(north_img, (0, 0))
                        combined_img.paste(south_img, (0, north_img.height))
                        combined_img.save(combined_path)
                        handle_news(
                            None, "info", f"Combined image saved: {combined_path}"
                        )

                        dds_output_dir = (
                            PLUGINS_DIR
                            / plugin_name
                            / "textures"
                            / plugin_name
                            / "planets"
                            / planet_name
                        )
                        dds_output_dir.mkdir(parents=True, exist_ok=True)

                        dds_name_map = {
                            "color": f"{planet_name}_color.dds",
                            "surface": f"{planet_name}_surface_metal.dds",
                            "ocean": f"{planet_name}_ocean_mask.dds",
                            "normal": f"{planet_name}_normal.dds",
                            "rough": f"{planet_name}_rough.dds",
                            "ao": f"{planet_name}_ao.dds",
                        }
                        dds_filename = dds_name_map[texture_type]

                        try:
                            dds_path = convert_png_to_dds(
                                combined_path,
                                dds_output_dir,
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

    subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetMaterials.pyt")], check=True)
    sys.stdout.flush()
    sys.exit()


if __name__ == "__main__":
    main()
