#!/usr/bin/env python3
"""
Planet Textures Generator

Generates PNG and DDS texture images for planet biomes.
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
from noise import pnoise2
import numpy as np
from typing import Dict, List, Set, Tuple, NamedTuple, cast
import colorsys
import subprocess
import json
import csv
import os
import sys
import shutil
import math
from PIL import Image, ImageEnhance
from PlanetNewsfeed import handle_news
from PlanetTerrain import generate_terrain_normal, generate_normal_map
from PlanetConstants import (
    get_config,
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
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    IMAGE_FILES,
    BOOLEAN_KEYS,
    PROCESSING_MAP,
)


# Global configuration
config = get_config()
plugin_name = config.get("plugin_name", "default_plugin")
planet_name = config.get("planet_name", "default_planet")

GRID_SIZE = (256, 256)
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

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


def load_biom_file(output_dir, planet_name, biome_data, config):
    color_path = output_dir / f"{planet_name}_color.png"
    print(f"Loading color PNG: {color_path}")
    if not color_path.exists():
        handle_news(None, "error", f"Color PNG not found: {color_path}")
        raise FileNotFoundError(f"Color PNG not found: {color_path}")

    try:
        biome_img = Image.open(color_path).convert("RGB")
        biome_array = np.array(biome_img, dtype=np.uint8)
        print(f"Color image shape: {biome_array.shape}")
    except Exception as e:
        handle_news(None, "error", f"Failed to load color PNG {color_path}: {e}")
        raise

    # Set GRID_SIZE once
    global GRID_SIZE, GRID_FLATSIZE
    GRID_SIZE = biome_array.shape[:2]
    GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]
    print(f"Updated GRID_SIZE: {GRID_SIZE}, GRID_FLATSIZE: {GRID_FLATSIZE}")

    return biome_array


def generate_noise(shape, scale=None):
    """Generate larger-patch high-contrast salt-and-pepper noise."""
    if config.get("enable_texture_noise", False):
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


def generate_shadows(shape, intensity=None, spread=None):
    """Generate shadow effect from planet center."""
    handle_news(None)
    if intensity is None:
        intensity = (config.get("texture_contrast", 0.27) * 1.5)
    if spread is None:
        spread = (config.get("texture_brightness", 0.81) * 2)
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y_grid, x_grid = np.indices(shape)
    distance_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    return np.exp(-spread * (distance_from_center / max_distance)) * intensity


def generate_shading(
    grid, light_source_x=0.5, light_source_y=0.5, fade_intensity=0.5, fade_spread=0.5
):
    """Generate anisotropic shading based on terrain gradients with fade controls."""
    if config.get("enable_texture_noise", False):
        handle_news(None)

    # Compute gradient
    grad_x = np.gradient(grid, axis=1)
    grad_y = np.gradient(grid, axis=0)

    # Light projection
    shading = grad_x * light_source_x + grad_y * light_source_y

    # Optional: apply fade_spread as a smoothing factor (e.g., Gaussian blur)
    if fade_spread > 0:
        sigma = max(0.1, fade_spread * 10)  # Spread control, scaled to usable blur
        shading = gaussian_filter(shading, sigma=sigma)

    # Normalize shading to [0, 1]
    shading_min = np.nanmin(shading)
    shading_max = np.nanmax(shading)
    range_val = shading_max - shading_min if shading_max != shading_min else 1.0
    shading = (shading - shading_min) / range_val

    # Apply intensity scaling (fade_intensity can be positive or negative)
    shading = 0.5 + (shading - 0.5) * fade_intensity

    # Sanitize
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
    combined -= erosion_map * 0.5  # Valleys deepen while mountains stabilize

    # **Normalize and Strengthen Terrain Growth**
    combined = safe_normalize(combined)
    combined = np.power(combined, 2.3)  # Enhances ridge prominence

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


def generate_edge_blend(
    rgb_grid: np.ndarray, biome_data: Dict[int, Dict], blend_radius=None
) -> np.ndarray:
    """
    Generate a localized Perlin noise blend map along biome boundaries.
    Non-edge areas stay untouched. Edge-adjacent pixels get a noise mask
    with smooth radial falloff (like an edge feather).
    """
    if config.get("enable_texture_edges", False):
        handle_news(None)

    if not config.get("enable_texture_edges", False):
        return np.zeros(rgb_grid.shape[:2], dtype=np.float32)

    if blend_radius is None:
        blend_radius = config.get("texture_edges", 0.53)

    h, w = rgb_grid.shape[:2]
    edge_map = np.zeros((h, w), dtype=np.float32)
    rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}

    # --- Step 1: Detect biome edge pixels ---
    for y in range(h):
        for x in range(w):
            current_rgb = tuple(rgb_grid[y, x])
            current_form_id = rgb_to_form_id.get(current_rgb, None)
            if current_form_id is None:
                continue

            neighbors = [
                tuple(rgb_grid[max(y - 1, 0), x]),
                tuple(rgb_grid[min(y + 1, h - 1), x]),
                tuple(rgb_grid[y, max(x - 1, 0)]),
                tuple(rgb_grid[y, min(x + 1, w - 1)]),
            ]
            neighbor_ids = [rgb_to_form_id.get(n, None) for n in neighbors]
            if any(nid != current_form_id and nid is not None for nid in neighbor_ids):
                edge_map[y, x] = 1.0

    # --- Step 2: Expand edge zones with Gaussian blur (falloff mask) ---
    edge_falloff = gaussian_filter(edge_map, sigma=max(1.0, 3 * (1 - blend_radius)))
    edge_falloff = np.clip(edge_falloff, 0, 1)

    # --- Step 3: Generate Perlin noise map ---
    from noise import pnoise2

    seed = int(config.get("global_seed", 42))
    scale = 0.03 + (1.0 - blend_radius) * 0.07  # smaller = smoother noise

    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    xv, yv = np.meshgrid(x, y)
    noise_func = np.vectorize(lambda x, y: pnoise2(x * scale, y * scale, base=seed))
    perlin = noise_func(xv, yv)
    perlin -= perlin.min()
    perlin /= perlin.max() + 1e-6

    # Optional: boost contrast for sharper transition (but still soft)
    contrast = (config.get("texture_contrast", 0.5) * 3)
    perlin = np.clip((perlin - 0.1) * contrast + 0.1, 0, 1)

    # --- Step 4: Multiply perlin noise only where edge falloff is active ---
    blend_mask = edge_falloff * perlin

    return blend_mask.astype(np.float32)


def enhance_brightness(image, bright_factor=None):
    """Enhance image brightness."""
    handle_news(None)
    if bright_factor is None:
        bright_factor = config.get("texture_brightness", 0.74)
    scaled_factor = bright_factor * 3
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(scaled_factor)


def adjust_tint(faded_color, texture_saturation, texture_tint):
    """Adjust color hue and saturation based on user input sliders."""
    r, g, b = [c / 255.0 for c in faded_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Shift hue: texture_tint in [0, 1], center 0.5 = no change
    h = (h + (texture_tint - 0.5) * 0.25) % 1.0

    # Adjust saturation: center 0.5 = no change
    s = s * (0.5 + texture_saturation)

    r, g, b = colorsys.hsv_to_rgb(h, min(s, 1.0), v)
    return tuple(int(np.clip(c * 255, 0, 255)) for c in (r, g, b))


def desaturate_color(rgb, texture_saturation):
    """Adjust color saturation: 0.5 = no change, <0.5 = desaturate, >0.5 = boost."""
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    if texture_saturation < 0.5:
        factor = texture_saturation * 2.0  # 0.0 → 0.0, 0.5 → 1.0
        s *= factor
    else:
        boost = (texture_saturation - 0.5) * 2.0  # 0.5 → 0.0, 1.0 → 1.0
        s = s + (1.0 - s) * boost

    s = min(max(s, 0.0), 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(np.clip(c * 255, 0, 255)) for c in (r, g, b))


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


def create_biome_image(
    rgb_grid: np.ndarray, biome_data: Dict[int, Dict]
) -> Dict[str, Image.Image]:
    """Generate biome texture images from RGB grid and biome data."""
    enable_basic_filters = config.get("enable_basic_filters", True)
    enable_texture_noise = config.get("enable_texture_noise", True)
    enable_texture_edges = config.get("enable_texture_edges", True)
    enable_texture_light = config.get("enable_texture_light", True)
    enable_texture_terrain = config.get("enable_texture_terrain", True)
    process_images = config.get("process_images", False)
    bright_factor = config.get("texture_brightness", 0.05)
    texture_saturation = config.get("texture_saturation", 0.5)
    texture_tint = config.get("texture_tint", 0.5)
    fractal_map = None

    png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
    ocean_image_path = png_dir / f"{planet_name}_ocean_mask.png"
    surface_image_path = png_dir / f"{planet_name}_height.png"
    terrain_image_path = png_dir / f"{planet_name}_terrain.png"
    color_image_path = (
        png_dir / f"{planet_name}_color.png"
    )  # Path to original color.png

    # Load input images
    ocean_image = Image.open(ocean_image_path).convert("L")
    surface_image = Image.open(surface_image_path).convert("L")

    # Load original color.png instead of using rgb_grid
    if not color_image_path.exists():
        handle_news(None, "error", f"Original color PNG not found: {color_image_path}")
        raise FileNotFoundError(f"Original color PNG not found: {color_image_path}")
    color_image = Image.open(color_image_path).convert("RGB")
    color = np.array(color_image, dtype=np.uint8)  # Use original color map

    height, width, _ = rgb_grid.shape
    if color.shape[:2] != (height, width):
        handle_news(
            None,
            "warning",
            f"Resizing color image to match biome grid {height}x{width}",
        )
        color_image = color_image.resize((width, height), Image.Resampling.LANCZOS)
        color = np.array(color_image, dtype=np.uint8)

    handle_news(None)

    if process_images:
        noise_map = generate_noise(
            (height, width), scale=config.get("noise_scale", 4.17)
        )
        elevation_map = generate_elevation(rgb_grid, biome_data)  # Use biome grid
        generate_terrain_normal(
            river_mask_path=str(png_dir / f"{planet_name}_river_mask.png"),
            terrain_image_path=str(terrain_image_path),
            mountain_mask_path=str(terrain_image_path),
            output_path=str(png_dir / f"{planet_name}_terrain_normal.png"),
        )
        edge_blend_map = generate_edge_blend(rgb_grid, biome_data)  # Use biome grid
        shading_map = generate_shading(elevation_map)
        elevation_norm = elevation_map / 255.0
        fractal_map = generate_fractal_noise(elevation_norm)
        atmospheric_fade_map = generate_shadows((height, width))
        if enable_basic_filters:
            bright_factor = config.get("texture_brightness", 0.74)

        rgb_to_form_id = {tuple(v["color"]): k for k, v in biome_data.items()}
        for y in range(height):
            for x in range(width):
                rgb = tuple(color[y, x])  # Start with original color
                form_id = rgb_to_form_id.get(tuple(rgb_grid[y, x]), None)  # Biome ID
                category = (
                    biome_data.get(form_id, {}).get("category", "")
                    if form_id is not None
                    else ""
                )
                lat_factor = abs((y / height) - 0.5) * 0.4
                elevation_factor = elevation_map[y, x] / 255.0

                # Apply elevation-based shading
                shaded_color = tuple(
                    int(c * (0.8 + 0.2 * elevation_factor)) for c in rgb
                )
                # Apply light-based shading
                if enable_basic_filters:
                    light_adjusted_color = tuple(
                        int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color
                    )
                else:
                    light_adjusted_color = shaded_color

                # Apply fractal noise
                fractal_adjusted_color = tuple(
                    int(c * (0.85 + 0.15 * fractal_map[y, x]))
                    for c in light_adjusted_color
                )
                # Apply latitude-based darkening
                lat_adjusted_color = tuple(
                    int(c * (1.0 - lat_factor)) for c in fractal_adjusted_color
                )
                # Apply edge blending
                if enable_texture_edges:
                    blended_color = tuple(
                        int(c * (1 - 0.5 * edge_blend_map[y, x]))
                        for c in lat_adjusted_color
                    )
                else:
                    blended_color = lat_adjusted_color

                # Apply noise
                if enable_texture_noise:
                    noisy_color = tuple(
                        np.clip(int(c * (0.95 + 0.05 * noise_map[y, x])), 0, 255)
                        for c in blended_color
                    )
                else:
                    noisy_color = blended_color

                # Apply atmospheric fade
                if enable_basic_filters:
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
                    )
                    # Apply desaturation
                    desaturated_color = desaturate_color(
                        tinted_color, texture_saturation
                    )
                    color[y, x] = desaturated_color
                else:
                    color[y, x] = faded_color

    color_image = Image.fromarray(color, mode="RGB")
    if process_images and enable_basic_filters:
        color_image = enhance_brightness(color_image, bright_factor)

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

    maps = {
        "color": color_image,
        "surface": surface_image,
        "ocean": ocean_image,
        "normal": normal_image,
        "rough": rough_image,
        "ao": ao_image,
    }

    print(f"Generated maps: {list(maps.keys())}")
    for key, img in maps.items():
        print(f"Map {key} mode: {img.mode}, size: {img.size}")
    return maps


def generate_ao_map(
    rgb_grid: np.ndarray,
    biome_data: Dict[int, Dict],
    fade_intensity: float = 0.1,  # Range: 0.1–1.0 (darkness strength)
    fade_spread: float = 0.1,  # Range: 0.1–1.0 (contrast shaping)
) -> Image.Image:
    if config.get("enable_texture_light", False):
        handle_news(None)
    fade_intensity = config.get("fade_intensity", 0.5)
    fade_spread = config.get("fade_spread", 0.5)
    elevation = generate_elevation(rgb_grid, biome_data)
    #blurred = gaussian_filter(elevation.astype(np.float32), sigma=0.2)
    ao = np.clip((elevation), 0, 255)

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
        "terrain_normal": config.get("normal_format", "BC5_SNORM"),
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
    global planet_name
    print(f"Landscaping permit approved for: {plugin_name}", flush=True)
    print("=== Starting PlanetTextures ===", flush=True)

    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handle_news(None, "info", f"PNG output directory: {PNG_OUTPUT_DIR}")

    # Load biome data
    biome_data = load_biome_data(CSV_PATH)
    if not biome_data:
        raise ValueError("No valid biome data loaded from Biomes.csv")

    output_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        biome_grid = load_biom_file(output_dir, planet_name, biome_data, config)
        handle_news(None, "info", f"Loaded unified biome for {planet_name}")
    except Exception as e:
        print(f"Error loading biome for {planet_name}: {e}")
        sys.exit(1)

    keep_pngs = config.get("keep_pngs_after_conversion", True)
    handle_news(None, "info", f"Processing blueprint for planet {planet_name}")

    try:
        handle_news(None, "info", f"Generating textures for {planet_name}...")
        maps = create_biome_image(biome_grid, biome_data)
        handle_news(None, "debug", f"Maps generated: {list(maps.keys())}")

        copied_textures = set()

        dds_output_dir = (
            PLUGINS_DIR
            / plugin_name
            / "textures"
            / plugin_name
            / "planets"
            / planet_name
        )

        dds_name_map = {
            "color": f"{planet_name}_color.dds",
            "surface": f"{planet_name}_surface_metal.dds",
            "ocean": f"{planet_name}_ocean_mask.dds",
            "normal": f"{planet_name}_normal.dds",
            "rough": f"{planet_name}_rough.dds",
            "ao": f"{planet_name}_ao.dds",
            "terrain_normal": f"{planet_name}_terrain_normal.dds",
        }

        for texture_type, img in maps.items():
            suffix_map = {
                "surface": "surface_metal",
                "ocean": "ocean_mask",
            }
            suffix = suffix_map.get(texture_type, texture_type)
            png_filename = f"{planet_name}_{suffix}.png"
            png_path = output_dir / png_filename

            img.save(png_path)
            handle_news(None, "info", f"Saved PNG: {png_path}")

            dds_filename = dds_name_map.get(texture_type, f"{planet_name}_{suffix}.dds")
            dds_path = convert_png_to_dds(
                png_path,
                dds_output_dir,
                plugin_name,
                texture_type,
                dds_filename,
            )
            handle_news(None, "info", f"DDS saved: {dds_path}")

            if not keep_pngs:
                os.remove(png_path)
                handle_news(None, "info", f"Deleted PNG: {png_path}")

        # Generate terrain normal if needed
        if "ocean" in maps:
            try:
                terrain_path = output_dir / f"{planet_name}_terrain.png"
                terrain_normal_path = output_dir / f"{planet_name}_terrain_normal.png"
                if terrain_path.exists():
                    generate_terrain_normal(
                        river_mask_path=str(
                            output_dir / f"{planet_name}_river_mask.png"
                        ),
                        terrain_image_path=str(terrain_path),
                        mountain_mask_path=str(terrain_path),
                        output_path=str(terrain_normal_path),
                    )
                    handle_news(
                        None, "info", f"Generated terrain normal: {terrain_normal_path}"
                    )
                    # Convert terrain normal to DDS
                    dds_path = convert_png_to_dds(
                        terrain_normal_path,
                        dds_output_dir,
                        plugin_name,
                        "terrain_normal",
                        dds_name_map["terrain_normal"],
                    )
                    handle_news(None, "info", f"DDS saved: {dds_path}")
                    if not keep_pngs:
                        os.remove(terrain_normal_path)
                        handle_news(None, "info", f"Deleted PNG: {terrain_normal_path}")
                else:
                    handle_news(
                        None, "warning", f"Terrain map not found: {terrain_path}"
                    )
            except Exception as e:
                handle_news(None, "error", f"Terrain normal generation failed: {e}")

        print(f"Generated textures for {planet_name}: {list(maps.keys())}")

    except Exception as e:
        print(f"Error processing {planet_name}: {e}")
        sys.exit(1)

    print(f"Visual inspection of {planet_name} complete.")
    if config.get("run_planet_materials", True):
        subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetMaterials.py")], check=True)
    else:
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
