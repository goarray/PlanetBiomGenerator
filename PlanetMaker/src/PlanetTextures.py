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
from typing import Dict, List, Set, Tuple, NamedTuple, cast, Optional
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
from PlanetUtils import biome_db
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


def get_output_paths(plugin_name, planet_name):
    base = PNG_OUTPUT_DIR / plugin_name / planet_name
    return {
        "color": base / f"{planet_name}_color.png",  # full color smooth image, res*2res
        "rough": base / f"{planet_name}_height.png",  # smooth greyscale heightmap,
        "ocean_mask": base / f"{planet_name}_ocean_mask.png",
        "surface_metal": base / f"{planet_name}_river_mask.png",
        "ao": base / f"{planet_name}_colony_mask.png",
    }


def load_image_layers(
    paths: dict[str, Path],
    resolution: int = 256,
    fallback: Path = DEFAULT_IMAGE_PATH,
) -> dict[str, np.ndarray]:
    """
    Load all available image layers from given paths, resizing them to match
    (2 * resolution, resolution). If a layer is missing, fall back to default image.
    """
    target_size = (resolution, 2 * resolution)  # width, height
    layers = {}

    for layer_name, path in paths.items():
        img_path = path if path.exists() else fallback

        if not img_path.exists():
            handle_news(
                None,
                "error",
                f"Missing both {path.name} and fallback image: {fallback}",
            )
            raise FileNotFoundError(
                f"Missing image for layer '{layer_name}' and no fallback available."
            )

        try:
            img = Image.open(img_path)
            img = img.convert("RGB") if img.mode == "RGB" else img.convert("L")

            if img.size != target_size:
                img = img.resize(target_size, resample=Image.Resampling.BILINEAR)

            arr = np.array(img)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]

            layers[layer_name] = arr
        except Exception as e:
            handle_news(None, "warning", f"Failed to load layer '{layer_name}': {e}")

    return layers


def generate_noise(shape, scale=None):
    """Generate larger-patch high-contrast salt-and-pepper noise."""
    if config.get("enable_texture_noise", False):
        handle_news(None)
    if scale is None:
        scale = config.get("texture_noise", 4.17)
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


def generate_elevation(rgb_grid: np.ndarray) -> np.ndarray:
    """Generate elevation from an RGB biome grid using heights from biome_db."""
    handle_news(None)
    height, width, _ = rgb_grid.shape
    elevation = np.zeros((height, width), dtype=np.uint8)

    # Build RGB → BiomeEntry map once
    rgb_to_biome = {b.color: b for b in biome_db.all_biomes()}

    for y in range(height):
        for x in range(width):
            rgb = tuple(rgb_grid[y, x])
            biome = rgb_to_biome.get(rgb)
            elevation[y, x] = biome.height if biome else 127

    # Optional: smooth elevation
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


def generate_rgb_based_roughness_map(
    rgb_grid: np.ndarray,
    height: np.ndarray,
    base_value: float = 0.2,
    biome_rough_lookup: dict[tuple[int, int, int], float] = {},
) -> np.ndarray:
    """
    Generate roughness values based on RGB grid.
    - Uses biome RGB as lookup key.
    - If no match, uses base_value.
    """
    biome_rough_map = np.full_like(height, base_value, dtype=np.float32)

    # Iterate only over unique RGB triplets to minimize cost
    unique_colors = np.unique(rgb_grid.reshape(-1, 3), axis=0)

    for rgb in unique_colors:
        rgb_tuple = tuple(rgb.tolist())
        rough = biome_rough_lookup.get(rgb_tuple, base_value)
        mask = np.all(rgb_grid == rgb, axis=2)
        biome_rough_map[mask] = rough * (1 + height[mask] * 0.5)

    return biome_rough_map


def generate_rough_map(
    height_img,
    rgb_grid,
    ocean_img=None,
    fractal_map=None,
    base_value=None,
    texture_noise=None,
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
    if texture_noise is None:
        texture_noise = config.get("texture_roughness", 0.15)

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
    biome_rough_map = generate_rgb_based_roughness_map(
        rgb_grid,
        height,
        base_value=base_value or 0.2,
        biome_rough_lookup={
            (0, 128, 255): 0.05,
            (34, 139, 34): 0.3,
            (210, 180, 140): 0.5,
            (255, 255, 255): 0.9,
        },
    )

    roughness += biome_rough_map

    # --- Fractal influence ---
    if fractal_map is not None:
        fractal_map = (fractal_map - fractal_map.min()) / (
            fractal_map.max() - fractal_map.min() + 1e-6
        )
        roughness += texture_noise * fractal_map

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


def create_biome_image(image_data: Dict[str, np.ndarray]) -> Dict[str, Image.Image]:
    """Process input image layers and generate final texture maps."""

    enable_basic_filters = config.get("enable_basic_filters", True)
    enable_texture_noise = config.get("enable_texture_noise", True)
    enable_texture_edges = config.get("enable_texture_edges", True)
    process_images = config.get("process_images", False)
    bright_factor = config.get("texture_brightness", 0.05)
    texture_saturation = config.get("texture_saturation", 0.5)
    texture_tint = config.get("texture_tint", 0.5)
    fractal_map = None

    # Required base maps
    color_input = image_data.get("color")
    if color_input is None:
        color_input = image_data.get("biome")
        if color_input is None:
            raise ValueError("No color or biome image found to base textures on.")
    surface_input = image_data.get("terrain", np.zeros_like(color_input[:, :, 0]))
    ocean_input = image_data.get("ocean", np.zeros_like(color_input[:, :, 0]))

    surface_image = Image.fromarray(surface_input).convert("L")
    ocean_image = Image.fromarray(ocean_input).convert("L")

    # Color input or fallback to biome
    color = np.array(color_input, dtype=np.uint8)
    height, width, _ = color.shape

    handle_news(None)

    if process_images:
        noise_map = generate_noise(
            (height, width), scale=config.get("texture_noise", 4.17)
        )
        elevation_map = color_input[:, :, 0]  # crude elevation from R channel
        edge_blend_map = np.zeros(
            (height, width)
        )  # optional: use edge detection from RGB
        shading_map = generate_shading(elevation_map)
        elevation_norm = elevation_map / 255.0
        fractal_map = generate_fractal_noise(elevation_norm)
        atmospheric_fade_map = generate_shadows((height, width))

        for y in range(height):
            for x in range(width):
                rgb = tuple(color[y, x])
                elevation_factor = elevation_map[y, x] / 255.0
                lat_factor = abs((y / height) - 0.5) * 0.4

                shaded_color = tuple(
                    int(c * (0.8 + 0.2 * elevation_factor)) for c in rgb
                )

                if enable_basic_filters:
                    light_adjusted_color = tuple(
                        int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color
                    )
                else:
                    light_adjusted_color = shaded_color

                fractal_adjusted_color = tuple(
                    int(c * (0.85 + 0.15 * fractal_map[y, x]))
                    for c in light_adjusted_color
                )

                lat_adjusted_color = tuple(
                    int(c * (1.0 - lat_factor)) for c in fractal_adjusted_color
                )

                if enable_texture_edges:
                    blended_color = tuple(
                        int(c * (1 - 0.5 * edge_blend_map[y, x]))
                        for c in lat_adjusted_color
                    )
                else:
                    blended_color = lat_adjusted_color

                if enable_texture_noise:
                    noisy_color = tuple(
                        np.clip(int(c * (0.95 + 0.05 * noise_map[y, x])), 0, 255)
                        for c in blended_color
                    )
                else:
                    noisy_color = blended_color

                if enable_basic_filters:
                    fade_factor = 1.0 - atmospheric_fade_map[y, x]
                    faded_color = tuple(
                        int(c * (1.0 - 0.3 * fade_factor)) for c in noisy_color
                    )
                else:
                    faded_color = noisy_color

                if enable_basic_filters:
                    tinted_color = adjust_tint(
                        faded_color, texture_saturation, texture_tint
                    )
                    desaturated_color = desaturate_color(
                        tinted_color, texture_saturation
                    )
                    color[y, x] = desaturated_color
                else:
                    color[y, x] = faded_color

    color_image = Image.fromarray(color, mode="RGB")
    if process_images and enable_basic_filters:
        color_image = enhance_brightness(color_image, bright_factor)

    ao_image = generate_ao_map(color_input)
    rough_image = generate_rough_map(
        height_img=surface_image,
        rgb_grid=color_input,
        ocean_img=ocean_image,
        fractal_map=fractal_map if process_images else None,
        base_value=config.get("texture_roughness_base", 0.2),
        texture_noise=config.get("texture_roughness", 0.15),
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


def generate_rgb_based_elevation(
    rgb_grid: np.ndarray,
    elevation_lookup: Optional[dict[tuple[int, int, int], float]] = None,
    default_elevation: float = 0.5,
) -> np.ndarray:
    """
    Converts biome RGB colors into an elevation map (0.0 to 1.0).
    """
    height, width, _ = rgb_grid.shape
    elevation = np.full((height, width), default_elevation, dtype=np.float32)

    if elevation_lookup is None:
        elevation_lookup = {
            (0, 128, 255): 0.0,  # ocean
            (34, 139, 34): 0.4,  # forest
            (210, 180, 140): 0.6,  # desert
            (255, 255, 255): 0.9,  # snow
        }

    unique_colors = np.unique(rgb_grid.reshape(-1, 3), axis=0)
    for rgb in unique_colors:
        rgb_tuple = tuple(rgb.tolist())
        elev_value = elevation_lookup.get(rgb_tuple, default_elevation)
        mask = np.all(rgb_grid == rgb, axis=2)
        elevation[mask] = elev_value

    # Normalize to 0–255
    elevation = np.clip(elevation * 255, 0, 255)
    return elevation


def generate_ao_map(rgb_grid: np.ndarray) -> Image.Image:
    if config.get("enable_texture_light", False):
        handle_news(None)
    fade_intensity = config.get("fade_intensity", 0.5)
    fade_spread = config.get("fade_spread", 0.5)
    elevation = generate_elevation(rgb_grid)
    # blurred = gaussian_filter(elevation.astype(np.float32), sigma=0.2)
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
    config = get_config()
    print("Config ID:", id(config))

    global plugin_name
    global planet_name

    print(f"Landscaping permit approved for: {plugin_name}", flush=True)
    print("=== Starting PlanetTextures ===", flush=True)

    resolution = config.get("texture_resolution", 256)
    keep_pngs = config.get("keep_pngs_after_conversion", True)

    output_dir = PNG_OUTPUT_DIR / plugin_name / planet_name
    output_dir.mkdir(parents=True, exist_ok=True)
    handle_news(None, "info", f"PNG output directory: {output_dir}")

    # Load input image files
    paths = get_output_paths(plugin_name, planet_name)
    image_data = load_image_layers(
        paths, resolution, fallback=IMAGE_DIR / "default.png"
    )

    # Output DDS path base
    dds_output_dir = (
        PLUGINS_DIR / plugin_name / "textures" / plugin_name / "planets" / planet_name
    )
    dds_output_dir.mkdir(parents=True, exist_ok=True)

    dds_name_map = {
        "color": f"{planet_name}_color.dds",
        "surface": f"{planet_name}_surface_metal.dds",
        "ocean": f"{planet_name}_ocean_mask.dds",
        "normal": f"{planet_name}_normal.dds",
        "rough": f"{planet_name}_rough.dds",
        "ao": f"{planet_name}_ao.dds",
        "terrain_normal": f"{planet_name}_terrain_normal.dds",
    }

    # Convert each image to DDS
    for texture_type, img in image_data.items():
        suffix = {
            "surface": "surface_metal",
            "ocean": "ocean_mask",
        }.get(texture_type, texture_type)

        png_path = output_dir / f"{planet_name}_{suffix}.png"
        if texture_type in ("ao", "surface_metal"):
            img = 255 - img
        Image.fromarray(img).save(png_path)
        handle_news(None, "info", f"Saved PNG: {png_path}")

        dds_filename = dds_name_map.get(texture_type, f"{planet_name}_{suffix}.dds")
        dds_path = convert_png_to_dds(
            png_path, dds_output_dir, plugin_name, texture_type, dds_filename
        )
        handle_news(None, "info", f"DDS saved: {dds_path}")

        if not keep_pngs:
            os.remove(png_path)
            handle_news(None, "info", f"Deleted PNG: {png_path}")

    # Special case: terrain_normal
    terrain_path = output_dir / f"{planet_name}_terrain.png"
    if terrain_path.exists():
        terrain_normal_path = output_dir / f"{planet_name}_terrain_normal.png"
        try:
            generate_terrain_normal(output_path=str(terrain_normal_path))
            handle_news(
                None, "info", f"Generated terrain normal: {terrain_normal_path}"
            )

            dds_path = convert_png_to_dds(
                terrain_normal_path,
                dds_output_dir,
                plugin_name,
                "terrain_normal",
                dds_name_map["normal"],
            )
            handle_news(None, "info", f"DDS saved: {dds_path}")

            if not keep_pngs:
                os.remove(terrain_normal_path)
                handle_news(None, "info", f"Deleted PNG: {terrain_normal_path}")
        except Exception as e:
            handle_news(None, "error", f"Terrain normal generation failed: {e}")
    else:
        handle_news(None, "warning", f"Terrain map not found: {terrain_path}")

    # Finalization
    print(f"Visual inspection of {planet_name} complete.")
    if config.get("run_planet_materials", True):
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "PlanetMaterials.py")], check=True
        )
    else:
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
