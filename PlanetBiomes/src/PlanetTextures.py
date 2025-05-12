#!/usr/bin/env python3
"""
Planet Textures Generator

Generates PNG texture images for planet biomes based on .biom files.
Applies effects like noise, elevation, shading, craters, and edge blending
to create realistic planetary visuals. Uses configuration from JSON and
biome colors from CSV.

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
"""

from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter
import numpy as np
import colorsys
import json
import csv
import sys
from PIL import Image, ImageEnhance

# Directory paths
BASE_DIR = Path(__file__).parent.parent
SCRIPT_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "Output"
PNG_OUTPUT_DIR = OUTPUT_DIR / "BiomePNGs"

# File paths
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
BIOMES_CSV_PATH = ASSETS_DIR / "Biomes.csv"

# Grid constants
GRID_SIZE = [256, 256]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]

# Global configuration
config = {}

# Define .biom file structure
CsSF_Biom = Struct(
    "magic" / Const(0x105, UInt16),
    "_numBiomes" / Rebuild(UInt32, len_(this.biomeIds)),
    "biomeIds" / UInt32[this._numBiomes],
    Const(2, UInt32),
    Const(GRID_SIZE, UInt32[2]),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridN" / UInt32[GRID_FLATSIZE],
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridN" / UInt8[GRID_FLATSIZE],
    Const(GRID_SIZE, UInt32[2]),
    Const(GRID_FLATSIZE, UInt32),
    "biomeGridS" / UInt32[GRID_FLATSIZE],
    Const(GRID_FLATSIZE, UInt32),
    "resrcGridS" / UInt8[GRID_FLATSIZE],
)

def load_config():
    """Load configuration from JSON file."""
    global config
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {CONFIG_PATH} not found.")
        config = {}

# Initialize configuration
load_config()

def load_biome_colors(csv_path, used_biome_ids, saturate_factor=None):
    """Load RGB colors for used biome IDs from CSV."""
    if saturate_factor is None:
        saturate_factor = config["image_pipeline"].get("saturation_factor", 1.0)

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

def load_biom_file(filepath):
    """Load .biom file and return biome grids as numpy arrays."""
    with open(filepath, "rb") as f:
        data = CsSF_Biom.parse_stream(f)
    biome_grid_n = np.array(data.biomeGridN, dtype=np.uint32).reshape(GRID_SIZE[1], GRID_SIZE[0])
    biome_grid_s = np.array(data.biomeGridS, dtype=np.uint32).reshape(GRID_SIZE[1], GRID_SIZE[0])
    return biome_grid_n, biome_grid_s

def upscale_image(image, target_size=(1024, 1024)):
    """Upscale image to target size if enabled in config."""
    if config["image_pipeline"]["upscale_image"]:
        return image.resize(target_size, Image.Resampling.LANCZOS)
    return image

def generate_noise(shape, scale=None):
    """Generate smoothed noise for texture variation."""
    if scale is None:
        scale = config["image_pipeline"]["noise_scale"]
    base_noise = np.random.rand(*shape)
    smoothed = gaussian_filter(base_noise, sigma=scale)
    return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

def generate_elevation(shape, scale=None):
    """Generate elevation map for shading effects."""
    if scale is None:
        scale = config["image_pipeline"]["elevation_scale"]
    base_noise = np.random.rand(*shape)
    smoothed = gaussian_filter(base_noise, sigma=scale)
    return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

def generate_atmospheric_fade(shape, intensity=None, spread=None):
    """Generate atmospheric fade effect from planet center."""
    if intensity is None:
        intensity = config["image_pipeline"]["fade_intensity"]
    if spread is None:
        spread = config["image_pipeline"]["fade_spread"]
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y_grid, x_grid = np.indices(shape)
    distance_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    return np.exp(-spread * (distance_from_center / max_distance)) * intensity

def generate_shading(grid, light_source_x=None, light_source_y=None):
    """Generate anisotropic shading based on terrain gradients."""
    if light_source_x is None:
        light_source_x = config["image_pipeline"]["light_source_x"]
    if light_source_y is None:
        light_source_y = config["image_pipeline"]["light_source_y"]
    grad_x = np.gradient(grid, axis=1)
    grad_y = np.gradient(grid, axis=0)
    shading = np.clip(grad_x * light_source_x + grad_y * light_source_y, -1, 1)
    return (shading - shading.min()) / (shading.max() - shading.min())

def generate_fractal_noise(shape, octaves=None, detail_smoothness=None, detail_strength_decay=None):
    """Generate fractal noise for terrain complexity."""
    if octaves is None:
        octaves = config["image_pipeline"].get("fractal_octaves", 4)
    if detail_smoothness is None:
        detail_smoothness = config["image_pipeline"].get("detail_smoothness", 2)
    if detail_strength_decay is None:
        detail_strength_decay = config["image_pipeline"].get("detail_strength_decay", 0.5)
    base = np.random.rand(*shape)
    combined = np.zeros_like(base)
    for i in range(octaves):
        sigma = detail_smoothness ** (i + 1)
        weight = detail_strength_decay ** i
        combined += gaussian_filter(base, sigma=sigma) * weight
    return (combined - combined.min()) / (combined.max() - combined.min())

def add_craters(grid, num_craters=30, max_radius=None, crater_depth_min=None, crater_depth_max=None):
    """Add impact craters to terrain grid."""
    if max_radius is None:
        max_radius = config["image_pipeline"].get("crater_max_radius", 20)
    if max_radius <= 5:
        print(f"WARNING: max_radius ({max_radius}) is too low, adjusting to 6.")
        max_radius = 6
    if crater_depth_min is None:
        crater_depth_min = config["image_pipeline"].get("crater_depth_min", 0.2)
    if crater_depth_max is None:
        crater_depth_max = config["image_pipeline"].get("crater_depth_max", 0.8)
    if crater_depth_min >= crater_depth_max:
        crater_depth_max = crater_depth_min + 0.01
        print(f"WARNING: Adjusted crater_depth_max to {crater_depth_max} to ensure min < max")
    crater_map = np.zeros_like(grid, dtype=np.float32)
    for _ in range(num_craters):
        cx, cy = np.random.randint(0, GRID_SIZE[0]), np.random.randint(0, GRID_SIZE[1])
        radius = np.random.randint(3, max_radius // 2)
        y_grid, x_grid = np.indices(grid.shape)
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        crater_depth = np.exp(-dist / radius) * np.random.uniform(crater_depth_min, crater_depth_max) * 1.5
        crater_map -= crater_depth
    return np.clip(grid + crater_map, 0, 1)

def generate_crater_shading(crater_map):
    """Generate shading for crater rims."""
    if config["image_pipeline"]["enable_crater_shading"]:
        shading = np.gradient(crater_map, axis=0) + np.gradient(crater_map, axis=1)
        min_shading, max_shading = shading.min(), shading.max()
        if max_shading == min_shading:
            return np.zeros_like(crater_map)
        normalized_shading = (shading - min_shading) / (max_shading - min_shading)
        normalized_shading[np.isnan(normalized_shading)] = 0
        return normalized_shading
    return np.zeros_like(crater_map)

def generate_edge_blend(grid, blend_radius=None):
    """Generate edge blending map for biome transitions."""
    if not config["image_pipeline"].get("enable_edge_blending", True):
        return np.zeros_like(grid, dtype=np.float32)
    if blend_radius is None:
        blend_radius = config["image_pipeline"].get("edge_blend_radius", 10)
    edge_map = np.zeros_like(grid, dtype=np.float32)
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            current_biome = grid[y, x]
            neighbors = [
                grid[max(y - 1, 0), x], grid[min(y + 1, GRID_SIZE[1] - 1), x],
                grid[y, max(x - 1, 0)], grid[y, min(x + 1, GRID_SIZE[0] - 1)]
            ]
            if any(neighbor != current_biome for neighbor in neighbors):
                edge_map[y, x] = 1.0
    return gaussian_filter(edge_map, sigma=blend_radius)

def desaturate_color(rgb, saturate_factor=1.2):
    """Adjust color saturation in HSV space."""
    if saturate_factor is None:
        saturate_factor = config["image_pipeline"].get("saturation_factor", 1.0)
    h, s, v = colorsys.rgb_to_hsv(*[c / 255.0 for c in rgb])
    s *= saturate_factor
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in (r, g, b))

def enhance_brightness(image, bright_factor=1.5):
    """Enhance image brightness."""
    if bright_factor is None:
        bright_factor = config["image_pipeline"].get("brightness_factor", 1.0)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(bright_factor)

def create_biome_image(grid, biome_colors, default_color=(128, 128, 128)):
    """Generate biome image with visual effects."""
    image = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
    noise_map = generate_noise((GRID_SIZE[1], GRID_SIZE[0]))
    elevation_map = generate_elevation((GRID_SIZE[1], GRID_SIZE[0]))
    edge_blend_map = generate_edge_blend(grid)
    shading_map = generate_shading(elevation_map)
    fractal_map = generate_fractal_noise((GRID_SIZE[1], GRID_SIZE[0]))
    crater_map = add_craters(elevation_map)
    crater_shading = generate_crater_shading(crater_map)

    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            form_id = int(grid[y, x])
            biome_color = biome_colors.get(form_id, default_color)
            lat_factor = abs((y / GRID_SIZE[1]) - 0.5) * 0.4
            shaded_color = tuple(int(c * (0.8 + 0.2 * elevation_map[y, x])) for c in biome_color)
            light_adjusted_color = tuple(int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color)
            fractal_adjusted_color = tuple(int(c * (0.85 + 0.15 * fractal_map[y, x])) for c in light_adjusted_color)
            crater_adjusted_color = tuple(int(c * (0.7 + 0.3 * crater_shading[y, x])) for c in fractal_adjusted_color)
            lat_adjusted_color = tuple(int(c * (1 - lat_factor)) for c in crater_adjusted_color)
            blended_color = tuple(int(c * (1 - 0.3 * edge_blend_map[y, x])) for c in lat_adjusted_color)
            final_color = tuple(int(c * (0.95 + 0.05 * noise_map[y, x])) for c in blended_color)
            image[y, x] = final_color

    biome_image = Image.fromarray(image)
    return enhance_brightness(biome_image, bright_factor=1.0)

def main():
    """Process .biom files and generate PNG textures."""
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    biom_files = [
        f for f in OUTPUT_DIR.rglob("*.biom")
        if f.parent != OUTPUT_DIR and "assets" not in str(f.parent)
    ]

    if not biom_files:
        print("No .biom files found in the output directory.")
        sys.exit(1)

    used_biome_ids = set()
    for biom_path in biom_files:
        print(f"Collecting biome IDs from {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)
            used_biome_ids.update(biome_grid_n.flatten())
            used_biome_ids.update(biome_grid_s.flatten())
        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")

    biome_colors = load_biome_colors(BIOMES_CSV_PATH, used_biome_ids)
    if not biome_colors:
        raise ValueError("No valid biome colors loaded from Biomes.csv")

    for biom_path in biom_files:
        print(f"Processing {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)
            image_n = create_biome_image(biome_grid_n, biome_colors)
            image_s = create_biome_image(biome_grid_s, biome_colors)
            image_n = upscale_image(image_n)
            image_s = upscale_image(image_s)
            planet_name = biom_path.stem
            image_n.save(PNG_OUTPUT_DIR / f"{planet_name}_North.png")
            image_s.save(PNG_OUTPUT_DIR / f"{planet_name}_South.png")
            print(f"Saved PNGs for {planet_name} (North and South)")
        except Exception as e:
            import traceback
            print(f"Error processing {biom_path.name}: {e}")
            traceback.print_exc()

    print("Processing complete.")
    sys.stdout.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()