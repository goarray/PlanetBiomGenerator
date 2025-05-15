#!/usr/bin/env python3
"""
Planet Textures Generator

Generates PNG and DDS texture images for planet biomes based on .biom files.
Outputs four maps per hemisphere: albedo (color), normal, rough, and alpha.
Applies effects like noise, elevation, shading, craters, and edge blending
to create realistic planetary visuals. Uses configuration from JSON and
biome colors from CSV. Converts PNGs to DDS format using texconv.exe for Starfield compatibility.

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
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter
import numpy as np
import colorsys
import argparse
import subprocess
import json
import csv
import sys
import os
from PIL import Image, ImageEnhance

# Directory paths
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS).resolve()
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()

SCRIPT_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
ASSETS_DIR = BASE_DIR / "assets"
CSV_DIR = BASE_DIR / "csv"
OUTPUT_DIR = BASE_DIR / "Output"
TEXTURE_OUTPUT_DIR = OUTPUT_DIR / "Textures"

# File paths
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
BIOMES_CSV_PATH = CSV_DIR / "Biomes.csv"
TEXCONV_PATH = BASE_DIR / "textconv" / "texconv.exe"

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

        # Set default values only when missing
        defaults = {
            "image_pipeline": {
                "noise_scale": 10.0,
                "elevation_scale": 5.0,
                "light_source_x": 1.0,
                "light_source_y": 1.0,
                "fade_intensity": 0.5,
                "fade_spread": 2.0,
                "saturation_factor": 1.2,
                "brightness_factor": 1.0,
                "upscale_image": False,
                "enable_edge_blending": True,
                "edge_blend_radius": 4,
                "fractal_octaves": 4,
                "detail_smoothness": 2,
                "detail_strength_decay": 0.5,
                "crater_max_radius": 20,
                "crater_depth_min": 0.2,
                "crater_depth_max": 0.8,
                "enable_crater_shading": True,
                "normal_strength": 1.0,
                "roughness_base": 0.5,
                "roughness_noise_scale": 0.1,
                "alpha_base": 1.0,
                "alpha_noise_scale": 0.05,
            },
            "dds_conversion": {
                "delete_pngs_after_conversion": False,
                "albedo_format": "BC7_UNORM",
                "normal_format": "BC7_UNORM",
                "rough_format": "BC4_UNORM",
                "alpha_format": "BC4_UNORM"
            },
        }

        # Only update missing keys, preserving user-defined values
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    config[key].setdefault(sub_key, sub_value)

    except FileNotFoundError:
        print(f"Error: Config file {CONFIG_PATH} not found.")
        config = defaults


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
    """Load .biom file and return biome grids as numpy arrays, plus plugin name."""
    biom_path = Path(filepath)
    plugin_name = biom_path.parent.name

    with open(filepath, "rb") as f:
        data = CsSF_Biom.parse_stream(f)

    biome_grid_n = np.array(data.biomeGridN, dtype=np.uint32).reshape(
        GRID_SIZE[1], GRID_SIZE[0]
    )
    biome_grid_s = np.array(data.biomeGridS, dtype=np.uint32).reshape(
        GRID_SIZE[1], GRID_SIZE[0]
    )

    return biome_grid_n, biome_grid_s, plugin_name


def upscale_image(image, target_size=(1024, 1024)):
    """Upscale image to target size if enabled in config."""
    if config["image_pipeline"].get("upscale_image"):
        return image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def generate_noise(shape, scale=None):
    """Generate larger-patch high-contrast salt-and-pepper noise."""
    if scale is None:
        scale = config["image_pipeline"].get("noise_scale")

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


def generate_elevation(shape, scale=None):
    """Generate elevation map with increased variation."""
    if scale is None:
        scale = config["image_pipeline"].get("elevation_scale", 5.0)
    base_noise = np.random.rand(*shape)
    detail_noise = np.random.rand(*shape)
    fine_noise = np.random.rand(*shape)
    # Combine noise layers with different scales
    smoothed = (
        gaussian_filter(base_noise, sigma=scale) * 0.5
        + gaussian_filter(detail_noise, sigma=scale / 2) * 0.3
        + gaussian_filter(fine_noise, sigma=scale / 4) * 0.2
    )
    elevation = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    print(
        f"Elevation map - min: {elevation.min():.4f}, max: {elevation.max():.4f}, range: {elevation.max() - elevation.min():.4f}"
    )
    if elevation.max() - elevation.min() < 0.1:
        print("Warning: Elevation map has low variation, reducing scale")
        scale = min(scale, 2.0)
        smoothed = (
            gaussian_filter(base_noise, sigma=scale) * 0.5
            + gaussian_filter(detail_noise, sigma=scale / 2) * 0.3
            + gaussian_filter(fine_noise, sigma=scale / 4) * 0.2
        )
        elevation = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        print(
            f"Adjusted elevation map - min: {elevation.min():.4f}, max: {elevation.max():.4f}"
        )
    return elevation


def generate_atmospheric_fade(shape, intensity=None, spread=None):
    """Generate atmospheric fade effect from planet center."""
    if intensity is None:
        intensity = config["image_pipeline"].get("fade_intensity")
    if spread is None:
        spread = config["image_pipeline"].get("fade_spread")
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y_grid, x_grid = np.indices(shape)
    distance_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    return np.exp(-spread * (distance_from_center / max_distance)) * intensity


def generate_shading(grid, light_source_x=None, light_source_y=None):
    """Generate anisotropic shading based terrain gradients."""
    if light_source_x is None:
        light_source_x = config["image_pipeline"].get("light_source_x")
    if light_source_y is None:
        light_source_y = config["image_pipeline"].get("light_source_y")
    grad_x = np.gradient(grid, axis=1)
    grad_y = np.gradient(grid, axis=0)
    shading = np.clip(grad_x * light_source_x + grad_y * light_source_y, -1, 1)
    return (shading - shading.min()) / (shading.max() - shading.min())


def generate_fractal_noise(
    shape, octaves=None, detail_smoothness=None, detail_strength_decay=None
):
    """Generate fractal noise for terrain complexity."""
    if octaves is None:
        octaves = config["image_pipeline"].get("fractal_octaves", 4)
    if detail_smoothness is None:
        detail_smoothness = config["image_pipeline"].get("detail_smoothness", 2)
    if detail_strength_decay is None:
        detail_strength_decay = config["image_pipeline"].get(
            "detail_strength_decay", 0.5
        )
    base = np.random.rand(*shape)
    combined = np.zeros_like(base)
    for i in range(octaves):
        sigma = max(1, detail_smoothness ** (i * 0.3))
        weight = detail_strength_decay ** (i * 1.5)
        combined += gaussian_filter(base, sigma=sigma) * weight

    combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = np.power(combined, 2)
    return (combined - combined.min()) / (combined.max() - combined.min())


def add_craters(
    grid, num_craters=100, max_radius=None, crater_depth_min=None, crater_depth_max=None
):
    """Add small impact craters to terrain grid."""
    if max_radius is None:
        max_radius = config["image_pipeline"].get("crater_max_radius", 20)
    if crater_depth_min is None:
        crater_depth_min = config["image_pipeline"].get("crater_depth_min", 0.2)
    if crater_depth_max is None:
        crater_depth_max = config["image_pipeline"].get("crater_depth_max", 0.8)

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
    if config["image_pipeline"].get("enable_crater_shading"):
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
    if not config["image_pipeline"].get("enable_edge_blending", True):
        return np.zeros_like(grid, dtype=np.float32)

    if blend_radius is None:
        blend_radius = config["image_pipeline"].get("edge_blend_radius", 4)

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


def desaturate_color(rgb, saturate_factor=1.2):
    """Adjust color saturation in HSV space."""
    if saturate_factor is None:
        saturate_factor = config["image_pipeline"].get("saturation_factor", 1.0)
    h, s, v = colorsys.rgb_to_hsv(*[c / 255.0 for c in rgb])
    s *= saturate_factor
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in (r, g, b))


def enhance_brightness(image, bright_factor=None):
    """Enhance image brightness."""
    if bright_factor is None:
        bright_factor = config["image_pipeline"].get("brightness_factor", 1.0)
    scaled_factor = bright_factor * 4
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(scaled_factor)


def generate_normal_map(elevation_map, strength=None):
    """Generate RGB normal map from elevation map with purple hue."""
    if strength is None:
        strength = config["image_pipeline"].get("normal_strength", 1.0)

    # Compute gradients
    grad_y, grad_x = np.gradient(elevation_map)
    print(f"Gradient X - min: {grad_x.min():.4f}, max: {grad_x.max():.4f}")
    print(f"Gradient Y - min: {grad_y.min():.4f}, max: {grad_y.max():.4f}")

    # Calculate normal components
    normal_x = -grad_x * strength
    normal_y = -grad_y * strength
    normal_z = np.ones_like(normal_x)  # Z points outward

    # Normalize the normal vector
    magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    print(f"Magnitude - min: {magnitude.min():.4f}, max: {magnitude.max():.4f}")
    normal_x = np.divide(
        normal_x, magnitude, where=magnitude != 0, out=np.zeros_like(normal_x)
    )
    normal_y = np.divide(
        normal_y, magnitude, where=magnitude != 0, out=np.zeros_like(normal_y)
    )
    normal_z = np.divide(
        normal_z, magnitude, where=magnitude != 0, out=np.ones_like(normal_z)
    )
    print(f"Normal X - min: {normal_x.min():.4f}, max: {normal_x.max():.4f}")
    print(f"Normal Y - min: {normal_y.min():.4f}, max: {normal_y.max():.4f}")
    print(f"Normal Z - min: {normal_z.min():.4f}, max: {normal_z.max():.4f}")

    # Map to RGB (0-255), centered at 128 for X and Y, 255 for flat Z
    normal_map = np.stack(
        (
            (normal_x + 1) * 127.5,  # Red: X
            (normal_y + 1) * 127.5,  # Green: Y
            normal_z * 255,  # Blue: Z
        ),
        axis=-1,
    ).astype(np.uint8)

    print(
        f"Normal map RGB - min: {normal_map.min(axis=(0,1))}, max: {normal_map.max(axis=(0,1))}"
    )
    return Image.fromarray(normal_map, mode="RGB")


def generate_roughness_map(
    elevation_map, fractal_map, base_value=None, noise_scale=None
):
    """Generate roughness map based on elevation and fractal noise."""
    if base_value is None:
        base_value = 1.0 - config["image_pipeline"].get("roughness_base", 0.5)
    if noise_scale is None:
        noise_scale = config["image_pipeline"].get("roughness_noise_scale", 0.1)
    roughness = base_value * np.ones_like(elevation_map)
    roughness += noise_scale * fractal_map
    roughness = np.clip(roughness, 0, 1)
    roughness_map = (roughness * 255).astype(np.uint8)
    return Image.fromarray(roughness_map, mode="L")


def generate_alpha_map(elevation_map, base_value=None, noise_scale=None):
    """Generate alpha map for transparency."""
    if base_value is None:
        base_value = config["image_pipeline"].get("alpha_base", 1.0)
    if noise_scale is None:
        noise_scale = config["image_pipeline"].get("alpha_noise_scale", 0.05)
    alpha = base_value * np.ones_like(elevation_map)
    alpha += noise_scale * generate_noise(elevation_map.shape, scale=noise_scale * 100)
    alpha = np.clip(alpha, 0, 1)
    alpha_map = (alpha * 255).astype(np.uint8)
    return Image.fromarray(alpha_map, mode="L")


def create_biome_image(grid, biome_colors, default_color=(128, 128, 128)):
    """Generate biome image with albedo, normal, rough, and alpha maps."""
    # Initialize maps
    albedo = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
    noise_map = generate_noise(
        (GRID_SIZE[1], GRID_SIZE[0]), scale=config["image_pipeline"]["noise_scale"]
    )
    elevation_map = generate_elevation((GRID_SIZE[1], GRID_SIZE[0]))
    edge_blend_map = generate_edge_blend(grid)
    shading_map = generate_shading(elevation_map)
    fractal_map = generate_fractal_noise((GRID_SIZE[1], GRID_SIZE[0]))
    crater_map = add_craters(elevation_map)
    crater_shading = generate_crater_shading(crater_map)
    bright_factor = config["image_pipeline"].get("brightness_factor", 1.0)

    # Generate albedo map
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            form_id = int(grid[y, x])
            biome_color = biome_colors.get(form_id, default_color)
            lat_factor = abs((y / GRID_SIZE[1]) - 0.5) * 0.4
            shaded_color = tuple(
                int(c * (0.8 + 0.2 * elevation_map[y, x])) for c in biome_color
            )
            light_adjusted_color = tuple(
                int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color
            )
            fractal_adjusted_color = tuple(
                int(c * (0.85 + 0.15 * fractal_map[y, x])) for c in light_adjusted_color
            )
            crater_adjusted_color = tuple(
                int(c * (0.7 + 0.3 * crater_shading[y, x]))
                for c in fractal_adjusted_color
            )
            lat_adjusted_color = tuple(
                int(c * (1 - lat_factor)) for c in crater_adjusted_color
            )
            blended_color = tuple(
                int(c * (1 - 0.5 * edge_blend_map[y, x])) for c in lat_adjusted_color
            )
            final_color = tuple(
                np.clip(int(c * (0.91 + 0.09 * noise_map[y, x])), 0, 255)
                for c in blended_color
            )
            albedo[y, x] = final_color

    albedo_image = Image.fromarray(albedo)
    albedo_image = enhance_brightness(albedo_image, bright_factor)

    # Generate normal, rough, and alpha maps
    normal_image = generate_normal_map(elevation_map)
    rough_image = generate_roughness_map(elevation_map, fractal_map)
    alpha_image = generate_alpha_map(elevation_map)

    return {
        "albedo": albedo_image,
        "normal": normal_image,
        "rough": rough_image,
        "alpha": alpha_image,
    }


def convert_png_to_dds(png_path, texture_output_dir, plugin_name, texture_type):
    """Convert a PNG file to DDS using texconv.exe with Starfield-compatible formats."""
    if not TEXCONV_PATH.exists():
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}")

    # Ensure output directory exists
    texture_output_dir = texture_output_dir / plugin_name
    texture_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine DDS format based on texture type
    format_map = {
        "albedo": config["dds_conversion"].get("albedo_format", "BC7_UNORM"),
        "normal": config["dds_conversion"].get("normal_format", "BC7_UNORM"),
        "rough": config["dds_conversion"].get("rough_format", "BC4_UNORM"),
        "alpha": config["dds_conversion"].get("alpha_format", "BC4_UNORM"),
    }
    dds_format = format_map.get(texture_type, "BC7_UNORM")

    # Construct output DDS path
    texture_filename = png_path.stem + ".dds"
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
        print(f"Converted {png_path.name} to {texture_path.name} ({dds_format})")
        return texture_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {png_path.name} to DDS: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: texconv.exe not found at {TEXCONV_PATH}")
        raise


processing_widget_process = None


def start_processing_widget(title):
    """Launch processing indicator and store process handle."""
    global processing_widget_process
    script_path = os.path.join(os.path.dirname(__file__), "processing_widget.py")

    if not os.path.exists(script_path):
        print(f"Error: {script_path} does not exist!")
        return

    processing_widget_process = subprocess.Popen(["python", script_path, title])


def stop_processing_widget():
    """Terminate processing widget when script completes."""
    global processing_widget_process
    if processing_widget_process:
        processing_widget_process.terminate()


_progress_started = False


def main():
    """Process .biom files and generate PNG and DDS textures."""
    global _progress_started
    if not _progress_started:
        _progress_started = True
        start_processing_widget("Processing Planet Textures")

    parser = argparse.ArgumentParser(
        description="Generate PNG and DDS textures from .biom files"
    )
    parser.add_argument(
        "biom_file", nargs="?", help="Path to the .biom file (for preview mode)"
    )
    parser.add_argument("--preview", action="store_true", help="Run in preview mode")
    args = parser.parse_args()

    TEXTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.preview and args.biom_file:
        biom_files = [Path(args.biom_file)]
        if not biom_files[0].exists():
            print(f"Error: Provided .biom file not found: {args.biom_file}")
            sys.exit(1)
    else:
        biom_files = [
            f
            for f in OUTPUT_DIR.rglob("*.biom")
            if f.parent != OUTPUT_DIR and "assets" not in str(f.parent)
        ]
        if not biom_files:
            print("No .biom files found in the output directory.")
            sys.exit(1)

    used_biome_ids = set()
    for biom_path in biom_files:
        plugin_name = biom_path.parent.name
        print(f"Collecting biome IDs from {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s, plugin_name = load_biom_file(biom_path)
            used_biome_ids.update(biome_grid_n.flatten())
            used_biome_ids.update(biome_grid_s.flatten())
        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")

    biome_colors = load_biome_colors(BIOMES_CSV_PATH, used_biome_ids)
    if not biome_colors:
        raise ValueError("No valid biome colors loaded from Biomes.csv")

    delete_pngs = config["dds_conversion"].get("delete_pngs_after_conversion", False)

    for biom_path in biom_files:
        print(f"Processing {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s, plugin_name = load_biom_file(biom_path)
            maps_n = create_biome_image(biome_grid_n, biome_colors)
            maps_s = create_biome_image(biome_grid_s, biome_colors)

            # Upscale all maps
            maps_n = {k: upscale_image(v) for k, v in maps_n.items()}
            maps_s = {k: upscale_image(v) for k, v in maps_s.items()}

            planet_name = biom_path.stem

            # Save PNGs and convert to DDS
            texture_types = ["albedo", "normal", "rough", "alpha"]
            for hemisphere, maps in [("North", maps_n), ("South", maps_s)]:
                for texture_type in texture_types:
                    # Save PNG
                    png_filename = f"{planet_name}_{hemisphere}_{texture_type}.png"
                    png_path = TEXTURE_OUTPUT_DIR / png_filename
                    maps[texture_type].save(png_path)
                    print(f"Saved PNG: {png_path}")

                    # Convert to DDS
                    texture_output_dir = TEXTURE_OUTPUT_DIR
                    texture_path = convert_png_to_dds(
                        png_path, texture_output_dir, plugin_name, texture_type
                    )

                    print(f"Converted DDS saved to: {texture_path}")

                    # Optionally delete PNG
                    if delete_pngs:
                        try:
                            png_path.unlink()
                            print(f"Deleted intermediate PNG: {png_path}")
                        except OSError as e:
                            print(f"Error deleting {png_path}: {e}")

            print(
                f"Generated textures for {planet_name} (North and South: albedo, normal, rough, alpha)"
            )
        except Exception as e:
            import traceback

            print(f"Error processing {biom_path.name}: {e}")
            traceback.print_exc()

    print("Processing complete.")
    stop_processing_widget()
    sys.stdout.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()
