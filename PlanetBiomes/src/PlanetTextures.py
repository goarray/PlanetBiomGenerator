#!/usr/bin/env python3
"""
Planet Textures Generator

Generates PNG and DDS texture images for planet biomes based on .biom files.
Outputs four maps per hemisphere: albedo (color), normal, rough, and alpha.
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
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter
import numpy as np
import colorsys
import argparse
import subprocess
import json
import csv
import sys
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
OUTPUT_DIR = BASE_DIR / "output"
TEXTURE_OUTPUT_DIR = OUTPUT_DIR / "textures"

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


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


# Initialize configuration
config = load_config(CONFIG_PATH)


def load_biome_colors(csv_path, used_biome_ids, saturate_factor=None):
    """Load RGB colors for used biome IDs from CSV."""
    if saturate_factor is None:
        saturate_factor = config.get("saturation_factor", 0.29)

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


def generate_elevation(shape, scale=None):
    """Generate elevation map with increased variation."""
    if scale is None:
        scale = config.get("noise_scale", 4.17)
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
    if elevation.max() - elevation.min() < 0.1:
        print("Warning: Elevation map has low variation, reducing scale")
        scale = min(scale, 2.0)
        smoothed = (
            gaussian_filter(base_noise, sigma=scale) * 0.5
            + gaussian_filter(detail_noise, sigma=scale / 2) * 0.3
            + gaussian_filter(fine_noise, sigma=scale / 4) * 0.2
        )
        elevation = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    return elevation


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
    """Generate anisotropic shading based terrain gradients."""
    if light_source_x is None:
        light_source_x = config.get("light_source_x", 0.5)
    if light_source_y is None:
        light_source_y = config.get("light_source_y", 0.5)
    grad_x = np.gradient(grid, axis=1)
    grad_y = np.gradient(grid, axis=0)
    shading = np.clip(grad_x * light_source_x + grad_y * light_source_y, -1, 1)
    return (shading - shading.min()) / (shading.max() - shading.min())


def generate_fractal_noise(
    shape, octaves=None, detail_smoothness=None, detail_strength_decay=None
):
    """Generate fractal noise for terrain complexity."""
    if octaves is None:
        octaves = config.get("fractal_octaves", 4.23)
    if detail_smoothness is None:
        detail_smoothness = config.get("detail_smoothness", 0.41)
    if detail_strength_decay is None:
        detail_strength_decay = config.get("detail_strength_decay", 0.67)
    base = np.random.rand(*shape)
    combined = np.zeros_like(base)
    for i in range(int(octaves)):
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
        blend_radius = config.get("edge_blend_radius", 0.53)

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
        saturate_factor = config.get("saturation_factor", 0.29)
    h, s, v = colorsys.rgb_to_hsv(*[c / 255.0 for c in rgb])
    s *= saturate_factor
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(c * 255) for c in (r, g, b))


def enhance_brightness(image, bright_factor=None):
    """Enhance image brightness."""
    if bright_factor is None:
        bright_factor = config.get("brightness_factor", 0.74)
    scaled_factor = bright_factor * 4
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(scaled_factor)


def generate_normal_map(elevation_map, strength=None):
    """Generate RGB normal map from elevation map with purple hue."""
    if strength is None:
        strength = config.get("normal_strength", 0.88)

    # Compute gradients
    grad_y, grad_x = np.gradient(elevation_map)

    # Calculate normal components
    normal_x = -grad_x * strength
    normal_y = -grad_y * strength
    normal_z = np.ones_like(normal_x)  # Z points outward

    # Normalize the normal vector
    magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x = np.divide(
        normal_x, magnitude, where=magnitude != 0, out=np.zeros_like(normal_x)
    )
    normal_y = np.divide(
        normal_y, magnitude, where=magnitude != 0, out=np.zeros_like(normal_y)
    )
    normal_z = np.divide(
        normal_z, magnitude, where=magnitude != 0, out=np.ones_like(normal_z)
    )

    # Map to RGB (0-255), centered at 128 for X and Y, 255 for flat Z
    normal_map = np.stack(
        (
            (normal_x + 1) * 127.5,  # Red: X
            (normal_y + 1) * 127.5,  # Green: Y
            normal_z * 255,  # Blue: Z
        ),
        axis=-1,
    ).astype(np.uint8)

    return Image.fromarray(normal_map, mode="RGB")


def generate_roughness_map(
    elevation_map, fractal_map, base_value=None, noise_scale=None
):
    """Generate roughness map based on elevation and fractal noise."""
    if base_value is None:
        base_value = 1.0 - config.get("roughness_base", 0.36)
    if noise_scale is None:
        noise_scale = config.get("roughness_noise_scale", 0.95)
    roughness = base_value * np.ones_like(elevation_map)
    roughness += noise_scale * fractal_map
    roughness = np.clip(roughness, 0, 1)
    roughness_map = (roughness * 255).astype(np.uint8)
    return Image.fromarray(roughness_map, mode="L")


def generate_alpha_map(elevation_map, base_value=None, noise_scale=None):
    """Generate alpha map for transparency."""
    if base_value is None:
        base_value = config.get("alpha_base", 1.0)
    if noise_scale is None:
        noise_scale = config.get("alpha_noise_scale", 0.05)
    alpha = base_value * np.ones_like(elevation_map)
    alpha += noise_scale * generate_noise(elevation_map.shape, scale=noise_scale * 100)
    alpha = np.clip(alpha, 0, 1)
    alpha_map = (alpha * 255).astype(np.uint8)
    return Image.fromarray(alpha_map, mode="L")


def create_biome_image(grid, biome_colors, default_color=(128, 128, 128)):
    """Generate biome image with albedo, normal, rough, and alpha maps."""
    process_images = config.get("process_images", False)

    if not process_images:
        # Direct 1-to-1 conversion without processing
        albedo = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                form_id = int(grid[y, x])
                albedo[y, x] = biome_colors.get(form_id, default_color)

        albedo_image = Image.fromarray(albedo)

        # Flat normal map (pointing straight up, RGB = 128, 128, 255)
        normal_map = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
        normal_map[..., 0] = 128  # X
        normal_map[..., 1] = 128  # Y
        normal_map[..., 2] = 255  # Z
        normal_image = Image.fromarray(normal_map, mode="RGB")

        # Constant roughness map
        base_roughness = 1.0 - config.get("roughness_base", 0.36)
        roughness_map = np.full(
            (GRID_SIZE[1], GRID_SIZE[0]), int(base_roughness * 255), dtype=np.uint8
        )
        rough_image = Image.fromarray(roughness_map, mode="L")

        # Constant alpha map
        base_alpha = config.get("alpha_base", 1.0)
        alpha_map = np.full(
            (GRID_SIZE[1], GRID_SIZE[0]), int(base_alpha * 255), dtype=np.uint8
        )
        alpha_image = Image.fromarray(alpha_map, mode="L")

        return {
            "albedo": albedo_image,
            "normal": normal_image,
            "rough": rough_image,
            "alpha": alpha_image,
        }

    # Original processing pipeline
    albedo = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
    noise_map = generate_noise(
        (GRID_SIZE[1], GRID_SIZE[0]), scale=config["noise_scale"]
    )
    elevation_map = generate_elevation((GRID_SIZE[1], GRID_SIZE[0]))
    edge_blend_map = generate_edge_blend(grid)
    shading_map = generate_shading(elevation_map)
    fractal_map = generate_fractal_noise((GRID_SIZE[1], GRID_SIZE[0]))
    crater_map = add_craters(elevation_map)
    crater_shading = generate_crater_shading(crater_map)
    bright_factor = config.get("brightness_factor", 0.74)

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
        "albedo": config.get("albedo_format", "BC7_UNORM"),
        "normal": config.get("normal_format", "BC7_UNORM"),
        "rough": config.get("rough_format", "BC4_UNORM"),
        "alpha": config.get("alpha_format", "BC4_UNORM"),
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


def main():
    """Process .biom files and generate PNG and DDS textures."""
    global _progress_started

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

    keep_pngs = config.get("keep_pngs_after_conversion", True)

    for biom_path in biom_files:
        print(f"Processing {biom_path.name}")
        try:
            biome_grid_n, biome_grid_s, plugin_name = load_biom_file(biom_path)
            maps_n = create_biome_image(biome_grid_n, biome_colors)
            maps_s = create_biome_image(biome_grid_s, biome_colors)

            # Upscale all maps
            maps_n = {k: upscale_image(v) for k, v in maps_n.items()}
            maps_s = {k: upscale_image(v) for k, v in maps_s.items()}

            once_per_run = False 
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

                    if not once_per_run:
                        print("Permits approved, site secured.")
                        once_per_run = True

                    # Skip DDS conversion in preview mode
                    if not args.preview:
                        texture_output_dir = TEXTURE_OUTPUT_DIR
                        texture_path = convert_png_to_dds(
                            png_path, texture_output_dir, plugin_name, texture_type
                        )
                        print(f"Converted DDS saved to: {texture_path}")

                        # Optionally delete PNG after DDS conversion
                        if not keep_pngs:
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

    print("Materials processing complete.")

    subprocess.run(["python", str(BASE_DIR / "src" / "PlanetMaterials.py")], check=True)
    sys.stdout.flush()
    sys.exit()


if __name__ == "__main__":
    main()
