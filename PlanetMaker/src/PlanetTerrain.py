import numpy as np
from PIL import Image
import json
import noise
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter, sobel
from PlanetNewsfeed import handle_news
from PlanetConstants import get_config, CONFIG_PATH

RIDGE_OCTAVES = 5
RIDGE_SCALE = 5
RIDGE_STRENGTH = 20.0
RIDGE_THRESHOLD = 0.55
INVERT_ELEVATION = False
FALLOFF_SCALE = 10.0
FLOOR_MASK = 0.4
RIDGE_SHARPNESS = 0.5


# Global configuration
config = get_config()

def ridged_noise(x, y, scale, octaves=RIDGE_OCTAVES):
    n = noise.pnoise2(x / scale, y / scale, octaves=octaves)
    return 1.0 - abs(fbm(n,n))

def fbm(x, y, octaves=2, persistence=0.7, lacunarity=0.55):
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        value += amplitude * noise.pnoise2(x * frequency, y * frequency)
        frequency *= lacunarity
        amplitude *= persistence
    return value

def soft_noise_mask(width, height, scale=300.0, strength=0.9, power=0.5):
    mask = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            nx = x / scale
            ny = y / scale
            val = fbm(nx, ny, octaves=5, persistence=0.5, lacunarity=2.0)
            val = (val + 1) / 2  # Normalize
            val = val ** power
            mask[y, x] = val * strength

    return mask


def generate_normal_map(height_img, invert_height=True):
    handle_news(None)
    strength = (config.get("texture_roughness", 0.5) * 0.2)
    height = np.asarray(height_img).astype(np.float32) / 255.0
    if invert_height:
        height = 1.0 - height
    dx = sobel(height, axis=1) * strength
    dy = sobel(height, axis=0) * strength
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


def generate_terrain_normal(
    river_mask_path: str,
    terrain_image_path: str,
    mountain_mask_path: str,
    output_path: str = "river_normal.png",
):
    """Generates a river-based normal map using river mask (and optional terrain + mountain masks)."""

    # Load river mask as height source
    river_mask = Image.open(river_mask_path).convert("L")
    river_height = np.asarray(river_mask).astype(np.float32) / 255.0

    # Optionally soften river pattern
    river_height = gaussian_filter(river_height, sigma=0.1)

    # Optional: blend with terrain to adjust base elevation
    if terrain_image_path:
        terrain_img = Image.open(terrain_image_path).convert("L")
        terrain = np.asarray(terrain_img).astype(np.float32) / 255.0
        river_height = (
            river_height * 1.0 + terrain * 0.2
        )  # blend 80% river + 20% terrain

    # Optional: dampen effect in mountain zones
    if mountain_mask_path:
        mountain_mask = Image.open(mountain_mask_path).convert("L")
        mountain_mask = np.asarray(mountain_mask).astype(np.float32) / 255.0
        mountain_mask = gaussian_filter(mountain_mask, sigma=1)
        river_height *= (
            1.0 - 0.5 * mountain_mask
        )  # reduce intensity in mountainous areas

    # Optional: sharpen river cuts
    river_height += laplace(river_height) * 0.25

    # Normalize height to [0, 1]
    river_height -= river_height.min()
    river_height /= river_height.max() + 1e-8

    # Convert to grayscale image
    river_img = Image.fromarray((river_height * 255).astype(np.uint8), mode="L")

    # Generate normal map
    normal_img = generate_normal_map(river_img)
    normal_img.save(output_path)

    handle_news(None, "info", f"Saved river-based normal map to {output_path}")
