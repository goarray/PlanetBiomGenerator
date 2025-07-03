import numpy as np
from PIL import Image
import json
import noise
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter, sobel
from PlanetNewsfeed import handle_news
from PlanetConstants import get_config, CONFIG_PATH, PNG_OUTPUT_DIR

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


def generate_terrain_normal(output_path: str = "terrain_normal.png"):
    """Generates a terrain-based normal map using internal config and standard mask paths."""

    # Load config to get planet and plugin names
    config = get_config()
    plugin_name = config.get("plugin_name")
    planet_name = config.get("planet_name")

    # Construct input paths based on naming convention
    base = PNG_OUTPUT_DIR / plugin_name / planet_name
    paths = {
        "road_mask": base / f"{planet_name}_road_mask.png",
        "mountain_mask": base / f"{planet_name}_mountain_mask.png",
        "terrain": base / f"{planet_name}_terrain.png",
        "river_mask": base / f"{planet_name}_river_mask.png",
    }

    # Load and normalize masks
    def load_grayscale(path):
        return np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0

    river = gaussian_filter(load_grayscale(paths["river_mask"]), sigma=0.1)
    terrain = gaussian_filter(load_grayscale(paths["terrain"]), sigma=0.1)
    mountain = gaussian_filter(load_grayscale(paths["mountain_mask"]), sigma=0.2)

    # Blend inputs into heightmap
    river = river * 1.0 + terrain * 0.2
    river *= 1.0 - 0.5 * mountain
    river += laplace(river) * 0.25

    # Normalize
    river -= river.min()
    river /= river.max() + 1e-8
    river_inverted = 1.0 - river

    bump = mountain * 0.7 + terrain * 0.3 + river_inverted * 0.4
    bump -= bump.min()
    bump /= bump.max() + 1e-8

    # Apply road flattening
    road = gaussian_filter(load_grayscale(paths["road_mask"]), sigma=2.0)
    flatten_strength = 0.6
    bump = bump * (1 - road * flatten_strength) + (0.5 * road * flatten_strength)

    bump += laplace(bump) * 0.2
    bump -= bump.min()
    bump /= bump.max() + 1e-8

    inverted_bump = 1.0 - bump
    bump_img = Image.fromarray((inverted_bump * 255).astype(np.uint8), mode="L")
    normal_img = generate_normal_map(bump_img)
    normal_img.save(output_path)

    handle_news(None, "info", f"Saved terrain normal to {output_path}")
