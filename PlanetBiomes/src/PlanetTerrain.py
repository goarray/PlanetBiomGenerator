import numpy as np
from PIL import Image
import json
import noise
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter, sobel
from PlanetNewsfeed import handle_news
from PlanetConstants import CONFIG_PATH

RIDGE_OCTAVES = 5
RIDGE_SCALE = 5
RIDGE_STRENGTH = 20.0
RIDGE_THRESHOLD = 0.55
INVERT_ELEVATION = False
FALLOFF_SCALE = 10.0
FLOOR_MASK = 0.4
RIDGE_SHARPNESS = 0.5


# Global configuration
def load_config():
    """Load plugin_name from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


config = load_config()

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


def generate_terrain(width, height, scale=RIDGE_SCALE, strength=RIDGE_STRENGTH, invert=INVERT_ELEVATION):
    handle_news(None)
    terrain = np.zeros((height, width))
    ridge_mask = np.zeros((height, width))
    falloff_mask = np.zeros((height, width))

    threshold = RIDGE_THRESHOLD
    ridge_noise = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            r = ridged_noise(x, y, scale)
            ridge_noise[y, x] = r
            if r < threshold:
                r = 0.0
            ridge_mask[y, x] = r

            f = noise.pnoise2(x / FALLOFF_SCALE, y / FALLOFF_SCALE, octaves=RIDGE_OCTAVES)
            falloff_mask[y, x] = (f + 1) / 2

    ridge_mask *= falloff_mask

    # Optional noise erosion mask to knock down some ridge areas
    erosion_mask = soft_noise_mask(width, height, scale=50.0, strength=0.95)
    Image.fromarray((erosion_mask * 255).astype(np.uint8)).save("erosion_mask.png")
    ridge_mask *= erosion_mask

    ridge_mask = np.where(ridge_mask < FLOOR_MASK, 0.0, ridge_mask)
    ridge_mask = gaussian_filter(ridge_mask, sigma=0.01)

    for y in range(height):
        for x in range(width):
            elevation = ridged_noise(x, y, scale, octaves=RIDGE_OCTAVES)
            elevation = np.sign(elevation) * (abs(elevation) ** RIDGE_SHARPNESS)
            terrain[y, x] = elevation * ridge_mask[y, x] * strength
            if invert:
                terrain[y, x] *= -1

    terrain += laplace(terrain) * 0.05

    return terrain


def generate_terrain_normal(
    ocean_mask_path: str, output_path: str = "terrain_normal.png"
):
    """Generates a terrain-based normal map, suppressing ocean areas."""

    ocean_mask_img = Image.open(ocean_mask_path).convert("L")
    ocean_mask = np.asarray(ocean_mask_img).astype(np.float32) / 255.0
    height, width = ocean_mask.shape

    # Step 1: Generate full terrain
    terrain = generate_terrain(width, height)

    # Soften terrain in the ocean
    land_mask = ocean_mask * 0.6 + 0.4

    # Step 3: Optional: Smooth land/ocean transitions
    land_mask = gaussian_filter(land_mask, sigma=2)

    # Step 4: Apply land mask to terrain
    terrain *= land_mask

    # Step 5: Optional: Post-process terrain
    terrain += laplace(terrain) * 0.2

    # Step 6: Normalize to [0,1] for normal map
    terrain -= terrain.min()
    terrain /= terrain.max() + 1e-8
    terrain_img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")

    # Step 7: Generate and save normal map
    normal_img = generate_normal_map(terrain_img)
    normal_img.save(output_path)
    handle_news(None, "info", f"Saved terrain-based normal map to {output_path}")
