from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter
import numpy as np
import csv
from PIL import Image

# Constants
GRID_SIZE = [256, 256]  # [0x100, 0x100]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = Path(__file__).parent.parent
BIOMES_CSV_PATH = SCRIPT_DIR / "Biomes.csv"
OUTPUT_DIR = SCRIPT_DIR
PNG_OUTPUT_DIR = OUTPUT_DIR / "BiomePNGs"

# Define .biom file structure (same as original script)
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


def load_biome_colors(csv_path):
    """Load Biomes.csv and return a dictionary mapping FormID to RGB tuple."""
    biome_colors = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                form_id = int(row[0], 16)  # Parse FormID from hex
                r, g, b = int(row[2]), int(row[3]), int(row[4])  # RGB values
                biome_colors[form_id] = (r, g, b)
            except (ValueError, IndexError):
                print(f"Warning: Invalid row in Biomes.csv: {row}. Skipping.")
    return biome_colors


def load_biom_file(filepath):
    """Load a .biom file and return biomeGridN and biomeGridS as numpy arrays."""
    with open(filepath, "rb") as f:
        data = CsSF_Biom.parse_stream(f)
    biome_grid_n = np.array(data.biomeGridN, dtype=np.uint32).reshape(
        GRID_SIZE[1], GRID_SIZE[0]
    )
    biome_grid_s = np.array(data.biomeGridS, dtype=np.uint32).reshape(
        GRID_SIZE[1], GRID_SIZE[0]
    )
    return biome_grid_n, biome_grid_s


def generate_noise(shape, scale=6): # Adjust scale : def 8
    """Create smooth noise for texture variation."""
    base_noise = np.random.rand(*shape)
    smoothed = gaussian_filter(base_noise, sigma=scale)  # Larger sigma for smoother blending
    return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min()) 

def generate_elevation(shape, scale=10): # Adjust scale : def 12
    """Generate elevation-based shading using smoothed noise."""
    base_noise = np.random.rand(*shape)
    smoothed = gaussian_filter(base_noise, sigma=scale)  # Larger sigma for broader elevation shifts
    elevation = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())  # Normalize [0,1]
    return elevation

def generate_atmospheric_fade(shape, intensity=1.2, spread=5): # Adjust intensity and spread : def 0.6 & 1.5
    """Creates an atmospheric gradient fading outward from the planet center."""
    center_y, center_x = shape[0] // 2, shape[1] // 2
    y_grid, x_grid = np.indices(shape)
    
    distance_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    fade_map = np.exp(-spread * (distance_from_center / max_distance)) * intensity
    return fade_map

def generate_shading(grid, light_source=(0.2, 0.8)):  # Light coming from top-right
    """Creates anisotropic shading based on terrain height differences."""
    grad_x = np.gradient(grid, axis=1)  # Horizontal shading
    grad_y = np.gradient(grid, axis=0)  # Vertical shading

    # Combine gradients using the light direction
    shading = np.clip(grad_x * light_source[0] + grad_y * light_source[1], -1, 1)

    return (shading - shading.min()) / (shading.max() - shading.min())  # Normalize [0,1]

def generate_fractal_noise(shape, octaves=4):
    """Creates fractal terrain noise using multiple layers of detail."""
    base = np.random.rand(*shape)
    combined = np.zeros_like(base)

    for i in range(octaves):
        sigma = 2 ** (i + 1)  # Higher octaves add finer details
        weight = 0.5 ** i  # Reduce strength for finer layers
        combined += gaussian_filter(base, sigma=sigma) * weight  # Layered smoothing

    return (combined - combined.min()) / (combined.max() - combined.min())  # Normalize [0,1]

def add_craters(grid, num_craters=30, max_radius=20):
    """Adds realistic impact craters by deepening terrain."""
    crater_map = np.zeros_like(grid, dtype=np.float32)

    for _ in range(num_craters):
        cx, cy = np.random.randint(0, GRID_SIZE[0]), np.random.randint(0, GRID_SIZE[1])
        radius = np.random.randint(5, max_radius)  # Larger minimum radius
        y_grid, x_grid = np.indices(grid.shape)
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        # Stronger impact effect
        crater_depth = np.exp(-dist / radius) * np.random.uniform(0.4, 0.8)  # Boost intensity
        crater_map -= crater_depth  # Lower terrain at impact sites

    return np.clip(grid + crater_map, 0, 1)  # Adjust grid to reflect crater depth

def generate_crater_shading(crater_map):
    """Creates crater rim shading for realistic depth."""
    shading = np.gradient(crater_map, axis=0) + np.gradient(crater_map, axis=1)
    return (shading - shading.min()) / (shading.max() - shading.min())

def generate_edge_blend(grid, blend_radius=10): # Adjust radius to increase blending : def 7
    """Creates an edge blending map by detecting transitions between biome IDs."""
    edge_map = np.zeros_like(grid, dtype=np.float32)

    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            current_biome = grid[y, x]

            # Check neighboring pixels to detect biome boundaries
            neighbors = [
                grid[max(y - 1, 0), x], grid[min(y + 1, GRID_SIZE[1] - 1), x],
                grid[y, max(x - 1, 0)], grid[y, min(x + 1, GRID_SIZE[0] - 1)]
            ]

            # If biome differs from neighbors, it's near an edge
            if any(neighbor != current_biome for neighbor in neighbors):
                edge_map[y, x] = 1.0  # Mark edge pixels

    # Apply Gaussian blur to smooth transition areas
    blended_edges = gaussian_filter(edge_map, sigma=blend_radius)
    return blended_edges

def create_biome_image(grid, biome_colors, default_color=(128, 128, 128)):
    """Create a PIL Image with biome colors, elevation shading, latitude shading, and edge blending."""
    image = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)

    noise_map = generate_noise((GRID_SIZE[1], GRID_SIZE[0]), scale=8)  # Fine texture
    elevation_map = generate_elevation((GRID_SIZE[1], GRID_SIZE[0]), scale=12)  # Height shading
    edge_blend_map = generate_edge_blend(grid, blend_radius=3)  # Smooth biome transitions
    shading_map = generate_shading(elevation_map, light_source=(0.3, 0.7))
    fractal_map = generate_fractal_noise((GRID_SIZE[1], GRID_SIZE[0]), octaves=4)  # **Now actually generating fractal detail**
    crater_map = add_craters(elevation_map, num_craters=30, max_radius=12)
    crater_shading = generate_crater_shading(crater_map)

    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            form_id = int(grid[y, x])
            biome_color = biome_colors.get(form_id, default_color)

            # **Latitude shading**
            lat_factor = abs((y / GRID_SIZE[1]) - 0.5) * 0.4

            # Apply **elevation shading** (lighter peaks, darker valleys)
            shaded_color = tuple(int(c * (0.8 + 0.2 * elevation_map[y, x])) for c in biome_color)
            
            # Apply anisotropic shading (directional light effect)
            light_adjusted_color = tuple(int(c * (0.9 + 0.1 * shading_map[y, x])) for c in shaded_color)

            # Apply **fractal terrain complexity**
            fractal_adjusted_color = tuple(int(c * (0.85 + 0.15 * fractal_map[y, x])) for c in light_adjusted_color)

            # Apply **Add craters**
            crater_adjusted_color = tuple(int(c * (0.7 + 0.3 * crater_shading[y, x])) for c in fractal_adjusted_color)

            # Apply **latitude-based adjustment**
            lat_adjusted_color = tuple(int(c * (1 - lat_factor)) for c in crater_adjusted_color)

            # Apply **edge blending effect**
            blended_color = tuple(int(c * (1 - 0.3 * edge_blend_map[y, x])) for c in lat_adjusted_color)

            # Apply **noise-based texture**
            final_color = tuple(int(c * (0.95 + 0.05 * noise_map[y, x])) for c in blended_color)

            image[y, x] = final_color

    return Image.fromarray(image)

def main():
    # Create output directory for PNGs
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load biome colors from Biomes.csv
    biome_colors = load_biome_colors(BIOMES_CSV_PATH)
    if not biome_colors:
        raise ValueError("No valid biome colors loaded from Biomes.csv")

    # Find all .biom files in the generated subdirectory
    biom_files = [
        f for f in OUTPUT_DIR.rglob("*.biom")
        if f.parent != OUTPUT_DIR  # Exclude files directly in the script directory
    ]

    if not biom_files:
        print("No .biom files found in the output directory.")
        return

    for biom_path in biom_files:
        print(f"Processing {biom_path.name}")
        try:
            # Load biome grids
            biome_grid_n, biome_grid_s = load_biom_file(biom_path)

            # Create images for both hemispheres
            image_n = create_biome_image(biome_grid_n, biome_colors)
            image_s = create_biome_image(biome_grid_s, biome_colors)

            # Save images
            planet_name = biom_path.stem
            image_n.save(PNG_OUTPUT_DIR / f"{planet_name}_North.png")
            image_s.save(PNG_OUTPUT_DIR / f"{planet_name}_South.png")
            print(f"Saved PNGs for {planet_name} (North and South)")

        except Exception as e:
            print(f"Error processing {biom_path.name}: {e}")


if __name__ == "__main__":
    main()
