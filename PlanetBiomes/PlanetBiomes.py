from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
from scipy.ndimage import gaussian_filter
import numpy as np
import csv

# Constants
GRID_SIZE = [0x100, 0x100]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]
SCRIPT_DIR = Path(__file__).parent

# Update the base directory to the root directory
BASE_DIR = Path(__file__).parent.parent  # Get the parent of the 'src' folder

# Path adjustments based on new directory structure
TEMPLATE_PATH = SCRIPT_DIR / "PlanetBiomes.biom"  # PlanetBiomes.biom
CSV_PATH = SCRIPT_DIR / "xEditOutput" / "PlanetBiomes.csv"  # PlanetBiomes.csv
OUTPUT_DIR = SCRIPT_DIR

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

def load_planet_biomes(csv_path):
    """Load PlanetBiomes.csv and return (plugin_name, planet_to_biomes dict)."""
    planet_biomes = {}
    with open(csv_path, newline="") as csvfile:
        # Read the first line separately as plugin name
        first_line = csvfile.readline().strip()
        plugin_name = first_line.strip()  # Use the first line as plugin name
        
        if not plugin_name:
            raise ValueError("Plugin name is missing or empty.")
        
        # Ensure no extra commas or whitespaces in the plugin name
        plugin_name = plugin_name.rstrip(",")  # Strip any trailing commas
        
        # Now, parse the remaining CSV data for planet-biome mappings
        reader = csv.DictReader(csvfile, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"])
        next(reader, None)  # Skip the CSV header row
        for row in reader:
            planet = row["PlanetName"].strip()  # Strip any leading/trailing spaces
            if not planet:
                continue  # Skip empty rows or rows with no planet name

            try:
                form_id = int(row["BIOM_FormID"], 16)  # Parse the FormID
                planet_biomes.setdefault(planet, []).append(form_id)
            except ValueError:
                print(f"Warning: Invalid FormID '{row['BIOM_FormID']}' for planet '{planet}'. Skipping.")

    return plugin_name, planet_biomes

class BiomFile:
    def __init__(self):
        self.biomeIds = []
        self.biomeGridN = []
        self.resrcGridN = []
        self.biomeGridS = []
        self.resrcGridS = []

    def load(self, filename):
        with open(filename, "rb") as f:
            data = CsSF_Biom.parse_stream(f)
        self.biomeIds = list(data.biomeIds)
        self.biomeGridN = np.array(data.biomeGridN)
        self.resrcGridN = np.array(data.resrcGridN)
        self.biomeGridS = np.array(data.biomeGridS)
        self.resrcGridS = np.array(data.resrcGridS)

    def overwrite_biome_ids(self, new_biome_ids):
        if not self.biomeIds:
            raise ValueError("No biome IDs found in file.")
        
        if not new_biome_ids or len(new_biome_ids) < 1:
            raise ValueError("At least one biome ID is required.")
        
        if len(new_biome_ids) == 1:
            print(f"Warning: Only one biome ID provided. Filling entire grid with {new_biome_ids[0]}.")

            # **Ensure the biome grids are properly updated**
            self.biomeIds = [new_biome_ids[0]]  # Keep only this one biome ID
            self.biomeGridN = np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
            self.biomeGridS = np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
            return  # **Return immediately to skip complex assignment logic**
        
        elif len(new_biome_ids) > 7:
            print(f"##### Warning: {len(new_biome_ids)} biomes provided, but max is 7. Truncating to first 7. #####")
            new_biome_ids = new_biome_ids[:7]
        
        num_biomes = len(new_biome_ids)
        zones = np.linspace(0, GRID_SIZE[1], num_biomes + 1, dtype=int)  # Latitude zones

        # Create zone-indexed maps
        def generate_zone_map(shape):
            base = np.random.rand(*shape)

            # Add structured noise layers with increased smoothing
            large = gaussian_filter(base, sigma=16)  # Larger sigma for smoother landmasses
            medium = gaussian_filter(np.random.rand(*shape), sigma=6)  # Smoother peninsulas
            small = gaussian_filter(np.random.rand(*shape), sigma=2)  # Reduced fine texture

            # Blend with emphasis on larger features
            combined = 0.6 * large + 0.3 * medium + 0.1 * small

            # Apply non-linear transformation to enhance contrast and merge small features
            combined = np.power(combined, 1.5)  # Power function to boost larger features

            # Normalize to 0–1
            combined = (combined - combined.min()) / (combined.max() - combined.min())
            return combined

        def assign_biomes(smoothed_noise, hemisphere):
            # Special case: If only one biome, create a uniform grid and return immediately
            if len(new_biome_ids) == 1:
                print(f"Single-biome planet detected. Filling grid with biome ID {new_biome_ids[0]}")
                return np.full(GRID_FLATSIZE, new_biome_ids[0], dtype=np.uint32)
    
            grid = np.zeros(GRID_FLATSIZE, dtype=np.uint32)
            center_y, center_x = GRID_SIZE[1] // 2, GRID_SIZE[0] // 2
            n = 4  # Superellipse exponent for squircle shape (n=4 approximates a squircle)

            # Generate distortion ONCE per hemisphere
            distortion = gaussian_filter(np.random.rand(GRID_SIZE[1], GRID_SIZE[0]), sigma=24)
            distortion = (distortion - distortion.min()) / (distortion.max() - distortion.min())  # Normalize to [0,1]

            # Define drag centers near equator and poles
            equator_drag_centers = [
                (center_x + np.random.randint(-40, 40), center_y + np.random.randint(-20, 20))
                for _ in range(3)
            ]
            pole_drag_centers = [
                (center_x + np.random.randint(-50, 50), center_y + np.random.randint(-100, -50))
                for _ in range(2)
            ]
            drag_radius = 18  # Increased drag influence

            for y in range(GRID_SIZE[1]):
                for x in range(GRID_SIZE[0]):
                    i = y * GRID_SIZE[0] + x
                    dx = (x - center_x) / (GRID_SIZE[0] / 2)  # Normalize to [-1, 1]
                    dy = (y - center_y) / (GRID_SIZE[1] / 2)  # Normalize to [-1, 1]

                    # Superellipse distance for squircle gradient
                    r = (abs(dx) ** n + abs(dy) ** n) ** (1 / n)
                    r = min(r, 1.0)  # Clamp to [0, 1]

                    # Adjust the weights for pole vs. equator biome
                    if hemisphere == "N":
                        lat_factor = r + 0.12 * (distortion[y, x] - 0.5)
                    else:
                        lat_factor = r + 0.08 * (distortion[y, x] - 0.5)

                    lat_factor = np.clip(lat_factor, 0, 1)

                    noise = smoothed_noise[y, x]
                    combined = 0.6 * noise + 0.6 * lat_factor

                    # Reverse biome order so first in CSV / top in xEdit or PNDT is equator, last is poles
                    reversed_biome_ids = list(reversed(new_biome_ids))

                    biome_index = int(combined * len(reversed_biome_ids))
                    biome_index = min(biome_index, len(reversed_biome_ids) - 1)

                    # Equator biome: stronger pull inland
                    if biome_index == 0:
                        for cx, cy in equator_drag_centers:
                            ddx = x - cx
                            ddy = y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2  # Stronger pull
                                combined -= weight * 0.84  # Increase dragging strength
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    # Pole biome: stronger pull inland
                    elif biome_index == len(reversed_biome_ids) - 1:
                        for cx, cy in pole_drag_centers:
                            ddx = x - cx
                            ddy = y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2  # Stronger pull
                                combined += weight * 0.59  # Increase dragging strength
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    # 2nd biome: allow intrusion from equator biome
                    elif biome_index == 1 and len(reversed_biome_ids) > 1:
                        for cx, cy in equator_drag_centers:
                            ddx = x - cx
                            ddy = y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2
                                combined -= weight * 0.32  # Allow intrusion without sharp boundaries
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    # 2nd-to-last biome: allow intrusion from pole biome
                    elif biome_index == len(reversed_biome_ids) - 2 and len(reversed_biome_ids) > 2:
                        for cx, cy in pole_drag_centers:
                            ddx = x - cx
                            ddy = y - cy
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < drag_radius:
                                weight = (1 - dist / drag_radius) ** 2
                                combined += weight * 0.55  # Allow intrusion without sharp boundaries
                        combined = np.clip(combined, 0, 1)
                        biome_index = min(int(combined * len(reversed_biome_ids)), len(reversed_biome_ids) - 1)

                    grid[i] = reversed_biome_ids[biome_index]

            return grid



        shape = (GRID_SIZE[1], GRID_SIZE[0])
        noise_n = generate_zone_map(shape)
        noise_s = generate_zone_map(shape[::-1])  # Flip for southern hemisphere

        self.biomeGridN = assign_biomes(noise_n, "N")
        self.biomeGridS = assign_biomes(noise_s, "S")
        self.biomeIds = list(set(new_biome_ids))  # Trim to used set

    def save(self, filename):
        obj = {
            "biomeIds": self.biomeIds,
            "biomeGridN": self.biomeGridN,
            "biomeGridS": self.biomeGridS,
            "resrcGridN": self.resrcGridN,
            "resrcGridS": self.resrcGridS,
        }
        with open(filename, "wb") as f:
            CsSF_Biom.build_stream(obj, f)
        print(f"Saved to: {filename}")

def clone_biom(biom):
    new = BiomFile()
    new.biomeIds = biom.biomeIds.copy()
    new.biomeGridN = biom.biomeGridN.copy()
    new.resrcGridN = biom.resrcGridN.copy()
    new.biomeGridS = biom.biomeGridS.copy()
    new.resrcGridS = biom.resrcGridS.copy()
    return new

def main():
    plugin_name, planet_biomes = load_planet_biomes(CSV_PATH)
    output_subdir = OUTPUT_DIR / plugin_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    template = BiomFile()
    template.load(TEMPLATE_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for planet, new_ids in planet_biomes.items():
        print(f"Processing {planet} with {len(new_ids)} biome(s)")
        new_biom = clone_biom(template)
        new_biom.overwrite_biome_ids(new_ids)
        out_path = output_subdir / f"{planet}.biom"
        new_biom.save(out_path)

if __name__ == "__main__":
    main()