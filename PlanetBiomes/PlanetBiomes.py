from pathlib import Path
from construct import Struct, Const, Rebuild, this, len_
from construct import Int32ul as UInt32, Int16ul as UInt16, Int8ul as UInt8
import numpy as np
import csv

# Constants
GRID_SIZE = [0x100, 0x100]
GRID_FLATSIZE = GRID_SIZE[0] * GRID_SIZE[1]
SCRIPT_DIR = Path(__file__).parent

# Update the base directory to the root directory
BASE_DIR = Path(__file__).parent.parent  # Get the parent of the 'src' folder

# Path adjustments based on new directory structure
TEMPLATE_PATH = SCRIPT_DIR / "PlanetBiomes.biom"      # PlanetBiomes.biom
CSV_PATH = SCRIPT_DIR / "xEditOutput" / "PlanetBiomes.csv"     # PlanetBiomes.csv
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
    with open(csv_path, newline='') as csvfile:
        first_line = csvfile.readline().strip()
        plugin_name = first_line.strip()
        reader = csv.DictReader(csvfile, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"])
        next(reader, None)  # Skip header row
        for row in reader:
            planet = row["PlanetName"]
            try:
                form_id = int(row["BIOM_FormID"], 16)
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
        if not new_biome_ids:
            raise ValueError("No new biome IDs provided.")
        mapping = {
            old_id: new_biome_ids[i % len(new_biome_ids)]
            for i, old_id in enumerate(self.biomeIds)
        }
        self.biomeGridN = np.array([mapping.get(x, new_biome_ids[0]) for x in self.biomeGridN], dtype=np.uint32)
        self.biomeGridS = np.array([mapping.get(x, new_biome_ids[0]) for x in self.biomeGridS], dtype=np.uint32)
        self.biomeIds = list(set(new_biome_ids))

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
