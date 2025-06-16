from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path
import csv
import json

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Configuration paths (assume these are imported correctly)
from PlanetConstants import CONFIG_PATH, CSV_PATH, BIOME_HUMIDITY, get_config, save_config
from PlanetThemes import get_biome_palette_stylesheet

# Global configuration
config = get_config()


# -----------------------------
# Biome Entry Data Structure
# -----------------------------
@dataclass
class BiomeEntry:
    editor_id: str
    form_id: int
    color: tuple[int, int, int]
    height: int
    category: str
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.extra.get(key, default)

    def humidity(self) -> float:
        return BIOME_HUMIDITY.get(self.category.lower(), 0.0)


# -----------------------------
# Biome Database
# -----------------------------
class BiomeDatabase:
    def __init__(self):
        self.biomes_by_id: Dict[int, BiomeEntry] = {}
        self.biomes_by_name: Dict[str, BiomeEntry] = {}
        self.biome_by_height: Dict[int, BiomeEntry] = {}

    def all_biomes(self) -> list[BiomeEntry]:
        return list(self.biomes_by_name.values())
    
    def active_biomes(self, config: dict) -> list[BiomeEntry]:
        active = []
        for i in range(7):  # Assuming 7 active biomes
            key = f"biome{i:02}_editor_id"
            editor_id = config.get(key)
            if editor_id:
                biome = self.biomes_by_name.get(editor_id)
                if biome:
                    active.append(biome)
        return active

    def load_csv(self, csv_path: Path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    editor_id = row.get("EditorID", "").strip()
                    form_id = int(row.get("FormID", "0"), 16)
                    height = int(row.get("HeightIndex", "0"), 16)
                    category = row.get("BiomeCategory", "").strip()

                    red = int(row.get("Red", 0))
                    green = int(row.get("Green", 0))
                    blue = int(row.get("Blue", 0))
                    color = (red, green, blue)

                    known_keys = {
                        "FormID",
                        "EditorID",
                        "Red",
                        "Green",
                        "Blue",
                        "HeightIndex",
                        "BlockPatternID",
                        "BlockPatternEditorID",
                        "Category",
                    }
                    extra = {k: v for k, v in row.items() if k not in known_keys}

                    biome = BiomeEntry(editor_id, form_id, color, height, category, extra)
                    self.biomes_by_id[form_id] = biome
                    self.biomes_by_name[editor_id] = biome
                    self.biome_by_height[height] = biome
                except Exception as e:
                    print(f"[BiomeDB] Skipping invalid row: {e}")


# -----------------------------
# Biome Dropdown Selection Handler
# -----------------------------
def update_biome_selection(ui, key, biome_db, index):
    config[key] = index
    base_key = key.replace("_qcombobox", "")

    editor_ids = list(biome_db.biomes_by_name.keys())
    if index < 0 or index >= len(editor_ids):
        print(f"[Error] Invalid biome index {index} for {key}")
        index = 0  # Fallback to first biome
        config[key] = index

    biome_name = editor_ids[index]
    biome = biome_db.biomes_by_name.get(biome_name)
    if not biome:
        print(f"[Error] Biome '{biome_name}' not found in database")
        return

    r, g, b = biome.color
    height = biome.height # do we need to store this? It's only ever used for processing
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)

    config[f"{base_key}_editor_id"] = biome.editor_id
    config[f"{base_key}_formid"] = biome.form_id
    config[f"{base_key}_color"] = hex_color

    config[f"{base_key}_color"] = hex_color

    print(
        f"[Debug] Updated {base_key}: editor_id={biome.editor_id}, formid={biome.form_id}, color={hex_color}, height={height}, index={index}"
    )

    save_config()

    # Apply only to the progress bar, using the centralized theme logic
    pg_stylesheet = get_biome_palette_stylesheet(config)
    ui.biome_palette_progressBar.setStyleSheet(pg_stylesheet)
    ui.biome_palette_progressBar.repaint()

def get_biome_colormaps(config: dict) -> tuple[ListedColormap, LinearSegmentedColormap]:
    # Pull and sort biome colors
    biome_colors = [config.get(f"biome{i:02}_color", "#000000") for i in range(7)]

    # Discrete colormap: exact biome ID mapping
    discrete_cmap = ListedColormap(
        [mcolors.to_rgb(c) for c in biome_colors], name="BiomeDiscrete"
    )

    # Smooth gradient colormap: for height/color maps
    # Use evenly spaced stops (same as progress bar)
    positions = [i / 6.0 for i in range(7)]
    gradient_data = list(zip(positions, biome_colors))
    gradient_cmap = LinearSegmentedColormap.from_list("BiomeGradient", gradient_data)

    return discrete_cmap, gradient_cmap


biome_db = BiomeDatabase()
biome_db.load_csv(CSV_PATH)
