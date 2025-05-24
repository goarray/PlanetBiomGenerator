import json
import sys
from pathlib import Path
import zlib
import os
from typing import List, Dict
import uuid
from PlanetConstants import (
    BASE_DIR,
    CRC_MAP,
    CONFIG_PATH,
    PLUGINS_DIR,
    BIOM_DIR,
    DDS_OUTPUT_DIR,
    MAT_OUTPUT_DIR,
    MATERIAL_PATH,
)

# Core directories
bundle_dir = getattr(sys, "_MEIPASS", None)
if bundle_dir:
    BASE_DIR = Path(bundle_dir).resolve()
else:
    BASE_DIR = Path(__file__).resolve().parent


def load_config():
    """Load configuration from custom_config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_planet_material_paths():
    """Yield (planet_name, texture_path, material_path) for each .biom in the plugin."""
    config = load_config()
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")

    biom_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    if not biom_dir.exists():
        raise FileNotFoundError(f"Expected biom directory not found: {biom_dir}")

    biom_files = sorted(biom_dir.glob("*.biom"))

    for biom_path in biom_files:
        planet_name = biom_path.stem

        texture_path = (
            DDS_OUTPUT_DIR
            / plugin_name
            / "planets"
            / planet_name
            / f"{planet_name}_color.dds"
        )
        material_path = (
            PLUGINS_DIR
            / f"{plugin_name}"
            / MAT_OUTPUT_DIR
            / f"{plugin_name}"
            / "planets"
            / f"{planet_name}.mat"
        )

        print(f"Debug: Material Path {material_path}")
        print(f"Exists {Path(material_path).exists()}")

        write_material_file(material_path, plugin_name, planet_name)

        yield planet_name, texture_path, material_path


def write_material_file(material_path, plugin_name, planet_name):
    """Generate and write a formatted .mat file using placeholders."""

    # Convert string path to Path object if necessary
    output_path = Path(material_path)

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the template file and replace placeholders
    template_path = MATERIAL_PATH  # ✅ Your template file location
    try:
        with open(template_path, "r") as f:
            mat_content = f.read()

        # Replace placeholders
        mat_content = mat_content.replace("plugin_name", plugin_name)
        mat_content = mat_content.replace("planet_name", planet_name)

        # Write the updated material file
        with open(output_path, "w") as f:
            f.write(mat_content)

        print(f"Material file written to: {output_path}")

    except FileNotFoundError:
        print(f"Template file not found: {template_path}", file=sys.stderr)


def main():
    """Run material processing and print results."""
    print("=== Starting Material Path Processing ===")


if __name__ == "__main__":
    main()  #
