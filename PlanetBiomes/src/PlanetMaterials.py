import json
import sys
from pathlib import Path
import zlib
import os
import random
import binascii
import copy
from typing import List, Dict
import uuid
from PlanetNewsfeed import handle_news
from PlanetHasher import create_resource_id, create_simple_crc_id
from PlanetConstants import (
    BASE_DIR,
    CRC_MAP,
    CONFIG_PATH,
    PLUGINS_DIR,
    BIOM_DIR,
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
            PLUGINS_DIR
            / plugin_name
            / "textures"
            / plugin_name
            / planet_name
            / f"{planet_name}_color.dds"
        )

        material_path = (
            PLUGINS_DIR
            / plugin_name
            / "materials"
            / plugin_name
            / "planets"
            / f"{planet_name}.mat"
        )

        handle_news(None, "info", f"Material Path: {material_path}")

        write_material_file(material_path, plugin_name, planet_name)

        yield planet_name, texture_path, material_path


def generate_crc_hashes(mat_data):
    mat_data = copy.deepcopy(mat_data)
    id_mapping = {}

    for obj_index, obj in enumerate(mat_data["Objects"]):
        obj_name = None
        texture_path = None

        for component in obj.get("Components", []):
            if component.get("Type") == "BSComponentDB::CTName":
                obj_name = component.get("Data", {}).get("Name")
            elif component.get("Type") == "BSMaterial::MRTextureFile":
                texture_path = component.get("Data", {}).get("FileName", "")

        if not obj_name:
            obj_name = f"object_{obj_index}"

        # --- Assign object ID based on texture path ---
        if "ID" in obj:
            old_id = obj["ID"]
            if texture_path:
                folder = os.path.dirname(texture_path)
                file = os.path.splitext(os.path.basename(texture_path))[0]
                new_id = create_resource_id(folder, file)
                id_mapping[old_id] = new_id
                obj["ID"] = new_id

        # --- Assign component IDs using simple CRC ---
        for comp_index, component in enumerate(obj.get("Components", [])):
            if "Data" in component and "ID" in component["Data"]:
                old_id = component["Data"]["ID"]
                if old_id:
                    comp_name = f"{obj_name}_comp_{comp_index}"
                    new_id = create_simple_crc_id(comp_name, index=comp_index)
                    id_mapping[old_id] = new_id
                    component["Data"]["ID"] = new_id

    # --- Update 'To' fields in Edges ---
    for obj in mat_data["Objects"]:
        for edge in obj.get("Edges", []):
            if edge.get("To") not in ("<this>", None) and edge["To"] in id_mapping:
                edge["To"] = id_mapping[edge["To"]]

    return mat_data


def write_material_file(material_path, plugin_name, planet_name):
    """Generate and write a formatted .mat file with hashed values."""

    output_path = Path(material_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read template file
    template_path = MATERIAL_PATH
    try:
        with open(template_path, "r") as f:
            mat_data = json.load(f)  # ✅ Load JSON directly instead of treating as text

        # Replace placeholders
        mat_data["Filename"] = (
            mat_data["Filename"]
            .replace("plugin_name", plugin_name)
            .replace("planet_name", planet_name)
        )

        for obj in mat_data.get("Objects", []):
            for component in obj.get("Components", []):
                if component.get("Type") == "BSMaterial::MRTextureFile":
                    file_data = component.get("Data", {})
                    if "FileName" in file_data:
                        file_data["FileName"] = (
                            file_data["FileName"]
                            .replace("plugin_name", plugin_name)
                            .replace("planet_name", planet_name)
                        )

        # Generate CRC-based hashes in one step
        updated_mat_data = generate_crc_hashes(mat_data)

        # Write final material file
        with open(output_path, "w") as f:
            json.dump(
                updated_mat_data, f, indent=4
            )  # ✅ Directly saving the final version

        handle_news(None, "info", f"Material file written to: {output_path}")

        # Print resource hash info for this .mat file
        relative_folder = f"materials/{plugin_name}/planets"
        filename = planet_name  # no extension

        folder_hash = create_resource_id(relative_folder, "")
        file_hash = create_resource_id(relative_folder, filename)

        print(f"[MATERIAL PATH]: {relative_folder} / {filename}")

        # Split and extract folder/file CRCs from the res string
        parts = file_hash.split(":")
        folder_crc = parts[1]
        file_crc = parts[2]

        print(f"[MATERIAL FOLDER HASH]: {folder_crc}")
        print(f"[MATERIAL FILE HASH]:   {file_crc}")

    except FileNotFoundError:
        print(f"Template file not found: {template_path}", file=sys.stderr)


def main():
    """Run material processing and print results."""
    print("=== Starting Material Path Processing ===")
    for planet_name, texture_path, material_path in get_planet_material_paths():
        handle_news(None, "info", f"Updated {planet_name} material. Saved at: {material_path}")


if __name__ == "__main__":
    main()  #
