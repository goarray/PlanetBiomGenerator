from pathlib import Path
import pyvista as pv
import numpy as np
import sys
import subprocess
from shutil import copyfile
from PlanetConstants import (
    PLUGINS_DIR,
    SCRIPT_DIR,
    ASSETS_DIR,
    MESH_OUTPUT_DIR,
    BIOM_DIR,
    MESH_PATH,
    TEMP_DIR,
    PNG_OUTPUT_DIR,
    IMAGE_DIR,
    get_config,
)
from PlanetNewsfeed import handle_news

config = get_config()

def patch_nif_material_path(nif_path: Path, new_mat_path: str):
    """
    Replace the material path in a .nif file with the given new path.
    """
    with open(nif_path, "rb") as f:
        data = f.read()

    byte_data = bytearray(data)

    start = byte_data.find(b"materials\\")
    if start == -1:
        raise ValueError(f"Material path not found in {nif_path.name}")

    end = byte_data.find(b"\x00", start)
    if end == -1:
        raise ValueError(
            f"Null terminator not found after material path in {nif_path.name}"
        )

    new_bytes = new_mat_path.encode("utf-8")
    length = end - start
    padded = new_bytes.ljust(length, b"\x00")

    if len(padded) > length:
        raise ValueError(f"New material path too long for {nif_path.name}")

    byte_data[start:end] = padded[:length]

    with open(nif_path, "wb") as f:
        f.write(byte_data)

    handle_news(None, "info", f"Patched: {nif_path.name} → {new_mat_path}")


def generate_and_patch_planet_meshes():
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")
    planet_name = config.get("planet_name", "PLUGINNOTFOUND")

    mesh_output_dir = PLUGINS_DIR / plugin_name / "meshes" / plugin_name / "planets"
    template_nif = MESH_PATH

    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    output_nif = mesh_output_dir / f"{planet_name}.nif"

    if not template_nif.exists():
        handle_news(
            None, "warn", f"Template not found for {planet_name}: {template_nif}"
        )

    copyfile(template_nif, output_nif)
    mat_path = f"materials\\{plugin_name}\\planets\\{planet_name}.mat"

    try:
        patch_nif_material_path(output_nif, mat_path)
    except Exception as e:
        handle_news(None, "error", f"Failed to patch {output_nif.name}: {e}")


# Optional entry point
if __name__ == "__main__":
    generate_and_patch_planet_meshes()

    if config.get("run_planet_surface", True):
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "PlanetSurface.py")], check=True
        )
    else:
        sys.stdout.flush()
        sys.exit(0)
