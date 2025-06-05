from pathlib import Path
import pyvista as pv
import numpy as np
from shutil import copyfile
from PlanetConstants import (
    PLUGINS_DIR,
    ASSETS_DIR,
    MESH_OUTPUT_DIR,
    BIOM_DIR,
    MESH_PATH,
    TEMP_DIR,
    PNG_OUTPUT_DIR,
    load_config,
)
from PlanetNewsfeed import handle_news


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
    config = load_config()
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")

    biom_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    mesh_output_dir = PLUGINS_DIR / plugin_name / "meshes" / plugin_name / "planets"
    template_nif = MESH_PATH

    if not biom_dir.exists() or not template_nif.exists():
        raise FileNotFoundError("Required biom or template directory missing.")

    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    for biom_path in sorted(biom_dir.glob("*.biom")):
        planet_name = biom_path.stem
        output_nif = mesh_output_dir / f"{planet_name}.nif"

        if not template_nif.exists():
            handle_news(
                None, "warn", f"Template not found for {planet_name}: {template_nif}"
            )
            continue

        copyfile(template_nif, output_nif)
        mat_path = f"materials\\{plugin_name}\\planets\\{planet_name}.mat"

        try:
            patch_nif_material_path(output_nif, mat_path)
        except Exception as e:
            handle_news(None, "error", f"Failed to patch {output_nif.name}: {e}")


def generate_sphere(plotter):
    config = load_config()
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")

    biom_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    biom_files = sorted(biom_dir.glob("*.biom"))
    if not biom_files:
        raise FileNotFoundError("No .biom files found.")

    planet_name = biom_files[0].stem
    texture_path = (
        PNG_OUTPUT_DIR / plugin_name / planet_name / f"{planet_name}_color.png"
    )
    if not texture_path.exists():
        raise FileNotFoundError(f"Missing texture: {texture_path}")

    texture = pv.read_texture(texture_path)

    # Create full sphere
    sphere = pv.Sphere(theta_resolution=2048, phi_resolution=1024)
    points = sphere.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    theta = np.mod(theta, 2 * np.pi)

    # Hemisphere masks
    north_mask = phi <= (np.pi / 2)
    south_mask = ~north_mask

    # --- North hemisphere UVs ---
    phi_n = phi[north_mask]
    theta_n = theta[north_mask]
    r_n = phi_n / (np.pi / 2)

    u_n = 0.5 + 0.5 * r_n * np.cos(theta_n)
    v_n = 0.5 + 0.5 * r_n * np.sin(theta_n)
    v_n = v_n * 0.5 + 0.5  # top half

    # --- South hemisphere UVs with longitude inversion ---
    phi_s = phi[south_mask]
    theta_s = np.mod(-theta[south_mask], 2 * np.pi)  # Invert longitude
    r_s = (np.pi - phi_s) / (np.pi / 2)

    u_s = 0.5 + 0.5 * r_s * np.cos(theta_s)
    v_s = 0.5 + 0.5 * r_s * np.sin(theta_s)
    v_s = v_s * 0.5  # bottom half

    # Assemble full texture coordinates
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    u[north_mask] = u_n
    v[north_mask] = v_n
    u[south_mask] = u_s
    v[south_mask] = v_s

    sphere.active_t_coords = np.column_stack((u, v))

    # Plot it in the embedded viewer
    plotter.clear()
    plotter.add_mesh(sphere, texture=texture)
    plotter.reset_camera()


# Optional entry point
if __name__ == "__main__":
    generate_and_patch_planet_meshes()
