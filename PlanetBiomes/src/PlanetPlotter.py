from pathlib import Path
import pyvista as pv
import numpy as np
import json
import sys
from shutil import copyfile
from PlanetConstants import (
    PLUGINS_DIR,
    ASSETS_DIR,
    MESH_OUTPUT_DIR,
    BIOM_DIR,
    MESH_PATH,
    TEMP_DIR,
    PNG_OUTPUT_DIR,
    TEMPLATE_PATH,
    IMAGE_DIR,
    CONFIG_PATH,
    load_config,
)
from PlanetNewsfeed import handle_news

import pyvista as pv
import numpy as np
from pathlib import Path

TEXTURE_TYPES = [
    "biome",
    "surface_metal",
    "terrain_normal",
    "ao",
    "resource",
    "ocean_mask",
    "normal",
    "rough",
    # "fault",
    "color",
]


# Global configuration
def load_config():
    """Load plugin_name from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

    plugin_name = config.get("plugin_name", "default_plugin")
    if plugin_name == "default_plugin" or not plugin_name.endswith(".esm"):
        handle_news(
            None,
            "error",
            f"[ERROR] Invalid or missing plugin_name in config: {plugin_name}",
        )
        sys.exit(1)


def generate_sphere(plotter):
    config = load_config()
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")

    biom_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    biom_files = sorted(biom_dir.glob("*.biom"))

    if not biom_files:
        print(f"No .biom files in {biom_dir}, checking template path...")
        biom_files = sorted(TEMPLATE_PATH.glob("*.biom"))

    if not biom_files:
        raise FileNotFoundError(f"No .biom files found in {biom_dir} or {TEMPLATE_PATH}")

    planet_name = biom_files[0].stem

    # Create base sphere geometry
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

    # North hemisphere UVs
    phi_n = phi[north_mask]
    theta_n = theta[north_mask]
    r_n = phi_n / (np.pi / 2)
    u_n = 0.5 + 0.5 * r_n * np.cos(theta_n)
    v_n = 0.5 + 0.5 * r_n * np.sin(theta_n)
    v_n = v_n * 0.5 + 0.5  # top half

    # South hemisphere UVs with longitude inversion
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
    sphere.active_texture_coordinates = np.column_stack((u, v))

    # Dictionary to store meshes for each texture type
    meshes = {}

    # Load textures and create meshes
    for texture_type in TEXTURE_TYPES:
        texture_path = (
            PNG_OUTPUT_DIR
            / plugin_name
            / planet_name
            / f"{planet_name}_{texture_type}.png"
        )
        if not texture_path.exists():
            print(f"Warning: Missing texture: {texture_path}")
            continue

        # Load texture
        texture = pv.read_texture(texture_path)

        # Create a copy of the sphere for this texture
        mesh = sphere.copy()
        mesh.active_texture_coordinates = np.column_stack((u, v))  # Reapply UVs to copy
        meshes[texture_type] = {"mesh": mesh, "texture": texture, "visible": True}

    # Plotting and toggling logic
    plotter.clear()
    plotter.set_background("black")

    # Add all meshes to the plotter, initially visible
    for texture_type, data in meshes.items():
        plotter.add_mesh(data["mesh"], texture=data["texture"], name=texture_type)

    # Function to toggle mesh visibility
    def toggle_mesh(texture_type, visible):
        if texture_type in meshes:
            meshes[texture_type]["visible"] = visible
            if visible:
                plotter.add_mesh(
                    meshes[texture_type]["mesh"],
                    texture=meshes[texture_type]["texture"],
                    name=texture_type,
                )
            else:
                plotter.remove_actor(texture_type)

    # Example: Bind toggle to a callback (e.g., for UI or key press)
    # This is a placeholder; integrate with your UI or keybinding system
    def toggle_biome(state):
        toggle_mesh("biome", state)

    def toggle_resource(state):
        toggle_mesh("resource", state)

    def toggle_color(state):
        toggle_mesh("color", state)

    def toggle_surface(state):
        toggle_mesh("surface_metal", state)

    def toggle_ocean(state):
        toggle_mesh("ocean_mask", state)

    def toggle_normal(state):
        toggle_mesh("normal", state)

    def toggle_ambient(state):
        toggle_mesh("ao", state)

    def toggle_rough(state):
        toggle_mesh("rough", state)

    def toggle_terrain(state):
        toggle_mesh("terrain_normal", state)

    # Add key bindings for toggling (example for PyVista's built-in key events)
    plotter.add_key_event(
        "b", lambda: toggle_biome(not meshes.get("biome", {}).get("visible", False))
    )
    plotter.add_key_event(
        "e", lambda: toggle_resource(not meshes.get("resource", {}).get("visible", False))
    )
    plotter.add_key_event(
        "c", lambda: toggle_color(not meshes.get("color", {}).get("visible", False))
    )
    plotter.add_key_event(
        "s", lambda: toggle_surface(not meshes.get("surface_metal", {}).get("visible", False))
    )
    plotter.add_key_event(
        "o", lambda: toggle_ocean(not meshes.get("ocean_mask", {}).get("visible", False))
    )
    plotter.add_key_event(
        "n", lambda: toggle_normal(not meshes.get("normal", {}).get("visible", False))
    )
    plotter.add_key_event(
        "a", lambda: toggle_ambient(not meshes.get("ao", {}).get("visible", False))
    )
    plotter.add_key_event(
        "r", lambda: toggle_rough(not meshes.get("rough", {}).get("visible", False))
    )
    plotter.add_key_event(
        "t", lambda: toggle_terrain(not meshes.get("terrain_normal", {}).get("visible", False))
    )

    plotter.reset_camera()
    return meshes  # Return meshes for external control if needed


# Example usage
if __name__ == "__main__":
    plotter = pv.Plotter()
    meshes = generate_sphere(plotter)
    plotter.show()
