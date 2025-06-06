import numpy as np
import pyvista as pv
from pathlib import Path
import json
import sys
import time
import re
from functools import partial
from PlanetConstants import (
    PLUGINS_DIR,
    BIOM_DIR,
    TEMPLATE_PATH,
    PNG_OUTPUT_DIR,
    CONFIG_PATH,
    load_config,
)
from PlanetNewsfeed import handle_news

config = load_config()

DEFAULT_OPACITIES = {
    "surface_metal": 1.0,
    "color": 1.0,
    "fault": 0.2,
    "resource": 0.1,
    "biome": 0.4,
    "rough": 0.2,
    "normal": 0.2,
    "ao": 0.2,
    "ocean_mask": 0.2,
}

TEXTURE_OPACITIES = {
    **DEFAULT_OPACITIES,
    **{
        key.replace("_opacity", ""): value
        for key, value in config.items()
        if key.endswith("_opacity")
    },
}

TEXTURE_TYPES = [
    "surface_metal",
    "color",
    "fault",
    "resource",
    "biome",
    "rough",
    "normal",
    "ao",
    "ocean_mask",
]


def toggle_mesh(main_window, texture_type, visible, plotter, meshes):
    """Toggle mesh visibility with correct opacity and sync checkbox."""
    if texture_type not in meshes:
        print(f"[WARN] Unknown texture_type: {texture_type}")
        return
    checkbox_name = f"toggle_{texture_type}_view"
    checkbox = getattr(main_window, checkbox_name, None)
    if checkbox:
        checkbox.setChecked(visible)
    meshes[texture_type]["visible"] = visible
    if visible:
        refresh_mesh_opacity(texture_type, plotter, meshes)
    else:
        plotter.remove_actor(texture_type, reset_camera=False)
        print(f"[TOGGLE] {texture_type}: visible={visible}")


def handle_toggle_view(main_window, texture_type, plotter, meshes):
    """Handle toggle view for a texture type."""
    if texture_type not in meshes:
        print(f"[WARN] Unknown texture_type: {texture_type}")
        return
    visible = not meshes[texture_type].get("visible", False)
    toggle_mesh(main_window, texture_type, visible, plotter, meshes)


def auto_connect_toggle_buttons(window, plotter, meshes):
    """Connect UI toggle buttons to handle_toggle_view."""
    pattern = re.compile(r"toggle_(.+)_view")
    for attr_name in dir(window):
        match = pattern.fullmatch(attr_name)
        if match:
            texture_type = match.group(1)
            button = getattr(window, attr_name)
            if callable(getattr(button, "clicked", None)):
                print(f"Connecting {attr_name} to toggle {texture_type}")
                button.clicked.connect(
                    partial(handle_toggle_view, window, texture_type, plotter, meshes))

def refresh_mesh_opacity(texture_type, plotter, meshes):
    """Refresh opacity for a single mesh."""
    config = load_config()
    if texture_type not in meshes:
        print(f"[WARN] Tried to refresh unknown texture: {texture_type}")
        return
    if not meshes[texture_type].get("visible", True):
        print(f"[DEBUG] {texture_type} is hidden; skipping opacity refresh.")
        return
    plotter.remove_actor(texture_type, reset_camera=False)
    opacity = config.get(
        f"{texture_type}_opacity", TEXTURE_OPACITIES.get(texture_type, 1.0)
    )
    plotter.add_mesh(
        meshes[texture_type]["mesh"],
        texture=meshes[texture_type]["texture"],
        name=texture_type,
        opacity=opacity,
    )
    print(
        f"[REFRESH] {texture_type}: opacity={opacity}, visible={meshes[texture_type].get('visible')}"
    )
    plotter.render()


def refresh_all_opacities(main_window, plotter, meshes):
    """Refresh opacities for all visible meshes."""
    for texture_type in meshes:
        if meshes[texture_type].get("visible", False):
            refresh_mesh_opacity(texture_type, plotter, meshes)
            # Sync checkbox state
            checkbox_name = f"toggle_{texture_type}_view"
            checkbox = getattr(main_window, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(True)


def generate_sphere(main_window, plotter, run_once=[False]):
    config = load_config()
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")

    biom_dir = PLUGINS_DIR / plugin_name / BIOM_DIR / plugin_name
    biom_files = sorted(biom_dir.glob("*.biom"))

    if not biom_files:
        print(f"No .biom files in {biom_dir}, checking template path...")
        biom_files = sorted(TEMPLATE_PATH.glob("*.biom"))

    if not biom_files:
        raise FileNotFoundError(
            f"No .biom files found in {biom_dir} or {TEMPLATE_PATH}"
        )

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
    north_mask = phi <= np.pi / 2
    south_mask = phi > np.pi / 2

    # North hemisphere UVs
    phi_n = phi[north_mask]
    theta_n = theta[north_mask]
    r_n = phi_n / (np.pi / 2)
    r_n = np.clip(r_n, 0, 1)
    u_n = 0.5 + 0.5 * r_n * np.cos(theta_n)
    v_n = 0.5 + 0.5 * r_n * np.sin(theta_n)
    v_n = v_n * 0.5 + 0.5  # top half

    # South hemisphere UVs with longitude inversion
    phi_s = phi[south_mask]
    theta_s = np.mod(-theta[south_mask], 2 * np.pi)
    r_s = (np.pi - phi_s) / (np.pi / 2)
    r_s = np.clip(r_s, 0, 1)
    u_s = 0.5 + 0.5 * r_s * np.cos(theta_s)
    v_s = 0.5 + 0.5 * r_s * np.sin(theta_s)
    v_s = v_s * 0.5  # bottom half

    # Ensure exact UV alignment at equator
    equator_mask = np.isclose(phi, np.pi / 2, atol=1e-6)
    if np.any(equator_mask):
        r_eq = 1.0
        u_eq = 0.5 + 0.5 * r_eq * np.cos(theta[equator_mask])
        v_eq = 0.5 + 0.5 * r_eq * np.sin(theta[equator_mask])
        v_eq = v_eq * 0.5 + 0.5
        u_n[phi_n >= 1.0 - 1e-6] = u_eq
        v_n[phi_n >= 1.0 - 1e-6] = v_eq
        u_s[phi_s <= np.pi / 2 + 1e-6] = u_eq
        v_s[phi_s <= np.pi / 2 + 1e-6] = v_eq

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
        mesh.active_texture_coordinates = np.column_stack((u, v))
        meshes[texture_type] = {"mesh": mesh, "texture": texture, "visible": True}

    # Plotting and toggling logic
    plotter.clear()
    plotter.set_background("black")
    plotter.enable_depth_peeling(number_of_peels=100)
    plotter.camera.azimuth += 0.01
    plotter.render()
    plotter.reset_camera()
    plotter.camera.zoom(0.9)

    # Add all meshes to the plotter, initially visible
    for index, (texture_type, data) in enumerate(meshes.items()):
        opacity = config.get(
            f"{texture_type}_opacity", TEXTURE_OPACITIES.get(texture_type, 1.0)
        )
        base_mesh = data["mesh"]
        offset_mesh = base_mesh.copy()
        offset = 0.001 * index
        offset_mesh.compute_normals(inplace=True)
        normals = offset_mesh.point_normals
        offset_mesh.points += normals * offset
        plotter.add_mesh(
            offset_mesh,
            texture=data["texture"],
            name=texture_type,
            opacity=opacity,
        )
        meshes[texture_type]["mesh"] = offset_mesh
        # Sync checkbox state
        checkbox_name = f"toggle_{texture_type}_view"
        checkbox = getattr(main_window, checkbox_name, None)
        if checkbox:
            checkbox.setChecked(True)
    plotter.update()
    if run_once[0]:
        time.sleep(0.5)
    run_once[0] = True

    # Add key bindings for toggling
    key_callbacks = {
        "f": "fault",
        "b": "biome",
        "e": "resource",
        "c": "color",
        "s": "surface_metal",
        "o": "ocean_mask",
        "n": "normal",
        "a": "ao",
        "r": "rough",
    }
    for key, texture_type in key_callbacks.items():
        plotter.add_key_event(
            key,
            lambda t=texture_type: toggle_mesh(
                main_window,
                t,
                not meshes.get(t, {}).get("visible", False),
                plotter,
                meshes,
            ),
        )

    # Example: Connect a config update callback (e.g., for sliders)
    def on_config_updated():
        global config, TEXTURE_OPACITIES
        config = load_config()
        TEXTURE_OPACITIES = {
            **DEFAULT_OPACITIES,
            **{
                key.replace("_opacity", ""): value
                for key, value in config.items()
                if key.endswith("_opacity")
            },
        }
        refresh_all_opacities(main_window, plotter, meshes)

    # Placeholder: Assume main_window has a signal for config updates
    if hasattr(main_window, "config_updated"):
        main_window.config_updated.connect(on_config_updated)

    return meshes

if __name__ == "__main__":
    plotter = pv.Plotter()
    meshes = generate_sphere(None, plotter)
    plotter.show()
