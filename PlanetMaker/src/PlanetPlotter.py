import numpy as np
import pyvista as pv
from pyvista import Plotter
import vtk
from pathlib import Path
from PIL import Image
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
    get_config,
)
from PlanetNewsfeed import handle_news
from PlanetThemes import theme_data

config = get_config()

print("Config ID:", id(config))

DEFAULT_OPACITIES = {
    "height": 0.2,
    "terrain": 0.2,
    "surface_metal": 1.0,
    "normal": 0.2,
    "rough": 0.2,
    "humidity": 0.2,
    "ocean_mask": 0.2,
    "color": 1.0,
    "terrain_normal": 0.2,
    "resource": 0.1,
    "biome": 0.4,
    "mountain_mask": 0.2,
    "river_mask": 0.2,
    "colony_mask": 0.2,
}  # "ao": 0.2,

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
    "normal",
    "rough",
    "terrain_normal",
    "terrain",
    "resource",
    "biome",
    "color",
    "humidity",
    "ocean_mask",  # white transparent
    "river_mask",  # white transparent
    "mountain_mask",  # black transparent
    "road_mask",  # black transparent
    "colony_mask",  # black transparent
]  # "ao",


def enable_mesh(main_window, texture_type, visible, plotter, meshes):
    """enable mesh visibility with correct opacity and sync checkbox."""
    if texture_type not in meshes:
        print(f"[WARN] Unknown texture_type: {texture_type}")
        return
    checkbox_name = f"enable_{texture_type}_view"
    checkbox = getattr(main_window, checkbox_name, None)
    if checkbox:
        checkbox.setChecked(visible)
    meshes[texture_type]["visible"] = visible
    if visible:
        refresh_mesh_opacity(texture_type, plotter, meshes)
    else:
        plotter.remove_actor(texture_type, reset_camera=False)
        handle_news(None, "info", f"[ENABLED] {texture_type}: visible={visible}")


def handle_enable_view(main_window, texture_type, plotter, meshes):
    """Handle enable view for a texture type."""
    if texture_type not in meshes:
        print(f"[WARN] Unknown texture_type: {texture_type}")
        return
    visible = not meshes[texture_type].get("visible", False)
    enable_mesh(main_window, texture_type, visible, plotter, meshes)


def auto_connect_enable_buttons(window, plotter, meshes):
    """Connect UI enable buttons to handle_enable_view."""
    pattern = re.compile(r"enable_(.+)_view")
    for attr_name in dir(window):
        match = pattern.fullmatch(attr_name)
        if match:
            texture_type = match.group(1)
            button = getattr(window, attr_name)
            if callable(getattr(button, "clicked", None)):
                handle_news(None, "info", f"Connecting {attr_name} to enable {texture_type}")
                button.clicked.connect(
                    partial(handle_enable_view, window, texture_type, plotter, meshes))

def refresh_mesh_opacity(texture_type, plotter, meshes):
    """Refresh opacity for a single mesh."""
    if texture_type not in meshes:
        print(f"[WARN] Tried to refresh unknown texture: {texture_type}")
        return
    if not meshes[texture_type].get("visible", True):
        handle_news(None, "info", f"[DEBUG] {texture_type} is hidden; skipping opacity refresh.")
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
    handle_news(None, "info", 
        f"[REFRESH] {texture_type}: opacity={opacity}, visible={meshes[texture_type].get('visible')}"
    )
    plotter.render()


def refresh_all_opacities(main_window, plotter, meshes):
    """Refresh opacities for all visible meshes."""
    for texture_type in meshes:
        if meshes[texture_type].get("visible", False):
            refresh_mesh_opacity(texture_type, plotter, meshes)
            # Sync checkbox state
            checkbox_name = f"enable_{texture_type}_view"
            checkbox = getattr(main_window, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(True)


def force_load_texture(
    path: Path,
    transparent_black: bool = False,
    transparent_white: bool = False,
    grayscale_opacity: bool = False,
) -> pv.Texture:
    # Load image as RGBA
    image = Image.open(path).convert("RGBA")
    tex_data = np.array(image)

    if transparent_black:
        # Make pure black pixels fully transparent
        black_pixels = np.all(tex_data[:, :, :3] == 0, axis=2)
        tex_data[black_pixels, 3] = 0  # alpha=0 for black pixels
        tex_data[~black_pixels, 3] = 255  # alpha=255 elsewhere

    elif transparent_white:
        # Make pure white pixels fully transparent
        white_pixels = np.all(tex_data[:, :, :3] == 255, axis=2)
        tex_data[white_pixels, 3] = 0  # alpha=0 for white pixels
        tex_data[~white_pixels, 3] = 255  # alpha=255 elsewhere

    elif grayscale_opacity:
        # Set alpha based on grayscale brightness: black (0) = transparent, white (255) = opaque
        gray = np.mean(tex_data[:, :, :3], axis=2).astype(np.uint8)
        tex_data[:, :, 3] = gray

    else:
        # Default: fully opaque
        tex_data[:, :, 3] = 255

    return pv.Texture(tex_data)


def generate_sphere(main_window, plotter, run_once=[False]):
    theme_name = config.get("theme", "Dark")
    plugin_name = config.get("plugin_name", "default_plugin")
    planet_name = config.get("planet_name", "default_planet")

    resolution = config.get("texture_resolution", 256)

    # Dictionary to store meshes for each texture type
    meshes = {}

    # Fully reset plotter
    plotter.clear()  # Clear all actors, lights, and cameras
    meshes.clear()

    # Create base sphere geometry
    MAX_RESOLUTION = 512  # or 512 depending on available memory
    resolution = min(resolution, MAX_RESOLUTION)
    sphere = pv.Sphere(theta_resolution=resolution, phi_resolution=(2 * resolution))
    points = sphere.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    theta = np.mod(theta, 2 * np.pi)

    # Hemisphere masks
    north_mask = phi <= np.pi / 2
    south_mask = phi > np.pi / 2

    def normalize_radius(theta):
        return 1.0 / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

    # North hemisphere UVs
    phi_n = phi[north_mask]
    theta_n = theta[north_mask]
    r_n = (phi_n / (np.pi / 2)) * normalize_radius(theta_n)
    u_n = 0.5 + 0.5 * r_n * np.cos(theta_n)
    v_n = 0.5 + 0.5 * r_n * np.sin(theta_n)
    v_n = v_n * 0.5 + 0.5  # top half

    # South hemisphere UVs with longitude inversion
    phi_s = phi[south_mask]
    theta_s = np.mod(-theta[south_mask], 2 * np.pi)
    r_s = ((np.pi - phi_s) / (np.pi / 2)) * normalize_radius(theta_s)
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

    # Debug UV ranges
    print(
        f"[DEBUG] North UV: u_min={u[north_mask].min():.3f}, u_max={u[north_mask].max():.3f}, "
        f"v_min={v[north_mask].min():.3f}, v_max={v[north_mask].max():.3f}"
    )
    print(
        f"[DEBUG] South UV: u_min={u[south_mask].min():.3f}, u_max={u[south_mask].max():.3f}, "
        f"v_min={v[south_mask].min():.3f}, v_max={v[south_mask].max():.3f}"
    )
    # Debug UV ranges with additional angles
    angles = [45, 135, 225, 315]  # 45° and 135° east/west in both hemispheres
    angle_radians = np.radians(angles)

    for angle in angle_radians:
        mask_angle = np.isclose(
            theta, angle, atol=np.radians(2)
        )  # Allow slight tolerance
        if np.any(mask_angle):
            print(
                f"[DEBUG] Angle {np.degrees(angle):.0f}° UV: "
                f"u_min={u[mask_angle].min():.3f}, u_max={u[mask_angle].max():.3f}, "
                f"v_min={v[mask_angle].min():.3f}, v_max={v[mask_angle].max():.3f}"
            )

    sphere.active_texture_coordinates = np.column_stack((u, v))

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
        if texture_type in ("colony_mask", "mountain_mask", "road_mask", "river_mask"):
            texture = force_load_texture(texture_path, transparent_black=True)
        elif texture_type in ("ocean_mask"):
            texture = force_load_texture(texture_path, transparent_white=True)
        elif texture_type in ("humidity"):
            texture = force_load_texture(texture_path, grayscale_opacity=True)
        else:
            texture = force_load_texture(texture_path)
        # Correct normal map for southern hemisphere

        if texture_type in ("normal", "terrain_normal"):
            texture_img = Image.open(texture_path).convert("RGB")
            texture_array = np.array(texture_img)
            height, width, _ = texture_array.shape
            south_region = slice(0, height // 2)  # bottom half
            texture_array[south_region, :, 0] = 255 - texture_array[south_region, :, 0]  # flip X
            texture = pv.Texture(texture_array)

        # Create a copy of the sphere for this texture
        mesh = sphere.copy()
        mesh.active_texture_coordinates = np.column_stack((u, v))
        meshes[texture_type] = {"mesh": mesh, "texture": texture, "visible": True}

    # Plotting and toggling logic
    plotter.clear()
    theme = theme_data.get(theme_name) or theme_data.get("Starfield", {})
    background_color = theme.get("background", "#000000")
    plotter.set_background(background_color)
    plotter.enable_depth_peeling(number_of_peels=100)
    plotter.camera.azimuth += 0.01
    plotter.render()
    plotter.reset_camera()
    if not run_once[0]:
        plotter.camera.zoom(0.9)
        run_once[0] = True

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
        checkbox_name = f"enable_{texture_type}_view"
        checkbox = getattr(main_window, checkbox_name, None)
        if checkbox:
            checkbox.setChecked(True)

    if not plotter._initialized:
        plotter.initialize()
    plotter.update()

    def suppress_vtk_stereo_mode():
        # Override stereo keys with no-op
        for key in ['1','2','3','4','5','6','7','8','9']:
            plotter.add_key_event(key, lambda: None)
    suppress_vtk_stereo_mode()

    # Add key bindings for toggling
    key_callbacks = {
        "8": "surface_metal",
        "7": "normal",
        "6": "rough",
        "5": "resource",
        "4": "biome",
        "3": "terrain_normal",
        "2": "color",
        "1": "colony_mask",
        "0": "ocean_mask",
    }
    for key, texture_type in key_callbacks.items():
        plotter.add_key_event(
            key,
            lambda t=texture_type: enable_mesh(
                main_window,
                t,
                not meshes.get(t, {}).get("visible", False),
                plotter,
                meshes,
            ),
        )

    refresh_all_opacities(main_window, plotter, meshes)

    return meshes


def add_background_image(plotter: Plotter, image_path: str):
    # Load the image
    reader = vtk.vtkJPEGReader()  # or vtkPNGReader
    reader.SetFileName(image_path)
    reader.Update()

    image_actor = vtk.vtkImageActor()
    image_actor.GetMapper().SetInputConnection(reader.GetOutputPort())

    # Insert into renderer background
    renderer = plotter.renderer
    renderer.SetLayer(0)
    renderer.InteractiveOff()

    # Set layers for background image support
    plotter.ren_win.SetNumberOfLayers(2)
    renderer_background = vtk.vtkRenderer()
    renderer_background.SetLayer(0)
    renderer_background.InteractiveOff()
    renderer_background.AddActor(image_actor)

    # Move current renderer to upper layer
    renderer.SetLayer(1)

    # Add both renderers to the render window
    plotter.ren_win.AddRenderer(renderer_background)

    plotter.render()


if __name__ == "__main__":
    plotter = pv.Plotter()
    meshes = generate_sphere(None, plotter)
    plotter.show()
