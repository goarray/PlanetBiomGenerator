#!/usr/bin/env python3
"""
Biome Config Editor

A Tkinter-based GUI application for editing biome configuration settings.
Allows users to modify numerical values, toggle boolean settings, and manage
image pipeline configurations. Supports loading/saving JSON configs and
running an external PlanetBiomes.py script.

Dependencies:
- Python 3.8+
- tkinter
- PIL (Pillow)
- subprocess
- json
- pathlib
"""

from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import subprocess
import json
import signal
import os
import sys

# Directory paths
BASE_DIR = Path(__file__).parent.parent
IMAGE_DIR = BASE_DIR / "assets" / "images"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "default.png"

SCRIPT_PATH = BASE_DIR / "src" / "PlanetBiomes.py"

CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"

# Configuration keys for boolean values
BOOLEAN_KEYS = {
    "enable_equator_drag",
    "enable_pole_drag",
    "enable_equator_intrusion",
    "enable_pole_intrusion",
    "apply_distortion",
    "apply_resource_gradient",
    "apply_latitude_blending"
}

# Human-readable labels for UI elements
LABELS = {
    "lat_weight_factor": "Zoom",
    "squircle_exponent": "Diamond < Circle > Square",
    "distortion_sigma": "Fine Distortion",
    "lat_distortion_factor": "Large Distortion",
    "drag_radius": "Drag Radius",
    "noise_factor": "Equator Weight Mult",
    "enable_equator_drag": "Allow Equator Dragging",
    "enable_pole_drag": "Allow Pole Dragging",
    "enable_equator_intrusion": "Enable Equator Intrusions",
    "enable_pole_intrusion": "Enable Pole Intrusions",
    "apply_distortion": "Apply Terrain Distortion",
    "apply_resource_gradient": "Use Resource Gradient",
    "apply_latitude_blending": "Blend Biomes by Latitude",
    "zone_seed": "Zone generation seed"
}

# Global configuration dictionary
config = {}

# UI element storage
checkbox_vars = {}
slider_vars = {}

# Subprocess for PlanetBiomes.py
planet_biomes_process = None

def load_config():
    """Load configuration from custom or default JSON file."""
    global config
    try:
        config_path = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
        with open(config_path, "r") as f:
            raw_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found.")
        raw_config = {}

    # Convert boolean fields
    for category, sub_config in raw_config.items():
        for key, value in sub_config.items():
            if key in BOOLEAN_KEYS and isinstance(value, (float, int)):
                raw_config[category][key] = bool(int(value))

    config = raw_config

def save_config():
    """Save current configuration to JSON file."""
    if CONFIG_PATH.exists():
        os.remove(CONFIG_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def update_value(category, key, val, index=None):
    """Update configuration value and save to file."""
    if isinstance(config[category][key], bool):
        config[category][key] = bool(int(val))
    elif isinstance(config[category][key], list) and len(config[category][key]) == 2 and index is not None:
        val = float(val)
        if index == 0:
            val = min(val, config[category][key][1] - 0.01)
        elif index == 1:
            val = max(val, config[category][key][0] + 0.01)
        config[category][key][index] = val
    elif isinstance(config[category][key], int):
        config[category][key] = int(float(val))
    elif isinstance(config[category][key], float):
        config[category][key] = round(float(val), 2)

    save_config()

def start_planet_biomes():
    """Start PlanetBiomes.py, wait for completion, and exit."""
    global planet_biomes_process
    if planet_biomes_process is None or planet_biomes_process.poll() is not None:
        planet_biomes_process = subprocess.Popen(["python", str(SCRIPT_PATH)], shell=True)
        planet_biomes_process.wait()
        sys.exit()

def cancel_and_exit():
    """Terminate subprocess and exit application."""
    if planet_biomes_process:
        planet_biomes_process.terminate()
        planet_biomes_process.wait()
        try:
            os.kill(planet_biomes_process.pid, signal.SIGTERM)
        except OSError:
            pass
    os._exit(0)

def save_and_continue():
    """Save config, start PlanetBiomes, and hide UI."""
    save_config()
    start_planet_biomes()
    root.withdraw()

def reset_to_defaults():
    """Reset configuration to defaults and update UI."""
    global config
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")
        config = {}

    for category, sub_config in config.items():
        for key, value in sub_config.items():
            if key in checkbox_vars:
                checkbox_vars[key].set(bool(value))
            elif key in slider_vars:
                if isinstance(slider_vars[key], tuple):
                    min_slider, max_slider = slider_vars[key]
                    min_slider.set(value[0])
                    max_slider.set(value[1])
                    for slider in [min_slider, max_slider]:
                        slider.event_generate("<Motion>")
                else:
                    slider_vars[key].set(value)
                    slider_vars[key].event_generate("<Motion>")

    save_config()

def show_image(label_text):
    """Display image corresponding to hovered label."""
    if label_text in images:
        image_label.configure(image=images[label_text])
        image_label.image = images[label_text]

def hide_image(event):
    """Clear image display on mouse leave."""
    image_label.config(image=default_image)

def update_slider_label(value_pair, label):
    """Update slider label with min/max or single value."""
    if isinstance(value_pair, (tuple, list)) and len(value_pair) == 2:
        label.config(text=f"{value_pair[0]:.2f} - {value_pair[1]:.2f}")
    else:
        label.config(text=f"{float(value_pair):.2f}")

# Initialize configuration
load_config()

# Create main UI window
root = tk.Tk()
root.title("Biome Config Editor")
root.geometry("1024x820")

# Create side-by-side frames
left_frame = ttk.Frame(root)
left_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

center_frame = ttk.Frame(root)
center_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

right_frame = ttk.Frame(root)
right_frame.grid(row=0, column=2, sticky="ns", padx=10, pady=10)

# Organize sections into panels
frame_sliders = ttk.LabelFrame(left_frame, text="Numerical Values")
frame_sliders.pack(fill="x", padx=10, pady=5)

frame_booleans = ttk.LabelFrame(center_frame, text="Boolean Toggles")
frame_booleans.pack(fill="x", padx=10, pady=5)

frame_image_pipeline = ttk.LabelFrame(right_frame, text="Image Pipeline Settings")
frame_image_pipeline.pack(fill="x", padx=10, pady=5)

# Group assignments for panels
left_groups = ["terrain_settings", "biome_drag_settings"]
center_groups = ["biome_intrusion_settings", "global_toggles", "global_seeds"]
right_groups = ["image_pipeline"]

# Load images
images = {
    image.stem: ImageTk.PhotoImage(Image.open(image))
    for image in IMAGE_DIR.glob("*.png")
}

# Create UI elements for configuration
for category, sub_config in config.items():
    target_frame = (
        left_frame if category in left_groups
        else right_frame if category in right_groups
        else center_frame
    )
    frame = ttk.LabelFrame(target_frame, text=category.replace("_", " ").title())
    frame.pack(fill="x", padx=10, pady=10)

    for key, value in sub_config.items():
        if isinstance(value, bool):
            label_text = LABELS.get(key, key.replace("_", " ").title())
            var = tk.BooleanVar(value=value)
            checkbox_vars[key] = var
            checkbox = ttk.Checkbutton(frame, text=label_text, variable=var,
                                      command=lambda k=key, c=category: update_value(c, k, checkbox_vars[k].get()))
            checkbox.pack(fill="x", padx=5, pady=5)
            continue

        elif isinstance(value, (int, float)):
            subframe = ttk.Frame(frame)
            subframe.pack(fill="x", padx=5, pady=5)

            label_text = LABELS.get(key, key.replace("_", " ").title())
            label = ttk.Label(subframe, text=label_text)
            label.pack(side="left")

            label.bind("<Enter>", lambda e, t=label_text: show_image(t))
            label.bind("<Leave>", hide_image)

            value_label = ttk.Label(subframe, text=f"{value:.2f}", width=6)
            value_label.pack(side="right", padx=5)

            min_val = 0.01
            if "strength" in key or "lat_weight_factor" in key:
                min_val, max_val = 0.01, 2
            elif "squircle" in key or "octaves" in key or "smoothness" in key:
                min_val, max_val = 1, 4
            elif "drags" in key:
                min_val, max_val = 1, 20
            elif "x_min" in key or "y_min" in key or "crater_depth_min" in key:
                min_val, max_val = -100, -0.01
            elif "x_max" in key or "y_max" in key or "crater_depth_max" in key or "drag_radius" in key:
                min_val, max_val = 0.01, 100
            elif "max_radius" in key:
                min_val, max_val = 6, 100
            elif "factor" in key or (value is not None and value <= 1):
                max_val = 1
            else:
                min_val, max_val = 1, 100

            slider = ttk.Scale(subframe, from_=min_val, to=max_val, orient="horizontal",
                              command=lambda val, k=key, c=category, lbl=value_label: (
                                  update_value(c, k, val),
                                  update_slider_label(val, lbl)
                              ))
            slider.set(value)
            slider.pack(side="right")
            slider_vars[key] = slider

# Configure button styles
style = ttk.Style()
style.configure("Red.TButton", background="red")
style.configure("Blue.TButton", background="blue", font=("Arial", 24))
style.configure("Green.TButton", background="green", font=("Arial", 24))

# Image display placeholder
default_image = ImageTk.PhotoImage(Image.open(DEFAULT_IMAGE_PATH)) if DEFAULT_IMAGE_PATH.exists() else None
image_label = ttk.Label(center_frame, image=default_image)
image_label.pack(side="top", anchor="center", pady=15)

# Button frame for center panel
button_frame = ttk.Frame(center_frame)
button_frame.pack(fill="x", padx=10, pady=10)

reset_button = ttk.Button(button_frame, text="Reset to Defaults", style="Blue.TButton", command=reset_to_defaults)
reset_button.pack(fill="x", pady=10)

cancel_button = ttk.Button(button_frame, text="Cancel and Exit", style="Red.TButton", command=cancel_and_exit)
cancel_button.pack(fill="x", pady=5)

# Button frame for right panel
button_frame_right = ttk.Frame(right_frame)
button_frame_right.pack(fill="x", padx=10, pady=10)

save_button = ttk.Button(button_frame_right, text="Save and Continue", style="Green.TButton", command=save_and_continue)
save_button.pack(fill="x", pady=5)

# Start the UI event loop
root.mainloop()