#!/usr/bin/env python3
"""
Biome Config Editor

A PyQt6-based GUI application for editing biome configuration settings.
Uses a .ui file for the interface and supports loading/saving JSON configs and running PlanetBiomes.py.

Dependencies:
- Python 3.8+
- PyQt6
- Pillow (PIL)
- subprocess
- json
- pathlib
"""

from pathlib import Path
import random
import sys
import os
import json
import subprocess
from PyQt6.uic import loadUi
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplashScreen
from PyQt6.QtCore import QTimer, QProcess, Qt
from PyQt6.QtGui import QPixmap, QFont, QMovie
from themes import THEMES

# Directory paths
BASE_DIR = (
    Path(sys._MEIPASS).resolve()
    if hasattr(sys, "_MEIPASS")
    else Path(__file__).parent.parent.resolve()
)
CONFIG_DIR = BASE_DIR / "config"
IMAGE_DIR = BASE_DIR / "assets" / "images"
PNG_OUTPUT_DIR = BASE_DIR / "Output" / "Textures"

# File Paths
UI_PATH = Path(__file__).parent / "mainwindow.ui"
SCRIPT_PATH = BASE_DIR / "src" / "PlanetBiomes.py"
PREVIEW_BIOME_PATH = BASE_DIR / "assets" / "PlanetBiomes.biom"
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "default.png"
GIF_PATHS = {
    1: IMAGE_DIR / "progress_1.gif",
    2: IMAGE_DIR / "progress_2.gif",
    3: IMAGE_DIR / "progress_3.gif",
}

# Image files for display
IMAGE_FILES = [
    "preview_North_albedo.png",
    "preview_North_normal.png",
    "preview_North_rough.png",
    "preview_North_alpha.png",
]

# Configuration keys
BOOLEAN_KEYS = {
    "enable_equator_drag",
    "enable_pole_drag",
    "enable_equator_intrusion",
    "enable_pole_intrusion",
    "apply_distortion",
    "apply_resource_gradient",
    "apply_latitude_blending",
    "keep_pngs_after_conversion",
    "enable_noise",
    "enable_anomalies",
    "enable_biases",
    "use_random",
    "enable_texture_light",
    "enable_texture_edges",
    "enable_basic_filters",
    "enable_texture_anomolies",
    "process_images",
    "enable_texture_noise",
    "upscale_image",
    "enable_texture_preview",
    "output_csv_files",
    "output_dds_files",
    "output_mat_files",
    "output_biom_files",
    "enable_random_drag",
    "random_distortion",
}

# Human-readable labels
LABELS = {
    "zoom": "Zoom",
    "squircle_exponent": "Diamond (1) Circle (2) Squircle (max)",
    "noise_factor": "Equator Weight",
    "global_seed": "Generation Seed",
    "noise_scale": "Anomaly Scale",
    "noise_amplitude": "Anomaly Distortion",
    "enable_equator_drag": "Enable Polar Anomalies",
    "enable_pole_drag": "Enable Equator Anomalies",
    "image_pipeline": "Image Settings",
    "brightness_factor": "Brightness",
    "saturation_factor": "Saturation",
    "enable_edge_blending": "Enable Edges",
    "edge_blend_radius": "Edge Detail",
    "distortion_sigma": "Fine Distortion",
    "lat_distortion_factor": "Large Distortion",
    "drag_radius": "Anomalies",
    "enable_equator_intrusion": "Enable Equator Intrusions",
    "enable_pole_intrusion": "Enable Pole Intrusions",
    "apply_distortion": "Apply Terrain Distortion",
    "apply_resource_gradient": "Use Resource Gradient",
    "apply_latitude_blending": "Blend Biomes by Latitude",
    "user_seed": "Seed",
    "elevation_scale": "Terrain Smoothness",
    "detail_smoothness": "Fractal Noise",
    "detail_strength_decay": "Fractal Decay",
    "normal_strength": "Normal Strength",
    "roughness_base": "Roughness Smoothness",
    "roughness_noise_scale": "Roughness Contrast",
    "alpha_base": "Alpha Base",
    "alpha_noise_scale": "Alpha Noise Scale",
    "fade_intensity": "Fade Intensity",
    "fade_spread": "Fade Spread",
    "perlin_noise": "Perlin Noise",
    "swap_factor": "Swap Factor",
    "fractal_octaves": "Fractal Octaves",
    "tint_factor": "Tint Factor",
    "equator_anomoly_count": "Equator Anomaly Count",
    "equator_anomoly_scale": "Equator Anomaly Scale",
    "polar_anomoly_count": "Polar Anomaly Count",
    "polar_anomoly_scale": "Polar Anomaly Scale",
    "distortion_scale": "Distortion Scale",
    "distortion_influence": "Distortion Influence",
    "biome_distortion": "Biome Distortion",
    "noise_scatter": "Noise Scatter",
    "biome_perlin": "Biome Perlin",
    "biome_swap": "Biome Swap",
    "biom_fractal": "Biome Fractal",
}

PROCESSING_MAP = {
    "Biome processing complete.": [1],
    "Texture processing complete.": [2],
    "Materials processing complete.": [3],
}

# Global variables
config = {}
checkbox_vars = {}
slider_vars = {}
planet_biomes_process = None

def load_config():
    """Load configuration from custom or default JSON file."""
    global config
    config_path = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            raw_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found. Creating default config.")
        raw_config = {
            "some_values": {
                "zoom": 1.0,
                "squircle_exponent": 2.0,
                "noise_factor": 0.5,
                "noise_scale": 0.1,
                "noise_amplitude": 0.1,
                "enable_equator_drag": False,
                "enable_pole_drag": False,
                "apply_distortion": False,
                "enable_noise": False,
                "enable_anomalies": False,
                "enable_biases": False,
                "brightness_factor": 1.0,
                "saturation_factor": 1.0,
                "edge_blend_radius": 0.5,
                "detail_smoothness": 0.5,
                "detail_strength_decay": 0.5,
                "normal_strength": 0.5,
                "roughness_base": 0.5,
                "roughness_noise_scale": 0.5,
                "fade_intensity": 0.5,
                "fade_spread": 0.5,
                "perlin_noise": 0.5,
                "swap_factor": 0.5,
                "fractal_octaves": 0.5,
                "tint_factor": 0.5,
                "equator_anomoly_count": 0.5,
                "equator_anomoly_scale": 0.5,
                "polar_anomoly_count": 0.5,
                "polar_anomoly_scale": 0.5,
                "distortion_scale": 0.5,
                "distortion_influence": 0.5,
                "biome_distortion": 0.5,
                "noise_scatter": 0.5,
                "biome_perlin": 0.5,
                "biome_swap": 0.5,
                "biom_fractal": 0.5,
                "enable_texture_light": False,
                "enable_texture_edges": False,
                "enable_basic_filters": False,
                "enable_texture_anomolies": False,
                "process_images": False,
                "enable_texture_noise": False,
                "enable_texture_preview": False,
                "enable_random_drag": False,
                "random_distortion": False,
            },
            "global_seed": {"user_seed": 12345, "use_random": False},
            "image_pipeline": {
                "enable_edge_blending": False,
                "upscale_image": False,
                "keep_pngs_after_conversion": False,
                "output_csv_files": False,
                "output_dds_files": False,
                "output_mat_files": False,
                "output_biom_files": False,
            },
            "biome_zones": {
                "zone_00": 0.5,
                "zone_01": 0.5,
                "zone_02": 0.5,
                "zone_03": 0.5,
                "zone_04": 0.5,
                "zone_05": 0.5,
                "zone_06": 0.5,
            },
        }
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(raw_config, f, indent=4)

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
        config[category][key] = bool(val)
    elif isinstance(config[category][key], (int, float)):
        config[category][key] = float(val) if isinstance(val, str) else val
    save_config()

def start_planet_biomes():
    """Start PlanetBiomes.py asynchronously."""
    global planet_biomes_process
    if not SCRIPT_PATH.exists():
        print(f"Error: PlanetBiomes.py not found at {SCRIPT_PATH}")
        return

    planet_biomes_process = QProcess()
    planet_biomes_process.setProgram("python")
    planet_biomes_process.setArguments([str(SCRIPT_PATH)])
    planet_biomes_process.setWorkingDirectory(str(BASE_DIR))

    planet_biomes_process.finished.connect(
        lambda exit_code: (
            print(f"PlanetBiomes.py finished with exit code {exit_code}"),
            cleanup_and_exit(exit_code),
        )
    )
    planet_biomes_process.errorOccurred.connect(
        lambda error: print(f"Error: {planet_biomes_process.errorString()}")
    )
    planet_biomes_process.readyReadStandardOutput.connect(
        lambda: print(
            f"Output: {planet_biomes_process.readAllStandardOutput().data().decode()}"
        )
    )
    planet_biomes_process.readyReadStandardError.connect(
        lambda: print(
            f"Error: {planet_biomes_process.readAllStandardError().data().decode()}"
        )
    )

    print(f"Starting PlanetBiomes.py at {SCRIPT_PATH}")
    planet_biomes_process.start()

def cleanup_and_exit(exit_code=0):
    """Clean up and exit the application."""
    global planet_biomes_process
    if planet_biomes_process and planet_biomes_process.state() != QProcess.ProcessState.NotRunning:
        planet_biomes_process.terminate()
        planet_biomes_process.waitForFinished(1000)
        if planet_biomes_process.state() != QProcess.ProcessState.NotRunning:
            planet_biomes_process.kill()
    planet_biomes_process = None
    sys.exit(exit_code)

def set_seed():
    """Update config with a random seed if 'use_random' is True."""
    seed_cfg = config.get("global_seed", {})
    if seed_cfg.get("use_random", False):
        seed_cfg["user_seed"] = random.randint(0, 99999)
        config["global_seed"] = seed_cfg


def reset_to_defaults(category, key):
    """Reset a single setting to its default using update_value()."""
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            default_config = json.load(f)

        default_value = default_config.get(category, {}).get(key)
        if default_value is not None:
            update_value(
                category, key, default_value
            )  # ✅ Ensure update_value is being called
        else:
            print(f"Warning: {category}/{key} not found in defaults.")

    except FileNotFoundError:
        print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")


def disable_upscaling():
    """Disable upscaling in the config file."""
    try:
        with open(CONFIG_PATH, "r") as file:
            config_data = json.load(file)
        config_data["image_pipeline"]["upscale_image"] = False
        with open(CONFIG_PATH, "w") as file:
            json.dump(config_data, file, indent=4)
    except Exception as e:
        print(f"Error disabling upscaling: {e}")

def generate_preview(main_window):
    """Start the preview script asynchronously."""
    set_seed()
    save_config()
    if (
        hasattr(main_window, "preview_process")
        and main_window.preview_process.state() != QProcess.ProcessState.NotRunning
    ):
        print("Preview is already running, please wait.")
        return

    if not SCRIPT_PATH.exists() or not PREVIEW_BIOME_PATH.exists():
        print(f"Error: Script or biome file not found.")
        main_window.preview_command_button.setEnabled(True)
        return

    disable_upscaling()
    start_processing(main_window)

    process = QProcess()
    main_window.preview_process = process
    main_window.preview_command_button.setEnabled(True)

    def handle_output():
        output = process.readAllStandardOutput().data().decode()
        print(f"Script output: {output}")
        main_window.stdout_widget.appendPlainText(output)
        for message, indices in PROCESSING_MAP.items():
            if message in output:
                for index in indices:
                    main_window.image_labels[index].setPixmap(
                        QPixmap(str(PNG_OUTPUT_DIR / IMAGE_FILES[index]))
                    )

    process.readyReadStandardOutput.connect(handle_output)
    process.finished.connect(main_window.refresh_images)
    process.setWorkingDirectory(str(BASE_DIR))
    process.start("python", [str(SCRIPT_PATH), str(PREVIEW_BIOME_PATH), "--preview"])
    if not process.waitForStarted(5000):
        print(f"Failed to start preview script: {process.errorString()}")
        main_window.preview_command_button.setEnabled(True)

def start_processing(main_window):
    """Start processing and display GIFs."""
    process = QProcess()
    main_window.processing_process = process
    for index in [1, 2, 3]:
        movie = QMovie(str(GIF_PATHS.get(index)))
        if movie.isValid():
            main_window.image_labels[index].setMovie(movie)
            movie.start()

    process.setProgram("python")
    process.setArguments([str(SCRIPT_PATH)])
    process.setWorkingDirectory(str(BASE_DIR))

    process.readyReadStandardOutput.connect(
        lambda: (
            main_window.stdout_widget.appendPlainText(
                process.readAllStandardOutput().data().decode()
            ),
            main_window.stdout_widget.repaint(),
        )
    )

    process.readyReadStandardError.connect(
        lambda: main_window.stdout_widget.appendPlainText(
            "Error: " + process.readAllStandardError().data().decode()
        )
    )
    process.start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(str(UI_PATH), self)
        self.setWindowTitle("Biome Config Editor")
        self.themes = THEMES
        self.image_labels = [
            self.albedo_preview_image,
            self.normal_preview_image,
            self.rough_preview_image,
            self.alpha_preview_image,
        ]
        self.default_image = (
            QPixmap(str(DEFAULT_IMAGE_PATH))
            if DEFAULT_IMAGE_PATH.exists()
            else QPixmap()
        )

        # Connect signals
        self.preview_command_button.clicked.connect(lambda: generate_preview(self))
        self.save_command_button.clicked.connect(self.save_and_continue)
        self.exit_command_button.clicked.connect(self.cancel_and_exit)
        self.reset_command_button.clicked.connect(reset_to_defaults)
        self.themes_dropdown.addItems(self.themes.keys())
        self.themes_dropdown.currentTextChanged.connect(self.change_theme)
        self.open_output_button.clicked.connect(self.open_output_folder)

        # Map checkboxes and sliders to config
        self.setup_config_controls()

        # Apply default theme
        self.change_theme("Light Sci-Fi")

    def setup_config_controls(self):
        """Map UI controls to configuration keys."""
        checkbox_mappings = {
            "enable_noise": ("some_values", "enable_noise"),
            "apply_distortion": ("some_values", "apply_distortion"),
            "enable_biases": ("some_values", "enable_biases"),
            "enable_anomalies": ("some_values", "enable_anomalies"),
            "use_random": ("global_seed", "use_random"),
            "enable_equator_drag": ("some_values", "enable_equator_drag"),
            "enable_pole_drag": ("some_values", "enable_pole_drag"),
            "enable_texture_light": ("some_values", "enable_texture_light"),
            "enable_texture_edges": ("some_values", "enable_texture_edges"),
            "enable_basic_filters": ("some_values", "enable_basic_filters"),
            "enable_texture_anomolies": ("some_values", "enable_texture_anomolies"),
            "process_images": ("some_values", "process_images"),
            "enable_texture_noise": ("some_values", "enable_texture_noise"),
            "upscale_image": ("image_pipeline", "upscale_image"),
            "enable_texture_preview": ("some_values", "enable_texture_preview"),
            "output_csv_files": ("image_pipeline", "output_csv_files"),
            "output_dds_files": ("image_pipeline", "output_dds_files"),
            "keep_pngs_after_conversion": ("image_pipeline", "keep_pngs_after_conversion"),
            "output_mat_files": ("image_pipeline", "output_mat_files"),
            "output_biom_files": ("image_pipeline", "output_biom_files"),
            "enable_random_drag": ("some_values", "enable_random_drag"),
            "random_distortion": ("some_values", "random_distortion"),
        }
        slider_mappings = {
            "zoom": ("some_values", "zoom"),
            "squircle_exponent": ("some_values", "squircle_exponent"),
            "noise_factor": ("some_values", "noise_factor"),
            "noise_scale": ("some_values", "noise_scale"),
            "noise_amplitude": ("some_values", "noise_amplitude"),
            "user_seed": ("global_seed", "user_seed"),
            "brightness_factor": ("some_values", "brightness_factor"),
            "saturation_factor": ("some_values", "saturation_factor"),
            "edge_blend_radius": ("some_values", "edge_blend_radius"),
            "detail_smoothness": ("some_values", "detail_smoothness"),
            "detail_strength_decay": ("some_values", "detail_strength_decay"),
            "normal_strength": ("some_values", "normal_strength"),
            "roughness_base": ("some_values", "roughness_base"),
            "roughness_noise_scale": ("some_values", "roughness_noise_scale"),
            "fade_intensity": ("some_values", "fade_intensity"),
            "fade_spread": ("some_values", "fade_spread"),
            "perlin_noise": ("some_values", "perlin_noise"),
            "swap_factor": ("some_values", "swap_factor"),
            "fractal_octaves": ("some_values", "fractal_octaves"),
            "tint_factor": ("some_values", "tint_factor"),
            "equator_anomoly_count": ("some_values", "equator_anomoly_count"),
            "equator_anomoly_scale": ("some_values", "equator_anomoly_scale"),
            "polar_anomoly_count": ("some_values", "polar_anomoly_count"),
            "polar_anomoly_scale": ("some_values", "polar_anomoly_scale"),
            "distortion_scale": ("some_values", "distortion_scale"),
            "distortion_influence": ("some_values", "distortion_influence"),
            "biome_distortion": ("some_values", "biome_distortion"),
            "noise_scatter": ("some_values", "noise_scatter"),
            "biome_perlin": ("some_values", "biome_perlin"),
            "biome_swap": ("some_values", "biome_swap"),
            "biom_fractal": ("some_values", "biom_fractal"),
            "zone_00": ("biome_zones", "zone_00"),
            "zone_01": ("biome_zones", "zone_01"),
            "zone_02": ("biome_zones", "zone_02"),
            "zone_03": ("biome_zones", "zone_03"),
            "zone_04": ("biome_zones", "zone_04"),
            "zone_05": ("biome_zones", "zone_05"),
            "zone_06": ("biome_zones", "zone_06"),
        }

        for checkbox_name, (category, key) in checkbox_mappings.items():
            checkbox = getattr(self, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(config.get(category, {}).get(key, False))
                checkbox.toggled.connect(
                    lambda val, c=category, k=key: update_value(c, k, val)
                )
                checkbox_vars[key] = checkbox

        for slider_name, (category, key) in slider_mappings.items():
            slider = getattr(self, slider_name, None)
            if slider:
                value = config.get(category, {}).get(key, 0)
                min_val, max_val = 0.01, 1.0
                if key == "user_seed":
                    min_val, max_val = 0, 99999
                elif key in ["zoom", "squircle_exponent"]:
                    max_val = 4
                elif key in ["noise_scale", "noise_amplitude", "noise_scatter"]:
                    max_val = 10
                elif key in ["equator_anomoly_count", "polar_anomoly_count"]:
                    min_val, max_val = 0, 10
                elif key in ["fractal_octaves"]:
                    min_val, max_val = 1, 8
                slider.setRange(int(min_val * 100), int(max_val * 100))
                slider.setValue(int(value * 100))
                slider.valueChanged.connect(
                    lambda val, c=category, k=key: update_value(c, k, val / 100)
                )
                slider_vars[key] = slider

        # Connect reset buttons
        reset_buttons = [
            "zoom_reset", "squircle_exponent_reset", "noise_factor_reset",
            "noise_scale_reset", "noise_amplitude_reset", "noise_scatter_reset",
            "biome_prelin_reset", "biome_swap_reset", "biome_fractal_reset",
            "brightness_factor_reset", "saturation_factor_reset", "tint_factor_reset",
            "detail_smoothness_reset", "detail_strength_decay_reset", "edge_blend_radius_reset",
            "perlin_noise_reset", "swap_factor_reset", "fractal_octaves_reset",
            "roughness_noise_scale_reset", "roughness_base_reset", "normal_strength_reset",
            "fade_intensity_reset", "fade_spread_reset",
            "equator_bias_reset", "pole_bias_reset", "balanced_bias_reset",
            "anomoly_count_reset", "anomoly_scale_reset",
            "distortion_influence_reset", "distortion_scale_reset", "biome_distortion_reset",
        ]
        for button_name in reset_buttons:
            button = getattr(self, button_name, None)
            if button:
                button.clicked.connect(
                    lambda _, c=category, k=key: reset_to_defaults(c, k)
                )

        # Setup seed display
        self.seed_display.display(config["global_seed"]["user_seed"])
        self.user_seed.valueChanged.connect(
            lambda val: (
                update_value("global_seed", "user_seed", val / 100),
                self.seed_display.display(val / 100)
            )
        )

    def change_theme(self, theme_name):
        """Apply the selected theme."""
        self.setStyleSheet(self.themes.get(theme_name, ""))

    def refresh_images(self):
        """Refresh image displays."""
        for i, image_file in enumerate(IMAGE_FILES):
            output_image = PNG_OUTPUT_DIR / image_file
            self.image_labels[i].setPixmap(
                QPixmap(str(output_image))
                if output_image.exists()
                else self.default_image
            )

    def save_and_continue(self):
        """Save config and start PlanetBiomes."""
        set_seed()
        save_config()
        start_processing(self)
        start_planet_biomes()

    def cancel_and_exit(self):
        """Terminate subprocess and exit."""
        cleanup_and_exit()

    def open_output_folder(self):
        """Open the output directory in the file explorer."""
        if PNG_OUTPUT_DIR.exists():
            if sys.platform == "win32":
                os.startfile(PNG_OUTPUT_DIR)
            elif sys.platform == "darwin":
                subprocess.run(["open", PNG_OUTPUT_DIR])
            else:
                subprocess.run(["xdg-open", PNG_OUTPUT_DIR])
        else:
            print(f"Output directory {PNG_OUTPUT_DIR} does not exist.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Orbitron", 10))
    splash = QSplashScreen(
        QPixmap(str(DEFAULT_IMAGE_PATH)) if DEFAULT_IMAGE_PATH.exists() else QPixmap()
    )
    splash.show()
    QTimer.singleShot(500, splash.close)

    load_config()
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
