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

PROCESSING_MAP = {
    "Biome processing complete.": [1],
    "Texture processing complete.": [2],
    "Materials processing complete.": [3],
}

# Global declarations
config = {}
checkbox_vars = {}
slider_vars = {}
planet_biomes_process = None
process_list = []


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
            "enable_edge_blending": False,
            "upscale_image": False,
            "keep_pngs_after_conversion": False,
            "output_csv_files": False,
            "output_dds_files": False,
            "output_mat_files": False,
            "output_biom_files": False,
            "zone_00": 0.5,
            "zone_01": 0.5,
            "zone_02": 0.5,
            "zone_03": 0.5,
            "zone_04": 0.5,
            "zone_05": 0.5,
            "zone_06": 0.5,
            "light_bias": "light_bias_cc",
            "biome_bias": "biome_bias_cc",
        }
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(raw_config, f, indent=4)

    for key, value in raw_config.items():
        if key in BOOLEAN_KEYS and isinstance(value, (float, int)):
            raw_config[key] = bool(int(value))
    config = raw_config


def save_config():
    """Save current configuration to JSON file."""
    if CONFIG_PATH.exists():
        os.remove(CONFIG_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def update_value(key, val, index=None):
    """Update configuration value and save to file."""
    if key not in config:
        print(f"Warning: Key '{key}' not found in config.")
        return

    if key == "user_seed":
        config[key] = int(val)
    elif isinstance(config[key], bool):
        config[key] = bool(val)
    elif isinstance(config[key], (int, float)):
        try:
            config[key] = float(val) if isinstance(val, str) else val
        except ValueError:
            print(f"Error: Cannot convert '{val}' to float for '{key}'")
            return

    save_config()


def update_bias_selection(self, button_name, is_checked):
    """Update the config when a new radio button is selected and update UI."""
    if is_checked:
        # Determine if it's a light_bias or biome_bias button
        bias_type = (
            "light_bias" if button_name.startswith("light_bias_") else "biome_bias"
        )
        config[bias_type] = button_name
        # Clear redundant bias keys to avoid conflicts
        bias_keys = [
            "biome_bias_nw",
            "biome_bias_nn",
            "biome_bias_ne",
            "biome_bias_ww",
            "biome_bias_cc",
            "biome_bias_ee",
            "biome_bias_sw",
            "biome_bias_ss",
            "biome_bias_se",
            "light_bias_nw",
            "light_bias_nn",
            "light_bias_ne",
            "light_bias_ww",
            "light_bias_cc",
            "light_bias_ee",
            "light_bias_sw",
            "light_bias_ss",
            "light_bias_se",
        ]
        for key in bias_keys:
            if key in config:
                del config[key]
        save_config()
        print(f"{bias_type} Updated: {button_name}")

        # Update UI for the respective bias group
        group_prefix = "light_bias_" if bias_type == "light_bias" else "biome_bias_"
        for btn_name in bias_keys:
            if btn_name.startswith(group_prefix):
                btn = getattr(self, btn_name, None)
                if btn:
                    btn.setChecked(btn_name == button_name)


def start_planet_biomes(mode=""):
    """Start PlanetBiomes.py asynchronously and track the process."""
    global planet_biomes_process, process_list
    if not SCRIPT_PATH.exists():
        print(f"Error: PlanetBiomes.py not found at {SCRIPT_PATH}")
        return

    planet_biomes_process = QProcess()
    planet_biomes_process.setProgram("python")
    planet_biomes_process.setArguments([str(SCRIPT_PATH), mode])  # ✅ Pass mode arg
    planet_biomes_process.setWorkingDirectory(str(BASE_DIR))
    process_list.append(planet_biomes_process)

    process_list.append(planet_biomes_process)  # ✅ Track the process

    planet_biomes_process.finished.connect(
        lambda exit_code: print(f"PlanetBiomes.py finished with exit code {exit_code}")
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

    print(f"Starting PlanetBiomes.py at {SCRIPT_PATH} with mode {mode}")
    planet_biomes_process.start()


def cleanup_and_exit(exit_code=0):
    """Clean up background processes before exiting the application."""
    cancel_processing()  # ✅ Ensure processes are killed before exiting
    print("Application cleanup complete. Exiting now...")
    sys.exit(exit_code)


def cancel_processing():
    """Kill any running background processes without closing the app."""
    global process_list  # Assuming you have a list of active processes
    for process in process_list:
        try:
            process.terminate()  # Gracefully stop the process
            process.wait()  # Ensure it's fully closed
        except Exception as e:
            print(f"Error stopping process: {e}")

    process_list.clear()  # Remove references to killed processes
    print("Processing canceled, but the app remains open.")


def cancel_and_exit():
    """Terminate all processes and close the application."""
    cancel_processing()  # First, kill any running processes
    print("Exiting application...")
    sys.exit()


def reset_to_defaults(key):
    """Reset a single setting to its default using update_value() and update UI sliders."""
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            default_config = json.load(f)

        default_value = default_config.get(key)
        if default_value is not None:
            update_value(key, default_value)

            slider = slider_vars.get(key, None)
            if slider:
                slider.setValue(int(default_value * 100))

        else:
            print(f"Warning: {key} not found in defaults.")

    except FileNotFoundError:
        print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")


def disable_upscaling():
    """Disable upscaling in the config file."""
    try:
        with open(CONFIG_PATH, "r") as file:
            config_data = json.load(file)
        config_data["upscale_image"] = False
        with open(CONFIG_PATH, "w") as file:
            json.dump(config_data, file, indent=4)
    except Exception as e:
        print(f"Error disabling upscaling: {e}")


def generate_preview(main_window):
    """Start the preview script asynchronously."""
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

    def handle_error():
        error_output = process.readAllStandardError().data().decode()
        print(f"Script error: {error_output}")
        main_window.stderr_widget.appendPlainText(error_output)

    process.readyReadStandardOutput.connect(handle_output)
    process.readyReadStandardError.connect(handle_error)
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
        lambda: (
            main_window.stderr_widget.appendPlainText(
                process.readAllStandardError().data().decode()
            ),
            main_window.stderr_widget.repaint(),
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
        self.preview_command_button.clicked.connect(
            lambda: (save_config(), start_planet_biomes("--preview"))
        )
        self.halt_command_button.clicked.connect(cancel_processing)
        self.exit_command_button.clicked.connect(cancel_and_exit)
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
            "enable_noise": "enable_noise",
            "apply_distortion": "apply_distortion",
            "enable_biases": "enable_biases",
            "enable_anomalies": "enable_anomalies",
            "use_random": "use_random",
            "enable_equator_drag": "enable_equator_drag",
            "enable_pole_drag": "enable_pole_drag",
            "enable_texture_light": "enable_texture_light",
            "enable_texture_edges": "enable_texture_edges",
            "enable_basic_filters": "enable_basic_filters",
            "enable_texture_anomolies": "enable_texture_anomolies",
            "process_images": "process_images",
            "enable_texture_noise": "enable_texture_noise",
            "upscale_image": "upscale_image",
            "enable_texture_preview": "enable_texture_preview",
            "output_csv_files": "output_csv_files",
            "output_dds_files": "output_dds_files",
            "keep_pngs_after_conversion": "keep_pngs_after_conversion",
            "output_mat_files": "output_mat_files",
            "output_biom_files": "output_biom_files",
            "enable_random_drag": "enable_random_drag",
            "random_distortion": "random_distortion",
        }

        slider_mappings = {
            "zoom": "zoom",
            "squircle_exponent": "squircle_exponent",
            "noise_factor": "noise_factor",
            "noise_scale": "noise_scale",
            "noise_amplitude": "noise_amplitude",
            "user_seed": "user_seed",
            "brightness_factor": "brightness_factor",
            "saturation_factor": "saturation_factor",
            "edge_blend_radius": "edge_blend_radius",
            "detail_smoothness": "detail_smoothness",
            "detail_strength_decay": "detail_strength_decay",
            "normal_strength": "normal_strength",
            "roughness_base": "roughness_base",
            "roughness_noise_scale": "roughness_noise_scale",
            "fade_intensity": "fade_intensity",
            "fade_spread": "fade_spread",
            "perlin_noise": "perlin_noise",
            "swap_factor": "swap_factor",
            "fractal_octaves": "fractal_octaves",
            "tint_factor": "tint_factor",
            "equator_anomoly_count": "equator_anomoly_count",
            "equator_anomoly_scale": "equator_anomoly_scale",
            "polar_anomoly_count": "polar_anomoly_count",
            "polar_anomoly_scale": "polar_anomoly_scale",
            "distortion_scale": "distortion_scale",
            "distortion_influence": "distortion_influence",
            "biome_distortion": "biome_distortion",
            "noise_scatter": "noise_scatter",
            "biome_perlin": "biome_perlin",
            "biome_swap": "biome_swap",
            "biom_fractal": "biom_fractal",
            "zone_00": "zone_00",
            "zone_01": "zone_01",
            "zone_02": "zone_02",
            "zone_03": "zone_03",
            "zone_04": "zone_04",
            "zone_05": "zone_05",
            "zone_06": "zone_06",
        }

        # Load saved bias values
        saved_light_bias = config.get("light_bias", "light_bias_cc")
        saved_biome_bias = config.get("biome_bias", "biome_bias_cc")

        bias_buttons = [
            "biome_bias_nw",
            "biome_bias_nn",
            "biome_bias_ne",
            "biome_bias_ww",
            "biome_bias_cc",
            "biome_bias_ee",
            "biome_bias_sw",
            "biome_bias_ss",
            "biome_bias_se",
            "light_bias_nw",
            "light_bias_nn",
            "light_bias_ne",
            "light_bias_ww",
            "light_bias_cc",
            "light_bias_ee",
            "light_bias_sw",
            "light_bias_ss",
            "light_bias_se",
        ]

        for checkbox_name, key in checkbox_mappings.items():
            checkbox = getattr(self, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(config.get(key, False))
                checkbox.toggled.connect(lambda val, k=key: update_value(k, val))
                checkbox_vars[key] = checkbox

        for slider_name, key in slider_mappings.items():
            slider = getattr(self, slider_name, None)
            if slider:
                value = config.get(key, 0)
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
                    lambda val, k=key: update_value(k, val / 100)
                )
                slider_vars[key] = slider

        # Set up bias buttons
        for button_name in bias_buttons:
            button = getattr(self, button_name, None)
            if button:
                # Set initial state based on config
                if button_name.startswith("light_bias_"):
                    button.setChecked(button_name == saved_light_bias)
                else:
                    button.setChecked(button_name == saved_biome_bias)
                # Connect toggled signal
                button.toggled.connect(
                    lambda checked, name=button_name: update_bias_selection(
                        self, name, checked
                    )
                )

        # Connect reset buttons
        reset_buttons = [
            "zoom_reset",
            "squircle_exponent_reset",
            "noise_factor_reset",
            "noise_scale_reset",
            "noise_amplitude_reset",
            "noise_scatter_reset",
            "biome_prelin_reset",
            "biome_swap_reset",
            "biome_fractal_reset",
            "brightness_factor_reset",
            "saturation_factor_reset",
            "tint_factor_reset",
            "detail_smoothness_reset",
            "detail_strength_decay_reset",
            "edge_blend_radius_reset",
            "perlin_noise_reset",
            "swap_factor_reset",
            "fractal_octaves_reset",
            "roughness_noise_scale_reset",
            "roughness_base_reset",
            "normal_strength_reset",
            "fade_intensity_reset",
            "fade_spread_reset",
            "equator_bias_reset",
            "pole_bias_reset",
            "balanced_bias_reset",
            "anomoly_count_reset",
            "anomoly_scale_reset",
            "distortion_influence_reset",
            "distortion_scale_reset",
            "biome_distortion_reset",
        ]
        for button_name in reset_buttons:
            button = getattr(self, button_name, None)
            key = slider_mappings.get(button_name.replace("_reset", ""), None)
            if button and key:
                button.clicked.connect(lambda _, k=key: reset_to_defaults(k))

        # Setup seed display
        self.seed_display.display(config["user_seed"])
        self.user_seed.valueChanged.connect(
            lambda val: (
                update_value("user_seed", int(val)),
                self.seed_display.display(int(val)),
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
        save_config()
        start_processing(self)
        start_planet_biomes()

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
