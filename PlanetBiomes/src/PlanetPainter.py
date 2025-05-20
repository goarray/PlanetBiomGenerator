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

# Standard Libraries
import sys
import os
import csv
import json
import subprocess
from pathlib import Path

# Third Party Libraries
from typing import cast
import numpy as np
from PyQt6.uic.load_ui import loadUi
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplashScreen,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QComboBox,
    QLCDNumber,
    QSlider,
)
from PyQt6.QtCore import QTimer, QProcess, Qt
from PyQt6.QtGui import QPixmap, QFont, QMovie
from themes import THEMES
from PlanetConstants import (
    # Core directories
    BASE_DIR,
    CONFIG_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    PLUGINS_DIR,
    PNG_OUTPUT_DIR,
    # Config and data files
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    TEMP_DIR,
    CSV_DIR,
    PREVIEW_PATH,
    # Script and template paths
    SCRIPT_PATH,
    FOLDER_PATHS,
    PREVIEW_BIOME_PATH,
    # UI and static assets
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    IMAGE_FILES,
    # Logic/data maps
    BOOLEAN_KEYS,
    PROCESSING_MAP,
)

# Global declarations
config = {}
checkbox_vars = {}
slider_vars = {}
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
            "user_seed": 1234567,
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
            "equator_anomaly_count": 0.5,
            "equator_anomaly_scale": 0.5,
            "polar_anomaly_count": 0.5,
            "polar_anomaly_scale": 0.5,
            "distortion_scale": 0.5,
            "distortion_influence": 0.5,
            "biome_distortion": 0.5,
            "noise_scatter": 0.5,
            "biome_perlin": 0.5,
            "biome_swap": 0.5,
            "biome_fractal": 0.5,
            "enable_texture_light": False,
            "enable_texture_edges": False,
            "enable_basic_filters": False,
            "enable_texture_anomalies": False,
            "process_images": False,
            "enable_texture_noise": False,
            "enable_preview_mode": False,
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
    return config


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


def update_selected_plugin(index):
    if "plugin_index" not in config or not config["plugin_index"]:
        print("plugin_index missing or empty, restoring fallback list.")
        config["plugin_index"] = ["preview.csv"]
        config["plugin_selected"] = 0
    config["plugin_selected"] = index
    selected_csv = config["plugin_index"][index]  # Get selected CSV file name

    # ✅ Correctly set csv_path for preview.csv vs. user-provided files
    if selected_csv == "preview.csv":
        csv_path = CSV_DIR / selected_csv  # ✅ Use CSV_DIR for preview.csv
    else:
        csv_path = (
            INPUT_DIR / selected_csv
        )  # ✅ Use INPUT_DIR for user-selected plugins

    config["enable_preview_mode"] = selected_csv == "preview.csv"

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)  # Get first row
            if first_row and len(first_row) > 0:
                config["plugin_name"] = first_row[0].strip()  # First column
            else:
                config["plugin_name"] = "Unknown"
    except FileNotFoundError:
        print(f"Error: CSV file {csv_path} not found.")
        config["plugin_name"] = "Unknown"

    update_value("enable_preview_mode", config["enable_preview_mode"])

    save_config()  # Ensure changes are saved


def get_seed():
    """Retrieve user seed from `_config.json`."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        return int(config.get("user_seed", 0))
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading config. Using default seed: 0")
        return 0


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


# At the module level
planet_biomes_process: QProcess | None = None


def get_planet_biomes_process() -> QProcess:
    """Retrieve the planet_biomes_process, ensuring it's initialized."""
    global planet_biomes_process
    if planet_biomes_process is None:
        raise RuntimeError("planet_biomes_process is not initialized")
    return planet_biomes_process


def start_planet_biomes(main_window, mode=""):
    """Start PlanetBiomes.py asynchronously, handle modes, and update UI."""
    global planet_biomes_process, process_list
    main_window.stdout_widget.clear()
    main_window.stderr_widget.clear()

    if not SCRIPT_PATH.exists():
        main_window.stderr_widget.appendPlainText(
            f"Error: PlanetBiomes.py not found at {SCRIPT_PATH}"
        )
        return

    # Disable upscaling for preview mode
    if "--preview" in mode:
        disable_upscaling()

    # Save config to ensure latest settings are used
    save_config()

    # Initialize planet_biomes_process
    planet_biomes_process = QProcess()
    planet_biomes_process.setProgram(sys.executable)
    args = [str(SCRIPT_PATH)]
    print(f"Mode argument received: {mode}")
    if mode:
        if "--preview" in mode:
            args.extend([str(PREVIEW_BIOME_PATH), "--preview"])
        else:
            args.extend(mode.split())
    planet_biomes_process.setArguments(args)
    planet_biomes_process.setWorkingDirectory(str(BASE_DIR))

    process_list.append(planet_biomes_process)

    # Handle output for image updates
    def handle_output():
        if planet_biomes_process is None:
            main_window.stderr_widget.appendPlainText(
                "Error: planet_biomes_process is not initialized"
            )
            return
        output = planet_biomes_process.readAllStandardOutput().data().decode()
        main_window.stdout_widget.appendPlainText(output)
        updated = False

        for message, indices in PROCESSING_MAP.items():
            if message in output:
                main_window.stderr_widget.appendPlainText(message)
                for index in indices:
                    output_image = TEMP_DIR / IMAGE_FILES[index]
                    if output_image.exists():
                        pixmap = QPixmap(str(output_image)).scaled(
                            main_window.image_labels[index].width(),
                            main_window.image_labels[index].height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        )
                        # Stop any running GIF
                        if main_window.image_labels[index].movie():
                            main_window.image_labels[index].movie().stop()
                            main_window.image_labels[index].setMovie(None)
                        main_window.image_labels[index].setPixmap(pixmap)
                        updated = True

        # Fallback: refresh all images if a completion-like message is detected
        if not updated and "complete" in output.lower():
            for index in range(len(IMAGE_FILES)):
                output_image = TEMP_DIR / IMAGE_FILES[index]
                if output_image.exists():
                    pixmap = QPixmap(str(output_image)).scaled(
                        main_window.image_labels[index].width(),
                        main_window.image_labels[index].height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                    # Stop any running GIF
                    if main_window.image_labels[index].movie():
                        main_window.image_labels[index].movie().stop()
                        main_window.image_labels[index].setMovie(None)
                    main_window.image_labels[index].setPixmap(pixmap)

        if "Visual inspection" in output:
            output_dir = Path(OUTPUT_DIR)
            num_files = sum(1 for f in output_dir.rglob("*") if f.is_file())
            main_window.stdout_widget.appendPlainText(
                f"Forms filed correctly: {num_files}"
            )

    def handle_error():
        if planet_biomes_process is None:
            main_window.stderr_widget.appendPlainText(
                "Error: planet_biomes_process is not initialized"
            )
            return
        error_output = planet_biomes_process.readAllStandardError().data().decode()
        if error_output:
            main_window.stderr_widget.appendPlainText(f"{error_output}")

    # Connect signals
    planet_biomes_process.readyReadStandardOutput.connect(handle_output)
    planet_biomes_process.readyReadStandardError.connect(handle_error)
    seed = get_seed()

    planet_biomes_process.finished.connect(
        lambda exit_code: main_window.stdout_widget.appendPlainText(
            f"Permit {seed} closed, don't panic!"
            if exit_code == 0
            else f"Permit denied, code {exit_code}: Construction halted."
        )
    )
    planet_biomes_process.errorOccurred.connect(
        lambda error: main_window.stderr_widget.appendPlainText(
            f"Error: {planet_biomes_process.errorString() if planet_biomes_process else 'Process not initialized'}"
        )
    )

    # Start GIFs for processing, but only for labels without existing images
    for index in [1, 2, 3]:
        output_image = TEMP_DIR / IMAGE_FILES[index]
        if not output_image.exists():  # Only set GIF if no image exists
            movie = QMovie(str(GIF_PATHS.get(index)))
            if movie.isValid():
                main_window.image_labels[index].setMovie(movie)
                movie.start()

    main_window.stderr_widget.appendPlainText(
        f"Starting PlanetBiomes.py with args: {args}"
    )
    planet_biomes_process.start()


def cleanup_and_exit(exit_code=0):
    """Clean up background processes before exiting the application."""
    cancel_processing()
    print("Application cleanup complete. Exiting now...")
    sys.exit(exit_code)


def cancel_processing():
    """Kill any running background processes without closing the app."""
    global process_list
    for process in process_list:
        if process.state() != QProcess.ProcessState.NotRunning:
            process.terminate()
            process.waitForFinished(500)
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
    process_list.clear()
    main_window.refresh_images()
    print("Processing halted, but the app remains open.")


def cancel_and_exit():
    """Terminate all processes and close the application."""
    cancel_processing()
    print("Exiting application...")
    sys.exit()


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


def start_processing(main_window):
    """Start processing by calling start_planet_biomes."""
    start_planet_biomes(main_window)


class MainWindow(QMainWindow):

    albedo_preview_image: QLabel
    normal_preview_image: QLabel
    rough_preview_image: QLabel
    alpha_preview_image: QLabel
    stdout_widget: QPlainTextEdit
    stderr_widget: QPlainTextEdit
    preview_command_button: QPushButton
    halt_command_button: QPushButton
    exit_command_button: QPushButton
    reset_command_button: QPushButton
    themes_dropdown: QComboBox
    plugins_dropdown: QComboBox
    folders_dropdown: QComboBox
    open_plugins_button: QPushButton
    open_output_button: QPushButton
    open_input_button: QPushButton
    seed_display: QLCDNumber
    user_seed: QSlider

    def __init__(self):
        super().__init__()

        # Determine base directory based on whether we're running as a PyInstaller bundle
        if getattr(sys, "frozen", False):
            base_dir = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to mainwindow.ui
        loadUi(UI_PATH, self)
        self.slider_mappings = {}
        self.checkbox_mappings = {}

        self.setWindowTitle("Planet Painter")
        self.themes = THEMES
        config = load_config()

        csv_files = list(INPUT_DIR.glob("*.csv"))
        csv_names = [f.name for f in csv_files]

        # Always include preview.csv if it's not already listed
        if PREVIEW_PATH.name not in csv_names:
            csv_names.append(PREVIEW_PATH.name)

        config["plugin_index"] = csv_names  # Used by update_selected_plugin
        config["plugin_list"] = csv_names   # Optional duplicate (can unify later)

        print(f"DEBUG: plugin_index = {config['plugin_index']}")

        print(f"DEBUG: Available themes: {self.themes.keys()}")
        theme = config.get("theme", "Starfield")
        print(f"DEBUG: Loaded theme from config: {theme}")
        plugin_list = config.get("plugin_index", [])  # Get the list or default to empty
        self.plugins_dropdown.clear()
        self.plugins_dropdown.addItems(config["plugin_index"])
        self.plugins_dropdown.setCurrentIndex(config.get("plugin_selected", 0))
        self.plugins_dropdown.currentIndexChanged.connect(
            lambda idx: (update_selected_plugin(idx), self.refresh_ui_from_config())
        )
        print("DEBUG: Available plugins:")
        for index, plugin in enumerate(plugin_list):
            print(f"  [{index}] {plugin}")

        self.image_labels = [
            self.albedo_preview_image,
            self.normal_preview_image,
            self.rough_preview_image,
            self.alpha_preview_image,
        ]

        temp_images = {img.name: img for img in TEMP_DIR.glob("*.png")}  # Fast lookup
        self.default_image = (
            QPixmap(str(DEFAULT_IMAGE_PATH)).scaled(
                self.image_labels[0].width(),
                self.image_labels[0].height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            if DEFAULT_IMAGE_PATH.exists()
            else QPixmap()
        )

        # Set initial images for all labels
        for i, image_file in enumerate(IMAGE_FILES):
            image_path = TEMP_DIR / image_file  # Look in TEMP_DIR, not PNG_OUTPUT_DIR
            pixmap = (
                QPixmap(str(image_path)).scaled(
                    self.image_labels[i].width(),
                    self.image_labels[i].height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                if image_path.exists()
                else self.default_image
            )
            self.image_labels[i].setPixmap(pixmap)

        message = f"Available themes: {', '.join(self.themes.keys())}"
        self.stdout_widget.appendPlainText(message)

        # Connect signals
        self.preview_command_button.clicked.connect(
            lambda: start_planet_biomes(self, "--preview" if config.get("enable_preview_mode") else "")
        )
        self.halt_command_button.clicked.connect(cancel_processing)
        self.exit_command_button.clicked.connect(cancel_and_exit)
        self.reset_command_button.clicked.connect(self.reset_all_to_defaults)
        self.themes_dropdown.addItems(self.themes.keys())

        # Populate themes dropdown
        self.themes_dropdown.clear()  # Clear any existing items
        self.themes_dropdown.addItems(self.themes.keys())
        if not self.themes_dropdown.count():
            print("Error: No themes loaded into themes_dropdown")

        self.themes_dropdown.currentTextChanged.connect(self.change_theme)

        for label, path in FOLDER_PATHS.items():
            self.folders_dropdown.addItem(label, path)
        self.folders_dropdown.currentIndexChanged.connect(self.open_selected_folder)

        # Map checkboxes and sliders to config
        self.setup_config_controls()

        self.change_theme(theme)

        message = "Available plugins:\n"
        for index, plugin in enumerate(plugin_list):
            message += f"  [{index}] {plugin}\n"
        self.stderr_widget.appendPlainText(message)

    def open_selected_folder(self, index):
        folder_path = self.folders_dropdown.itemData(index)
        if folder_path:
            self.open_folder(folder_path)
        self.folders_dropdown.setCurrentIndex(0)

    def reset_to_defaults(self, key):
        """Reset a single setting to its default using update_value() and update UI sliders."""
        print(f"Resetting key: '{key}'")  # Debug print
        try:
            with open(DEFAULT_CONFIG_PATH, "r") as f:
                default_config = json.load(f)

            if not isinstance(key, str):
                print(f"Error: Invalid key '{key}', expected a string.")
                return

            default_value = default_config.get(key)
            if default_value is not None:
                update_value(key, default_value)

                # Update sliders
                slider = slider_vars.get(key)
                if slider:
                    slider.setValue(int(default_value * 100))

                # Update checkboxes
                checkbox = checkbox_vars.get(key)
                if checkbox and isinstance(default_value, bool):
                    checkbox.setChecked(default_value)

            else:
                print(f"Warning: '{key}' not found in defaults.")
        except FileNotFoundError:
            print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")

    def reset_all_to_defaults(self):
        print("Resetting all configuration settings to defaults")
        try:
            with open(DEFAULT_CONFIG_PATH, "r") as f:
                default_config = json.load(f)

            # Overwrite custom config file entirely
            with open(CONFIG_PATH, "w") as f:
                json.dump(default_config, f, indent=4)

            # Reload config in memory
            global config
            config = default_config

            # Re-apply UI control values
            for key in self.slider_mappings:
                slider = slider_vars.get(key)
                if slider:
                    value = config.get(key, 0)
                    slider.setValue(int(value * 100))

            for key in self.checkbox_mappings:
                checkbox = checkbox_vars.get(key)
                if checkbox:
                    checkbox.setChecked(config.get(key, False))

            # Set radio buttons for biases
            saved_light_bias = config.get("light_bias", "light_bias_cc")
            saved_biome_bias = config.get("biome_bias", "biome_bias_cc")
            for button_name in [
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
            ]:
                button = getattr(self, button_name, None)
                if button:
                    button.setChecked(
                        button_name == saved_light_bias
                        or button_name == saved_biome_bias
                    )

            # Update seed display
            self.seed_display.display(config.get("user_seed", 0))
            self.refresh_plugin_list()
            plugin_name = config.get("plugin_name", "preview.csv")
            if "plugin_index" in config and plugin_name in config["plugin_index"]:
                config["plugin_selected"] = config["plugin_index"].index(plugin_name)
            else:
                config["plugin_selected"] = -1

            self.plugins_dropdown.setCurrentIndex(config["plugin_selected"])

            self.refresh_ui_from_config()
            save_config()

        except FileNotFoundError:
            print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")

    def refresh_plugin_list(self):
        """Scan for CSVs and update plugin list in config and dropdown."""
        global config

        csv_files = list(INPUT_DIR.glob("*.csv"))
        csv_names = [f.name for f in csv_files]

        # Always include preview.csv if not already
        if PREVIEW_PATH.name not in csv_names:
            csv_names.append(PREVIEW_PATH.name)

        config["plugin_index"] = csv_names
        config["plugin_list"] = csv_names  # Optional, depending on what you're using

        # Reset selected plugin
        config["plugin_selected"] = 0
        config["plugin_name"] = csv_names[0].replace(".csv", ".esm")
        config["enable_preview_mode"] = csv_names[0] == PREVIEW_PATH.name

        # Refresh dropdown UI
        self.plugins_dropdown.blockSignals(True)
        self.plugins_dropdown.clear()
        self.plugins_dropdown.addItems(csv_names)
        self.plugins_dropdown.setCurrentIndex(0)
        self.plugins_dropdown.blockSignals(False)

    def open_folder(self, directory):
        """Open the specified directory in the file explorer."""
        if directory.exists():
            if sys.platform == "win32":
                os.startfile(directory)
            elif sys.platform == "darwin":
                subprocess.run(["open", directory])
            else:
                subprocess.run(["xdg-open", directory])
        else:
            print(f"Error: Directory {directory} does not exist.")

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
            "enable_texture_anomalies": "enable_texture_anomalies",
            "process_images": "process_images",
            "enable_texture_noise": "enable_texture_noise",
            "upscale_image": "upscale_image",
            "enable_preview_mode": "enable_preview_mode",
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
            "equator_anomaly_count": "equator_anomaly_count",
            "equator_anomaly_scale": "equator_anomaly_scale",
            "polar_anomaly_count": "polar_anomaly_count",
            "polar_anomaly_scale": "polar_anomaly_scale",
            "distortion_scale": "distortion_scale",
            "distortion_influence": "distortion_influence",
            "biome_distortion": "biome_distortion",
            "noise_scatter": "noise_scatter",
            "biome_perlin": "biome_perlin",
            "biome_swap": "biome_swap",
            "biome_fractal": "biome_fractal",
            "zone_00": "zone_00",
            "zone_01": "zone_01",
            "zone_02": "zone_02",
            "zone_03": "zone_03",
            "zone_04": "zone_04",
            "zone_05": "zone_05",
            "zone_06": "zone_06",
        }

        reset_buttons = [
            "zoom_reset",
            "squircle_exponent_reset",
            "noise_factor_reset",
            "noise_scale_reset",
            "noise_amplitude_reset",
            "noise_scatter_reset",
            "biome_perlin_reset",  # Fixed typo from "biome_prelin_reset"
            "biome_swap_reset",
            "biome_fractal_reset",  # Fixed typo from "biome_fractal_reset"
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
            "distortion_influence_reset",
            "distortion_scale_reset",
            "biome_distortion_reset",
        ]

        # Custom reset functions for grouped settings
        def reset_anomaly_counts():
            print("Resetting anomaly counts")
            self.reset_to_defaults("equator_anomaly_count")
            self.reset_to_defaults("polar_anomaly_count")

        def reset_anomaly_scales():
            print("Resetting anomaly scales")
            self.reset_to_defaults("equator_anomaly_scale")
            self.reset_to_defaults("polar_anomaly_scale")

        def reset_equator_bias():
            print("Resetting equator bias")
            for zone in ["zone_00", "zone_01", "zone_02"]:  # Adjust zones as needed
                self.reset_to_defaults(zone)

        def reset_pole_bias():
            print("Resetting pole bias")
            for zone in ["zone_03", "zone_04"]:  # Adjust zones as needed
                self.reset_to_defaults(zone)

        def reset_balanced_bias():
            print("Resetting balanced bias")
            for zone in ["zone_05", "zone_06"]:  # Adjust zones as needed
                self.reset_to_defaults(zone)

        # Connect checkboxes
        for checkbox_name, key in checkbox_mappings.items():
            checkbox = getattr(self, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(config.get(key, False))
                checkbox.toggled.connect(lambda val, k=key: update_value(k, val))
                checkbox_vars[key] = checkbox
            else:
                print(f"Warning: Checkbox '{checkbox_name}' not found in UI")

        # Connect sliders
        for slider_name, key in slider_mappings.items():
            slider = getattr(self, slider_name, None)
            if slider:
                value = config.get(key, 0)
                min_val, max_val = 0.01, 1.0
                if key == "user_seed":
                    min_val, max_val = 0, 99999
                elif key in ["squircle_exponent"]:
                    max_val = 4
                elif key in ["noise_scale", "noise_amplitude", "noise_scatter"]:
                    max_val = 10
                elif key in ["equator_anomaly_count", "polar_anomaly_count"]:
                    min_val, max_val = 0, 10
                elif key in ["fractal_octaves"]:
                    min_val, max_val = 1, 8
                slider.setRange(int(min_val * 100), int(max_val * 100))
                slider.setValue(int(value * 100))
                slider.valueChanged.connect(
                    lambda val, k=key: update_value(k, val / 100)
                )
                slider_vars[key] = slider
            else:
                print(f"Warning: Slider '{slider_name}' not found in UI")

        # Connect reset buttons
        for button_name in reset_buttons:
            button = getattr(self, button_name, None)
            key = slider_mappings.get(button_name.replace("_reset", ""), None)
            if button and key:
                button.clicked.connect(
                    lambda checked=False, k=key: self.reset_to_defaults(k)
                )
            else:
                print(
                    f"Error: Button '{button_name}' or key '{key}' not found in UI or mappings"
                )

        # Connect special reset buttons
        special_reset_buttons = {
            "anomaly_count_reset": reset_anomaly_counts,
            "anomaly_scale_reset": reset_anomaly_scales,
            "equator_bias_reset": reset_equator_bias,
            "pole_bias_reset": reset_pole_bias,
            "balanced_bias_reset": reset_balanced_bias,
        }
        for button_name, reset_func in special_reset_buttons.items():
            button = getattr(self, button_name, None)
            if button:
                button.clicked.connect(reset_func)
            else:
                print(f"Error: Special reset button '{button_name}' not found in UI")

        # Connect bias buttons
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
        for button_name in bias_buttons:
            button = getattr(self, button_name, None)
            if button:
                if button_name.startswith("light_bias_"):
                    button.setChecked(button_name == saved_light_bias)
                else:
                    button.setChecked(button_name == saved_biome_bias)
                button.toggled.connect(
                    lambda checked, name=button_name: update_bias_selection(
                        self, name, checked
                    )
                )
            else:
                print(f"Warning: Bias button '{button_name}' not found in UI")

        # Setup seed display
        self.seed_display.display(config["user_seed"])
        self.user_seed.valueChanged.connect(
            lambda val: (
                update_value("user_seed", int(val)),
                self.seed_display.display(int(val)),
            )
        )

    def change_theme(self, theme_name):
        """Apply the selected theme, update stdout, and save to config."""

        if theme_name in self.themes:
            self.setStyleSheet(self.themes[theme_name])
            message = f"Applied theme: {theme_name}"

            # Save selected theme to config
            config["theme"] = theme_name
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=4)

        else:
            self.setStyleSheet(self.themes.get("Starfield", ""))
            message = f"Error: Theme '{theme_name}' not found in themes"

        self.stdout_widget.appendPlainText(message)

    def refresh_images(self):
        for i, image_file in enumerate(IMAGE_FILES):
            output_image = TEMP_DIR / image_file  # Use TEMP_DIR
            pixmap = (
                QPixmap(str(output_image))
                if output_image.exists()
                else QPixmap(str(DEFAULT_IMAGE_PATH))
            )
            # Clear any running GIFs
            movie = self.image_labels[i].movie()
            if movie is not None:
                movie.stop()
            self.image_labels[i].setMovie(None)
            self.image_labels[i].setPixmap(
                pixmap.scaled(
                    self.image_labels[i].width(),
                    self.image_labels[i].height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )

    def refresh_ui_from_config(self):
        """Refresh the entire UI to reflect current config values."""
        for key, slider in slider_vars.items():
            if slider and key in config:
                slider.setValue(int(config[key] * 100))

        for key, checkbox in checkbox_vars.items():
            if checkbox and key in config:
                checkbox.setChecked(config[key])

        # Update seed display
        self.seed_display.display(config.get("user_seed", 0))

        # Update bias radio buttons
        light_bias = config.get("light_bias", "light_bias_cc")
        biome_bias = config.get("biome_bias", "biome_bias_cc")

        for name in [
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
        ]:
            button = getattr(self, name, None)
            if button:
                button.setChecked(
                    name == light_bias
                    if name.startswith("light_bias_")
                    else name == biome_bias
                )

    def save_and_continue(self):
        """Save config and start PlanetBiomes."""
        save_config()
        start_planet_biomes(self)


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
