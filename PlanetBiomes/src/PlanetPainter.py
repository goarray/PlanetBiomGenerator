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
import random
import subprocess
from pathlib import Path
from typing import Dict

# Third Party Libraries
from pyvistaqt import QtInteractor
from typing import cast
import numpy as np
from PyQt6.uic.load_ui import loadUi
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplashScreen,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QLCDNumber,
    QSlider,
    QProgressBar,
    QFrame,
    QVBoxLayout,
    QRadioButton,
    QCheckBox,
)
from PyQt6.QtCore import QTimer, QProcess, Qt
from PyQt6.QtGui import QPixmap, QFont, QMovie, QTextCursor
from PlanetThemes import THEMES
from PlanetPlotter import (
    generate_sphere,
    auto_connect_enable_buttons,
    handle_enable_view,
    refresh_mesh_opacity,
)
from PlanetNewsfeed import (
    handle_news,
    news_count,
    news_percent,
    biom_percent,
    text_percent,
    total_news,
    total_biom,
    total_text,
    total_other,
    reset_news_count,
    precompute_total_news,
    load_global_config,
)
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
    TEMPLATE_PATH,
    # UI and static assets
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    IMAGE_FILES,
    # Logic/data maps
    BOOLEAN_KEYS,
    PROCESSING_MAP,
    SPHERE_PATH
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
        handle_news(
            None, "error", f"Error: Config file {config_path} not found. Creating default config."
        )
        raw_config = {
            "total_news": 0,
            "process_biomes": False,
            "_program_options": "Pluging related options",
            "plugin_selected": 0,
            "plugin_index": ["preview.csv"],
            "plugin_name": "preview.esm",
            "enable_preview_mode": False,
            "_theme_group": "Parameters related to themes",
            "theme": "Starfield",
            "_biome_toggles_group": "Biome-related options",
            "enable_noise": False,
            "enable_distortion": False,
            "enable_biases": False,
            "enable_anomalies": False,
            "_generation_seed_group": "Random seed settings",
            "use_random": False,
            "user_seed": 99999,
            "_biome_basic_group": "Basic biome parameters",
            "zoom_factor": 0.5,
            "enable_tectonic_plates": False,
            "number_faults": 8,
            "fault_width": 8,
            "fault_jitter": 0.5,
            "fault_smooth": 0.5,
            "distort_scale": 5,
            "save_tectonic_maps": True,
            "squircle_factor": 0.5,
            "_biome_noise_group": "Noise configuration",
            "noise_scale": 0.5,
            "noise_amplitude": 0.5,
            "noise_scatter": 0.5,
            "biome_perlin": 0.5,
            "biome_swap": 0.5,
            "biome_fractal": 0.5,
            "_biome_bias_group": "Bias settings for different zones",
            "set_equator_bias": "Button used to adjust to an equator bias",
            "set_balanced_bias": "Button used to adjust to an equator bias",
            "set_polar_bias": "Button used to adjust to an equator bias",
            "zone_00": 0.5,
            "zone_01": 0.5,
            "zone_02": 0.5,
            "zone_03": 0.5,
            "zone_04": 0.5,
            "zone_05": 0.5,
            "zone_06": 0.5,
            "_biome_distortion_group": "Distortion settings",
            "random_distortion": False,
            "distortion_scale": 0.5,
            "biome_bias": "biome_bias_cc",
            "_biome_anomaly_group": "Anomaly settings",
            "enable_equator_anomalies": True,
            "enable_seed_anomalies": False,
            "enable_polar_anomalies": True,
            "equator_anomaly_count": 0.5,
            "equator_anomaly_spray": 0.5,
            "polar_anomaly_count": 0.5,
            "polar_anomaly_spray": 0.5,
            "_file_toggles_group": "File output settings",
            "output_biom_files": False,
            "keep_pngs_after_conversion": True,
            "output_dds_files": False,
            "output_mat_files": False,
            "output_csv_files": False,
            "enable_output_log": False,
            "_texture_toggles_group": "Texture processing toggles",
            "upscale_image": True,
            "texture_resolution_scale": 8,
            "texture_resolution": 2048,
            "process_images": False,
            "enable_texture_light": True,
            "enable_basic_filters": False,
            "enable_texture_edges": False,
            "enable_texture_noise": False,
            "enable_texture_terrain": False,
            "_texture_basics_group": "Texture basics",
            "texture_brightness": 0.5,
            "texture_saturation": 0.5,
            "texture_edges": 0.5,
            "texture_contrast": 0.5,
            "texture_tint": 0.5,
            "_texture_noise_group": "Noise parameters for textures",
            "texture_roughness": 0.5,
            "texture_roughness_base": 0.5,
            "texture_noise": 0.5,
            "texture_perlin": 0.5,
            "texture_swap": 0.5,
            "texture_fractal": 0.5,
            "_texture_lighting_group": "Lighting settings",
            "fade_intensity": 0.5,
            "fade_spread": 0.5,
            "Texture Terrain Settings": "Texture Terrain Settings",
            "texture_mountains": 0.5,
            "texture_canyons": 0.5,
            "3D Viewport Settings": "3D Viewport Settings",
            "enable_surface_metal_view": True,
            "enable_color_view": True,
            "enable_fault_view": True,
            "enable_resource_view": True,
            "enable_biome_view": True,
            "enable_rough_view": True,
            "enable_normal_view": True,
            "enable_ao_view": True,
            "enable_ocean_mask_view": True,
            "surface_metal_opacity": 1.0,
            "color_opacity": 1.0,
            "fault_opacity": 0.2,
            "resource_opacity": 0.1,
            "biome_opacity": 0.4,
            "rough_opacity": 0.2,
            "normal_opacity": 0.2,
            "ao_opacity": 0.2,
            "ocean_mask_opacity": 1.0,
            "_image_options_group": "Other image options",
            "enable_normal": True,
            "enable_rough": False,
            "enable_alpha": False,
            # "plugin_list": ["PlanetBiomes.csv", "preview.csv"],
            "elevation_influence": 0.5,
        }
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(raw_config, f, indent=4)

    for key, value in raw_config.items():
        if key in BOOLEAN_KEYS and isinstance(value, (float, int)):
            raw_config[key] = bool(int(value))
    config = raw_config

    # Ensure plugin settings are initialized
    if "plugin_selected" not in config:
        config["plugin_selected"] = 0
    if "plugin_index" not in config or not config["plugin_index"]:
        config["plugin_index"] = ["preview.csv"]
    if "plugin_name" not in config:
        config["plugin_name"] = "preview.esm"
    # if "plugin_list" not in config:
    #   config["plugin_list"] = ["preview.csv"]

    return config


def save_config():
    """Save current configuration to JSON file with error handling."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
        global total_news
        total_news = config.get("total_news", 37)
    except Exception as e:
        handle_news(
            None, "error",
            f"Error saving JSON: {e}"
        )


def update_value(key, val, index=None, plotter=None, meshes=None):
    if key not in config:
        print(f"Warning: Key '{key}' not found in config.")
        return
    elif (
        key == "user_seed"
        or key == "texture_resolution_scale"
        or key == "number_faults"
        or key == "fault_width"
    ):
        config[key] = int(val)
    elif isinstance(config[key], bool):
        config[key] = bool(val)
    elif isinstance(config[key], (int, float)):
        try:
            config[key] = float(val) if isinstance(val, str) else val
        except ValueError:
            print(f"Error: Cannot convert '{val}' to float for '{key}'")
            return

    # Update opacity on change
    if key.endswith("_opacity") and plotter and meshes:
        texture_type = key.replace("_opacity", "")
        refresh_mesh_opacity(texture_type, plotter, meshes)

    save_config()


def update_selected_plugin(index, main_window, force=False):
    """Update the selected plugin in config and UI when the user selects a plugin."""
    if "plugin_index" not in config or not config["plugin_index"]:
        handle_news(
            None, "error", "plugin_index missing or empty, restoring fallback list."
        )
        config["plugin_index"] = ["preview.csv"]
        #config["plugin_list"] = ["preview.csv"]
        config["plugin_selected"] = 0
        config["plugin_name"] = "preview.esm"
        save_config()

    if index < 0 or index >= len(config["plugin_index"]):
        handle_news(
            None, "warn", f"Invalid plugin index {index}, defaulting to preview.csv"
        )
        index = (
            config["plugin_index"].index("preview.csv")
            if "preview.csv" in config["plugin_index"]
            else 0
        )

    # Only update if the selection has changed or forced
    if config.get("plugin_selected") != index or force:
        config["plugin_selected"] = index
        selected_csv = config["plugin_index"][index]
        csv_path = (
            CSV_DIR / selected_csv
            if selected_csv == "preview.csv"
            else INPUT_DIR / selected_csv
        )

        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row and len(first_row) > 0:
                    config["plugin_name"] = first_row[0].strip()
                else:
                    config["plugin_name"] = selected_csv  # Use CSV name as fallback
            handle_news(
                None,
                "info",
                f"Read plugin name '{config['plugin_name']}' from {csv_path}",
            )
        except FileNotFoundError:
            handle_news(None, "error", f"CSV file {csv_path} not found.")
            config["plugin_name"] = "preview.esm"
            config["plugin_selected"] = (
                config["plugin_index"].index("preview.csv")
                if "preview.csv" in config["plugin_index"]
                else 0
            )
            selected_csv = "preview.csv"
            index = config["plugin_selected"]

        save_config()
        handle_news(
            None,
            "info",
            f"Selected plugin: index={index}, name={config['plugin_name']}, csv={selected_csv}",
        )

    main_window.plugin_name.setText("Plugin: " + config["plugin_name"])
    main_window.plugins_dropdown.setCurrentIndex(config["plugin_selected"])


def get_seed(config) -> int:
    """Return either a random seed or the user-defined seed from config."""
    use_random = config.get("use_random", False)

    if use_random:
        seed = random.randint(0, 99999)
        config["user_seed"] = seed
        save_config()
    else:
        seed = int(config.get("user_seed", 0))

    return seed


def update_seed_display(main_window, config):
    """Update the seed display widget with the current seed value."""
    seed = config.get("user_seed", "N/A")
    main_window.seed_display.display(seed)

# At the module level
planet_biomes_process: QProcess | None = None


def apply_bias(bias_type: str) -> Dict[str, float]:
    """Adjust biome zone values with a single-direction bias."""
    if bias_type == "set_equator_bias":
        # Strongest at the equator (zone_00), fading toward the poles
        zone_values = [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
    elif bias_type == "set_polar_bias":
        # Strongest at the poles (zone_06), fading toward the equator
        zone_values = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    else:  # Balanced bias
        zone_values = [0.5] * 7  # Uniform values for a neutral effect

    return {f"zone_0{i}": zone_values[i] for i in range(7)}


def update_ui_bias(bias_type: str):
    """Apply bias settings to UI sliders."""
    new_values = apply_bias(bias_type)
    for key, slider in slider_vars.items():
        if key in new_values:
            slider.setValue(int(new_values[key] * 100))


def get_planet_biomes_process() -> QProcess:
    """Retrieve the planet_biomes_process, ensuring it's initialized."""
    global planet_biomes_process
    if planet_biomes_process is None:
        raise RuntimeError("planet_biomes_process is not initialized")
    return planet_biomes_process


def start_planet_biomes(main_window):
    global planet_biomes_process, process_list
    global total_news, total_biom, total_text, total_other
    global news_count, news_percent, biom_percent, text_percent
    handle_news(None)

    main_window.stdout_widget.clear()
    main_window.stderr_widget.clear()

    # Reset progress counters
    news_count = 0
    news_percent = 0.0
    biom_percent = 0.0
    text_percent = 0.0

    # Precompute totals based on input
    totals = precompute_total_news(config)
    total_news = totals["total_news"]
    total_biom = totals["total_biom"]
    total_text = totals["total_text"]
    total_other = totals["total_other"]

    reset_news_count()

    # Initialize progress bars and labels
    main_window.news_count_progressBar.setValue(0)
    main_window.biom_count_progressBar.setValue(0)
    main_window.text_count_progressBar.setValue(0)
    main_window.news_label.setText("Working...")

    if not SCRIPT_PATH.exists():
        main_window.stderr_widget.insertPlainText(
            f"Error: PlanetBiomes.py not found at {SCRIPT_PATH}"
        )
        main_window.stderr_widget.moveCursor(QTextCursor.MoveOperation.End)
        return

    for key in config:
        if key.startswith("enable_") and key.endswith("_view"):
            # Force it True in the config
            update_value(key, True)

            # Also update the UI checkbox if it exists
            checkbox = getattr(main_window, f"{key}_checkbox", None)
            if checkbox is not None:
                checkbox.setChecked(True)
                checkbox.toggled.emit(True)  # optional
                checkbox.click()

    seed = get_seed(config)
    update_seed_display(main_window, config)
    handle_news(main_window, "success", f"Permit application: {seed} received.")

    if config.get("enable_preview_mode", False):
        disable_upscaling()

    save_config()

    # Initialize planet_biomes_process
    planet_biomes_process = QProcess()
    planet_biomes_process.setProgram(sys.executable)
    args = [str(SCRIPT_PATH)]
    if config.get("enable_preview_mode", False):
        args.append(str(TEMPLATE_PATH))
    planet_biomes_process.setArguments(args)
    planet_biomes_process.setWorkingDirectory(str(BASE_DIR))

    process_list.append(planet_biomes_process)

    # Handle output for image updates
    def handle_output():
        if planet_biomes_process is None:
            handle_news(
                main_window, "error", "Error: planet_biomes_process is not initialized"
            )
            return

        output = planet_biomes_process.readAllStandardOutput().data().decode()
        updated = False

        for line in output.splitlines():
            handle_news(
                main_window, "success", line
            )  # assumes success-type lines for stdout

        for message, indices in PROCESSING_MAP.items():
            if message in output:
                handle_news(main_window, "debug", message)
                for index in indices:
                    output_image = TEMP_DIR / IMAGE_FILES[index]
                    if output_image.exists():
                        pixmap = QPixmap(str(output_image)).scaled(
                            main_window.image_labels[index].width(),
                            main_window.image_labels[index].height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        )
                        if main_window.image_labels[index].movie():
                            main_window.image_labels[index].movie().stop()
                            main_window.image_labels[index].setMovie(None)
                        main_window.image_labels[index].setPixmap(pixmap)
                        updated = True

        if not updated and "complete" in output.lower():
            for index in range(len(IMAGE_FILES)):
                output_image = TEMP_DIR / IMAGE_FILES[index]
                if output_image.exists():
                    pixmap = QPixmap(str(output_image)).scaled(
                        main_window.image_labels[index].width(),
                        main_window.image_labels[index].height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                    if main_window.image_labels[index].movie():
                        main_window.image_labels[index].movie().stop()
                        main_window.image_labels[index].setMovie(None)
                    main_window.image_labels[index].setPixmap(pixmap)

            generate_sphere(main_window, main_window.plotter)

        if "Visual inspection" in output:
            output_dir = Path(OUTPUT_DIR)
            num_files = sum(1 for f in output_dir.rglob("*") if f.is_file())
            handle_news(main_window, "success", f"Forms filed correctly: {num_files}")

    def handle_error():
        if planet_biomes_process is None:
            handle_news(
                main_window, "error", "Error: planet_biomes_process is not initialized"
            )
            return

        error_output = planet_biomes_process.readAllStandardError().data().decode()
        for line in error_output.splitlines():
            handle_news(main_window, "error", line)

    # Connect signals
    planet_biomes_process.readyReadStandardOutput.connect(handle_output)
    planet_biomes_process.readyReadStandardError.connect(handle_error)

    def on_planet_biomes_finished(exit_code):
        message = (
            f"Permit {seed} complete!\nDon't panic!"
            if exit_code == 0
            else f"Permit denied, code {exit_code}: Construction halted."
        )
        main_window.news_label.setText("Done.")
        kind = "success" if exit_code == 0 else "error"
        handle_news(main_window, kind, message)

        main_window.news_count_progressBar.setValue(int(100))
        main_window.biom_count_progressBar.setValue(int(100))
        main_window.text_count_progressBar.setValue(int(100))

        # Clear the plotter and regenerate all meshes
        main_window.plotter.clear()  # removes all meshes

        # Recreate all meshes from current data
        main_window.meshes = generate_sphere(main_window, main_window.plotter)

        # Force PyVista to redraw the scene
        main_window.plotter.render()

    planet_biomes_process.finished.connect(on_planet_biomes_finished)

    planet_biomes_process.errorOccurred.connect(
        lambda _: handle_news(
            main_window,
            "error",
            f"Error: {planet_biomes_process.errorString() if planet_biomes_process else 'Process not initialized'}",
        )
    )

    # Start GIFs for processing, but only for labels without existing images
    for index in [1, 2, 3, 4, 5, 6, 7]:
        output_image = TEMP_DIR / IMAGE_FILES[index]
        if not output_image.exists():  # Only set GIF if no image exists
            movie = QMovie(str(GIF_PATHS.get(index)))
            if movie.isValid():
                main_window.image_labels[index].setMovie(movie)
                movie.start()

    main_window.stderr_widget.insertPlainText(
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


def create_planet():

    # Initialize planet_sphere_process
    planet_sphere_process = QProcess()
    planet_sphere_process.setProgram(sys.executable)
    args = [str(SPHERE_PATH)]
    if config.get("enable_preview_mode", False):
        args.append(str(TEMPLATE_PATH))
    planet_sphere_process.setArguments(args)
    planet_sphere_process.setWorkingDirectory(str(BASE_DIR))

    process_list.append(planet_sphere_process)

    planet_sphere_process.start()


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

    color_preview_image: QLabel
    biome_preview_image: QLabel
    surface_preview_image: QLabel
    resource_preview_image: QLabel
    ocean_preview_image: QLabel
    normal_preview_image: QLabel
    ao_preview_image: QLabel
    rough_preview_image: QLabel
    fault_preview_image: QLabel
    news_label: QLabel
    plugin_name: QLabel
    sphere_preview_frame: QFrame
    stdout_widget: QTextEdit
    stderr_widget: QTextEdit
    themes_dropdown: QComboBox
    plugins_dropdown: QComboBox
    folders_dropdown: QComboBox
    seed_display: QLCDNumber
    texture_resolution_display: QLCDNumber
    news_count_progressBar: QProgressBar
    biom_count_progressBar: QProgressBar
    text_count_progressBar: QProgressBar
    user_seed: QSlider
    texture_resolution_scale: QSlider
    generate_sphere_button: QPushButton
    start_command_button: QPushButton
    halt_command_button: QPushButton
    exit_command_button: QPushButton
    reset_command_button: QPushButton
    set_equator_bias_button: QPushButton
    set_balanced_bias_button: QPushButton
    set_polar_bias_button: QPushButton
    open_plugins_button: QPushButton
    open_output_button: QPushButton
    open_input_button: QPushButton
    enable_color_view: QCheckBox
    enable_ocean_mask_view: QCheckBox
    enable_normal_view: QCheckBox
    enable_ao_view: QCheckBox
    enable_rough_view: QCheckBox
    enable_biome_view: QCheckBox
    enable_resource_view: QCheckBox
    enable_fault_view: QCheckBox

    def __init__(self):
        super().__init__()
        self.slider_vars = {}

        # Determine base directory based on whether we're running as a PyInstaller bundle
        if getattr(sys, "frozen", False):
            base_dir = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to mainwindow.ui
        loadUi(UI_PATH, self)
        self.slider_mappings = {}
        self.checkbox_mappings = {}
        self.dropdown_vars = {
            "plugin_selected": self.plugins_dropdown,
            # Add others here if needed
        }

        self.setWindowTitle("Planet Painter")
        self.themes = THEMES
        config = load_config()

        self.news_label.setText("Progress")

        # Initialize plugin list
        if not config.get("plugin_index"):
            config["plugin_index"] = ["preview.csv"]
            # config["plugin_list"] = ["preview.csv"]
            config["plugin_selected"] = 0
            config["plugin_name"] = "preview.esm"
            save_config()

        self.plugins_dropdown.clear()
        self.plugins_dropdown.addItems(config["plugin_index"])
        selected_index = config.get("plugin_selected", 0)
        if selected_index >= 0 and selected_index < len(config["plugin_index"]):
            self.plugins_dropdown.setCurrentIndex(selected_index)
        else:
            self.plugins_dropdown.setCurrentIndex(0)
            config["plugin_selected"] = 0
            config["plugin_name"] = "preview.esm"
            save_config()

        update_selected_plugin(
            selected_index if selected_index >= 0 else 0, self, force=True
        )
        self.plugins_dropdown.currentIndexChanged.connect(
            lambda idx: (
                update_selected_plugin(idx, self),
                self.refresh_ui_from_config(),
            )
        )

        self.image_labels = [
            self.color_preview_image,
            self.biome_preview_image,
            self.surface_preview_image,
            self.resource_preview_image,
            self.ocean_preview_image,
            self.normal_preview_image,
            self.ao_preview_image,
            self.rough_preview_image,
            self.fault_preview_image,
        ]

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
        for image_file, label in zip(IMAGE_FILES, self.image_labels):
            image_path = TEMP_DIR / image_file
            pixmap = (
                QPixmap(str(image_path)).scaled(
                    label.width(),
                    label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                if image_path.exists()
                else self.default_image
            )
            label.setPixmap(pixmap)

        message = f"Available themes: {', '.join(self.themes.keys())}"
        self.stdout_widget.insertPlainText(message)
        self.stdout_widget.moveCursor(QTextCursor.MoveOperation.End)

        # Connect signals
        self.generate_sphere_button.clicked.connect(create_planet)
        self.start_command_button.clicked.connect(lambda: start_planet_biomes(self))
        self.halt_command_button.clicked.connect(cancel_processing)
        self.exit_command_button.clicked.connect(cancel_and_exit)
        self.reset_command_button.clicked.connect(self.reset_all_to_defaults)
        self.themes_dropdown.addItems(self.themes.keys())
        self.set_equator_bias_button.clicked.connect(self.set_equator_bias)
        self.set_polar_bias_button.clicked.connect(self.set_polar_bias)
        self.set_balanced_bias_button.clicked.connect(self.set_balanced_bias)

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

        self.change_theme(config.get("theme", "Starfield"))

        save_config()

        message = "Available plugins:\n"
        for index, plugin in enumerate(config["plugin_index"]):
            message += f"  [{index}] {plugin}\n"
        self.stderr_widget.insertPlainText(message)

        # Embed a PyVista interactor into your Qt layout
        # Create the interactor (no parent)
        self.plotter = QtInteractor()

        # Create a layout and add the interactor
        layout = QVBoxLayout(self.sphere_preview_frame)
        layout.addWidget(self.plotter)
        self.sphere_preview_frame.setLayout(layout)

        # Update 3D Display
        self.meshes = generate_sphere(self, self.plotter)
        auto_connect_enable_buttons(self, self.plotter, self.meshes)

        for texture_type in self.meshes:
            checkbox_name = f"enable_{texture_type}_view"
            checkbox = getattr(self, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(self.meshes[texture_type]["visible"])

    def open_selected_folder(self, index):
        folder_path = self.folders_dropdown.itemData(index)
        if folder_path:
            self.open_folder(folder_path)
        self.folders_dropdown.setCurrentIndex(0)

    def reset_to_defaults(self, key):
        print(f"Resetting key: '{key}'")
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
                    if key in [
                        "number_faults",
                        "fault_width",
                        "user_seed",
                        "texture_resolution_scale",
                    ]:
                        slider.setValue(
                            int(default_value)
                        )  # No scaling for integer keys
                    else:
                        slider.setValue(
                            int(default_value * 100)
                        )  # Scale for float keys

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

            # --- Set plugin_selected to last item ---
            plugin_list = config.get("plugin_index", [])
            last_index = len(plugin_list) - 1 if plugin_list else 0
            config["plugin_selected"] = last_index
            config["plugin_name"] = plugin_list[last_index] if plugin_list else ""
            dropdown = self.dropdown_vars.get("plugin_selected")
            if dropdown:
                dropdown.setCurrentIndex(last_index)

            # Re-apply UI control values
            for key in self.slider_mappings:
                slider = self.slider_vars.get(key)
                if slider:
                    value = config.get(key, 0)
                    if key in [
                        "number_faults",
                        "fault_width",
                        "user_seed",
                        "texture_resolution_scale",
                    ]:
                        slider.setValue(int(value))
                    else:
                        slider.setValue(int(value * 100))

            for key in self.dropdown_vars:
                dropdown = self.dropdown_vars[key]
                if key == "plugin_selected":
                    plugin_list = config.get("plugin_index", [])
                    last_index = len(plugin_list) - 1 if plugin_list else 0
                    dropdown.setCurrentIndex(last_index)

            # Update displays
            self.seed_display.display(config.get("user_seed", 0))
            self.texture_resolution_display.display(
                config.get("texture_resolution", 111)
            )

            self.refresh_plugin_list()
            self.plugins_dropdown.setCurrentIndex(last_index)
            update_selected_plugin(last_index, self, force=True)

            self.refresh_ui_from_config()
            save_config()

        except FileNotFoundError:
            print("Default config file not found.")
            print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")

    def refresh_plugin_list(self):
        """Scan for CSVs and update plugin list in config and dropdown."""
        global config
        csv_files = list(INPUT_DIR.glob("*.csv"))
        csv_names = [f.name for f in csv_files]
        if PREVIEW_PATH.name not in csv_names:
            csv_names.append(PREVIEW_PATH.name)

        # Update plugin_index only if new CSVs are found
        current_index = config.get("plugin_index", [])
        if set(csv_names) != set(current_index):
            if "preview.csv" in csv_names:
                csv_names.remove("preview.csv")
                csv_names.insert(0, "preview.csv")  # Always first
            config["plugin_index"] = csv_names
            save_config()

        # Preserve existing selection if valid
        selected_index = config.get("plugin_selected", 0)
        if (
            selected_index >= 0
            and selected_index < len(csv_names)
            and config.get("plugin_name")
        ):
            # Keep current selection
            pass
        else:
            # Default to preview.csv
            config["plugin_selected"] = (
                csv_names.index("preview.csv") if "preview.csv" in csv_names else 0
            )
            config["plugin_name"] = "preview.esm"
            save_config()

        self.plugins_dropdown.blockSignals(True)
        self.plugins_dropdown.clear()
        self.plugins_dropdown.addItems(csv_names)
        self.plugins_dropdown.setCurrentIndex(config["plugin_selected"])
        self.plugins_dropdown.blockSignals(False)

        update_selected_plugin(config["plugin_selected"], self, force=True)

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

    def set_equator_bias(self):
        bias_values = apply_bias("set_equator_bias")
        for zone in bias_values:
            self.slider_vars[zone].setValue(
                int(bias_values[zone] * 100)
            )

    def set_polar_bias(self):
        bias_values = apply_bias("set_polar_bias")
        for zone in bias_values:
            self.slider_vars[zone].setValue(int(bias_values[zone] * 100))

    def set_balanced_bias(self):
        bias_values = apply_bias("set_balanced_bias")
        for zone in bias_values:
            self.slider_vars[zone].setValue(int(bias_values[zone] * 100))

    def setup_config_controls(self):
        """Map UI controls to configuration keys."""
        self.slider_vars["zone_00"] = self.findChild(QSlider, "zone_00")
        self.slider_vars["zone_01"] = self.findChild(QSlider, "zone_01")
        self.slider_vars["zone_02"] = self.findChild(QSlider, "zone_02")
        self.slider_vars["zone_03"] = self.findChild(QSlider, "zone_03")
        self.slider_vars["zone_04"] = self.findChild(QSlider, "zone_04")
        self.slider_vars["zone_05"] = self.findChild(QSlider, "zone_05")
        self.slider_vars["zone_06"] = self.findChild(QSlider, "zone_06")
        checkbox_mappings = {
            "process_biomes": "process_biomes",
            "enable_noise": "enable_noise",
            "enable_distortion": "enable_distortion",
            "enable_biases": "enable_biases",
            "enable_anomalies": "enable_anomalies",
            "use_random": "use_random",
            "enable_equator_anomalies": "enable_equator_anomalies",
            "enable_polar_anomalies": "enable_polar_anomalies",
            "enable_texture_light": "enable_texture_light",
            "enable_texture_edges": "enable_texture_edges",
            "enable_basic_filters": "enable_basic_filters",
            "process_images": "process_images",
            "enable_texture_noise": "enable_texture_noise",
            "enable_texture_terrain": "enable_texture_terrain",
            "upscale_image": "upscale_image",
            "output_dds_files": "output_dds_files",
            "keep_pngs_after_conversion": "keep_pngs_after_conversion",
            "output_mat_files": "output_mat_files",
            "output_biom_files": "output_biom_files",
            "enable_seed_anomalies": "enable_seed_anomalies",
            "random_distortion": "random_distortion",
            "enable_tectonic_plates": "enable_tectonic_plates",
            "enable_surface_metal_view": "enable_surface_metal_view",
            "enable_color_view": "enable_color_view",
            "enable_fault_view": "enable_fault_view",
            "enable_resource_view": "enable_resource_view",
            "enable_biome_view": "enable_biome_view",
            "enable_rough_view": "enable_rough_view",
            "enable_normal_view": "enable_normal_view",
            "enable_ao_view": "enable_ao_view",
            "enable_ocean_mask_view": "enable_ocean_mask_view",
        }

        slider_mappings = {
            "zoom_factor": "zoom_factor",
            "squircle_factor": "squircle_factor",
            "number_faults": "number_faults",
            "fault_width": "fault_width",
            "fault_jitter": "fault_jitter",
            "fault_smooth": "fault_smooth",
            "noise_scale": "noise_scale",
            "noise_amplitude": "noise_amplitude",
            "user_seed": "user_seed",
            "fade_intensity": "fade_intensity",
            "texture_mountains": "texture_mountains",
            "texture_canyons": "texture_canyons",
            "fade_spread": "fade_spread",
            "equator_anomaly_count": "equator_anomaly_count",
            "equator_anomaly_spray": "equator_anomaly_spray",
            "polar_anomaly_count": "polar_anomaly_count",
            "polar_anomaly_spray": "polar_anomaly_spray",
            "distortion_scale": "distortion_scale",
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
            "texture_resolution_scale": "texture_resolution_scale",
            "texture_brightness": "texture_brightness",
            "texture_saturation": "texture_saturation",
            "texture_edges": "texture_edges",
            "texture_contrast": "texture_contrast",
            "texture_tint": "texture_tint",
            "texture_roughness": "texture_roughness",
            "texture_roughness_base": "texture_roughness_base",
            "texture_noise": "texture_noise",
            "texture_perlin": "texture_perlin",
            "texture_swap": "texture_swap",
            "texture_fractal": "texture_fractal",
            "ocean_mask_opacity": "ocean_mask_opacity",
            "ao_opacity": "ao_opacity",
            "normal_opacity": "normal_opacity",
            "rough_opacity": "rough_opacity",
            "biome_opacity": "biome_opacity",
            "resource_opacity": "resource_opacity",
            "fault_opacity": "fault_opacity",
            "color_opacity": "color_opacity",
        }

        reset_buttons = [
            "zoom_factor_reset",
            "squircle_factor_reset",
            "number_faults_reset",
            "fault_width_reset",
            "fault_jitter_reset",
            "fault_smooth_reset",
            "noise_scale_reset",
            "noise_amplitude_reset",
            "noise_scatter_reset",
            "biome_perlin_reset",
            "biome_swap_reset",
            "biome_fractal_reset",
            "texture_brightness_reset",
            "texture_saturation_reset",
            "texture_tint_reset",
            "texture_edges_reset",
            "texture_perlin_reset",
            "texture_swap_reset",
            "texture_fractal_reset",
            "texture_roughness_reset",
            "texture_roughness_base_reset",
            "fade_intensity_reset",
            "texture_canyons_reset",
            "texture_mountains_reset",
            "fade_spread_reset",
            "distortion_scale_reset",
        ]

        # Custom reset functions for grouped settings
        def reset_anomaly_counts():
            print("Resetting anomaly counts")
            self.reset_to_defaults("equator_anomaly_count")
            self.reset_to_defaults("polar_anomaly_count")

        def reset_anomaly_sprays():
            print("Resetting anomaly scales")
            self.reset_to_defaults("equator_anomaly_spray")
            self.reset_to_defaults("polar_anomaly_spray")

        # Connect checkboxes
        for checkbox_name, key in checkbox_mappings.items():
            checkbox = getattr(self, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(config.get(key, False))
                checkbox.toggled.connect(lambda val, k=key: update_value(k, val))
                checkbox_vars[key] = checkbox
            else:
                print(f"PlanetPainter: Warning: Checkbox '{checkbox_name}' not found in UI", file=sys.stderr)

        # Connect sliders
        for slider_name, key in slider_mappings.items():
            slider = getattr(self, slider_name, None)
            if slider:
                value = config.get(key, 0)
                min_val, max_val = 0.1, 1
                if key == "user_seed":
                    slider.setRange(0, 99999)
                    slider.setValue(int(value))
                    slider.valueChanged.connect(lambda val, k=key: update_value(k, val))
                elif key in (
                    "number_faults",
                    "texture_resolution_scale",
                    "fault_width",
                    ):
                    slider.setRange(1, 8)
                    slider.setValue(int(value))
                    slider.valueChanged.connect(lambda val, k=key: update_value(k, val))
                else:
                    if key in (
                        "noise_amplitude",
                        "texture_roughness_base",
                        "distortion_scale",
                    ):
                        max_val = 0.25
                    if key in ("ocean_mask_opacity", "ao_opacity"):
                        min_val, max_val = 0.01, 0.2
                    slider.setRange(int(min_val * 100), int(max_val * 100))
                    slider.setValue(int(value * 100))
                    slider.valueChanged.connect(
                        lambda val, k=key: update_value(
                            k, val / 100, plotter=self.plotter, meshes=self.meshes
                        )
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
                    f"PlanetPainter: Error: Button '{button_name}' or key '{key}' not found in UI or mappings", file=sys.stderr
                )

        # Connect special reset buttons
        special_reset_buttons = {
            "anomaly_count_reset": reset_anomaly_counts,
            "anomaly_spray_reset": reset_anomaly_sprays,
        }
        for button_name, reset_func in special_reset_buttons.items():
            button = getattr(self, button_name, None)
            if button:
                button.clicked.connect(reset_func)
            else:
                print(f"PlanetPainter: Error: Special reset button '{button_name}' not found in UI", file=sys.stderr)

        # Setup seed display
        self.seed_display.display(config["user_seed"])
        self.user_seed.valueChanged.connect(
            lambda val: (
                update_value("user_seed", int(val)),
                self.seed_display.display(int(val)),
            )
        )

        # Setup resolution display
        texture_resolution = config["texture_resolution_scale"] * 256
        self.texture_resolution_display.display(texture_resolution)
        self.texture_resolution_scale.valueChanged.connect(
            lambda val: (
                update_value(
                    "texture_resolution", 256 * int(val)
                ),  # Update config correctly
                update_value("texture_resolution_scale", int(val)),  # Keep scale synced
                self.texture_resolution_display.display(
                    256 * int(val)
                ),  # Show multiplied resolution
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

        self.stdout_widget.insertPlainText(message)
        self.stdout_widget.moveCursor(QTextCursor.MoveOperation.End)

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
            if slider:
                value = config.get(key, 0)
                if key in ["number_faults", "fault_width", "user_seed", "texture_resolution_scale"]:
                    slider.setValue(int(value))
                else:
                    slider.setValue(int(value * 100))

        for key, checkbox in checkbox_vars.items():
            if checkbox and key in config:
                checkbox.setChecked(config[key])

        # Update seed display
        self.seed_display.display(config.get("user_seed", 0))
        self.texture_resolution_display.display(config.get("texture_resolution", 1))

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
