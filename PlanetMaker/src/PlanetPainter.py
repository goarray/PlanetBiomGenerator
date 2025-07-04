#!/usr/bin/env python3
"""
Biome Config Editor

A PyQt6-based GUI application for editing biome configuration settings.
Uses a .ui file for the interface and supports loading/saving JSON configs and running PlanetMaker.py.

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
from functools import partial
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
from PlanetThemes import (
    THEMES,
    get_biome_palette_stylesheet,
    get_height_palette_stylesheet,
)
from PlanetUtils import (
    BiomeDatabase,
    update_biome_selection,
    get_average_biome_humidity, 
    biome_db,
)
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
    make_percent,
    text_percent,
    total_news,
    total_make,
    total_text,
    total_other,
    reset_news_count,
    precompute_total_news,
)
from PlanetConstants import (
    # Modules
    get_config,
    save_config,
    # Core directories
    BASE_DIR,
    CONFIG_DIR,
    SCRIPT_DIR,
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
    FOLDER_PATHS,
    TEMPLATE_PATH,
    # UI and static assets
    UI_PATH,
    DEFAULT_IMAGE_PATH,
    GIF_PATHS,
    # Logic/data maps
    BOOLEAN_KEYS,
    PROCESSING_MAP,
    MAKER_PATH
)

# Global declarations
# config = {}
checkbox_vars = {}
slider_vars = {}
process_list = []

config = get_config()
print("Config ID:", id(config))


def update_value(key, val, index=None, plotter=None, meshes=None, main_window=None):
    if key not in config:
        print(f"Warning: Key '{key}' not found in config.")
        return
    elif (
        key in ("user_seed", "texture_resolution_scale")
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

    update_stat_config_from_ui(main_window)
    save_config()


def update_stat_config_from_ui(main_window):
    try:
        config["ttl_river"] = int(main_window.river_label.text())
        config["ttl_mountain"] = int(main_window.mountain_label.text())
        config["coastal_census_total"] = int(main_window.coastal_census_label.text())
        config["inland_census_total"] = int(main_window.inland_census_label.text())
    except ValueError:
        print("[Warning] Invalid label value(s), not updating config.")


def update_stat_labels(main_window, config):
    main_window.river_label.setText(str(config.get("ttl_river", 0)))
    main_window.mountain_label.setText(str(config.get("ttl_mountain", 0)))
    main_window.coastal_census_label.setText(str(config.get("coastal_census_total", 0)))
    main_window.inland_census_label.setText(str(config.get("inland_census_total", 0)))


def update_selected_plugin(index, main_window, force=False):
    print(f"[Debug] update_selected_plugin called with index={index}, force={force}")
    # Preserve critical values
    preserved_values = {
        "coastal_census_total": config.get("coastal_census_total", 0),
        "inland_census_total": config.get("inland_census_total", 0),
        "ttl_river": config.get("ttl_river", 0),
        "ttl_mountain": config.get("ttl_mountain", 0),
    }

    if "plugin_index" not in config or not config["plugin_index"]:
        handle_news(
            None, "error", "plugin_index missing or empty, restoring fallback list."
        )
        config["plugin_index"] = ["preview.csv"]
        config["plugin_selected"] = 0
        config["plugin_name"] = "preview.esm"
        config.update(preserved_values)

        save_config()
        return

    if index < 0 or index >= len(config["plugin_index"]):
        handle_news(
            None, "warn", f"Invalid plugin index {index}, defaulting to preview.csv"
        )
        index = (
            config["plugin_index"].index("preview.csv")
            if "preview.csv" in config["plugin_index"]
            else 0
        )

    # Only update if plugin actually changes or force is True
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
                    config["plugin_name"] = selected_csv
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

        # Restore preserved values
        config.update(preserved_values)

        save_config()

    main_window.plugins_dropdown.setCurrentIndex(config["plugin_selected"])

    # Handle biome dropdowns
    biome_keys = [f"biome0{i}_qcombobox" for i in range(7)]  # Fixed key format
    biome_boxes = [
        main_window.biome00_qcombobox,
        main_window.biome01_qcombobox,
        main_window.biome02_qcombobox,
        main_window.biome03_qcombobox,
        main_window.biome04_qcombobox,
        main_window.biome05_qcombobox,
        main_window.biome06_qcombobox,
    ]

    for key, box in zip(biome_keys, biome_boxes):
        value = config.get(key, 0)
        box.setCurrentIndex(int(value))

    print(
        f"[Debug] Config after update_selected_plugin: coastal={config.get('coastal_census_total', 0)}, inland={config.get('inland_census_total', 0)}, river={config.get('ttl_river', 0)}, mountain={config.get('ttl_mountain', 0)}"
    )


def get_seed(config) -> int:
    """Return either a random seed or the user-defined seed from config."""
    config = get_config()
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
planet_maker_process: QProcess | None = None


def get_planet_maker_process() -> QProcess:
    """Retrieve the planet_maker_process, ensuring it's initialized."""
    global planet_maker_process
    if planet_maker_process is None:
        raise RuntimeError("planet_maker_process is not initialized")
    return planet_maker_process


def start_planet_maker(main_window):
    global planet_maker_process, process_list
    global total_news, total_make, total_text, total_other
    global news_count, news_percent, make_percent, text_percent 
    update_stat_config_from_ui(main_window)
    planet_maker_process = QProcess()
    handle_news(None)

    main_window.stdout_widget.clear()
    main_window.stderr_widget.clear()

    # Reset progress counters
    news_count = 0
    news_percent = 0.0
    make_percent = 0.0
    text_percent = 0.0

    # Precompute totals based on input
    totals = precompute_total_news(config)
    total_news = totals["total_news"]
    total_make = totals["total_make"]
    total_text = totals["total_text"]
    total_other = totals["total_other"]

    reset_news_count()

    # Initialize progress bars and labels
    main_window.biome_palette_progressBar.setValue(100)
    main_window.biome_height_progressBar.setValue(100)
    main_window.news_count_progressBar.setValue(0)
    main_window.make_count_progressBar.setValue(0)
    main_window.text_count_progressBar.setValue(0)
    main_window.news_label.setText("Working...")

    if not MAKER_PATH.exists():
        main_window.stderr_widget.insertPlainText(
            f"Error: PlanetMaker.py not found at {MAKER_PATH}"
        )
        main_window.stderr_widget.moveCursor(QTextCursor.MoveOperation.End)
        return

    for key in config:
        if key.startswith("enable_") and key.endswith("_view"):
            # Force it True in the config
            update_value(key, True, main_window=main_window)

            # Also update the UI checkbox if it exists
            checkbox = getattr(main_window, f"{key}_checkbox", None)
            if checkbox is not None:
                checkbox.setChecked(True)
                checkbox.toggled.emit(True)  # optional
                checkbox.click()

    seed = get_seed(config)
    update_seed_display(main_window, config)
    handle_news(main_window, "success", f"Permit application: {seed} received.")

    args = [str(MAKER_PATH)]

    if config.get("run_planet_scripts", True):
        if config.get("run_planet_maker", True):
            # Launch PlanetMaker asynchronously — it will call subsequent steps internally
            planet_maker_process.setProgram(sys.executable)
            planet_maker_process.setArguments(args)
            planet_maker_process.setWorkingDirectory(str(BASE_DIR))
            process_list.append(planet_maker_process)

            # Connect signals and start process
            planet_maker_process.start()
        else:
            # No PlanetMaker, so run full pipeline synchronously from UI:
            steps = [
                ("run_planet_textures", SCRIPT_DIR / "PlanetTextures.py", "PlanetTextures"),
                (
                    "run_planet_materials",
                    SCRIPT_DIR / "PlanetMaterials.py",
                    "PlanetMaterials",
                ),
                ("run_planet_meshes", SCRIPT_DIR / "PlanetMeshes.py", "PlanetMeshes"),
                ("run_planet_surface", SCRIPT_DIR / "PlanetSurface.py", "PlanetSurface"),
            ] #("run_planet_biomes", SCRIPT_DIR / "PlanetBiomes.py", "PlanetBiomes"),
            for config_key, script_path, description in steps:
                if config.get(config_key, True):
                    subprocess.run([sys.executable, str(script_path)], check=True)
                else:
                    handle_news(main_window, "info", f"{description} step skipped.")

    # Handle output for image updates
    def handle_output():
        if planet_maker_process is None:
            handle_news(
                main_window, "error", "Error: planet_maker_process is not initialized"
            )
            return

        output = planet_maker_process.readAllStandardOutput().data().decode()

        for line in output.splitlines():
            handle_news(main_window, "success", line)

        if "complete" in output.lower():
            generate_sphere(main_window, main_window.plotter)

        if "Visual inspection" in output:
            output_dir = Path(OUTPUT_DIR)
            num_files = sum(1 for f in output_dir.rglob("*") if f.is_file())
            handle_news(main_window, "success", f"Forms filed correctly: {num_files}")

    def handle_error():
        if planet_maker_process is None:
            handle_news(
                main_window, "error", "Error: planet_maker_process is not initialized"
            )
            return

        error_output = planet_maker_process.readAllStandardError().data().decode()
        for line in error_output.splitlines():
            handle_news(main_window, "error", line)

    # Connect signals
    planet_maker_process.readyReadStandardOutput.connect(handle_output)
    planet_maker_process.readyReadStandardError.connect(handle_error)

    def on_planet_maker_finished(exit_code):
        # global config
        config_path = CONFIG_PATH  # or wherever your config lives
        with open(config_path, "r") as f:
            updated_config = json.load(f)
        print(f"[Debug] on_planet_maker_finished called with exit_code={exit_code}")
        print(
            f"[Debug] Config before reload: coastal={config.get('coastal_census_total', 0)}, inland={config.get('inland_census_total', 0)}, river={config.get('ttl_river', 0)}, mountain={config.get('ttl_mountain', 0)}"
        )

        # Reload config to reflect PlanetMaker's changes
        print(
            f"[Debug] Config after reload: coastal={config.get('coastal_census_total', 0)}, inland={config.get('inland_census_total', 0)}, river={config.get('ttl_river', 0)}, mountain={config.get('ttl_mountain', 0)}"
        )

        message = (
            f"Permit {seed} complete!\nDon't panic!"
            if exit_code == 0
            else f"Permit denied, code {exit_code}: Construction halted."
        )
        main_window.news_label.setText("Done.")
        kind = "success" if exit_code == 0 else "error"
        handle_news(main_window, kind, message)

        main_window.biome_palette_progressBar.setValue(100)
        main_window.biome_height_progressBar.setValue(100)
        main_window.news_count_progressBar.setValue(100)
        main_window.make_count_progressBar.setValue(100)
        main_window.text_count_progressBar.setValue(100)

        # Disconnect plugins_dropdown signal to prevent save_config
        try:
            main_window.plugins_dropdown.currentIndexChanged.disconnect()
            print("[Debug] Disconnected plugins_dropdown.currentIndexChanged")
        except TypeError:
            print("[Debug] plugins_dropdown.currentIndexChanged was not connected")

        # Update UI
        main_window.refresh_ui_from_config(updated_config)

        # Clear and regenerate meshes
        main_window.plotter.clear()
        main_window.meshes = generate_sphere(main_window, main_window.plotter)
        main_window.plotter.render()

        # Reconnect plugins_dropdown signal
        main_window.plugins_dropdown.currentIndexChanged.connect(
            lambda idx: (
                update_selected_plugin(idx, main_window),
                main_window.refresh_ui_from_config(updated_config),
            )
        )
        print("[Debug] Reconnected plugins_dropdown.currentIndexChanged")

    planet_maker_process.finished.connect(on_planet_maker_finished)

    planet_maker_process.errorOccurred.connect(
        lambda _: handle_news(
            main_window,
            "error",
            f"Error: {planet_maker_process.errorString() if planet_maker_process else 'Process not initialized'}",
        )
    )

    main_window.stderr_widget.insertPlainText(
        f"Starting PlanetMaker.py with args: {args}"
    )
    # planet_maker_process.start()


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
    print("Processing halted, but the app remains open.")


def create_planet():

    # Initialize planet_sphere_process
    planet_sphere_process = QProcess()
    planet_sphere_process.setProgram(sys.executable)
    args = [str(MAKER_PATH)]
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


class MainWindow(QMainWindow):

    news_label: QLabel
    humidity_label: QLabel
    river_label: QLabel
    mountain_label: QLabel
    coastal_census_label: QLabel
    inland_census_label: QLabel
    sphere_preview_frame: QFrame
    stdout_widget: QTextEdit
    stderr_widget: QTextEdit
    themes_dropdown: QComboBox
    plugins_dropdown: QComboBox
    folders_dropdown: QComboBox
    biome00_qcombobox: QComboBox
    biome01_qcombobox: QComboBox
    biome02_qcombobox: QComboBox
    biome03_qcombobox: QComboBox
    biome04_qcombobox: QComboBox
    biome05_qcombobox: QComboBox
    biome06_qcombobox: QComboBox
    seed_display: QLCDNumber
    resolution_display: QLCDNumber
    biome_palette_progressBar: QProgressBar
    biome_height_progressBar: QProgressBar
    news_count_progressBar: QProgressBar
    make_count_progressBar: QProgressBar
    text_count_progressBar: QProgressBar
    user_seed: QSlider
    texture_resolution_scale: QSlider
    generate_sphere_button: QPushButton
    start_command_button: QPushButton
    halt_command_button: QPushButton
    exit_command_button: QPushButton
    reset_command_button: QPushButton
    open_plugins_button: QPushButton
    open_output_button: QPushButton
    open_input_button: QPushButton
    enable_color_view: QCheckBox
    enable_ocean_mask_view: QCheckBox
    enable_normal_view: QCheckBox
    enable_colony_mask_view: QCheckBox
    enable_rough_view: QCheckBox
    enable_biome_view: QCheckBox
    enable_resource_view: QCheckBox
    enable_terrain_view: QCheckBox
    enable_terrain_normal_view: QCheckBox
    enable_river_mask_view: QCheckBox
    enable_mountain_mask_view: QCheckBox
    enable_road_mask_view: QCheckBox
    enable_humidity_view: QCheckBox

    def __init__(self):
        super().__init__()
        self.slider_vars = {}
        self.biome_db = BiomeDatabase()
        self.biome_db.load_csv(CSV_DIR / "Biomes.csv")

        # Determine base directory based on whether we're running as a PyInstaller bundle
        if getattr(sys, "frozen", False):
            base_dir = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to mainwindow.ui
        loadUi(UI_PATH, self)
        print(f"river_label: {self.river_label}")  # Should print a QLabel object
        print(f"Has river_label: {hasattr(self, 'river_label')}")  # Should print True
        self.biome_color_pickers = {}
        self.slider_mappings = {}
        self.checkbox_mappings = {}
        self.dropdown_vars = {
            "plugin_selected": self.plugins_dropdown,
            "biome00_qcombobox": self.biome00_qcombobox,
            "biome01_qcombobox": self.biome01_qcombobox,
            "biome02_qcombobox": self.biome02_qcombobox,
            "biome03_qcombobox": self.biome03_qcombobox,
            "biome04_qcombobox": self.biome04_qcombobox,
            "biome05_qcombobox": self.biome05_qcombobox,
            "biome06_qcombobox": self.biome06_qcombobox,
        }

        self.setWindowTitle("Planet Painter")
        self.themes = THEMES
        print(
            f"[Debug] Initial config in PlanetMaker: coastal={config.get('coastal_census_total', 0)}, inland={config.get('inland_census_total', 0)}, river={config.get('ttl_river', 0)}, mountain={config.get('ttl_mountain', 0)}"
        )

        plugin_name = config.get("plugin_name", "default_plugin")
        planet_name = config.get("planet_name", "default_planet")

        self.png_dir = PNG_OUTPUT_DIR / plugin_name / planet_name

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
                self.refresh_ui_from_config(config),
            )
        )

        # Set initial index for biome dropdowns
        for key, box in self.dropdown_vars.items():
            if key.startswith("biome") and isinstance(box, QComboBox):
                box.setCurrentIndex(config.get(key, 0))

        message = f"Available themes: {', '.join(self.themes.keys())}"
        self.stdout_widget.insertPlainText(message)
        self.stdout_widget.moveCursor(QTextCursor.MoveOperation.End)

        # Connect signals
        self.generate_sphere_button.clicked.connect(create_planet)
        self.start_command_button.clicked.connect(lambda: start_planet_maker(self))
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

        self.change_theme(config.get("theme", "Starfield"))
        self.biome_palette_progressBar.setObjectName("BiomePaletteBar")
        self.biome_height_progressBar.setObjectName("HeightPaletteBar")
        self.setStyleSheet(self.themes.get(config.get("theme", "Starfield"), ""))

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

        self.coastal_census_label.setText(str(config.get("coastal_census_total", 0)))
        self.inland_census_label.setText(str(config.get("inland_census_total", 0)))
        self.river_label.setText(str(config.get("ttl_river", 0)))
        self.mountain_label.setText(str(config.get("ttl_mountain", 0)))

        self.init_biome_dropdowns()

        avg_humidity = get_average_biome_humidity(config, self.biome_db)
        self.humidity_label.setText(f"{avg_humidity:.2f}")

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
                update_value(key, default_value, main_window=self)

                # Update sliders
                slider = slider_vars.get(key)
                if slider:
                    if key in [
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
            config = default_config

            # Initialize biome colors based on default biome indices
            self.biome_db = BiomeDatabase()
            self.biome_db.load_csv(CSV_DIR / "Biomes.csv")
            editor_ids = list(self.biome_db.biomes_by_name.keys())
            for i in range(7):
                key = f"biome{i:02}_qcombobox"
                color_key = f"biome{i:02}_color"
                biome_name = config.get(key, editor_ids[0])
                if biome_name not in editor_ids:
                    biome_name = editor_ids[0]
                biome = self.biome_db.biomes_by_name.get(biome_name)
                if biome:
                    r, g, b = biome.color
                    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                    config[color_key] = hex_color
                    print(f"[Debug] Reset {color_key} to {hex_color}")
                else:
                    config[color_key] = "#000000"
                    print(f"[Debug] Fallback to #000000 for {color_key}")

            # Save updated config
            print(
                f"[CONFIG] reset_all_to_defaults, ttl_river={config.get("ttl_river")}"
            )
            save_config()

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
                    if key in ["user_seed", "texture_resolution_scale"]:
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
            self.resolution_display.display(config.get("texture_resolution", 111))
            self.coastal_census_label.setText(str(config["coastal_census_total"]))
            self.inland_census_label.setText(str(config["inland_census_total"]))

            self.refresh_plugin_list()
            self.plugins_dropdown.setCurrentIndex(last_index)
            update_selected_plugin(last_index, self, force=True)

            self.init_biome_dropdowns()

            self.refresh_ui_from_config(config)

        except FileNotFoundError:
            print(f"Error: Default config file {DEFAULT_CONFIG_PATH} not found.")

    def init_biome_dropdowns(self):
        editor_ids = list(self.biome_db.biomes_by_name.keys())

        for key, box in self.dropdown_vars.items():
            if key.startswith("biome"):
                box.blockSignals(True)  # Block signals while initializing
                box.clear()
                box.addItems(editor_ids)

                # Load saved index or default to 0
                index = config.get(key, 0)
                if not isinstance(index, int) or index < 0 or index >= len(editor_ids):
                    index = 0
                box.setCurrentIndex(index)

                # Update corresponding config values
                base_key = key.replace("_qcombobox", "")
                biome_name = editor_ids[index]
                biome = self.biome_db.biomes_by_name.get(biome_name)
                if biome:
                    r, g, b = biome.color
                    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                    config[f"{base_key}_editor_id"] = biome.editor_id
                    config[f"{base_key}_formid"] = biome.form_id
                    config[f"{base_key}_color"] = hex_color
                    print(
                        f"[Debug] Initialized {base_key}: editor_id={biome.editor_id}, formid={biome.form_id}, color={hex_color}"
                    )
                else:
                    config[f"{base_key}_editor_id"] = editor_ids[0]
                    config[f"{base_key}_formid"] = 0
                    config[f"{base_key}_color"] = "#000000"
                    print(
                        f"[Debug] Fallback for {base_key}: editor_id={editor_ids[0]}, formid=0, color=#000000"
                    )

                box.blockSignals(False)

                # Connect selection change to config
                box.currentIndexChanged.connect(
                    partial(update_biome_selection, self, config, key, self.biome_db)
                )

        # self.coastal_census_label.setText(str(config.get("coastal_census_total", 0)))
        # self.inland_census_label.setText(str(config.get("inland_census_total", 0)))
        # self.river_label.setText(str(config.get("ttl_river", 0)))
        # self.mountain_label.setText(str(config.get("ttl_mountain", 0)))

        self.biome_palette_progressBar.setStyleSheet(get_biome_palette_stylesheet(config))
        self.biome_palette_progressBar.repaint()
        self.biome_height_progressBar.setStyleSheet(get_height_palette_stylesheet(config))
        self.biome_height_progressBar.repaint()

        avg_humidity = get_average_biome_humidity(config, self.biome_db)
        self.humidity_label.setText(f"{avg_humidity:.2f}")

    def refresh_plugin_list(self):
        """Scan for CSVs and update plugin list in config and dropdown."""
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

            # save_config()

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

    def setup_config_controls(self):
        """Map UI controls to configuration keys."""
        checkbox_mappings = {
            "run_planet_scripts": "run_planet_scripts",
            "run_planet_maker": "run_planet_maker",
            "run_planet_textures": "run_planet_textures",
            "run_planet_materials": "run_planet_materials",
            "run_planet_meshes": "run_planet_meshes",
            "run_planet_biomes": "run_planet_biomes",
            "run_planet_surface": "run_planet_surface",
            "use_random": "use_random",
            "enable_coastal_population": "enable_coastal_population",
            "enable_inland_population": "enable_inland_population",
            "enable_texture_light": "enable_texture_light",
            "enable_texture_edges": "enable_texture_edges",
            "enable_basic_filters": "enable_basic_filters",
            "process_images": "process_images",
            "enable_texture_noise": "enable_texture_noise",
            "enable_texture_terrain": "enable_texture_terrain",
            "output_dds_files": "output_dds_files",
            "keep_pngs_after_conversion": "keep_pngs_after_conversion",
            "output_mat_files": "output_mat_files",
            "output_biom_files": "output_biom_files",
            "enable_ocean_population": "enable_ocean_population",
            "random_distortion": "random_distortion",
            "enable_surface_metal_view": "enable_surface_metal_view",
            "enable_color_view": "enable_color_view",
            "enable_terrain_normal_view": "enable_terrain_normal_view",
            "enable_terrain_view": "enable_terrain_view",
            "enable_resource_view": "enable_resource_view",
            "enable_biome_view": "enable_biome_view",
            "enable_rough_view": "enable_rough_view",
            "enable_normal_view": "enable_normal_view",
            "enable_colony_mask_view": "enable_colony_mask_view",
            "enable_ocean_mask_view": "enable_ocean_mask_view",
            "enable_river_mask_view": "enable_river_mask_view",
            "enable_mountain_mask_view": "enable_mountain_mask_view",
            "enable_road_mask_view": "enable_road_mask_view",
            "enable_humidity_view": "enable_humidity_view",
        }

        slider_mappings = {
            "biome_order": "biome_order",
            "biome_chaos": "biome_chaos",
            "user_seed": "user_seed",
            "fade_intensity": "fade_intensity",
            "fade_spread": "fade_spread",
            "coastal_population_count": "coastal_population_count",
            "coastal_population_density": "coastal_population_density",
            "inland_population_count": "inland_population_count",
            "inland_population_density": "inland_population_density",
            "distortion_scale": "distortion_scale",
            "biome_perlin": "biome_perlin",
            "biome_swap": "biome_swap",
            "biome_fractal": "biome_fractal",
            "humidity_bias": "humidity_bias",
            "river_bias": "river_bias",
            "mountain_bias": "mountain_bias",
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
            "colony_mask_opacity": "colony_mask_opacity",
            "normal_opacity": "normal_opacity",
            "rough_opacity": "rough_opacity",
            "biome_opacity": "biome_opacity",
            "resource_opacity": "resource_opacity",
            "terrain_opacity": "terrain_opacity",
            "terrain_normal_opacity": "terrain_normal_opacity",
            "color_opacity": "color_opacity",
            "river_mask_opacity": "river_mask_opacity",
            "mountain_mask_opacity": "mountain_mask_opacity",
            "road_mask_opacity": "road_mask_opacity",
            "humidity_opacity": "humidity_opacity",
        }

        reset_buttons = [
            "humidity_bias_reset",
            "river_bias_reset",
            "mountain_bias_reset",
            "biome_order_reset",
            "biome_chaos_reset",
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
            "fade_spread_reset",
            "distortion_scale_reset",
        ]

        # self.coastal_census_label.setText(str(config.get("coastal_census_total", 0)))
        # self.inland_census_label.setText(str(config.get("inland_census_total", 0)))
        # self.river_label.setText(str(config.get("ttl_river", 0)))
        # self.mountain_label.setText(str(config.get("ttl_mountain", 0)))

        # Custom reset functions for grouped settings
        def reset_population_counts():
            print("Resetting population counts")
            self.reset_to_defaults("coastal_population_count")
            self.reset_to_defaults("inland_population_count")

        def reset_population_densities():
            print("Resetting population scales")
            self.reset_to_defaults("coastal_population_density")
            self.reset_to_defaults("inland_population_density")

        # Connect checkboxes
        for checkbox_name, key in checkbox_mappings.items():
            checkbox = getattr(self, checkbox_name, None)
            if checkbox:
                checkbox.setChecked(config.get(key, False))
                checkbox.toggled.connect(
                    lambda val, k=key: update_value(k, val, main_window=self)
                )
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
                    slider.valueChanged.connect(
                        lambda val, k=key: update_value(k, val, main_window=self)
                    )
                elif key in (
                    "texture_resolution_scale",
                ):
                    slider.setRange(1, 8)
                    slider.setValue(int(value))
                    slider.valueChanged.connect(
                        lambda val, k=key: update_value(k, val, main_window=self)
                    )
                else:
                    if key in (
                        "texture_roughness_base",
                        "distortion_scale",
                    ):
                        max_val = 0.25
                    if key in ("ocean_mask_opacity", "colony_mask_opacity"):
                        min_val = 0.01
                    slider.setRange(int(min_val * 100), int(max_val * 100))
                    slider.setValue(int(value * 100))
                    slider.valueChanged.connect(
                        lambda val, k=key: update_value(
                            k, val / 100, plotter=self.plotter, meshes=self.meshes, main_window=self
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
            "population_count_reset": reset_population_counts,
            "population_density_reset": reset_population_densities,
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
                update_value("user_seed", int(val), main_window=self),
                self.seed_display.display(int(val)),
            )
        )

        avg_humidity = get_average_biome_humidity(config, self.biome_db)
        self.humidity_label.setText(f"{avg_humidity:.2f}")

        # Setup resolution display
        resolution = config["texture_resolution_scale"] * 256
        self.resolution_display.display(resolution)
        self.texture_resolution_scale.valueChanged.connect(
            lambda val: (
                update_value(
                    "texture_resolution", 256 * int(val), main_window=self
                ),  # Update config correctly
                update_value(
                    "texture_resolution_scale", int(val), main_window=self
                ),  # Keep scale synced
                self.resolution_display.display(
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

        else:
            self.setStyleSheet(self.themes.get("Starfield", ""))
            message = f"Error: Theme '{theme_name}' not found in themes"

        self.stdout_widget.insertPlainText(message)
        self.stdout_widget.moveCursor(QTextCursor.MoveOperation.End)

        save_config()

    def refresh_ui_from_config(self, config):
        """Refresh the entire UI to reflect current config values."""
        print("[Debug] refresh_ui_from_config called")
        print(f"[Debug] Config in refresh_ui: coastal={config.get('coastal_census_total', 0)}, inland={config.get('inland_census_total', 0)}, river={config.get('ttl_river', 0)}, mountain={config.get('ttl_mountain', 0)}")
        for key, slider in slider_vars.items():
            if slider:
                value = config.get(key, 0)
                if key in ["user_seed", "texture_resolution_scale"]:
                    slider.setValue(int(value))
                else:
                    slider.setValue(int(value * 100))

        for key, checkbox in checkbox_vars.items():
            if checkbox and key in config:
                checkbox.setChecked(config[key])

        # Update displays
        self.seed_display.display(config.get("user_seed", 0))
        self.resolution_display.display(config.get("texture_resolution", 111))

        update_stat_labels(main_window, config)

        for key in config:
            if key.startswith("biome") and key.endswith("_qcombobox"):
                index = config[key]
                update_biome_selection(self, config, key, self.biome_db, index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Orbitron", 10))
    splash = QSplashScreen(
        QPixmap(str(DEFAULT_IMAGE_PATH)) if DEFAULT_IMAGE_PATH.exists() else QPixmap()
    )
    splash.show()
    QTimer.singleShot(500, splash.close)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
