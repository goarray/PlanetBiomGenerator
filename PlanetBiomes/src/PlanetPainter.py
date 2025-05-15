#!/usr/bin/env python3
"""
Biome Config Editor

A PySide6-based GUI application for editing biome configuration settings with a modern, sci-fi themed interface.
Allows users to modify numerical values, toggle boolean settings, and manage image pipeline configurations.
Supports loading/saving JSON configs and running an external PlanetBiomes.py script.

Dependencies:
- Python 3.8+
- PySide6
- Pillow (PIL)
- subprocess
- json
- pathlib
"""

from pathlib import Path
from PIL import Image
from themes import THEMES
import time
import sys
import os
import signal
import json
import subprocess
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QSlider,
    QPushButton,
    QSplashScreen,
    QComboBox,
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt, QTimer, QProcess

# Directory paths
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS).resolve()
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()
    if not (BASE_DIR / "src" / "PlanetBiomes.py").exists():
        print(
            f"Warning: PlanetBiomes.py not found in {BASE_DIR / 'src'}. Adjusting BASE_DIR."
        )
        BASE_DIR = Path(__file__).parent.resolve()

IMAGE_DIR = BASE_DIR / "assets" / "images"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "default.png"
PNG_OUTPUT_DIR = BASE_DIR / "Output" / "Textures"

# File Paths
SCRIPT_PATH = BASE_DIR / "src" / "PlanetBiomes.py"
PREVIEW_BIOME_PATH = BASE_DIR / "assets" / "PlanetBiomes.biom"
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"

# Image paths for display
IMAGE_FILES = [
    "preview_North_albedo.png",
    "preview_North_normal.png",
    "preview_North_rough.png",
    "preview_North_alpha.png",
]

# Configuration keys for boolean values
BOOLEAN_KEYS = {
    "enable_equator_drag",
    "enable_pole_drag",
    "enable_equator_intrusion",
    "enable_pole_intrusion",
    "apply_distortion",
    "apply_resource_gradient",
    "apply_latitude_blending",
    "delete_pngs_after_conversion"
}

# Human-readable labels for UI elements
LABELS = {
    # .biom manipulation labels
    "lat_weight_factor": "Zoom",
    "squircle_exponent": "Diamond (1) Circle (2) Squircle (max)",
    "noise_factor": "Equator Weight Mult",
    "global_seed": "Generation Seed",
    "noise_scale": "Anomoly Scale",
    "noise_amplitude": "Anomoly Distortion",
    "enable_equator_drag": "Enable Polar Anomolies",
    "enable_pole_drag": "Enable Equator Anomolies",
    # .png manipulation labels
    "image_pipeline": "Image Settings",
    "brightness_factor": "Brightness",
    "saturation_factor": "Saturation",
    "enable_edge_blending": "Enable Edges",
    "edge_blend_radius": "Edge Detail",
    "distortion_sigma": "Fine Distortion",
    "lat_distortion_factor": "Large Distortion",
    "drag_radius": "Anomolies",
    "enable_equator_intrusion": "Enable Equator Intrusions",
    "enable_pole_intrusion": "Enable Pole Intrusions",
    "apply_distortion": "Apply Terrain Distortion",
    "apply_resource_gradient": "Use Resource Gradient",
    "apply_latitude_blending": "Blend Biomes by Latitude",
    "zone_seed": "Seed",
    "elevation_scale": "Terrain Smoothness",
    "detail_smoothness": "Fractal Noise",
    "detail_strength_decay": "Fractal Decay",
    "normal_strength": "Normal Strength",
    "roughness_base": "Roughness Smoothness",
    "roughness_noise_scale": "Roughness Contrast",
    "alpha_base": "Alpha Base",
    "alpha_noise_scale": "Alpha Noise Scale",
}

# Global configuration dictionary
config = {}

# UI element storage
checkbox_vars = {}
spinbox_vars = {}

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
        config[category][key] = bool(val)
    elif (
        isinstance(config[category][key], list)
        and len(config[category][key]) == 2
        and index is not None
    ):
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

    print(json.dumps(config, indent=4))


def start_planet_biomes():
    """Start PlanetBiomes.py asynchronously and handle completion."""
    global planet_biomes_process

    # Validate script path
    if not SCRIPT_PATH.exists():
        print(f"Error: PlanetBiomes.py not found at {SCRIPT_PATH}")
        return

    # Initialize QProcess without a parent to avoid destruction when window closes
    planet_biomes_process = QProcess()
    planet_biomes_process.setProgram("python")
    planet_biomes_process.setArguments([str(SCRIPT_PATH)])
    planet_biomes_process.setWorkingDirectory(str(BASE_DIR))

    # Connect signals for feedback
    planet_biomes_process.finished.connect(
        lambda exit_code, exit_status: (
            print(f"PlanetBiomes.py finished with exit code {exit_code}"),
            cleanup_and_exit(exit_code),
        )
    )
    planet_biomes_process.errorOccurred.connect(
        lambda error: print(
            f"Error running PlanetBiomes.py: {planet_biomes_process.errorString()}"
        )
    )
    planet_biomes_process.readyReadStandardOutput.connect(
        lambda: print(
            f"PlanetBiomes.py output: {planet_biomes_process.readAllStandardOutput().data().decode()}"
        )
    )
    planet_biomes_process.readyReadStandardError.connect(
        lambda: print(
            f"PlanetBiomes.py error: {planet_biomes_process.readAllStandardError().data().decode()}"
        )
    )

    # Start the process
    print(f"Starting PlanetBiomes.py at {SCRIPT_PATH}")
    planet_biomes_process.start()
    if not planet_biomes_process.waitForStarted(5000):
        print(f"Failed to start PlanetBiomes.py: {planet_biomes_process.errorString()}")
        planet_biomes_process = None
        cleanup_and_exit(1)


def cleanup_and_exit(exit_code=0):
    """Clean up and exit the application."""
    global planet_biomes_process
    if planet_biomes_process and planet_biomes_process.state() != QProcess.NotRunning:
        planet_biomes_process.terminate()
        planet_biomes_process.waitForFinished(1000)
        if planet_biomes_process.state() != QProcess.NotRunning:
            planet_biomes_process.kill()
    planet_biomes_process = None
    sys.exit(exit_code)


def cancel_and_exit():
    """Terminate subprocess and exit application."""
    global planet_biomes_process
    if planet_biomes_process and planet_biomes_process.state() != QProcess.NotRunning:
        planet_biomes_process.terminate()
        planet_biomes_process.waitForFinished(1000)
        if planet_biomes_process.state() != QProcess.NotRunning:
            planet_biomes_process.kill()
    sys.exit()


def save_and_continue():
    """Save config, start PlanetBiomes, and hide UI."""
    save_config()
    main_window.hide()
    start_planet_biomes()


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
                checkbox_vars[key].setChecked(value)
            elif key in spinbox_vars:
                if isinstance(spinbox_vars[key], tuple):
                    min_spinbox, max_spinbox = spinbox_vars[key]
                    min_spinbox.setValue(value[0])
                    max_spinbox.setValue(value[1])
                else:
                    spinbox_vars[key].setValue(value)

    save_config()


def disable_upscaling():
    """Disable upscaling in the config file."""
    config_path = CONFIG_PATH
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
        config_data["image_pipeline"]["upscale_image"] = False
        with open(config_path, "w") as file:
            json.dump(config_data, file, indent=4)
    except Exception as e:
        print(f"Error disabling upscaling: {e}")


def generate_preview(main_window):
    """Start the preview script asynchronously and set up non-blocking wait."""
    # Check if a preview process is already running
    if (
        hasattr(main_window, "preview_process")
        and main_window.preview_process.state() != QProcess.NotRunning
    ):
        print("Preview is already running, please wait.")
        return

    # Validate paths
    if not SCRIPT_PATH.exists():
        print(f"Error: Preview script not found at {SCRIPT_PATH}")
        main_window.preview_button.setEnabled(True)
        return
    if not PREVIEW_BIOME_PATH.exists():
        print(f"Error: Biome file not found at {PREVIEW_BIOME_PATH}")
        main_window.preview_button.setEnabled(True)
        return

    disable_upscaling()

    process = QProcess()
    main_window.preview_process = process
    main_window.progress_started = False

    main_window.preview_button.setEnabled(False)

    process.finished.connect(lambda: wait_for_preview(main_window))
    process.errorOccurred.connect(
        lambda error: (
            print(f"Preview script error: {process.errorString()}"),
            main_window.preview_button.setEnabled(True),
        )
    )
    process.readyReadStandardOutput.connect(
        lambda: print(
            f"Script output: {process.readAllStandardOutput().data().decode()}"
        )
    )
    process.readyReadStandardError.connect(
        lambda: print(f"Script error: {process.readAllStandardError().data().decode()}")
    )

    print(f"Starting preview process with {SCRIPT_PATH} {PREVIEW_BIOME_PATH} --preview")
    process.setWorkingDirectory(str(BASE_DIR))
    process.start("python", [str(SCRIPT_PATH), str(PREVIEW_BIOME_PATH), "--preview"])
    if not process.waitForStarted(5000):
        print(f"Failed to start preview script: {process.errorString()}")
        main_window.preview_button.setEnabled(True)
        process = None


def start_processing_widget(main_window, title):
    """Launch processing indicator as a separate process and return the process."""
    print(f"Starting processing widget with title: {title}")
    script_path = os.path.join(os.path.dirname(__file__), "processing_widget.py")

    if not os.path.exists(script_path):
        print(f"Error: {script_path} does not exist!")
        return None

    try:
        process = subprocess.Popen(
            ["python", script_path, title], stderr=subprocess.PIPE
        )
        return process
    except Exception as e:
        print(f"Failed to start processing widget: {e}")
        return None


def wait_for_preview(main_window):
    """Wait for preview process to complete and refresh images."""
    if not main_window.progress_started:
        main_window.progress_started = True
        main_window.processing_widget_process = start_processing_widget(
            main_window, "Processing Planet Preview"
        )

    start_time = time.time()
    timeout = 60

    timer = QTimer(main_window)

    def update_progress():
        elapsed = time.time() - start_time
        if main_window.preview_process.state() == QProcess.NotRunning:
            print("Preview script finished, refreshing images")
            main_window.refresh_images()
            main_window.preview_button.setEnabled(True)
            # Terminate the processing widget
            if main_window.processing_widget_process:
                main_window.processing_widget_process.terminate()
                main_window.processing_widget_process.wait(1000)
                if main_window.processing_widget_process.poll() is None:
                    main_window.processing_widget_process.kill()
                main_window.processing_widget_process = None
            timer.stop()
            timer.deleteLater()
            main_window.progress_started = False
        elif elapsed > timeout:
            print("Preview timed out")
            main_window.preview_process.terminate()
            main_window.preview_process.waitForFinished(1000)
            main_window.preview_button.setEnabled(True)
            # Terminate the processing widget on timeout
            if main_window.processing_widget_process:
                main_window.processing_widget_process.terminate()
                main_window.processing_widget_process.wait(1000)
                if main_window.processing_widget_process.poll() is None:
                    main_window.processing_widget_process.kill()
                main_window.processing_widget_process = None
            timer.stop()
            timer.deleteLater()
            main_window.progress_started = False
        else:
            print("Waiting for preview completion...")

    timer.timeout.connect(update_progress)
    timer.start(500)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.themes = THEMES
        self.setWindowTitle("Biome Config Editor")
        self.setGeometry(50, 50, 1280, 900)
        self.progress_started = False
        self.processing_widget_process = None

        self.slider_vars = {}
        self.image_labels = []

        # Load images
        self.images = {
            image.stem: QPixmap(str(image)) for image in IMAGE_DIR.glob("*.png")
        }
        self.default_image = (
            QPixmap(str(DEFAULT_IMAGE_PATH))
            if DEFAULT_IMAGE_PATH.exists()
            else QPixmap()
        )

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Create side-by-side frames
        left_frame = QWidget()
        center_frame = QWidget()
        right_frame = QWidget()

        left_layout = QVBoxLayout()
        center_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_frame.setLayout(left_layout)
        center_frame.setLayout(center_layout)
        right_frame.setLayout(right_layout)

        main_layout.addWidget(left_frame)
        main_layout.addWidget(center_frame)
        main_layout.addWidget(right_frame)

        # Theme selector
        theme_selector = QComboBox()
        theme_selector.addItems(self.themes.keys())
        theme_selector.currentTextChanged.connect(self.change_theme)
        center_layout.addWidget(theme_selector)

        # Organize sections into panels
        sliders_layout = QVBoxLayout()
        booleans_layout = QVBoxLayout()
        image_pipeline_layout = QVBoxLayout()

        left_layout.addLayout(sliders_layout)
        center_layout.addLayout(booleans_layout)
        right_layout.addLayout(image_pipeline_layout)

        # Group assignments for panels
        left_groups = ["distortion_settings", "equator_anomolies", "pole_anomolies"]
        center_groups = ["global_toggles", "global_seed"]
        right_groups = ["image_pipeline"]

        # Image layout
        image_container = QWidget()
        image_layout = QHBoxLayout()
        image_layout.setAlignment(Qt.AlignCenter)
        image_container.setLayout(image_layout)

        # Albedo image (256x256)
        albedo_label = QLabel()
        albedo_path = PNG_OUTPUT_DIR / IMAGE_FILES[0]
        albedo_label.setPixmap(
            self.default_image
            if not albedo_path.exists()
            else QPixmap(str(albedo_path))
        )
        albedo_label.setAlignment(Qt.AlignCenter)
        albedo_label.setFixedSize(256, 256)
        albedo_label.setStyleSheet("border-radius: 6px")
        self.image_labels.append(albedo_label)
        image_layout.addWidget(albedo_label)

        # Vertical stack for normal, rough, alpha images (128x128 each)
        secondary_images_container = QWidget()
        secondary_images_layout = QVBoxLayout()
        secondary_images_container.setLayout(secondary_images_layout)
        image_names = ["Normal", "Rough", "Alpha"]
        for index, image_file in enumerate(IMAGE_FILES[1:]):
            label_text = QLabel(image_names[index])
            label_text.setAlignment(Qt.AlignCenter)
            secondary_images_layout.addWidget(label_text)
            label = QLabel()
            image_path = PNG_OUTPUT_DIR / image_file
            label.setPixmap(
                self.default_image.scaled(
                    80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                if not image_path.exists()
                else QPixmap(str(image_path)).scaled(
                    80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(80, 80)
            label.setScaledContents(True)
            label.setStyleSheet("border: 1px solid #4a4a8e; border-radius: 4px;")
            self.image_labels.append(label)
            secondary_images_layout.addWidget(label)
        image_layout.addWidget(secondary_images_container)

        center_layout.addWidget(image_container)

        # Buttons
        button_frame = QWidget()
        button_layout = QVBoxLayout()
        button_frame.setLayout(button_layout)

        self.preview_button = QPushButton("Preview Planet")
        self.preview_button.clicked.connect(lambda: generate_preview(self))

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(reset_to_defaults)

        cancel_button = QPushButton("Cancel and Exit")
        cancel_button.clicked.connect(cancel_and_exit)

        save_button = QPushButton("Save and Continue")
        save_button.clicked.connect(save_and_continue)

        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)

        center_layout.addWidget(button_frame)

        # Create UI elements for configuration
        group_heights = {
            "distortion_settings": 0.38,
            "equator_anomolies": 0.3,
            "pole_anomolies": 0.3,
            "global_toggles": 0.2,
            "global_seed": 0.1,
            "image_pipeline": 0.99,
        }

        window_height = self.height()
        scaled_heights = {key: int(window_height * ratio) for key, ratio in group_heights.items()}

        for category, sub_config in config.items():
            if category in left_groups:
                target_layout = sliders_layout
            elif category in center_groups:
                target_layout = booleans_layout
            elif category in right_groups:
                target_layout = image_pipeline_layout
            else:
                target_layout = booleans_layout

            group_box = QGroupBox(category.replace("_", " ").title())
            group_layout = QVBoxLayout()
            group_box.setLayout(group_layout)
            group_box.setFixedHeight(scaled_heights.get(category, int(window_height * 0.2)))

            for key, value in sub_config.items():
                if isinstance(value, bool):
                    checkbox = QCheckBox(LABELS.get(key, key.replace("_", " ").title()))
                    checkbox.setChecked(value)
                    checkbox_vars[key] = checkbox
                    checkbox.toggled.connect(
                        lambda val, k=key, c=category: update_value(c, k, val)
                    )
                    group_layout.addWidget(checkbox)
                elif isinstance(value, (int, float)):
                    sub_widget = QWidget()
                    sub_layout = QHBoxLayout()
                    sub_widget.setLayout(sub_layout)

                    label_text = str(LABELS.get(key, key.replace("_", " ").title()))
                    label = QLabel(label_text)
                    value_label = QLabel(f"{value:.2f}")

                    slider = QSlider(Qt.Horizontal)

                    min_val = 0.01
                    if "drag_strength" in key or "lat_weight_factor" in key or "edge_blend_radius" in key:
                        max_val = 4
                    elif "octaves" in key or "smoothness" in key or "squircle" in key:
                        max_val = 4
                    elif (
                        "drags" in key or "elevation_scale" in key or "drag_radius" in key
                    ):
                        max_val = 20
                    elif "x_min" in key or "y_min" in key or "crater_depth_min" in key:
                        min_val, max_val = -100, -0.01
                    elif "x_max" in key or "y_max" in key or "crater_depth_max" in key:
                        max_val = 100
                    else:
                        max_val = 1

                    slider.setRange(int(min_val * 100), int(max_val * 100))
                    slider.setValue(int(value * 100))
                    slider.setTickInterval((max_val - min_val) // 10)

                    slider.valueChanged.connect(
                        lambda val, c=category, k=key, lbl=value_label: (
                            update_value(c, k, val / 100),
                            lbl.setText(f"{val / 100:.2f}"),
                        )
                    )
                    self.slider_vars[key] = slider

                    sub_layout.addWidget(label)
                    sub_layout.addWidget(value_label)
                    sub_layout.addWidget(slider)
                    group_layout.addWidget(sub_widget)

            target_layout.addWidget(group_box)

        # Apply default theme
        self.change_theme("Light Sci-Fi")
        # Set sci-fi font
        app.setFont(QFont("Orbitron", 10))

    def change_theme(self, theme_name):
        """Apply the selected theme's stylesheet."""
        self.setStyleSheet(self.themes.get(theme_name, ""))

    def refresh_images(self):
        """Refresh the preview images from the output directory."""
        for i, image_file in enumerate(IMAGE_FILES):
            output_image = PNG_OUTPUT_DIR / image_file
            if output_image.exists():
                self.image_labels[i].setPixmap(QPixmap(str(output_image)))
            else:
                print(f"Preview image not found at {output_image}")
                self.image_labels[i].setPixmap(self.default_image)
            self.image_labels[i].update()

    def closeEvent(self, event):
        """Clean up processes on window close."""
        if hasattr(self, "preview_process") and self.preview_process:
            self.preview_process.terminate()
            self.preview_process.waitForFinished(1000)
        if self.processing_widget_process:
            self.processing_widget_process.terminate()
            self.processing_widget_process.wait(1000)
            if self.processing_widget_process.poll() is None:
                self.processing_widget_process.kill()
            self.processing_widget_process = None
        cleanup_and_exit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = QSplashScreen(
        QPixmap(str(DEFAULT_IMAGE_PATH)) if DEFAULT_IMAGE_PATH.exists() else QPixmap()
    )
    splash.show()
    QTimer.singleShot(500, splash.close)

    load_config()
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
