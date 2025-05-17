import json
import sys
from pathlib import Path

# Ensure BASE_DIR is the actual root project folder
BASE_DIR = (
    Path(sys._MEIPASS).resolve()
    if hasattr(sys, "_MEIPASS")
    else Path(__file__).parent.parent.resolve()  # ✅ Move up from /src/
)

CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "custom_config.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"


def load_config():
    """Load configuration from custom or default JSON file, or create a default config if neither exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # If neither config exists, create default first
    if not CONFIG_PATH.exists() and not DEFAULT_CONFIG_PATH.exists():
        print(f"Warning: No config found, creating {DEFAULT_CONFIG_PATH}...")
        default_config = {
            "some_values": {
                "zoom": 1.0,
                "noise_factor": 0.5,
                "enable_noise": False,
            },
            "global_seed": {"user_seed": 12345, "use_random": False},
        }
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(default_config, f, indent=4)

    # Check for custom config first, else fallback to default
    active_config_path = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH

    with open(active_config_path, "r") as file:
        return json.load(file)


# Load config at startup
config = load_config()

# Extract values for themes
FONT_PRIMARY = config.get("fonts", {}).get("primary", "Orbitron")
FONT_SECONDARY = config.get("fonts", {}).get("secondary", "Exo 2")
FONT_SIZE_SMALL = config.get("fonts", {}).get("size_small", "12px")
FONT_SIZE_LARGE = config.get("fonts", {}).get("size_large", "16px")
BUTTON_WIDTH = config.get("button", {}).get("width", "100px")
BUTTON_HEIGHT = config.get("button", {}).get("height", "40px")

# Generate themes dynamically from JSON
THEMES = {
    theme_name: f"""
        QMainWindow, QWidget {{
            background-color: {theme_data['background']}; 
            color: {theme_data['color']}; 
            font-family: {theme_data['font_family'].format(primary=FONT_PRIMARY, secondary=FONT_SECONDARY)};
        }}
        QPushButton {{
            background-color: {theme_data['button_color']}; 
            border: 1px solid {theme_data['border_color']}; 
            font-size: {FONT_SIZE_LARGE}; 
            min-width: {BUTTON_WIDTH}; 
            min-height: {BUTTON_HEIGHT};
        }}
    """
    for theme_name, theme_data in config.get("themes", {}).items()
}
