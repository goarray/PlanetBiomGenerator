import json
import sys
from pathlib import Path

# Ensure BASE_DIR is the actual root project folder
BASE_DIR = (
    Path(sys._MEIPASS).resolve()
    if hasattr(sys, "_MEIPASS")
    else Path(__file__).parent.parent.resolve()
)

CONFIG_DIR = BASE_DIR / "config"
DEFAULT_THEME_PATH = CONFIG_DIR / "default_themes.json"
CUSTOM_THEME_PATH = CONFIG_DIR / "custom_themes.json"

DEFAULT_THEMES = {
    "Light": {
        "background": "#f0f0f0",
        "color": "#202020",
        "font_family": "{primary}, sans-serif",
        "button_color": "#ffffff",
        "border_color": "#cccccc",
        "fonts": {
            "primary": "Orbitron",
            "secondary": "Exo 2",
            "size_small": "12px",
            "size_large": "14px",
        },
    },
    "Dark": {
        "background": "#121212",
        "color": "#dddddd",
        "font_family": "{secondary}, monospace",
        "button_color": "#1e1e1e",
        "border_color": "#555555",
        "fonts": {
            "primary": "Orbitron",
            "secondary": "Exo 2",
            "size_small": "12px",
            "size_large": "14px",
        },
    },
    "Sci-Fi": {
        "background": "#0d1117",
        "color": "#39ff14",
        "font_family": "{primary}, {secondary}, monospace",
        "button_color": "#111a24",
        "border_color": "#00ff99",
        "fonts": {
            "primary": "Orbitron",
            "secondary": "Exo 2",
            "size_small": "12px",
            "size_large": "14px",
        },
    },
    "Sci-Fi Light": {
        "background": "#e8f0ff",
        "color": "#002244",
        "font_family": "{primary}, {secondary}, sans-serif",
        "button_color": "#d0e0ff",
        "border_color": "#6699cc",
        "fonts": {
            "primary": "Orbitron",
            "secondary": "Exo 2",
            "size_small": "12px",
            "size_large": "14px",
        },
    },
}


def load_theme():
    """Load theme from custom or default JSON file, or create a default theme file if none exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CUSTOM_THEME_PATH.exists():
        try:
            with open(CUSTOM_THEME_PATH, "r") as theme_file:
                return json.load(theme_file)
        except json.JSONDecodeError as e:
            print(
                f"Error: Invalid JSON in {CUSTOM_THEME_PATH}. Falling back to default themes. {e}"
            )
            return DEFAULT_THEMES

    elif DEFAULT_THEME_PATH.exists():
        try:
            with open(DEFAULT_THEME_PATH, "r") as theme_file:
                return json.load(theme_file)
        except json.JSONDecodeError as e:
            print(
                f"Error: Invalid JSON in {DEFAULT_THEME_PATH}. Falling back to default themes. {e}"
            )
            return DEFAULT_THEMES

    else:
        with open(DEFAULT_THEME_PATH, "w") as theme_file:
            json.dump(DEFAULT_THEMES, theme_file, indent=4)
        return DEFAULT_THEMES


# Load theme at startup
theme_data = load_theme()

# Extract global theme-related values with defaults
FONT_PRIMARY = theme_data.get("fonts", {}).get("primary", "Orbitron")
FONT_SECONDARY = theme_data.get("fonts", {}).get("secondary", "Exo 2")
FONT_SIZE_SMALL = theme_data.get("fonts", {}).get("size_small", "12px")
FONT_SIZE_LARGE = theme_data.get("fonts", {}).get("size_large", "14px")

# Generate final Qt style sheet dictionary
THEMES = {
    theme_name: f"""
        QMainWindow, QWidget {{
            background-color: {theme['background']}; 
            color: {theme['color']}; 
            font-family: {theme['font_family'].format(primary=FONT_PRIMARY, secondary=FONT_SECONDARY)};
            font-size: {theme.get('fonts', {}).get('size_small', FONT_SIZE_SMALL)};
        }}

        QPushButton {{
            background-color: {theme['button_color']}; 
            border: 1px solid {theme['border_color']}; 
            font-size: {theme.get('fonts', {}).get('size_large', FONT_SIZE_LARGE)};
            padding: 5px;
        }}

        QPushButton:hover {{
            background-color: {theme['border_color']};
        }}

        QPushButton.primary {{
            background-color: {theme['border_color']};
            color: {theme['background']};
            border: 2px solid {theme['color']};
            font-weight: bold;
        }}

        QPushButton.danger {{
            background-color: #cc0000;
            color: white;
            border: 1px solid #990000;
        }}

        QPushButton.flat {{
            background-color: transparent;
            border: none;
            color: {theme['color']};
        }}

        QSlider::groove:horizontal {{
            height: 6px;
            background: {theme['border_color']};
            margin: 2px 0;
            border-radius: 3px;
        }}

        QSlider::groove:vertical {{
            width: 6px;
            background: {theme['border_color']};
            margin: 0 2px;
            border-radius: 3px;
        }}

        QSlider::handle:horizontal {{
            width: 14px;
            height: 14px;
            background: {theme['color']};
            border: 1px solid {theme['background']};
            border-radius: 7px;
            margin: -5px 0;
        }}

        QSlider::handle:vertical {{
            width: 14px;
            height: 14px;
            background: {theme['color']};
            border: 1px solid {theme['background']};
            border-radius: 7px;
            margin: 0 -5px;
        }}

        QCheckBox, QRadioButton {{
            spacing: 5px;
            color: {theme['color']};
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 14px;
            height: 14px;
        }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {theme['color']};
            border: 1px solid {theme['border_color']};
        }}
        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {{
            background-color: {theme['button_color']};
            border: 1px solid {theme['border_color']};
        }}
        QComboBox {{
            padding: 3px;
            border: 1px solid {theme['border_color']};
            background-color: {theme['button_color']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {theme['button_color']};
            selection-background-color: {theme['border_color']};
            color: {theme['color']};
        }}
    """
    for theme_name, theme in theme_data.items()
}
