import json
import sys
from pathlib import Path
from PlanetConstants import BASE_DIR, CONFIG_DIR, THEME_PATH, DEFAULT_THEME_PATH

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

    if THEME_PATH.exists():
        try:
            with open(THEME_PATH, "r") as theme_file:
                return json.load(theme_file)
        except json.JSONDecodeError as e:
            print(
                f"Error: Invalid JSON in {THEME_PATH}. Falling back to default themes. {e}"
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
raw_size = theme_data.get("slider_handle_height", 10)
SLIDER_HANDLE_HEIGHT = int(raw_size) if isinstance(raw_size, (int, float, str)) else 14
SLIDER_HANDLE_RADIUS = SLIDER_HANDLE_HEIGHT // 2
SLIDER_HANDLE_WIDTH = theme_data.get("slider_handle_width", 20)
SLIDER_HANDLE_MARGIN = theme_data.get("slider_handle_margin", -5)

THEMES = {
    theme_name: f"""
        QMainWindow, QWidget {{
            background-color: {theme.get('background', '#1e1e1e')}; 
            color: {theme.get('color', '#ffffff')}; 
            font-family: {theme.get('font_family', '{primary}, sans-serif').format(primary=FONT_PRIMARY, secondary=FONT_SECONDARY)};
            font-size: {theme.get('fonts', {}).get('size_small', FONT_SIZE_SMALL)};
        }}

        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {theme.get('input_background', theme.get('button_color', '#2a2b2f'))};
            color: {theme.get('color', '#ffffff')};
            border: 1px solid {theme.get('input_border', theme.get('border_color', '#6e9bac'))};
            border-radius: {theme.get('border_radius', 5)}px;
        }}

        QPushButton {{
            background-color: {theme.get('button_color', '#2c2e33')}; 
            color: {theme.get('color', '#ffffff')};
            selection-color: {theme.get('selection_color', '#ffffff')};
            border: {theme.get('border_width', 1)}px {theme.get('border_style', 'solid')} {theme.get('border_color', '#6e9bac')};
            font-size: {theme.get('fonts', {}).get('size_large', FONT_SIZE_LARGE)};
            padding: 5px;
            border-radius: {theme.get('border_radius', 5)}px;
        }}

        QPushButton:hover {{
            background-color: {theme.get('button_hover_color', theme['button_color'])};
            border: 1px solid {theme.get('hover_border_color', theme['color'])};
            color: {theme.get('hover_text_color', theme['color'])};
            font-weight: {theme.get('hover_font_weight', 'bold')};
        }}

        QPushButton.primary {{
            background-color: {theme.get('primary_hover_color', theme.get('button_hover_color', '#00ffff'))};
            color: {theme.get('background', '#1e1e1e')};
            border: 2px solid {theme.get('color', '#ffffff')};
            font-weight: bold;
        }}

        QPushButton.danger {{
            background-color: {theme.get('danger_hover_color', '#ff4c4c')};
            color: white;
            border: 1px solid #990000;
        }}

        QPushButton.flat {{
            background-color: transparent;
            border: none;
            color: {theme.get('color', '#ffffff')};
        }}

        QSlider::handle:horizontal {{
            width: {SLIDER_HANDLE_WIDTH}px;
            height: {SLIDER_HANDLE_HEIGHT}px;
            background: {theme['color']};
            border: 1px solid {theme['background']};
            border-radius: {SLIDER_HANDLE_RADIUS}px;
            margin: {SLIDER_HANDLE_MARGIN}px 0;
        }}

        QSlider::handle:vertical {{
            width: {SLIDER_HANDLE_WIDTH}px;
            height: {SLIDER_HANDLE_HEIGHT}px;
            background: {theme['color']};
            border: 1px solid {theme['background']};
            border-radius: {SLIDER_HANDLE_RADIUS}px;
            margin: 0 {SLIDER_HANDLE_MARGIN}px;
        }}

        QSlider::groove:horizontal {{
            height: 6px;
            background: {theme.get('border_color', '#6e9bac')};
            margin: 2px 0;
            border-radius: 3px;
        }}

        QSlider::sub-page:horizontal {{
            background: {theme.get('primary_hover_color', '#00FFFF')};
            border-radius: 3px;
        }}

        QSlider::sub-page:vertical {{
            background: {theme.get('primary_hover_color', '#00FFFF')};
            border-radius: 3px;
        }}

        QSlider::groove:vertical {{
            width: 6px;
            background: {theme.get('border_color', '#6e9bac')};
            margin: 0 2px;
            border-radius: 3px;
        }}

        QProgressBar {{
            border: 1px solid {theme.get("border_color", "#00FFFF")};
            border-radius: 2px;
            text-align: center;
            height: 5px;
            background: {theme.get('background', '#10141a')};
            color: {theme.get('color', '#FFFFFF')};
        }}

        QProgressBar::chunk {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {theme.get('primary_hover_color', '#00FFFF')},
                stop: 1 {theme.get('color', '#0099FF')}
            );
            border-radius: 4px;
        }}

        QLCDNumber {{
            color: {theme.get("color", "#00FFAA")};
            background-color: {theme.get("background", "#1e1e1e")};
            border: 1px solid {theme.get("primary_hover_color", "#00FFFF")};
            border-radius: 6px;
            padding: 4px;
        }}

        QCheckBox, QRadioButton {{
            spacing: 5px;
            color: {theme.get('color', '#ffffff')};
        }}

        QCheckBox::indicator, QRadioButton::indicator {{
            width: 14px;
            height: 14px;
        }}

        QCheckBox::indicator:checked {{
            background-color: {theme.get('checkbox_checked_color', theme.get('selection_color', '#00ffff'))};
            border: 1px solid {theme.get('border_color', '#6e9bac')};
        }}

        QRadioButton::indicator:checked {{
            background-color: {theme.get('radio_checked_color', theme.get('selection_color', '#00ffff'))};
            border: 1px solid {theme.get('border_color', '#6e9bac')};
        }}

        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {{
            background-color: {theme.get('button_color', '#2c2e33')};
            border: 1px solid {theme.get('border_color', '#6e9bac')};
        }}

        QComboBox {{
            padding: 3px;
            border: 1px solid {theme.get('border_color', '#6e9bac')};
            background-color: {theme.get('button_color', '#2c2e33')};
            color: {theme.get('color', '#ffffff')};
        }}

        QComboBox QAbstractItemView {{
            background-color: {theme.get('button_color', '#2c2e33')};
            selection-background-color: {theme.get('border_color', '#6e9bac')};
            selection-color: {theme.get('hover_text_color', '#ffffff')};
            color: {theme.get('color', '#ffffff')};
            border: {theme.get('background', '#1e1e1e')}; 
        }}

        QComboBox QAbstractItemView::item:hover {{
            background-color: {theme.get('button_hover_color', '#3a3f44')};
            color: {theme.get('hover_text_color', '#ffffff')};
            font-weight: {theme.get('hover_font_weight', 'bold')};
        }}

        QToolTip {{
            background-color: {theme.get('tooltip_background', '#33363a')};
            color: {theme.get('tooltip_color', '#d9f6ff')};
            border: 1px solid {theme.get('border_color', '#6e9bac')};
            padding: 4px;
            border-radius: 4px;
        }}

        QLabel#Header {{
            font-size: 16px;
            font-weight: bold;
            color: {theme.get('header_color', theme.get('color', '#ffffff'))};
            background-color: {theme.get('header_background', theme.get('background', '#262729'))};
            padding: 5px;

        }}

        QLabel {{
            color: {theme.get('color', '#ffffff')};
            border: 1px solid {theme.get("primary_hover_color", "#00FFFF")};
            font-size: {theme.get('fonts', {}).get('size_large', FONT_SIZE_LARGE)};
            padding: 5px;
            border-radius: {theme.get('border_radius', 5)}px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {theme.get('border_color', '#6e9bac')};
            background: #222;
            padding: 5px;
        }}

        QTabBar::tab {{
            background:{theme.get('background', '#1e1e1e')}; 
            color: {theme.get('tooltip_color', '#d9f6ff')};
            padding: 8px 15px;
            border: 1px solid {theme.get('border_color', '#6e9bac')};
            border-bottom: none; /* To blend with the pane */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }}

        QTabBar::tab:selected {{
            background: {theme.get('button_hover_color', '#3a3f44')};
            color: {theme.get('hover_text_color', theme['color'])};
            font-weight: {theme.get('hover_font_weight', 'bold')};
            border-color: 1px solid {theme.get('border_color', '#6e9bac')};
        }}

        QTabBar::tab:hover {{
            background: {theme.get('primary_hover_color', '#00FFFF')};
            color: {theme.get('hover_text_color', theme['color'])};
        }}

        /* Scoped styles only for images_tabwidget */
        QTabWidget#images_tab_widget QTabBar::tab {{
            font-size: 10px;
            height: 15px;
            width: 40px;
            padding: 3px 6px;
        }}

        QTabWidget#images_tab_widget::pane {{
            selection-background-color: {theme.get('border_color', '#6e9bac')};
            selection-color: {theme.get('hover_text_color', '#ffffff')};
            border: 1px solid {theme.get('border_color', '#6e9bac')};
            background: #1a1a1a;
        }}
    """
    for theme_name, theme in theme_data.items()
}
