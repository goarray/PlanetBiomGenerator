# PlanetNewsfeed.py
from PyQt6.QtGui import QTextCursor
from datetime import datetime
from PyQt6.QtWidgets import QApplication
import sys
import csv
import re
import json
from PlanetConstants import CONFIG_PATH, DEFAULT_CONFIG_PATH, PREVIEW_PATH, INPUT_DIR

# Shared global variables
news_count = 0
news_percent = 0
biom_percent = 0
text_percent = 0
total_news = 0
total_biom = 0
total_text = 0
total_other = 0
unique_planets = 0

config = {}


def save_json(CONFIG_PATH, data: dict):
    """Save dictionary data to a JSON file."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        handle_news(None, "error", f"Error saving JSON: {e}")


def format_message(message: str, kind: str = "info", timestamp: bool = False) -> str:
    prefix = f"[{datetime.now():%H:%M:%S}] " if timestamp else ""
    full_message = prefix + message

    styles = {
        "info": "<span style='font-weight: normal;'>{}</span>",
        "success": "<div align='center' <br><b>{}</b><br></div>",
        "error": "<div style='font-weight: bold; border: 1px solid; padding: 2px;'>{}</div>",
        "warn": "<i>{}</i>",
        "debug": "<span style='font-family: monospace;'>{}</span>",
        "header": "<div style='font-weight: bold; border-bottom: 1px solid; margin-top: 6px;'>{}</div>",
    }

    template = styles.get(kind, "{}")
    return template.format(full_message)


def update_news_count(main_window=None):
    """Increment news_count and update progress bar percentages."""
    global news_count, news_percent, biom_percent, text_percent
    global total_news, total_biom, total_text, total_other

    news_count += 1

    if total_news > 0:
        news_percent = (news_count / total_news) * 100

    # Biomes percentage (first N entries)
    if news_count <= total_biom:
        biom_percent = (news_count / total_biom) * 100
    else:
        biom_percent = 100.0

    # Textures percentage (starts after total_biom)
    if news_count > total_biom:
        completed = news_count - total_biom
        text_percent = min((completed / total_text) * 100, 100.0)
    else:
        text_percent = 0.0

    # Update progress bars if UI present
    if main_window:
        if hasattr(main_window, "news_count_progressBar"):
            main_window.news_count_progressBar.setValue(int(news_percent))
        if hasattr(main_window, "biom_count_progressBar"):
            main_window.biom_count_progressBar.setValue(int(biom_percent))
        if hasattr(main_window, "text_count_progressBar"):
            main_window.text_count_progressBar.setValue(int(text_percent))
    
    """print(
        f"news_count: {news_count}, "
        f"biom_percent: {biom_percent:.1f}%, "
        f"text_percent: {text_percent:.1f}%, "
        f"news_percent: {news_percent:.1f}%"
    )"""


def reset_news_count():
    """Reset news_count and news_percent."""
    global news_count, news_percent, biom_percent, text_percent
    news_count = 0
    news_percent = 0
    biom_percent = 0
    text_percent = 0


def handle_news(main_window, kind: str = "info", message: str = "", flush=False):
    """Handle news messages and update counters."""
    update_news_count(main_window)  # Always increment counter

    if not message:
        return

    kind_clean = kind.lower().strip()
    timestamp = kind_clean != "success"
    formatted = format_message(message, kind_clean, timestamp)

    if main_window:
        widget = getattr(
            main_window,
            "stdout_widget" if kind_clean == "success" else "stderr_widget",
            None,
        )
        if widget:
            widget.append(formatted)
            widget.moveCursor(QTextCursor.MoveOperation.End)
            return

    # Fallback to terminal output
    clean = re.sub(r"<[^>]+>", "", formatted)
    stream = sys.stdout if kind_clean == "success" else sys.stderr
    print(f"{clean}", file=stream, flush=flush)


def calc_biom_count(config):
    base = 10 * unique_planets  # base number of biomes
    if config.get("process_biomes", False):
        base += sum(
            2
            for key in [
                "enable_noise",
                "enable_distortion",
                "enable_biases",
                "enable_anomalies",
            ]
            if config.get(key, False)
        )
        if config.get("enable_tectonic_plates", False):
            base += 15
    return base


def calc_text_count(config):
    base = 127 + calc_biom_count(config) * unique_planets  # 127 = textures base
    if config.get("process_images", False):
        base += 18
        base += sum(
            2
            for key in [
                "enable_basic_filters",
                #"enable_texture_noise",
                #"enable_texture_edges",
                #"enable_texture_light",
                #"enable_texture_terrain",
            ]
            if config.get(key, False)
        )
    return base


def calc_other_count(config):
    base = (total_news - total_biom - total_text) * unique_planets
    # if config.get("process_other", False):
    #    base += 18
    #    base += sum(
    #        2
    #        for key in [
    #            "enable_basic_filters",
    #            #"enable_texture_noise",
    #            #"enable_texture_edges",
    #            #"enable_texture_light",
    #            #"enable_texture_terrain",
    #        ]
    #        if config.get(key, False)
    #    )
    return base


def load_global_config():
    global config, total_news, total_biom, total_text

    config_path = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

            total_biom = calc_biom_count(config)
            total_text = calc_text_count(config)
            total_news = total_biom + total_text + total_other

            handle_news(
                None,
                "info",
                f"Loaded config: biom={total_biom}, text={total_text}, total_news={total_news}",
            )
    except FileNotFoundError:
        handle_news(
            None, "error", f"Config file {config_path} not found. Using default totals."
        )
    except json.JSONDecodeError as e:
        handle_news(None, "error", f"Error parsing config file {config_path}: {e}")
        total_news = 351


# Initialize total_news after all functions are defined
load_global_config()


def precompute_total_news(config: dict):
    global unique_planets
    if config.get("enable_preview_mode", True):
        csv_files = [PREVIEW_PATH]
    else:
        csv_files = list(INPUT_DIR.glob("*.csv"))
        if not csv_files:
            csv_files = [PREVIEW_PATH]

    selected_index = min(config.get("plugin_selected", 0), max(len(csv_files) - 1, 0))
    selected_index = max(0, selected_index)
    input_path = csv_files[selected_index]

    planet_names = set()
    with open(input_path, newline="") as f:
        f.readline()
        reader = csv.DictReader(
            f, fieldnames=["PlanetName", "BIOM_FormID", "BIOM_EditorID"]
        )
        next(reader, None)
        for row in reader:
            name = row["PlanetName"].strip()
            if name:
                planet_names.add(name)

    unique_planets = len(planet_names)

    total = len(planet_names) * 120
    config["total_news"] = total
    save_json(CONFIG_PATH, config)

    # ✅ SET GLOBAL COUNTS HERE
    global total_news, total_biom, total_text, total_other
    total_news = total
    total_biom = calc_biom_count(config)
    total_text = calc_text_count(config)
    total_other = calc_other_count(config)

    return {
        "total_news": total_news,
        "total_biom": total_biom,
        "total_text": total_text,
        "total_other": total_other,
    }
