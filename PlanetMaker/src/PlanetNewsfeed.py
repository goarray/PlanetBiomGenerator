# PlanetNewsfeed.py
from PyQt6.QtGui import QTextCursor
from datetime import datetime
from PyQt6.QtWidgets import QApplication
import sys
import csv
import re
import json
from pathlib import Path
from typing import List
from PlanetConstants import CONFIG_PATH, DEFAULT_CONFIG_PATH, PREVIEW_PATH, INPUT_DIR

# Shared global variables
news_count = 0
news_percent = 0
make_percent = 0
text_percent = 0
total_news = 0
total_make = 0
total_text = 0
total_other = 0

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
    global news_count, news_percent, make_percent, text_percent
    global total_news, total_make, total_text, total_other

    news_count += 1

    if total_news > 0:
        news_percent = (news_count / total_news) * 100

    # Maker percentage (first N entries)
    if news_count <= total_make:
        make_percent = (news_count / total_make) * 100
    else:
        make_percent = 100.0

    # Textures percentage (starts after total_make)
    if news_count > total_make:
        completed = news_count - total_make
        text_percent = min((completed / (1 + total_text)) * 100, 100.0)
    else:
        text_percent = 0.0

    # Update progress bars if UI present
    if main_window:
        if hasattr(main_window, "news_count_progressBar"):
            main_window.news_count_progressBar.setValue(int(news_percent))
        if hasattr(main_window, "make_count_progressBar"):
            main_window.make_count_progressBar.setValue(int(make_percent))
        if hasattr(main_window, "text_count_progressBar"):
            main_window.text_count_progressBar.setValue(int(text_percent))
    
    """print(
        f"news_count: {news_count}, "
        f"make_percent: {make_percent:.1f}%, "
        f"text_percent: {text_percent:.1f}%, "
        f"news_percent: {news_percent:.1f}%"
    )"""

def reset_news_count():
    """Reset news_count and news_percent."""
    global news_count, news_percent, make_percent, text_percent
    news_count = 0
    news_percent = 0
    make_percent = 0
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


def calc_make_count(config: dict) -> int:
    base = 15 #(-39 shifted to planettextures)
    
    if config.get("run_planet_scripts", False):
        # Optional extras
        if config.get("run_planet_materials", False):
            base += 2
        if config.get("run_planet_meshes", False):
            base += 2
        if config.get("enable_tectonic_plates", False):
            base += 18
        if config.get("run_planet_textures", False):
            base += 2
        if config.get("run_planet_maker", False):
            base += 2

        # Baseline finalization steps
        base += 2

    return base


def calc_text_count(config: dict) -> int:
    base = 74 #(+32 taken from planetmaker)
    if config.get("process_images", False):
        if config.get("enable_basic_filters", False):
            base += 2
        if config.get("enable_texture_noise", False):
            base += 4
        if config.get("enable_texture_edges", False):
            base += 2
        if config.get("enable_texture_light", False):
            base += 4
        if config.get("enable_texture_terrain", False):
            base += 2

        base += 8

    return base


def calc_other_count(config: dict) -> int:
    base = 668  # (+2 taken from planetMaker)
    if config.get("process_other", False):
        base += 10  # baseline count
        base += sum(
            2
            for key in [

            ]
            if config.get(key, False)
        )
    return base


def load_global_config():
    global config, total_news, total_make, total_text

    config_path = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

            total_make = calc_make_count(config)
            total_text = calc_text_count(config)
            total_news = total_make + total_text + total_other

            handle_news(
                None,
                "info",
                f"Loaded config: make={total_make}, text={total_text}, total_news={total_news}",
            )
    
    except json.JSONDecodeError as e:
        handle_news(None, "error", f"Error parsing config file {config_path}: {e}")
        total_news = 100


def precompute_total_news(config: dict):

    # Calculate totals
    global total_news, total_make, total_text, total_other
    total_make = calc_make_count(config)
    total_text = calc_text_count(config)
    total_other = calc_other_count(config)
    total_news = total_make + total_text + total_other

    config["total_news"] = total_news
    save_json(CONFIG_PATH, config)

    return {
        "total_news": total_news,
        "total_make": total_make,
        "total_text": total_text,
        "total_other": total_other,
    }


# Initialize total_news after all functions are defined
load_global_config()
