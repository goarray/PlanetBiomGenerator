# PlanetNewsfeed.py
from PyQt6.QtGui import QTextCursor
from datetime import datetime
from PyQt6.QtWidgets import QApplication
import sys
import re
import json
from PlanetConstants import CONFIG_PATH, DEFAULT_CONFIG_PATH

# Shared global variables
news_count = 0
news_percent = 0
total_news = 37


def set_total_news_from_config(config):
    global total_news
    total_news = config.get("total_news", 37)


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
    """Increment news_count and update news_percent."""
    global news_count, news_percent
    news_count += 1
    if total_news > 0:
        news_percent = (news_count / total_news) * 100
    if main_window:
        main_window.news_count = news_count
        main_window.news_percent = news_percent
        if hasattr(main_window, "news_count_progressbar"):
            main_window.news_count_progressbar.setValue(int(news_percent))


def reset_news_count():
    """Reset news_count and news_percent."""
    global news_count, news_percent
    news_count = 0
    news_percent = 0


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


def load_global_config():
    """Load total_news from config.json and initialize global variables."""
    global total_news
    config_path = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            total_news = config.get("total_news", 37)
            handle_news(
                None, "info", f"Loaded total_news: {total_news} from {config_path}"
            )
    except FileNotFoundError:
        handle_news(
            None,
            "error",
            f"Config file {config_path} not found. Using default total_news: {total_news}",
        )
    except json.JSONDecodeError as e:
        handle_news(None, "error", f"Error parsing config file {config_path}: {e}")
        total_news = 37


# Initialize total_news after all functions are defined
load_global_config()
