from PyQt6.QtGui import QTextCursor
from datetime import datetime
from PyQt6.QtWidgets import QApplication


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


def handle_news(main_window, kind: str, message: str, flush=False):
    import sys
    import re

    kind_clean = kind.lower().strip()  # Normalize kind
    timestamp = kind_clean != "success"
    formatted = format_message(message, kind_clean, timestamp)

    # Determine widget target by kind_clean
    if main_window:
        if kind_clean == "success":
            widget = getattr(main_window, "stdout_widget", None)
        else:
            widget = getattr(main_window, "stderr_widget", None)
        if widget:
            widget.append(formatted)
            widget.moveCursor(QTextCursor.MoveOperation.End)
            return

    # No GUI widget, fallback to terminal streams
    clean = re.sub(r"<[^>]+>", "", formatted)

    if kind_clean == "success":
        stream = sys.stdout
    else:
        stream = sys.stderr

    print(f"{clean}", file=stream, flush=flush)
