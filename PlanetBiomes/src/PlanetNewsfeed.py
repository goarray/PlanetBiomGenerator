from PyQt6.QtGui import QTextCursor
from datetime import datetime
from PyQt6.QtWidgets import QApplication
import sys, re

news_count = 0
news_percent = 0

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
    global news_count

    kind_clean = kind.lower().strip()
    timestamp = kind_clean != "success"
    formatted = format_message(message, kind_clean, timestamp)

    if main_window:
        main_window.news_count += 1
        news_count += 1
        

        # Use main_window.total_news from config
        if main_window.total_news > 0:
            main_window.news_percent = (main_window.news_count / main_window.total_news) * 100
            main_window.news_count_progressbar.setValue(int(main_window.news_percent))

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
