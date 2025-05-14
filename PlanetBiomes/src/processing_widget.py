from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
from PySide6.QtGui import QMovie, QKeyEvent
from PySide6.QtCore import Qt
import sys
from pathlib import Path

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
GIF_PATH = IMAGE_DIR / "progress.gif"


class ProcessingDialog(QDialog):
    def __init__(self, title="Processing...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(150, 150)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 255);")

        layout = QVBoxLayout(self)
        self.label = QLabel(self)

        # Load the GIF
        if not GIF_PATH.exists():
            print(f"Error: GIF not found at {GIF_PATH}")
            self.label.setText("GIF not found")
        else:
            self.movie = QMovie(str(GIF_PATH))
            if not self.movie.isValid():
                print(f"Error: Failed to load GIF at {GIF_PATH}")
                self.label.setText("Invalid GIF")
            else:
                print(f"Successfully loaded GIF at {GIF_PATH}")
                self.label.setMovie(self.movie)
                self.label.setScaledContents(True)
                self.movie.start()

        layout.addWidget(self.label)
        self.setLayout(layout)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        print("Processing widget closed.")
        sys.exit(0)


def main():
    title = "Processing..."
    if len(sys.argv) > 1:
        title = sys.argv[1]

    app = QApplication([])
    progress_dialog = ProcessingDialog(title=title)
    progress_dialog.show()
    app.exec()


if __name__ == "__main__":
    main()
