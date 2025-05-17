from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.uic import loadUi

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("PlanetBiomes.ui", self)  # ✅ Loads the .ui file directly

app = QApplication([])
window = MainWindow()
window.show()
app.exec()