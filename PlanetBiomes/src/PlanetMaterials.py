import time
import subprocess
import psutil
import sys
from pathlib import Path


# Determine base directory depending on execution mode
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS).resolve()  # PyInstaller temp directory
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()

RESTART_PLANET_PAINTER = True


class ZeroPlanetWorks:
    def __init__(self, restart_on_finish=True):
        self.base_dir = self.get_base_dir()
        self.restart_planet_painter = restart_on_finish

    @staticmethod
    def get_base_dir():
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS).resolve()
        return Path(__file__).parent.parent.resolve()

    def is_process_running(self, process_name):
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline")
                if cmdline and any(process_name in str(arg) for arg in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def run_materials_processing(self):
        print("Materials invoice processing.")
        time.sleep(2)  # Simulated processing time
        print("Materials application approved.")

        if self.restart_planet_painter and not self.is_process_running(
            "PlanetPainter.py"
        ):
            print("Restarting PlanetPainter...")
            subprocess.run(["python", str(self.base_dir / "PlanetPainter.py")])

        sys.stdout.flush()

    # Dummy extension methods for future use
    def check_resources(self):
        print("Materials shipment received.")

    def export_report(self):
        print("Exporting processing report...")

    def log_status(self, message):
        print(f"{message}")


def main():
    print("=== Starting PlanetMaterials ===", flush=True)
    zpw = ZeroPlanetWorks()
    zpw.log_status("Materialization ordered.")
    zpw.check_resources()
    zpw.run_materials_processing()


if __name__ == "__main__":
    main()
