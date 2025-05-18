import time
import subprocess
import psutil
import sys
from pathlib import Path

# Directory paths
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS).resolve()
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()

RESTART_PLANET_PAINTER = True


def is_process_running(process_name):
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline and any(process_name in str(arg) for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def run_materials_processing():
    """Simulate material processing by waiting a few seconds."""
    print("Starting materials processing...")

    # Simulated processing time
    time.sleep(2)

    print("Materials processing complete.")

    if RESTART_PLANET_PAINTER and not is_process_running("PlanetPainter.py"):
        print("Restarting PlanetPainter...")
        subprocess.run(["python", str(BASE_DIR / "PlanetPainter.py")])

    sys.stdout.flush()
    sys.exit()


if __name__ == "__main__":
    run_materials_processing()
