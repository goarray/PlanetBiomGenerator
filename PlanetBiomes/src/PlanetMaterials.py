import time
import subprocess
import psutil
import sys
from pathlib import Path

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
    print("Processing materials...")
    time.sleep(3)
    print("Materials processing complete.")

    if RESTART_PLANET_PAINTER and not is_process_running("PlanetPainter.py"):
        print("Restarting PlanetPainter...")
        subprocess.Popen(["python", str(Path(__file__).parent / "PlanetPainter.py")])

    sys.stdout.flush()
    sys.exit()


if __name__ == "__main__":
    run_materials_processing()
