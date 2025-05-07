@echo off
cd /d "%~dp0"

echo Running PlanetBiomes.py...
python PlanetBiomes.py

echo Running PlanetTextures.py...
python PlanetTextures.py

pause
