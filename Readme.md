🌍 Starfield Planet Biome Generator
⚠️ This tool does not paint detailed terrain or generate landscape detail.
It only generates ANAM.biom files with generic biome layouts around the equator.
For detailed landscape painting, use additional modding tools.

🔧 What It Does
Exports biome info from selected planets using xEdit

Generates .biom files based on those planets

Outputs files into the correct biomemaps/ folder structure

Includes helper scripts for batch processing and testing

📁 Files Included
PlanetBiomes.bat — Main batch runner

PlanetBiomes.py — Python processor

Biomes.csv — List of base game generic biomes

Edit Scripts/ — xEdit scripts:

Starfield - ExportBiomesToPlanetBiomesCSV.pas

Starfield - AddBiomesFromBiomesCSV.pas

🚀 Quick Start
1. Install
Drop both folders into your xEdit directory:

swift
Copy
Edit
/Edit Scripts/
/PlanetBiomes/
The xEdit scripts will now appear under your script list.

2. Export Biomes from a Planet
Open xEdit and load your Starfield plugin(s)

Select the planet(s) you want

Run:

nginx
Copy
Edit
Starfield - ExportBiomesToPlanetBiomesCSV
This generates:

bash
Copy
Edit
/PlanetBiomes/xEditOutput/PlanetBiomes.csv
3. Generate .biom Files
Run the batch file:

swift
Copy
Edit
/PlanetBiomes/PlanetBiomes.bat
This creates a folder:

css
Copy
Edit
/PlanetBiomes/[YourPluginName.esm or .esp]/
Each planet gets its own .biom file.

4. Install .biom Files
Move your output folder into Starfield’s biome path:

swift
Copy
Edit
Starfield/Data/planetdata/biomemaps/[YourPluginName.esm or .esp]/
