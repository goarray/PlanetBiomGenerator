# Starfield Planet Biome Generator

⚠️ This tool does not paint your planet! ⚠️

It only generates ANAM.biom files for planets and wraps the biomes around the equator.
Use other tools for actual biome placement.

----  

## Features

- Exports all biome FormIDs from selected planets in xEdit to a PlanetBiomes.csv

- Generates one [ANAM].biom file per planet

- Includes a batch tool to automate generation

- `Biomes.csv` includes all generic base game biomes, excluding named ones (like sandyearth, etc.)


## xEdit Automation Scripts

- `Starfield - ExportBiomesToPlanetBiomesCSV.pas` - Exports the selected planets’ biomes to PlanetBiomes.CSV

- `Starfield - AddBiomesFromBiomesCSV.pas` - Assigns 7 random generic biomes to selected planet(s)

- Tip: You can batch-assign specific biomes to new planets by editing the Biomes.csv and re-running the xEdit `AddBiomes..` script.

----

## Installation

Drop the two folders into your xEdit.exe directory:

- `/Edit Scripts/` → will auto-load into xEdit script list

- `/PlanetBiomes/` → script and batch runner live here

Note: This is not necessary for the tool to work. But, the export script will need added to xEdit and the /PlanetBiomes`/xEditOutput/PlanetBiomes.csv/` the script creates will need to be moved to the tool's `/PlanetBiomes/` directory.

## How to Use

### 1. Export Biomes from Planet(s)

- In xEdit, select one or more planet records

-  Run the script: `Starfield - ExportBiomesToPlanetBiomesCSV`

- This generates the file: `/PlanetBiomes/xEditOutput/PlanetBiomes.csv` (in xEdit/)

### 2. Generate .biom Files

- Run the batch script: `/PlanetBiomes/PlanetBiomes.bat`

- This will create a `plugin` folder like: /PlanetBiomes/`[yourpluginname.esm or .esp]/`

- It will contain one `[ANAM].biom` file per planet in the plugin

- Tip: Create you own PlanetBiome.csv using the template found in /PlanetBiomes.csv.Template/ for granual generation.

### 3. Install the Generated Biomes

- Move the generated 'plugin' folder into your plugins directory: /Data/planetdata/biomemaps/`[yourpluginname.esm or .esp]/`

- Each .biom will now be recognized by the game when loading the corresponding planet.

----  

## Done!

