# Starfield Planet Biome Generator

⚠️ This tool does not paint detailed landscapes or terrain.
It only generates ANAM.biom files for selected planets and wraps biomes around the equator.
Use other tools for visual/terrain biome editing.

----  

## Features

- Exports all biomes from selected Starfield planets in xEdit

- Generates one .biom file per planet for inclusion in your mod

- Includes a batch tool to automate generation

- Compatible with Starfield Creation Kit planet definitions

- - `Biomes.csv` includes all generic base game biomes, excluding named ones (like sandyearth, etc.)


## xEdit Automation Scripts

- `Starfield - ExportBiomesToPlanetBiomesCSV.pas` - Exports the selected planets’ biomes to CSV

- `Starfield - AddBiomesFromBiomesCSV.pas` - Assigns 7 random generic biomes to selected planet(s)

- Tip: You can batch-assign specific biomes to new planets by editing the Biomes.csv and re-running the xEdit `Add` script.

----

## Installation

Drop the two folders into your xEdit.exe directory:

- `/Edit Scripts/` → will auto-load into xEdit script list

- `/PlanetBiomes/` → script and batch runner live here

Note: This is not necessary for the tool to work.
But, the export script will need added to xEdit and the `/xEditOutput/PlanetBiomes.csv/` the script creates will need to be moved to the `/PlanetBiomes/` directory.

## How to Use

### 1. Export Biomes from Planet(s)

- In xEdit, select one or more planet records

-  Run the script: `Starfield - ExportBiomesToPlanetBiomesCSV`

- This generates the file: `/PlanetBiomes/xEditOutput/PlanetBiomes.csv` (in xEdit/)

### 2. Generate .biom Files

- Run the batch script: `/PlanetBiomes/PlanetBiomes.bat`

- This will create a folder like: `/PlanetBiomes/[yourpluginname.esm or .esp]/`

- It will contain one `[ANAM].biom` file per planet in the plugin

### 3. Install the Generated Biomes

- Move the generated plugin folder into your Starfield directory: `Starfield/Data/planetdata/biomemaps/[yourpluginname.esm or .esp]/`

- Each .biom will now be recognized by the game when loading the corresponding planet.

----  

## Done!

You’re now ready to populate and scan your planets in-game with proper biome references.

