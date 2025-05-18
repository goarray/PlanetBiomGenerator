# Expiremental UI added (now QtPy6)

- .biom and .png files will now be created a a new /Output/ folder
- Most setting are adjustable for .biom creation, a simple planet texture pipeline added

# USE at your own risk! Mostly works pretty ok.

# Starfield Planet Biome Generator

### Quick Start
- Place in xEdit.exe directory > select planets in xEdit > run the `Starfield - ExportBiomes...` script in xEdit > move generated PlanetBiomes.csv to /PlanetBiomes/assets/ > run the PlanetBiomes.bat found in /PlanetBiomes/ > place the generated /my.esm/ directory in your /biomemaps/ directory.

----  

## Features

- Exports all biome FormIDs from selected planets in xEdit to a PlanetBiomes.csv

- Generates one [ANAM].biom file per planet

- **NEW** now attempts to organically distribute the biomes. The top biome in PNDT (or per planet name in the PlanetBiomes.csv) will be located around the equator, the last biome the pole.

- **New** generates planet """textures""" (seriously, I'm just figuring this out)

- Includes a batch tool to automate generation

- `Biomes.csv` includes all generic base game biomes and RGB color codes used by the texture generator. Excludes named biomes (like sandyearth, etc.)


## xEdit Automation Scripts

- `Starfield - ExportBiomesToPlanetBiomesCSV.pas` - Exports the selected planets’ biomes to PlanetBiomes.CSV

- `Starfield - AddBiomesFromBiomesCSV.pas` - Assigns 7 random generic biomes to selected planet(s)

- Tip: You can batch-assign specific biomes to new planets by editing the Biomes.csv and re-running the xEdit `AddBiomes..` script.

----

## Installation

Drop the two folders into your xEdit.exe directory:

- `/Edit Scripts/` → will auto-load into xEdit script list

- `/PlanetBiomes/` → scripts and batch runner live here

Note: This is not necessary for the tool to work. But, the export script will need added to xEdit and the /PlanetBiomes`/xEditOutput/PlanetBiomes.csv/` the script creates will need to be moved to the tool's `/PlanetBiomes/assets/` directory.

## How to Use

### 1. Export Biomes from Planet(s)

- In xEdit, select one or more planet records (even just base game if you want to try it out)

- Run the script: `Starfield - ExportBiomesToPlanetBiomesCSV`

- This generates the file: `/PlanetBiomes/xEditOutput/PlanetBiomes.csv` (in /xEdit/) (move this to /PlanetBiomes//assets/)

### 2. Generate .biom Files

- Run the batch script: `/PlanetBiomes/PlanetBiomes.bat`

- This will launch the .biom scrambler UI, press `Create Planet`

- This will create a `plugin` folder like: /PlanetBiomes/Output/`[yourpluginname.esm or .esp]/`

- This directory will contain one `[ANAM].biom` file per planet found in the PlanetBiones.csv

- **NEW** It will also create the directory /output/textures/`[yourpluginname.esm or esp]`, this will contain a North and South texture.png for each planet.

- Tip: Create you own PlanetBiome.csv using the template found in /PlanetBiomes/assets/ for granual generation.

### 3. Install the Generated Biomes

- Move the generated 'plugin' folder into your plugins directory: /Data/planetdata/biomemaps/`[yourpluginname.esm or .esp]/`

- Each .biom will now be recognized by the game when loading the corresponding planet.

- Do whatever with BiomePNGs, they're really bad and would probably make your planet cry.

----  

## Done!
