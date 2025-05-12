BiomePalette.py
- Creates a palette of EditorID labeled colors from Biomes.csv 

BiomePalette.png
- Palette created from running the above script

PlanetBiomes.biom
- Template .biom file for use by PlanetBiomes.py

PlanetBiomes.csv
- The csv containing planets to build the .biom files and images from
- example:
```sh
examplePlanet.esm
PlanetName,BIOM_FormID,BIOM_EditorID,ResourceID
examplePlanet,0007F88B,OceanLife01,00
examplePlanet,001DFE07,MountainsLife15,00
examplePlanet,0025FC04,ForestTropicalLife01,00
examplePlanet,002C9984,VolcanicRoughLife03,00
examplePlanet,0018A89C,SavannaLife10,00
examplePlanet,00285AFB,DesertRockyLifeExtreme,00
```

Biomes.csv
- The csv containing the various 'generic' biomes found in starfield
- Each FormID should be followed by EditorID and R,G,B values
- example:
```sh
000FD012,ArchipelagoLife01,122,231,240
000FD00F,ArchipelagoLife02,132,236,237
000FD010,ArchipelagoLife03,142,239,233
000FD015,ArchipelagoLife04,152,240,228
000F1F0D,ArchipelagoLife05,162,238,222
000F1F17,ArchipelagoLife06,172,234,215
```