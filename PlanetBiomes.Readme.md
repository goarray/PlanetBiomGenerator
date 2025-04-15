Note, this is not going to 'paint' a pretty landscape on the planet(s), you'll need to use a separate method / tool for that. This simple generates the ANAM.biom files needed for proper planets and wraps the biomes around the equator.

The Biomes.CSV file contains all of the base biomes minus named ones, eg, sandyearth. 

The included "Starfield - AddBiomesFromBiomesCSV" will add 7 generic (random) biomes to any selected planet(s) in xEdit from the Biomes.CSV (edit this to batch in xEdit).

----

Instructions:

Drop the two folders into your xEdit.exe directory. 

The scripts in /Edit Scripts/ should wind up alongside your user created scripts.

In xEdit, select the planet(s) you wish to generate the .biom files for, run the "Starfield - ExportBiomesToPlanetBiomesCSV" xEdit script. 

This will (re)generate a /PlanetBiomes/xEditOutput/PlanetBiomes.CSV which contains all of the biome info for the selected planet(s) 

Run /PlanetBiomes/PlanetBiomes.bat, this will create /[yourpluginname.esm/esp]/ and fill it with the new (ANAM).biom files.  

Drop the created folder into your: "/planetdata/biomemaps/" directory. 

Done.

