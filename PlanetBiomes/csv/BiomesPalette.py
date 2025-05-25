from PIL import Image, ImageDraw, ImageFont
import csv

# Constants
GRID_SIZE = (10, 40)  # Adjust based on total biomes
TILE_WIDTH = 120
TILE_HEIGHT = 25
IMG_SIZE = (GRID_SIZE[0] * TILE_WIDTH, GRID_SIZE[1] * TILE_HEIGHT)

# Load biome colors from CSV
biome_colors = []

with open("Biomes.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 5:
            continue
        biome_id, biome_name, r, g, b = row[0], row[1], int(row[2]), int(row[3]), int(row[4])
        biome_colors.append((biome_name, (r, g, b)))

# Create image
palette_img = Image.new("RGB", IMG_SIZE, (255, 255, 255))
draw = ImageDraw.Draw(palette_img)

# Font for text labels
try:
    font = ImageFont.truetype("arial.ttf", 10)
except:
    font = ImageFont.load_default()

# Draw biomes as color tiles
for i, (biome_name, color) in enumerate(biome_colors):
    x = (i % GRID_SIZE[0]) * TILE_WIDTH
    y = (i // GRID_SIZE[0]) * TILE_HEIGHT
    draw.rectangle([x, y, x + TILE_WIDTH, y + TILE_HEIGHT], fill=color)
    draw.text((x + 1, y + 1), biome_name[:25], fill="black", font=font)

# Save or show the image
palette_img.save("BiomesPalette.png")
palette_img.show()
