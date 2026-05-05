import geopandas as gpd
import matplotlib.pyplot as plt

folder = "/home/cs24m119/hackathon/shp-file/"

files = [
"Road.shp",
"Railway.shp",
"Bridge.shp",
"Water_Body.shp",
"Built_Up_Area_type.shp"
]

plt.figure(figsize=(12,12))

for f in files:
    gdf = gpd.read_file(folder + f)
    gdf.plot()

plt.title("All SHP Layers Overview")
plt.savefig("dataset_overview.png")