import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

# -------------------------------------------------
# INPUT
# -------------------------------------------------

MASK_PATH = "generated_masks/28996_NADALA_ORTHO_mask.tif"

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------

OUTPUT_DIR = "reconstructed_shp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIS_DIR = "reconstructed_visualization"
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------------------------------------
# CLASS MAPPING (same as shp_to_mask)
# -------------------------------------------------

layers = {
    1: "buildings",
    2: "roads",
    3: "road_center",
    4: "water",
    5: "water_line",
    6: "water_point",
    7: "utility",
    8: "utility_poly",
    9: "railway",
    10: "bridge"
}

# -------------------------------------------------
# LOAD MASK
# -------------------------------------------------

print("\nOpening mask...")

with rasterio.open(MASK_PATH) as src:

    mask = src.read(1)
    transform = src.transform
    crs = src.crs

print("Mask loaded")

# -------------------------------------------------
# VECTORIZE MASK
# -------------------------------------------------

print("\nConverting mask to polygons...")

polygons = []

for geom, value in shapes(mask.astype(np.int16), transform=transform):

    if value == 0:
        continue

    polygons.append({
        "geometry": shape(geom),
        "class_id": int(value),
        "class_name": layers.get(int(value), "unknown")
    })

print("Total extracted polygons:", len(polygons))

# -------------------------------------------------
# CREATE GEODATAFRAME
# -------------------------------------------------

gdf = gpd.GeoDataFrame(polygons, crs=crs)

# -------------------------------------------------
# SAVE PER CLASS SHAPEFILES
# -------------------------------------------------

print("\nSaving shapefiles...")

for class_id, class_name in layers.items():

    class_gdf = gdf[gdf["class_id"] == class_id]

    if len(class_gdf) == 0:
        continue

    shp_path = os.path.join(OUTPUT_DIR, f"{class_name}.shp")

    class_gdf.to_file(shp_path)

    print(class_name, "features:", len(class_gdf))

print("\nShapefiles saved to:", OUTPUT_DIR)

# -------------------------------------------------
# CORRECTNESS CHECK
# -------------------------------------------------

print("\nRunning correctness checks...")

unique_classes = np.unique(mask)

print("Classes present in mask:", unique_classes)

for class_id in layers.keys():

    pixel_count = np.sum(mask == class_id)

    print(f"{layers[class_id]} pixels:", pixel_count)

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------

print("\nGenerating visualization...")

fig, ax = plt.subplots(figsize=(10,10))

ax.imshow(mask, cmap="tab20")

for class_id, class_name in layers.items():

    class_gdf = gdf[gdf["class_id"] == class_id]

    if len(class_gdf) == 0:
        continue

    class_gdf.boundary.plot(ax=ax, linewidth=0.5)

ax.set_title("Mask → SHP Reconstruction")
ax.axis("off")

vis_path = os.path.join(VIS_DIR, "mask_to_shp_visualization.png")

plt.savefig(vis_path, dpi=300, bbox_inches="tight")

plt.close()

print("Visualization saved at = ", vis_path)