import geopandas as gpd
import rasterio

# Paths
shp_path = "Road.shp"
tif_path = "/home/cs24m119/hackathon/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif"

# 1. Check Shapefile CRS
roads = gpd.read_file(shp_path)
shp_crs = roads.crs

# 2. Check TIF CRS (Using rasterio)
with rasterio.open(tif_path) as src:
    tif_crs = src.crs

# 3. Compare and Print Results
print(f"{'Data Source':<15} | {'EPSG / CRS Definition'}")
print("-" * 50)
print(f"{'Roads (SHP)':<15} | {shp_crs}")
print(f"{'Ortho (TIF)':<15} | {tif_crs}")

# Direct Comparison Logic
if shp_crs == tif_crs:
    print("\n✅ MATCH: The coordinate systems are the same. They will overlap perfectly.")
else:
    print("\n❌ MISMATCH: The coordinate systems are different.")
    print(f"To fix this, use: roads = roads.to_crs('{tif_crs}')")

# Extra Info
print("\n--- Road Geometry Info ---")
print(roads.geom_type.value_counts())