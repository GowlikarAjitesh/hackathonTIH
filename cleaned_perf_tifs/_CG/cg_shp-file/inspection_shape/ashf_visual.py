import geopandas as gpd

SHP = "/home/cs24m119/hackathon/shp-file/Bridge.shp"

gdf = gpd.read_file(SHP)

print("\nColumns:")
print(gdf.columns)

print("\nFirst rows:")
print(gdf.head())

print("\nTotal objects:", len(gdf))

print("\nGeometry types:")
print(gdf.geom_type.value_counts())

print("\nCRS:")
print(gdf.crs)

print("\nBounds:")
print(gdf.total_bounds)