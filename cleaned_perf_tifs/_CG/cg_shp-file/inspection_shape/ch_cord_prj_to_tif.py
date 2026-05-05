import os
import geopandas as gpd
import rasterio
import time

# -------- 1. CONFIGURATION --------
# The TIF file provides the "Master" coordinate system
tif_path = "/home/cs24m119/hackathon/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif"

# Source of shapefiles
input_shp_dir = "/home/cs24m119/hackathon/shp-file"

# Where you want to save the correctly aligned shapefiles
output_shp_dir = "/home/cs24m119/hackathon/shp-file/inspection_shape"
os.makedirs(output_shp_dir, exist_ok=True)

# -------- 2. DYNAMIC CRS EXTRACTION --------
with rasterio.open(tif_path) as src:
    target_crs = src.crs  # Automatically gets EPSG:32644 or whatever the TIF is

print(f"Master Projection (from TIF): {target_crs}")

# -------- 3. BATCH PROCESSING --------
shp_files = [f for f in os.listdir(input_shp_dir) if f.endswith('.shp')]
print(f"Found {len(shp_files)} shapefiles. Syncing projections...")

for shp_file in shp_files:
    start = time.time()
    input_path = os.path.join(input_shp_dir, shp_file)
    output_path = os.path.join(output_shp_dir, shp_file)
    
    try:
        # Load vector data
        gdf = gpd.read_file(input_path)
        
        # Check if the Shapefile matches the TIF
        if gdf.crs != target_crs:
            print(f"🔄 Syncing {shp_file}: {gdf.crs} -> {target_crs}")
            # The 'to_crs' function uses the TIF's specific CRS object
            gdf_converted = gdf.to_crs(target_crs)
            gdf_converted.to_file(output_path)
        else:
            print(f"✅ {shp_file} already matches. Copying...")
            gdf.to_file(output_path)
            
        print(f"   Completed in {time.time() - start:.2f}s")
            
    except Exception as e:
        print(f"❌ Error syncing {shp_file}: {e}")

print(f"\nAll files are now synced to the TIF's projection and saved in: {output_shp_dir}")