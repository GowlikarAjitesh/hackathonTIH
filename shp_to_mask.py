import os
import sys

# --- FORCED CUDA PATH (MUST BE FIRST) ---
def set_cuda_env():
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/local/cuda/bin"
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

set_cuda_env() # Run for the main process
# ----------------------------------------

import dask.array as da
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import cupy as cp
from pathlib import Path

TIF_FILE = "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif"
SHP_DIR = "/home/vishnu/Ajitesh/ajitesh/cg_shp-file"
MASK_OUTPUT_DIR = "/home/vishnu/Ajitesh/ajitesh/generated_masks"
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

def process_block(block, block_info=None, shapes_by_cid=None, transform=None):
    # Ensure workers know where CUDA is
    set_cuda_env()
    
    if block_info is None or not shapes_by_cid:
        return block

    loc = block_info[0]['array-location']
    y_start, y_end = loc[0]
    x_start, x_end = loc[1]
    
    tile_window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
    tile_transform = rasterio.windows.transform(tile_window, transform)
    
    h, w = block.shape
    # Initializing on GPU
    gpu_mask = cp.zeros((h, w), dtype=cp.uint8)

    for cid, shapes in shapes_by_cid.items():
        tile_raster = rasterize(
            shapes,
            out_shape=(h, w),
            transform=tile_transform,
            dtype="uint8"
        )
        # Using the direct assignment to avoid complex kernel compilation
        gpu_layer = cp.array(tile_raster)
        gpu_mask[gpu_layer == cid] = cid
        del gpu_layer

    res = cp.asnumpy(gpu_mask)
    cp.get_default_memory_pool().free_all_blocks()
    return res

def main():
    print("Opening BigTIFF...")
    with rasterio.open(TIF_FILE) as src:
        h, w = src.height, src.width
        profile = src.profile
        full_transform = src.transform
        crs = src.crs

    print("Loading shapefiles...")
    layer_defs = {
        "buildings": (["Built_Up_Area_type.shp", "Built_Up_Area_typ.shp"], 1),
        "roads": (["Road.shp", "Road_32643.shp"], 2),
        "water": (["Water_Body.shp"], 4),
    }

    shapes_by_cid = {}
    for name, (files, cid) in layer_defs.items():
        for f in files:
            path = os.path.join(SHP_DIR, f)
            if os.path.exists(path):
                gdf = gpd.read_file(path).to_crs(crs)
                shapes_by_cid[cid] = [(geom, cid) for geom in gdf.geometry if geom is not None]
                print(f"  Loaded {name}")
                break

    # Using 10k x 10k chunks
    dask_arr = da.zeros((h, w), chunks=(10000, 10000), dtype=np.uint8)

    print("Executing GPU Graph...")
    final_graph = dask_arr.map_blocks(
        process_block, 
        shapes_by_cid=shapes_by_cid, 
        transform=full_transform, 
        dtype=np.uint8
    )

    base_name = Path(TIF_FILE).stem
    out_path = os.path.join(MASK_OUTPUT_DIR, f"{base_name}_FINAL.tif")
    profile.update(count=1, dtype='uint8', compress='lzw', bigtiff='YES')

    with rasterio.open(out_path, "w", **profile) as dst:
        da.store(final_graph, dst, lock=False)

    print(f"Success! Result: {out_path}")

if __name__ == "__main__":
    main()