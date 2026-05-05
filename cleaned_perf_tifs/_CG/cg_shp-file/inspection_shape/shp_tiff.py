import geopandas as gpd
import rasterio

shp_path = "/home/cs24m119/hackathon/shp-file/inspection_shape/Bridge.shp"
tif_path = "/home/cs24m119/hackathon/Training_dataSet_2/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif"

roads = gpd.read_file(shp_path)

with rasterio.open(tif_path) as src:
    raster_crs = src.crs

roads = roads.to_crs(raster_crs)

print("New CRS:", roads.crs)

from shapely.geometry import box

with rasterio.open(tif_path) as src:

    bounds = src.bounds

tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

roads_tile = roads.clip(tile_box)

print("Shapes inside tile:", len(roads_tile))

from rasterio.features import rasterize
import numpy as np

with rasterio.open(tif_path) as src:

    mask = rasterize(

        [(geom, 1) for geom in roads_tile.geometry],

        out_shape=(src.height, src.width),

        transform=src.transform,

        fill=0,

        dtype=np.uint8
    )
with rasterio.open(
    "mask.tif",
    "w",
    driver="GTiff",
    height=mask.shape[0],
    width=mask.shape[1],
    count=1,
    dtype="uint8",
    crs=src.crs,
    transform=src.transform
) as dst:

    dst.write(mask, 1)
import matplotlib.pyplot as plt

plt.imshow(mask)
plt.title("Rasterized Mask")
plt.show()