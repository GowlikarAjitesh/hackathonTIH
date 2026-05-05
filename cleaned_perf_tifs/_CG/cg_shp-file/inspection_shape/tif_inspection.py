import rasterio

with rasterio.open("/home/cs24m119/hackathon/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif") as src:
    print(src.crs)
    print(src.bounds)
    print(src.width, src.height)