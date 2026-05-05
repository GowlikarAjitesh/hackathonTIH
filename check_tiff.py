import rasterio
try:
    with rasterio.open("/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif") as src:
        print("Success! File is readable.")
        print(src.profile)
except Exception as e:
    print(f"File is still broken: {e}")