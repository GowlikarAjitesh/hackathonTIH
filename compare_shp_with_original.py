import os
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

# -------------------------------------------------
# INPUT
# -------------------------------------------------

TIF_FILE = "/home/cs24m112/hackathon/dataset/feature_extraction/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif"

ORIGINAL_SHP_DIR = "/home/cs24m112/hackathon/dataset/feature_extraction/PB_training_dataSet_shp_file/shp-file"

RECONSTRUCTED_DIR = "reconstructed_shp"

OUTPUT_DIR = "comparison_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# LAYER MAPPING
# -------------------------------------------------

layers = {
    "buildings": "Built_Up_Area_typ.shp",
    "roads": "Road.shp",
    "road_center": "Road_Centre_Line.shp",
    "water": "Water_Body.shp",
    "water_line": "Water_Body_Line.shp",
    "water_point": "Waterbody_Point.shp",
    "utility": "Utility.shp",
    "utility_poly": "Utility_Poly_.shp",
    "railway": "Railway.shp",
    "bridge": "Bridge.shp"
}

# -------------------------------------------------
# GET TIFF BOUNDS
# -------------------------------------------------

with rasterio.open(TIF_FILE) as src:
    bounds = src.bounds
    crs = src.crs

print("TIFF bounds:", bounds)

# -------------------------------------------------
# COMPARISON LOOP
# -------------------------------------------------

for layer_name, file_name in layers.items():

    original_path = os.path.join(ORIGINAL_SHP_DIR, file_name)
    recon_path = os.path.join(RECONSTRUCTED_DIR, f"{layer_name}.shp")

    if not os.path.exists(original_path) or not os.path.exists(recon_path):
        print("Skipping:", layer_name)
        continue

    print("\n----------------------------------")
    print("Layer:", layer_name)

    original = gpd.read_file(original_path)
    recon = gpd.read_file(recon_path)

    original = original.to_crs(crs)
    recon = recon.to_crs(crs)

    # -------------------------------------------------
    # CLIP ORIGINAL TO TIFF AREA
    # -------------------------------------------------

    clipped = original.cx[
        bounds.left:bounds.right,
        bounds.bottom:bounds.top
    ]

    print("Original (clipped):", len(clipped))
    print("Reconstructed:", len(recon))

    # -------------------------------------------------
    # INTERSECTION
    # -------------------------------------------------

    intersection = gpd.overlay(clipped, recon, how="intersection", keep_geom_type=False)

    if len(intersection) > 0:
        inter_area = intersection.area.sum()
    else:
        inter_area = 0

    orig_area = clipped.area.sum()
    recon_area = recon.area.sum()

    iou = inter_area / (orig_area + recon_area - inter_area + 1e-9)

    print("IoU:", round(iou,4))

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(8,8))

    clipped.boundary.plot(ax=ax, color="blue", linewidth=1, label="Original (clipped)")
    recon.boundary.plot(ax=ax, color="red", linewidth=0.5, label="Reconstructed")

    ax.set_title(f"{layer_name} comparison")
    ax.legend()

    save_path = os.path.join(OUTPUT_DIR, f"{layer_name}_comparison.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Visualization saved:", save_path)

print("\nComparison finished")