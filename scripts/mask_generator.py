import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from pathlib import Path

# --- PATHS ---
# Add all your TIF files here
TIF_FILES = [
    # CG region
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/KUTRU_451189_AAKLANKA_451163_ORTHO.tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO(3).tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO(4).tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO(5).tif",
    # PB region
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/28996_NADALA_ORTHO(8).tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/37458_fattu_bhila_ortho_3857(6).tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/37774_bagga ortho_3857(7).tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/PINDORI MAYA SINGH-TUGALWAL_28456_ortho(10).tif",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/TIMMOWAL_37695_ORI(9).tif",
]

# Each region has its own SHP folder
SHP_DIRS = {
    "CG": "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/cg_shp-file",
    "PB": "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/pb_shp-file",
}

MASK_OUTPUT_DIR = "/home/vishnu/Ajitesh/ajitesh/newly_generated_masks"
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 10000

# --- ALL 10 CLASSES (merged from both your code and teammate's code) ---
# Format: "layer_name": ([possible filenames], class_id, all_touched)
# all_touched=True for thin features like roads, lines
LAYER_DEFS = {
    "buildings":    (["Built_Up_Area_type.shp", "Built_Up_Area_typ.shp"], 1,  False),
    "roads":        (["Road.shp", "Road_32643.shp"],                       2,  True ),  # thin → all_touched
    "road_center":  (["Road_Centre_Line.shp"],                             3,  True ),  # line → all_touched
    "water":        (["Water_Body.shp"],                                   4,  False),
    "water_line":   (["Water_Body_Line.shp"],                              5,  True ),  # line → all_touched
    "water_point":  (["Waterbody_Point.shp"],                              6,  True ),
    "utility":      (["Utility.shp"],                                      7,  True ),
    "utility_poly": (["Utility_Poly.shp", "Utility_Poly_.shp"],            8,  False),
    "railway":      (["Railway.shp"],                                      9,  True ),  # line → all_touched
    "bridge":       (["Bridge.shp"],                                       10, True ),
}

def get_region(tif_path):
    if "/CG/" in tif_path or "\\CG\\" in tif_path:
        return "CG"
    return "PB"

def load_shapes(shp_dir, crs, bounds):
    """Load all available shapefiles from a directory."""
    available = {p.name for p in Path(shp_dir).glob("*.shp")}
    print(f"  Available shapefiles: {available}")

    shapes_by_cid = {}
    loaded_layers = []
    skipped_layers = []

    for layer_name, (file_candidates, cid, all_touched) in LAYER_DEFS.items():
        matched = next((f for f in file_candidates if f in available), None)

        if matched is None:
            skipped_layers.append(layer_name)
            continue

        shp_path = os.path.join(shp_dir, matched)
        try:
            gdf = gpd.read_file(shp_path).to_crs(crs)

            # Clip to image bounds — ignore shapes outside image
            clipped = gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

            if clipped.empty:
                print(f"  ⚠️  {layer_name}: no features inside image bounds")
                skipped_layers.append(layer_name)
                continue

            valid = [
                g for g in clipped.geometry
                if g is not None and g.is_valid and not g.is_empty
            ]

            if not valid:
                skipped_layers.append(layer_name)
                continue

            shapes_by_cid[cid] = {
                "shapes":      [(geom, cid) for geom in valid],
                "all_touched": all_touched,
                "name":        layer_name,
            }
            loaded_layers.append(f"{layer_name}(class {cid}): {len(valid)} shapes")

        except Exception as e:
            print(f"  ❌ Failed to load {layer_name}: {e}")
            skipped_layers.append(layer_name)

    print(f"\n  Loaded layers:")
    for l in loaded_layers:
        print(f"    ✅ {l}")
    print(f"\n  Skipped layers (not found or empty):")
    for l in skipped_layers:
        print(f"    ⬜ {l}")

    return shapes_by_cid

def process_tif(tif_path):
    print(f"\n{'='*60}")
    print(f"Processing: {Path(tif_path).name}")
    print(f"{'='*60}")

    region  = get_region(tif_path)
    shp_dir = SHP_DIRS[region]
    print(f"Region: {region} | SHP dir: {shp_dir}")

    with rasterio.open(tif_path) as src:
        h, w           = src.height, src.width
        full_transform = src.transform
        crs            = src.crs
        bounds         = src.bounds

    print(f"Image size: {w} x {h}")

    # Load all shapefiles
    shapes_by_cid = load_shapes(shp_dir, crs, bounds)

    if not shapes_by_cid:
        print("  ❌ No valid layers found, skipping this TIF")
        return

    # Output profile with proper compression
    out_profile = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     w,
        "height":    h,
        "count":     1,
        "crs":       crs,
        "transform": full_transform,
        "compress":  "deflate",
        "predictor": 2,
        "zlevel":    6,
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff":   "YES",
    }

    base_name = Path(tif_path).stem
    out_path  = os.path.join(MASK_OUTPUT_DIR, f"{base_name}_mask.tif")

    print(f"\nWriting mask: {out_path}")

    with rasterio.open(out_path, "w", **out_profile) as dst:
        total_blocks = (h // CHUNK_SIZE + 1) * (w // CHUNK_SIZE + 1)
        block_num    = 0

        for y in range(0, h, CHUNK_SIZE):
            for x in range(0, w, CHUNK_SIZE):
                ah = min(CHUNK_SIZE, h - y)
                aw = min(CHUNK_SIZE, w - x)

                window         = Window(x, y, aw, ah)
                tile_transform = rasterio.windows.transform(window, full_transform)
                tile_mask      = np.zeros((ah, aw), dtype=np.uint8)

                for cid, info in shapes_by_cid.items():
                    try:
                        burned = rasterize(
                            info["shapes"],
                            out_shape=(ah, aw),
                            transform=tile_transform,
                            dtype="uint8",
                            fill=0,
                            all_touched=info["all_touched"],
                        )
                        tile_mask[burned == cid] = cid
                    except Exception as e:
                        print(f"  ⚠️  Rasterize error cid={cid}: {e}")

                dst.write(tile_mask, 1, window=window)
                block_num += 1
                print(f"  Block {block_num}/{total_blocks} y={y} x={x} done", end="\r")

    print(f"\n  ✅ Mask saved: {out_path}")

    # --- Quick class distribution check ---
    print(f"\n  Checking class distribution...")
    with rasterio.open(out_path) as src:
        # Read downsampled for quick check
        scale   = 0.05
        check_h = int(h * scale)
        check_w = int(w * scale)
        sample  = src.read(1, out_shape=(check_h, check_w))

    total = sample.size
    print(f"  Class distribution (sampled at 5%):")
    print(f"    Class  0 background : {np.sum(sample==0)/total*100:.2f}%")
    for cid, info in shapes_by_cid.items():
        pct = np.sum(sample == cid) / total * 100
        print(f"    Class {cid:2d} {info['name']:15s}: {pct:.4f}%")


def main():
    print(f"Processing {len(TIF_FILES)} TIF files\n")

    success = 0
    failed  = []

    for tif_path in TIF_FILES:
        if not os.path.exists(tif_path):
            print(f"⚠️  File not found: {tif_path}")
            failed.append(tif_path)
            continue
        try:
            process_tif(tif_path)
            success += 1
        except Exception as e:
            print(f"❌ Failed: {Path(tif_path).name} → {e}")
            failed.append(tif_path)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Success : {success}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        for f in failed:
            print(f"    ❌ {Path(f).name}")
    print(f"\nMasks saved to: {MASK_OUTPUT_DIR}")

if __name__ == "__main__":
    main()