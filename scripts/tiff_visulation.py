import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- PATHS ---
TIF_DIRS = [
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG",
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB",
]

SHP_DIRS = {
    "CG": "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/CG/cg_shp-file",
    "PB": "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/PB/pb_shp-file",
}

OUT_DIR = "/home/vishnu/Ajitesh/ajitesh/tiff_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

SCALE = 0.03  # 3% of original — increase for more detail, decrease if slow

LAYER_DEFS = {
    "Buildings": (["Built_Up_Area_type.shp", "Built_Up_Area_typ.shp"], "orange"),
    "Roads":     (["Road.shp", "Road_32643.shp"],                       "red"),
    "Water":     (["Water_Body.shp"],                                    "blue"),
}

# --- COLLECT ALL TIF FILES ---
def collect_tifs():
    tif_files = []
    for tif_dir in TIF_DIRS:
        region = Path(tif_dir).name  # "CG" or "PB"
        for f in os.listdir(tif_dir):
            if f.endswith(".tif"):
                tif_files.append((os.path.join(tif_dir, f), region))
    return tif_files

# --- PROCESS ONE TIF ---
def process_tif(args):
    tif_path, region = args
    tif_name = Path(tif_path).stem
    out_path = os.path.join(OUT_DIR, f"{tif_name}.png")

    if os.path.exists(out_path):
        print(f"  Already exists, skipping: {tif_name}")
        return tif_name, True

    try:
        print(f"  Processing: {tif_name}")

        # --- Read downsampled image ---
        with rasterio.open(tif_path) as src:
            out_h = int(src.height * SCALE)
            out_w = int(src.width  * SCALE)
            bands = min(src.count, 3)

            img = src.read(
                list(range(1, bands + 1)),
                out_shape=(bands, out_h, out_w),
            )
            crs    = src.crs
            bounds = src.bounds

        # Fix shape
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        img = np.transpose(img, (1, 2, 0))[:, :, :3]

        # Normalize
        img = img.astype(np.float32)
        p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
        img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)

        # --- Load shapefiles ---
        shp_dir = SHP_DIRS[region]
        layers  = {}
        for name, (files, color) in LAYER_DEFS.items():
            for f in files:
                path = os.path.join(shp_dir, f)
                if os.path.exists(path):
                    try:
                        gdf = gpd.read_file(path).to_crs(crs)
                        if not gdf.empty:
                            layers[name] = (gdf, color)
                    except:
                        pass
                    break

        # --- Plot: 3 panels ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"{tif_name}  |  Region: {region}", fontsize=13, fontweight="bold")

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Panel 1 — Raw image
        axes[0].imshow(img)
        axes[0].set_title("Raw Satellite Image\n(RGB pixels only)", fontsize=10)
        axes[0].axis("off")

        # Panel 2 — Annotations only (on white background)
        axes[1].set_facecolor("white")
        axes[1].set_xlim(bounds.left,  bounds.right)
        axes[1].set_ylim(bounds.bottom, bounds.top)
        legend_handles = []
        for name, (gdf, color) in layers.items():
            gdf.plot(ax=axes[1], color=color, alpha=0.7, linewidth=0.8)
            legend_handles.append(mpatches.Patch(color=color, label=name))
        axes[1].set_title("Shapefile Annotations (.shp)\n(what experts labeled)", fontsize=10)
        axes[1].legend(handles=legend_handles, loc="lower right", fontsize=8)
        axes[1].axis("off")

        # Panel 3 — Overlay
        axes[2].imshow(img, extent=extent, aspect='auto')
        for name, (gdf, color) in layers.items():
            gdf.plot(ax=axes[2], color=color, alpha=0.5, linewidth=0.8)
        axes[2].set_title("Overlay\n(image + annotations together)", fontsize=10)
        axes[2].legend(handles=legend_handles, loc="lower right", fontsize=8)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()

        print(f"  ✅ Saved: {tif_name}.png")
        return tif_name, True

    except Exception as e:
        print(f"  ❌ Failed: {tif_name} → {e}")
        return tif_name, False


if __name__ == "__main__":
    tif_files = collect_tifs()
    print(f"Found {len(tif_files)} TIF files")
    print(f"Saving visualizations to: {OUT_DIR}\n")

    # Use multiple CPU workers — one per TIF file
    workers = min(cpu_count(), len(tif_files))
    print(f"Using {workers} CPU workers...\n")

    with Pool(workers) as p:
        results = p.map(process_tif, tif_files)

    success = sum(1 for _, ok in results if ok)
    failed  = sum(1 for _, ok in results if not ok)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Saved : {success} visualizations")
    print(f"  Failed: {failed}")
    print(f"\nOpen folder: {OUT_DIR}")
    print("All PNGs are viewable in any image viewer")