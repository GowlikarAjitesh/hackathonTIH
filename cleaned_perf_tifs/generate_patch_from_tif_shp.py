import os
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from multiprocessing import Pool
from tqdm import tqdm
import geopandas as gpd
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ================= CONFIG ================= #
BASE_DIR = "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs"
PATCH_DIR = os.path.join(BASE_DIR, "generated_paches")

PATCH_SIZE = 512
NUM_WORKERS = 6
SAVE_VISUALS = True   # 🔥 toggle ON/OFF

os.makedirs(PATCH_DIR, exist_ok=True)

# ================= CLASS MAP ================= #
CLASS_MAP = {
    "Bridge": 1,
    "Railway": 2,
    "Road": 3,
    "Road_Centre_Line": 4,
    "Built_Up_Area_type": 5,
    "Utility": 6,
    "Utility_Poly": 7,
    "Water_Body": 8,
    "Water_Body_Line": 9,
    "Waterbody_Point": 10
}

# ================= NORMALIZE ================= #
def normalize_name(name):
    name = name.replace(".shp", "")
    if "Road_" in name:
        return "Road"
    if "Built_Up_Area_typ" in name:
        return "Built_Up_Area_type"
    if "Utility_Poly_" in name:
        return "Utility_Poly"
    return name

# ================= LOAD SHAPES ================= #
def load_shapes(shp_dir):
    shapes = []

    for file in os.listdir(shp_dir):
        if not file.endswith(".shp"):
            continue

        class_name = normalize_name(file)

        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]

        try:
            gdf = gpd.read_file(os.path.join(shp_dir, file))

            # fix invalid geometry
            gdf["geometry"] = gdf["geometry"].buffer(0)

            for geom in gdf.geometry:
                if geom is not None and not geom.is_empty:
                    shapes.append((geom, class_id))

        except:
            continue

    return shapes

# ================= VISUAL ================= #
def create_visual(img, mask):
    img = img.transpose(1, 2, 0)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    vis = img.copy().astype(np.float32)

    # fixed colors (better than random)
    COLORS = {
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [255, 255, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
        7: [128, 128, 0],
        8: [0, 128, 255],
        9: [255, 128, 0],
        10: [128, 0, 255]
    }

    for cls in np.unique(mask):
        if cls == 0:
            continue
        vis[mask == cls] = 0.6 * vis[mask == cls] + 0.4 * np.array(COLORS.get(cls, [255,255,255]))

    return vis.astype(np.uint8)

# ================= PROCESS ================= #
def process_tif(args):

    tif_path, shp_dir = args

    try:
        with rasterio.open(tif_path) as src:

            height, width = src.height, src.width
            transform = src.transform

            if height == 0 or width == 0:
                return 0

            shapes = load_shapes(shp_dir)
            if len(shapes) == 0:
                return 0

            mask = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8"
            )

            tif_name = os.path.splitext(os.path.basename(tif_path))[0]

            img_dir = os.path.join(PATCH_DIR, tif_name, "images")
            mask_dir = os.path.join(PATCH_DIR, tif_name, "masks")
            vis_dir = os.path.join(PATCH_DIR, tif_name, "visuals")

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            if SAVE_VISUALS:
                os.makedirs(vis_dir, exist_ok=True)

            saved = 0

            for y in range(0, height, PATCH_SIZE):
                for x in range(0, width, PATCH_SIZE):

                    try:
                        window = Window(x, y, PATCH_SIZE, PATCH_SIZE)

                        img_patch = src.read(window=window)

                        if img_patch.shape[1] != PATCH_SIZE:
                            continue

                        mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                        if mask_patch.size == 0:
                            continue

                        # 🔥 skip empty patches
                        if np.max(mask_patch) == 0:
                            continue

                        patch_name = f"{y}_{x}.tif"

                        img_file = os.path.join(img_dir, patch_name)
                        mask_file = os.path.join(mask_dir, patch_name)

                        # ===== SAVE IMAGE ===== #
                        with rasterio.open(
                            img_file,
                            "w",
                            driver="GTiff",
                            height=PATCH_SIZE,
                            width=PATCH_SIZE,
                            count=img_patch.shape[0],
                            dtype=img_patch.dtype,
                            transform=transform,
                            compress="lzw"
                        ) as dst:
                            dst.write(img_patch)

                        # ===== SAVE MASK ===== #
                        with rasterio.open(
                            mask_file,
                            "w",
                            driver="GTiff",
                            height=PATCH_SIZE,
                            width=PATCH_SIZE,
                            count=1,
                            dtype="uint8",
                            transform=transform,
                            compress="lzw"
                        ) as dst:
                            dst.write(mask_patch, 1)

                        # ===== SAVE VISUAL ===== #
                        if SAVE_VISUALS:
                            vis = create_visual(img_patch, mask_patch)

                            vis_file = os.path.join(
                                vis_dir,
                                patch_name.replace(".tif", ".png")
                            )

                            plt.imshow(vis)
                            plt.axis("off")
                            plt.savefig(vis_file, bbox_inches='tight')
                            plt.close()

                        saved += 1

                    except:
                        continue

        return saved

    except:
        print(f"❌ Skipping {os.path.basename(tif_path)}")
        return 0

# ================= TASKS ================= #
def collect_tasks():

    tasks = []

    pb_dir = os.path.join(BASE_DIR, "_PB")
    pb_shp = os.path.join(pb_dir, "pb_shp-file")

    for f in os.listdir(pb_dir):
        if f.endswith(".tif"):
            tasks.append((os.path.join(pb_dir, f), pb_shp))

    return tasks

# ================= MAIN ================= #
if __name__ == "__main__":

    tasks = collect_tasks()

    print(f"🚀 Total TIFFs: {len(tasks)}")

    start = time.time()
    total_saved = 0

    with Pool(NUM_WORKERS) as p:
        for res in tqdm(p.imap(process_tif, tasks), total=len(tasks)):
            total_saved += res

    print("\n✅ DONE")
    print(f"🧩 Total patches: {total_saved}")
    print(f"⏱️ Time: {str(timedelta(seconds=int(time.time()-start)))}")