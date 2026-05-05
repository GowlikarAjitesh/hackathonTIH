import os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.windows import Window
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

IMAGE_PATH = "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/TIMMOWAL_37695_ORI(9).tif"
MASK_PATH  = "/home/vishnu/Ajitesh/ajitesh/generated_masks/TIMMOWAL_37695_ORI(9)_mask.tif"

PATCH_SIZE     = 512
MIN_ANNO_RATIO = 0.02   # at least 2% non-background pixels — key filter

base_name = Path(IMAGE_PATH).stem
OUT_DIR   = f"/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/{base_name}"
IMG_DIR   = os.path.join(OUT_DIR, "images")
MASK_DIR  = os.path.join(OUT_DIR, "masks")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

with rasterio.open(MASK_PATH) as src:
    height, width = src.height, src.width

img_src  = None
mask_src = None

def init_worker():
    global img_src, mask_src
    img_src  = rasterio.open(IMAGE_PATH)
    mask_src = rasterio.open(MASK_PATH)

def process_patch(coords):
    y, x = coords
    window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
    try:
        mask_patch = mask_src.read(1, window=window)

        # ✅ Skip if patch is too small (edge) 
        if mask_patch.shape != (PATCH_SIZE, PATCH_SIZE):
            return 0

        # ✅ Skip if not enough annotation
        ratio = np.sum(mask_patch > 0) / mask_patch.size
        if ratio < MIN_ANNO_RATIO:
            return 0

        img_patch = img_src.read(window=window)

        # ✅ Save
        with rasterio.open(
            os.path.join(IMG_DIR, f"img_{y}_{x}.tif"), "w",
            driver="GTiff", height=PATCH_SIZE, width=PATCH_SIZE,
            count=img_patch.shape[0], dtype=img_patch.dtype
        ) as dst:
            dst.write(img_patch)

        with rasterio.open(
            os.path.join(MASK_DIR, f"mask_{y}_{x}.tif"), "w",
            driver="GTiff", height=PATCH_SIZE, width=PATCH_SIZE,
            count=1, dtype="uint8"
        ) as dst:
            dst.write(mask_patch, 1)

        return 1
    except Exception as e:
        return 0

if __name__ == "__main__":
    coords = [(y, x) for y in range(0, height, PATCH_SIZE)
                      for x in range(0, width, PATCH_SIZE)]
    print(f"Total candidate patches: {len(coords)}")

    with Pool(6, initializer=init_worker) as p:
        results = list(tqdm(p.imap(process_patch, coords),
                            total=len(coords), desc="Patching", unit="patch"))

    saved   = sum(results)
    skipped = len(results) - saved
    print(f"\nSaved: {saved} | Skipped (background): {skipped}")
    print(f"Kept {saved/(saved+skipped)*100:.1f}% of all patches")