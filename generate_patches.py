import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
from multiprocessing import Pool
import warnings
from tqdm import tqdm  # <--- Added for progress and ETA

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ... [Keep your IMAGE_PATH, MASK_PATH, and Directory setup as is] ...
IMAGE_PATH = "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/TIMMOWAL_37695_ORI(9).tif"
MASK_PATH  = "/home/vishnu/Ajitesh/ajitesh/generated_masks/TIMMOWAL_37695_ORI(9)_mask.tif"
PATCH_SIZE = 512
base_name = Path(IMAGE_PATH).stem
OUT_DIR = f"/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/{base_name}"
IMG_DIR = os.path.join(OUT_DIR, "images")
MASK_DIR = os.path.join(OUT_DIR, "masks")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

with rasterio.open(MASK_PATH) as src:
    height, width = src.height, src.width

# --- GLOBAL HANDLES ---
img_src = None
mask_src = None

def init_worker():
    global img_src, mask_src
    img_src = rasterio.open(IMAGE_PATH)
    mask_src = rasterio.open(MASK_PATH)

# --- PATCH FUNCTION ---
def process_patch(coords):
    y, x = coords
    window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
    
    # Pre-check if indices are valid to avoid crashing workers
    try:
        img_patch = img_src.read(window=window)
        mask_patch = mask_src.read(1, window=window)

        if img_patch.shape[1] != PATCH_SIZE or img_patch.shape[2] != PATCH_SIZE:
            return 0

        img_file = os.path.join(IMG_DIR, f"img_{y}_{x}.tif")
        mask_file = os.path.join(MASK_DIR, f"mask_{y}_{x}.tif")

        with rasterio.open(img_file, "w", driver="GTiff", height=PATCH_SIZE, width=PATCH_SIZE,
                           count=img_patch.shape[0], dtype=img_patch.dtype) as dst:
            dst.write(img_patch)

        with rasterio.open(mask_file, "w", driver="GTiff", height=PATCH_SIZE, width=PATCH_SIZE,
                           count=1, dtype="uint8") as dst:
            dst.write(mask_patch, 1)
        return 1
    except Exception:
        return 0

# --- MAIN ---
if __name__ == "__main__":
    coords = [(y, x) for y in range(0, height, PATCH_SIZE)
                      for x in range(0, width, PATCH_SIZE)]

    total_patches = len(coords)
    print(f"🚀 Total patches to process: {total_patches}")

    # Using tqdm to track progress through the multiprocessing pool
    with Pool(6, initializer=init_worker) as p:
        # p.imap allows tqdm to update as each task finishes
        results = list(tqdm(p.imap(process_patch, coords), total=total_patches, desc="Patching", unit="patch"))

    print(f"\n✅ Done! Total patches successfully saved: {sum(results)}")