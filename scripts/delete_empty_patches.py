import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

MASK_DIR = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho(10)/masks"
IMG_DIR  = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho(10)/images"
MIN_ANNO_RATIO = 0.02

def filter_patch(f):
    if not f.endswith(".tif"):
        return 0
    try:
        mask_path = os.path.join(MASK_DIR, f)
        img_path  = os.path.join(IMG_DIR, f.replace("mask_", "img_"))

        mask  = tiff.imread(mask_path)
        ratio = np.sum(mask > 0) / mask.size

        if ratio < MIN_ANNO_RATIO:
            os.remove(mask_path)
            if os.path.exists(img_path):
                os.remove(img_path)
            return 0  # deleted
        return 1      # kept
    except:
        return 0

if __name__ == "__main__":
    files   = [f for f in os.listdir(MASK_DIR) if f.endswith(".tif")]
    workers = min(cpu_count(), 16)

    print(f"Filtering {len(files)} patches with {workers} workers...")

    with Pool(workers) as p:
        results = list(tqdm(p.imap(filter_patch, files),
                            total=len(files), desc="Filtering"))

    kept    = sum(results)
    deleted = len(results) - kept
    print(f"\nKept   : {kept}")
    print(f"Deleted: {deleted}")
    print(f"Your training set: {kept} patches")