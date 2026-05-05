import os
import random
import shutil
from multiprocessing import Pool

IMAGES_DIR = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho(10)/images"
MASKS_DIR  = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho(10)/masks"

BASE_OUTPUT_PATH = "/home/vishnu/Ajitesh/ajitesh/dataset_after_train_test_split"
FOLDER_NAME = os.path.basename(os.path.dirname(IMAGES_DIR))
OUTPUT_DIR = os.path.join(BASE_OUTPUT_PATH, FOLDER_NAME)

SPLIT = (0.7, 0.15, 0.15)


def process_file(args):
    f, split_name = args

    img_path = os.path.join(IMAGES_DIR, f)
    mask_name = f.replace("img_", "mask_")
    mask_path = os.path.join(MASKS_DIR, mask_name)

    if not os.path.exists(mask_path):
        return 0, 1

    img_dst = os.path.join(OUTPUT_DIR, split_name, "images", f)
    mask_dst = os.path.join(OUTPUT_DIR, split_name, "masks", mask_name)

    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    os.makedirs(os.path.dirname(mask_dst), exist_ok=True)

    shutil.copy2(img_path, img_dst)
    shutil.copy2(mask_path, mask_dst)

    return 1, 0


def split_dataset():
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".tif")]
    random.shuffle(images)

    n = len(images)
    train_end = int(SPLIT[0]*n)
    val_end = int((SPLIT[0]+SPLIT[1])*n)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    tasks = []
    for split_name, files in splits.items():
        for f in files:
            tasks.append((f, split_name))

    print(f"🚀 Processing {len(tasks)} files...")

    with Pool(32) as p:   # sweet spot (don’t go 112)
        results = p.map(process_file, tasks)

    copied = sum(r[0] for r in results)
    skipped = sum(r[1] for r in results)

    print(f"\n✅ Done!")
    print(f"Copied: {copied}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    split_dataset()