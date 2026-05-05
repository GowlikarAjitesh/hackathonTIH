import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import random

IMG_DIR = "patched_tif_masks/37774_bagga ortho_3857(7)_patches/images"
MASK_DIR = "patched_tif_masks/37774_bagga ortho_3857(7)_patches/masks"

OUT_DIR = "patch_mask_sideby/37774_bagga ortho_3857(7"
os.makedirs(OUT_DIR, exist_ok=True)

img_files = sorted(os.listdir(IMG_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

# -----------------------------
# Filter useful patches
# -----------------------------
def is_valid_patch(img, mask):
    if np.sum(mask > 0) < 50:
        return False
    if np.mean(img) < 5:
        return False
    return True

valid_indices = []

for i in range(len(img_files)):
    with rasterio.open(os.path.join(IMG_DIR, img_files[i])) as src:
        img = src.read()
    with rasterio.open(os.path.join(MASK_DIR, mask_files[i])) as src:
        mask = src.read(1)

    if is_valid_patch(img, mask):
        valid_indices.append(i)

print("Valid:", len(valid_indices))

# -----------------------------
# Select patches
# -----------------------------
selected = valid_indices[:50] + random.sample(valid_indices, min(30, len(valid_indices)))

# -----------------------------
# Create comparison grid
# -----------------------------
def save_compare_grid(indices, grid_id):
    
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 4 * len(indices)))

    for row, idx in enumerate(indices):

        img_path = os.path.join(IMG_DIR, img_files[idx])
        mask_path = os.path.join(MASK_DIR, mask_files[idx])

        with rasterio.open(img_path) as src:
            img = src.read()

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        img = img.transpose(1, 2, 0).astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        # LEFT → IMAGE
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Image {idx}")
        axes[row, 0].axis("off")

        # RIGHT → MASK
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title(f"Mask {idx}")
        axes[row, 1].axis("off")

    save_path = os.path.join(OUT_DIR, f"compare_{grid_id}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


# -----------------------------
# Generate grids (2 samples per image)
# -----------------------------
chunk_size = 2  # 2 rows → each row = (img | mask)

for i in range(0, len(selected), chunk_size):
    chunk = selected[i:i+chunk_size]

    if len(chunk) < chunk_size:
        break

    save_compare_grid(chunk, i // chunk_size)

print("\n✅ Done: Proper side-by-side comparisons ready.")