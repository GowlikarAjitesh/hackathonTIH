import os
import rasterio
import numpy as np

# -----------------------------------
# PATHS
# -----------------------------------

MASK_PATH = "generated_masks/28996_NADALA_ORTHO_mask.tif"
PATCH_DIR = "patch_dataset/masks"

PATCH_SIZE = 512

# -----------------------------------
# LOAD ORIGINAL MASK
# -----------------------------------

with rasterio.open(MASK_PATH) as src:
    original_mask = src.read(1)

errors = 0
checked = 0

# -----------------------------------
# VERIFY EACH PATCH
# -----------------------------------

for file in os.listdir(PATCH_DIR):

    if not file.endswith(".tif"):
        continue

    parts = file.replace(".tif","").split("_")

    y = int(parts[1])
    x = int(parts[2])

    patch_path = os.path.join(PATCH_DIR,file)

    with rasterio.open(patch_path) as src:
        patch = src.read(1)

    original_region = original_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    if not np.array_equal(patch, original_region):
        print("Mismatch:", file)
        errors += 1

    checked += 1

print("\nTotal patches checked:", checked)
print("Errors:", errors)

if errors == 0:
    print("\nPatch extraction is LOSSLESS ✅")
else:
    print("\nPatch mismatch detected ❌")



import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# PATHS
# -----------------------------------

MASK_PATH = "generated_masks/28996_NADALA_ORTHO_mask.tif"
PATCH_DIR = "patch_dataset/masks"

PATCH_SIZE = 512

OUT_DIR = "reconstruction_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------
# LOAD ORIGINAL MASK
# -----------------------------------

with rasterio.open(MASK_PATH) as src:
    original_mask = src.read(1)

height, width = original_mask.shape

# -----------------------------------
# RECONSTRUCT MASK
# -----------------------------------

reconstructed = np.zeros_like(original_mask)

for file in os.listdir(PATCH_DIR):

    if not file.endswith(".tif"):
        continue

    parts = file.replace(".tif","").split("_")

    y = int(parts[1])
    x = int(parts[2])

    path = os.path.join(PATCH_DIR,file)

    with rasterio.open(path) as src:
        patch = src.read(1)

    reconstructed[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch

# -----------------------------------
# DIFFERENCE MAP
# -----------------------------------

difference = original_mask - reconstructed

# -----------------------------------
# SAVE VISUALIZATIONS
# -----------------------------------

plt.figure(figsize=(10,10))
plt.imshow(original_mask, cmap="tab20")
plt.title("Original Mask")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR,"original_mask.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(reconstructed, cmap="tab20")
plt.title("Reconstructed Mask")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR,"reconstructed_mask.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(difference, cmap="RdBu")
plt.title("Difference Map")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR,"difference.png"), dpi=300)
plt.close()

print("\nVisualization saved in:", OUT_DIR)


import os
import rasterio
import numpy as np

# --------------------------------
# PATHS
# --------------------------------

IMAGE_PATH = "/home/cs24m112/hackathon/dataset/feature_extraction/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif"

PATCH_DIR = "patch_dataset/images"

PATCH_SIZE = 512

# --------------------------------
# LOAD ORIGINAL IMAGE
# --------------------------------

with rasterio.open(IMAGE_PATH) as src:
    original_image = src.read()

errors = 0
checked = 0

# --------------------------------
# VERIFY EACH PATCH
# --------------------------------

for file in os.listdir(PATCH_DIR):

    if not file.endswith(".tif"):
        continue

    # filename format → img_y_x.tif
    parts = file.replace(".tif","").split("_")

    y = int(parts[1])
    x = int(parts[2])

    patch_path = os.path.join(PATCH_DIR,file)

    with rasterio.open(patch_path) as src:
        patch = src.read()

    original_region = original_image[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    if not np.array_equal(patch, original_region):
        print("Mismatch:", file)
        errors += 1

    checked += 1

# --------------------------------
# RESULTS
# --------------------------------

print("\nTotal patches checked:", checked)
print("Errors:", errors)

if errors == 0:
    print("\nImage patches are LOSSLESS ✅")
else:
    print("\nPatch mismatch detected ❌")


import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# PATHS
# -----------------------------------

IMAGE_PATH = "/home/cs24m112/hackathon/dataset/feature_extraction/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif"
PATCH_DIR = "patch_dataset/images"

PATCH_SIZE = 512

OUT_DIR = "image_reconstruction_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

DOWNSAMPLE = 20  # reduce size for visualization

# -----------------------------------
# LOAD ORIGINAL IMAGE
# -----------------------------------

with rasterio.open(IMAGE_PATH) as src:
    original = src.read([1,2,3])   # RGB only

original = original.transpose(1,2,0)

height, width, _ = original.shape

# -----------------------------------
# RECONSTRUCT IMAGE
# -----------------------------------

reconstructed = np.zeros_like(original)

for file in os.listdir(PATCH_DIR):

    if not file.endswith(".tif"):
        continue

    parts = file.replace(".tif","").split("_")

    y = int(parts[1])
    x = int(parts[2])

    path = os.path.join(PATCH_DIR,file)

    with rasterio.open(path) as src:
        patch = src.read([1,2,3])

    patch = patch.transpose(1,2,0)

    reconstructed[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch

# -----------------------------------
# DIFFERENCE MAP
# -----------------------------------

difference = original.astype(int) - reconstructed.astype(int)

# -----------------------------------
# DOWNSAMPLE FOR VISUALIZATION
# -----------------------------------

orig_vis = original[::DOWNSAMPLE, ::DOWNSAMPLE]
recon_vis = reconstructed[::DOWNSAMPLE, ::DOWNSAMPLE]
diff_vis = difference[::DOWNSAMPLE, ::DOWNSAMPLE]

# -----------------------------------
# SAVE VISUALIZATIONS
# -----------------------------------

plt.figure(figsize=(10,10))
plt.imshow(orig_vis)
plt.title("Original Orthophoto")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR,"original_image.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(recon_vis)
plt.title("Reconstructed Image")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR,"reconstructed_image.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(np.abs(diff_vis))
plt.title("Difference Image")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR,"difference_image.png"), dpi=300)
plt.close()

print("\nVisualization saved in:", OUT_DIR)