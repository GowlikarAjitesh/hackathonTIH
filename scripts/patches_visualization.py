import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import re

# --- PATHS ---
PATCH_BASE_DIR = "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks"
OUT_DIR        = "/home/vishnu/Ajitesh/ajitesh/patch_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_COLORS = {
    1: [255, 140, 0],   # buildings — orange
    2: [255, 0,   0],   # roads     — red
    4: [0,   0, 255],   # water     — blue
}

# --- STEP 1: Get all patch folders ---
def get_patch_folders():
    folders = []
    for f in os.listdir(PATCH_BASE_DIR):
        full     = os.path.join(PATCH_BASE_DIR, f)
        img_dir  = os.path.join(full, "images")
        mask_dir = os.path.join(full, "masks")
        if os.path.isdir(full) and os.path.exists(img_dir) and os.path.exists(mask_dir):
            folders.append((f, full))
    return folders

# --- STEP 2: Reconstruct canvas from patches ---
def reconstruct_from_patches(patch_folder):
    img_dir  = os.path.join(patch_folder, "images")
    mask_dir = os.path.join(patch_folder, "masks")

    patch_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
    if not patch_files:
        return None, None

    # Parse y, x from filenames
    coords = []
    for f in patch_files:
        match = re.search(r"img_(\d+)_(\d+)\.tif", f)
        if match:
            y, x = int(match.group(1)), int(match.group(2))
            coords.append((y, x, f))

    if not coords:
        return None, None

    # Get patch size from first patch
    sample = tiff.imread(os.path.join(img_dir, coords[0][2]))
    if sample.ndim == 3 and sample.shape[0] in [3, 4]:
        patch_h, patch_w = sample.shape[1], sample.shape[2]
    else:
        patch_h, patch_w = sample.shape[0], sample.shape[1]

    # Canvas size
    max_y = max(c[0] for c in coords) + patch_h
    max_x = max(c[1] for c in coords) + patch_w

    canvas_img  = np.zeros((max_y, max_x, 3), dtype=np.uint8)
    canvas_mask = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    print(f"  Canvas: {max_y} x {max_x} | Patches: {len(coords)}")

    for y, x, fname in tqdm(coords, desc="  Merging", leave=False):
        img_path  = os.path.join(img_dir,  fname)
        mask_name = fname.replace("img_", "mask_")
        mask_path = os.path.join(mask_dir, mask_name)

        try:
            # --- Image ---
            img = tiff.imread(img_path)
            if img.ndim == 3 and img.shape[0] in [3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            img = img[:, :, :3].astype(np.float32)
            p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
            img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
            img = (img * 255).astype(np.uint8)

            h, w = img.shape[:2]
            canvas_img[y:y+h, x:x+w] = img

            # --- Mask ---
            if os.path.exists(mask_path):
                mask     = tiff.imread(mask_path)
                mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for cid, color in CLASS_COLORS.items():
                    mask_rgb[mask == cid] = color
                canvas_mask[y:y+h, x:x+w] = mask_rgb

        except:
            continue

    return canvas_img, canvas_mask

# --- STEP 3: Build overlay ---
def build_overlay(canvas_img, canvas_mask):
    overlay = canvas_img.copy().astype(np.float32)
    for cid, color in CLASS_COLORS.items():
        region = np.all(canvas_mask == color, axis=-1)
        overlay[region] = (
            np.array(color) * 0.55 +
            overlay[region] * 0.45
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)

# --- STEP 4: Save as high quality PNG ---
def save_patch_visualization(folder_name, canvas_img, canvas_mask):
    overlay = build_overlay(canvas_img, canvas_mask)

    # Resize for display — keep aspect ratio, max height 1200px
    max_h   = 1200
    scale   = min(1.0, max_h / canvas_img.shape[0])
    disp_h  = int(canvas_img.shape[0] * scale)
    disp_w  = int(canvas_img.shape[1] * scale)

    img_disp     = np.array(Image.fromarray(canvas_img).resize( (disp_w, disp_h), Image.LANCZOS))
    mask_disp    = np.array(Image.fromarray(canvas_mask).resize((disp_w, disp_h), Image.NEAREST))
    overlay_disp = np.array(Image.fromarray(overlay).resize(    (disp_w, disp_h), Image.LANCZOS))

    # --- 3 panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"Patch Visualization — {folder_name}",
        fontsize=15, fontweight="bold", color="white", y=1.01
    )

    titles = [
        "Patched Image\n(kept patches only, black = dropped)",
        "Mask\n(orange=Buildings  red=Roads  blue=Water)",
        "Image + Mask Overlay\n(how well annotations align with image)",
    ]
    images = [img_disp, mask_disp, overlay_disp]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=11, color="white", pad=8)
        ax.axis("off")

    # Legend
    legend = [
        mpatches.Patch(color=[1, 0.55, 0], label="Buildings (class 1)"),
        mpatches.Patch(color=[1, 0,    0], label="Roads (class 2)"),
        mpatches.Patch(color=[0, 0,    1], label="Water (class 4)"),
        mpatches.Patch(color=[0, 0,    0], label="Dropped patches"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4,
        fontsize=11, bbox_to_anchor=(0.5, -0.04),
        facecolor="#2c2c54", labelcolor="white", edgecolor="white"
    )

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(OUT_DIR, f"{folder_name}_visualization.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Figure saved : {folder_name}_visualization.png")

    # Also save raw overlay as standalone PNG (no axes/titles — pure image)
    raw_path = os.path.join(OUT_DIR, f"{folder_name}_overlay_raw.png")
    Image.fromarray(overlay_disp).save(raw_path)
    print(f"  ✅ Raw overlay  : {folder_name}_overlay_raw.png")


# --- MAIN ---
if __name__ == "__main__":
    patch_folders = get_patch_folders()
    print(f"Found {len(patch_folders)} patch folders\n")

    for folder_name, folder_path in patch_folders:
        print(f"\nProcessing: {folder_name}")

        canvas_img, canvas_mask = reconstruct_from_patches(folder_path)
        if canvas_img is None:
            print(f"  ⚠️  No valid patches, skipping")
            continue

        save_patch_visualization(folder_name, canvas_img, canvas_mask)

    print(f"\n{'='*50}")
    print(f"All saved to: {OUT_DIR}")