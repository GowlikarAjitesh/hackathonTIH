import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import random

# --- PATHS ---
# Add all your patch dataset folders here
PATCH_DATASETS = [
    "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho(10)",
    "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/TIMMOWAL_37695_ORI(9)",
    "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/28996_NADALA_ORTHO(8)",
    "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/37458_fattu_bhila_ortho_3857(6)",
    "/home/vishnu/Ajitesh/ajitesh/patched_tif_masks/37774_bagga_ortho_3857(7)",
    # add CG ones too if patched
]

OUT_DIR = "/home/vishnu/Ajitesh/ajitesh/tiff_visualization/patch_overview"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_COLORS = {
    0: [0,   0,   0  ],   # background — black
    1: [255, 140, 0  ],   # buildings  — orange
    2: [255, 0,   0  ],   # roads      — red
    4: [0,   0,   255],   # water      — blue
}

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        rgb[mask == cid] = color
    return rgb

# --- STEP 1: Collect stats from ALL patches ---
def check_patch(args):
    img_path, mask_path = args
    try:
        mask  = tiff.imread(mask_path)
        ratio = np.sum(mask > 0) / mask.size

        classes = {}
        for cid in [1, 2, 4]:
            classes[cid] = np.sum(mask == cid) / mask.size

        return {
            "img":       img_path,
            "mask":      mask_path,
            "ratio":     ratio,
            "has_anno":  ratio >= 0.02,
            "buildings": classes[1],
            "roads":     classes[2],
            "water":     classes[4],
        }
    except:
        return None

def collect_all_patches():
    tasks = []
    for dataset in PATCH_DATASETS:
        img_dir  = os.path.join(dataset, "images")
        mask_dir = os.path.join(dataset, "masks")
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(mask_dir):
            if f.endswith(".tif"):
                img_file = f.replace("mask_", "img_")
                tasks.append((
                    os.path.join(img_dir,  img_file),
                    os.path.join(mask_dir, f),
                ))
    return tasks

# --- STEP 2: Dataset overview plot ---
def plot_dataset_overview(stats):
    total    = len(stats)
    has_anno = [s for s in stats if s["has_anno"]]
    empty    = [s for s in stats if not s["has_anno"]]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Full Patch Dataset Overview\nTotal: {total} patches across all TIF files",
                 fontsize=15, fontweight="bold")

    # --- Plot 1: Kept vs Dropped pie ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.pie(
        [len(has_anno), len(empty)],
        labels=[f"Kept\n{len(has_anno)} ({len(has_anno)/total*100:.1f}%)",
                f"Dropped\n{len(empty)} ({len(empty)/total*100:.1f}%)"],
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        textprops={"fontsize": 11}
    )
    ax1.set_title("Kept vs Dropped Patches", fontweight="bold")

    # --- Plot 2: Annotation ratio distribution ---
    ax2 = fig.add_subplot(2, 3, 2)
    ratios = [s["ratio"] for s in stats]
    ax2.hist(ratios, bins=50, color="#3498db", edgecolor="white", linewidth=0.5)
    ax2.axvline(0.02, color="red", linestyle="--", linewidth=2, label="2% threshold")
    ax2.set_xlabel("Annotation ratio (non-background pixels)")
    ax2.set_ylabel("Number of patches")
    ax2.set_title("Annotation Density Distribution", fontweight="bold")
    ax2.legend()

    # --- Plot 3: Class breakdown in KEPT patches ---
    ax3 = fig.add_subplot(2, 3, 3)
    if has_anno:
        b = np.mean([s["buildings"] for s in has_anno]) * 100
        r = np.mean([s["roads"]     for s in has_anno]) * 100
        w = np.mean([s["water"]     for s in has_anno]) * 100
        bg = 100 - b - r - w

        bars = ax3.bar(
            ["Background", "Buildings", "Roads", "Water"],
            [bg, b, r, w],
            color=["#7f8c8d", "orange", "red", "blue"]
        )
        for bar, val in zip(bars, [bg, b, r, w]):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{val:.2f}%", ha="center", fontsize=10)
        ax3.set_ylabel("Average % of pixels")
        ax3.set_title("Average Class Distribution\n(in kept patches)", fontweight="bold")

    # --- Plot 4: Per-dataset breakdown ---
    ax4 = fig.add_subplot(2, 3, 4)
    dataset_names = []
    dataset_kept  = []
    dataset_drop  = []
    for dataset in PATCH_DATASETS:
        name     = Path(dataset).name[:20]  # truncate long names
        mask_dir = os.path.join(dataset, "masks")
        if not os.path.exists(mask_dir):
            continue
        ds_stats = [s for s in stats if dataset in s["mask"]]
        kept = sum(1 for s in ds_stats if s["has_anno"])
        drop = sum(1 for s in ds_stats if not s["has_anno"])
        dataset_names.append(name)
        dataset_kept.append(kept)
        dataset_drop.append(drop)

    x = np.arange(len(dataset_names))
    w = 0.35
    ax4.bar(x - w/2, dataset_kept, w, label="Kept",    color="#2ecc71")
    ax4.bar(x + w/2, dataset_drop, w, label="Dropped", color="#e74c3c")
    ax4.set_xticks(x)
    ax4.set_xticklabels(dataset_names, rotation=20, ha="right", fontsize=8)
    ax4.set_ylabel("Number of patches")
    ax4.set_title("Kept vs Dropped per TIF file", fontweight="bold")
    ax4.legend()

    # --- Plot 5: Summary text ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis("off")
    summary = (
        f"DATASET SUMMARY\n"
        f"{'─'*30}\n"
        f"Total patches     : {total}\n"
        f"Kept (≥2% anno)   : {len(has_anno)}\n"
        f"Dropped (<2% anno): {len(empty)}\n\n"
        f"KEPT PATCH STATS\n"
        f"{'─'*30}\n"
    )
    if has_anno:
        b  = np.mean([s["buildings"] for s in has_anno]) * 100
        r  = np.mean([s["roads"]     for s in has_anno]) * 100
        w  = np.mean([s["water"]     for s in has_anno]) * 100
        bg = 100 - b - r - w
        summary += (
            f"Avg background  : {bg:.2f}%\n"
            f"Avg buildings   : {b:.2f}%\n"
            f"Avg roads       : {r:.2f}%\n"
            f"Avg water       : {w:.2f}%\n\n"
            f"Train (70%)     : {int(len(has_anno)*0.70)}\n"
            f"Val   (15%)     : {int(len(has_anno)*0.15)}\n"
            f"Test  (15%)     : {int(len(has_anno)*0.15)}\n"
        )
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "00_dataset_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: 00_dataset_overview.png")


# --- STEP 3: Sample patch grid (image | mask | overlay) ---
def plot_patch_grid(stats, num_samples=30):
    has_anno = [s for s in stats if s["has_anno"]]
    samples  = random.sample(has_anno, min(num_samples, len(has_anno)))

    cols = 6   # 2 patches per column (image + overlay)
    rows = len(samples)

    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 3))
    fig.suptitle(f"Sample Kept Patches — Image | Mask | Overlay\n"
                 f"(randomly sampled from {len(has_anno)} kept patches)",
                 fontsize=13, fontweight="bold")

    if rows == 1:
        axes = [axes]

    for i, s in enumerate(samples):
        try:
            img  = tiff.imread(s["img"])
            mask = tiff.imread(s["mask"])

            # Fix channels
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[0] in [3, 4] and img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
            img = img[:, :, :3].astype(np.float32)
            p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
            img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)

            mask_rgb = mask_to_rgb(mask)

            # Overlay
            overlay = (img * 255).astype(np.uint8).copy()
            for cid, color in CLASS_COLORS.items():
                if cid == 0:
                    continue
                overlay[mask == cid] = (
                    np.array(color) * 0.6 +
                    overlay[mask == cid] * 0.4
                ).astype(np.uint8)

            axes[i][0].imshow(img)
            axes[i][0].axis("off")
            axes[i][0].set_ylabel(Path(s["mask"]).stem[:15],
                                  fontsize=7, rotation=0, labelpad=60)

            axes[i][1].imshow(mask_rgb)
            axes[i][1].axis("off")

            axes[i][2].imshow(overlay)
            axes[i][2].axis("off")

        except Exception as e:
            for j in range(3):
                axes[i][j].axis("off")

    # Column headers
    axes[0][0].set_title("Image",   fontweight="bold", fontsize=11)
    axes[0][1].set_title("Mask",    fontweight="bold", fontsize=11)
    axes[0][2].set_title("Overlay", fontweight="bold", fontsize=11)

    # Legend
    legend = [
        mpatches.Patch(color="orange", label="Buildings (1)"),
        mpatches.Patch(color="red",    label="Roads (2)"),
        mpatches.Patch(color="blue",   label="Water (4)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 0))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(OUT_DIR, "01_sample_patches.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: 01_sample_patches.png")


# --- MAIN ---
if __name__ == "__main__":
    print("Collecting all patches...")
    tasks = collect_all_patches()
    print(f"Found {len(tasks)} total patches across all datasets\n")

    print("Analyzing patches with multiprocessing...")
    workers = min(cpu_count(), 16)
    with Pool(workers) as p:
        results = list(tqdm(p.imap(check_patch, tasks),
                            total=len(tasks), desc="Scanning"))

    stats = [r for r in results if r is not None]
    print(f"\nAnalyzed {len(stats)} patches")

    print("\nGenerating dataset overview plot...")
    plot_dataset_overview(stats)

    print("Generating sample patch grid...")
    plot_patch_grid(stats, num_samples=30)

    print(f"\nAll visualizations saved to:\n{OUT_DIR}")
    print("Open the PNG files to see your full dataset!")