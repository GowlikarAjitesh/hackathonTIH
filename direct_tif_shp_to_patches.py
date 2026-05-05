from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: geopandas. Use the project venv and install the required geospatial packages."
    ) from exc

try:
    from shapely.geometry import box
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: shapely. Use the project venv and install the required geospatial packages."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **_: object):
        return iterable


DEFAULT_TIF = Path(
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/"
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho(10).tif"
)
DEFAULT_SHP_DIR = Path("/home/vishnu/Ajitesh/ajitesh/cg_shp-file")
DEFAULT_OUTPUT_ROOT = Path("/home/vishnu/Ajitesh/ajitesh/patched_tif_masks")

CLASS_CONFIG: dict[str, tuple[list[str], int]] = {
    "buildings": (["Built_Up_Area_type.shp", "Built_Up_Area_typ.shp"], 1),
    "roads": (["Road.shp", "Road_32643.shp"], 2),
    "water": (["Water_Body.shp"], 4),
}


@dataclass
class LayerIndex:
    name: str
    class_id: int
    gdf: gpd.GeoDataFrame
    sindex: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse image/mask patches directly from a GeoTIFF and shapefiles."
    )
    parser.add_argument("--tif", type=Path, default=DEFAULT_TIF, help=f"Input TIFF. Default: {DEFAULT_TIF}")
    parser.add_argument(
        "--shp-dir",
        type=Path,
        default=DEFAULT_SHP_DIR,
        help=f"Directory containing shapefiles. Default: {DEFAULT_SHP_DIR}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for generated patches. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--patch-size", type=int, default=512, help="Patch width and height in pixels.")
    parser.add_argument(
        "--min-positive-ratio",
        type=float,
        default=0.0,
        help="Minimum fraction of non-zero mask pixels required to keep a patch.",
    )
    parser.add_argument(
        "--include-edge-patches",
        action="store_true",
        help="Include incomplete edge patches. Default behavior skips them.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        default=True,
        help="Burn all touched pixels when rasterizing. Enabled by default.",
    )
    parser.add_argument(
        "--no-all-touched",
        action="store_false",
        dest="all_touched",
        help="Disable all_touched during rasterization.",
    )
    return parser.parse_args()


def find_shapefile(shp_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = shp_dir / name
        if path.exists():
            return path
    return None


def load_layer_indexes(shp_dir: Path, tif_crs: object) -> list[LayerIndex]:
    layer_indexes: list[LayerIndex] = []

    for layer_name, (candidate_files, class_id) in CLASS_CONFIG.items():
        shp_path = find_shapefile(shp_dir, candidate_files)
        if shp_path is None:
            print(f"Skipping {layer_name}: no shapefile found in {shp_dir}")
            continue

        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            print(f"Skipping {layer_name}: {shp_path.name} is empty")
            continue

        if gdf.crs is None:
            print(f"Skipping {layer_name}: {shp_path.name} has no CRS")
            continue

        gdf = gdf.to_crs(tif_crs)
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.is_valid]
        gdf = gdf[~gdf.geometry.is_empty].copy()

        if gdf.empty:
            print(f"Skipping {layer_name}: no valid geometries after cleanup")
            continue

        layer_indexes.append(
            LayerIndex(
                name=layer_name,
                class_id=class_id,
                gdf=gdf,
                sindex=gdf.sindex,
            )
        )
        print(f"Loaded {layer_name}: {len(gdf)} geometries from {shp_path.name}")

    if not layer_indexes:
        raise RuntimeError("No usable shapefiles were loaded.")

    return layer_indexes


def prepare_output_dirs(output_root: Path, tif_path: Path) -> tuple[Path, Path, Path]:
    base_name = tif_path.stem
    dataset_dir = output_root / base_name
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, img_dir, mask_dir


def tile_block_size(size: int) -> int:
    bounded = min(256, size)
    if bounded < 16:
        return 16
    return max(16, (bounded // 16) * 16)


def iter_windows(width: int, height: int, patch_size: int, include_edges: bool) -> list[tuple[int, int, Window]]:
    windows: list[tuple[int, int, Window]] = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            w = min(patch_size, width - x)
            h = min(patch_size, height - y)
            if not include_edges and (w != patch_size or h != patch_size):
                continue
            windows.append((y, x, Window(x, y, w, h)))
    return windows


def get_window_geometries(layer: LayerIndex, window_bounds: tuple[float, float, float, float]) -> list[tuple[object, int]]:
    minx, miny, maxx, maxy = window_bounds
    query_geom = box(minx, miny, maxx, maxy)

    candidate_idx = list(layer.sindex.intersection(query_geom.bounds))
    if not candidate_idx:
        return []

    candidates = layer.gdf.iloc[candidate_idx]
    intersects = candidates.intersects(query_geom)
    if not intersects.any():
        return []

    return [(geom, layer.class_id) for geom in candidates.loc[intersects, "geometry"]]


def build_mask_patch(
    layers: list[LayerIndex],
    window: Window,
    transform: object,
    patch_height: int,
    patch_width: int,
    all_touched: bool,
) -> tuple[np.ndarray, dict[int, int]]:
    tile_transform = rasterio.windows.transform(window, transform)
    bounds = rasterio.windows.bounds(window, transform)
    mask_patch = np.zeros((patch_height, patch_width), dtype=np.uint8)
    class_pixel_counts: dict[int, int] = defaultdict(int)

    for layer in layers:
        shapes = get_window_geometries(layer, bounds)
        if not shapes:
            continue

        burned = rasterize(
            shapes=shapes,
            out_shape=(patch_height, patch_width),
            transform=tile_transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )
        positive = burned == layer.class_id
        if not np.any(positive):
            continue

        mask_patch[positive] = layer.class_id
        class_pixel_counts[layer.class_id] += int(np.count_nonzero(positive))

    return mask_patch, class_pixel_counts


def save_image_patch(img_path: Path, image_patch: np.ndarray, profile: dict, window_transform: object) -> None:
    patch_profile = profile.copy()
    patch_profile.update(
        driver="GTiff",
        width=image_patch.shape[2],
        height=image_patch.shape[1],
        count=image_patch.shape[0],
        transform=window_transform,
        compress=profile.get("compress") or "lzw",
        tiled=True,
        blockxsize=tile_block_size(image_patch.shape[2]),
        blockysize=tile_block_size(image_patch.shape[1]),
    )
    patch_profile.pop("photometric", None)

    with rasterio.open(img_path, "w", **patch_profile) as dst:
        dst.write(image_patch)


def save_mask_patch(mask_path: Path, mask_patch: np.ndarray, crs: object, window_transform: object) -> None:
    mask_profile = {
        "driver": "GTiff",
        "width": mask_patch.shape[1],
        "height": mask_patch.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": window_transform,
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": tile_block_size(mask_patch.shape[1]),
        "blockysize": tile_block_size(mask_patch.shape[0]),
    }

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask_patch, 1)


def main() -> None:
    args = parse_args()
    tif_path = args.tif.expanduser().resolve()
    shp_dir = args.shp_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not tif_path.exists():
        raise FileNotFoundError(f"TIFF not found: {tif_path}")
    if not shp_dir.exists():
        raise FileNotFoundError(f"Shapefile directory not found: {shp_dir}")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be a positive integer.")
    if not 0.0 <= args.min_positive_ratio <= 1.0:
        raise ValueError("--min-positive-ratio must be between 0 and 1.")

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Input TIFF has no CRS: {tif_path}")

        dataset_dir, img_dir, mask_dir = prepare_output_dirs(output_root, tif_path)
        layers = load_layer_indexes(shp_dir, src.crs)
        windows = iter_windows(src.width, src.height, args.patch_size, args.include_edge_patches)
        total_windows = len(windows)

        saved = 0
        skipped = 0
        class_pixel_totals: dict[int, int] = defaultdict(int)
        kept_patch_counts: dict[int, int] = defaultdict(int)

        print(f"Saving sparse patches to: {dataset_dir}")
        print(f"Candidate windows: {total_windows}")

        for y, x, window in tqdm(windows, desc="Generating patches", unit="patch"):
            patch_width = int(window.width)
            patch_height = int(window.height)
            mask_patch, class_counts = build_mask_patch(
                layers=layers,
                window=window,
                transform=src.transform,
                patch_height=patch_height,
                patch_width=patch_width,
                all_touched=args.all_touched,
            )

            positive_pixels = int(np.count_nonzero(mask_patch))
            if positive_pixels == 0:
                skipped += 1
                continue

            positive_ratio = positive_pixels / float(mask_patch.size)
            if positive_ratio < args.min_positive_ratio:
                skipped += 1
                continue

            image_patch = src.read(window=window)
            patch_suffix = f"{y}_{x}.tif"
            patch_transform = rasterio.windows.transform(window, src.transform)

            save_image_patch(img_dir / f"img_{patch_suffix}", image_patch, src.profile, patch_transform)
            save_mask_patch(mask_dir / f"mask_{patch_suffix}", mask_patch, src.crs, patch_transform)

            saved += 1
            for class_id, count in class_counts.items():
                class_pixel_totals[class_id] += count
                if count > 0:
                    kept_patch_counts[class_id] += 1

    print("\nDone.")
    print(f"Saved patches: {saved}")
    print(f"Skipped windows: {skipped}")
    print(f"Keep rate: {saved / total_windows * 100:.2f}%" if total_windows else "Keep rate: 0.00%")
    print("Per-class kept patch counts:")
    for _, (_, class_id) in CLASS_CONFIG.items():
        print(f"  class {class_id}: {kept_patch_counts[class_id]}")
    print("Per-class positive pixel totals:")
    for _, (_, class_id) in CLASS_CONFIG.items():
        print(f"  class {class_id}: {class_pixel_totals[class_id]}")


if __name__ == "__main__":
    main()
