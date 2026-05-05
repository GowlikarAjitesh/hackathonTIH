from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np
import rasterio
from folium import LayerControl
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from rasterio.vrt import WarpedVRT


DEFAULT_TIF_PATH = Path(
    "/home/vishnu/Ajitesh/ajitesh/cleaned_perf_tifs/"
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho(10).tif"
)
DEFAULT_OUTPUT_HTML = Path("/home/vishnu/Ajitesh/ajitesh/tif_visualization.html")
WEB_CRS = "EPSG:4326"
MAX_DISPLAY_SIZE = 1024


def _compute_display_shape(width: int, height: int, max_size: int = MAX_DISPLAY_SIZE) -> tuple[int, int]:
    longest_edge = max(width, height)
    if longest_edge <= max_size:
        return width, height

    scale = max_size / float(longest_edge)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32, copy=False)
    valid_mask = np.isfinite(band)
    if not np.any(valid_mask):
        return np.zeros_like(band, dtype=np.uint8)

    valid_values = band[valid_mask]
    low, high = np.percentile(valid_values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(valid_values.min())
        high = float(valid_values.max())

    if high <= low:
        return np.zeros_like(band, dtype=np.uint8)

    scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
    scaled[~valid_mask] = 0.0
    return (scaled * 255).astype(np.uint8)


def _prepare_overlay_image(data: np.ndarray) -> np.ndarray:
    if data.shape[0] >= 3:
        rgb = np.stack([_normalize_band(data[idx]) for idx in range(3)], axis=0)
        return reshape_as_image(rgb)

    if data.shape[0] == 1:
        gray = _normalize_band(data[0])
        return np.dstack([gray, gray, gray])

    raise ValueError("GeoTIFF has no readable bands.")


def build_map(tif_path: Path, zoom_start: int = 16) -> folium.Map:
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path} has no CRS, so it cannot be placed on a web map.")

        display_width, display_height = _compute_display_shape(src.width, src.height)
        read_shape = (src.count, display_height, display_width)

        with WarpedVRT(src, crs=WEB_CRS, resampling=Resampling.bilinear) as vrt:
            data = vrt.read(out_shape=read_shape)
            bounds = vrt.bounds

    image = _prepare_overlay_image(data)

    min_lon, min_lat, max_lon, max_lat = bounds
    center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True)

    folium.raster_layers.ImageOverlay(
        image=image,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.7,
        name=tif_path.name,
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color="#ff3b30",
        weight=2,
        fill=False,
        tooltip=(
            f"{tif_path.name}\n"
            f"Lat: {min_lat:.6f} to {max_lat:.6f}\n"
            f"Lon: {min_lon:.6f} to {max_lon:.6f}"
        ),
    ).add_to(m)

    folium.Marker(
        location=center,
        tooltip=tif_path.name,
        popup=(
            f"<b>{tif_path.name}</b><br>"
            f"Center: {center[0]:.6f}, {center[1]:.6f}<br>"
            f"Bounds: [{min_lat:.6f}, {min_lon:.6f}] to [{max_lat:.6f}, {max_lon:.6f}]"
        ),
    ).add_to(m)

    LayerControl().add_to(m)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a GeoTIFF on an interactive map and save it as HTML."
    )
    parser.add_argument(
        "--tif",
        type=Path,
        default=DEFAULT_TIF_PATH,
        help=f"Path to the GeoTIFF to visualize. Default: {DEFAULT_TIF_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_HTML,
        help=f"Where to save the generated HTML map. Default: {DEFAULT_OUTPUT_HTML}",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=16,
        help="Initial zoom level before the map is fit to the TIFF bounds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tif_path = args.tif.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not tif_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object = build_map(tif_path=tif_path, zoom_start=args.zoom_start)
    map_object.save(str(output_path))

    print(f"Saved map to: {output_path}")
    print(f"Visualized TIFF: {tif_path}")


if __name__ == "__main__":
    main()
