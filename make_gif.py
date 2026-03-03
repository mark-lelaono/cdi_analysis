"""
CDI Raster → H3 Hexagon Animated GIF
-------------------------------------
Simulates the Honeycomb zoom-in effect:
  - Starts with coarse hexagons (wide overview)
  - Progressively zooms into finer hexagons (local detail)

Uses the CDI Hex Engine (cdi_hexmap.py) for raster extraction
and H3 aggregation. Renders frames with matplotlib, stitches
with Pillow.

Usage:
    python make_gif.py -i cdi_monthly/2020/eadw-cdi-data-2020-apr.tif
    python make_gif.py --persistence
"""

import logging
import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon, box
from PIL import Image

from cdi_hexmap import (
    RasterProcessor,
    H3Aggregator,
    CDI_CATEGORY_MAP,
    CDI_CATEGORY_COLORS,
    DEFAULT_SHAPEFILE,
    ICPAC_CENTER,
)
from cdi_config import PATHS, CDI, MONTH_ABBR_TO_NAME, MONTH_ORDER
import h3

log = logging.getLogger(__name__)


# ── CONFIG ────────────────────────────────────────────────────────────────────

FRAMES_DIR = str(PATHS.OUTPUT_DIR / "gif_frames")
OUTPUT_DIR = str(PATHS.OUTPUT_DIR / "hexmaps")

# Admin1 sub-national boundaries (regions/districts)
ADMIN1_SHAPEFILE = str(PATHS.ADMIN1_SHP)

# Background color for frames
BG_COLOR = "#FFFFFF"
BG_COLOR_RGB = (255, 255, 255)

# H3 resolutions to animate through (coarse → fine)
H3_RESOLUTIONS = [3, 4, 5, 6, 7]

# Frames per resolution level
FRAMES_PER_RES = 3

# GIF frame duration in milliseconds
FRAME_DURATION_MS = 500

# CDI discrete colormap: category → hex color
CDI_COLORS_HEX = {
    "No Drought": "#F5F5F5",
    "Watch":      "#FFFF04",
    "Warning":    "#FFA604",
    "Alert":      "#FF0304",
    "No Data":    "#C1C1C1",
}

# Severity weights for scoring (from shared config)
SEVERITY_WEIGHTS = CDI.SEVERITY_WEIGHTS

# Month ordering for chronological sorting (from shared config)
MONTH_NAMES = MONTH_ABBR_TO_NAME


# ── RASTER DISCOVERY ─────────────────────────────────────────────────────────

def discover_rasters(base_dir="cdi_monthly"):
    """Find all CDI .tif files and sort chronologically."""
    pattern = os.path.join(base_dir, "**", "*.tif")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {base_dir}/")

    def sort_key(p):
        m = re.search(r"(\d{4})-(\w{3})", os.path.basename(p))
        if m:
            return (int(m.group(1)), MONTH_ORDER.get(m.group(2), 0))
        return (0, 0)

    paths.sort(key=sort_key)
    return paths


def parse_month_label(filename):
    """Extract 'April 2020' from 'eadw-cdi-data-2020-apr.tif'."""
    m = re.search(r"(\d{4})-(\w{3})", os.path.basename(filename))
    if m:
        year = m.group(1)
        month_abbr = m.group(2).lower()
        month_name = MONTH_NAMES.get(month_abbr, month_abbr.capitalize())
        return f"{month_name} {year}"
    return os.path.splitext(os.path.basename(filename))[0]


# ── COUNTRY SCORING (vectorized) ─────────────────────────────────────────────

def score_countries_single(gdf, boundaries_gdf, country_col="COUNTRY"):
    """
    Score countries by drought severity for a single month's GeoDataFrame.

    Uses vectorized groupby + crosstab instead of iterrows().
    Score is normalized: weighted_severe_hexagons / total_hexagons_in_country,
    so large countries don't dominate purely by area.

    Returns sorted DataFrame with columns:
        COUNTRY, alert, warning, watch, total, severity_density, score
    """
    joined = gpd.sjoin(
        gdf, boundaries_gdf[[country_col, "geometry"]],
        how="inner", predicate="intersects",
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[country_col, "alert", "warning", "watch",
                     "total", "severity_density", "score"],
        )

    # Vectorized crosstab: country × category counts
    ct = pd.crosstab(joined[country_col], joined["cdi_class"])

    # Ensure all category columns exist
    for cat in ("Alert", "Warning", "Watch"):
        if cat not in ct.columns:
            ct[cat] = 0

    result = pd.DataFrame({
        country_col: ct.index,
        "alert": ct["Alert"].values,
        "warning": ct["Warning"].values,
        "watch": ct["Watch"].values,
        "total": ct.sum(axis=1).values,
    })

    # Raw score = weighted sum
    result["score"] = (
        SEVERITY_WEIGHTS["Alert"] * result["alert"]
        + SEVERITY_WEIGHTS["Warning"] * result["warning"]
        + SEVERITY_WEIGHTS["Watch"] * result["watch"]
    )

    # Severe count = Alert + Warning hexagons
    result["severe"] = result["alert"] + result["warning"]

    # Sort by severe count (highest first), break ties with raw score
    return (
        result
        .sort_values(["severe", "score"], ascending=[False, False])
        .reset_index(drop=True)
    )


def find_hotspot(gdf_country, severity_cats=("Alert", "Warning")):
    """
    Find the center of the densest cluster of Alert/Warning hexagons.
    Returns (center_lon, center_lat).
    """
    severe = gdf_country[gdf_country["cdi_class"].isin(severity_cats)].copy()
    if len(severe) == 0:
        centroid = gdf_country.dissolve().centroid.iloc[0]
        return (centroid.x, centroid.y)

    cluster = _find_densest_cluster(severe)
    weights = cluster["cdi_class"].map({"Alert": 2.0, "Warning": 1.0}).fillna(0.5)
    centroids = cluster.geometry.representative_point()
    w_sum = weights.sum()
    cx = (centroids.x * weights).sum() / w_sum
    cy = (centroids.y * weights).sum() / w_sum
    return (float(cx), float(cy))


def _find_densest_cluster(severe_gdf, search_radius=1.5):
    """
    Find the densest spatial cluster of severe hexagons.

    Uses a sliding-window approach: for each severe hexagon centroid,
    count how many other severe hexagons fall within `search_radius`
    degrees. The hexagon with the highest neighbor count is the
    cluster center. Returns the subset of hexagons within the cluster.

    Parameters
    ----------
    severe_gdf : GeoDataFrame
        GeoDataFrame of severe (Alert/Warning) hexagons.
    search_radius : float
        Search radius in degrees (~1.5° ≈ 165 km at equator).

    Returns
    -------
    GeoDataFrame
        Subset of severe_gdf belonging to the densest cluster.
    """
    if len(severe_gdf) <= 3:
        return severe_gdf

    # Weight: Alert=3, Warning=1
    weights = severe_gdf["cdi_class"].map({"Alert": 3.0, "Warning": 1.0}).fillna(0.5)
    centroids = severe_gdf.geometry.representative_point()
    xs = centroids.x.values
    ys = centroids.y.values
    ws = weights.values

    # For each hexagon, compute weighted count of neighbors within radius
    best_score = -1
    best_idx = 0
    for i in range(len(xs)):
        dists = np.sqrt((xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
        mask = dists <= search_radius
        score = ws[mask].sum()
        if score > best_score:
            best_score = score
            best_idx = i

    # Select all hexagons within radius of the densest point
    cx, cy = xs[best_idx], ys[best_idx]
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    cluster_mask = dists <= search_radius

    return severe_gdf.iloc[cluster_mask]


def hotspot_bbox_from_data(gdf_country, severity_cats=("Alert", "Warning"),
                           pad_factor=0.25):
    """
    Compute hotspot bounding box centered on the densest cluster of
    Alert/Warning hexagons, so the zoom targets where drought is
    most concentrated rather than spanning all scattered severe hexagons.

    Falls back to all hexagons if no severe hexagons exist.
    """
    severe = gdf_country[gdf_country["cdi_class"].isin(severity_cats)]
    if len(severe) == 0:
        severe = gdf_country  # fallback to all hexagons
        cluster = severe
    else:
        cluster = _find_densest_cluster(severe)

    bounds = cluster.total_bounds  # (minx, miny, maxx, maxy)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]

    # Ensure minimum size so we don't get a pinpoint bbox
    min_span = 2.0  # ~2 degrees minimum
    w = max(w, min_span)
    h = max(h, min_span)

    # Add padding
    pad_w = w * pad_factor
    pad_h = h * pad_factor
    return (
        bounds[0] - pad_w, bounds[1] - pad_h,
        bounds[2] + pad_w, bounds[3] + pad_h,
    )


# ── H3 → GeoDataFrame ────────────────────────────────────────────────────────

def h3_df_to_geodataframe(df):
    """Convert H3 aggregated DataFrame to GeoDataFrame with hex polygons."""
    def make_polygon(h3_idx):
        boundary = h3.cell_to_boundary(h3_idx)
        return Polygon([(lng, lat) for lat, lng in boundary])

    geometries = [make_polygon(idx) for idx in df["h3_index"]]
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf


# ── ZOOM BOUNDING BOX ────────────────────────────────────────────────────────

def get_zoom_bbox(gdf_full, zoom_fraction, center_lon, center_lat):
    """Compute viewport bbox for a given zoom fraction (1.0=full, 0.15=tight)."""
    bounds = gdf_full.total_bounds
    full_w = bounds[2] - bounds[0]
    full_h = bounds[3] - bounds[1]
    half_w = (full_w * zoom_fraction) / 2
    half_h = (full_h * zoom_fraction) / 2
    return (
        center_lon - half_w,
        center_lat - half_h,
        center_lon + half_w,
        center_lat + half_h,
    )


def _interpolate_bbox(bbox_a, bbox_b, t):
    """Linearly interpolate between two bounding boxes. t=0→a, t=1→b."""
    return tuple(a + (b - a) * t for a, b in zip(bbox_a, bbox_b))


# ── RENDER FRAME ─────────────────────────────────────────────────────────────

def render_frame(gdf, bbox, h3_res, frame_idx, total_frames, output_dir,
                 boundaries_gdf=None, subtitle=None, highlight_country=None,
                 admin1_gdf=None, admin1_label_col="NAME_1",
                 show_country_names=False, country_col="COUNTRY"):
    """Render one matplotlib frame and save as PNG.

    Parameters
    ----------
    subtitle : str, optional
        Second line of text below the main title (e.g. month label).
    highlight_country : GeoDataFrame, optional
        Single-country GeoDataFrame drawn with a thicker border.
    admin1_gdf : GeoDataFrame, optional
        Admin1 sub-national boundaries. Plotted with labels on zoomed frames.
    admin1_label_col : str
        Column name in admin1_gdf to use for labels.
    show_country_names : bool
        If True, label each country with its name on the map.
    country_col : str
        Column name in boundaries_gdf for country names.
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    minx, miny, maxx, maxy = bbox

    # Clip hexagons to viewport
    gdf_clip = gdf.cx[minx:maxx, miny:maxy]

    # Country boundaries (underneath)
    if boundaries_gdf is not None:
        boundaries_gdf.boundary.plot(
            ax=ax, color="#999999", linewidth=0.8, zorder=1,
        )

        # Country name labels
        if show_country_names and country_col in boundaries_gdf.columns:
            viewport = box(minx, miny, maxx, maxy)
            pts = boundaries_gdf.geometry.representative_point()
            visible = boundaries_gdf[pts.within(viewport)]
            for idx, row in visible.iterrows():
                pt = pts.loc[idx]
                ax.annotate(
                    row[country_col], xy=(pt.x, pt.y),
                    fontsize=8, color="#333333", ha="center", va="center",
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.7),
                )

    # Admin1 boundaries with labels
    if admin1_gdf is not None:
        admin1_gdf.boundary.plot(
            ax=ax, color="#AAAAAA", linewidth=0.5, linestyle="--", zorder=1,
        )
        viewport = box(minx, miny, maxx, maxy)
        centroids = admin1_gdf.geometry.representative_point()
        visible = admin1_gdf[centroids.within(viewport)]
        for idx, row in visible.iterrows():
            centroid = centroids.loc[idx]
            ax.annotate(
                row[admin1_label_col], xy=(centroid.x, centroid.y),
                fontsize=7, color="#666666", ha="center", va="center",
                fontweight="bold", zorder=5,
            )

    # Highlight target country
    if highlight_country is not None:
        highlight_country.boundary.plot(
            ax=ax, color="#333333", linewidth=2.0, zorder=3,
        )

    # Color hexagons by CDI category (vectorized)
    if len(gdf_clip) > 0:
        cat_col = gdf_clip["cdi_class"]
        if not cat_col.dtype == object:
            cat_col = cat_col.map(CDI_CATEGORY_MAP).fillna("No Data")
        colors = cat_col.map(CDI_COLORS_HEX).fillna("#C1C1C1")

        gdf_clip.plot(
            ax=ax, color=colors.values,
            edgecolor="#CCCCCC", linewidth=0.3, alpha=0.90, zorder=2,
        )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    # Title
    n_hex = len(gdf_clip)
    if n_hex > 0:
        title = f"CDI Drought Index  |  H3 res {h3_res}  |  {n_hex:,} hexagons"
    else:
        title = "CDI Drought Index  |  No drought detected in this area"
    ax.set_title(title, color="#222222", fontsize=13, fontweight="bold", pad=14)

    # Subtitle
    if subtitle:
        fig.text(
            0.50, 0.91, subtitle,
            color="#555555", fontsize=11, ha="center", fontstyle="italic",
        )

    # Legend — horizontal bar below the map
    legend_items = [
        ("Watch",   CDI_COLORS_HEX["Watch"]),
        ("Warning", CDI_COLORS_HEX["Warning"]),
        ("Alert",   CDI_COLORS_HEX["Alert"]),
    ]
    patches = [Patch(facecolor=c, edgecolor="#999", label=l) for l, c in legend_items]
    ax.legend(
        handles=patches, loc="upper center",
        bbox_to_anchor=(0.5, -0.02), ncol=3,
        fontsize=8, framealpha=0.9, facecolor=BG_COLOR,
        edgecolor="#CCCCCC", labelcolor="#333333",
    )

    # Progress bar
    progress = frame_idx / max(total_frames - 1, 1)
    bar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.010])
    bar_ax.barh(0, progress, color="#FFA604", height=1)
    bar_ax.barh(0, 1, color="#DDDDDD", height=1, zorder=0)
    bar_ax.set_xlim(0, 1)
    bar_ax.set_axis_off()

    fig.text(
        0.91, 0.025, f"Frame {frame_idx + 1}/{total_frames}",
        color="#888888", fontsize=8, ha="right",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    fig.savefig(frame_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_path


# ── TITLE FRAME ──────────────────────────────────────────────────────────────

def render_title_frame(title_text, subtitle_text, output_path, figsize=(12, 8)):
    """Render a title card frame (white background, centered text)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(
        0.5, 0.55, title_text,
        ha="center", va="center", fontsize=22, fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.5, 0.40, subtitle_text,
        ha="center", va="center", fontsize=16,
        color="#555555", fontstyle="italic",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ── STITCH GIF ────────────────────────────────────────────────────────────────

def frames_to_gif(frame_paths, output_path, durations=None,
                  duration_ms=FRAME_DURATION_MS):
    """Combine PNG frames into an animated GIF using Pillow.

    Processes frames one at a time to avoid holding all images in memory.
    """
    if not frame_paths:
        return

    if durations is None:
        durations = [duration_ms] * len(frame_paths)
        durations[0] = duration_ms * 3
        durations[-1] = duration_ms * 4

    # Load first frame as the base
    first = Image.open(frame_paths[0]).convert("RGBA")
    bg = Image.new("RGB", first.size, BG_COLOR_RGB)
    bg.paste(first, mask=first.split()[3])
    first.close()

    # Stream remaining frames to avoid holding all in memory
    remaining = []
    for path in frame_paths[1:]:
        img = Image.open(path).convert("RGBA")
        frame_bg = Image.new("RGB", img.size, BG_COLOR_RGB)
        frame_bg.paste(img, mask=img.split()[3])
        img.close()
        remaining.append(frame_bg)

    bg.save(
        output_path, save_all=True, append_images=remaining,
        duration=durations, loop=0, optimize=True,
    )

    # Free memory
    bg.close()
    for img in remaining:
        img.close()

    log.info("GIF saved → %s  (%d frames)", output_path, len(frame_paths))


# ── MAIN (single raster) ────────────────────────────────────────────────────

def generate_gif(
    raster_path,
    output_gif=None,
    resolutions=None,
    frames_per_res=FRAMES_PER_RES,
    zoom_center=None,
):
    """
    Generate an animated hexagon zoom GIF from a CDI raster.

    Parameters
    ----------
    raster_path : str
        Path to CDI GeoTIFF.
    output_gif : str, optional
        Output GIF path. Auto-generated if None.
    resolutions : list[int], optional
        H3 resolutions to animate through. Default: [3,4,5,6,7].
    frames_per_res : int
        Number of frames per resolution level.
    zoom_center : tuple(lon, lat), optional
        Center point for zoom. Uses data center if None.
    """
    if resolutions is None:
        resolutions = H3_RESOLUTIONS
    if output_gif is None:
        stem = os.path.splitext(os.path.basename(raster_path))[0]
        output_gif = os.path.join(OUTPUT_DIR, f"{stem}_hex_zoom.gif")

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    log.info("CDI Raster → H3 Hexagon Animated GIF")

    # 0. Load country boundaries
    boundaries_gdf = None
    shapefile_path = str(DEFAULT_SHAPEFILE)
    if os.path.exists(shapefile_path):
        boundaries_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        log.info("Loaded %d boundary features", len(boundaries_gdf))

    # 1. Extract drought pixels (once)
    log.info("Extracting drought pixels...")
    lats, lngs, values = RasterProcessor.extract_valid_pixels(
        raster_path, drought_only=True,
    )
    log.info("  %s drought pixels", f"{len(values):,}")

    # Compute center
    if zoom_center:
        center_lon, center_lat = zoom_center
    else:
        center_lon = float(np.mean(lngs))
        center_lat = float(np.mean(lats))

    # 2. Pre-aggregate at all resolutions
    log.info("Pre-aggregating at all H3 resolutions")
    aggregated = {}
    for res in resolutions:
        agg = H3Aggregator(res)
        h3_indices = agg.latlngs_to_h3(lats, lngs)
        df = agg.aggregate(h3_indices, values, method="mode", by_category=True)
        aggregated[res] = h3_df_to_geodataframe(df)
        log.info("  H3 res %d: %s hexagons", res, f"{len(aggregated[res]):,}")

    # 3. Build zoom sequence
    gdf_full = aggregated[resolutions[0]]
    total_frames = len(resolutions) * frames_per_res
    zoom_levels = np.linspace(1.0, 0.15, total_frames)

    frame_plan = []
    for i, res in enumerate(resolutions):
        start = i * frames_per_res
        end = start + frames_per_res
        for frac in zoom_levels[start:end]:
            frame_plan.append((res, frac))

    # 4. Render frames
    log.info("Rendering %d frames", total_frames)
    frame_paths = []
    for idx, (res, zoom_frac) in enumerate(frame_plan):
        bbox = get_zoom_bbox(gdf_full, zoom_frac, center_lon, center_lat)
        path = render_frame(
            gdf=aggregated[res], bbox=bbox, h3_res=res,
            frame_idx=idx, total_frames=total_frames,
            output_dir=FRAMES_DIR, boundaries_gdf=boundaries_gdf,
        )
        if path:
            frame_paths.append(path)

    # 5. Stitch GIF
    frames_to_gif(frame_paths, output_gif)
    log.info("Done! %s", output_gif)
    return output_gif


# ── MONTHLY GIF GENERATOR ────────────────────────────────────────────────────

def _generate_single_month_gif(
    raster_path, boundaries_gdf, admin1_gdf, icpac_bbox, output_gif,
    month_label, country_col="COUNTRY", admin1_label_col="NAME_1",
    pixel_data=None,
):
    """
    Generate one GIF for a single month with 3-act zoom:
      Act 1 — ICPAC overview (coarse hexagons, full region)
      Act 2 — Zoom into worst-Alert country
      Act 3 — Zoom into admin1 hotspot with fine hexagons

    Parameters
    ----------
    pixel_data : tuple(lats, lngs, values), optional
        Pre-extracted pixel data. If None, extracts from raster_path.
    country_col : str
        Column name for country identifier in boundaries_gdf.
    admin1_label_col : str
        Column name for admin1 labels in admin1_gdf.
    """
    log.info("%s", month_label)

    # ── Extract pixels (reuse if provided) ───────────────────────────
    if pixel_data is not None:
        lats, lngs, values = pixel_data
    else:
        lats, lngs, values = RasterProcessor.extract_valid_pixels(
            raster_path, drought_only=True,
        )
    log.info("  %s drought pixels", f"{len(values):,}")

    # ── Aggregate at required resolutions ────────────────────────────
    aggregated = {}
    for res in [4, 5, 6]:
        agg = H3Aggregator(res)
        h3_indices = agg.latlngs_to_h3(lats, lngs)
        df = agg.aggregate(h3_indices, values, method="mode", by_category=True)
        aggregated[res] = h3_df_to_geodataframe(df)
        log.info("  H3 res %d: %s hexagons", res, f"{len(aggregated[res]):,}")

    # ── Score countries (vectorized) ─────────────────────────────────
    ranking = score_countries_single(
        aggregated[5], boundaries_gdf, country_col=country_col,
    )
    log.info("  Country ranking (%s):", month_label)
    for _, row in ranking.head(5).iterrows():
        log.info(
            "    %-15s  Alert:%3d  Warning:%3d  Severe:%4d  Watch:%4d",
            row[country_col], row["alert"], row["warning"],
            row["severe"], row["watch"],
        )

    top_country = ranking.iloc[0][country_col]
    log.info("  → Target: %s (severe: %d)", top_country,
             ranking.iloc[0]["severe"])

    top_country_gdf = boundaries_gdf[boundaries_gdf[country_col] == top_country]

    # ── Find hotspot ─────────────────────────────────────────────────
    joined = gpd.sjoin(
        aggregated[5], top_country_gdf[[country_col, "geometry"]],
        how="inner", predicate="intersects",
    )
    hotspot_lon, hotspot_lat = find_hotspot(joined)
    log.info("  Hotspot: (%.4f, %.4f)", hotspot_lon, hotspot_lat)

    # ── Bounding boxes ───────────────────────────────────────────────
    cb = top_country_gdf.total_bounds
    pad_w = (cb[2] - cb[0]) * 0.15
    pad_h = (cb[3] - cb[1]) * 0.15
    country_bbox = (cb[0] - pad_w, cb[1] - pad_h,
                    cb[2] + pad_w, cb[3] + pad_h)

    # Data-driven hotspot bbox from severe hexagon extent
    hotspot_bbox = hotspot_bbox_from_data(joined)

    # ── Filter admin1 to target country ──────────────────────────────
    country_admin1 = None
    if admin1_gdf is not None and country_col in admin1_gdf.columns:
        country_admin1 = admin1_gdf[admin1_gdf[country_col] == top_country]

    # ── Build frame plan ─────────────────────────────────────────────
    act1_n = 5   # ICPAC overview
    act2_n = 5   # Country zoom
    act3_n = 5   # Admin1 hotspot zoom
    total_frames = act1_n + act2_n + act3_n

    frames_dir = os.path.join(FRAMES_DIR, month_label.replace(" ", "_"))
    os.makedirs(frames_dir, exist_ok=True)

    frame_plan = []

    # Act 1: ICPAC overview → start zooming toward country
    for i in range(act1_n):
        t = i / max(act1_n - 1, 1)
        bbox = _interpolate_bbox(icpac_bbox, country_bbox, t * 0.3)
        frame_plan.append({
            "gdf": aggregated[4], "bbox": bbox, "h3_res": 4,
            "subtitle": f"ICPAC Region — {month_label}",
            "highlight": top_country_gdf, "admin1": None,
            "country_names": True,
        })

    # Act 2: Zoom into country with admin1 boundaries
    for i in range(act2_n):
        t = i / max(act2_n - 1, 1)
        bbox_start = _interpolate_bbox(icpac_bbox, country_bbox, 0.3)
        bbox = _interpolate_bbox(bbox_start, country_bbox, t)
        frame_plan.append({
            "gdf": aggregated[5], "bbox": bbox, "h3_res": 5,
            "subtitle": f"Zooming into {top_country} — {month_label}",
            "highlight": top_country_gdf, "admin1": country_admin1,
            "country_names": False,
        })

    # Act 3: Zoom to admin1 hotspot with fine hexagons
    for i in range(act3_n):
        t = i / max(act3_n - 1, 1)
        bbox = _interpolate_bbox(country_bbox, hotspot_bbox, t)
        frame_plan.append({
            "gdf": aggregated[6], "bbox": bbox, "h3_res": 6,
            "subtitle": f"{top_country} — {month_label}",
            "highlight": top_country_gdf, "admin1": country_admin1,
            "country_names": False,
        })

    # ── Render frames ────────────────────────────────────────────────
    log.info("  Rendering %d frames...", total_frames)
    frame_paths = []
    for idx, plan in enumerate(frame_plan):
        path = render_frame(
            gdf=plan["gdf"], bbox=plan["bbox"], h3_res=plan["h3_res"],
            frame_idx=idx, total_frames=total_frames, output_dir=frames_dir,
            boundaries_gdf=boundaries_gdf, subtitle=plan["subtitle"],
            highlight_country=plan.get("highlight"),
            admin1_gdf=plan.get("admin1"),
            admin1_label_col=admin1_label_col,
            show_country_names=plan.get("country_names", False),
            country_col=country_col,
        )
        if path:
            frame_paths.append(path)

    if not frame_paths:
        log.warning("  No frames rendered — skipping GIF")
        return None

    # ── Stitch GIF ───────────────────────────────────────────────────
    durations = [FRAME_DURATION_MS] * len(frame_paths)
    durations[0] = 2000                    # Hold first frame
    durations[act1_n - 1] = 1000           # Pause before country zoom
    durations[act1_n + act2_n - 1] = 1000  # Pause before hotspot zoom
    durations[-1] = 3000                   # Hold final frame

    frames_to_gif(frame_paths, output_gif, durations=durations)
    size_kb = os.path.getsize(output_gif) / 1024
    log.info("  → %s  (%d frames, %.0f KB)", output_gif, len(frame_paths), size_kb)
    return output_gif


def generate_monthly_gifs(raster_dir="cdi_monthly", country_col="COUNTRY",
                          admin1_label_col="NAME_1", months=None):
    """
    Generate one GIF per month. Each GIF follows a 3-act zoom:
      1. ICPAC overview (coarse hexagons)
      2. Zoom into the country with highest severe count
      3. Zoom into admin1 hotspot with fine hexagons

    Parameters
    ----------
    raster_dir : str
        Directory containing CDI rasters.
    country_col : str
        Column name for country in shapefile.
    admin1_label_col : str
        Column name for admin1 labels in admin1 shapefile.
    months : list[str], optional
        Month abbreviations to process (e.g. ["aug", "sep"]).
        If None, all discovered rasters are processed.

    Returns list of output GIF paths.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("CDI Monthly Drought GIFs — One per Month")
    log.info("=" * 60)

    # Discover rasters
    raster_paths = discover_rasters(raster_dir)

    # Filter to specified months if provided
    if months:
        month_set = {m.lower()[:3] for m in months}
        raster_paths = [
            p for p in raster_paths
            if re.search(r"-(\w{3})\.", os.path.basename(p))
            and re.search(r"-(\w{3})\.", os.path.basename(p)).group(1).lower()
            in month_set
        ]
        if not raster_paths:
            log.error("No rasters found for months: %s", ", ".join(months))
            return []

    log.info("Found %d CDI rasters:", len(raster_paths))
    for rp in raster_paths:
        log.info("  • %-15s  %s", parse_month_label(rp), rp)

    # Load boundaries (shared across all months)
    shapefile_path = str(DEFAULT_SHAPEFILE)
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    boundaries_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    log.info("Loaded %d country boundaries", len(boundaries_gdf))

    admin1_gdf = None
    if os.path.exists(ADMIN1_SHAPEFILE):
        admin1_gdf = gpd.read_file(ADMIN1_SHAPEFILE).to_crs(epsg=4326)
        log.info("Loaded %d admin1 boundaries", len(admin1_gdf))

    # ICPAC bbox (shared)
    full_bounds = boundaries_gdf.total_bounds
    icpac_bbox = (full_bounds[0] - 1, full_bounds[1] - 1,
                  full_bounds[2] + 1, full_bounds[3] + 1)

    # Generate one GIF per month
    output_gifs = []
    for rpath in raster_paths:
        month_label = parse_month_label(rpath)

        # Extract once, pass to _generate_single_month_gif
        try:
            pixel_data = RasterProcessor.extract_valid_pixels(
                rpath, drought_only=True,
            )
        except Exception as e:
            log.warning("SKIPPING %s (corrupt: %s)", month_label, e)
            continue

        stem = os.path.splitext(os.path.basename(rpath))[0]
        output_gif = os.path.join(OUTPUT_DIR, f"{stem}_monthly.gif")

        result = _generate_single_month_gif(
            raster_path=rpath,
            boundaries_gdf=boundaries_gdf,
            admin1_gdf=admin1_gdf,
            icpac_bbox=icpac_bbox,
            output_gif=output_gif,
            month_label=month_label,
            country_col=country_col,
            admin1_label_col=admin1_label_col,
            pixel_data=pixel_data,
        )
        if result:
            output_gifs.append(result)

    log.info("=" * 60)
    log.info("Done! Generated %d GIFs:", len(output_gifs))
    for g in output_gifs:
        log.info("  → %s", g)
    return output_gifs


# ── HEX PANEL VISUALIZATION ─────────────────────────────────────────────────

# Panel theme colors
_PANEL_BG = "#FFFFFF"           # White background
_PANEL_CARD_BG = "#FFFFFF"     # White card background for maps
_PANEL_ACCENT = "#00A5CF"      # ICPAC-style teal accent
_PANEL_TEXT = "#2C3E50"        # Dark text on white bg
_PANEL_TEXT_DARK = "#2C3E50"   # Dark text
_PANEL_CARD_BORDER = "#B0BEC5" # Subtle grey card border
_PANEL_MONTH_RIBBON = {
    "January": "#3498DB", "February": "#2980B9", "March": "#27AE60",
    "April": "#2ECC71", "May": "#F1C40F", "June": "#F39C12",
    "July": "#E67E22", "August": "#E74C3C", "September": "#C0392B",
    "October": "#8E44AD", "November": "#9B59B6", "December": "#2C3E50",
}


def _render_hex_panel_cell(ax, gdf, bbox, boundaries_gdf, title,
                           country_col="COUNTRY", logo_img=None):
    """Render a single hexagonal map cell within a panel axis."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.patches import FancyBboxPatch

    minx, miny, maxx, maxy = bbox
    gdf_clip = gdf.cx[minx:maxx, miny:maxy]

    ax.set_facecolor(_PANEL_CARD_BG)

    # Hexagons — drawn first (lower zorder)
    if len(gdf_clip) > 0:
        cat_col = gdf_clip["cdi_class"]
        if not cat_col.dtype == object:
            cat_col = cat_col.map(CDI_CATEGORY_MAP).fillna("No Data")
        colors = cat_col.map(CDI_COLORS_HEX).fillna("#C1C1C1")
        gdf_clip.plot(
            ax=ax, color=colors.values,
            edgecolor="#E0E0E0", linewidth=0.15, alpha=0.92, zorder=1,
        )

    # Country boundaries — on top of hexagons
    if boundaries_gdf is not None:
        boundaries_gdf.boundary.plot(
            ax=ax, color="#555555", linewidth=0.7, zorder=3,
        )
        # Country labels
        if country_col in boundaries_gdf.columns:
            viewport = box(minx, miny, maxx, maxy)
            pts = boundaries_gdf.geometry.representative_point()
            visible = boundaries_gdf[pts.within(viewport)]
            for idx, row in visible.iterrows():
                pt = pts.loc[idx]
                ax.annotate(
                    row[country_col], xy=(pt.x, pt.y),
                    fontsize=5.5, color="#333333", ha="center", va="center",
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.75),
                )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    # Month ribbon banner at top
    month_name = title.split()[0] if " " in title else title
    ribbon_color = _PANEL_MONTH_RIBBON.get(month_name, _PANEL_ACCENT)
    ribbon = FancyBboxPatch(
        (0.02, 0.88), 0.96, 0.10, transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=ribbon_color, edgecolor="none", alpha=0.88, zorder=8,
    )
    ax.add_patch(ribbon)
    ax.text(
        0.50, 0.935, title, transform=ax.transAxes,
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", zorder=9,
    )

    # ICPAC logo inside bottom-left corner
    if logo_img is not None:
        imagebox = OffsetImage(logo_img, zoom=0.045)
        ab = AnnotationBbox(
            imagebox, (0.04, 0.04), xycoords="axes fraction",
            frameon=False, box_alignment=(0.0, 0.0), zorder=10,
        )
        ax.add_artist(ab)

    # Hex count badge — bottom right
    n_hex = len(gdf_clip)
    if n_hex > 0:
        badge = FancyBboxPatch(
            (0.72, 0.02), 0.26, 0.06, transform=ax.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#37474F", edgecolor="none", alpha=0.80, zorder=8,
        )
        ax.add_patch(badge)
        ax.text(
            0.85, 0.05, f"{n_hex:,} hex", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold", zorder=9,
        )

    # Card border with rounded feel
    card_border = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        facecolor="none", edgecolor=_PANEL_CARD_BORDER,
        linewidth=1.2, zorder=11,
    )
    ax.add_patch(card_border)


def generate_hex_panel(raster_dir="cdi_monthly", months=None,
                       h3_res=6, country_col="COUNTRY",
                       output_path=None):
    """
    Generate a panel visualization with one hexagonal map per month.

    Parameters
    ----------
    raster_dir : str
        Directory containing CDI rasters.
    months : list[str], optional
        Month abbreviations to include (e.g. ["aug", "sep"]).
        If None, all rasters in raster_dir are used.
    h3_res : int
        H3 resolution for hexagons (default 6 = fine detail).
    country_col : str
        Column name for country in shapefile.
    output_path : str, optional
        Output PNG path. Auto-generated if None.

    Returns path to output panel PNG.
    """
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.gridspec import GridSpec
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    log.info("=" * 60)
    log.info("CDI Hexagonal Panel Visualization")
    log.info("=" * 60)

    # Discover and filter rasters
    raster_paths = discover_rasters(raster_dir)
    if months:
        month_set = {m.lower()[:3] for m in months}
        raster_paths = [
            p for p in raster_paths
            if re.search(r"-(\w{3})\.", os.path.basename(p))
            and re.search(r"-(\w{3})\.", os.path.basename(p)).group(1).lower()
            in month_set
        ]
    if not raster_paths:
        log.error("No rasters found")
        return None

    log.info("Processing %d months at H3 res %d:", len(raster_paths), h3_res)
    for rp in raster_paths:
        log.info("  • %s", parse_month_label(rp))

    # Load boundaries
    shapefile_path = str(DEFAULT_SHAPEFILE)
    boundaries_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    full_bounds = boundaries_gdf.total_bounds
    icpac_bbox = (full_bounds[0] - 1, full_bounds[1] - 1,
                  full_bounds[2] + 1, full_bounds[3] + 1)

    # Load ICPAC logo
    logo_path = os.path.join(os.path.dirname(__file__), "assets",
                             "ICPAC_LOGO_WithSLogan.width-800 (1) (1).png")
    logo_img = None
    if os.path.exists(logo_path):
        logo_img = plt.imread(logo_path)
        log.info("Loaded ICPAC logo from %s", logo_path)
    else:
        log.warning("ICPAC logo not found at %s — skipping", logo_path)

    # Pre-aggregate all months
    month_data = []
    for rpath in raster_paths:
        month_label = parse_month_label(rpath)
        try:
            lats, lngs, values = RasterProcessor.extract_valid_pixels(
                rpath, drought_only=True,
            )
        except Exception as e:
            log.warning("  SKIPPING %s (corrupt: %s)", month_label, e)
            continue

        agg = H3Aggregator(h3_res)
        h3_indices = agg.latlngs_to_h3(lats, lngs)
        df = agg.aggregate(h3_indices, values, method="mode", by_category=True)
        gdf = h3_df_to_geodataframe(df)
        log.info("  %s: %s hexagons", month_label, f"{len(gdf):,}")
        month_data.append((month_label, gdf))

    if not month_data:
        log.error("No valid months to render")
        return None

    # ── Layout: header row + map grid + footer ───────────────────
    n = len(month_data)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    # Figure: dark background, generous size
    fig_w = ncols * 7
    fig_h = nrows * 6 + 2.4  # extra space for header + footer
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_PANEL_BG)

    # GridSpec: row 0 = header, rows 1..nrows = maps, last row = footer
    gs = GridSpec(
        nrows + 2, ncols, figure=fig,
        height_ratios=[0.6] + [1.0] * nrows + [0.25],
        wspace=0.06, hspace=0.10,
    )

    # ── Header ────────────────────────────────────────────────────
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(_PANEL_BG)
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.set_axis_off()

    # Dynamic title
    first_m = re.search(r"(\d{4})", os.path.basename(raster_paths[0]))
    year_str = first_m.group(1) if first_m else ""
    month_labels_list = [ml.split()[0] for ml, _ in month_data]
    if len(month_labels_list) > 2:
        months_str = f"{month_labels_list[0]}\u2013{month_labels_list[-1]}"
    else:
        months_str = " & ".join(month_labels_list)

    ax_header.text(
        0.50, 0.72,
        "Combined Drought Index (CDI)",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color="#1a1a2e", family="sans-serif",
    )
    ax_header.text(
        0.50, 0.28,
        f"ICPAC Region  |  {months_str} {year_str}",
        ha="center", va="center", fontsize=11,
        color="#607D8B", family="sans-serif",
    )
    # Accent underline
    ax_header.plot(
        [0.25, 0.75], [0.05, 0.05],
        color=_PANEL_ACCENT, linewidth=2.5, transform=ax_header.transAxes,
        solid_capstyle="round",
    )

    # ── Map cells ─────────────────────────────────────────────────
    for i, (month_label, gdf) in enumerate(month_data):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row + 1, col])

        _render_hex_panel_cell(
            ax=ax, gdf=gdf, bbox=icpac_bbox,
            boundaries_gdf=boundaries_gdf, title=month_label,
            country_col=country_col, logo_img=logo_img,
        )

    # Hide unused cells (e.g. 5 months = 2×3 grid with 1 empty)
    for i in range(n, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax_empty = fig.add_subplot(gs[row + 1, col])
        ax_empty.set_facecolor(_PANEL_BG)
        ax_empty.set_axis_off()

    # ── Footer: legend bar ────────────────────────────────────────
    ax_footer = fig.add_subplot(gs[-1, :])
    ax_footer.set_facecolor(_PANEL_BG)
    ax_footer.set_xlim(0, 1)
    ax_footer.set_ylim(0, 1)
    ax_footer.set_axis_off()

    legend_items = [
        ("Watch",   CDI_COLORS_HEX["Watch"]),
        ("Warning", CDI_COLORS_HEX["Warning"]),
        ("Alert",   CDI_COLORS_HEX["Alert"]),
    ]
    # Draw legend as colored pills with labels
    x_start = 0.30
    x_step = 0.15
    for j, (label, color) in enumerate(legend_items):
        cx = x_start + j * x_step
        pill = FancyBboxPatch(
            (cx - 0.025, 0.35), 0.05, 0.30, transform=ax_footer.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=color, edgecolor="none", alpha=0.95,
        )
        ax_footer.add_patch(pill)
        ax_footer.text(
            cx + 0.04, 0.50, label, transform=ax_footer.transAxes,
            ha="left", va="center", fontsize=10, color="#333333",
            fontweight="bold",
        )

    # Source credit
    ax_footer.text(
        0.98, 0.50, "Source: ICPAC Drought Watch",
        transform=ax_footer.transAxes, ha="right", va="center",
        fontsize=7.5, color="#607D8B", fontstyle="italic",
    )

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        if months:
            tag = "_".join(m.lower()[:3] for m in months)
        else:
            tag = "all"
        year_tag = year_str if year_str else "panel"
        output_path = os.path.join(OUTPUT_DIR, f"cdi_hex_panel_{year_tag}_{tag}.png")

    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info("Panel saved → %s  (%.1f MB)", output_path, size_mb)
    return output_path


# ── COMBINED GIF ─────────────────────────────────────────────────────────────

def generate_combined_gif(raster_dir="cdi_monthly", months=None,
                          country_col="COUNTRY", admin1_label_col="NAME_1"):
    """
    Generate individual monthly GIFs, then stitch them into a single
    combined GIF with a title card before each month.

    Each month section:
      1. Title frame: "Combined Drought Indicator Analysis, {Year} {Month}"
      2. The 15-frame monthly GIF (ICPAC → country → hotspot)

    Parameters
    ----------
    raster_dir : str
        Directory containing CDI rasters.
    months : list[str]
        Month abbreviations (e.g. ["aug", "sep", "oct", "nov", "dec"]).
    country_col : str
        Column name for country in shapefile.
    admin1_label_col : str
        Column name for admin1 labels in admin1 shapefile.

    Returns path to combined GIF.
    """
    # Step 1: Generate individual monthly GIFs
    monthly_gifs = generate_monthly_gifs(
        raster_dir=raster_dir, country_col=country_col,
        admin1_label_col=admin1_label_col, months=months,
    )
    if not monthly_gifs:
        log.error("No monthly GIFs generated — cannot create combined GIF")
        return None

    log.info("\n" + "=" * 60)
    log.info("Combining %d months into single GIF ...", len(monthly_gifs))

    # Step 2: Build combined frame list with title cards
    title_dir = os.path.join(FRAMES_DIR, "title_cards")
    os.makedirs(title_dir, exist_ok=True)

    combined_frames = []
    combined_durations = []

    for gif_path in monthly_gifs:
        # Parse month label from filename
        stem = os.path.splitext(os.path.basename(gif_path))[0]
        month_label = parse_month_label(stem)
        m = re.search(r"(\d{4})-(\w{3})", stem)
        if m:
            year = m.group(1)
            month_name = MONTH_NAMES.get(m.group(2).lower(), m.group(2))
        else:
            year = ""
            month_name = month_label

        # Render title card
        title_path = os.path.join(title_dir, f"title_{stem}.png")
        render_title_frame(
            title_text="Combined Drought Indicator Analysis",
            subtitle_text=f"{year}  {month_name}",
            output_path=title_path,
        )
        combined_frames.append(title_path)
        combined_durations.append(2500)  # Hold title for 2.5s

        # Extract frames from the monthly GIF
        monthly_img = Image.open(gif_path)
        n_frames = getattr(monthly_img, "n_frames", 1)
        for i in range(n_frames):
            monthly_img.seek(i)
            frame_path = os.path.join(title_dir, f"{stem}_f{i:03d}.png")
            monthly_img.save(frame_path)
            combined_frames.append(frame_path)

            # Preserve original frame durations
            duration = monthly_img.info.get("duration", FRAME_DURATION_MS)
            combined_durations.append(duration)

        monthly_img.close()

    # Step 3: Stitch combined GIF
    # Determine output filename from months
    if months:
        month_tag = "_".join(m.lower()[:3] for m in months)
    else:
        month_tag = "all"

    # Extract year from first GIF path
    first_m = re.search(r"(\d{4})", os.path.basename(monthly_gifs[0]))
    year_tag = first_m.group(1) if first_m else "combined"

    output_path = os.path.join(
        OUTPUT_DIR, f"cdi_combined_{year_tag}_{month_tag}.gif",
    )

    frames_to_gif(combined_frames, output_path, durations=combined_durations)
    size_kb = os.path.getsize(output_path) / 1024
    log.info("Combined GIF → %s  (%d frames, %.0f KB)",
             output_path, len(combined_frames), size_kb)
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CDI Raster → H3 Hexagon Animated GIF",
    )
    parser.add_argument("-i", "--input", default=None,
                        help="Input CDI raster (GeoTIFF)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output GIF path")
    parser.add_argument(
        "--persistence", action="store_true",
        help="Generate one GIF per month (auto-discovers all rasters)",
    )
    parser.add_argument(
        "--panel", action="store_true",
        help="Generate a hex panel PNG (one cell per month)",
    )
    parser.add_argument(
        "--months", nargs="+", default=None,
        metavar="MON",
        help="Month abbreviations to process (e.g. aug sep oct nov dec)",
    )
    parser.add_argument(
        "--raster-dir", default="cdi_monthly",
        help="Directory containing CDI rasters (for --persistence mode)",
    )
    parser.add_argument(
        "--country-col", default="COUNTRY",
        help="Column name for country in shapefile (default: COUNTRY)",
    )
    parser.add_argument(
        "--admin1-col", default="NAME_1",
        help="Column name for admin1 labels (default: NAME_1)",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=H3_RESOLUTIONS,
        help=f"H3 resolutions to animate (default: {H3_RESOLUTIONS})",
    )
    parser.add_argument("--frames-per-res", type=int, default=FRAMES_PER_RES)
    parser.add_argument(
        "--center", type=float, nargs=2, default=None,
        metavar=("LON", "LAT"), help="Zoom center (lon lat)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.panel:
        # Panel mode: one hex map per month in a grid
        generate_hex_panel(
            raster_dir=args.raster_dir,
            months=args.months,
            country_col=args.country_col,
            output_path=args.output,
        )
    elif args.persistence:
        if args.months:
            # Combined mode: individual GIFs + stitched combined GIF
            generate_combined_gif(
                raster_dir=args.raster_dir,
                months=args.months,
                country_col=args.country_col,
                admin1_label_col=args.admin1_col,
            )
        else:
            generate_monthly_gifs(
                raster_dir=args.raster_dir,
                country_col=args.country_col,
                admin1_label_col=args.admin1_col,
            )
    else:
        if not args.input:
            parser.error("-i/--input is required unless using --persistence or --panel")
        generate_gif(
            raster_path=args.input,
            output_gif=args.output,
            resolutions=args.resolutions,
            frames_per_res=args.frames_per_res,
            zoom_center=tuple(args.center) if args.center else None,
        )
