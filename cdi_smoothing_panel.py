#!/usr/bin/env python3
"""
cdi_smoothing_panel.py
----------------------
CDI Gaussian Smoothed Panel — Aug to Dec 2025.

Applies the same smoothing pipeline from cdi_smoothing.py to each month
and produces a publication-quality side-by-side panel (original vs smoothed).

Uses shared configuration from cdi_config.py.
"""

import os
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.plot import show
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from cdi_config import (
    PATHS, CDI, MONTH_ABBR_TO_NAME,
    load_boundaries, load_lakes,
)

# ==============================
# CONFIGURATION
# ==============================
PATHS.ensure_dirs()
output_dir = str(PATHS.OUTPUT_DIR)

gaussian_sigma = 5

months = [
    ("aug", "August"),
    ("sep", "September"),
    ("oct", "October"),
    ("nov", "November"),
    ("dec", "December"),
]

# CDI color scheme (from shared config)
cdi_colors = ListedColormap(CDI.SMOOTH_COLORS)

legend_labels = [
    mpatches.Patch(facecolor="#ffffff", label="No Drought", edgecolor="#cccccc"),
    mpatches.Patch(color="#ffff00", label="Watch"),
    mpatches.Patch(color="#ff9900", label="Warning"),
    mpatches.Patch(color="#cc0000", label="Alert"),
]

# ==============================
# LOAD SHARED VECTOR DATA
# ==============================
admin_boundaries = load_boundaries("countries")


def process_month(month_abbr):
    """
    Run the full Gaussian smoothing pipeline for one month.
    Returns (input_path, smoothed_path, smoothed_lakes_gdf).
    """
    input_raster = str(PATHS.monthly_raster(2025, month_abbr))
    output_gaussian = os.path.join(output_dir, f"cdi_gaussian_2025_{month_abbr}.tif")

    # --- Load raster ---
    with rasterio.open(input_raster) as src:
        cdi = src.read(1).astype(np.float64)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata
        if nodata is not None:
            cdi = np.where(cdi == nodata, np.nan, cdi)

    # --- Clip & smooth lakes via shared loader ---
    raster_bounds = rasterio.transform.array_bounds(
        cdi.shape[0], cdi.shape[1], transform
    )
    lakes_smooth = load_lakes(clip_bounds=raster_bounds, smooth=True)

    # --- Burn lakes as nodata ---
    if not lakes_smooth.empty:
        lake_mask = geometry_mask(
            lakes_smooth.geometry,
            out_shape=cdi.shape,
            transform=transform,
            invert=True,
        )
        cdi[lake_mask] = np.nan

    # --- Normalized Gaussian smoothing ---
    valid_mask = ~np.isnan(cdi)
    cdi_filled = np.where(valid_mask, cdi, 0.0)
    mask_float = valid_mask.astype(np.float64)

    smoothed_values = gaussian_filter(cdi_filled, sigma=gaussian_sigma)
    smoothed_weights = gaussian_filter(mask_float, sigma=gaussian_sigma)
    smoothed_weights[smoothed_weights == 0] = 1
    cdi_gaussian = smoothed_values / smoothed_weights
    cdi_gaussian[~valid_mask] = np.nan

    # --- Write output raster ---
    profile.update(dtype=rasterio.float32)
    with rasterio.open(output_gaussian, "w", **profile) as dst:
        dst.write(cdi_gaussian.astype(np.float32), 1)

    print(f"  {month_abbr.upper()} smoothing complete → {os.path.basename(output_gaussian)}")
    return input_raster, output_gaussian, lakes_smooth


# ==============================
# PROCESS ALL MONTHS
# ==============================
print("Processing months...")
results = {}
for abbr, full_name in months:
    raw_path, smooth_path, lakes_sm = process_month(abbr)
    results[abbr] = {
        "raw": raw_path,
        "smooth": smooth_path,
        "name": full_name,
        "lakes": lakes_sm,
    }

# ==============================
# CREATE PANEL FIGURE (Original vs Smoothed)
# ==============================
print("Building panel figure...")

n_months = len(months)
fig = plt.figure(figsize=(16, n_months * 5.5), facecolor="white")

gs = GridSpec(
    n_months + 1, 2,
    figure=fig,
    wspace=0.04,
    hspace=0.18,
    left=0.04, right=0.96,
    top=0.94, bottom=0.03,
    height_ratios=[0.12] + [1.0] * n_months,
)

# Suptitle
fig.suptitle(
    "Combined Drought Indicator (CDI)\n"
    "August – December 2025",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

# Column headers in the first row
for col_idx, col_title in enumerate(["Pixelated CDI", "Gaussian Smoothed"]):
    ax_header = fig.add_subplot(gs[0, col_idx])
    ax_header.set_axis_off()
    ax_header.text(
        0.5, 0.5, col_title,
        transform=ax_header.transAxes,
        ha="center", va="center",
        fontsize=15, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#E3F2FD" if col_idx == 0 else "#E8F5E9",
            edgecolor="#90CAF9" if col_idx == 0 else "#A5D6A7",
            alpha=0.9,
        ),
    )

# Month label accent colours
month_accent = {
    "aug": "#FFF8E1",
    "sep": "#FFF3E0",
    "oct": "#FFE0B2",
    "nov": "#FFCCBC",
    "dec": "#FFCDD2",
}

# Load ICPAC logo once
logo_img = plt.imread(str(PATHS.LOGO))


def plot_map(ax, raster_path, lakes_gdf, title_text, title_bg, show_ylabel, show_xlabel):
    """Plot a CDI raster with lakes and boundaries on an axis."""
    with rasterio.open(raster_path) as src:
        show(src, ax=ax, cmap=cdi_colors, vmin=0, vmax=3)
    lakes_gdf.plot(ax=ax, color="#87CEEB", edgecolor="#4682B4", linewidth=0.3)
    admin_boundaries.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

    # ICPAC logo — bottom right
    imagebox = OffsetImage(logo_img, zoom=0.045, alpha=0.85)
    ab = AnnotationBbox(
        imagebox, (1.0, 0.0),
        xycoords="axes fraction",
        box_alignment=(1.0, 0.0),
        pad=0.2,
        frameon=False,
    )
    ax.add_artist(ab)

    ax.set_title(
        title_text,
        fontsize=12,
        fontweight="bold",
        pad=6,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=title_bg,
            edgecolor="#999999",
            alpha=0.9,
        ),
    )

    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=9)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    if show_xlabel:
        ax.set_xlabel("Longitude", fontsize=9)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    ax.tick_params(labelsize=7)


for idx, (abbr, full_name) in enumerate(months):
    row = idx + 1  # offset by 1 for header row
    info = results[abbr]
    is_last_row = (idx == n_months - 1)

    # Left column — Original (pixelated)
    ax_raw = fig.add_subplot(gs[row, 0])
    plot_map(
        ax_raw, info["raw"], info["lakes"],
        title_text=f"{full_name} 2025",
        title_bg=month_accent[abbr],
        show_ylabel=True,
        show_xlabel=is_last_row,
    )

    # Right column — Gaussian smoothed
    ax_smooth = fig.add_subplot(gs[row, 1])
    plot_map(
        ax_smooth, info["smooth"], info["lakes"],
        title_text=f"{full_name} 2025",
        title_bg=month_accent[abbr],
        show_ylabel=False,
        show_xlabel=is_last_row,
    )

# Shared legend at the bottom
fig.legend(
    handles=legend_labels,
    loc="lower center",
    ncol=4,
    fontsize=12,
    title="CDI Classification",
    title_fontsize=13,
    frameon=True,
    fancybox=True,
    shadow=True,
    edgecolor="#666666",
    bbox_to_anchor=(0.5, 0.005),
)

# Data source note — bottom right
fig.text(
    0.96, 0.008,
    "Data: East Africa Drought Watch  |  Boundaries: ICPAC Countries",
    ha="right", va="bottom",
    fontsize=8, color="#555555", style="italic",
)

# Save
output_panel = os.path.join(output_dir, "CDI_Gaussian_Panel_Aug_Dec_2025.png")
fig.savefig(output_panel, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nPanel saved → {output_panel}")
