#!/usr/bin/env python3
"""
cdi_smoothing.py
----------------
CDI Raster Smoothing and Visualization.

Applies Gaussian, mean, and median filters to a CDI raster with lake
masking and normalized convolution for smooth borders. Produces
classified polygons and a publication-quality map.

Uses shared configuration from cdi_config.py.
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes, geometry_mask
from rasterio.plot import show
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import geopandas as gpd

from cdi_config import PATHS, CDI, load_boundaries, load_lakes

# ==============================
# USER INPUTS
# ==============================
PATHS.ensure_dirs()

input_raster   = str(PATHS.monthly_raster(2025, "aug"))
output_dir     = str(PATHS.OUTPUT_DIR)
output_gaussian  = os.path.join(output_dir, "cdi_gaussian.tif")
output_mean      = os.path.join(output_dir, "cdi_mean.tif")
output_median    = os.path.join(output_dir, "cdi_median.tif")
output_resampled = os.path.join(output_dir, "cdi_resampled.tif")
output_polygons  = os.path.join(output_dir, "cdi_smoothed_polygons.shp")

gaussian_sigma = 5       # controls smoothing strength
window_size    = 5       # moving window size (3x3, 5x5 etc.)
resample_scale = 2       # increase resolution factor (2 = double pixel size)

# ==============================
# LOAD CDI RASTER
# ==============================
with rasterio.open(input_raster) as src:
    cdi = src.read(1)
    profile = src.profile
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    if nodata is not None:
        cdi = np.where(cdi == nodata, np.nan, cdi)

# ==============================
# MASK LAKES FROM CDI BEFORE SMOOTHING
# ==============================
raster_bounds = rasterio.transform.array_bounds(cdi.shape[0], cdi.shape[1], transform)
lakes_smoothed = load_lakes(clip_bounds=raster_bounds, smooth=True)

if not lakes_smoothed.empty:
    lake_mask = geometry_mask(
        lakes_smoothed.geometry,
        out_shape=cdi.shape,
        transform=transform,
        invert=True,
    )
    cdi[lake_mask] = np.nan

# ==============================
# PREPARE DATA FOR SMOOTH BORDERS
# ==============================
valid_mask = ~np.isnan(cdi)
cdi_filled = np.where(valid_mask, cdi, 0.0)
mask_float = valid_mask.astype(np.float64)

# ==============================
# 1. GAUSSIAN SMOOTHING
# ==============================
smoothed_values = gaussian_filter(cdi_filled, sigma=gaussian_sigma)
smoothed_weights = gaussian_filter(mask_float, sigma=gaussian_sigma)
smoothed_weights[smoothed_weights == 0] = 1
cdi_gaussian = smoothed_values / smoothed_weights
cdi_gaussian[~valid_mask] = np.nan

profile.update(dtype=rasterio.float32)
with rasterio.open(output_gaussian, "w", **profile) as dst:
    dst.write(cdi_gaussian.astype(np.float32), 1)
print("Gaussian smoothing complete.")

# ==============================
# 2. MOVING WINDOW MEAN FILTER
# ==============================
mean_values = uniform_filter(cdi_filled, size=window_size)
mean_weights = uniform_filter(mask_float, size=window_size)
mean_weights[mean_weights == 0] = 1
cdi_mean = mean_values / mean_weights
cdi_mean[~valid_mask] = np.nan

with rasterio.open(output_mean, "w", **profile) as dst:
    dst.write(cdi_mean.astype(np.float32), 1)
print("Mean filter smoothing complete.")

# ==============================
# 3. MOVING WINDOW MEDIAN FILTER
# ==============================
cdi_median = median_filter(cdi_filled.astype(np.float32), size=window_size)
cdi_median[~valid_mask] = np.nan

with rasterio.open(output_median, "w", **profile) as dst:
    dst.write(cdi_median.astype(np.float32), 1)
print("Median filter smoothing complete.")

# ==============================
# 4. OPTIONAL RESAMPLING (COARSER OUTPUT)
# ==============================
with rasterio.open(input_raster) as src:
    new_height = int(src.height / resample_scale)
    new_width = int(src.width / resample_scale)
    data = src.read(
        1,
        out_shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )
    new_transform = src.transform * src.transform.scale(
        (src.width / new_width),
        (src.height / new_height),
    )
    profile_resampled = src.profile.copy()
    profile_resampled.update({
        "height": new_height,
        "width": new_width,
        "transform": new_transform,
    })
with rasterio.open(output_resampled, "w", **profile_resampled) as dst:
    dst.write(data.astype(np.float32), 1)
print("Resampling complete.")

# ==============================
# 5. CONVERT SMOOTHED CDI TO POLYGONS
#    (Watch / Warning / Alert zones)
# ==============================
classified = np.zeros_like(cdi_gaussian)
classified[(cdi_gaussian > 0.5) & (cdi_gaussian <= 1.5)] = 1
classified[(cdi_gaussian > 1.5) & (cdi_gaussian <= 2.5)] = 2
classified[cdi_gaussian > 2.5] = 3

results = (
    {"properties": {"class": v}, "geometry": s}
    for s, v in shapes(classified.astype(np.int16), transform=transform)
    if v > 0
)
gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)
gdf.to_file(output_polygons)
print("Polygon extraction complete.")

# ==============================
# 6. VISUALIZATION
# ==============================
cdi_colors = ListedColormap(CDI.SMOOTH_COLORS)

legend_labels = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(CDI.SMOOTH_COLORS, CDI.LABELS)
]

admin_boundaries = load_boundaries("countries")

fig, ax = plt.subplots(figsize=(10, 8))
with rasterio.open(output_gaussian) as src:
    show(src, ax=ax, cmap=cdi_colors, vmin=0, vmax=3)
lakes_smoothed.plot(ax=ax, color="#87CEEB", edgecolor="#4682B4", linewidth=0.3)
admin_boundaries.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)
ax.set_title("Gaussian Smoothed CDI")
ax.legend(handles=legend_labels, loc="lower left", title="CDI Status")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "Gaussian_Smoothed_CDI.png"),
    dpi=600, bbox_inches="tight", facecolor="white",
)
print("Visualization complete.")
