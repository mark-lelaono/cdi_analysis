#!/usr/bin/env python3
"""
cdi_config.py
-------------
Shared configuration for all CDI analysis scripts.

Centralises paths, constants, color schemes, and helper utilities so that
every script in the project imports from one place instead of redefining
its own versions.

Usage:
    from cdi_config import BASE_DIR, PATHS, CDI, load_boundaries, load_lakes
"""

import os
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


# ──────────────────────────────────────────────────────────────────────────────
# PATHS — all relative to BASE_DIR, overridable via .env
# ──────────────────────────────────────────────────────────────────────────────
class PATHS:
    INPUT_DIR   = Path(os.environ.get("CDI_INPUT_DIR",   BASE_DIR / "cdi_input"))
    OUTPUT_DIR  = Path(os.environ.get("CDI_OUTPUT_DIR",  BASE_DIR / "cdi_output"))
    MONTHLY_DIR = Path(os.environ.get("CDI_MONTHLY_DIR", BASE_DIR / "cdi_monthly"))
    DEKADAL_DIR = Path(os.environ.get("CDI_DEKADAL_DIR", BASE_DIR / "cdi_dekadal"))

    COUNTRIES_SHP = INPUT_DIR / "icpac_countries_merged.shp"
    ADMIN1_SHP    = INPUT_DIR / "Administrative1_Boundaries_ICPAC_Countries.shp"
    ADMIN2_SHP    = INPUT_DIR / "Administrative2_Boundaries_ICPAC_Countries.shp"
    LAKES_SHP     = INPUT_DIR / "ne_10m_lakes" / "ne_10m_lakes.shp"
    LOGO          = BASE_DIR / "assets" / "ICPAC_LOGO_WithSLogan.width-800 (1) (1).png"

    # FTP base URLs
    MONTHLY_BASE_URL = os.environ.get(
        "MONTHLY_BASE_URL",
        "https://droughtwatch.icpac.net/ftp/monthly/geotif/",
    )
    DEKADAL_BASE_URL = os.environ.get(
        "DEKADAL_BASE_URL",
        "https://droughtwatch.icpac.net/ftp/dekadal/geotif/",
    )

    @classmethod
    def monthly_raster(cls, year: int, month_abbr: str) -> Path:
        """Return the path to a monthly CDI raster file."""
        return cls.MONTHLY_DIR / str(year) / f"eadw-cdi-data-{year}-{month_abbr}.tif"

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        for d in [cls.OUTPUT_DIR, cls.MONTHLY_DIR, cls.DEKADAL_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# CDI DROUGHT CLASSIFICATION — single source of truth
# ──────────────────────────────────────────────────────────────────────────────
class CDI:
    # Value → category name  (raw CDI integer scale 0-10)
    CATEGORY_MAP = {
        0:  "No Drought",
        1:  "Watch",   2: "Watch",   3: "Watch",
        4:  "Warning", 5: "Warning", 6: "Warning",
        7:  "Alert",   8: "Alert",   9: "Alert",  10: "Alert",
    }

    # Category → hex color
    CATEGORY_COLORS = {
        "No Drought": "#FFFFFF",
        "Watch":      "#FFD700",
        "Warning":    "#FFA500",
        "Alert":      "#FF0000",
        "No Data":    "#C1C1C1",
    }

    # Detailed per-value colors (used by the pipeline for fine-grained maps)
    VALUE_COLORS = {
        1: "#FFFF99", 2: "#FFFF66", 3: "#FFEE33",             # Watch
        4: "#FFCC66", 5: "#FF9933", 6: "#FF6600",             # Warning
        7: "#FF3333", 8: "#CC0000", 9: "#990000", 10: "#660000",  # Alert
    }

    # Boundary / norm arrays for matplotlib BoundaryNorm
    BOUNDARIES  = [0, 1, 4, 7, 11]
    COLORS_4    = ["#FFFFFF", "#FFFF00", "#FFA500", "#FF0000"]
    LABELS      = ["No Drought", "Watch", "Warning", "Alert"]

    # Smoothing-specific 4-class colormap (white → yellow → orange → red)
    SMOOTH_COLORS = ["#ffffff", "#ffff00", "#ff9900", "#cc0000"]

    # Severity weights for spatial scoring
    SEVERITY_WEIGHTS = {"Alert": 3, "Warning": 2, "Watch": 1}

    # ICPAC region center (lat, lon)
    ICPAC_CENTER = (5.5, 36.5)


# ──────────────────────────────────────────────────────────────────────────────
# MONTH UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
MONTH_ABBR_TO_NAME = {
    "jan": "January",  "feb": "February", "mar": "March",
    "apr": "April",    "may": "May",      "jun": "June",
    "jul": "July",     "aug": "August",   "sep": "September",
    "oct": "October",  "nov": "November", "dec": "December",
}

MONTH_ORDER = {abbr: i for i, abbr in enumerate(MONTH_ABBR_TO_NAME, start=1)}


# ──────────────────────────────────────────────────────────────────────────────
# SHARED DATA LOADERS (cached at module level)
# ──────────────────────────────────────────────────────────────────────────────
_boundaries_cache = {}
_lakes_cache = None


def load_boundaries(level: str = "countries") -> gpd.GeoDataFrame:
    """Load ICPAC boundary shapefile. level: 'countries', 'admin1', 'admin2'."""
    if level in _boundaries_cache:
        return _boundaries_cache[level]
    path_map = {
        "countries": PATHS.COUNTRIES_SHP,
        "admin1":    PATHS.ADMIN1_SHP,
        "admin2":    PATHS.ADMIN2_SHP,
    }
    gdf = gpd.read_file(path_map[level])
    _boundaries_cache[level] = gdf
    return gdf


def load_lakes(clip_bounds=None, smooth=True, buffer_deg=0.02) -> gpd.GeoDataFrame:
    """
    Load Natural Earth 10m lakes, optionally clipped to bounds and smoothed.

    Parameters
    ----------
    clip_bounds : tuple or None
        (left, bottom, right, top) to clip lakes. None returns full dataset.
    smooth : bool
        Buffer + simplify to produce smooth polygon edges.
    buffer_deg : float
        Degree buffer for smoothing (only used if smooth=True).
    """
    global _lakes_cache
    if _lakes_cache is None:
        _lakes_cache = gpd.read_file(PATHS.LAKES_SHP)

    result = _lakes_cache
    if clip_bounds is not None:
        from shapely.geometry import box
        roi = box(*clip_bounds)
        result = result.clip(roi)

    if smooth and not result.empty:
        result = result.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result["geometry"] = (
                result.geometry
                .buffer(buffer_deg)
                .buffer(-buffer_deg * 0.5)
                .simplify(tolerance=0.01)
            )
    return result
