#!/usr/bin/env python3
"""
cdi_pipeline.py
---------------
Comprehensive CDI (Combined Drought Index) Analysis Pipeline for ICPAC Region

This script automatically:
1. Detects the latest available year from ICPAC FTP server
2. Downloads monthly and dekadal CDI GeoTIFF data
3. Merges country shapefiles for the 11 ICPAC countries
4. Clips, classifies, and visualizes drought data
5. Generates panel visualizations

Author: Mark Lelaono
Date: 2025
"""

import os
import sys
import glob
import calendar
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import geopandas as gpd
import pandas as pd  # Only used for shapefile merging
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# === CONFIGURATION ===
class Config:
    """Configuration settings for the CDI pipeline."""

    # FTP Base URLs
    MONTHLY_BASE_URL = "https://droughtwatch.icpac.net/ftp/monthly/geotif/"
    DEKADAL_BASE_URL = "https://droughtwatch.icpac.net/ftp/dekadal/geotif/"

    # Default paths (can be overridden via arguments)
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "cdi_input"
    OUTPUT_DIR = BASE_DIR / "cdi_output"
    MONTHLY_DIR = BASE_DIR / "cdi_monthly"
    DEKADAL_DIR = BASE_DIR / "cdi_dekadal"

    # Shapefile settings
    SHAPEFILE_NAME = "icpac_countries_merged.shp"
    WATERBODY_SHAPEFILE = "water_bodies.shp"
    LOGO_PATH = BASE_DIR / "assets" / "ICPAC_LOGO.png"

    # CDI Color scheme
    CDI_COLORS = {
        1: "#FFFF99", 2: "#FFFF66", 3: "#FFEE33",     # Watch (Yellow)
        4: "#FFCC66", 5: "#FF9933", 6: "#FF6600",     # Warning (Orange)
        7: "#FF3333", 8: "#CC0000", 9: "#990000", 10: "#660000"  # Alert (Red)
    }

    # Drought categories
    DROUGHT_BOUNDARIES = [0, 1, 4, 7, 11]
    DROUGHT_COLORS = ['#FFFFFF', '#FFFF00', '#FFA500', '#FF0000']
    DROUGHT_LABELS = ['No Drought', 'Watch', 'Warning', 'Alert']


class CDIPipeline:
    """Main CDI analysis pipeline class."""

    def __init__(self, config: Config = None):
        """Initialize the pipeline with configuration."""
        self.config = config or Config()
        self._setup_directories()
        self.region_gdf = None
        self.waterbodies_gdf = None

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [
            self.config.INPUT_DIR,
            self.config.OUTPUT_DIR,
            self.config.MONTHLY_DIR,
            self.config.DEKADAL_DIR,
            self.config.BASE_DIR / "assets"
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")

    # =========================================================================
    # FTP UTILITIES
    # =========================================================================

    def get_available_years(self, base_url: str) -> List[int]:
        """Fetch available years from the FTP server."""
        try:
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            years = []
            for link in soup.find_all("a"):
                href = link.get("href", "").strip("/")
                if href.isdigit() and len(href) == 4:
                    years.append(int(href))

            years.sort()
            logger.info(f"Available years: {years}")
            return years

        except Exception as e:
            logger.error(f"Error fetching years from {base_url}: {e}")
            return []

    def get_latest_year(self, data_type: str = "monthly") -> int:
        """Get the latest available year from the FTP server."""
        base_url = (
            self.config.MONTHLY_BASE_URL
            if data_type == "monthly"
            else self.config.DEKADAL_BASE_URL
        )
        years = self.get_available_years(base_url)
        if years:
            latest = max(years)
            logger.info(f"Latest {data_type} year detected: {latest}")
            return latest
        return datetime.now().year

    def get_tif_links(self, year_url: str) -> List[str]:
        """Scrape .tif links for a given year folder."""
        try:
            response = requests.get(year_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            links = [
                a["href"] for a in soup.find_all("a")
                if a.get("href", "").lower().endswith(".tif")
            ]
            return links

        except Exception as e:
            logger.error(f"Error accessing {year_url}: {e}")
            return []

    def download_file(self, url: str, save_path: Path) -> bool:
        """Download a file with progress bar."""
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))

                with open(save_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=save_path.name,
                    ncols=80
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    # =========================================================================
    # DATA DOWNLOAD
    # =========================================================================

    def download_monthly_data(
        self,
        years: List[int] = None,
        force_download: bool = False
    ) -> Dict[int, List[Path]]:
        """Download monthly CDI data for specified years."""
        if years is None:
            latest_year = self.get_latest_year("monthly")
            years = [latest_year]

        downloaded_files = {}

        for year in years:
            year_url = f"{self.config.MONTHLY_BASE_URL}{year}/"
            logger.info(f"Processing monthly data for {year}: {year_url}")

            tif_links = self.get_tif_links(year_url)
            if not tif_links:
                logger.warning(f"No .tif files found for {year}")
                continue

            year_dir = self.config.MONTHLY_DIR / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            downloaded_files[year] = []

            for tif_name in tif_links:
                file_url = f"{year_url}{tif_name}"
                save_path = year_dir / tif_name

                if save_path.exists() and not force_download:
                    logger.info(f"Skipping existing file: {tif_name}")
                    downloaded_files[year].append(save_path)
                    continue

                logger.info(f"Downloading {tif_name}...")
                if self.download_file(file_url, save_path):
                    downloaded_files[year].append(save_path)

        return downloaded_files

    def download_dekadal_data(
        self,
        years: List[int] = None,
        force_download: bool = False
    ) -> Dict[int, List[Path]]:
        """Download dekadal CDI data for specified years."""
        if years is None:
            latest_year = self.get_latest_year("dekadal")
            years = [latest_year]

        downloaded_files = {}

        for year in years:
            year_url = f"{self.config.DEKADAL_BASE_URL}{year}/"
            logger.info(f"Processing dekadal data for {year}: {year_url}")

            tif_links = self.get_tif_links(year_url)
            if not tif_links:
                logger.warning(f"No .tif files found for {year}")
                continue

            year_dir = self.config.DEKADAL_DIR / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            downloaded_files[year] = []

            for tif_name in tif_links:
                file_url = f"{year_url}{tif_name}"
                save_path = year_dir / tif_name

                if save_path.exists() and not force_download:
                    logger.info(f"Skipping existing file: {tif_name}")
                    downloaded_files[year].append(save_path)
                    continue

                logger.info(f"Downloading {tif_name}...")
                if self.download_file(file_url, save_path):
                    downloaded_files[year].append(save_path)

        return downloaded_files

    # =========================================================================
    # SHAPEFILE UTILITIES
    # =========================================================================

    def merge_shapefiles(self, input_folder: Path, output_path: Path = None) -> gpd.GeoDataFrame:
        """Merge all shapefiles in a folder into one GeoDataFrame."""
        if output_path is None:
            output_path = self.config.INPUT_DIR / self.config.SHAPEFILE_NAME

        shapefiles = list(input_folder.glob("*.shp"))

        if not shapefiles:
            raise FileNotFoundError(f"No .shp files found in {input_folder}")

        logger.info(f"Found {len(shapefiles)} shapefiles to merge")

        gdfs = []
        for shp in shapefiles:
            logger.info(f"Loading {shp.name}...")
            gdf = gpd.read_file(shp)
            gdfs.append(gdf)

        merged = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            crs=gdfs[0].crs
        )

        merged.to_file(output_path)
        logger.info(f"Merged shapefile saved: {output_path}")

        return merged

    def load_region_shapefile(self, shapefile_path: Path = None) -> gpd.GeoDataFrame:
        """Load the ICPAC region shapefile."""
        if shapefile_path is None:
            shapefile_path = self.config.INPUT_DIR / self.config.SHAPEFILE_NAME

        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

        self.region_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        logger.info(f"Loaded region shapefile: {shapefile_path}")

        return self.region_gdf

    def load_waterbodies(self, shapefile_path: Path = None) -> Optional[gpd.GeoDataFrame]:
        """Load waterbodies shapefile if available."""
        if shapefile_path is None:
            shapefile_path = self.config.INPUT_DIR / self.config.WATERBODY_SHAPEFILE

        if shapefile_path.exists():
            self.waterbodies_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
            logger.info(f"Loaded waterbodies shapefile: {shapefile_path}")
            return self.waterbodies_gdf

        logger.warning(f"Waterbodies shapefile not found: {shapefile_path}")
        return None

    # =========================================================================
    # CDI PROCESSING UTILITIES
    # =========================================================================

    @staticmethod
    def classify_cdi(data: np.ndarray, colors: Dict[int, str]) -> np.ndarray:
        """Classify CDI values into drought categories."""
        classified = np.full_like(data, np.nan, dtype=float)
        for val in colors.keys():
            classified[data == val] = val
        return classified

    @staticmethod
    def parse_monthly_date(filename: str) -> Optional[datetime]:
        """Parse date from monthly CDI filename."""
        name = Path(filename).stem

        # Format: eadw-cdi-data-2025-jan
        patterns = [
            (r"eadw-cdi-data-(\d{4})-(\w{3})", "%Y-%b"),
            (r"(\d{4})-(\w{3})", "%Y-%b"),
        ]

        import re
        for pattern, date_format in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                try:
                    date_str = f"{match.group(1)}-{match.group(2)}"
                    return datetime.strptime(date_str, date_format)
                except ValueError:
                    continue

        return None

    @staticmethod
    def parse_dekadal_date(filename: str) -> Optional[datetime]:
        """Parse date from dekadal CDI filename."""
        name = Path(filename).stem

        # Format: eadw-cdi-data-2025-01-01
        import re
        match = re.search(r"eadw-cdi-data-(\d{4})-(\d{2})-(\d{2})", name)
        if match:
            try:
                return datetime.strptime(
                    f"{match.group(1)}-{match.group(2)}-{match.group(3)}",
                    "%Y-%m-%d"
                )
            except ValueError:
                pass

        return None

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def create_cdi_map(
        self,
        data: np.ndarray,
        transform,
        date: datetime,
        output_path: Path,
        title_prefix: str = "Combined Drought Index (CDI)",
        show_logo: bool = True,
        show_disclaimer: bool = True
    ) -> None:
        """Create a styled CDI map with legend, labels, and optional logo."""

        # Classify CDI values
        classified = self.classify_cdi(data, self.config.CDI_COLORS)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Define colormap
        cmap = LinearSegmentedColormap.from_list(
            'cdi_drought',
            self.config.DROUGHT_COLORS,
            N=len(self.config.DROUGHT_COLORS)
        )
        norm = BoundaryNorm(self.config.DROUGHT_BOUNDARIES, cmap.N, clip=True)

        # Plot classified raster
        im = show(classified, transform=transform, ax=ax, cmap=cmap, norm=norm)

        # Plot region boundaries
        if self.region_gdf is not None:
            self.region_gdf.boundary.plot(ax=ax, color="black", linewidth=0.6, alpha=0.5)

        # Plot waterbodies
        if self.waterbodies_gdf is not None:
            self.waterbodies_gdf.plot(ax=ax, color="#90D5FF", alpha=0.5)

        # Add country labels
        if self.region_gdf is not None:
            self.region_gdf["centroid"] = self.region_gdf.geometry.centroid
            for _, row in self.region_gdf.iterrows():
                country_name = row.get("COUNTRY", row.get("NAME_0", ""))
                if country_name:
                    x, y = row.centroid.x, row.centroid.y

                    # Adjust Somalia label
                    if country_name.lower().startswith("som"):
                        x += 2.5
                        y += 0.5

                    ax.text(
                        x, y, country_name,
                        fontsize=7, color="black",
                        ha="center", va="center",
                        alpha=0.8, weight="normal"
                    )

        # Add gridlines
        ax.set_xticks(np.arange(20, 55, 5))
        ax.set_yticks(np.arange(-12, 25, 5))
        ax.set_xticklabels([f"{x}°E" for x in np.arange(20, 55, 5)], fontsize=8)
        ax.set_yticklabels([f"{y}°N" for y in np.arange(-12, 25, 5)], fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(
            im.get_images()[0],
            ax=ax,
            fraction=0.046, pad=0.04,
            boundaries=self.config.DROUGHT_BOUNDARIES,
            ticks=[0.5, 2, 5, 8.5]
        )
        cbar.ax.set_yticklabels(self.config.DROUGHT_LABELS, fontsize=8)

        # Title
        ax.set_title(
            f"{title_prefix} – {date.strftime('%B %Y')}",
            fontsize=13, fontweight='bold', pad=15
        )

        # Add frame
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        frame = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, color='black', linewidth=1.2,
            zorder=10, transform=ax.transData
        )
        ax.add_patch(frame)

        # Add ICPAC logo
        if show_logo and self.config.LOGO_PATH.exists():
            try:
                logo = Image.open(self.config.LOGO_PATH)
                logo.thumbnail((90, 90))
                imagebox = OffsetImage(logo, zoom=0.35)
                ab = AnnotationBbox(
                    imagebox, (xmax - 1.5, ymax - 1.5),
                    frameon=False, box_alignment=(1, 1),
                    xycoords='data'
                )
                ax.add_artist(ab)
            except Exception as e:
                logger.warning(f"Could not add logo: {e}")

        # Add disclaimer
        if show_disclaimer:
            ax.text(
                (xmin + xmax) / 2, ymin + 0.4,
                "Disclaimer: The country boundaries are not endorsed by ICPAC.",
                ha="center", va="bottom",
                fontsize=7, color="black", style="italic",
                transform=ax.transData
            )

        # Save
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Map saved: {output_path}")

    def create_dekadal_map(
        self,
        data: np.ndarray,
        transform,
        date: datetime,
        output_path: Path
    ) -> None:
        """Create a dekadal CDI map."""
        self.create_cdi_map(
            data, transform, date, output_path,
            title_prefix=f"Combined Drought Index (CDI) – Dekad {date.day}"
        )

    # =========================================================================
    # MONTHLY ANALYSIS
    # =========================================================================

    def process_monthly_data(
        self,
        years: List[int] = None,
        download: bool = True
    ) -> int:
        """Process monthly CDI data for specified years."""

        # Ensure region shapefile is loaded
        if self.region_gdf is None:
            self.load_region_shapefile()

        # Load waterbodies if available
        self.load_waterbodies()

        # Download data if requested
        if download:
            if years is None:
                years = [self.get_latest_year("monthly")]
            self.download_monthly_data(years)

        # Collect all TIF files
        tif_files = []
        for year_dir in sorted(self.config.MONTHLY_DIR.iterdir()):
            if year_dir.is_dir():
                tif_files.extend(sorted(year_dir.glob("*.tif")))

        # Filter and sort by date
        tif_files = [f for f in tif_files if self.parse_monthly_date(f.name)]
        tif_files.sort(key=lambda x: self.parse_monthly_date(x.name))

        if not tif_files:
            logger.warning("No monthly TIF files found to process")
            return 0

        logger.info(f"Processing {len(tif_files)} monthly CDI files...")

        output_dir = self.config.OUTPUT_DIR / "monthly_maps"
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0

        for tif_path in tqdm(tif_files, desc="Processing monthly CDI"):
            date = self.parse_monthly_date(tif_path.name)
            if not date:
                continue

            try:
                with rasterio.open(tif_path) as src:
                    # Clip to region
                    clipped, transform = mask(src, self.region_gdf.geometry, crop=True)
                    data = clipped[0]
                    data = np.where(data == src.nodata, np.nan, data)

                    # Create map
                    output_path = output_dir / f"CDI_Monthly_{date.strftime('%Y_%m')}.png"
                    self.create_cdi_map(data, transform, date, output_path)
                    processed_count += 1

            except Exception as e:
                logger.error(f"Error processing {tif_path}: {e}")

        logger.info(f"Processed {processed_count} monthly CDI maps")
        return processed_count

    # =========================================================================
    # DEKADAL ANALYSIS
    # =========================================================================

    def process_dekadal_data(
        self,
        years: List[int] = None,
        download: bool = True
    ) -> int:
        """Process dekadal CDI data for specified years."""

        # Ensure region shapefile is loaded
        if self.region_gdf is None:
            self.load_region_shapefile()

        # Load waterbodies if available
        self.load_waterbodies()

        # Download data if requested
        if download:
            if years is None:
                years = [self.get_latest_year("dekadal")]
            self.download_dekadal_data(years)

        # Collect all TIF files
        tif_files = []
        for year_dir in sorted(self.config.DEKADAL_DIR.iterdir()):
            if year_dir.is_dir():
                tif_files.extend(sorted(year_dir.glob("*.tif")))

        # Filter and sort by date
        tif_files = [f for f in tif_files if self.parse_dekadal_date(f.name)]
        tif_files.sort(key=lambda x: self.parse_dekadal_date(x.name))

        if not tif_files:
            logger.warning("No dekadal TIF files found to process")
            return 0

        logger.info(f"Processing {len(tif_files)} dekadal CDI files...")

        output_dir = self.config.OUTPUT_DIR / "dekadal_maps"
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0

        for tif_path in tqdm(tif_files, desc="Processing dekadal CDI"):
            date = self.parse_dekadal_date(tif_path.name)
            if not date:
                continue

            try:
                with rasterio.open(tif_path) as src:
                    # Clip to region
                    clipped, transform = mask(src, self.region_gdf.geometry, crop=True)
                    data = clipped[0]
                    data = np.where(data == src.nodata, np.nan, data)

                    # Create map
                    output_path = output_dir / f"CDI_Dekadal_{date.strftime('%Y_%m_%d')}.png"
                    self.create_dekadal_map(data, transform, date, output_path)
                    processed_count += 1

            except Exception as e:
                logger.error(f"Error processing {tif_path}: {e}")

        logger.info(f"Processed {processed_count} dekadal CDI maps")
        return processed_count

    # =========================================================================
    # PANEL VISUALIZATIONS
    # =========================================================================

    def create_monthly_panel(
        self,
        years: List[int] = None,
        output_path: Path = None
    ) -> None:
        """Create a panel visualization of monthly CDI maps (rows=years, cols=months)."""

        if output_path is None:
            output_path = self.config.OUTPUT_DIR / "CDI_Monthly_Panel.png"

        map_dir = self.config.OUTPUT_DIR / "monthly_maps"

        if years is None:
            # Auto-detect years from available maps
            years = sorted(set(
                int(f.stem.split('_')[2])
                for f in map_dir.glob("CDI_Monthly_*.png")
                if f.stem.split('_')[2].isdigit()
            ))

        if not years:
            logger.warning("No monthly maps found for panel creation")
            return

        months = range(1, 13)

        # Collect images
        images = []
        for year in years:
            row_imgs = []
            for month in months:
                img_path = map_dir / f"CDI_Monthly_{year}_{month:02d}.png"
                if img_path.exists():
                    img = mpimg.imread(img_path)
                    row_imgs.append(img)
                else:
                    # Create blank placeholder
                    blank = np.ones((200, 200, 3))
                    row_imgs.append(blank)
                    logger.debug(f"Missing: {img_path}")
            images.append(row_imgs)

        # Create grid panel
        nrows = len(years)
        ncols = len(months)

        fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.5))
        gs = GridSpec(nrows, ncols, figure=fig, wspace=0.03, hspace=0.03)

        for i, year in enumerate(years):
            for j, month in enumerate(months):
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(images[i][j])
                ax.axis("off")

                # Box border
                rect = Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    linewidth=0.8,
                    edgecolor="black",
                    facecolor="none"
                )
                ax.add_patch(rect)

                # Column headers (months)
                if i == 0:
                    ax.set_title(calendar.month_abbr[month], fontsize=9, pad=6)

                # Row labels (years)
                if j == 0:
                    ax.text(
                        -0.15, 0.5, str(year),
                        fontsize=10, fontweight="bold",
                        va="center", ha="right", rotation=90,
                        transform=ax.transAxes
                    )

        fig.suptitle(
            "Combined Drought Index (CDI) – Monthly Overview",
            fontsize=14, fontweight='bold', y=1.02
        )

        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Monthly panel saved: {output_path}")

    def create_dekadal_panel(
        self,
        year: int = None,
        output_path: Path = None
    ) -> None:
        """Create a panel visualization of dekadal CDI maps for a single year."""

        if year is None:
            year = self.get_latest_year("dekadal")

        if output_path is None:
            output_path = self.config.OUTPUT_DIR / f"CDI_Dekadal_Panel_{year}.png"

        map_dir = self.config.OUTPUT_DIR / "dekadal_maps"

        # 36 dekads per year (3 per month × 12 months)
        months = range(1, 13)
        dekads = [1, 11, 21]

        # Collect images
        images = []
        for month in months:
            row_imgs = []
            for day in dekads:
                img_path = map_dir / f"CDI_Dekadal_{year}_{month:02d}_{day:02d}.png"
                if img_path.exists():
                    img = mpimg.imread(img_path)
                    row_imgs.append(img)
                else:
                    blank = np.ones((200, 200, 3))
                    row_imgs.append(blank)
            images.append(row_imgs)

        # Create grid: 12 rows (months) × 3 columns (dekads)
        nrows = 12
        ncols = 3

        fig = plt.figure(figsize=(ncols * 3, nrows * 2))
        gs = GridSpec(nrows, ncols, figure=fig, wspace=0.03, hspace=0.03)

        for i, month in enumerate(months):
            for j, day in enumerate(dekads):
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(images[i][j])
                ax.axis("off")

                rect = Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    linewidth=0.8,
                    edgecolor="black",
                    facecolor="none"
                )
                ax.add_patch(rect)

                # Column headers (dekads)
                if i == 0:
                    ax.set_title(f"Dekad {j+1}", fontsize=9, pad=6)

                # Row labels (months)
                if j == 0:
                    ax.text(
                        -0.1, 0.5, calendar.month_abbr[month],
                        fontsize=9, fontweight="bold",
                        va="center", ha="right",
                        transform=ax.transAxes
                    )

        fig.suptitle(
            f"Combined Drought Index (CDI) – Dekadal Overview {year}",
            fontsize=14, fontweight='bold', y=1.01
        )

        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Dekadal panel saved: {output_path}")

    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================

    def run(
        self,
        analysis_type: str = "both",
        years: List[int] = None,
        download: bool = True,
        create_panels: bool = True
    ) -> Dict[str, int]:
        """
        Run the complete CDI analysis pipeline.

        Parameters:
        -----------
        analysis_type : str
            Type of analysis: "monthly", "dekadal", or "both"
        years : list
            List of years to process. If None, uses latest year
        download : bool
            Whether to download data from FTP
        create_panels : bool
            Whether to create panel visualizations

        Returns:
        --------
        dict : Dictionary with processed counts for each analysis type
        """

        results = {}

        logger.info("=" * 60)
        logger.info("Starting CDI Analysis Pipeline")
        logger.info("=" * 60)

        # Detect latest year if not specified
        if years is None:
            if analysis_type in ["monthly", "both"]:
                monthly_year = self.get_latest_year("monthly")
                logger.info(f"Latest monthly year: {monthly_year}")
            if analysis_type in ["dekadal", "both"]:
                dekadal_year = self.get_latest_year("dekadal")
                logger.info(f"Latest dekadal year: {dekadal_year}")

        # Load shapefiles
        try:
            self.load_region_shapefile()
        except FileNotFoundError:
            logger.error("Region shapefile not found. Please ensure it exists in cdi_input/")
            return results

        self.load_waterbodies()

        # Process monthly data
        if analysis_type in ["monthly", "both"]:
            logger.info("-" * 40)
            logger.info("Processing MONTHLY CDI data")
            logger.info("-" * 40)

            monthly_years = years or [self.get_latest_year("monthly")]
            results["monthly"] = self.process_monthly_data(monthly_years, download)

            if create_panels and results["monthly"] > 0:
                self.create_monthly_panel(monthly_years)

        # Process dekadal data
        if analysis_type in ["dekadal", "both"]:
            logger.info("-" * 40)
            logger.info("Processing DEKADAL CDI data")
            logger.info("-" * 40)

            dekadal_years = years or [self.get_latest_year("dekadal")]
            results["dekadal"] = self.process_dekadal_data(dekadal_years, download)

            if create_panels and results["dekadal"] > 0:
                for year in dekadal_years:
                    self.create_dekadal_panel(year)

        logger.info("=" * 60)
        logger.info("CDI Analysis Pipeline Complete!")
        logger.info(f"Output directory: {self.config.OUTPUT_DIR}")
        logger.info("=" * 60)

        return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command line usage."""

    parser = argparse.ArgumentParser(
        description="CDI Analysis Pipeline for ICPAC Region",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis for latest year (auto-detected)
  python cdi_pipeline.py

  # Run only monthly analysis
  python cdi_pipeline.py --type monthly

  # Run dekadal analysis for specific years
  python cdi_pipeline.py --type dekadal --years 2023 2024

  # Skip download (use existing data)
  python cdi_pipeline.py --no-download

  # Process all available years
  python cdi_pipeline.py --all-years
        """
    )

    parser.add_argument(
        "--type", "-t",
        choices=["monthly", "dekadal", "both"],
        default="both",
        help="Type of analysis to run (default: both)"
    )

    parser.add_argument(
        "--years", "-y",
        type=int,
        nargs="+",
        help="Specific years to process (default: latest year)"
    )

    parser.add_argument(
        "--all-years", "-a",
        action="store_true",
        help="Process all available years from FTP"
    )

    parser.add_argument(
        "--no-download", "-n",
        action="store_true",
        help="Skip downloading data (use existing local files)"
    )

    parser.add_argument(
        "--no-panels", "-p",
        action="store_true",
        help="Skip creating panel visualizations"
    )

    parser.add_argument(
        "--merge-shapefiles", "-m",
        type=str,
        metavar="FOLDER",
        help="Merge shapefiles from specified folder before processing"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Custom output directory"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize pipeline
    config = Config()

    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)

    pipeline = CDIPipeline(config)

    # Merge shapefiles if requested
    if args.merge_shapefiles:
        input_folder = Path(args.merge_shapefiles)
        if input_folder.exists():
            pipeline.merge_shapefiles(input_folder)
        else:
            logger.error(f"Shapefile folder not found: {input_folder}")
            sys.exit(1)

    # Determine years to process
    years = args.years
    if args.all_years:
        if args.type in ["monthly", "both"]:
            years = pipeline.get_available_years(config.MONTHLY_BASE_URL)
        elif args.type == "dekadal":
            years = pipeline.get_available_years(config.DEKADAL_BASE_URL)

    # Run pipeline
    try:
        results = pipeline.run(
            analysis_type=args.type,
            years=years,
            download=not args.no_download,
            create_panels=not args.no_panels
        )

        # Print summary
        for data_type, count in results.items():
            if count > 0:
                print(f"\n{data_type.upper()} Summary:")
                print(f"  Maps processed: {count}")

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
