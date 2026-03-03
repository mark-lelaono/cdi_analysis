# CDI Analysis Pipeline for ICPAC Region

A comprehensive Python toolkit for downloading, analysing, smoothing, and visualising Combined Drought Index (CDI) data across the 11 ICPAC countries in East Africa.

## Features

- **Auto-detection** of latest available year from the ICPAC FTP server
- **Automated download** of monthly and dekadal CDI GeoTIFF data
- **Shapefile merging** for ICPAC region countries
- **Drought classification** and color-coded map generation
- **Raster smoothing** — Gaussian, mean, and median filters with lake masking
- **Panel visualisations** — multi-month side-by-side comparison (original vs smoothed)
- **H3 hexagon maps** — interactive Folium-based hex grids at multiple resolutions
- **Animated GIFs** — honeycomb zoom-in storytelling GIFs per month

## ICPAC Countries Covered

Burundi, Djibouti, Eritrea, Ethiopia, Kenya, Rwanda, Somalia, South Sudan, Sudan, Tanzania, Uganda

## Project Structure

```
cdi_analysis/
├── cdi_config.py           # Shared configuration (paths, colors, constants)
├── cdi_pipeline.py         # Main pipeline: download, classify, visualise
├── cdi_smoothing.py        # Raster smoothing (Gaussian, mean, median)
├── cdi_smoothing_panel.py  # Multi-month comparison panel generator
├── cdi_hexmap.py           # H3 hexagonal grid engine (interactive maps)
├── make_gif.py             # Animated honeycomb GIF generator
├── download_and_gif.py     # Download helper + GIF batch runner
│
├── cdi_analysis.ipynb      # Interactive analysis notebook
├── hexagon.ipynb           # H3 hexagon experimentation notebook
│
├── cdi_input/              # Input shapefiles
│   ├── icpac_countries_merged.shp
│   ├── Administrative1_Boundaries_ICPAC_Countries.shp
│   ├── Administrative2_Boundaries_ICPAC_Countries.shp
│   └── ne_10m_lakes/       # Natural Earth 10m lakes
│
├── cdi_monthly/            # Downloaded monthly CDI GeoTIFFs
│   └── {YYYY}/
├── cdi_dekadal/            # Downloaded dekadal CDI GeoTIFFs
│   └── {YYYY}/
│
├── cdi_output/             # Generated outputs
│   ├── monthly_maps/       # Individual monthly CDI PNGs
│   ├── dekadal_maps/       # Individual dekadal CDI PNGs
│   ├── hexmaps/            # Interactive HTML hex maps & GIFs
│   ├── gif_frames/         # Intermediate GIF frame PNGs
│   ├── cdi_gaussian*.tif   # Smoothed rasters
│   └── CDI_*.png           # Panel visualisations
│
├── assets/                 # Logo and branding
│   └── ICPAC_LOGO.png
├── .env.example            # Environment variable template
├── .env                    # Local environment overrides (not tracked)
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd cdi_analysis
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install geopandas pandas numpy rasterio scipy matplotlib pillow \
    beautifulsoup4 tqdm requests h3 folium shapely python-dotenv
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env only if you need to override default paths
```

## Scripts

### `cdi_config.py` — Shared Configuration

Central module imported by all other scripts. Defines:
- `PATHS` — directory paths, shapefile locations, FTP URLs
- `CDI` — drought classification colors, category maps, severity weights
- `MONTH_ABBR_TO_NAME` / `MONTH_ORDER` — month utilities
- `load_boundaries()` / `load_lakes()` — cached data loaders

### `cdi_pipeline.py` — Main Pipeline

Downloads, clips, classifies, and visualises CDI data.

```bash
# Full analysis for latest year (auto-detected)
python cdi_pipeline.py

# Monthly analysis only
python cdi_pipeline.py --type monthly

# Specific years
python cdi_pipeline.py --years 2023 2024 2025

# All available years (2010-present)
python cdi_pipeline.py --all-years

# Skip download / skip panels
python cdi_pipeline.py --no-download
python cdi_pipeline.py --no-panels

# Merge shapefiles from a folder
python cdi_pipeline.py --merge-shapefiles /path/to/admin/shapefiles

# Verbose logging
python cdi_pipeline.py --verbose
```

### `cdi_smoothing.py` — Raster Smoothing

Applies Gaussian, mean, and median filters to a single CDI raster with:
- Lake masking (Natural Earth 10m, buffered for smooth boundaries)
- Normalized convolution (clean edges, no NaN bleeding)
- Classified polygon extraction (Watch / Warning / Alert zones)
- Publication-quality map with country boundaries

```bash
python cdi_smoothing.py
```

**Configurable parameters** (edit at the top of the script):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gaussian_sigma` | 5 | Gaussian smoothing strength |
| `window_size` | 5 | Moving window size for mean/median |
| `resample_scale` | 2 | Resolution coarsening factor |

### `cdi_smoothing_panel.py` — Multi-Month Comparison Panel

Processes Aug–Dec 2025 and generates a side-by-side panel:
- Left column: original pixelated CDI
- Right column: Gaussian smoothed CDI
- Country boundaries, smoothed lakes, and ICPAC logo on each map

```bash
python cdi_smoothing_panel.py
```

### `cdi_hexmap.py` — H3 Hexagonal Map Engine

Converts raster CDI to Uber's H3 hexagonal grid for interactive Folium maps.

```bash
python cdi_hexmap.py -i cdi_monthly/2025/eadw-cdi-data-2025-aug.tif -o output.html
python cdi_hexmap.py -i input.tif --coarse 6 --by-category
```

### `make_gif.py` — Animated GIF Generator

Creates honeycomb zoom-in GIFs (coarse H3 → fine H3) with country/region labels.

```bash
python make_gif.py -i cdi_monthly/2025/eadw-cdi-data-2025-aug.tif
```

### `download_and_gif.py` — Download + GIF Batch Runner

Downloads monthly rasters and generates GIFs in one step.

```bash
python download_and_gif.py                    # download + GIFs
python download_and_gif.py --skip-download    # GIFs only
python download_and_gif.py --force            # re-download
python download_and_gif.py --months aug sep   # specific months
```

## Data Sources

### Monthly CDI
- **URL**: `https://droughtwatch.icpac.net/ftp/monthly/geotif/`
- **Format**: `eadw-cdi-data-YYYY-MMM.tif`
- **Frequency**: 12 files per year (one per month)

### Dekadal CDI
- **URL**: `https://droughtwatch.icpac.net/ftp/dekadal/geotif/`
- **Format**: `eadw-cdi-data-YYYY-MM-DD.tif`
- **Frequency**: 36 files per year (1st, 11th, 21st of each month)

### Shapefiles
- **ICPAC Countries**: Merged boundaries for 11 East African countries
- **Admin1/Admin2**: Sub-national boundaries from ICPAC
- **Lakes**: [Natural Earth 10m lakes](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-lakes/)

## Drought Classification

The CDI uses a 1–10 scale classified into four categories:

| Category   | CDI Values | Color  | Description                     |
|------------|------------|--------|---------------------------------|
| No Drought | 0          | White  | Normal conditions               |
| Watch      | 1–3        | Yellow | Drought conditions developing   |
| Warning    | 4–6        | Orange | Drought conditions worsening    |
| Alert      | 7–10       | Red    | Severe drought conditions       |

## Requirements

- Python 3.8+
- geopandas
- pandas
- numpy
- rasterio
- scipy
- matplotlib
- pillow
- beautifulsoup4
- tqdm
- requests
- h3
- folium
- shapely
- python-dotenv

## Notes

- All downloaded TIF files and generated outputs are excluded from version control via `.gitignore`
- The pipeline automatically creates necessary directories
- Existing files are skipped during download (use `--force` to re-download)
- Internet connection required for FTP data download
- All scripts import shared config from `cdi_config.py` — edit that file to change paths or colors globally

## Author

Mark Lelaono

## License

This project is for drought monitoring and analysis purposes within the ICPAC region.

## Acknowledgments

- ICPAC (IGAD Climate Prediction and Applications Centre) for providing CDI data
- Natural Earth for lakes and physical vector data
- GADM for administrative boundary data
