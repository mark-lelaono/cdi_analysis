# CDI Analysis Pipeline for ICPAC Region

A comprehensive Python pipeline for analyzing Combined Drought Index (CDI) data across the 11 ICPAC (IGAD Climate Prediction and Applications Centre) countries in East Africa.

## Features

- **Auto-detection** of latest available year from ICPAC FTP server
- **Automated download** of monthly and dekadal CDI GeoTIFF data
- **Shapefile merging** for ICPAC region countries
- **Drought classification** and visualization with color-coded maps
- **Panel visualizations** for multi-year/multi-period comparisons

## ICPAC Countries Covered

Burundi, Djibouti, Eritrea, Ethiopia, Kenya, Rwanda, Somalia, South Sudan, Sudan, Tanzania, Uganda

## Project Structure

```
cdi_analysis/
├── cdi_pipeline.py         # Main pipeline script
├── cdi_analysis.ipynb      # Jupyter notebook for interactive analysis
├── cdi_input/              # Input shapefiles
│   └── icpac_countries_merged.shp
├── cdi_output/             # Generated outputs
│   ├── monthly_maps/       # Monthly CDI map PNGs
│   ├── dekadal_maps/       # Dekadal CDI map PNGs
│   └── *.png               # Panel visualizations
├── cdi_monthly/            # Downloaded monthly TIF files
├── cdi_dekadal/            # Downloaded dekadal TIF files
├── assets/                 # Logo and other assets
│   └── ICPAC_LOGO.png
├── .venv/                  # Python virtual environment
└── README.md
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
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install dependencies

```bash
pip install geopandas pandas rasterio numpy matplotlib pillow beautifulsoup4 tqdm requests
```

## Usage

### Command Line Interface

```bash
# Activate virtual environment
source .venv/bin/activate

# Run full analysis for latest year (auto-detected from FTP)
python cdi_pipeline.py

# Run only monthly analysis
python cdi_pipeline.py --type monthly

# Run only dekadal analysis
python cdi_pipeline.py --type dekadal

# Process specific years
python cdi_pipeline.py --years 2023 2024 2025

# Process all available years (2010-present)
python cdi_pipeline.py --all-years

# Skip download (use existing local files)
python cdi_pipeline.py --no-download

# Skip panel visualizations
python cdi_pipeline.py --no-panels

# Merge shapefiles from a folder before processing
python cdi_pipeline.py --merge-shapefiles /path/to/admin/shapefiles

# Custom output directory
python cdi_pipeline.py --output-dir /path/to/output

# Enable verbose logging
python cdi_pipeline.py --verbose
```

### Python API

```python
from cdi_pipeline import CDIPipeline, Config

# Initialize pipeline
pipeline = CDIPipeline()

# Run full analysis for latest year
results = pipeline.run()

# Run specific analysis type
results = pipeline.run(analysis_type="monthly", years=[2024, 2025])

# Access results (returns count of processed maps)
monthly_count = results.get("monthly")
dekadal_count = results.get("dekadal")
```

## Data Sources

### Monthly CDI Data
- **URL**: `https://droughtwatch.icpac.net/ftp/monthly/geotif/`
- **Format**: `eadw-cdi-data-YYYY-MMM.tif` (e.g., `eadw-cdi-data-2025-jan.tif`)
- **Frequency**: 12 files per year (one per month)

### Dekadal CDI Data
- **URL**: `https://droughtwatch.icpac.net/ftp/dekadal/geotif/`
- **Format**: `eadw-cdi-data-YYYY-MM-DD.tif` (e.g., `eadw-cdi-data-2025-01-01.tif`)
- **Frequency**: 36 files per year (3 dekads per month: 1st, 11th, 21st)

## Drought Classification

The CDI uses a 1-10 scale classified into four categories:

| Category | CDI Values | Color | Description |
|----------|------------|-------|-------------|
| No Drought | 0 | White | Normal conditions |
| Watch | 1-3 | Yellow | Drought conditions developing |
| Warning | 4-6 | Orange | Drought conditions worsening |
| Alert | 7-10 | Red | Severe drought conditions |

## Output Files

### Maps
- `CDI_Monthly_YYYY_MM.png` - Individual monthly drought maps
- `CDI_Dekadal_YYYY_MM_DD.png` - Individual dekadal drought maps
- `CDI_Monthly_Panel.png` - Grid panel of all monthly maps
- `CDI_Dekadal_Panel_YYYY.png` - Grid panel of dekadal maps for a year

## Requirements

- Python 3.8+
- geopandas
- pandas
- numpy
- rasterio
- matplotlib
- pillow
- beautifulsoup4
- tqdm
- requests

## Notes

- All downloaded TIF files and generated outputs are excluded from version control via `.gitignore`
- The pipeline automatically creates necessary directories
- Existing files are skipped during download (use `--force` to re-download)
- Internet connection required for FTP data download

## Author

Mark Lelaono

## License

This project is for drought monitoring and analysis purposes within the ICPAC region.

## Acknowledgments

- ICPAC (IGAD Climate Prediction and Applications Centre) for providing CDI data
- GADM for administrative boundary data
