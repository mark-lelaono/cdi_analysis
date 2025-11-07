# Drought Monitoring Analysis

This project contains analysis of Combined Drought Indicator (CDI) data for monitoring drought conditions.

## Project Structure

```
Drought_Monitoring/
├── cdi_analysis.ipynb      # Main analysis notebook
├── cdi_input/              # Input TIF files (not tracked in git)
├── cdi_output/             # Generated outputs (not tracked in git)
│   ├── *.png              # Visualization outputs
│   └── *.csv              # Data exports
└── README.md              # This file
```

## Data Files

Data files are excluded from version control to keep the repository lightweight. The following data is expected:

### Input Data (`cdi_input/`)
- Monthly CDI TIF files (2020-2023)
- Format: `YYYY-MMM.tif` (e.g., `2020-jan.tif`)

### Output Data (`cdi_output/`)
- Monthly CDI visualizations: `CDI_YYYY_MM.png`
- Time series plots: `CDI_Timeseries.png`, `CDI_Timeseries_2020_2023.png`
- CSV export: `CDI_mean_timeseries.csv`

## Getting Started

1. Clone this repository
2. Place your CDI input TIF files in the `cdi_input/` directory
3. Open and run `cdi_analysis.ipynb` to generate the analysis
4. Outputs will be saved to `cdi_output/`

## Requirements

See the notebook for required Python packages. Typically includes:
- numpy
- pandas
- rasterio
- matplotlib
- geopandas (if applicable)

## Notes

All data files (`.tif`, `.png`, `.csv`) are excluded from version control via `.gitignore`.
