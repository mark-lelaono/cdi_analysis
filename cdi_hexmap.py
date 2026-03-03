# ==========================================================
# CDI HEX ENGINE — Production Version
# ==========================================================

import os
import json
import numpy as np
import pandas as pd
import rasterio
import h3
import folium
from dataclasses import dataclass, field
from shapely.geometry import Polygon
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).parent
DEFAULT_SHAPEFILE = Path(os.environ.get(
    "REGION_SHAPEFILE",
    BASE_DIR / "cdi_input" / "icpac_countries_merged.shp"
))

# ICPAC region center
ICPAC_CENTER = (5.5, 36.5)


@dataclass
class CDIConfig:
    coarse_resolution: int = 6
    fine_resolution: Optional[int] = 8
    aggregation_method: str = "mode"  # mode | mean | max
    by_category: bool = True
    opacity: float = 0.7
    shapefile: str = str(DEFAULT_SHAPEFILE)


CDI_CATEGORY_MAP = {
    0: "No Drought",
    1: "Watch",
    2: "Watch",
    3: "Watch",
    4: "Warning",
    5: "Warning",
    6: "Warning",
    7: "Alert",
    8: "Alert",
    9: "Alert",
    10: "Alert",
}

CDI_CATEGORY_COLORS = {
    "No Drought": "#FFFFFF",
    "Watch": "#FFD700",     # Yellow
    "Warning": "#FFA500",   # Orange
    "Alert": "#FF0000",     # Red
    "No Data": "#808080",   # Grey
}


# ==========================================================
# RASTER PROCESSING
# ==========================================================

class RasterProcessor:

    @staticmethod
    def extract_valid_pixels(
        raster_path: str,
        drought_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(float)
            transform = src.transform
            nodata = src.nodata

            valid_mask = ~np.isnan(data)
            if nodata is not None:
                valid_mask &= (data != nodata)
            if drought_only:
                valid_mask &= (data >= 1)

            rows, cols = np.nonzero(valid_mask)

            lons, lats = rasterio.transform.xy(transform, rows, cols)
            values = data[rows, cols]

        return np.array(lats), np.array(lons), np.array(values)


# ==========================================================
# H3 AGGREGATION ENGINE
# ==========================================================

class H3Aggregator:

    def __init__(self, resolution: int):
        self.resolution = resolution

    def latlngs_to_h3(self, lats: np.ndarray, lngs: np.ndarray) -> List[str]:
        return [
            h3.latlng_to_cell(lat, lng, self.resolution)
            for lat, lng in zip(lats, lngs)
        ]

    def aggregate(
        self,
        h3_indices: List[str],
        values: np.ndarray,
        method: str = "mode",
        by_category: bool = False
    ) -> pd.DataFrame:

        df = pd.DataFrame({
            "h3_index": h3_indices,
            "value": values
        })

        if by_category:
            df["category"] = df["value"].map(CDI_CATEGORY_MAP)

            grouped = (
                df.groupby(["h3_index", "category"])
                .size()
                .reset_index(name="count")
            )

            idx = grouped.groupby("h3_index")["count"].idxmax()
            result = grouped.loc[idx].reset_index(drop=True)

            result.rename(columns={"category": "cdi_class"}, inplace=True)

        else:
            if method == "mean":
                result = df.groupby("h3_index")["value"].mean().reset_index()
            elif method == "max":
                result = df.groupby("h3_index")["value"].max().reset_index()
            else:  # mode
                result = (
                    df.groupby("h3_index")["value"]
                    .agg(lambda x: x.mode().iloc[0])
                    .reset_index()
                )

            result.rename(columns={"value": "cdi_class"}, inplace=True)

        # Add centroids
        centroids = {
            h: h3.cell_to_latlng(h)
            for h in result["h3_index"].unique()
        }

        result["lat"] = result["h3_index"].map(lambda h: centroids[h][0])
        result["lng"] = result["h3_index"].map(lambda h: centroids[h][1])

        return result


# ==========================================================
# MAP BUILDER
# ==========================================================

class CDIHexMapBuilder:

    def __init__(self, center_lat: float, center_lng: float, zoom: int = 6):
        self.map = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom,
            tiles="CartoDB positron",
        )

    def add_country_boundaries(self, shapefile_path: str):
        """Add ICPAC country boundary outlines from shapefile."""
        if not Path(shapefile_path).exists():
            return
        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        geojson = json.loads(gdf.to_json())
        folium.GeoJson(
            geojson,
            style_function=lambda feat: {
                "fillColor": "transparent",
                "color": "#333333",
                "weight": 1.5,
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["COUNTRY"] if "COUNTRY" in gdf.columns else [],
                aliases=["Country:"] if "COUNTRY" in gdf.columns else [],
            ),
        ).add_to(self.map)

    @staticmethod
    def _h3_to_geojson_feature(h3_index: str, cdi_class, count: int = 0):
        """Convert a single H3 cell to a GeoJSON Feature with styling properties."""
        boundary = h3.cell_to_boundary(h3_index)
        # h3 returns (lat, lng) pairs; GeoJSON needs [lng, lat]
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])  # close ring

        if isinstance(cdi_class, str):
            category = cdi_class
        else:
            category = CDI_CATEGORY_MAP.get(int(cdi_class), "No Data")
        color = CDI_CATEGORY_COLORS.get(category, "#808080")

        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "h3": h3_index,
                "category": category,
                "color": color,
                "count": count,
            },
        }

    def add_hex_layer(self, df: pd.DataFrame, opacity: float = 0.7):
        """Add all hexagons as a single GeoJSON layer (fast batch rendering)."""
        features = []
        for _, row in df.iterrows():
            count = int(row["count"]) if "count" in row.index else 0
            features.append(
                self._h3_to_geojson_feature(row["h3_index"], row["cdi_class"], count)
            )

        geojson = {"type": "FeatureCollection", "features": features}

        folium.GeoJson(
            geojson,
            style_function=lambda feat: {
                "fillColor": feat["properties"]["color"],
                "color": feat["properties"]["color"],
                "weight": 0.5,
                "fillOpacity": opacity,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["category", "h3", "count"],
                aliases=["Drought Class:", "H3 Index:", "Pixel Count:"],
            ),
        ).add_to(self.map)

    def save(self, output_path: str):
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.map.save(output_path)


# ==========================================================
# MAIN ENGINE FUNCTION
# ==========================================================

def generate_cdi_hex_map(
    raster_path: str,
    output_html: str,
    config: CDIConfig
):

    print("Extracting raster pixels...")
    lats, lngs, values = RasterProcessor.extract_valid_pixels(raster_path)

    print("Aggregating coarse resolution...")
    coarse_agg = H3Aggregator(config.coarse_resolution)
    coarse_h3 = coarse_agg.latlngs_to_h3(lats, lngs)
    coarse_df = coarse_agg.aggregate(
        coarse_h3,
        values,
        method=config.aggregation_method,
        by_category=config.by_category
    )

    center_lat, center_lng = ICPAC_CENTER

    builder = CDIHexMapBuilder(center_lat, center_lng, zoom=5)
    builder.add_country_boundaries(config.shapefile)
    builder.add_hex_layer(coarse_df, config.opacity)

    builder.save(output_html)

    print(f"Map saved → {output_html}")


# ==========================================================
# CLI ENTRYPOINT (FIXED)
# ==========================================================

if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate CDI Hex Map")
    parser.add_argument("-i", "--input", required=True, help="Input CDI raster (GeoTIFF)")
    parser.add_argument("-o", "--output", default="cdi_hexmap.html", help="Output HTML file")
    parser.add_argument("--coarse", type=int, default=6, help="Coarse H3 resolution")
    parser.add_argument("--fine", type=int, default=None, help="Fine H3 resolution (optional)")
    parser.add_argument("--method", default="mode", choices=["mode", "mean", "max"])
    parser.add_argument("--by-category", action="store_true", help="Aggregate by drought category")
    parser.add_argument("--opacity", type=float, default=0.75)

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    config = CDIConfig(
        coarse_resolution=args.coarse,
        fine_resolution=args.fine,
        aggregation_method=args.method,
        by_category=args.by_category,
        opacity=args.opacity
    )

    generate_cdi_hex_map(
        raster_path=str(input_path),
        output_html=args.output,
        config=config
    )