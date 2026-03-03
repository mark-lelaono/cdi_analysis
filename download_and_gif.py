#!/usr/bin/env python3
"""
download_and_gif.py
-------------------
Downloads monthly CDI rasters for August–December 2025 from ICPAC's
public FTP server, then generates one storytelling GIF per month
using make_gif.py's generate_monthly_gifs().

Usage:
    python download_and_gif.py                  # download + generate GIFs
    python download_and_gif.py --skip-download  # GIFs only (files already local)
    python download_and_gif.py --force           # re-download even if files exist
    python download_and_gif.py --months aug sep  # only specific months
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from make_gif import generate_monthly_gifs
from cdi_config import PATHS, MONTH_ABBR_TO_NAME

log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

MONTHLY_BASE_URL = PATHS.MONTHLY_BASE_URL
YEAR = 2025
TARGET_MONTHS = ["aug", "sep", "oct", "nov", "dec"]
LOCAL_DIR = PATHS.MONTHLY_DIR / str(YEAR)
MONTH_NAMES = MONTH_ABBR_TO_NAME


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def build_file_url(month_abbr: str) -> str:
    """Build the full download URL for a given month."""
    filename = f"eadw-cdi-data-{YEAR}-{month_abbr}.tif"
    return f"{MONTHLY_BASE_URL}{YEAR}/{filename}"


def download_file(url: str, save_path: Path, timeout: int = 120) -> bool:
    """Download a file with progress bar. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with open(save_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=save_path.name, ncols=80,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            log.warning("  Not available on server (404): %s", save_path.name)
        else:
            log.error("  HTTP error downloading %s: %s", save_path.name, e)
        return False

    except Exception as e:
        log.error("  Failed to download %s: %s", save_path.name, e)
        return False


def download_monthly_rasters(months: list[str], force: bool = False) -> list[Path]:
    """
    Download CDI monthly rasters for the specified months.

    Parameters
    ----------
    months : list[str]
        Month abbreviations, e.g. ["aug", "sep", "oct", "nov", "dec"].
    force : bool
        Re-download even if the file already exists locally.

    Returns
    -------
    list[Path]
        Paths to successfully downloaded (or already existing) rasters.
    """
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for month in months:
        filename = f"eadw-cdi-data-{YEAR}-{month}.tif"
        save_path = LOCAL_DIR / filename
        url = build_file_url(month)

        if save_path.exists() and not force:
            size_mb = save_path.stat().st_size / (1024 * 1024)
            log.info("  Already exists: %s (%.1f MB)", filename, size_mb)
            downloaded.append(save_path)
            continue

        log.info("  Downloading %s ...", MONTH_NAMES.get(month, month))
        if download_file(url, save_path):
            size_mb = save_path.stat().st_size / (1024 * 1024)
            log.info("  Saved: %s (%.1f MB)", filename, size_mb)
            downloaded.append(save_path)
        else:
            # Clean up partial download
            if save_path.exists():
                save_path.unlink()

    return downloaded


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Aug–Dec 2025 CDI rasters and generate GIFs",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use existing files in cdi_monthly/2025/",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--months", nargs="+", default=TARGET_MONTHS,
        help=f"Month abbreviations to process (default: {' '.join(TARGET_MONTHS)})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    months = [m.lower()[:3] for m in args.months]
    invalid = [m for m in months if m not in MONTH_NAMES]
    if invalid:
        log.error("Invalid month abbreviations: %s", invalid)
        sys.exit(1)

    # ── Step 1: Download ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("CDI Monthly Download & GIF Pipeline")
    log.info("Year: %d  |  Months: %s", YEAR,
             ", ".join(MONTH_NAMES[m] for m in months))
    log.info("=" * 60)

    if not args.skip_download:
        log.info("\nStep 1: Downloading CDI rasters from ICPAC FTP ...")
        log.info("  Source: %s%d/", MONTHLY_BASE_URL, YEAR)
        downloaded = download_monthly_rasters(months, force=args.force)

        if not downloaded:
            log.error("No rasters downloaded. Check your internet connection "
                      "or verify files are available at %s%d/", MONTHLY_BASE_URL, YEAR)
            sys.exit(1)

        log.info("\n  Downloaded %d / %d months", len(downloaded), len(months))
    else:
        log.info("\nStep 1: Skipping download (--skip-download)")
        existing = list(LOCAL_DIR.glob("*.tif"))
        # Filter to requested months only
        existing = [
            p for p in existing
            if any(f"-{m}." in p.name for m in months)
        ]
        log.info("  Found %d existing rasters in %s", len(existing), LOCAL_DIR)
        if not existing:
            log.error("No .tif files found in %s for months: %s",
                      LOCAL_DIR, ", ".join(months))
            sys.exit(1)

    # ── Step 2: Generate GIFs ─────────────────────────────────────
    log.info("\nStep 2: Generating monthly GIFs ...")
    log.info("  Each GIF: ICPAC overview → worst-hit country → admin1 hotspot")
    log.info("")

    raster_dir = str(LOCAL_DIR)  # cdi_monthly/2025 — only process this year
    output_gifs = generate_monthly_gifs(raster_dir=raster_dir)

    # ── Summary ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("COMPLETE")
    log.info("=" * 60)
    if output_gifs:
        log.info("Generated %d GIFs:", len(output_gifs))
        for gif in output_gifs:
            size_kb = os.path.getsize(gif) / 1024
            log.info("  → %s  (%.0f KB)", gif, size_kb)
    else:
        log.info("No GIFs generated. Check raster files for corruption.")

    log.info("\nOutput directory: cdi_output/hexmaps/")


if __name__ == "__main__":
    main()
