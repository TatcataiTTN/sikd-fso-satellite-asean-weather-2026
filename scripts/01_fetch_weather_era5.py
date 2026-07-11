"""
01_fetch_weather_era5.py — ERA5 Hourly Climatology via Open-Meteo (10-year record)
====================================================================================
Fetches HOURLY cloud_cover + precipitation for 8 ASEAN cities, 2015-01-01 to
2024-12-31 (10 years), from the Open-Meteo Historical Weather API (ERA5 /
ERA5-Land reanalysis, ECMWF). Supersedes the earlier 6-year DAILY-only fetch
(2015-2020) used up to 02/07/2026.

Raw hourly data is saved to data/raw/era5_hourly_{city}.csv.gz (one row per
hour, ~87,672 rows/city) so downstream modules (modules/weather_stats.py,
Task 2) can compute diurnal climatology P_cloud(month, local_hour) and
inter-city daily correlation (Task 3) without re-fetching.

The legacy monthly aggregate (data/real_climate_data.json, consumed by
modules/weather_model.py) is recomputed from this same 10-year hourly record
for consistency. R_mm_month / P_cloud / rain_fraction definitions unchanged.

Task 1, plan 07-5. Full spec: 07-6.Project_Prompts.md PROMPT-T1.

Usage:
  cd 05_Code_v2 && python scripts/01_fetch_weather_era5.py
"""
import csv
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import save_provenance, save_verify_numbers  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"          # 10-year record (was 2015-2020, 6 years, daily-only)
CLOUD_OUTAGE_THRESHOLD = 85.0    # % daily mean cloud cover treated as thick/overcast
RAIN_DAY_MM_MIN = 1.0            # mm/day minimum to count as a "rain day"

# (lat, lon) — identical to GROUND_STATIONS in orbital_mechanics.py, verified
# 25/06/2026 via web search against latlong.net / geodatos.net / OpenStreetMap /
# dateandtime.info. Intentionally does NOT reuse Nguyen/Le/Pham/Dang (2023)'s
# own HCMC coordinate (10.4632N, 106.4207E), which sits ~35-40km south of the
# actual city center.
CITIES = {
    "hanoi":         (21.0285, 105.8542),
    "danang":        (16.0544, 108.2022),
    "hcmc":          (10.8231, 106.6297),
    "bangkok":       (13.7563, 100.5018),
    "singapore":     (1.3521, 103.8198),
    "manila":        (14.5995, 120.9842),
    "jakarta":       (-6.2088, 106.8456),
    "kuala_lumpur":  (3.1390, 101.6869),
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city_hourly(city: str, lat: float, lon: float, retries: int = 3) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "cloud_cover,precipitation",
        "timezone": "auto",
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def save_raw_hourly_csv(city: str, hourly_json: dict) -> str:
    times = hourly_json["hourly"]["time"]
    cloud = hourly_json["hourly"]["cloud_cover"]
    precip = hourly_json["hourly"]["precipitation"]
    path = os.path.join(RAW_DIR, f"era5_hourly_{city}.csv.gz")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_local", "cloud_cover_pct", "precip_mm"])
        for t, c, p in zip(times, cloud, precip):
            writer.writerow([t, "" if c is None else c, "" if p is None else p])
    return path


def daily_from_hourly(times, cloud, precip) -> dict:
    """Aggregate hourly records -> daily {date_str: (mean_cloud, total_precip)}."""
    by_day = defaultdict(lambda: {"cloud": [], "precip": []})
    for t, c, p in zip(times, cloud, precip):
        day = t[:10]
        if c is not None:
            by_day[day]["cloud"].append(c)
        if p is not None:
            by_day[day]["precip"].append(p)
    daily = {}
    for day, vals in by_day.items():
        mean_cloud = float(np.mean(vals["cloud"])) if vals["cloud"] else None
        total_precip = float(np.sum(vals["precip"])) if vals["precip"] else 0.0
        daily[day] = (mean_cloud, total_precip)
    return daily


def monthly_climatology(daily: dict):
    """Return (list of 12 (R_mm_month, P_cloud, rain_frac) tuples Jan-Dec, n_years)."""
    months = {m: {"precip_totals": [], "cloud_flags": [], "rain_flags": []}
              for m in range(1, 13)}
    years_seen = set()
    for day_str, (mean_cloud, total_precip) in daily.items():
        dt = datetime.fromisoformat(day_str)
        years_seen.add(dt.year)
        m = dt.month
        if mean_cloud is not None:
            months[m]["cloud_flags"].append(1.0 if mean_cloud >= CLOUD_OUTAGE_THRESHOLD else 0.0)
        months[m]["rain_flags"].append(1.0 if total_precip >= RAIN_DAY_MM_MIN else 0.0)
        months[m]["precip_totals"].append(total_precip)

    n_years = len(years_seen)
    out = []
    for m in range(1, 13):
        d = months[m]
        r_mm_month = sum(d["precip_totals"]) / n_years if n_years else 0.0
        p_cloud = float(np.mean(d["cloud_flags"])) if d["cloud_flags"] else 0.0
        rain_frac = float(np.mean(d["rain_flags"])) if d["rain_flags"] else 0.0
        out.append((round(r_mm_month, 1), round(p_cloud, 3), round(rain_frac, 3)))
    return out, n_years


def main():
    t0 = time.time()
    print("=" * 70)
    print("Fetching ERA5 hourly climatology (10-year record) from Open-Meteo")
    print(f"Period: {START_DATE} to {END_DATE} | cloud outage threshold: "
          f"{CLOUD_OUTAGE_THRESHOLD}% daily mean cloud cover")
    print("=" * 70)

    results = {}
    raw_files = {}
    n_years_check = None

    for city, (lat, lon) in CITIES.items():
        print(f"\nFetching {city} ({lat}, {lon})...")
        hourly_json = fetch_city_hourly(city, lat, lon)
        times = hourly_json["hourly"]["time"]
        cloud = hourly_json["hourly"]["cloud_cover"]
        precip = hourly_json["hourly"]["precipitation"]
        print(f"  {len(times)} hourly records ({times[0]} .. {times[-1]})")

        raw_path = save_raw_hourly_csv(city, hourly_json)
        raw_files[city] = raw_path
        print(f"  Saved raw: {os.path.basename(raw_path)}")

        daily = daily_from_hourly(times, cloud, precip)
        monthly, n_years = monthly_climatology(daily)
        n_years_check = n_years
        results[city] = monthly
        for m, (r, pc, rf) in enumerate(monthly, start=1):
            print(f"    Month {m:2d}: R={r:7.1f} mm  P_cloud={pc:.3f}  rain_frac={rf:.3f}")

        time.sleep(1.0)  # be polite to the API between cities

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis, ECMWF)",
        "source_url": "https://open-meteo.com/en/docs/historical-weather-api",
        "date_range": {"start": START_DATE, "end": END_DATE},
        "n_years": n_years_check,
        "resolution": "hourly (raw, data/raw/era5_hourly_{city}.csv.gz), "
                       "aggregated to daily then to monthly climatology below",
        "cities_lat_lon": CITIES,
        "definitions": {
            "R_mm_month": "mean total monthly precipitation (mm), averaged across years",
            "P_cloud": f"fraction of days with daily mean cloud cover >= {CLOUD_OUTAGE_THRESHOLD}%",
            "rain_fraction": f"fraction of days with total precipitation >= {RAIN_DAY_MM_MIN} mm",
        },
        "cross_check_reference": {
            "citation": "Nguyen, Le, Pham, Dang (2023), doi:10.4108/eetinis.v10i3.3327",
            "note": ("Independent ECMWF ERA-Interim CLWC analysis (2015-2020) for Hanoi, "
                     "Da Nang, Ho Chi Minh City reports system availability 80-87% "
                     "(lowest in autumn, highest in spring) at a 30 dB link budget, "
                     "used here only as a qualitative sanity check, not merged numerically."),
        },
        "monthly_data_R_Pcloud_rainfrac": results,
        "raw_hourly_files": {c: os.path.basename(p) for c, p in raw_files.items()},
    }
    meta_path = os.path.join(DATA_DIR, "real_climate_data.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved aggregated + provenance metadata: {meta_path}")

    verify = {}
    for city, monthly in results.items():
        p_clouds = [m[1] for m in monthly]
        verify[f"{city}_Pcloud_min"] = f"{min(p_clouds):.3f}"
        verify[f"{city}_Pcloud_max"] = f"{max(p_clouds):.3f}"
    save_verify_numbers(verify, "weather_climatology_10yr")

    save_provenance(
        script_name="01_fetch_weather_era5",
        params={
            "START_DATE": START_DATE,
            "END_DATE": END_DATE,
            "n_years": n_years_check,
            "CLOUD_OUTAGE_THRESHOLD_pct": CLOUD_OUTAGE_THRESHOLD,
            "RAIN_DAY_MM_MIN": RAIN_DAY_MM_MIN,
            "n_cities": len(CITIES),
        },
        key_numbers=verify,
        runtime_secs=time.time() - t0,
        output_files=[meta_path] + list(raw_files.values()),
        data_sources={"Open-Meteo archive API": ARCHIVE_URL},
        formulas={
            "P_cloud (daily)": "1 if daily_mean(cloud_cover_hourly) >= 85% else 0, "
                               "averaged over all days of that calendar month across "
                               "10 years",
            "R_mm_month": "sum(daily_precip_totals in month) / n_years",
            "Visibility (derived, no historical source)": "V_km = 15 - 10 * P_cloud",
        },
    )
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
