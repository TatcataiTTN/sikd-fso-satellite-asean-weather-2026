"""
weather_stats.py — Diurnal & Inter-City Weather Statistics (Task 2-3, plan 07-5)
==================================================================================
Builds hourly cloud-cover climatology (month x local-hour) and inter-city daily
cloud correlation from the ERA5 hourly record (data/raw/era5_hourly_{city}.csv.gz,
Task 1, scripts/01_fetch_weather_era5.py, 2015-01-01 to 2024-12-31, 10 years).
Backs figures F2 (diurnal heatmap) and F3 (joint availability / correlation
matrix) of the paper redesign (07-2 v2) and the tests in
tests/test_weather_stats.py.

Definitions
-----------
P_cloud (hourly climatology) : fraction of hours in a (month, local-hour) bin
    with cloud_cover_pct >= CLOUD_OUTAGE_THRESHOLD, over the full record.
    Same 85% threshold as scripts/01_fetch_weather_era5.py's monthly P_cloud,
    but resolved by local hour instead of averaged into a single monthly
    number.
Clear day : a calendar day whose MEAN hourly cloud_cover_pct is below
    CLOUD_OUTAGE_THRESHOLD. Used for daily_cloud_correlation and
    joint_clear_probability; consistent with the monthly P_cloud definition
    (a "cloudy day" there is exactly a non-clear day here).

Caching
-------
Per-city hourly climatology matrices are cached to
data/intermediate/hourly_climatology_{city}.npy (12x24 float array) so the
~87,600-row raw CSV is parsed once. Daily aggregates (needed for correlation
and joint availability) are cached separately to
data/intermediate/daily_cloud_{city}.npz.

Diurnal finding (investigated 03/07/2026, see tests/test_weather_stats.py)
---------------------------------------------------------------------------
Total cloud cover (ERA5 `cloud_cover`, an areal fraction across all cloud
layers) and precipitation have DIFFERENT diurnal phases in this record:
  - Precipitation cleanly peaks in the afternoon (13-17h local) for all 8
    ASEAN cities (convective thunderstorm signature) — verified 8/8.
  - Total cloud cover instead peaks near dawn/overnight for 6/8 cities and
    is lowest mid-morning (9-12h local), consistent with nocturnal
    radiative-cooling stratus/fog that dissipates as the morning sun
    breaks the boundary layer, before afternoon convection rebuilds cloud.
This is a known meteorological effect, not an extraction bug (cross-checked
directly against the raw precipitation series). The practical consequence
for FSO/SIKD link scheduling is that the best combined cloud+rain window is
mid-morning (roughly 05:00-10:00 local), not overnight as originally
assumed — overnight has the least rain but not the least cloud, and
afternoon has both cloud and heavy rain. See load_hourly_rain_climatology()
below for the precipitation-based diurnal profile used to cross-check this.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'intermediate')
os.makedirs(CACHE_DIR, exist_ok=True)

CLOUD_OUTAGE_THRESHOLD = 85.0  # % — matches scripts/01_fetch_weather_era5.py

CITIES = ["hanoi", "danang", "hcmc", "bangkok",
          "singapore", "manila", "jakarta", "kuala_lumpur"]

_START_DATE = "2015-01-01"
_END_DATE = "2024-12-31"

# In-process caches, keyed by city, avoid re-reading disk cache repeatedly
# within a single run (e.g. building the full 8x8 correlation matrix).
_daily_cache: dict = {}
_climatology_cache: dict = {}


def _raw_csv_path(city: str) -> str:
    return os.path.join(RAW_DIR, f"era5_hourly_{city}.csv.gz")


def _daily_cache_path(city: str) -> str:
    return os.path.join(CACHE_DIR, f"daily_cloud_{city}.npz")


def _climatology_cache_path(city: str) -> str:
    return os.path.join(CACHE_DIR, f"hourly_climatology_{city}.npy")


def _load_raw_hourly(city: str) -> pd.DataFrame:
    path = _raw_csv_path(city)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing raw ERA5 hourly file for '{city}': {path}. "
            "Run scripts/01_fetch_weather_era5.py first (Task 1)."
        )
    df = pd.read_csv(path, compression='gzip', parse_dates=['time_local'])
    return df


def _daily_series(city: str) -> pd.Series:
    """Daily mean cloud_cover_pct for `city`, indexed by local calendar date.

    Cached to disk after first computation (see module docstring)."""
    if city in _daily_cache:
        return _daily_cache[city]

    cache_path = _daily_cache_path(city)
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        dates = pd.to_datetime(z['dates'].astype(str))
        s = pd.Series(z['values'], index=dates)
        _daily_cache[city] = s
        return s

    df = _load_raw_hourly(city)
    df['date'] = df['time_local'].dt.floor('D')
    daily = df.groupby('date')['cloud_cover_pct'].mean()

    np.savez(
        cache_path,
        dates=daily.index.values.astype('datetime64[D]').astype(str),
        values=daily.values.astype(float),
    )
    _daily_cache[city] = daily
    return daily


def load_hourly_climatology(city: str) -> np.ndarray:
    """P_cloud per (month, local hour) -> shape (12, 24), months 1-12 rows,
    hours 0-23 columns."""
    if city in _climatology_cache:
        return _climatology_cache[city]

    cache_path = _climatology_cache_path(city)
    if os.path.exists(cache_path):
        matrix = np.load(cache_path)
        _climatology_cache[city] = matrix
        return matrix

    df = _load_raw_hourly(city)
    month = df['time_local'].dt.month.values       # 1-12
    hour = df['time_local'].dt.hour.values          # 0-23
    is_cloudy = (df['cloud_cover_pct'].values >= CLOUD_OUTAGE_THRESHOLD).astype(float)

    matrix = np.zeros((12, 24))
    for m_idx in range(12):
        month_mask = month == (m_idx + 1)
        for h_idx in range(24):
            mask = month_mask & (hour == h_idx)
            if mask.any():
                matrix[m_idx, h_idx] = is_cloudy[mask].mean()

    np.save(cache_path, matrix)
    _climatology_cache[city] = matrix
    return matrix


def _rain_climatology_cache_path(city: str) -> str:
    return os.path.join(CACHE_DIR, f"hourly_rain_climatology_{city}.npy")


def load_hourly_rain_climatology(city: str) -> np.ndarray:
    """Mean precipitation (mm/h) per (month, local hour) -> shape (12, 24).

    This is the TRUE convective-timing signal (afternoon peak, verified 8/8
    ASEAN cities) — distinct from load_hourly_climatology's total cloud
    cover, which instead peaks near dawn/overnight for most cities (see
    module docstring). Used to cross-check diurnal cloud claims and to
    build the combined cloud+rain "best FSO window" figure (Task 15, F2)."""
    cache_path = _rain_climatology_cache_path(city)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    df = _load_raw_hourly(city)
    month = df['time_local'].dt.month.values
    hour = df['time_local'].dt.hour.values
    precip = df['precip_mm'].values

    matrix = np.zeros((12, 24))
    for m_idx in range(12):
        month_mask = month == (m_idx + 1)
        for h_idx in range(24):
            mask = month_mask & (hour == h_idx)
            if mask.any():
                matrix[m_idx, h_idx] = precip[mask].mean()

    np.save(cache_path, matrix)
    return matrix


def diurnal_amplitude(city: str, month: int) -> float:
    """max - min of the 24-hour P_cloud profile for the given calendar
    month (1-12)."""
    matrix = load_hourly_climatology(city)
    profile = matrix[month - 1, :]
    return float(profile.max() - profile.min())


def _daily_clear_flags(city: str, months: list) -> pd.Series:
    """Boolean series (indexed by date, restricted to `months`) marking
    'clear day' (daily mean cloud cover below the outage threshold)."""
    daily = _daily_series(city)
    sel = daily[daily.index.month.isin(months)]
    return sel < CLOUD_OUTAGE_THRESHOLD


def daily_cloud_correlation(city_a: str, city_b: str, months: list) -> float:
    """Pearson correlation of daily mean cloud cover between two cities,
    restricted to the given calendar months, over the full 10-year record."""
    if city_a == city_b:
        return 1.0

    da = _daily_series(city_a)
    db = _daily_series(city_b)
    da_m = da[da.index.month.isin(months)]
    db_m = db[db.index.month.isin(months)]

    common_idx = da_m.index.intersection(db_m.index)
    a = da_m.loc[common_idx].values
    b = db_m.loc[common_idx].values

    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def correlation_matrix(months: list) -> np.ndarray:
    """8x8 Pearson correlation matrix of daily mean cloud cover across
    CITIES, restricted to the given calendar months."""
    n = len(CITIES)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = daily_cloud_correlation(CITIES[i], CITIES[j], months)
            M[i, j] = r
            M[j, i] = r
    return M


def joint_clear_probability(cities: list, month: int = None, months: list = None) -> float:
    """P(at least one city in `cities` has a clear day), computed directly
    from the joint daily record (not an independence assumption) —
    comparable to Dang 2023 Fig. 7.

    Restricted to a single calendar month via `month` (1-12, kept for
    backward compatibility with the original single-month contract), or to
    an arbitrary set via `months` (e.g. months=list(range(1,13)) for an
    annual figure comparable to Dang 2023's reported 81.92/96.86/99.44%,
    which is not month-specific in the source paper). If neither is given,
    all 12 calendar months are used."""
    if months is None:
        months = [month] if month is not None else list(range(1, 13))

    series_list = []
    for city in cities:
        flags = _daily_clear_flags(city, months)
        flags = flags.rename(city)
        series_list.append(flags)

    df = pd.concat(series_list, axis=1, join='inner')
    if df.empty:
        return 0.0

    any_clear = df.any(axis=1)
    return float(any_clear.mean())


def record_metadata() -> dict:
    return {
        "start": _START_DATE,
        "end": _END_DATE,
        "cities": list(CITIES),
    }
