"""
orbital_mechanics.py — Satellite Orbital Mechanics for SIKD/FSO
================================================================

Tính toán quỹ đạo vệ tinh LEO (Starlink) và các tham số liên kết
FSO cho hệ thống SIKD trên vùng ASEAN.

Chức năng chính
---------------
    Nhóm 1 — TLE Management   : parse_tle_block, load_tle_file,
                                  fetch_tle_celestrak
    Nhóm 2 — Geometry         : compute_elevation, compute_slant_path,
                                  compute_zenith_angle
    Nhóm 3 — Visibility        : get_visible_satellites, get_best_satellite
    Nhóm 4 — Time Series       : compute_elevation_timeseries,
                                  find_pass_windows
    Nhóm 5 — Coverage          : compute_coverage_grid
    Nhóm 6 — Convenience       : compute_link_geometry

Thư viện
--------
    skyfield  — SGP4 propagation, coordinate transforms (primary)
    numpy     — array operations
    requests  — TLE download (optional, only for fetch_tle_celestrak)

Tham số mặc định
----------------
    H_S_KM_DEFAULT   = 550.0   km  (Starlink shell 1)
    MIN_ELEVATION    = 10.0    deg (minimum usable elevation)
    ASEAN_BBOX       = (0, 25, 95, 130)  lat/lon bounding box

References
----------
    [P1] Vu et al., IEEE Access 2022
    [P2] Vu, PTIT 2025
    [P3] Toka et al., Computer Networks 2025 — LEO routing
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from skyfield.api import EarthSatellite, wgs84, load
from skyfield.timelib import Time as SkyfieldTime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H_S_KM_DEFAULT  = 550.0    # km — Starlink shell 1 nominal altitude
MIN_ELEVATION   = 10.0     # deg — minimum elevation for usable FSO link
RE_KM           = 6371.0   # km — Earth mean radius

# ASEAN bounding box (lat_min, lat_max, lon_min, lon_max)
ASEAN_BBOX = (0.0, 25.0, 95.0, 130.0)

# Timescale — loaded once at import (uses built-in Skyfield data, no download)
_TS = load.timescale()

# ---------------------------------------------------------------------------
# ASEAN ground station registry (mirrors weather_model.CITY_COORDS)
# ---------------------------------------------------------------------------
GROUND_STATIONS: dict[str, dict] = {
    "hanoi":        {"lat": 21.0285, "lon": 105.8542, "alt_m":  16.0},
    "hcmc":         {"lat": 10.8231, "lon": 106.6297, "alt_m":  19.0},
    "danang":       {"lat": 16.0544, "lon": 108.2022, "alt_m":  10.0},
    "bangkok":      {"lat": 13.7563, "lon": 100.5018, "alt_m":   2.0},
    "singapore":    {"lat":  1.3521, "lon": 103.8198, "alt_m":  15.0},
    "manila":       {"lat": 14.5995, "lon": 120.9842, "alt_m":  16.0},
    "jakarta":      {"lat": -6.2088, "lon": 106.8456, "alt_m":   8.0},
    "kuala_lumpur": {"lat":  3.1390, "lon": 101.6869, "alt_m":  68.0},
}


# ---------------------------------------------------------------------------
# 1. TLE Management
# ---------------------------------------------------------------------------

def parse_tle_block(tle_text: str) -> list[dict]:
    """
    Parse một khối TLE text (3-line format) thành list of dicts.

    Định dạng đầu vào (mỗi vệ tinh 3 dòng):
        STARLINK-1234
        1 NNNNNC YYYYDDD.DDDDDDDD ...
        2 NNNNN  III.IIII ...

    Parameters
    ----------
    tle_text : chuỗi TLE thô (nhiều vệ tinh, mỗi vệ tinh 3 dòng)

    Returns
    -------
    list of dicts, mỗi dict có:
        name   : tên vệ tinh
        line1  : TLE line 1
        line2  : TLE line 2
    """
    lines = [ln.strip() for ln in tle_text.strip().splitlines() if ln.strip()]
    satellites = []
    i = 0
    while i + 2 < len(lines):
        name  = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        # Validate: line1 starts with '1 ', line2 starts with '2 '
        if line1.startswith("1 ") and line2.startswith("2 "):
            satellites.append({"name": name, "line1": line1, "line2": line2})
            i += 3
        else:
            i += 1  # skip malformed entry
    return satellites


def load_tle_file(filepath: str) -> list[dict]:
    """
    Đọc file TLE từ đĩa và parse thành list of dicts.

    Parameters
    ----------
    filepath : đường dẫn đến file TLE (.txt hoặc .tle)

    Returns
    -------
    list of dicts (xem parse_tle_block)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return parse_tle_block(f.read())


def fetch_tle_celestrak(
    group: str = "starlink",
    timeout: int = 15,
) -> list[dict]:
    """
    Tải TLE từ CelesTrak (không cần authentication).

    URL: https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle

    Parameters
    ----------
    group   : tên nhóm vệ tinh ('starlink', 'oneweb', 'iridium', ...)
    timeout : HTTP timeout (giây)

    Returns
    -------
    list of dicts (xem parse_tle_block)

    Raises
    ------
    ImportError  : nếu requests không được cài
    RuntimeError : nếu HTTP request thất bại
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required: pip install requests") from exc

    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"CelesTrak returned HTTP {resp.status_code} for group '{group}'"
        )
    return parse_tle_block(resp.text)


def make_skyfield_satellite(tle_dict: dict) -> EarthSatellite:
    """
    Tạo Skyfield EarthSatellite từ TLE dict.

    Parameters
    ----------
    tle_dict : dict với keys 'name', 'line1', 'line2'

    Returns
    -------
    EarthSatellite object
    """
    return EarthSatellite(tle_dict["line1"], tle_dict["line2"],
                          tle_dict["name"], _TS)


# ---------------------------------------------------------------------------
# 2. Geometry
# ---------------------------------------------------------------------------

def compute_elevation(
    satellite: EarthSatellite,
    lat_deg: float,
    lon_deg: float,
    t: SkyfieldTime,
    alt_m: float = 0.0,
) -> float:
    """
    Tính elevation angle (độ cao góc) của vệ tinh từ một điểm mặt đất.

    Parameters
    ----------
    satellite : Skyfield EarthSatellite
    lat_deg   : vĩ độ trạm mặt đất (degrees, + = North)
    lon_deg   : kinh độ trạm mặt đất (degrees, + = East)
    t         : Skyfield Time object
    alt_m     : độ cao trạm mặt đất so với mực nước biển (m)

    Returns
    -------
    elevation_deg : elevation angle (degrees), âm = dưới đường chân trời
    """
    ground = wgs84.latlon(lat_deg, lon_deg, elevation_m=alt_m)
    diff   = satellite - ground
    topo   = diff.at(t)
    alt, _az, _dist = topo.altaz()
    return float(alt.degrees)


def compute_zenith_angle(elevation_deg: float) -> float:
    """
    Tính zenith angle từ elevation angle.

    zeta = 90° - elevation

    Parameters
    ----------
    elevation_deg : elevation angle (degrees)

    Returns
    -------
    zeta_deg : zenith angle (degrees)
    """
    return 90.0 - elevation_deg


def compute_slant_path(
    elevation_deg: float,
    H_S_km: float = H_S_KM_DEFAULT,
) -> float:
    """
    Tính slant path length từ mặt đất đến vệ tinh.

    Công thức gần đúng (flat Earth + spherical correction):
        L_slant = H_S / sin(elevation)

    Lưu ý: dùng sin(elevation) thay vì cos(zeta) — tương đương nhau
    vì zeta = 90° - elevation → cos(zeta) = sin(elevation).

    Parameters
    ----------
    elevation_deg : elevation angle (degrees), phải > 0
    H_S_km        : satellite altitude (km)

    Returns
    -------
    L_slant_km : slant path length (km)

    Raises
    ------
    ValueError : nếu elevation ≤ 0
    """
    if elevation_deg <= 0:
        raise ValueError(
            f"elevation_deg must be > 0, got {elevation_deg:.2f}°"
        )
    zeta_rad = np.radians(compute_zenith_angle(elevation_deg))
    return H_S_km / np.cos(zeta_rad)


# ---------------------------------------------------------------------------
# 3. Visibility
# ---------------------------------------------------------------------------

def get_visible_satellites(
    satellites: list[EarthSatellite],
    lat_deg: float,
    lon_deg: float,
    t: SkyfieldTime,
    alt_m: float = 0.0,
    min_elevation: float = MIN_ELEVATION,
) -> list[dict]:
    """
    Lọc danh sách vệ tinh visible từ một điểm mặt đất tại thời điểm t.

    Parameters
    ----------
    satellites    : list of EarthSatellite
    lat_deg       : vĩ độ (degrees)
    lon_deg       : kinh độ (degrees)
    t             : Skyfield Time
    alt_m         : độ cao trạm (m)
    min_elevation : ngưỡng elevation tối thiểu (degrees)

    Returns
    -------
    list of dicts, mỗi dict:
        satellite     : EarthSatellite object
        name          : tên vệ tinh
        elevation_deg : elevation angle (degrees)
        zenith_deg    : zenith angle (degrees)
        slant_km      : slant path (km)
    Sorted by elevation_deg descending (best satellite first).
    """
    visible = []
    ground  = wgs84.latlon(lat_deg, lon_deg, elevation_m=alt_m)

    for sat in satellites:
        diff = sat - ground
        topo = diff.at(t)
        alt, _az, _dist = topo.altaz()
        el = float(alt.degrees)
        if el >= min_elevation:
            zeta = compute_zenith_angle(el)
            visible.append({
                "satellite":     sat,
                "name":          sat.name,
                "elevation_deg": el,
                "zenith_deg":    zeta,
                "slant_km":      compute_slant_path(el),
            })

    visible.sort(key=lambda x: x["elevation_deg"], reverse=True)
    return visible


def get_best_satellite(
    satellites: list[EarthSatellite],
    lat_deg: float,
    lon_deg: float,
    t: SkyfieldTime,
    alt_m: float = 0.0,
    min_elevation: float = MIN_ELEVATION,
) -> Optional[dict]:
    """
    Trả về vệ tinh có elevation angle cao nhất (best FSO link).

    Parameters
    ----------
    (xem get_visible_satellites)

    Returns
    -------
    dict (xem get_visible_satellites) hoặc None nếu không có vệ tinh visible
    """
    visible = get_visible_satellites(
        satellites, lat_deg, lon_deg, t, alt_m, min_elevation
    )
    return visible[0] if visible else None


# ---------------------------------------------------------------------------
# 4. Time Series
# ---------------------------------------------------------------------------

def make_time_array(
    t_start: datetime,
    duration_hours: float = 24.0,
    step_minutes: float = 1.0,
) -> SkyfieldTime:
    """
    Tạo mảng thời gian Skyfield cho time series analysis.

    Parameters
    ----------
    t_start        : thời điểm bắt đầu (datetime, UTC)
    duration_hours : tổng thời gian (giờ)
    step_minutes   : bước thời gian (phút)

    Returns
    -------
    Skyfield Time array
    """
    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=timezone.utc)
    n_steps = int(duration_hours * 60 / step_minutes) + 1
    times   = [t_start + timedelta(minutes=i * step_minutes) for i in range(n_steps)]
    return _TS.from_datetimes(times)


def compute_elevation_timeseries(
    satellite: EarthSatellite,
    lat_deg: float,
    lon_deg: float,
    t_array: SkyfieldTime,
    alt_m: float = 0.0,
) -> np.ndarray:
    """
    Tính elevation angle theo thời gian cho một vệ tinh và một trạm mặt đất.

    Parameters
    ----------
    satellite : EarthSatellite
    lat_deg   : vĩ độ (degrees)
    lon_deg   : kinh độ (degrees)
    t_array   : Skyfield Time array (từ make_time_array)
    alt_m     : độ cao trạm (m)

    Returns
    -------
    elevations : np.ndarray, shape (n_steps,), đơn vị degrees
    """
    ground = wgs84.latlon(lat_deg, lon_deg, elevation_m=alt_m)
    diff   = satellite - ground
    topo   = diff.at(t_array)
    alt, _az, _dist = topo.altaz()
    return np.array(alt.degrees)


def find_pass_windows(
    satellite: EarthSatellite,
    lat_deg: float,
    lon_deg: float,
    t_array: SkyfieldTime,
    alt_m: float = 0.0,
    min_elevation: float = MIN_ELEVATION,
) -> list[dict]:
    """
    Tìm các cửa sổ liên lạc (pass windows) khi vệ tinh visible.

    Parameters
    ----------
    satellite     : EarthSatellite
    lat_deg       : vĩ độ (degrees)
    lon_deg       : kinh độ (degrees)
    t_array       : Skyfield Time array
    alt_m         : độ cao trạm (m)
    min_elevation : ngưỡng elevation tối thiểu (degrees)

    Returns
    -------
    list of dicts, mỗi dict:
        start_idx     : index bắt đầu trong t_array
        end_idx       : index kết thúc trong t_array
        peak_idx      : index elevation cao nhất
        peak_elevation: elevation cao nhất (degrees)
        duration_steps: số bước thời gian trong pass
        elevations    : np.ndarray elevation trong pass
    """
    elevations = compute_elevation_timeseries(
        satellite, lat_deg, lon_deg, t_array, alt_m
    )
    visible_mask = elevations >= min_elevation

    windows = []
    in_pass  = False
    start_i  = 0

    for i, vis in enumerate(visible_mask):
        if vis and not in_pass:
            in_pass = True
            start_i = i
        elif not vis and in_pass:
            in_pass = False
            end_i   = i - 1
            pass_el = elevations[start_i:end_i + 1]
            peak_i  = int(np.argmax(pass_el)) + start_i
            windows.append({
                "start_idx":      start_i,
                "end_idx":        end_i,
                "peak_idx":       peak_i,
                "peak_elevation": float(elevations[peak_i]),
                "duration_steps": end_i - start_i + 1,
                "elevations":     pass_el,
            })

    # Handle pass still active at end of array
    if in_pass:
        end_i   = len(elevations) - 1
        pass_el = elevations[start_i:end_i + 1]
        peak_i  = int(np.argmax(pass_el)) + start_i
        windows.append({
            "start_idx":      start_i,
            "end_idx":        end_i,
            "peak_idx":       peak_i,
            "peak_elevation": float(elevations[peak_i]),
            "duration_steps": end_i - start_i + 1,
            "elevations":     pass_el,
        })

    return windows


# ---------------------------------------------------------------------------
# 5. Coverage Grid
# ---------------------------------------------------------------------------

def compute_coverage_grid(
    satellites: list[EarthSatellite],
    t: SkyfieldTime,
    lat_range: tuple[float, float] = (ASEAN_BBOX[0], ASEAN_BBOX[1]),
    lon_range: tuple[float, float] = (ASEAN_BBOX[2], ASEAN_BBOX[3]),
    resolution_deg: float = 1.0,
    min_elevation: float = MIN_ELEVATION,
) -> dict:
    """
    Tính lưới coverage: max elevation angle tại mỗi điểm lat/lon.

    Với mỗi điểm lưới, tìm vệ tinh có elevation cao nhất tại thời điểm t.
    Điểm có max_elevation ≥ min_elevation → có coverage.

    Parameters
    ----------
    satellites      : list of EarthSatellite
    t               : Skyfield Time (một thời điểm)
    lat_range       : (lat_min, lat_max) degrees
    lon_range       : (lon_min, lon_max) degrees
    resolution_deg  : độ phân giải lưới (degrees)
    min_elevation   : ngưỡng elevation tối thiểu

    Returns
    -------
    dict
        lats           : np.ndarray vĩ độ (1D)
        lons           : np.ndarray kinh độ (1D)
        max_elevation  : np.ndarray (n_lat, n_lon) — max elevation (degrees)
        zenith_best    : np.ndarray (n_lat, n_lon) — zenith angle of best sat
        has_coverage   : np.ndarray bool (n_lat, n_lon)
        coverage_frac  : fraction of grid points with coverage ∈ [0, 1]
    """
    lats = np.arange(lat_range[0], lat_range[1] + resolution_deg, resolution_deg)
    lons = np.arange(lon_range[0], lon_range[1] + resolution_deg, resolution_deg)

    max_el  = np.full((len(lats), len(lons)), -90.0)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            ground = wgs84.latlon(lat, lon)
            best_el = -90.0
            for sat in satellites:
                diff = sat - ground
                topo = diff.at(t)
                alt, _az, _dist = topo.altaz()
                el = float(alt.degrees)
                if el > best_el:
                    best_el = el
            max_el[i, j] = best_el

    has_coverage = max_el >= min_elevation
    zenith_best  = np.where(has_coverage, 90.0 - max_el, np.nan)

    return {
        "lats":          lats,
        "lons":          lons,
        "max_elevation": max_el,
        "zenith_best":   zenith_best,
        "has_coverage":  has_coverage,
        "coverage_frac": float(np.mean(has_coverage)),
    }


# ---------------------------------------------------------------------------
# 6. Convenience wrapper
# ---------------------------------------------------------------------------

def compute_link_geometry(
    satellite: EarthSatellite,
    lat_deg: float,
    lon_deg: float,
    t: SkyfieldTime,
    alt_m: float = 0.0,
    H_S_km: float = H_S_KM_DEFAULT,
) -> dict:
    """
    Tính toàn bộ tham số hình học liên kết FSO cho một cặp (vệ tinh, trạm).

    Đây là hàm wrapper chính — trả về tất cả tham số cần thiết để
    gọi channel_model.compute_channel().

    Parameters
    ----------
    satellite : EarthSatellite
    lat_deg   : vĩ độ trạm (degrees)
    lon_deg   : kinh độ trạm (degrees)
    t         : Skyfield Time
    alt_m     : độ cao trạm (m)
    H_S_km    : satellite altitude (km) — dùng cho slant path

    Returns
    -------
    dict
        elevation_deg : elevation angle (degrees)
        zenith_deg    : zenith angle (degrees)
        slant_km      : slant path length (km)
        is_visible    : True nếu elevation ≥ MIN_ELEVATION
        sat_name      : tên vệ tinh
    """
    el   = compute_elevation(satellite, lat_deg, lon_deg, t, alt_m)
    zeta = compute_zenith_angle(el)

    if el > 0:
        slant = compute_slant_path(el, H_S_km)
    else:
        slant = np.inf

    return {
        "elevation_deg": el,
        "zenith_deg":    zeta,
        "slant_km":      slant,
        "is_visible":    el >= MIN_ELEVATION,
        "sat_name":      satellite.name,
    }
