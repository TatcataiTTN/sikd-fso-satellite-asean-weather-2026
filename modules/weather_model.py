"""
weather_model.py — ASEAN Climatological Weather Data for FSO/SIKD
==================================================================

Cung cấp dữ liệu thời tiết trung bình theo tháng (climatological averages)
cho 8 thành phố ASEAN, dùng để tính ảnh hưởng thời tiết lên kênh FSO.

Chiến lược thiết kế
-------------------
Climatological monthly averages, KHÔNG phải real-time API. Từ 03/07/2026
(Task 1, plan 07-5), R_mm_month và P_cloud được tính từ dữ liệu HOURLY
thật (ERA5/ERA5-Land, Open-Meteo) qua scripts/01_fetch_weather_era5.py,
giai đoạn 2015-01-01 đến 2024-12-31 (10 năm, nâng từ 6 năm/daily-only
trước đó), rồi nén thành climatology tháng. Raw hourly lưu ở
data/raw/era5_hourly_{city}.csv.gz — dùng cho modules/weather_stats.py
(climatology giờ×tháng, tương quan liên thành phố, Task 2-3) mà không cần
fetch lại.

Ba tham số thời tiết chính:
    R_mm_h   — rainfall rate (mm/h), tính từ mm/month
    V_km     — visibility (km), ảnh hưởng fog/haze attenuation
    P_cloud  — xác suất mây dày che phủ (link OFF), ∈ [0, 1]

Pipeline tích hợp với channel_model.py
---------------------------------------
    get_city_params(city, month)
        → (R_mm_h, V_km, P_cloud)
        → channel_model.compute_hl(zeta, V_km, R_mm_h)
        → sikd_performance.compute_sikd_performance(hg, hl, sigma_X2)
        → SKR_effective = SKR_clear × (1 - P_cloud)

Nguồn dữ liệu (cập nhật 03/07/2026 — 10 năm hourly, Task 1)
------------------------------------------------------------
    - R_mm_month : trung bình tổng mưa/tháng, 2015-2024 (10 năm), từ
      Open-Meteo archive API (ERA5/ERA5-Land), hourly precipitation nén
      thành ngày rồi thành tháng. Toạ độ = đúng GROUND_STATIONS
      (orbital_mechanics.py), đã verify qua web search (latlong.net,
      geodatos.net, OpenStreetMap) — KHÔNG dùng toạ độ riêng của Nguyen,
      Le, Pham, Dang (2023) vì điểm HCMC của bài đó (10.4632N, 106.4207E)
      lệch ~35-40km so với trung tâm TP.HCM thật.
    - P_cloud : tỷ lệ ngày/tháng có cloud_cover_mean ≥ 85%, tính từ hourly
      cloud_cover trung bình theo ngày (xem scripts/01_fetch_weather_era5.py,
      CLOUD_OUTAGE_THRESHOLD). So với bản 6 năm/daily-only cũ, chênh lệch
      P_cloud từng tháng lên tới 0.12 (xem data/PROVENANCE.md, Task 4) —
      biến động liên năm thực sự, không phải lỗi.
      Đối chiếu định tính với Nguyen, Le, Pham, Dang (2023) — bài dùng
      ECMWF ERA-Interim CLWC 2015-2020 cho Hà Nội/Đà Nẵng/TP.HCM, fit
      Gamma distribution, báo cáo system availability 80-87% theo mùa
      (EAI EETINIS, doi:10.4108/eetinis.v10i3.3327) — cùng xu hướng mùa,
      không phải số khớp tuyệt đối (phương pháp/ngưỡng khác nhau).
    - Visibility V_km : KHÔNG có trong Open-Meteo archive (không có biến
      visibility lịch sử) — vẫn là ước lượng suy ra từ P_cloud
      (V = 15 - 10×P_cloud), cùng logic như bản cũ và như Eq.(3) của
      Nguyen et al. (2023), nơi visibility cũng được suy ra từ CLWC chứ
      không đo trực tiếp.
    - Provenance đầy đủ (ngày tải, toạ độ, ngưỡng, raw daily data):
      05_Code/data/real_climate_data.json + open_meteo_raw_daily.json
    - rain_fraction (RAIN_FRACTION_DEFAULT, tỷ lệ giờ mưa trong ngày mưa):
      vẫn giữ ước lượng 0.15 — Open-Meteo archive chỉ có dữ liệu ngày,
      không đủ để suy ra phân bố trong ngày.

Thành phố hỗ trợ
----------------
    Vietnam  : hanoi, hcmc, danang
    ASEAN    : bangkok, singapore, manila, jakarta, kuala_lumpur

References
----------
    [P5] Koné et al., IJP 2024 — tropical FSO, Abidjan (rain attenuation model)
    [P3] Toka et al., Computer Networks 2025 — weather-adaptive routing
    [P4] Potter et al., JPL/NASA 1969 — cloud = complete blockage at optical
    [P6] Nguyen, Le, Pham, Dang, EAI EETINIS 2023, doi:10.4108/eetinis.v10i3.3327
         — ECMWF ERA-Interim CLWC, Gamma model, Hanoi/Da Nang/HCMC 2015-2020
"""

import numpy as np
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CityName = Literal[
    "hanoi", "hcmc", "danang",
    "bangkok", "singapore", "manila", "jakarta", "kuala_lumpur",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAIN_FRACTION_DEFAULT = 0.15
"""Fraction of hours in a month that it actually rains (tropical average).
   R_mm_h = R_mm_month / (30 × 24 × RAIN_FRACTION)
   0.15 ≈ 3.6 h/day — typical for humid tropical convective rain."""

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ---------------------------------------------------------------------------
# ASEAN Climatological Data
# ---------------------------------------------------------------------------
# Format per city: list of 12 tuples (one per month Jan–Dec):
#   (R_mm_month, V_km, P_cloud)
#
# R_mm_month : monthly rainfall total (mm/month)
# V_km       : mean visibility (km) — clear-air + haze
# P_cloud    : probability of thick cloud cover causing FSO outage ∈ [0, 1]
#
# Sources:
#   Vietnam data: Formula_Compendium_Part3_Weather.md + Vietnam Meteorological
#   ASEAN data  : tropical climatology estimates analogous to Paper 5 (Abidjan)
# ---------------------------------------------------------------------------

# R_mm_month, P_cloud: real Open-Meteo (ERA5/ERA5-Land) climatology,
# 2015-2020 daily data aggregated per calendar month — see
# 09_Fetch_Real_Weather_Data.py and data/real_climate_data.json for the
# fetch script, exact API parameters, and raw daily cache.
# V_km: still a derived estimate (V = 15 - 10*P_cloud); Open-Meteo's
# historical archive has no visibility variable, and neither does the
# reference lab study (Nguyen et al. 2023 also derive V from CLWC, not
# measure it directly).
ASEAN_CLIMATE_DATA: dict[str, list[tuple[float, float, float]]] = {

    # ── Hà Nội (21.0285°N, 105.8542°E) ──────────────────────────────────────
    "hanoi": [
        #  R_mm_month  V_km  P_cloud
        (     65.6,    10.4, 0.461),  # Jan
        (     48.2,    10.3, 0.473),  # Feb
        (     74.3,     9.7, 0.532),  # Mar
        (     94.1,    11.5, 0.350),  # Apr
        (    220.5,    10.8, 0.416),  # May
        (    258.8,     8.0, 0.697),  # Jun
        (    316.4,     8.3, 0.665),  # Jul
        (    418.9,     8.7, 0.626),  # Aug
        (    357.5,    10.3, 0.473),  # Sep
        (    179.8,    12.1, 0.287),  # Oct
        (     60.8,    12.6, 0.243),  # Nov
        (     36.0,    11.1, 0.390),  # Dec
    ],

    # ── TP. Hồ Chí Minh (10.8231°N, 106.6297°E) ─────────────────────────────
    "hcmc": [
        #  R_mm_month  V_km  P_cloud
        (     29.4,    12.7, 0.232),  # Jan
        (     13.9,    14.0, 0.102),  # Feb
        (     17.2,    14.3, 0.074),  # Mar
        (     83.9,    13.4, 0.157),  # Apr
        (    228.3,     9.7, 0.526),  # May
        (    250.8,     8.5, 0.647),  # Jun
        (    297.6,     8.2, 0.684),  # Jul
        (    299.4,     9.1, 0.594),  # Aug
        (    349.0,     7.8, 0.717),  # Sep
        (    314.2,     8.2, 0.681),  # Oct
        (    144.6,    10.7, 0.433),  # Nov
        (     64.7,    11.2, 0.381),  # Dec
    ],

    # ── Đà Nẵng (16.0544°N, 108.2022°E) ─────────────────────────────────────
    "danang": [
        #  R_mm_month  V_km  P_cloud
        (    170.0,    10.9, 0.410),  # Jan
        (     78.3,    12.6, 0.244),  # Feb
        (     51.7,    13.7, 0.132),  # Mar
        (     74.8,    13.5, 0.147),  # Apr
        (     73.0,    11.6, 0.339),  # May
        (     65.7,    10.3, 0.467),  # Jun
        (    117.5,    10.4, 0.455),  # Jul
        (    116.4,     9.7, 0.532),  # Aug
        (    241.8,    10.6, 0.443),  # Sep
        (    467.2,     9.8, 0.516),  # Oct
        (    346.5,    10.9, 0.407),  # Nov
        (    345.6,     9.3, 0.574),  # Dec
    ],

    # ── Bangkok (13.7563°N, 100.5018°E) ─────────────────────────────────────
    "bangkok": [
        #  R_mm_month  V_km  P_cloud
        (     24.0,    13.9, 0.106),  # Jan
        (     29.9,    14.7, 0.035),  # Feb
        (     44.4,    14.2, 0.081),  # Mar
        (     83.8,    12.9, 0.210),  # Apr
        (    158.1,     9.3, 0.568),  # May
        (    144.6,     7.8, 0.723),  # Jun
        (    195.3,     6.9, 0.813),  # Jul
        (    187.0,     6.9, 0.810),  # Aug
        (    293.3,     7.6, 0.740),  # Sep
        (    241.0,     9.5, 0.552),  # Oct
        (     71.7,    12.3, 0.270),  # Nov
        (     19.4,    13.7, 0.132),  # Dec
    ],

    # ── Singapore (1.3521°N, 103.8198°E) ────────────────────────────────────
    "singapore": [
        #  R_mm_month  V_km  P_cloud
        (    271.3,     8.6, 0.642),  # Jan
        (    123.3,    10.2, 0.481),  # Feb
        (    208.4,    11.5, 0.348),  # Mar
        (    291.2,     9.4, 0.560),  # Apr
        (    312.6,     9.2, 0.581),  # May
        (    293.5,    10.1, 0.493),  # Jun
        (    199.4,    10.7, 0.429),  # Jul
        (    230.1,    10.6, 0.445),  # Aug
        (    210.8,    10.0, 0.500),  # Sep
        (    301.8,     8.2, 0.681),  # Oct
        (    382.2,     6.9, 0.810),  # Nov
        (    315.3,     7.4, 0.761),  # Dec
    ],

    # ── Manila (14.5995°N, 120.9842°E) ──────────────────────────────────────
    "manila": [
        #  R_mm_month  V_km  P_cloud
        (     73.8,    13.2, 0.184),  # Jan
        (     31.8,    13.8, 0.117),  # Feb
        (     24.3,    14.1, 0.094),  # Mar
        (     39.9,    13.8, 0.117),  # Apr
        (    136.7,    11.9, 0.306),  # May
        (    253.6,    10.1, 0.493),  # Jun
        (    402.7,     7.7, 0.735),  # Jul
        (    343.2,     7.8, 0.716),  # Aug
        (    342.9,     7.9, 0.707),  # Sep
        (    244.9,     9.5, 0.548),  # Oct
        (    111.7,    11.6, 0.343),  # Nov
        (    171.4,    11.1, 0.387),  # Dec
    ],

    # ── Jakarta (6.2088°S, 106.8456°E) ──────────────────────────────────────
    "jakarta": [
        #  R_mm_month  V_km  P_cloud
        (    291.2,     6.6, 0.842),  # Jan
        (    321.7,     6.9, 0.813),  # Feb
        (    275.4,     8.0, 0.697),  # Mar
        (    233.4,     9.2, 0.583),  # Apr
        (    156.2,    11.4, 0.358),  # May
        (    119.4,    12.6, 0.240),  # Jun
        (     66.3,    13.2, 0.184),  # Jul
        (     58.7,    13.2, 0.177),  # Aug
        (     93.8,    12.8, 0.223),  # Sep
        (    161.2,    10.8, 0.419),  # Oct
        (    234.0,     8.4, 0.657),  # Nov
        (    259.7,     6.9, 0.806),  # Dec
    ],

    # ── Kuala Lumpur (3.1390°N, 101.6869°E) ─────────────────────────────────
    "kuala_lumpur": [
        #  R_mm_month  V_km  P_cloud
        (    222.0,     8.7, 0.632),  # Jan
        (    117.3,    11.0, 0.399),  # Feb
        (    222.7,    10.7, 0.432),  # Mar
        (    287.7,     8.4, 0.657),  # Apr
        (    273.6,     7.7, 0.726),  # May
        (    186.5,     8.7, 0.630),  # Jun
        (    181.4,     9.6, 0.542),  # Jul
        (    227.2,     8.9, 0.610),  # Aug
        (    253.3,     8.4, 0.663),  # Sep
        (    273.8,     7.3, 0.765),  # Oct
        (    380.5,     6.1, 0.893),  # Nov
        (    312.9,     7.3, 0.765),  # Dec
    ],
}

# Ground station coordinates (lat, lon, altitude_m)
CITY_COORDS: dict[str, tuple[float, float, float]] = {
    "hanoi":        (21.0285,   105.8542,  16.0),
    "hcmc":         (10.8231,   106.6297,  19.0),
    "danang":       (16.0544,   108.2022,  10.0),
    "bangkok":      (13.7563,   100.5018,   2.0),
    "singapore":    ( 1.3521,   103.8198,  15.0),
    "manila":       (14.5995,   120.9842,  16.0),
    "jakarta":      (-6.2088,   106.8456,   8.0),
    "kuala_lumpur": ( 3.1390,   101.6869,  68.0),
}


# ---------------------------------------------------------------------------
# 1. Unit conversion
# ---------------------------------------------------------------------------

def rain_mm_month_to_mm_h(
    R_mm_month: float,
    rain_fraction: float = RAIN_FRACTION_DEFAULT,
) -> float:
    """
    Chuyển đổi lượng mưa tháng (mm/month) sang rainfall rate (mm/h).

    Công thức:
        R_mm_h = R_mm_month / (30 × 24 × rain_fraction)

    Giải thích rain_fraction:
        Mưa nhiệt đới thường là mưa rào (convective), không mưa liên tục.
        rain_fraction = 0.15 → mưa ~3.6 h/ngày trong tháng mưa nhiều.
        Giá trị này phù hợp với khí hậu nhiệt đới ẩm (Paper 5, Abidjan).

    Parameters
    ----------
    R_mm_month   : lượng mưa tháng (mm/month)
    rain_fraction: tỷ lệ giờ có mưa trong tháng ∈ (0, 1]

    Returns
    -------
    R_mm_h : rainfall rate (mm/h)
    """
    if R_mm_month <= 0.0:
        return 0.0
    hours_per_month = 30.0 * 24.0
    return R_mm_month / (hours_per_month * rain_fraction)


# ---------------------------------------------------------------------------
# 2. Data access
# ---------------------------------------------------------------------------

def get_city_params(
    city: str,
    month: int,
    rain_fraction: float = RAIN_FRACTION_DEFAULT,
) -> dict:
    """
    Lấy tham số thời tiết trung bình cho một thành phố và tháng cụ thể.

    Parameters
    ----------
    city         : tên thành phố (không phân biệt hoa/thường).
                   Hỗ trợ: hanoi, hcmc, danang, bangkok, singapore,
                            manila, jakarta, kuala_lumpur
    month        : tháng (1–12)
    rain_fraction: tỷ lệ giờ có mưa (dùng để convert mm/month → mm/h)

    Returns
    -------
    dict
        R_mm_month : lượng mưa tháng (mm/month)
        R_mm_h     : rainfall rate (mm/h) — dùng cho channel_model
        V_km       : visibility (km) — dùng cho channel_model
        P_cloud    : xác suất mây dày (link OFF) ∈ [0, 1]
        city       : tên thành phố (normalized)
        month      : tháng (1–12)
        month_name : tên tháng (Jan, Feb, ...)
    """
    city_key = city.lower().replace(" ", "_").replace("-", "_")
    if city_key not in ASEAN_CLIMATE_DATA:
        raise ValueError(
            f"City '{city}' not found. Available: {list(ASEAN_CLIMATE_DATA.keys())}"
        )
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be 1–12, got {month}")

    R_mm_month, V_km, P_cloud = ASEAN_CLIMATE_DATA[city_key][month - 1]
    R_mm_h = rain_mm_month_to_mm_h(R_mm_month, rain_fraction)

    return {
        "R_mm_month": R_mm_month,
        "R_mm_h":     R_mm_h,
        "V_km":       V_km,
        "P_cloud":    P_cloud,
        "city":       city_key,
        "month":      month,
        "month_name": MONTHS[month - 1],
    }


def get_all_months(
    city: str,
    rain_fraction: float = RAIN_FRACTION_DEFAULT,
) -> list[dict]:
    """
    Lấy tham số thời tiết cho tất cả 12 tháng của một thành phố.

    Parameters
    ----------
    city         : tên thành phố
    rain_fraction: tỷ lệ giờ có mưa

    Returns
    -------
    list of 12 dicts (từ get_city_params), index 0 = tháng 1
    """
    return [get_city_params(city, m, rain_fraction) for m in range(1, 13)]


def list_cities() -> list[str]:
    """Trả về danh sách tên thành phố hỗ trợ."""
    return list(ASEAN_CLIMATE_DATA.keys())


# ---------------------------------------------------------------------------
# 3. Seasonal classification
# ---------------------------------------------------------------------------

def get_season(city: str, month: int) -> str:
    """
    Phân loại mùa (wet/dry) cho một thành phố và tháng.

    Dùng phân loại TƯƠNG ĐỐI: 6 tháng có R_mm_month cao nhất trong NĂM của
    chính thành phố đó = wet, 6 tháng thấp nhất = dry. KHÔNG dùng ngưỡng
    tuyệt đối (R > 100mm) — với thành phố mưa quanh năm như Singapore
    (R > 100mm cả 12 tháng theo dữ liệu Open-Meteo thật), ngưỡng tuyệt đối
    khiến nhóm "dry" rỗng, tạo ra kết luận sai "không có mùa khô" trong khi
    thực ra Singapore vẫn có 6 tháng tương đối khô hơn 6 tháng còn lại.

    Parameters
    ----------
    city  : tên thành phố
    month : tháng (1–12)

    Returns
    -------
    'wet' hoặc 'dry'
    """
    rain_by_month = [ASEAN_CLIMATE_DATA[city][m][0] for m in range(12)]
    order = sorted(range(12), key=lambda m: rain_by_month[m])  # ascending
    dry_months_idx = set(order[:6])
    return "dry" if (month - 1) in dry_months_idx else "wet"


def get_wet_months(city: str) -> list[int]:
    """Trả về 6 tháng mưa nhiều nhất (tương đối) trong năm của một thành phố."""
    return [m for m in range(1, 13) if get_season(city, m) == "wet"]


def get_dry_months(city: str) -> list[int]:
    """Trả về 6 tháng mưa ít nhất (tương đối) trong năm của một thành phố."""
    return [m for m in range(1, 13) if get_season(city, m) == "dry"]


# ---------------------------------------------------------------------------
# 4. Turbulence model selection
# ---------------------------------------------------------------------------

def get_turbulence_model(sigma_R2: float) -> str:
    """
    Chọn mô hình turbulence phù hợp dựa trên Rytov variance.

    Theo khuyến nghị từ Formula_Compendium_Part3_Weather.md và Paper 5:
        σR² < 0.3  → Log-normal (weak turbulence, như Paper 1 & 2)
        σR² ≥ 0.3  → Gamma-Gamma (moderate to strong)

    Parameters
    ----------
    sigma_R2 : Rytov variance (từ compute_rytov_variance)

    Returns
    -------
    'lognormal' hoặc 'gamma_gamma'
    """
    return "lognormal" if sigma_R2 < 0.3 else "gamma_gamma"


# ---------------------------------------------------------------------------
# 5. Seasonal hl computation (integrates with channel_model)
# ---------------------------------------------------------------------------

def compute_seasonal_hl(
    city: str,
    month: int,
    zeta_deg: float,
    lambda_nm: float = 1550.0,
    H_U_km: float = 0.0,
    rain_fraction: float = RAIN_FRACTION_DEFAULT,
) -> dict:
    """
    Tính hệ số suy hao khí quyển hl theo điều kiện thời tiết tháng.

    Gọi channel_model.compute_hl() với (R_mm_h, V_km) từ dữ liệu khí hậu.

    Parameters
    ----------
    city         : tên thành phố
    month        : tháng (1–12)
    zeta_deg     : zenith angle (degrees)
    lambda_nm    : wavelength (nm)
    H_U_km       : ground station altitude (km)
    rain_fraction: tỷ lệ giờ có mưa

    Returns
    -------
    dict
        hl         : atmospheric transmission ∈ (0, 1]
        hl_dB      : hl in dB (âm)
        R_mm_h     : rainfall rate used (mm/h)
        V_km       : visibility used (km)
        P_cloud    : cloud outage probability
        season     : 'wet' or 'dry'
        city, month, month_name
    """
    # Import here to avoid circular dependency
    from modules.channel_model import compute_hl as _compute_hl

    params = get_city_params(city, month, rain_fraction)
    hl = _compute_hl(
        zeta_deg=zeta_deg,
        V_km=params["V_km"],
        R_mm_h=params["R_mm_h"],
        lambda_nm=lambda_nm,
        H_U_km=H_U_km,
    )

    return {
        "hl":         hl,
        "hl_dB":      10.0 * np.log10(hl) if hl > 0 else -np.inf,
        "R_mm_h":     params["R_mm_h"],
        "V_km":       params["V_km"],
        "P_cloud":    params["P_cloud"],
        "season":     get_season(city, month),
        "city":       params["city"],
        "month":      params["month"],
        "month_name": params["month_name"],
    }


# ---------------------------------------------------------------------------
# 6. Availability analysis
# ---------------------------------------------------------------------------

def compute_link_availability(
    skr_norm: float,
    P_cloud: float,
    skr_threshold: float = 0.0,
) -> float:
    """
    Tính xác suất link available (SKR > threshold) cho một điều kiện thời tiết.

    Mô hình đơn giản hóa:
        P(available) = (1 - P_cloud) × P(SKR > threshold | no cloud)

    Vì SKR_norm đã được tính cho điều kiện không có mây (clear sky),
    P(SKR > threshold | no cloud) = 1 nếu skr_norm > threshold, else 0.

    Parameters
    ----------
    skr_norm      : normalized SKR (bit/s/Hz) tính cho clear sky
    P_cloud       : xác suất mây dày (link OFF)
    skr_threshold : ngưỡng SKR tối thiểu (default 0 = bất kỳ SKR > 0)

    Returns
    -------
    availability : P(link available) ∈ [0, 1]
    """
    p_clear_sky = 1.0 - P_cloud
    p_skr_ok = 1.0 if skr_norm > skr_threshold else 0.0
    return p_clear_sky * p_skr_ok


def compute_effective_skr(
    skr_norm: float,
    P_cloud: float,
) -> float:
    """
    Tính SKR hiệu dụng có tính đến xác suất mây che phủ (mô hình đơn trạng thái).

    SKR_effective = SKR_clear × (1 - P_cloud)

    LƯU Ý: Đây là mô hình đơn giản. Nếu skr_norm được tính với hl bao gồm
    mưa, kết quả sẽ đánh giá thấp SKR thực tế (~340× trong mùa mưa).
    Dùng compute_effective_skr_3state() cho kết quả chính xác hơn.

    Parameters
    ----------
    skr_norm : normalized SKR cho clear sky (bit/s/Hz)
    P_cloud  : xác suất mây dày

    Returns
    -------
    skr_effective : SKR hiệu dụng (bit/s/Hz)
    """
    return skr_norm * (1.0 - P_cloud)


def compute_effective_skr_3state(
    skr_clear: float,
    skr_rain: float,
    P_cloud: float,
    rain_fraction: float = RAIN_FRACTION_DEFAULT,
) -> dict:
    """
    Tính SKR hiệu dụng theo mô hình 3 trạng thái (chính xác hơn).

    3 trạng thái:
        1. Trời quang (p1 = 1 - P_cloud - rain_fraction): SKR = skr_clear
        2. Mưa (p2 = rain_fraction): SKR = skr_rain
        3. Mây dày (p3 = P_cloud): SKR = 0 (link OFF)

    SKR_effective = p1 × skr_clear + p2 × skr_rain

    Parameters
    ----------
    skr_clear     : SKR khi trời quang (R=0, V=V_dry), kbps hoặc bit/s/Hz
    skr_rain      : SKR khi mưa (R=R_peak, V=V_rain), cùng đơn vị
    P_cloud       : xác suất mây dày ∈ [0, 1]
    rain_fraction : tỷ lệ thời gian mưa ∈ (0, 1]

    Returns
    -------
    dict
        skr_effective : SKR hiệu dụng (trung bình có trọng số)
        p_clear       : xác suất trời quang
        p_rain        : xác suất mưa
        p_cloud       : xác suất mây
        skr_clear     : SKR trời quang (input)
        skr_rain      : SKR mưa (input)
    """
    p_rain = min(rain_fraction, 1.0 - P_cloud)
    p_clear = max(0.0, 1.0 - P_cloud - p_rain)
    p_cloud = P_cloud

    skr_eff = p_clear * skr_clear + p_rain * skr_rain

    return {
        "skr_effective": skr_eff,
        "p_clear": p_clear,
        "p_rain": p_rain,
        "p_cloud": p_cloud,
        "skr_clear": skr_clear,
        "skr_rain": skr_rain,
    }


def compute_annual_stats(
    city: str,
    skr_by_month: list[float],
) -> dict:
    """
    Tính thống kê SKR hàng năm cho một thành phố.

    Parameters
    ----------
    city         : tên thành phố
    skr_by_month : list 12 giá trị SKR_norm (tháng 1–12), clear sky

    Returns
    -------
    dict
        skr_annual_mean      : SKR trung bình năm (clear sky)
        skr_effective_mean   : SKR hiệu dụng trung bình (có tính mây)
        availability_mean    : availability trung bình năm
        skr_wet_mean         : SKR trung bình mùa mưa
        skr_dry_mean         : SKR trung bình mùa khô
        wet_months           : danh sách tháng mùa mưa
        dry_months           : danh sách tháng mùa khô
    """
    if len(skr_by_month) != 12:
        raise ValueError(f"skr_by_month must have 12 elements, got {len(skr_by_month)}")

    all_months_data = get_all_months(city)
    p_clouds = [d["P_cloud"] for d in all_months_data]

    skr_effective = [
        compute_effective_skr(skr, pc)
        for skr, pc in zip(skr_by_month, p_clouds)
    ]
    availability = [
        compute_link_availability(skr, pc)
        for skr, pc in zip(skr_by_month, p_clouds)
    ]

    wet_months = get_wet_months(city)
    dry_months = get_dry_months(city)

    skr_wet = [skr_by_month[m - 1] for m in wet_months]
    skr_dry = [skr_by_month[m - 1] for m in dry_months]

    return {
        "skr_annual_mean":    float(np.mean(skr_by_month)),
        "skr_effective_mean": float(np.mean(skr_effective)),
        "availability_mean":  float(np.mean(availability)),
        "skr_wet_mean":       float(np.mean(skr_wet)) if skr_wet else 0.0,
        "skr_dry_mean":       float(np.mean(skr_dry)) if skr_dry else 0.0,
        "wet_months":         wet_months,
        "dry_months":         dry_months,
    }


# ============================================================
# RELIABILITY METRICS (for GLOBECOM WS-03 paper)
# ============================================================

def compute_reliability_metrics(city, month, skr_eff_kbps=None, rain_fraction=0.15):
    """
    Compute formal reliability metrics for a given city/month.

    Parameters
    ----------
    city : str
        City key (e.g. 'hanoi', 'hcmc')
    month : int
        Month number (1-12)
    skr_eff_kbps : float, optional
        Effective SKR in kbps (if None, K_day not computed)
    rain_fraction : float
        Fraction of non-cloud time that is rainy (default 0.15)

    Returns
    -------
    dict with keys:
        A       — Link availability = 1 - P_cloud
        P_out   — Outage probability = P_cloud
        K_day   — Daily key delivery in bits (None if skr_eff_kbps not given)
        p_clear — Probability of clear state
        p_rain  — Probability of rain state
        p_cloud — Probability of cloud/outage state
    """
    params = get_city_params(city, month, rain_fraction=rain_fraction)
    P_cloud = params['P_cloud']

    p_rain = min(rain_fraction, 1.0 - P_cloud)
    p_clear = max(0.0, 1.0 - P_cloud - p_rain)

    A = p_clear + p_rain  # = 1 - P_cloud
    P_out = P_cloud

    K_day = None
    if skr_eff_kbps is not None:
        K_day = skr_eff_kbps * 1000.0 * 86400.0  # bits/day

    return {
        "A": A,
        "P_out": P_out,
        "K_day": K_day,
        "p_clear": p_clear,
        "p_rain": p_rain,
        "p_cloud": P_cloud,
    }
