"""
weather_model.py — ASEAN Climatological Weather Data for FSO/SIKD
==================================================================

Cung cấp dữ liệu thời tiết trung bình theo tháng (climatological averages)
cho 8 thành phố ASEAN, dùng để tính ảnh hưởng thời tiết lên kênh FSO.

Chiến lược thiết kế
-------------------
Không dùng real-time API. Dùng climatological monthly averages — đây là
chuẩn trong FSO research (Paper 5, Koné 2024 dùng cùng phương pháp).

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

Nguồn dữ liệu
-------------
    - Rainfall: Formula_Compendium_Part3_Weather.md (từ Paper 5, Koné 2024)
      + ước lượng khí hậu học nhiệt đới cho ASEAN
    - Visibility: ước lượng từ khí hậu học (mùa khô ~12–15 km,
      mùa mưa ~5–8 km cho khí hậu nhiệt đới ẩm)
    - Cloud outage: ước lượng từ thống kê mây nhiệt đới
      (mùa mưa ~50–65%, mùa khô ~15–25%)
    - rain_fraction: ~0.15 (mưa ~3.6h/ngày trong mùa mưa nhiệt đới)

Thành phố hỗ trợ
----------------
    Vietnam  : hanoi, hcmc, danang
    ASEAN    : bangkok, singapore, manila, jakarta, kuala_lumpur

References
----------
    [P5] Koné et al., IJP 2024 — tropical FSO, Abidjan (khí hậu tương tự VN)
    [P3] Toka et al., Computer Networks 2025 — weather-adaptive routing
    [P4] Potter et al., JPL/NASA 1969 — cloud = complete blockage at optical
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

ASEAN_CLIMATE_DATA: dict[str, list[tuple[float, float, float]]] = {

    # ── Hà Nội (21.03°N, 105.85°E) ──────────────────────────────────────
    # Mùa mưa: tháng 5–10 (mưa nhiều, tầm nhìn kém)
    # Mùa khô: tháng 11–4 (ít mưa, sương mù nhẹ tháng 1–3)
    "hanoi": [
        #  R_mm_month  V_km  P_cloud
        (   5,         12,   0.20),  # Jan — khô, sương mù nhẹ
        (  10,         12,   0.22),  # Feb
        (  20,         11,   0.28),  # Mar — bắt đầu ẩm
        (  50,         10,   0.32),  # Apr
        ( 150,          8,   0.50),  # May — bắt đầu mùa mưa
        ( 200,          7,   0.58),  # Jun — đỉnh mưa
        ( 180,          7,   0.55),  # Jul
        ( 160,          8,   0.52),  # Aug
        ( 200,          7,   0.58),  # Sep
        ( 150,          8,   0.50),  # Oct
        (  80,         10,   0.38),  # Nov — cuối mùa mưa
        (  20,         13,   0.22),  # Dec — khô
    ],

    # ── TP. Hồ Chí Minh (10.82°N, 106.63°E) ─────────────────────────────
    # Mùa mưa: tháng 5–11 (mưa nhiều, đặc trưng nhiệt đới)
    # Mùa khô: tháng 12–4 (ít mưa, tầm nhìn tốt)
    "hcmc": [
        #  R_mm_month  V_km  P_cloud
        (   5,         15,   0.15),  # Jan — khô nhất
        (   5,         15,   0.15),  # Feb
        (  10,         14,   0.18),  # Mar
        (  40,         12,   0.28),  # Apr — bắt đầu mưa
        ( 180,          7,   0.55),  # May
        ( 250,          6,   0.62),  # Jun — đỉnh mưa
        ( 280,          6,   0.65),  # Jul — đỉnh mưa
        ( 260,          6,   0.63),  # Aug
        ( 280,          6,   0.65),  # Sep
        ( 240,          7,   0.60),  # Oct
        ( 120,          9,   0.45),  # Nov
        (  30,         13,   0.20),  # Dec
    ],

    # ── Đà Nẵng (16.05°N, 108.20°E) ─────────────────────────────────────
    # Mùa mưa: tháng 9–12 (mưa lớn, bão)
    # Mùa khô: tháng 1–8 (khô, nóng)
    "danang": [
        #  R_mm_month  V_km  P_cloud
        (  50,         12,   0.30),  # Jan — cuối mùa mưa
        (  20,         13,   0.22),  # Feb
        (  15,         14,   0.18),  # Mar
        (  20,         14,   0.18),  # Apr
        (  40,         13,   0.25),  # May
        (  50,         12,   0.28),  # Jun
        (  50,         12,   0.28),  # Jul
        (  60,         11,   0.32),  # Aug
        ( 200,          7,   0.55),  # Sep — bắt đầu mùa mưa
        ( 350,          5,   0.68),  # Oct — đỉnh mưa
        ( 300,          6,   0.65),  # Nov
        ( 150,          8,   0.52),  # Dec
    ],

    # ── Bangkok (13.75°N, 100.52°E) ──────────────────────────────────────
    # Mùa mưa: tháng 5–10
    # Mùa khô: tháng 11–4
    "bangkok": [
        #  R_mm_month  V_km  P_cloud
        (  10,         15,   0.15),  # Jan
        (  15,         15,   0.15),  # Feb
        (  30,         13,   0.20),  # Mar
        (  60,         11,   0.30),  # Apr
        ( 180,          8,   0.52),  # May
        ( 150,          8,   0.50),  # Jun
        ( 140,          8,   0.48),  # Jul
        ( 160,          7,   0.52),  # Aug
        ( 200,          7,   0.58),  # Sep
        ( 180,          7,   0.55),  # Oct
        (  50,         11,   0.30),  # Nov
        (  10,         14,   0.18),  # Dec
    ],

    # ── Singapore (1.35°N, 103.82°E) ─────────────────────────────────────
    # Mưa quanh năm, không có mùa khô rõ rệt
    # Đỉnh mưa: tháng 11–1 (Northeast Monsoon)
    "singapore": [
        #  R_mm_month  V_km  P_cloud
        ( 230,          8,   0.55),  # Jan — Northeast Monsoon
        ( 150,          9,   0.48),  # Feb
        ( 170,          9,   0.48),  # Mar
        ( 160,          9,   0.48),  # Apr
        ( 170,          9,   0.48),  # May
        ( 130,         10,   0.42),  # Jun
        ( 150,         10,   0.42),  # Jul
        ( 150,         10,   0.42),  # Aug
        ( 160,          9,   0.45),  # Sep
        ( 170,          9,   0.48),  # Oct
        ( 250,          8,   0.58),  # Nov
        ( 280,          7,   0.62),  # Dec
    ],

    # ── Manila (14.60°N, 120.98°E) ───────────────────────────────────────
    # Mùa mưa: tháng 6–11 (bão nhiều)
    # Mùa khô: tháng 12–5
    "manila": [
        #  R_mm_month  V_km  P_cloud
        (  10,         15,   0.15),  # Jan
        (  10,         15,   0.15),  # Feb
        (  15,         14,   0.18),  # Mar
        (  20,         14,   0.18),  # Apr
        (  90,         10,   0.38),  # May
        ( 250,          6,   0.62),  # Jun
        ( 350,          5,   0.68),  # Jul — đỉnh mưa + bão
        ( 350,          5,   0.68),  # Aug
        ( 300,          5,   0.65),  # Sep
        ( 200,          7,   0.55),  # Oct
        ( 100,          9,   0.40),  # Nov
        (  30,         13,   0.22),  # Dec
    ],

    # ── Jakarta (6.21°S, 106.85°E) ───────────────────────────────────────
    # Mùa mưa: tháng 11–4 (ngược với VN)
    # Mùa khô: tháng 5–10
    "jakarta": [
        #  R_mm_month  V_km  P_cloud
        ( 300,          6,   0.65),  # Jan — đỉnh mưa
        ( 280,          6,   0.62),  # Feb
        ( 220,          7,   0.58),  # Mar
        ( 150,          8,   0.50),  # Apr
        ( 100,         10,   0.38),  # May
        (  60,         12,   0.28),  # Jun
        (  40,         13,   0.22),  # Jul — khô nhất
        (  40,         13,   0.22),  # Aug
        (  60,         12,   0.28),  # Sep
        ( 100,         10,   0.38),  # Oct
        ( 150,          8,   0.50),  # Nov
        ( 250,          7,   0.60),  # Dec
    ],

    # ── Kuala Lumpur (3.14°N, 101.69°E) ─────────────────────────────────
    # Mưa quanh năm, hai đỉnh: tháng 4–5 và tháng 10–11
    "kuala_lumpur": [
        #  R_mm_month  V_km  P_cloud
        ( 160,          9,   0.48),  # Jan
        ( 150,          9,   0.45),  # Feb
        ( 200,          8,   0.52),  # Mar
        ( 250,          7,   0.60),  # Apr — đỉnh 1
        ( 200,          8,   0.52),  # May
        ( 120,         10,   0.40),  # Jun
        ( 100,         11,   0.35),  # Jul
        ( 130,         10,   0.40),  # Aug
        ( 160,          9,   0.48),  # Sep
        ( 250,          7,   0.60),  # Oct — đỉnh 2
        ( 280,          7,   0.62),  # Nov
        ( 200,          8,   0.52),  # Dec
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

    Dựa trên ngưỡng lượng mưa: R_mm_month > 100 mm → wet season.

    Parameters
    ----------
    city  : tên thành phố
    month : tháng (1–12)

    Returns
    -------
    'wet' hoặc 'dry'
    """
    params = get_city_params(city, month)
    return "wet" if params["R_mm_month"] > 100.0 else "dry"


def get_wet_months(city: str) -> list[int]:
    """Trả về danh sách các tháng mùa mưa (R > 100 mm/month) cho một thành phố."""
    return [m for m in range(1, 13) if get_season(city, m) == "wet"]


def get_dry_months(city: str) -> list[int]:
    """Trả về danh sách các tháng mùa khô (R ≤ 100 mm/month) cho một thành phố."""
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
