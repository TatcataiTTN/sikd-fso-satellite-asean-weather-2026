"""
routing.py — Weather-Adaptive Satellite Routing for SIKD/FSO
=============================================================

Chọn vệ tinh tối ưu và tính toán hiệu năng routing cho hệ thống
SIKD trên vùng ASEAN, kết hợp orbital mechanics + weather + SIKD performance.

Chiến lược routing
------------------
    Greedy (mặc định):
        Tại mỗi time step, chọn vệ tinh có SKR cao nhất trong số các
        vệ tinh visible. Đơn giản, hiệu quả, phù hợp cho bài báo hội nghị.

    No-routing (baseline):
        Chọn vệ tinh đầu tiên trong danh sách (không tối ưu).
        Dùng để so sánh và tính improvement ratio.

Pipeline tích hợp
-----------------
    orbital_mechanics  → elevation, zenith, slant_km
    channel_model      → hg, hl, sigma_X2
    weather_model      → R_mm_h, V_km, P_cloud
    sikd_performance   → SKR_norm, QBER, BER_CC
    routing            → select_best_satellite, compute_daily_key_bits,
                          compute_routing_improvement

Cấu trúc module
---------------
    Nhóm 1 — SKR per satellite : compute_skr_for_satellite,
                                   compute_skr_all_visible
    Nhóm 2 — Routing strategies: greedy_best_satellite,
                                   no_routing_satellite
    Nhóm 3 — Time series        : compute_skr_timeseries,
                                   compute_daily_key_bits
    Nhóm 4 — Analysis           : compute_routing_improvement,
                                   compute_seasonal_routing_stats

References
----------
    [P2] Vu, PTIT 2025 — SIKD system model
    [P3] Toka et al., Computer Networks 2025 — weather-adaptive routing LEO
"""

import numpy as np
from datetime import datetime, timezone
from typing import Optional

from skyfield.api import EarthSatellite

# ---------------------------------------------------------------------------
# Default system parameters (Paper 2 Table)
# ---------------------------------------------------------------------------
PT_DEFAULT       = 1.0      # W    — transmit power (Paper 2 Table I: 30 dBm = 1 W)
MK_DEFAULT       = 0.05    # —    — QKD modulation index
MD_DEFAULT       = 0.5     # —    — data channel modulation index
ISO_DB_DEFAULT   = 15.0    # dB   — BPF filter isolation (Paper 2 Table I)
ZETA_SCALE_DEF   = 2.0     # —    — DT threshold scale
RB_DEFAULT       = 1e9     # bps  — raw bit rate
H_S_KM_DEFAULT   = 500.0   # km   — satellite altitude (Paper 2 Table I)


# ---------------------------------------------------------------------------
# 1. SKR per satellite
# ---------------------------------------------------------------------------

def compute_skr_for_satellite(
    zenith_deg: float,
    R_mm_h: float,
    V_km: float,
    P_cloud: float,
    PT: float = PT_DEFAULT,
    mK: float = MK_DEFAULT,
    mD: float = MD_DEFAULT,
    Iso_dB: float = ISO_DB_DEFAULT,
    zeta_scale: float = ZETA_SCALE_DEF,
    Rb: float = RB_DEFAULT,
    H_S_km: float = H_S_KM_DEFAULT,
) -> dict:
    """
    Tính SKR cho một vệ tinh tại một điều kiện kênh và thời tiết cụ thể.

    Pipeline:
        zenith_deg + H_S_km → channel_model.compute_channel()
        R_mm_h, V_km        → channel_model.compute_hl() (override hl)
        hg, hl, sigma_X2    → sikd_performance.compute_sikd_performance()
        SKR_norm × (1-P_cloud) → SKR_effective

    Parameters
    ----------
    zenith_deg  : zenith angle (degrees), từ orbital_mechanics
    R_mm_h      : rainfall rate (mm/h), từ weather_model
    V_km        : visibility (km), từ weather_model
    P_cloud     : cloud outage probability ∈ [0, 1]
    PT          : transmit power (W)
    mK          : QKD modulation index
    mD          : data channel modulation index
    Iso_dB      : BPF filter isolation (dB)
    zeta_scale  : DT threshold scale coefficient
    Rb          : raw bit rate (bps)
    H_S_km      : satellite altitude (km)

    Returns
    -------
    dict
        SKR_norm      : normalized SKR clear-sky (bit/s/Hz)
        SKR_kbps      : SKR clear-sky (kbps)
        SKR_effective : SKR × (1 - P_cloud) — thực tế có tính mây
        QBER          : quantum bit error rate
        BER_CC        : classical channel BER
        hl            : atmospheric transmission (weather-dependent)
        hg            : geometric loss
        sigma_X2      : log-amplitude variance
        zenith_deg    : zenith angle used
        is_feasible   : True nếu SKR_norm > 0
    """
    from modules.channel_model import compute_channel, compute_hl
    from modules.sikd_performance import compute_sikd_performance

    # Geometric + turbulence từ zenith angle
    ch = compute_channel(H_S_km, zenith_deg)
    hg       = ch["hg"]
    sigma_X2 = ch["sigma_X2"]

    # 3-state weather model:
    #   State 1 (clear): R=0, V=V_km → hl_clear
    #   State 2 (rain):  R=R_mm_h, V=V_km → hl_rain
    #   State 3 (cloud): link OFF → SKR=0
    # Probabilities: p_rain=0.15 (of non-cloud time), p_cloud, p_clear=1-P_cloud-p_rain
    RAIN_FRAC = 0.15

    hl_clear = compute_hl(zeta_deg=zenith_deg, V_km=V_km, R_mm_h=0.0)
    hl_rain  = compute_hl(zeta_deg=zenith_deg, V_km=V_km, R_mm_h=R_mm_h)

    # SKR for clear state
    perf_clear = compute_sikd_performance(
        hg=hg, hl=hl_clear, sigma_X2=sigma_X2,
        PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_scale, Rb=Rb,
    )
    # SKR for rain state
    perf_rain = compute_sikd_performance(
        hg=hg, hl=hl_rain, sigma_X2=sigma_X2,
        PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_scale, Rb=Rb,
    )

    # Weighted average (expected value)
    p_rain  = min(RAIN_FRAC, 1.0 - P_cloud)
    p_clear = max(0.0, 1.0 - P_cloud - p_rain)

    skr_eff = (p_clear * perf_clear["SKR_norm"]
               + p_rain * perf_rain["SKR_norm"])

    # Use clear-sky performance for reporting (dominant state)
    perf = perf_clear

    return {
        "SKR_norm":      perf["SKR_norm"],
        "SKR_kbps":      perf["SKR_kbps"],
        "SKR_effective": skr_eff,
        "QBER":          perf["QBER"],
        "BER_CC":        perf["BER_CC"],
        "hl":            hl_clear,
        "hg":            hg,
        "sigma_X2":      sigma_X2,
        "zenith_deg":    zenith_deg,
        "is_feasible":   perf["SKR_norm"] > 0.0,
    }


def compute_skr_all_visible(
    visible_sats: list[dict],
    R_mm_h: float,
    V_km: float,
    P_cloud: float,
    PT: float = PT_DEFAULT,
    mK: float = MK_DEFAULT,
    mD: float = MD_DEFAULT,
    Iso_dB: float = ISO_DB_DEFAULT,
    zeta_scale: float = ZETA_SCALE_DEF,
    Rb: float = RB_DEFAULT,
) -> list[dict]:
    """
    Tính SKR cho tất cả vệ tinh visible tại một time step.

    Parameters
    ----------
    visible_sats : list of dicts từ orbital_mechanics.get_visible_satellites()
                   Mỗi dict phải có 'zenith_deg' và 'name'
    R_mm_h, V_km, P_cloud : weather params từ weather_model
    PT, mK, mD, Iso_dB, zeta_scale, Rb : system params

    Returns
    -------
    list of dicts, mỗi dict = visible_sat_info + SKR results,
    sorted by SKR_effective descending
    """
    results = []
    for sat_info in visible_sats:
        skr_result = compute_skr_for_satellite(
            zenith_deg=sat_info["zenith_deg"],
            R_mm_h=R_mm_h,
            V_km=V_km,
            P_cloud=P_cloud,
            PT=PT, mK=mK, mD=mD,
            Iso_dB=Iso_dB, zeta_scale=zeta_scale, Rb=Rb,
        )
        entry = {**sat_info, **skr_result}
        results.append(entry)

    results.sort(key=lambda x: x["SKR_effective"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 2. Routing strategies
# ---------------------------------------------------------------------------

def greedy_best_satellite(
    skr_candidates: list[dict],
) -> Optional[dict]:
    """
    Greedy routing: chọn vệ tinh có SKR_effective cao nhất.

    Đây là chiến lược routing đơn giản nhất — tại mỗi time step,
    chọn vệ tinh tốt nhất hiện tại mà không xét lịch sử hay tương lai.

    Parameters
    ----------
    skr_candidates : list of dicts từ compute_skr_all_visible()
                     (đã sorted by SKR_effective descending)

    Returns
    -------
    dict của vệ tinh được chọn, hoặc None nếu không có vệ tinh visible
    """
    if not skr_candidates:
        return None
    return skr_candidates[0]


def no_routing_satellite(
    skr_candidates: list[dict],
) -> Optional[dict]:
    """
    Baseline (no routing): chọn vệ tinh đầu tiên trong danh sách visible.

    Trong compute_skr_timeseries, baseline thực sự được xử lý bởi
    fixed_satellite_logic (track 1 vệ tinh cho đến khi lặn).

    Hàm này chỉ dùng cho single-step comparison:
    chọn vệ tinh đầu tiên theo thứ tự TLE (không tối ưu).

    Parameters
    ----------
    skr_candidates : list of dicts từ compute_skr_all_visible()

    Returns
    -------
    dict của vệ tinh được chọn (đầu tiên theo tên), hoặc None
    """
    if not skr_candidates:
        return None
    # Sort by name để có thứ tự deterministic
    sorted_by_name = sorted(skr_candidates, key=lambda x: x["name"])
    return sorted_by_name[0]


# ---------------------------------------------------------------------------
# 3. Time series
# ---------------------------------------------------------------------------

def compute_skr_timeseries(
    satellites: list[EarthSatellite],
    lat_deg: float,
    lon_deg: float,
    t_start: datetime,
    R_mm_h: float,
    V_km: float,
    P_cloud: float,
    duration_hours: float = 24.0,
    step_minutes: float = 1.0,
    alt_m: float = 0.0,
    min_elevation: float = 30.0,
    PT: float = PT_DEFAULT,
    mK: float = MK_DEFAULT,
    mD: float = MD_DEFAULT,
    Iso_dB: float = ISO_DB_DEFAULT,
    zeta_scale: float = ZETA_SCALE_DEF,
    Rb: float = RB_DEFAULT,
) -> dict:
    """
    Tính SKR theo thời gian (time series) cho một trạm mặt đất.

    Tại mỗi time step:
        1. Tìm tất cả vệ tinh visible (elevation ≥ min_elevation)
        2. Tính SKR cho từng vệ tinh
        3. Greedy: chọn vệ tinh SKR cao nhất
        4. No-routing: chọn vệ tinh đầu tiên (baseline)

    Parameters
    ----------
    satellites     : list of EarthSatellite
    lat_deg        : vĩ độ trạm (degrees)
    lon_deg        : kinh độ trạm (degrees)
    t_start        : thời điểm bắt đầu (datetime UTC)
    R_mm_h         : rainfall rate (mm/h) — constant trong simulation
    V_km           : visibility (km) — constant trong simulation
    P_cloud        : cloud outage probability
    duration_hours : tổng thời gian (giờ)
    step_minutes   : bước thời gian (phút)
    alt_m          : độ cao trạm (m)
    min_elevation  : ngưỡng elevation tối thiểu (degrees)
    PT, mK, mD, Iso_dB, zeta_scale, Rb : system params

    Returns
    -------
    dict
        skr_greedy    : np.ndarray SKR_effective theo thời gian (greedy)
        skr_baseline  : np.ndarray SKR_effective theo thời gian (no-routing)
        n_visible     : np.ndarray số vệ tinh visible tại mỗi step
        time_hours    : np.ndarray thời gian (giờ từ t_start)
        n_steps       : tổng số time steps
    """
    from modules.orbital_mechanics import (
        make_time_array, get_visible_satellites,
    )

    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=timezone.utc)

    t_array = make_time_array(t_start, duration_hours, step_minutes)
    n_steps = len(t_array)

    skr_greedy   = np.zeros(n_steps)
    skr_baseline = np.zeros(n_steps)
    n_visible    = np.zeros(n_steps, dtype=int)

    for i in range(n_steps):
        t_i = t_array[i]

        visible = get_visible_satellites(
            satellites, lat_deg, lon_deg, t_i, alt_m, min_elevation
        )
        n_visible[i] = len(visible)

        if not visible:
            # No satellite visible → gap for both strategies
            continue

        candidates = compute_skr_all_visible(
            visible, R_mm_h, V_km, P_cloud,
            PT, mK, mD, Iso_dB, zeta_scale, Rb,
        )

        # --- GREEDY: chọn vệ tinh có SKR_effective cao nhất (weather-aware),
        # re-evaluate mỗi step ---
        best = greedy_best_satellite(candidates)
        skr_greedy[i] = best["SKR_effective"] if best else 0.0

        # --- BASELINE (elevation-priority): chọn vệ tinh có zenith angle
        # thấp nhất (elevation cao nhất) tại MỖI step, KHÔNG xét thời tiết.
        # Re-evaluate mỗi step giống greedy — sự khác biệt DUY NHẤT giữa
        # 2 chiến lược là TIÊU CHÍ chọn (SKR_effective vs zenith), không
        # phải tần suất handover. Khớp đúng công thức trong paper:
        # s_base(t) = argmin_s zeta_s(t). (Trước đây baseline "dính" 1 vệ
        # tinh đến khi lặn — với constellation mật độ cao, cách đó làm
        # baseline tụt giả tạo và phóng đại % cải thiện của greedy.)
        baseline_choice = min(candidates, key=lambda c: c["zenith_deg"])
        skr_baseline[i] = baseline_choice["SKR_effective"]

    time_hours = np.arange(n_steps) * step_minutes / 60.0

    return {
        "skr_greedy":   skr_greedy,
        "skr_baseline": skr_baseline,
        "n_visible":    n_visible,
        "time_hours":   time_hours,
        "n_steps":      n_steps,
    }


def compute_daily_key_bits(
    skr_timeseries: np.ndarray,
    step_minutes: float = 1.0,
    Rb: float = RB_DEFAULT,
) -> float:
    """
    Tính tổng số key bits tích lũy trong một ngày từ SKR time series.

    Total_key_bits = Σ SKR_norm(t) × Rb × Δt

    Parameters
    ----------
    skr_timeseries : np.ndarray SKR_norm (bit/s/Hz) theo thời gian
    step_minutes   : bước thời gian (phút)
    Rb             : raw bit rate (bps)

    Returns
    -------
    total_key_bits : tổng số key bits (bits)
    """
    dt_seconds = step_minutes * 60.0
    return float(np.sum(skr_timeseries) * Rb * dt_seconds)


# ---------------------------------------------------------------------------
# 4. Analysis
# ---------------------------------------------------------------------------

def compute_routing_improvement(
    skr_greedy: np.ndarray,
    skr_baseline: np.ndarray,
    step_minutes: float = 1.0,
    Rb: float = RB_DEFAULT,
) -> dict:
    """
    Tính các metrics so sánh greedy routing vs no-routing baseline.

    Parameters
    ----------
    skr_greedy   : np.ndarray SKR_effective (greedy), từ compute_skr_timeseries
    skr_baseline : np.ndarray SKR_effective (baseline)
    step_minutes : bước thời gian (phút)
    Rb           : raw bit rate (bps)

    Returns
    -------
    dict
        mean_skr_greedy    : SKR trung bình greedy (bit/s/Hz)
        mean_skr_baseline  : SKR trung bình baseline
        improvement_ratio  : mean_greedy / mean_baseline (≥ 1)
        improvement_pct    : (ratio - 1) × 100 %
        key_bits_greedy    : tổng key bits greedy (bits)
        key_bits_baseline  : tổng key bits baseline
        key_bits_gain      : key_bits_greedy - key_bits_baseline
        availability_greedy  : fraction of time steps với SKR > 0
        availability_baseline: fraction of time steps với SKR > 0
    """
    mean_g = float(np.mean(skr_greedy))
    mean_b = float(np.mean(skr_baseline))

    ratio = mean_g / mean_b if mean_b > 0 else np.inf

    kb_g = compute_daily_key_bits(skr_greedy,   step_minutes, Rb)
    kb_b = compute_daily_key_bits(skr_baseline, step_minutes, Rb)

    avail_g = float(np.mean(skr_greedy   > 0))
    avail_b = float(np.mean(skr_baseline > 0))

    return {
        "mean_skr_greedy":      mean_g,
        "mean_skr_baseline":    mean_b,
        "improvement_ratio":    ratio,
        "improvement_pct":      (ratio - 1.0) * 100.0,
        "key_bits_greedy":      kb_g,
        "key_bits_baseline":    kb_b,
        "key_bits_gain":        kb_g - kb_b,
        "availability_greedy":  avail_g,
        "availability_baseline": avail_b,
    }


def compute_seasonal_routing_stats(
    city: str,
    satellites: list[EarthSatellite],
    lat_deg: float,
    lon_deg: float,
    t_start: datetime,
    months: list[int],
    duration_hours: float = 24.0,
    step_minutes: float = 5.0,
    alt_m: float = 0.0,
    zeta_deg_fixed: float = 45.0,
    **sikd_kwargs,
) -> dict:
    """
    Tính routing stats theo mùa (wet vs dry) cho một thành phố.

    Với mỗi tháng trong danh sách, lấy weather params từ weather_model
    và tính routing improvement.

    Parameters
    ----------
    city           : tên thành phố (dùng weather_model.get_city_params)
    satellites     : list of EarthSatellite
    lat_deg        : vĩ độ trạm
    lon_deg        : kinh độ trạm
    t_start        : thời điểm bắt đầu simulation
    months         : list tháng cần tính (1–12)
    duration_hours : thời gian simulation mỗi tháng (giờ)
    step_minutes   : bước thời gian (phút)
    alt_m          : độ cao trạm (m)
    zeta_deg_fixed : zenith angle cố định (dùng khi không có orbital data)
    **sikd_kwargs  : tham số hệ thống SIKD (PT, mK, mD, Iso_dB, ...)

    Returns
    -------
    dict
        monthly_stats : list of dicts, mỗi dict chứa routing stats cho 1 tháng
        wet_improvement_pct  : improvement % trung bình mùa mưa
        dry_improvement_pct  : improvement % trung bình mùa khô
    """
    from modules.weather_model import get_city_params, get_season

    monthly_stats = []

    for month in months:
        wx = get_city_params(city, month)
        ts = compute_skr_timeseries(
            satellites=satellites,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            t_start=t_start,
            R_mm_h=wx["R_mm_h"],
            V_km=wx["V_km"],
            P_cloud=wx["P_cloud"],
            duration_hours=duration_hours,
            step_minutes=step_minutes,
            alt_m=alt_m,
            **sikd_kwargs,
        )
        imp = compute_routing_improvement(
            ts["skr_greedy"], ts["skr_baseline"], step_minutes
        )
        monthly_stats.append({
            "month":       month,
            "season":      get_season(city, month),
            "R_mm_h":      wx["R_mm_h"],
            "V_km":        wx["V_km"],
            "P_cloud":     wx["P_cloud"],
            **imp,
        })

    wet_pcts = [s["improvement_pct"] for s in monthly_stats if s["season"] == "wet"]
    dry_pcts = [s["improvement_pct"] for s in monthly_stats if s["season"] == "dry"]

    return {
        "monthly_stats":         monthly_stats,
        "wet_improvement_pct":   float(np.mean(wet_pcts))  if wet_pcts else 0.0,
        "dry_improvement_pct":   float(np.mean(dry_pcts))  if dry_pcts else 0.0,
    }
