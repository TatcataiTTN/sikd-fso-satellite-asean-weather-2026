#!/usr/bin/env python3
"""
debug_comprehensive.py — Kiểm tra toàn bộ pipeline SIKD/FSO
Chạy độc lập, in ra từng giá trị trung gian với precision cao.
So sánh với expected values từ Paper 1/2.
Flag bất kỳ kết quả vô lý nào.

Usage:
    cd "05_Code"
    python debug_comprehensive.py
"""

import sys
import os
import numpy as np

# Add modules path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

from channel_model import (
    compute_beam_waist, compute_beam_radius, compute_hg,
    compute_sigma_visibility, compute_sigma_rain,
    compute_hl, compute_Cn2_HV, compute_rytov_variance,
    compute_channel
)
from sikd_performance import (
    compute_noise, compute_thresholds,
    compute_Psift_QBER, compute_SKR, compute_SKR_bps,
    compute_BER_CC, compute_sikd_performance
)
from weather_model import (
    get_city_params, compute_effective_skr_3state,
    ASEAN_CLIMATE_DATA, rain_mm_month_to_mm_h
)

# ============================================================
# CONFIGURATION
# ============================================================
results = {"pass": 0, "fail": 0, "warn": 0}

def check(name, value, expected_min, expected_max, unit=""):
    """Check if value is within expected range."""
    if expected_min <= value <= expected_max:
        results["pass"] += 1
        status = "PASS"
    else:
        results["fail"] += 1
        status = "FAIL"
    print(f"  [{status}] {name}: {value:.8g} {unit} "
          f"(expected: [{expected_min:.6g}, {expected_max:.6g}])")
    return expected_min <= value <= expected_max

def warn(name, value, condition_desc):
    """Flag a warning."""
    results["warn"] += 1
    print(f"  [WARN] {name}: {value:.8g} -- {condition_desc}")


# ============================================================
# TEST GROUP 1: CHANNEL MODEL — Geometric Loss
# ============================================================
print("\n" + "="*70)
print("GROUP 1: CHANNEL MODEL — Geometric Loss (hg)")
print("="*70)

# Paper 1/2 parameters
H_S_km = 500.0
zeta_deg = 30.0
a_R = 0.05  # 5 cm receiver aperture
lambda_nm = 1550.0
theta_C_urad = 10.0

print(f"\n  Parameters: H_S={H_S_km} km, zeta={zeta_deg} deg, a_R={a_R} m, "
      f"lambda={lambda_nm} nm, theta_C={theta_C_urad} urad")

# Beam waist
lambda_m = lambda_nm * 1e-9
theta_C = theta_C_urad * 1e-6
w0 = compute_beam_waist(lambda_m, theta_C)
print(f"\n  w0 (beam waist) = {w0:.6f} m")
check("w0", w0, 0.05, 0.15, "m")

# Beam radius at receiver
zeta_rad = np.radians(zeta_deg)
L_slant_m = (H_S_km * 1e3) / np.cos(zeta_rad)
wL = compute_beam_radius(w0, L_slant_m, lambda_m)
print(f"  L_slant = {L_slant_m/1e3:.2f} km")
print(f"  wL (beam radius at receiver) = {wL:.4f} m")
check("wL", wL, 1.0, 20.0, "m")

# Geometric loss
hg, wL_out, nu_R = compute_hg(H_S_km, zeta_deg, a_R, lambda_nm, theta_C_urad)
print(f"\n  hg = {hg:.10f}")
print(f"  hg (dB) = {10*np.log10(hg):.4f} dB")
print(f"  nu_R = {nu_R:.8f}")
check("hg", hg, 1e-8, 1.0)
check("hg_dB", 10*np.log10(hg), -80, -10, "dB")

# Test at multiple zenith angles
print("\n  --- hg vs zenith angle ---")
for z in [0, 15, 30, 45, 60, 75]:
    hg_z, _, _ = compute_hg(H_S_km, z, a_R, lambda_nm, theta_C_urad)
    print(f"    zeta={z:2d} deg: hg={hg_z:.8e}, hg_dB={10*np.log10(hg_z):.2f} dB")

# ============================================================
# TEST GROUP 2: CHANNEL MODEL — Atmospheric Attenuation
# ============================================================
print("\n" + "="*70)
print("GROUP 2: CHANNEL MODEL — Atmospheric Attenuation (hl)")
print("="*70)

# Clear sky
V_km = 10.0
R_mm_h = 0.0
hl_clear = compute_hl(zeta_deg, V_km, R_mm_h, lambda_nm)
print(f"\n  Clear sky: V={V_km} km, R=0 mm/h")
print(f"  hl_clear = {hl_clear:.10f}")
print(f"  hl_clear (dB) = {10*np.log10(hl_clear):.4f} dB")
# L_atm = 20km/cos(30) = 23.1 km, sigma_vis ~ 0.1 km^-1 -> hl ~ exp(-2.3) ~ 0.1
check("hl_clear", hl_clear, 0.01, 1.0)

# Rain
R_mm_h_rain = 1.67  # Typical tropical
hl_rain = compute_hl(zeta_deg, V_km, R_mm_h_rain, lambda_nm)
print(f"\n  Rain: V={V_km} km, R={R_mm_h_rain} mm/h")
print(f"  hl_rain = {hl_rain:.10f}")
print(f"  hl_rain (dB) = {10*np.log10(hl_rain):.4f} dB")
check("hl_rain", hl_rain, 0.0001, 0.99)

# Sigma values
sigma_vis = compute_sigma_visibility(V_km, lambda_nm)
sigma_rain = compute_sigma_rain(R_mm_h_rain)
print(f"\n  sigma_vis = {sigma_vis:.8f} km^-1 (V={V_km} km)")
print(f"  sigma_rain = {sigma_rain:.8f} km^-1 (R={R_mm_h_rain} mm/h)")
check("sigma_vis", sigma_vis, 0.01, 2.0, "km^-1")
check("sigma_rain", sigma_rain, 0.01, 5.0, "km^-1")

# Heavy rain
R_heavy = 25.0
hl_heavy = compute_hl(zeta_deg, V_km, R_heavy, lambda_nm)
print(f"\n  Heavy rain: R={R_heavy} mm/h")
print(f"  hl_heavy = {hl_heavy:.10e}")
print(f"  hl_heavy (dB) = {10*np.log10(max(hl_heavy, 1e-100)):.2f} dB")
if hl_heavy < 1e-10:
    warn("hl_heavy", hl_heavy, "Extremely low -- link effectively OFF")

# ============================================================
# TEST GROUP 3: CHANNEL MODEL — Turbulence
# ============================================================
print("\n" + "="*70)
print("GROUP 3: CHANNEL MODEL — Turbulence (sigma_R2, sigma_X2)")
print("="*70)

H_U_km = 0.0
W = 21.0  # Wind speed m/s (H-V 5/7 default)
Cn2_0 = 1.7e-14  # Ground-level Cn2

sigma_R2, sigma_X2 = compute_rytov_variance(
    zeta_deg, H_S_km, H_U_km * 1e3, lambda_nm, W, Cn2_0
)
print(f"\n  Parameters: zeta={zeta_deg} deg, H_S={H_S_km} km, H_U={H_U_km} km")
print(f"  sigma_R2 = {sigma_R2:.8f}")
print(f"  sigma_X2 = {sigma_X2:.8f}")
check("sigma_R2", sigma_R2, 0.001, 5.0)
check("sigma_X2", sigma_X2, 0.0001, 1.5)

# Turbulence regime
if sigma_R2 < 0.3:
    print(f"  Regime: WEAK turbulence (sigma_R2 < 0.3) -> log-normal valid")
elif sigma_R2 < 1.0:
    print(f"  Regime: MODERATE turbulence (0.3 < sigma_R2 < 1.0)")
else:
    print(f"  Regime: STRONG turbulence (sigma_R2 > 1.0) -> Gamma-Gamma needed")

# Test Cn2 profile
print("\n  --- Cn2 profile (H-V 5/7) ---")
for h_m in [100, 1000, 5000, 10000, 20000]:
    cn2 = compute_Cn2_HV(h_m, W, Cn2_0)
    print(f"    h={h_m:6d} m: Cn2={cn2:.4e} m^(-2/3)")
    if cn2 < 0:
        results["fail"] += 1
        print(f"    [FAIL]: Cn2 negative!")

# sigma_R2 vs zenith
print("\n  --- sigma_R2 vs zenith ---")
prev_sr2 = 0.0
for z in [0, 15, 30, 45, 60, 75]:
    sr2, sx2 = compute_rytov_variance(z, H_S_km, 0, lambda_nm, W, Cn2_0)
    print(f"    zeta={z:2d} deg: sigma_R2={sr2:.6f}, sigma_X2={sx2:.6f}")
    if z > 0 and sr2 < prev_sr2:
        warn("sigma_R2 monotonicity", sr2, f"Should increase with zeta (prev={prev_sr2:.6f})")
    prev_sr2 = sr2

# ============================================================
# TEST GROUP 4: SIKD PERFORMANCE — Full pipeline
# ============================================================
print("\n" + "="*70)
print("GROUP 4: SIKD PERFORMANCE — Noise, Thresholds, Psift, QBER, SKR")
print("="*70)

# System parameters (Paper 2 defaults)
PT = 1.0        # W
mK = 0.05
mD = 0.5
Iso_dB = 15.0
Rb = 1e9        # 1 Gbps
Re = 0.9        # A/W
T_K = 280.0     # K
RL = 1000.0     # Ohm
Pb = 1e-9       # Background power (W)
zeta_sc = 2.0   # Threshold factor

print(f"\n  System: PT={PT} W, mK={mK}, mD={mD}, Iso={Iso_dB} dB, Rb={Rb/1e9} Gbps")
print(f"  Receiver: Re={Re} A/W, T={T_K} K, RL={RL} Ohm, zeta_sc={zeta_sc}")

# Use channel values from Group 1-3
ch = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                     theta_C_urad=theta_C_urad, V_km=V_km, R_mm_h=0.0,
                     W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
hg_val = ch["hg"]
hl_val = ch["hl"]
sigma_X2_val = ch["sigma_X2"]

print(f"\n  Channel: hg={hg_val:.8e}, hl={hl_val:.8f}, sigma_X2={sigma_X2_val:.8f}")

# Use compute_sikd_performance (convenience wrapper)
perf = compute_sikd_performance(
    hg=hg_val, hl=hl_val, sigma_X2=sigma_X2_val,
    PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB,
    zeta_scale=zeta_sc, Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
)

print(f"\n  Noise components:")
noise = perf["noise"]
print(f"    sigma2_shot    = {noise['sigma2_shot']:.6e}")
print(f"    sigma2_thermal = {noise['sigma2_thermal']:.6e}")
print(f"    sigma2_bg      = {noise['sigma2_bg']:.6e}")
print(f"    sigma2_CT      = {noise['sigma2_CT']:.6e}")
print(f"    sigma2_total   = {noise['sigma2_total']:.6e}")
print(f"    sigma_N        = {noise['sigma_N']:.6e}")

check("sigma2_shot", noise["sigma2_shot"], 1e-25, 1e-12)
check("sigma2_thermal", noise["sigma2_thermal"], 1e-25, 1e-12)
check("sigma2_CT", noise["sigma2_CT"], 0, 1e-8)
check("sigma2_total", noise["sigma2_total"], 1e-25, 1e-8)

# Dominant noise source
noises = {"shot": noise["sigma2_shot"], "thermal": noise["sigma2_thermal"],
           "bg": noise["sigma2_bg"], "CT": noise["sigma2_CT"]}
dominant = max(noises, key=noises.get)
print(f"    Dominant noise: {dominant} ({noises[dominant]:.4e})")

# Thresholds
print(f"\n  Thresholds:")
print(f"    d0 = {perf['d0']:.8e} A")
print(f"    d1 = {perf['d1']:.8e} A")

# Performance
Psift = perf["Psift"]
QBER = perf["QBER"]
SKR_norm = perf["SKR_norm"]
SKR_kbps = perf["SKR_kbps"]
BER_CC = perf["BER_CC"]

print(f"\n  Performance:")
print(f"    P_sift   = {Psift:.10f}")
print(f"    QBER     = {QBER:.10f} ({QBER*100:.4f}%)")
print(f"    SKR_norm = {SKR_norm:.10f}")
print(f"    SKR      = {SKR_kbps:.4f} kbps")
print(f"    BER_CC   = {BER_CC:.10e}")

check("Psift", Psift, 0.0, 1.0)
check("QBER", QBER, 0.0, 0.5)
check("SKR_norm", SKR_norm, 0.0, 1.0)
check("SKR_kbps", SKR_kbps, 0.0, 500000.0, "kbps")
check("BER_CC", BER_CC, 0.0, 0.5)

if QBER > 0.11:
    warn("QBER", QBER, "QBER > 11% -> SKR will be 0 (security threshold)")

if SKR_kbps > 0:
    print(f"    >> System operational: SKR > 0")
else:
    print(f"    >> System NOT operational: SKR = 0")

if BER_CC < 1e-9:
    print(f"    >> Classical channel: BER < 1e-9 (excellent)")
elif BER_CC < 1e-3:
    print(f"    >> Classical channel: BER < 1e-3 (acceptable with FEC)")
else:
    warn("BER_CC", BER_CC, "BER > 1e-3 -- classical channel degraded")

# ============================================================
# TEST GROUP 5: WEATHER MODEL — 3-State
# ============================================================
print("\n" + "="*70)
print("GROUP 5: WEATHER MODEL — 3-State SKR_effective")
print("="*70)

# Test Hanoi January (dry) vs July (wet)
for city, month, season in [("hanoi", 1, "DRY"), ("hanoi", 7, "WET"),
                             ("singapore", 1, ""), ("bangkok", 12, "DRY")]:
    params = get_city_params(city, month)
    R_mm_month = params["R_mm_month"]
    V_km_city = params["V_km"]
    P_cloud = params["P_cloud"]
    R_mm_h_city = params["R_mm_h"]

    # Compute SKR for clear and rain conditions
    ch_clear = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                               theta_C_urad=theta_C_urad, V_km=V_km_city,
                               R_mm_h=0.0, W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
    perf_clear = compute_sikd_performance(
        hg=ch_clear["hg"], hl=ch_clear["hl"], sigma_X2=ch_clear["sigma_X2"],
        PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
        Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
    )

    ch_rain = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                              theta_C_urad=theta_C_urad, V_km=V_km_city,
                              R_mm_h=R_mm_h_city, W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
    perf_rain = compute_sikd_performance(
        hg=ch_rain["hg"], hl=ch_rain["hl"], sigma_X2=ch_rain["sigma_X2"],
        PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
        Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
    )

    skr_3state = compute_effective_skr_3state(
        skr_clear=perf_clear["SKR_kbps"],
        skr_rain=perf_rain["SKR_kbps"],
        P_cloud=P_cloud
    )
    skr_eff = skr_3state["skr_effective"]

    print(f"\n  {city.upper()} month={month} ({season}):")
    print(f"    R_mm_month={R_mm_month}, V={V_km_city} km, P_cloud={P_cloud}")
    print(f"    SKR_clear={perf_clear['SKR_kbps']:.4f} kbps")
    print(f"    SKR_rain={perf_rain['SKR_kbps']:.4f} kbps")
    print(f"    SKR_effective = {skr_eff:.4f} kbps")

    check(f"SKR_eff_{city}_m{month}", skr_eff, 0.0, 50000.0, "kbps")

    if skr_eff <= 0:
        warn(f"SKR_eff_{city}_m{month}", skr_eff, "Zero SKR -- system non-operational")

# Wet/dry ratio check for Hanoi
params_dry = get_city_params("hanoi", 1)
params_wet = get_city_params("hanoi", 7)

ch_dry = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                         theta_C_urad=theta_C_urad, V_km=params_dry["V_km"],
                         R_mm_h=0.0, W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
perf_dry = compute_sikd_performance(
    hg=ch_dry["hg"], hl=ch_dry["hl"], sigma_X2=ch_dry["sigma_X2"],
    PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
    Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
)
ch_dry_rain = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                              theta_C_urad=theta_C_urad, V_km=params_dry["V_km"],
                              R_mm_h=params_dry["R_mm_h"], W=W, Cn2_0=Cn2_0,
                              H_U_km=H_U_km)
perf_dry_rain = compute_sikd_performance(
    hg=ch_dry_rain["hg"], hl=ch_dry_rain["hl"], sigma_X2=ch_dry_rain["sigma_X2"],
    PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
    Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
)
skr_dry_3s = compute_effective_skr_3state(
    perf_dry["SKR_kbps"], perf_dry_rain["SKR_kbps"], params_dry["P_cloud"]
)["skr_effective"]

ch_wet = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                         theta_C_urad=theta_C_urad, V_km=params_wet["V_km"],
                         R_mm_h=0.0, W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
perf_wet = compute_sikd_performance(
    hg=ch_wet["hg"], hl=ch_wet["hl"], sigma_X2=ch_wet["sigma_X2"],
    PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
    Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
)
ch_wet_rain = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                              theta_C_urad=theta_C_urad, V_km=params_wet["V_km"],
                              R_mm_h=params_wet["R_mm_h"], W=W, Cn2_0=Cn2_0,
                              H_U_km=H_U_km)
perf_wet_rain = compute_sikd_performance(
    hg=ch_wet_rain["hg"], hl=ch_wet_rain["hl"], sigma_X2=ch_wet_rain["sigma_X2"],
    PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
    Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
)
skr_wet_3s = compute_effective_skr_3state(
    perf_wet["SKR_kbps"], perf_wet_rain["SKR_kbps"], params_wet["P_cloud"]
)["skr_effective"]

if skr_wet_3s > 0:
    ratio = skr_dry_3s / skr_wet_3s
    print(f"\n  Hanoi dry/wet ratio = {ratio:.4f}")
    check("dry_wet_ratio", ratio, 1.0, 10.0)
    if ratio > 100:
        warn("dry_wet_ratio", ratio, "Ratio > 100 -- likely single-state bug!")
else:
    print(f"\n  Hanoi wet SKR = 0, cannot compute ratio")
    warn("skr_wet_hanoi", skr_wet_3s, "Wet season SKR = 0")

# ============================================================
# TEST GROUP 6: SANITY CHECKS — Physics Consistency
# ============================================================
print("\n" + "="*70)
print("GROUP 6: SANITY CHECKS — Physics Consistency")
print("="*70)

# 1. SKR should decrease with zenith angle
print("\n  --- SKR vs zenith (should decrease monotonically) ---")
prev_skr = float('inf')
monotonic_ok = True
for z in [0, 15, 30, 45, 60, 70, 75]:
    ch_z = compute_channel(H_S_km, z, a_R=a_R, lambda_nm=lambda_nm,
                           theta_C_urad=theta_C_urad, V_km=V_km, R_mm_h=0.0,
                           W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
    perf_z = compute_sikd_performance(
        hg=ch_z["hg"], hl=ch_z["hl"], sigma_X2=ch_z["sigma_X2"],
        PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
        Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
    )
    skr_z = perf_z["SKR_kbps"]
    if skr_z > prev_skr and prev_skr > 0:
        monotonic_ok = False
        warn(f"SKR_zeta={z}", skr_z, f"Non-monotonic! prev={prev_skr:.2f}")
    print(f"    zeta={z:2d} deg: SKR={skr_z:.2f} kbps, QBER={perf_z['QBER']*100:.3f}%")
    prev_skr = skr_z

if monotonic_ok:
    results["pass"] += 1
    print("  [PASS]: SKR decreases monotonically with zenith angle")

# 2. hl should decrease with rain rate
print("\n  --- hl vs rain rate (should decrease) ---")
prev_hl_val = 1.0
for R in [0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]:
    hl_r = compute_hl(30.0, 10.0, R, lambda_nm)
    if hl_r > prev_hl_val:
        warn(f"hl_R={R}", hl_r, f"Non-monotonic! prev={prev_hl_val:.6f}")
    print(f"    R={R:5.1f} mm/h: hl={hl_r:.8f} ({10*np.log10(max(hl_r,1e-100)):.2f} dB)")
    prev_hl_val = hl_r

# 3. SKR_eff should be between 0 and SKR_clear for all cities
print("\n  --- SKR_eff bounds check (all 8 cities x 12 months) ---")
bound_violations = 0
for city in ASEAN_CLIMATE_DATA.keys():
    for month in range(1, 13):
        params_cm = get_city_params(city, month)
        ch_cm_clear = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                                      theta_C_urad=theta_C_urad, V_km=params_cm["V_km"],
                                      R_mm_h=0.0, W=W, Cn2_0=Cn2_0, H_U_km=H_U_km)
        perf_cm_clear = compute_sikd_performance(
            hg=ch_cm_clear["hg"], hl=ch_cm_clear["hl"], sigma_X2=ch_cm_clear["sigma_X2"],
            PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
            Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
        )
        ch_cm_rain = compute_channel(H_S_km, zeta_deg, a_R=a_R, lambda_nm=lambda_nm,
                                     theta_C_urad=theta_C_urad, V_km=params_cm["V_km"],
                                     R_mm_h=params_cm["R_mm_h"], W=W, Cn2_0=Cn2_0,
                                     H_U_km=H_U_km)
        perf_cm_rain = compute_sikd_performance(
            hg=ch_cm_rain["hg"], hl=ch_cm_rain["hl"], sigma_X2=ch_cm_rain["sigma_X2"],
            PT=PT, mK=mK, mD=mD, Iso_dB=Iso_dB, zeta_scale=zeta_sc,
            Re=Re, T=T_K, RL=RL, Rb=Rb, Pbg=Pb
        )
        skr_3s = compute_effective_skr_3state(
            perf_cm_clear["SKR_kbps"], perf_cm_rain["SKR_kbps"], params_cm["P_cloud"]
        )["skr_effective"]

        if skr_3s < 0:
            bound_violations += 1
            warn(f"{city}_m{month}", skr_3s, "SKR_eff < 0!")
        if skr_3s > perf_cm_clear["SKR_kbps"] * 1.01:
            bound_violations += 1
            warn(f"{city}_m{month}", skr_3s,
                 f"SKR_eff > SKR_clear ({perf_cm_clear['SKR_kbps']:.2f})!")

if bound_violations == 0:
    results["pass"] += 1
    print("  [PASS]: All 96 city-month combinations within bounds")
else:
    print(f"  [FAIL]: {bound_violations} bound violations found")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
total = results["pass"] + results["fail"] + results["warn"]
print(f"\n  PASS: {results['pass']}")
print(f"  FAIL: {results['fail']}")
print(f"  WARN: {results['warn']}")
print(f"  Total checks: {total}")
print()

if results["fail"] == 0:
    print("  ALL CHECKS PASSED -- Pipeline is consistent")
else:
    print(f"  {results['fail']} FAILURES DETECTED -- Review needed")
    sys.exit(1)
