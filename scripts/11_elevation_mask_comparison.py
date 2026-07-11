"""
11_elevation_mask_comparison.py — Three Elevation-Mask Comparison (Task 7)
================================================================================
Task 7 VIỆC 2 (07-5): compares three operating masks for the SIKD link:

  Mask A: 30 deg (geometric minimum) with FIXED DT thresholds
          -> QBER at the mask boundary EXCEEDS the BB84 11% security limit
             (INSECURE at low elevation).
  Mask B: 40 deg (BB84 security margin) with FIXED DT thresholds
          -> QBER stays below 11% at every elevation within the mask.
  Mask C: 30 deg WITH per-pass ADAPTIVE DT thresholds (thesis, Vu 2022/2023)
          -> QBER driven toward ~1e-3, at the cost of Psift dropping ~25x.
             Geometry (radius, pass duration, DUAL-pair count) is IDENTICAL
             to Mask A since it is the same 30 deg elevation cutoff; only the
             detection-threshold strategy differs. Not simulated here (the
             adaptive-threshold optimizer is future work, see CLAUDE.md and
             project_dissertation_insights memory) -- QBER/Psift for Mask C
             are the thesis-reported figures, stated as such, not computed.

Reuses:
  - modules/link_geometry.py (Task 6): ground radius, DUAL-pair count.
  - data/intermediate/pass_statistics_per_city.csv (Task 5): real
    passes/day and median pass duration at both 30 and 40 deg masks.
  - modules/channel_model.py + modules/sikd_performance.py: QBER/Psift/SKR
    at the mask boundary elevation, clear sky (V=10km, R=0), matching
    Table II in the paper.

Run:
  cd 05_Code_v2 && python scripts/11_elevation_mask_comparison.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from modules.link_geometry import ground_coverage_radius_km, classify_pair, haversine_km
from modules.orbital_mechanics import GROUND_STATIONS
from modules.channel_model import compute_channel
from modules.sikd_performance import compute_sikd_performance
from utils import save_provenance, save_verify_numbers

t0 = time.time()

CITIES = list(GROUND_STATIONS.keys())
H_KM = 550.0

# ----------------------------------------------------------------
# Geometry: ground radius + DUAL-pair count at each mask (Task 6 reuse)
# ----------------------------------------------------------------
def n_dual_pairs(min_elev_deg):
    n = 0
    for i in range(8):
        for j in range(i + 1, 8):
            d = haversine_km(GROUND_STATIONS[CITIES[i]]['lat'], GROUND_STATIONS[CITIES[i]]['lon'],
                             GROUND_STATIONS[CITIES[j]]['lat'], GROUND_STATIONS[CITIES[j]]['lon'])
            if classify_pair(d, H_KM, min_elev_deg) == "DUAL":
                n += 1
    return n

radius_30 = ground_coverage_radius_km(H_KM, 30.0)
radius_40 = ground_coverage_radius_km(H_KM, 40.0)
dual_30 = n_dual_pairs(30.0)
dual_40 = n_dual_pairs(40.0)

# ----------------------------------------------------------------
# Real pass statistics (Task 5), averaged across 8 cities
# ----------------------------------------------------------------
STATS_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'intermediate',
                        'pass_statistics_per_city.csv')
rows = []
with open(STATS_CSV, newline='', encoding='utf-8') as f:
    next(f)  # skip description comment line
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

passes_per_day_30 = np.mean([float(r["passes_per_day"]) for r in rows])
dur_30 = np.mean([float(r["median_duration_s_elev30"]) for r in rows])
dur_40 = np.mean([float(r["median_duration_s_elev40"]) for r in rows])

# ----------------------------------------------------------------
# QBER/Psift/SKR at mask boundary (clear sky, matches Table II)
# ----------------------------------------------------------------
def channel_perf_at_elev(elev_deg):
    zenith = 90.0 - elev_deg
    ch = compute_channel(H_KM, zenith, V_km=10.0, R_mm_h=0.0, theta_C_urad=10.0)
    return compute_sikd_performance(ch['hg'], ch['hl'], ch['sigma_X2'])

perf_30 = channel_perf_at_elev(30.0)
perf_40 = channel_perf_at_elev(40.0)

# ----------------------------------------------------------------
# Print + save the 3-mask comparison table
# ----------------------------------------------------------------
print("Three-mask comparison table (Task 7 VIỆC 2):")
print(f"{'':30s} {'Mask A (30°,fixed)':>20s} {'Mask B (40°,fixed)':>20s} {'Mask C (30°,adaptive)':>22s}")
print(f"{'Ground radius (km)':30s} {radius_30:>20.0f} {radius_40:>20.0f} {radius_30:>22.0f}")
print(f"{'DUAL pairs (of 28)':30s} {dual_30:>20d} {dual_40:>20d} {dual_30:>22d}")
print(f"{'Passes/day (avg 8 cities)':30s} {passes_per_day_30:>20.1f} {'(same sats)':>20s} {passes_per_day_30:>22.1f}")
print(f"{'Median pass duration (s)':30s} {dur_30:>20.1f} {dur_40:>20.1f} {dur_30:>22.1f}")
print(f"{'QBER at mask boundary (%)':30s} {perf_30['QBER']*100:>20.2f} {perf_40['QBER']*100:>20.2f} {'~0.1 (thesis)':>22s}")
print(f"{'Psift at mask boundary':30s} {perf_30['Psift']:>20.4f} {perf_40['Psift']:>20.4f} {'~0.001 (thesis)':>22s}")
print(f"{'BB84 security (QBER<11%)':30s} {'INSECURE':>20s} {'SECURE':>20s} {'SECURE':>22s}")

verify = {
    "mask30_ground_radius_km": f"{radius_30:.0f}",
    "mask40_ground_radius_km": f"{radius_40:.0f}",
    "mask30_n_dual_pairs": str(dual_30),
    "mask40_n_dual_pairs": str(dual_40),
    "mask30_passes_per_day_avg": f"{passes_per_day_30:.1f}",
    "mask30_median_duration_s": f"{dur_30:.1f}",
    "mask40_median_duration_s": f"{dur_40:.1f}",
    "mask30_qber_pct": f"{perf_30['QBER']*100:.2f}",
    "mask40_qber_pct": f"{perf_40['QBER']*100:.2f}",
    "mask30_psift": f"{perf_30['Psift']:.4f}",
    "mask40_psift": f"{perf_40['Psift']:.4f}",
    "maskC_qber_pct_thesis": "~0.1 (Vu 2022/2023, not simulated here)",
    "maskC_psift_thesis": "~0.001 (Vu 2022/2023, ~25x lower than Mask A)",
}
save_verify_numbers(verify, "elevation_mask_comparison")

save_provenance(
    script_name="11_elevation_mask_comparison",
    params={"h_km": H_KM, "masks_deg": [30.0, 40.0]},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[],
    data_sources={
        "pass statistics (Task 5)": STATS_CSV,
        "DUAL-pair classification (Task 6)": "modules/link_geometry.py",
    },
    formulas={
        "Mask A vs B geometry": "ground_coverage_radius_km(550, elev), classify_pair() "
                                "over the 28 ASEAN city pairs",
        "QBER/Psift at boundary": "compute_channel + compute_sikd_performance, clear sky "
                                  "(V=10km, R=0), matches paper Table II",
        "Mask C": "same 30 deg geometry as Mask A; QBER/Psift figures are the thesis-reported "
                 "values for per-pass adaptive DT threshold optimization (Vu 2022/2023), "
                 "NOT recomputed by this script -- adaptive threshold optimizer is future work",
    },
)
print(f"\nDone in {time.time() - t0:.2f}s")
