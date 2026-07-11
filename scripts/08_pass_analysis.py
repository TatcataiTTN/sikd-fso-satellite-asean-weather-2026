"""
08_pass_analysis.py — Satellite Pass Table & Statistics (Task 5, plan 07-5)
================================================================================
Extracts the full satellite pass table for all 8 ASEAN ground stations over
a 7-day window, using the real 1,019-satellite Shell-1 subset (CelesTrak
25/06/2026), at both the 30 deg (geometric minimum) and 40 deg (BB84 security
margin with fixed DT thresholds) elevation masks.

Cross-checks pass timing against the diurnal cloud/rain "best FSO window"
found in Task 2 (modules/weather_stats.py): mid-morning, roughly 05:00-11:00
local, NOT overnight as originally assumed (see 07-5 section 0.2b). This is
the input Task 18 needs to state whether enough real passes actually fall
inside that window.

Run:
  cd 05_Code_v2 && python scripts/08_pass_analysis.py
"""
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from modules.orbital_mechanics import parse_tle_block, make_skyfield_satellite, GROUND_STATIONS
from modules.pass_analysis import (extract_passes, pass_frequency_per_day,
                                   passes_dataframe, UTC_OFFSET_HOURS, _local_hour)
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 9,
    'font.family': 'serif',
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

t0 = time.time()

CITIES = list(GROUND_STATIONS.keys())
CITY_LABELS = {
    'hanoi': 'Hanoi', 'danang': 'Da Nang', 'hcmc': 'HCMC', 'bangkok': 'Bangkok',
    'singapore': 'Singapore', 'manila': 'Manila', 'jakarta': 'Jakarta',
    'kuala_lumpur': 'Kuala Lumpur',
}
T_START = datetime(2026, 3, 12, 0, 0, 0, tzinfo=timezone.utc)
DURATION_HOURS = 7 * 24.0
BEST_WINDOW = (5, 11)  # from data/verify/diurnal_cloud_rain_window.txt (Task 2)

TLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'starlink_shell1_real_tle.txt')
# --- Epoch-variant overrides (Phase G, 05/07/2026): set SIKD_T_START /
# SIKD_TLE_PATH / SIKD_VARIANT_DIR to rerun this script under a different
# epoch without touching canonical outputs. Defaults preserve old behavior.
if os.environ.get('SIKD_T_START'):
    T_START = datetime.fromisoformat(os.environ['SIKD_T_START'])
if os.environ.get('SIKD_TLE_PATH'):
    TLE_PATH = os.environ['SIKD_TLE_PATH']
_VARIANT = os.environ.get('SIKD_VARIANT_DIR')
_FIG_BASE = (os.path.join(_VARIANT, 'figures') if _VARIANT else
             os.path.join(os.path.dirname(__file__), '..', '..', 'latex_paper_3', 'figures'))

tle_dicts = parse_tle_block(open(TLE_PATH).read())
satellites = [make_skyfield_satellite(d) for d in tle_dicts]
print(f"Loaded {len(satellites)} real Shell-1 satellites (CelesTrak 25/06/2026)")
print(f"Window: {T_START.isoformat()} + {DURATION_HOURS/24:.0f} days\n")

# ----------------------------------------------------------------
# Extract passes for all cities at both masks
# ----------------------------------------------------------------
passes_30 = {}
passes_40 = {}
for city in CITIES:
    tc0 = time.time()
    passes_30[city] = extract_passes(satellites, city, T_START, DURATION_HOURS, min_elev_deg=30.0)
    passes_40[city] = extract_passes(satellites, city, T_START, DURATION_HOURS, min_elev_deg=40.0)
    print(f"  {city:14s}  n_pass@30={len(passes_30[city]):5d}  "
          f"n_pass@40={len(passes_40[city]):5d}  ({time.time()-tc0:.1f}s)")

# ----------------------------------------------------------------
# Master CSV (30 deg mask, primary table)
# ----------------------------------------------------------------
all_rows = []
for city in CITIES:
    rows = passes_dataframe(passes_30[city])
    for r in rows:
        r["station"] = city
    all_rows.extend(rows)
all_rows.sort(key=lambda r: r["t_rise"])

save_intermediate_csv(
    all_rows, "pass_table_8cities_7days_elev30",
    "Full satellite pass table, 8 ASEAN stations x 7 days, real Shell-1 "
    "TLE (elev >= 30 deg). See modules/pass_analysis.py for rise/set "
    "interpolation method.",
)

# ----------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------
print("\nPer-city statistics:")
stats_rows = []
window_hits_total = 0
n_passes_total = 0
for city in CITIES:
    p30 = passes_30[city]
    p40 = passes_40[city]
    freq = pass_frequency_per_day(p30, DURATION_HOURS)
    dur30 = np.median([p["duration_s"] for p in p30]) if p30 else 0.0
    dur40 = np.median([p["duration_s"] for p in p40]) if p40 else 0.0
    elevs = [p["max_elev_deg"] for p in p30]
    hours = np.array([p["local_hour_peak"] for p in p30])
    in_window = ((hours >= BEST_WINDOW[0]) & (hours < BEST_WINDOW[1])).sum()
    window_hits_total += in_window
    n_passes_total += len(p30)
    stats_rows.append({
        "city": city,
        "passes_per_day": round(freq, 1),
        "median_duration_s_elev30": round(dur30, 1),
        "median_duration_s_elev40": round(dur40, 1),
        "median_max_elev_deg": round(float(np.median(elevs)), 1) if elevs else 0.0,
        "pct_passes_in_best_window": round(100.0 * in_window / len(p30), 1) if p30 else 0.0,
    })
    print(f"  {city:14s} passes/day={freq:6.1f}  median_dur@30={dur30:5.1f}s  "
          f"median_dur@40={dur40:5.1f}s  in_best_window={100.0*in_window/len(p30) if p30 else 0:.1f}%")

save_intermediate_csv(stats_rows, "pass_statistics_per_city",
                       "Pass frequency, duration, and best-window overlap per city")

overall_pct_in_window = 100.0 * window_hits_total / n_passes_total if n_passes_total else 0.0
print(f"\nOverall: {overall_pct_in_window:.1f}% of all passes (any city) peak within "
      f"the best FSO window ({BEST_WINDOW[0]}:00-{BEST_WINDOW[1]}:00 local) "
      f"[uniform-24h expectation: {100.0*(BEST_WINDOW[1]-BEST_WINDOW[0])/24:.1f}%]")

# ----------------------------------------------------------------
# Figure 1: Gantt-style pass timeline, ONE representative day (local hour)
# ----------------------------------------------------------------
# Plotting all 7 days' passes (thousands of points per station) collapses
# into a solid line at these marker sizes and shows no structure; a single
# day, drawn as an actual rise-to-set bar per pass, is what makes individual
# passes and the gaps between them visible.
day1_end = T_START + timedelta(hours=24)
fig, ax = plt.subplots(figsize=(7.16, 3.2))
for i, city in enumerate(CITIES):
    utc_offset = UTC_OFFSET_HOURS[city]
    bars = []
    for p in passes_30[city]:
        if not (T_START <= p["t_rise"] < day1_end):
            continue
        h_rise = _local_hour(p["t_rise"], utc_offset)
        h_set = _local_hour(p["t_set"], utc_offset)
        if h_set < h_rise:
            h_set = 24.0  # pass straddles local midnight -- clip, negligible for 30 deg passes
        bars.append((h_rise, max(h_set - h_rise, 0.05)))
    ax.broken_barh(bars, (i - 0.4, 0.8), facecolor='#1f77b4', edgecolor='none')
ax.axvspan(BEST_WINDOW[0], BEST_WINDOW[1], color='green', alpha=0.12,
           label=f'Lowest cloud/rain-risk window ({BEST_WINDOW[0]}:00-{BEST_WINDOW[1]}:00)')
ax.set_yticks(range(len(CITIES)))
ax.set_yticklabels([CITY_LABELS[c] for c in CITIES])
ax.set_xlabel('Local hour of day')
ax.set_xlim(0, 24)
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.legend(loc='upper right', fontsize=6.5)
ax.set_title('Satellite Pass Timeline, One Representative Day (elev $\\geq 30^\\circ$)',
             fontsize=8.5, fontweight='bold')
out1 = os.path.join(_FIG_BASE, 'fig04a_pass_timeline.png')
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ----------------------------------------------------------------
# Figure 2: Histogram of local_hour_peak, pooled across all cities
# ----------------------------------------------------------------
all_hours = np.concatenate([[p["local_hour_peak"] for p in passes_30[c]] for c in CITIES])
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.hist(all_hours, bins=24, range=(0, 24), color='#1f77b4', alpha=0.8, edgecolor='white')
ax.axvspan(BEST_WINDOW[0], BEST_WINDOW[1], color='green', alpha=0.15,
           label='Lowest cloud/rain-risk window')
ax.set_xlabel('Local hour of day')
ax.set_ylabel('Pass count (all 8 stations, 7 days)')
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.legend(fontsize=7)
ax.set_title('Distribution of Pass Peak Times', fontsize=8.5, fontweight='bold')
out2 = os.path.join(_FIG_BASE, 'fig04b_pass_hour_histogram.png')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ----------------------------------------------------------------
# Verify + provenance
# ----------------------------------------------------------------
verify = {"overall_pct_passes_in_best_window": f"{overall_pct_in_window:.1f}",
          "uniform_expectation_pct": f"{100.0*(BEST_WINDOW[1]-BEST_WINDOW[0])/24:.1f}"}
for r in stats_rows:
    verify[f"{r['city']}_passes_per_day"] = str(r["passes_per_day"])
    verify[f"{r['city']}_median_dur_s_elev30"] = str(r["median_duration_s_elev30"])
    verify[f"{r['city']}_median_dur_s_elev40"] = str(r["median_duration_s_elev40"])
save_verify_numbers(verify, "pass_analysis_stats")

save_provenance(
    script_name="08_pass_analysis",
    params={
        "n_satellites": len(satellites),
        "t_start_utc": T_START.isoformat(),
        "duration_days": DURATION_HOURS / 24.0,
        "elev_masks_deg": [30.0, 40.0],
        "step_seconds": 30.0,
        "best_window_local_hour": BEST_WINDOW,
    },
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1, out2],
    data_sources={
        "Shell-1 TLE": TLE_PATH,
        "best FSO window (Task 2)": "data/verify/diurnal_cloud_rain_window.txt",
    },
    formulas={
        "local_hour_peak": "(UTC hour of peak elevation + fixed civil UTC offset) mod 24; "
                            "see modules/pass_analysis.py UTC_OFFSET_HOURS",
        "Rise/set refinement": "linear interpolation of the elevation-mask crossing time "
                               "between the two samples straddling it (30s sampling step)",
    },
)
print(f"\nDone in {time.time() - t0:.1f}s")
