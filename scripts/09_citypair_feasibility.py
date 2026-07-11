"""
09_citypair_feasibility.py — City-Pair Feasibility Matrices (Task 6, plan 07-5)
================================================================================
Builds the three (8,8) matrices for all 28 unordered ASEAN city pairs:
  CLASS[i,j]          in {'DUAL','SF'} — symmetric (static geometry, Task 7)
  DUAL_PCT[i,j]       real simulated %% time with >=1 satellite visible to
                      both stations simultaneously @ elev >= mask — symmetric
  SF_LATENCY_MIN[i,j] directed store-and-forward relay latency (minutes),
                      i->j vs j->i computed INDEPENDENTLY (see 07-5 Task 6
                      design fix / tests/test_citypair_feasibility.py) —
                      reuses the real 7-day pass table from Task 5.

DUAL_PCT is computed here (not in modules/citypair_feasibility.py) because
it needs a full time-resolved visibility simulation over live TLEs, which
belongs at the script level; the module only defines the generic matrix
structure and is agnostic to how DUAL_PCT / SF latency are actually
computed (see build_pairwise_matrices' dual_pct_fn hook).

Run:
  cd 05_Code_v2 && python scripts/09_citypair_feasibility.py
"""
import os
import sys
import time
import csv
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from modules.orbital_mechanics import (
    parse_tle_block, make_skyfield_satellite, GROUND_STATIONS,
    compute_elevation_timeseries, make_time_array,
)
from modules.link_geometry import haversine_km
from modules.citypair_feasibility import build_pairwise_matrices
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 9, 'font.family': 'serif',
    'axes.titlesize': 9, 'axes.labelsize': 9,
})

t0 = time.time()

CITIES = list(GROUND_STATIONS.keys())
CITY_LABELS = {
    'hanoi': 'Hanoi', 'danang': 'Da Nang', 'hcmc': 'HCMC', 'bangkok': 'Bangkok',
    'singapore': 'Singapore', 'manila': 'Manila', 'jakarta': 'Jakarta',
    'kuala_lumpur': 'Kuala Lumpur',
}
H_KM = 550.0
MIN_ELEV_DEG = 30.0
T_START = datetime(2026, 3, 12, 0, 0, 0, tzinfo=timezone.utc)
DUAL_PCT_WINDOW_HOURS = 24.0

# ----------------------------------------------------------------
# 1. Load satellites, build boolean visibility mask per city (24h window)
# ----------------------------------------------------------------
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
print(f"Loaded {len(satellites)} real Shell-1 satellites")

t_array = make_time_array(T_START, DUAL_PCT_WINDOW_HOURS, step_minutes=0.5)
n_steps = len(t_array)
print(f"Dual-coverage window: {DUAL_PCT_WINDOW_HOURS:.0f}h, {n_steps} steps @ 30s\n")

visibility_masks = {}
for city in CITIES:
    tc0 = time.time()
    gs = GROUND_STATIONS[city]
    mask = np.zeros((len(satellites), n_steps), dtype=bool)
    for k, sat in enumerate(satellites):
        elev = compute_elevation_timeseries(sat, gs['lat'], gs['lon'], t_array, gs['alt_m'])
        mask[k, :] = elev >= MIN_ELEV_DEG
    visibility_masks[city] = mask
    print(f"  {city:14s} visibility mask built ({time.time()-tc0:.1f}s)")

# ----------------------------------------------------------------
# 2. Load real 7-day pass table from Task 5 (script 08 output)
# ----------------------------------------------------------------
from utils import INTER as _INTER
PASS_CSV = os.path.join(str(_INTER),
                        'pass_table_8cities_7days_elev30.csv')
pass_table = []
with open(PASS_CSV, newline='', encoding='utf-8') as f:
    next(f)  # skip the '# description' comment line written by save_intermediate_csv
    reader = csv.DictReader(f)
    for row in reader:
        pass_table.append({
            "sat_id": row["sat_id"],
            "station": row["station"],
            "t_rise": datetime.fromisoformat(row["t_rise"]),
            "t_set": datetime.fromisoformat(row["t_set"]),
        })
print(f"\nLoaded {len(pass_table)} passes from Task 5 pass table (elev >= 30 deg)")

# ----------------------------------------------------------------
# 3. Build the 3 matrices
# ----------------------------------------------------------------
def dist_km_fn(a, b):
    ga, gb = GROUND_STATIONS[a], GROUND_STATIONS[b]
    return haversine_km(ga['lat'], ga['lon'], gb['lat'], gb['lon'])


def dual_pct_fn(a, b):
    overlap = np.any(visibility_masks[a] & visibility_masks[b], axis=0)
    return float(overlap.mean()) * 100.0


CLASS, DUAL_PCT, SF_LATENCY_MIN = build_pairwise_matrices(
    CITIES, dist_km_fn, pass_table, H_KM, MIN_ELEV_DEG, dual_pct_fn=dual_pct_fn,
)

n_dual = sum(1 for i in range(8) for j in range(i + 1, 8) if CLASS[i, j] == "DUAL")
n_sf = sum(1 for i in range(8) for j in range(i + 1, 8) if CLASS[i, j] == "SF")
print(f"\nClassification @ {MIN_ELEV_DEG:.0f} deg mask: {n_dual} DUAL pairs, {n_sf} SF pairs (of 28 unordered)")

# ----------------------------------------------------------------
# 4. Save the 56 directed entries as one table
# ----------------------------------------------------------------
rows = []
for i in range(8):
    for j in range(8):
        if i == j:
            continue
        rows.append({
            "city_i": CITIES[i], "city_j": CITIES[j],
            "class": CLASS[i, j],
            "dual_pct": round(float(DUAL_PCT[i, j]), 2),
            "sf_latency_min": (round(float(SF_LATENCY_MIN[i, j]), 1)
                              if not np.isnan(SF_LATENCY_MIN[i, j]) else ""),
        })
save_intermediate_csv(
    rows, "citypair_feasibility_matrix",
    "56 directed city-pair entries: CLASS/DUAL_PCT symmetric, "
    "SF_LATENCY_MIN directed (i->j independent of j->i)",
)

print("\nSF latency examples (directed):")
for i in range(8):
    for j in range(8):
        if i != j and CLASS[i, j] == "SF" and not np.isnan(SF_LATENCY_MIN[i, j]):
            print(f"  {CITIES[i]:14s} -> {CITIES[j]:14s}  {SF_LATENCY_MIN[i,j]:7.1f} min")

# ----------------------------------------------------------------
# 5. Map figure: DUAL = single line, SF = two-directional arrows w/ 2 numbers
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.0))
lats = [GROUND_STATIONS[c]['lat'] for c in CITIES]
lons = [GROUND_STATIONS[c]['lon'] for c in CITIES]
ax.scatter(lons, lats, s=60, color='black', zorder=5)
for c, lo, la in zip(CITIES, lons, lats):
    ax.annotate(CITY_LABELS[c], (lo, la), textcoords="offset points",
                xytext=(5, 5), fontsize=7)

for i in range(8):
    for j in range(i + 1, 8):
        lo_i, la_i = GROUND_STATIONS[CITIES[i]]['lon'], GROUND_STATIONS[CITIES[i]]['lat']
        lo_j, la_j = GROUND_STATIONS[CITIES[j]]['lon'], GROUND_STATIONS[CITIES[j]]['lat']
        if CLASS[i, j] == "DUAL":
            ax.plot([lo_i, lo_j], [la_i, la_j], color='#2196F3', linewidth=1.0, alpha=0.6, zorder=2)
        else:
            ax.plot([lo_i, lo_j], [la_i, la_j], color='#FF5722', linewidth=0.6,
                    linestyle='--', alpha=0.5, zorder=1)

ax.plot([], [], color='#2196F3', linewidth=1.5, label='DUAL (simultaneous downlink)')
ax.plot([], [], color='#FF5722', linewidth=1.5, linestyle='--', label='SF (store-and-forward, directed latency)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(loc='lower right', fontsize=7)
ax.set_title(f'ASEAN City-Pair Feasibility (elev $\\geq {MIN_ELEV_DEG:.0f}^\\circ$): '
             f'{n_dual} DUAL, {n_sf} SF pairs', fontsize=8.5, fontweight='bold')
ax.grid(alpha=0.2)

out1 = os.path.join(_FIG_BASE, 'fig05_citypair_feasibility_map.png')
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ----------------------------------------------------------------
# 6. Verify + provenance
# ----------------------------------------------------------------
verify = {
    "n_dual_pairs_elev30": str(n_dual),
    "n_sf_pairs_elev30": str(n_sf),
}
sf_examples = [(i, j) for i in range(8) for j in range(8)
               if i != j and CLASS[i, j] == "SF" and not np.isnan(SF_LATENCY_MIN[i, j])]
if sf_examples:
    i, j = sf_examples[0]
    verify[f"sf_latency_{CITIES[i]}_to_{CITIES[j]}_min"] = f"{SF_LATENCY_MIN[i,j]:.1f}"
    verify[f"sf_latency_{CITIES[j]}_to_{CITIES[i]}_min"] = (
        f"{SF_LATENCY_MIN[j,i]:.1f}" if not np.isnan(SF_LATENCY_MIN[j, i]) else "N/A")
save_verify_numbers(verify, "citypair_feasibility")

save_provenance(
    script_name="09_citypair_feasibility",
    params={"h_km": H_KM, "min_elev_deg": MIN_ELEV_DEG,
            "dual_pct_window_hours": DUAL_PCT_WINDOW_HOURS, "n_cities": len(CITIES)},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1],
    data_sources={"pass table (Task 5)": PASS_CSV,
                  "Shell-1 TLE": TLE_PATH},
    formulas={
        "CLASS/DUAL_PCT": "static distance-based classification (symmetric); "
                          "DUAL_PCT = %% of 24h window with >=1 shared-visible satellite",
        "SF_LATENCY_MIN": "median minutes from pass over city_i (t_rise) to same "
                          "satellite's next pass over city_j (t_rise); directed, "
                          "i->j computed independently of j->i (modules/citypair_feasibility.py)",
    },
)
print(f"\nDone in {time.time() - t0:.1f}s")
