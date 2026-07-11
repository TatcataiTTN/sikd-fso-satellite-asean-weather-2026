"""
11f_sf_relay_detail.py — Directed SF Relay Detail, 56 Entries (Task 23, Phase G)
================================================================================
Makes the 56 directed city-pair entries fully explicit. Task 6 established
that store-and-forward relay latency is DIRECTIONAL (satellite ground-track
heading differs between ascending and descending passes; measured extreme:
jakarta->manila 7.0 min vs manila->jakarta 711.4 min, ~100x) but only
reported the median per direction. This script adds, for every directed SF
entry (i -> j): min/median/max latency, the number of relay opportunities,
the top-5 relay satellites by frequency, and — for the single FASTEST relay
— which satellite it was and the local pickup/dropoff clock times at each
end. DUAL-classified pairs carry their simultaneous-coverage percentage
instead (from Task 6's output; no re-simulation).

Definition (identical to modules/citypair_feasibility.sf_latency_minutes,
cross-checked below): one relay opportunity = a pass of satellite s over
city i (pickup, at t_rise) followed by the SAME satellite's next pass over
city j (dropoff, at t_rise). No inter-satellite links (07-5 scope).

Independent of Tasks 21-22: needs only the Task 5 pass table and the
Task 6 feasibility matrix.

Run:
  cd 05_Code_v2 && python scripts/11f_sf_relay_detail.py
"""
import os
import sys
import csv
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.orbital_mechanics import GROUND_STATIONS
from modules.pass_analysis import UTC_OFFSET_HOURS
from modules.citypair_feasibility import sf_latency_minutes
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 9, 'font.family': 'serif',
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
})

t0 = time.time()
CITIES = list(GROUND_STATIONS.keys())
CITY_LABELS = {
    'hanoi': 'Hanoi', 'danang': 'Da Nang', 'hcmc': 'HCMC', 'bangkok': 'Bangkok',
    'singapore': 'Singapore', 'manila': 'Manila', 'jakarta': 'Jakarta',
    'kuala_lumpur': 'Kuala Lumpur',
}

from utils import INTER as _INTER
_VARIANT = os.environ.get('SIKD_VARIANT_DIR')
PASS_CSV = os.path.join(str(_INTER),
                        'pass_table_8cities_7days_elev30.csv')
FEAS_CSV = os.path.join(str(_INTER),
                        'citypair_feasibility_matrix.csv')
# v1 (single-hop store-and-forward) is legacy/superseded by the ISL
# multi-hop model (Task 24.3, 12b_isl_relay_recompute.py) -- kept on disk
# for historical "before" comparison only, so its figure is NOT written
# into the active latex_paper_3/figures/ folder (that would silently
# resurrect a retired file every time this script reruns).
FIG_DIR = (os.path.join(_VARIANT, 'figures') if _VARIANT else
           os.path.join(os.path.dirname(__file__), '..', 'temp', 'unused_paper_figs'))
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------
# 1. Load pass table (Task 5) and feasibility classes (Task 6)
# ----------------------------------------------------------------
df = pd.read_csv(PASS_CSV, comment='#')
df['t_rise_dt'] = pd.to_datetime(df['t_rise'], format='ISO8601')
print(f"Loaded {len(df)} passes (Task 5, elev >= 30 deg, 7 days)")

feas = pd.read_csv(FEAS_CSV, comment='#')
class_of = {(r['city_i'], r['city_j']): r['class'] for _, r in feas.iterrows()}
dual_pct_of = {(r['city_i'], r['city_j']): r['dual_pct'] for _, r in feas.iterrows()}

# Per (station, sat): chronologically sorted pickup times (numpy datetime64)
times_by = defaultdict(dict)  # station -> sat_id -> sorted np.array of t_rise
for (station, sat_id), grp in df.groupby(['station', 'sat_id']):
    times_by[station][sat_id] = np.sort(grp['t_rise_dt'].values)


def local_hour_str(ts64, station):
    """Local clock time 'HH:MM' for a numpy datetime64 UTC instant."""
    ts = pd.Timestamp(ts64)
    local = ts + pd.Timedelta(hours=UTC_OFFSET_HOURS[station])
    return local.strftime('%H:%M')


def relay_opportunities(city_i, city_j):
    """All (latency_min, sat_id, t_pickup, t_dropoff): for each pass of each
    satellite over city_i, the SAME satellite's next pass over city_j.
    Same definition as modules/citypair_feasibility.sf_latency_minutes."""
    out = []
    sats_i = times_by[city_i]
    sats_j = times_by[city_j]
    for sat_id in sats_i.keys() & sats_j.keys():
        t_i = sats_i[sat_id]
        t_j = sats_j[sat_id]
        idx = np.searchsorted(t_j, t_i, side='right')
        valid = idx < len(t_j)
        for k in np.nonzero(valid)[0]:
            t_pick = t_i[k]
            t_drop = t_j[idx[k]]
            lat_min = (t_drop - t_pick) / np.timedelta64(60, 's')
            out.append((float(lat_min), sat_id, t_pick, t_drop))
    return out


# ----------------------------------------------------------------
# 2. Build the 56 directed rows
# ----------------------------------------------------------------
rows = []
median_matrix = np.full((8, 8), np.nan)
class_matrix = np.empty((8, 8), dtype=object)

for i, ci in enumerate(CITIES):
    for j, cj in enumerate(CITIES):
        if ci == cj:
            class_matrix[i, j] = "SELF"
            continue
        cls = class_of[(ci, cj)]
        class_matrix[i, j] = cls

        if cls == "DUAL":
            rows.append({
                "city_i": ci, "city_j": cj, "class": "DUAL",
                "dual_pct": dual_pct_of[(ci, cj)],
                "n_relay_opportunities": "", "latency_min_min": "",
                "latency_median_min": "", "latency_max_min": "",
                "top5_relay_sats": "", "fastest_sat": "",
                "fastest_latency_min": "", "fastest_pickup_local_i": "",
                "fastest_dropoff_local_j": "",
            })
            continue

        opps = relay_opportunities(ci, cj)
        lats = np.array([o[0] for o in opps])
        freq = defaultdict(int)
        for _, sat_id, _, _ in opps:
            freq[sat_id] += 1
        top5 = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
        fastest = min(opps, key=lambda o: o[0])

        median_matrix[i, j] = float(np.median(lats))
        rows.append({
            "city_i": ci, "city_j": cj, "class": "SF", "dual_pct": "",
            "n_relay_opportunities": len(opps),
            "latency_min_min": round(float(lats.min()), 1),
            "latency_median_min": round(float(np.median(lats)), 1),
            "latency_max_min": round(float(lats.max()), 1),
            "top5_relay_sats": ";".join(f"{s}({n})" for s, n in top5),
            "fastest_sat": fastest[1],
            "fastest_latency_min": round(fastest[0], 1),
            "fastest_pickup_local_i": local_hour_str(fastest[2], ci),
            "fastest_dropoff_local_j": local_hour_str(fastest[3], cj),
        })
        print(f"  {ci:13s} -> {cj:13s}  n={len(opps):5d}  "
              f"median={np.median(lats):7.1f}'  fastest={fastest[0]:6.1f}' "
              f"({fastest[1]}, pickup {local_hour_str(fastest[2], ci)} local)")

assert len(rows) == 56, f"expected 56 directed rows, got {len(rows)}"
save_intermediate_csv(rows, "sf_relay_detail_56pairs",
                       "56 directed city-pair entries: SF relay latency detail "
                       "(min/median/max, opportunities, top-5 relay satellites, "
                       "fastest relay pickup/dropoff local times); DUAL entries "
                       "carry simultaneous-coverage pct from Task 6")

# ----------------------------------------------------------------
# 3. Cross-check medians against the tested module function
# ----------------------------------------------------------------
pass_table = [{"sat_id": r.sat_id, "station": r.station,
               "t_rise": r.t_rise_dt.to_pydatetime(), "t_set": None}
              for r in df.itertuples()]
for ci, cj in [("jakarta", "manila"), ("manila", "jakarta"), ("hanoi", "singapore")]:
    mod_median = sf_latency_minutes(pass_table, ci, cj)
    i, j = CITIES.index(ci), CITIES.index(cj)
    assert abs(mod_median - median_matrix[i, j]) < 0.05, \
        f"median mismatch {ci}->{cj}: script {median_matrix[i,j]:.1f} vs module {mod_median:.1f}"
print("\nCross-check vs modules/citypair_feasibility.sf_latency_minutes: OK (3 spot pairs)")

jm = median_matrix[CITIES.index("jakarta"), CITIES.index("manila")]
mj = median_matrix[CITIES.index("manila"), CITIES.index("jakarta")]
print(f"Asymmetry landmark reproduced: jakarta->manila {jm:.1f}'  vs  manila->jakarta {mj:.1f}'")

# ----------------------------------------------------------------
# 4. Heatmap: 8x8 median latency, DUAL cells masked
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 5.2))
masked = np.ma.masked_invalid(median_matrix)
im = ax.imshow(masked, cmap='YlOrRd', vmin=0)
for i in range(8):
    for j in range(8):
        if i == j:
            continue
        if class_matrix[i, j] == "DUAL":
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        color='#c8e6c9', zorder=2))
            ax.text(j, i, "DUAL", ha='center', va='center', fontsize=6,
                    color='#1b5e20', zorder=3)
        else:
            ax.text(j, i, f"{median_matrix[i, j]:.0f}", ha='center', va='center',
                    fontsize=6.5, zorder=3,
                    color='white' if median_matrix[i, j] > np.nanmax(median_matrix) * 0.6 else 'black')
ax.set_xticks(range(8))
ax.set_xticklabels([CITY_LABELS[c] for c in CITIES], rotation=90)
ax.set_yticks(range(8))
ax.set_yticklabels([CITY_LABELS[c] for c in CITIES])
ax.set_xlabel('Drop-off city $j$')
ax.set_ylabel('Pickup city $i$')
ax.set_title('Directed Store-and-Forward Relay: Median Latency $i \\to j$ (minutes)\n'
             'Green cells: DUAL pairs (simultaneous downlink, no relay needed)',
             fontsize=8.5, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8, label='Median relay latency (min)')
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, 'fig10_sf_relay_latency_heatmap.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out_fig}")

# ----------------------------------------------------------------
# 5. Verify + provenance
# ----------------------------------------------------------------
verify = {
    "n_directed_rows": str(len(rows)),
    "jakarta_to_manila_median_min": f"{jm:.1f}",
    "manila_to_jakarta_median_min": f"{mj:.1f}",
    "asymmetry_ratio": f"{mj/jm:.1f}x",
}
sf_rows = [r for r in rows if r["class"] == "SF"]
worst = max(sf_rows, key=lambda r: r["latency_median_min"])
best = min(sf_rows, key=lambda r: r["latency_median_min"])
verify["sf_best_direction"] = (f"{best['city_i']}->{best['city_j']} "
                                f"median {best['latency_median_min']}'")
verify["sf_worst_direction"] = (f"{worst['city_i']}->{worst['city_j']} "
                                 f"median {worst['latency_median_min']}'")
save_verify_numbers(verify, "sf_relay_detail")

save_provenance(
    script_name="11f_sf_relay_detail",
    params={"n_passes": len(df), "n_directed_entries": 56,
            "n_sf_directed": len(sf_rows), "min_elev_deg": 30.0},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out_fig],
    data_sources={"pass table (Task 5)": PASS_CSV,
                  "feasibility classes + dual_pct (Task 6)": FEAS_CSV},
    formulas={
        "Relay opportunity": "pass of sat s over city i (t_rise) -> SAME satellite's "
                             "next pass over city j (t_rise); no inter-satellite links; "
                             "identical definition to modules/citypair_feasibility."
                             "sf_latency_minutes (cross-checked on 3 spot pairs)",
        "Directionality": "i->j and j->i computed independently (07-5 Task 6 design fix); "
                          "landmark asymmetry jakarta/manila must reproduce 7.0'/711.4'",
    },
)
print(f"\nTotal runtime: {time.time() - t0:.1f}s")
