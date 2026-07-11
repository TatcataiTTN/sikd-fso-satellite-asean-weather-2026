"""
12b_isl_relay_recompute.py — ISL Multi-Hop Relay Recompute, 28 Directed SF Pairs (Task 24.3)
================================================================================
Computes store-and-forward relay latency between city pairs using the ISL
(inter-satellite link) multi-hop mesh model (modules/isl_relay.py), which
replaces the single-hop model of Task 6/23 (modules/citypair_feasibility.py
-- that older model required the SAME satellite to physically revisit
both cities, forcing an unrealistic wait of up to ~700 minutes in the
worst case). This is now the ONLY relay model this project reports
(decision 08/07/2026: the single-hop legacy comparison was dropped --
only the corrected ISL model matters going forward).

Two sub-algorithms per directed pair, both drawing from the same feasible-
candidate set built via BFS over the ISL mesh (modules/isl_topology.py):
  time_optimal_relay      minimize wall-clock delivery latency.
  capacity_optimal_relay  within a latency budget, pick the delivery pass
                         with the best elevation (-> highest SKR/capacity).

Sampling (NOT full enumeration -- documented honestly, mirroring the
Task 21 v1-v3-vs-v4 lesson about being upfront when sampling): for each
origin city, up to N_SAMPLE_PICKUPS=200 pickup events are drawn
SYSTEMATICALLY (evenly spaced through that city's full 7-day pass list,
sorted by t_rise) rather than exhaustively enumerating every pickup
(~10,500 passes/city over 7 days would make this a much longer run). The
same 200-pickup sample per origin is reused across all of that origin's
SF-classified destinations, and the BFS hop-distance (which depends only
on the pickup satellite, not on time or destination) is cached per unique
satellite to avoid redundant graph traversals.

Run:
  cd 05_Code_v2 && python scripts/12b_isl_relay_recompute.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.orbital_mechanics import parse_tle_block, GROUND_STATIONS
from modules.isl_topology import build_isl_graph
from modules.isl_relay import bfs_hop_distance, time_optimal_relay, capacity_optimal_relay
from utils import save_provenance, save_intermediate_csv, save_verify_numbers, INTER as _INTER

plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 9, 'font.family': 'serif',
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
})

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


CITIES = list(GROUND_STATIONS.keys())
CITY_LABELS = {
    'hanoi': 'Hanoi', 'danang': 'Da Nang', 'hcmc': 'HCMC', 'bangkok': 'Bangkok',
    'singapore': 'Singapore', 'manila': 'Manila', 'jakarta': 'Jakarta',
    'kuala_lumpur': 'Kuala Lumpur',
}

TLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'starlink_shell1_real_tle.txt')
if os.environ.get('SIKD_TLE_PATH'):
    TLE_PATH = os.environ['SIKD_TLE_PATH']
_VARIANT = os.environ.get('SIKD_VARIANT_DIR')
CANONICAL_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'latex_paper_3', 'figures')
FIG_DIR = (os.path.join(_VARIANT, 'figures') if _VARIANT else CANONICAL_FIG_DIR)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CANONICAL_FIG_DIR, exist_ok=True)

PASS_CSV = os.path.join(str(_INTER), 'pass_table_8cities_7days_elev30.csv')
FEAS_CSV = os.path.join(str(_INTER), 'citypair_feasibility_matrix.csv')       # Task 6, still valid (DUAL/SF class is static geometry, unaffected by the relay-model fix)

BIN_WIDTH_DEG = 2.0             # same choice as 12a, justified there
ISL_HOP_LATENCY_S = 30.0        # 0.5 min/hop, conservative -- see isl_relay.py docstring
LATENCY_BUDGET_FACTOR = 3.0
N_SAMPLE_PICKUPS = 200

# ----------------------------------------------------------------
# 1. Load graph + pass table + feasibility classes (Task 6, still valid)
# ----------------------------------------------------------------
tle_dicts = parse_tle_block(open(TLE_PATH).read())
log(f"Loaded {len(tle_dicts)} real Shell-1 satellites")

graph = build_isl_graph(tle_dicts, bin_width_deg=BIN_WIDTH_DEG)
degrees = np.array([len(v) for v in graph.values()])
log(f"ISL graph: {len(graph)} nodes, degree median={np.median(degrees):.0f}")

df = pd.read_csv(PASS_CSV, comment='#')
df['t_rise_dt'] = pd.to_datetime(df['t_rise'], format='ISO8601')
df['t_set_dt'] = pd.to_datetime(df['t_set'], format='ISO8601')
log(f"Loaded {len(df)} passes (Task 5, elev>=30, 7 days)")

feas = pd.read_csv(FEAS_CSV, comment='#')
class_of = {(r['city_i'], r['city_j']): r['class'] for _, r in feas.iterrows()}

by_station = {city: df[df['station'] == city].sort_values('t_rise_dt').reset_index(drop=True)
             for city in CITIES}

# ----------------------------------------------------------------
# 2. Compute each directed SF pair
# ----------------------------------------------------------------
hop_cache = {}


def get_hops(pickup_sat):
    if pickup_sat not in hop_cache:
        hop_cache[pickup_sat] = bfs_hop_distance(graph, pickup_sat)
    return hop_cache[pickup_sat]


rows = []
for ci in CITIES:
    sub_i = by_station[ci]
    if len(sub_i) == 0:
        continue
    n_sample = min(N_SAMPLE_PICKUPS, len(sub_i))
    sample_idx = np.linspace(0, len(sub_i) - 1, n_sample).round().astype(int)
    sample_idx = np.unique(sample_idx)
    pickups = [(r.sat_id, r.t_rise_dt) for r in sub_i.iloc[sample_idx].itertuples()]

    destinations = [cj for cj in CITIES if cj != ci and class_of.get((ci, cj)) == "SF"]
    if not destinations:
        continue

    for cj in destinations:
        sub_j = by_station[cj]
        pass_table_j = [
            {"sat_id": r.sat_id, "t_rise": r.t_rise_dt, "t_set": r.t_set_dt,
             "max_elev_deg": r.max_elev_deg}
            for r in sub_j.itertuples()
        ]

        time_lat, time_hops = [], []
        cap_lat, cap_hops, cap_elev, cap_gbit = [], [], [], []
        n_no_candidate = 0
        diff_sat_count = 0
        for pickup_sat, t_pick in pickups:
            hops = get_hops(pickup_sat)
            t_opt = time_optimal_relay(pickup_sat, t_pick, graph, pass_table_j,
                                       hop_latency_s=ISL_HOP_LATENCY_S, precomputed_hops=hops)
            c_opt = capacity_optimal_relay(pickup_sat, t_pick, graph, pass_table_j,
                                          hop_latency_s=ISL_HOP_LATENCY_S,
                                          latency_budget_factor=LATENCY_BUDGET_FACTOR,
                                          precomputed_hops=hops)
            if t_opt is None or c_opt is None:
                n_no_candidate += 1
                continue
            time_lat.append(t_opt["latency_min"])
            time_hops.append(t_opt["n_hops"])
            cap_lat.append(c_opt["latency_min"])
            cap_hops.append(c_opt["n_hops"])
            cap_elev.append(c_opt["elev_deg"])
            cap_gbit.append(c_opt["delivered_gbit"])
            # Per-PICKUP satellite identity comparison -- a median-n_hops
            # comparison across the aggregate is a weak/misleading proxy
            # here (median n_hops turned out nearly identical, ~4 vs ~4.4,
            # for BOTH algorithms on real data, which does NOT mean they
            # picked the same satellite each time; two different
            # satellites can easily sit at the same hop distance). Track
            # the actual sat_id match per pickup event instead.
            diff_sat_count += int(t_opt["sat_id"] != c_opt["sat_id"])

        n_ok = len(time_lat)
        pct_diff_sat = round(100.0 * diff_sat_count / n_ok, 1) if n_ok else ""
        rows.append({
            "city_i": ci, "city_j": cj,
            "n_pickups_sampled": len(pickups),
            "n_pickups_with_candidate": n_ok,
            "latency_time_optimal_min": round(float(np.median(time_lat)), 2) if n_ok else "",
            "n_hops_time_optimal": round(float(np.median(time_hops)), 1) if n_ok else "",
            "latency_capacity_optimal_min": round(float(np.median(cap_lat)), 2) if n_ok else "",
            "n_hops_capacity_optimal": round(float(np.median(cap_hops)), 1) if n_ok else "",
            "elev_deg_capacity_optimal": round(float(np.median(cap_elev)), 1) if n_ok else "",
            "delivered_gbit_capacity_optimal": round(float(np.median(cap_gbit)), 4) if n_ok else "",
            "pct_pickups_diff_satellite": pct_diff_sat,
        })
        log(f"  {ci:13s} -> {cj:13s}  n_ok={n_ok:3d}/{len(pickups):3d}  "
            f"time_opt_median={np.median(time_lat) if n_ok else float('nan'):7.2f}'  "
            f"cap_opt_median={np.median(cap_lat) if n_ok else float('nan'):7.2f}' "
            f"(elev {np.median(cap_elev) if n_ok else float('nan'):5.1f} deg)  "
            f"diff_sat={pct_diff_sat}%")

assert len(rows) == 28, f"expected 28 directed SF entries, got {len(rows)}"
save_intermediate_csv(
    rows, "sf_relay_detail_56pairs_isl",
    f"ISL multi-hop relay (Task 24.3) for the 28 directed SF entries "
    f"(DUAL pairs unaffected, class from Task 6) -- SAMPLE of "
    f"{N_SAMPLE_PICKUPS} pickups/origin (not full enumeration), "
    f"hop_latency_s={ISL_HOP_LATENCY_S}, bin_width_deg={BIN_WIDTH_DEG}. "
    f"pct_pickups_diff_satellite = %% of sampled pickups where time_optimal "
    f"and capacity_optimal picked a DIFFERENT satellite (the correct way to "
    f"check the two algorithms actually diverge; comparing median n_hops "
    f"across the aggregate is NOT reliable -- two different satellites can "
    f"sit at the same hop distance).",
)
v2 = pd.DataFrame(rows)
v2['latency_time_optimal_min'] = pd.to_numeric(v2['latency_time_optimal_min'], errors='coerce')

# ----------------------------------------------------------------
# 3. fig10 — ISL time-optimal latency heatmap (8x8)
# ----------------------------------------------------------------
median_matrix = np.full((8, 8), np.nan)
class_matrix = np.empty((8, 8), dtype=object)
for i, ci in enumerate(CITIES):
    for j, cj in enumerate(CITIES):
        if ci == cj:
            class_matrix[i, j] = "SELF"
            continue
        cls = class_of.get((ci, cj))
        class_matrix[i, j] = cls
        if cls == "SF":
            row = v2[(v2.city_i == ci) & (v2.city_j == cj)]
            if len(row) and pd.notna(row.iloc[0]['latency_time_optimal_min']):
                median_matrix[i, j] = row.iloc[0]['latency_time_optimal_min']

fig, ax = plt.subplots(figsize=(6.0, 5.2))
masked = np.ma.masked_invalid(median_matrix)
im = ax.imshow(masked, cmap='YlOrRd', vmin=0)
for i in range(8):
    for j in range(8):
        if i == j:
            continue
        if class_matrix[i, j] == "DUAL":
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#c8e6c9', zorder=2))
            ax.text(j, i, "DUAL", ha='center', va='center', fontsize=6, color='#1b5e20', zorder=3)
        elif not np.isnan(median_matrix[i, j]):
            ax.text(j, i, f"{median_matrix[i, j]:.1f}", ha='center', va='center',
                    fontsize=6.5, zorder=3,
                    color='white' if median_matrix[i, j] > np.nanmax(median_matrix) * 0.6 else 'black')
ax.set_xticks(range(8)); ax.set_xticklabels([CITY_LABELS[c] for c in CITIES], rotation=90)
ax.set_yticks(range(8)); ax.set_yticklabels([CITY_LABELS[c] for c in CITIES])
ax.set_xlabel('Drop-off city $j$'); ax.set_ylabel('Pickup city $i$')
ax.set_title('ISL Multi-Hop Relay: Time-Optimal Median Latency $i \\to j$ (minutes)\n'
             'Green cells: DUAL pairs (simultaneous downlink, no relay needed)',
             fontsize=8.5, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8, label='Median relay latency (min)')
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, 'fig10_isl_relay_latency_heatmap.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
log(f"Saved: {out_fig}")

# ----------------------------------------------------------------
# 4. Verify + provenance
# ----------------------------------------------------------------
jm_row = v2[(v2.city_i == 'jakarta') & (v2.city_j == 'manila')]
mj_row = v2[(v2.city_i == 'manila') & (v2.city_j == 'jakarta')]
verify = {
    "n_directed_sf_pairs": str(len(v2)),
    "n_sample_pickups_per_origin": str(N_SAMPLE_PICKUPS),
    "isl_hop_latency_s": str(ISL_HOP_LATENCY_S),
    "median_latency_time_optimal_min": f"{v2['latency_time_optimal_min'].median():.2f}",
    "min_latency_time_optimal_min": f"{v2['latency_time_optimal_min'].min():.2f}",
    "max_latency_time_optimal_min": f"{v2['latency_time_optimal_min'].max():.2f}",
    "median_pct_pickups_diff_satellite": f"{pd.to_numeric(v2['pct_pickups_diff_satellite'], errors='coerce').median():.1f}",
    "n_pairs_with_zero_diff_satellite": str(int((pd.to_numeric(v2['pct_pickups_diff_satellite'], errors='coerce') == 0).sum())),
}
if len(jm_row):
    verify["jakarta_to_manila_time_optimal_min"] = f"{jm_row.iloc[0]['latency_time_optimal_min']:.2f}"
if len(mj_row):
    verify["manila_to_jakarta_time_optimal_min"] = f"{mj_row.iloc[0]['latency_time_optimal_min']:.2f}"
save_verify_numbers(verify, "isl_relay_recompute")

save_provenance(
    script_name="12b_isl_relay_recompute",
    params={"isl_hop_latency_s": ISL_HOP_LATENCY_S, "bin_width_deg": BIN_WIDTH_DEG,
            "latency_budget_factor": LATENCY_BUDGET_FACTOR,
            "n_sample_pickups_per_origin": N_SAMPLE_PICKUPS,
            "sampling_note": "SYSTEMATIC evenly-spaced sample, NOT full enumeration "
                            "-- mirrors the Task 21 lesson about being upfront when "
                            "sampling (07-5 Phase G); full enumeration is a natural "
                            "follow-up if the paper needs it"},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out_fig],
    data_sources={"pass table (Task 5)": PASS_CSV,
                  "feasibility classes (Task 6)": FEAS_CSV,
                  "Shell-1 TLE": TLE_PATH},
    formulas={
        "time_optimal_relay": "BFS hop-distance from pickup satellite; earliest "
                              "feasible drop-off pass over city_j (modules/isl_relay.py)",
        "capacity_optimal_relay": "among candidates within latency_budget_factor x "
                                 "time-optimal latency, pick highest delivered_gbit "
                                 "(elevation -> channel_model -> sikd_performance)",
    },
)

if _VARIANT and os.path.abspath(FIG_DIR) != os.path.abspath(CANONICAL_FIG_DIR):
    import shutil
    shutil.copy2(out_fig, os.path.join(CANONICAL_FIG_DIR, os.path.basename(out_fig)))
    log(f"Also copied figure to canonical {CANONICAL_FIG_DIR}")

log(f"Total runtime: {time.time() - t0:.1f}s")
