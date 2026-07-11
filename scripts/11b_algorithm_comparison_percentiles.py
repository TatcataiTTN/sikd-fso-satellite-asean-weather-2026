"""
11b_algorithm_comparison_percentiles.py — Fair ALG-0/1/2 Comparison, v2 (Task 14 extension)
================================================================================================
VERSION 2 of the Task 14 comparison (see 11_algorithm_comparison.py for v1
and its full design rationale, unchanged here). v1's min-pair-key metric
was found to be exactly 0 in 83-88% of real-day draws: a single hub
station (in 7 of 28 pairs) having a fully-clouded real day zeroes every
pair through it (a genuine physical limit, not a bug). Since "the worst
pair" collapses to 0 almost always, it cannot distinguish between
algorithms. This version ADDITIONALLY records the 10th-percentile and
median of the 28 pair-key values per draw -- a fairness measure that is
not dominated by a single hub-outage day -- alongside everything v1
already computed. No scheduling logic changes from v1; this is purely an
additional metric extracted from the SAME ALG-2 allocation.

Outputs go to temp/v2_percentiles/ (kept separate from v1's
temp/v1_throughput_only/ per user request to preserve all three analysis
versions distinctly).

--- v1 docstring below, unchanged ---

The main result of the paper redesign: a FAIR comparison of the three
scheduling algorithms (Section~IV/V of the paper), replacing the invalid
"+678%" 8-vs-1-station comparison (07-4). Every algorithm here runs on the
SAME 8 stations, the SAME real 1,019-satellite Shell-1 constellation, and
the SAME 24-hour window (2026-03-12, elev >= 40 deg, Mask B per Task 7's
security decision).

Algorithms compared
--------------------
  ALG-0          schedule_baseline: sticky, weather-blind, per-station.
  ALG-1-blind    match_weather_aware with weight = elevation (not SKR_eff):
                 isolates the value of MATCHING (no double-booking) alone,
                 with no weather information at all.
  ALG-1-aware    match_weather_aware with weight = real SKR_eff: adds
                 weather-awareness on top of matching.
  ALG-2          schedule_pairs_greedy: pass-level allocation to the 28
                 ASEAN city pairs under the trusted-relay model (Eq.
                 pair_key). A fundamentally different metric (pair-usable
                 key after the min() bottleneck) from the station-level
                 raw generation of ALG-0/1 -- both are reported, not
                 conflated (this is the "phân rã" the task asks for).

Key computational insight (why this fits in the time budget): ALG-0's and
ALG-1-blind's SCHEDULING DECISIONS do not depend on weather at all (only on
elevation, which is fixed real orbital geometry). Their assignment
schedules are therefore computed ONCE and reused across all Monte Carlo
weather draws; only the resulting KEY/DATA METRICS are re-evaluated per
draw against that draw's real bootstrapped weather. Only ALG-1-aware needs
genuine re-scheduling per draw (its decisions DO depend on weather).
Benchmarked: ~16 us per matching call, ~100 us per channel+SKR evaluation
-- projects to ~11 minutes for 400 total draws, well inside the 30-minute
budget.

Weather: Monte Carlo bootstrap of REAL historical days (not the monthly
climatological mean). Each draw picks ONE real calendar date (from the
2015-2024 record, filtered to the target month) SHARED across all 8
cities, preserving the real inter-city weather correlation of that actual
day (Task 2/3 finding: correlation is real and sometimes strongly
negative, e.g. Hanoi-Jakarta in July). Each city's ACTUAL hourly cloud
cover and precipitation for that date are used directly -- a genuine
per-hour realization, not the p_clear/p_rain expectation used elsewhere in
this project for monthly-average reporting.

Run:
  cd 05_Code_v2 && python scripts/11b_algorithm_comparison_percentiles.py
"""
import os
import sys
import time
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.orbital_mechanics import parse_tle_block, make_skyfield_satellite, GROUND_STATIONS
from modules.pass_analysis import extract_passes, UTC_OFFSET_HOURS
from modules.channel_model import compute_channel
from modules.sikd_performance import compute_sikd_performance
from modules.schedulers import (
    schedule_baseline, match_weather_aware, schedule_pairs_greedy, pair_key_totals,
)
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 9, 'font.family': 'serif',
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
})

def log(msg):
    """Print with a timestamp and force-flush, so progress is visible
    immediately even when this script's stdout is redirected to a file by
    a background runner (which would otherwise buffer output)."""
    elapsed = time.time() - t0
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


t0 = time.time()
random.seed(42)

TEMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp', 'v2_percentiles')
os.makedirs(TEMP_DIR, exist_ok=True)
RAW_RUNS_CSV = os.path.join(TEMP_DIR, 'raw_runs.csv')
PROGRESS_JSON = os.path.join(TEMP_DIR, 'progress.json')

# Fresh raw_runs.csv for this invocation (each run's numbers appended live below)
with open(RAW_RUNS_CSV, 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow([
        "season", "run", "date_str",
        "alg0_gbit", "alg1blind_gbit", "alg1aware_gbit", "alg2_pairkey_gbit",
        "alg0_handovers", "alg1blind_handovers", "alg1aware_handovers",
        "alg2_min_pair_gbit", "outage_bridged_pct",
        # v2 additions: fairness measures across the 28 pair-key values that
        # are not dominated by a single hub-outage day the way min() is.
        "alg2_p10_pair_gbit", "alg2_median_pair_gbit", "alg2_max_pair_gbit",
    ])


def append_run_row(row):
    with open(RAW_RUNS_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


def write_progress(season, run, n_total, extra=None):
    payload = {
        "season": season, "run": run + 1, "n_total": n_total,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if extra:
        payload.update(extra)
    with open(PROGRESS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

CITIES = list(GROUND_STATIONS.keys())
PAIRS = [(CITIES[i], CITIES[j]) for i in range(8) for j in range(i + 1, 8)]  # 28 pairs
H_KM = 550.0
MIN_ELEV_DEG = 40.0  # Mask B (Task 7): secure default for main results
T_START = datetime(2026, 3, 12, 0, 0, 0, tzinfo=timezone.utc)
DURATION_HOURS = 24.0
SEASONS = {"dry": 1, "wet": 7}
N_RUNS_PER_SEASON = 200
CLOUD_OUTAGE_THRESHOLD = 85.0

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'starlink_shell1_real_tle.txt')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'latex_paper_3', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------
# 1. Fixed orbital template: real passes, 24h window, elev >= 40 deg
# ----------------------------------------------------------------
tle_dicts = parse_tle_block(open(TLE_PATH).read())
satellites = [make_skyfield_satellite(d) for d in tle_dicts]
log(f"Loaded {len(satellites)} real Shell-1 satellites")

all_passes = []
for city in CITIES:
    passes = extract_passes(satellites, city, T_START, DURATION_HOURS, min_elev_deg=MIN_ELEV_DEG)
    for p in passes:
        p['station'] = city
    all_passes.extend(passes)
log(f"Built {len(all_passes)} real passes @ {MIN_ELEV_DEG:.0f} deg mask, 24h window (fixed orbital template)")

events = []
for p in all_passes:
    events.append((p['t_rise'], +1, p['station'], p['sat_id'], p['max_elev_deg']))
    events.append((p['t_set'], -1, p['station'], p['sat_id'], p['max_elev_deg']))
events.sort(key=lambda e: e[0])

# ----------------------------------------------------------------
# 2. Load raw hourly weather for all 8 cities, indexed by exact date
# ----------------------------------------------------------------
def load_raw_by_date(city):
    path = os.path.join(RAW_DIR, f"era5_hourly_{city}.csv.gz")
    df = pd.read_csv(path, compression='gzip', parse_dates=['time_local'])
    df['date'] = df['time_local'].dt.date.astype(str)
    df['hour'] = df['time_local'].dt.hour
    by_date = {}
    for date_str, grp in df.groupby('date'):
        cloud = np.zeros(24)
        precip = np.zeros(24)
        for _, row in grp.iterrows():
            cloud[row['hour']] = row['cloud_cover_pct']
            precip[row['hour']] = row['precip_mm']
        by_date[date_str] = (cloud, precip)
    return by_date


log("\nLoading raw hourly weather (10-year record) for bootstrap sampling...")
raw_by_date = {}
for city in CITIES:
    tcity = time.time()
    raw_by_date[city] = load_raw_by_date(city)
    log(f"  {city:14s} loaded ({len(raw_by_date[city])} dates, {time.time()-tcity:.1f}s)")
log("All 8 cities loaded.")


def sample_real_day(month):
    """One real calendar date (shared across all 8 cities -> preserves
    real inter-city correlation), drawn uniformly from 2015-2024."""
    year = random.randint(2015, 2024)
    day = random.randint(1, 28)  # 28 is valid for every month, incl. Feb; simplicity over exhaustive day-31 coverage
    return f"{year}-{month:02d}-{day:02d}"


# ----------------------------------------------------------------
# 3. Per-draw weight/channel lookup with simple caching
# ----------------------------------------------------------------
def make_weight_fn(date_str):
    """Returns weight_fn(station, elev_deg, local_hour) -> SKR_kbps for
    ONE bootstrapped real day, using that city's ACTUAL hourly cloud cover
    and precipitation on that date (not the monthly climatological mean)."""
    cache = {}

    def weight_fn(station, elev_deg, local_hour):
        hour_bin = int(local_hour) % 24
        elev_r = round(elev_deg)
        key = (station, elev_r, hour_bin)
        if key in cache:
            return cache[key]

        cloud_arr, precip_arr = raw_by_date[station].get(date_str, (None, None))
        if cloud_arr is None:
            cache[key] = 0.0
            return 0.0

        cloud_pct = cloud_arr[hour_bin]
        if cloud_pct >= CLOUD_OUTAGE_THRESHOLD:
            cache[key] = 0.0
            return 0.0

        v_km = max(15.0 - 10.0 * (cloud_pct / 100.0), 0.5)
        r_mm_h = float(precip_arr[hour_bin])
        zenith = 90.0 - elev_r
        ch = compute_channel(H_KM, zenith, V_km=v_km, R_mm_h=r_mm_h)
        perf = compute_sikd_performance(ch['hg'], ch['hl'], ch['sigma_X2'])
        cache[key] = perf['SKR_kbps']
        return perf['SKR_kbps']

    return weight_fn


def local_hour_of(dt, station):
    offset = UTC_OFFSET_HOURS[station]
    return (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + offset) % 24.0


# ----------------------------------------------------------------
# 4. Event sweep: ALG-0 and ALG-1-blind schedules (weather-independent,
#    computed ONCE); ALG-1-aware re-scheduled per draw.
# ----------------------------------------------------------------
def sweep_segments(events, use_matching, weight_of=None, handover_cost=0.0):
    """Return list of (t_start, t_end, assignment_dict) segments, where
    assignment_dict is {station: sat_id or None}, from the event sweep.
    weight_of(t, station, sat_id, elev_deg) -> float, used only if
    use_matching=True (schedule_baseline never looks at weight). `t` is
    the CURRENT event time, passed through so a weather-aware weight_of
    can look up the correct local hour at that specific moment (a bug
    caught before the full run: using a fixed reference time here would
    have frozen the local hour for the whole 24h sweep)."""
    open_by_station = defaultdict(dict)
    prev = None
    cur_assignment = {}
    segments = []
    last_t = events[0][0]

    for t, typ, station, sat_id, elev in events:
        if t > last_t:
            segments.append((last_t, t, cur_assignment))
        if typ == 1:
            open_by_station[station][sat_id] = elev
        else:
            open_by_station[station].pop(sat_id, None)

        visibility = {s: [{"sat_id": sid, "elev_deg": e} for sid, e in sats.items()]
                     for s, sats in open_by_station.items()}
        if use_matching:
            weights = {(s, sid): weight_of(t, s, sid, e)
                      for s, sats in open_by_station.items() for sid, e in sats.items()}
            cur_assignment = match_weather_aware(visibility, weights, prev=prev,
                                                 handover_cost=handover_cost)
        else:
            cur_assignment = schedule_baseline(visibility, prev=prev)
        prev = cur_assignment
        last_t = t

    return segments


log("\nPrecomputing ALG-0 and ALG-1-blind schedules (weather-independent, once)...")
tsched0 = time.time()
segments_alg0 = sweep_segments(events, use_matching=False)
segments_alg1_blind = sweep_segments(events, use_matching=True, weight_of=lambda t, s, sid, e: e)
log(f"  done in {time.time() - tsched0:.1f}s "
      f"({len(segments_alg0)} ALG-0 segments, {len(segments_alg1_blind)} ALG-1-blind segments)")


def evaluate_segments(segments, weight_fn):
    """Total network key (bits) delivered by a FIXED segment schedule,
    evaluated against ONE draw's realized weather; also per-station
    breakdown and handover count."""
    total_bits = 0.0
    per_station_bits = defaultdict(float)
    n_handovers = 0
    prev_sat = {}

    for t_start, t_end, assignment in segments:
        duration_s = (t_end - t_start).total_seconds()
        for station, sat_id in assignment.items():
            if sat_id is None:
                continue
            if station in prev_sat and prev_sat[station] != sat_id:
                n_handovers += 1
            prev_sat[station] = sat_id

            # Need the elevation of sat_id for this station at this segment;
            # recover from the pass list (max_elev_deg used as representative).
            elev = elev_lookup.get((station, sat_id), None)
            if elev is None:
                continue
            local_hour = local_hour_of(t_start, station)
            skr_kbps = weight_fn(station, elev, local_hour)
            bits = skr_kbps * 1e3 * duration_s
            total_bits += bits
            per_station_bits[station] += bits

    return total_bits, per_station_bits, n_handovers


elev_lookup = {(p['station'], p['sat_id']): p['max_elev_deg'] for p in all_passes}

# ----------------------------------------------------------------
# 5. Monte Carlo loop
# ----------------------------------------------------------------
results = {season: {"ALG-0": [], "ALG-1-blind": [], "ALG-1-aware": [], "ALG-2": []}
          for season in SEASONS}
handover_counts = {season: {"ALG-0": [], "ALG-1-blind": [], "ALG-1-aware": []} for season in SEASONS}
pair_min_key = {season: [] for season in SEASONS}
pair_p10_key = {season: [] for season in SEASONS}
pair_median_key = {season: [] for season in SEASONS}
pair_max_key = {season: [] for season in SEASONS}
outage_bridged_pct = {season: [] for season in SEASONS}

for season, month in SEASONS.items():
    log(f"\n=== {season.upper()} (month={month}) — {N_RUNS_PER_SEASON} Monte Carlo draws ===")
    trun0 = time.time()

    for run in range(N_RUNS_PER_SEASON):
        date_str = sample_real_day(month)
        weight_fn = make_weight_fn(date_str)

        bits0, per_st0, hov0 = evaluate_segments(segments_alg0, weight_fn)
        bits1b, per_st1b, hov1b = evaluate_segments(segments_alg1_blind, weight_fn)

        segments_alg1_aware = sweep_segments(
            events, use_matching=True,
            weight_of=lambda t, s, sid, e: weight_fn(s, e, local_hour_of(t, s)),
        )
        bits1a, per_st1a, hov1a = evaluate_segments(segments_alg1_aware, weight_fn)

        pass_rows = []
        for p in all_passes:
            skr_kbps = weight_fn(p['station'], p['max_elev_deg'], p['local_hour_peak'])
            duration_s = (p['t_set'] - p['t_rise']).total_seconds()
            pass_rows.append({
                "pass_id": f"{p['sat_id']}_{p['station']}_{p['t_rise'].isoformat()}",
                "station": p['station'],
                "key_bits": skr_kbps * 1e3 * duration_s,
                "t": p['t_rise'],
            })
        alloc2 = schedule_pairs_greedy(pass_rows, PAIRS, demands=None)
        totals2 = pair_key_totals(alloc2, pass_rows)
        bits2 = sum(totals2.values())
        pair_vals = np.array(list(totals2.values())) if totals2 else np.zeros(len(PAIRS))
        min_pair_key = float(pair_vals.min()) if len(pair_vals) else 0.0
        # v2: fairness measures not dominated by a single hub-outage day
        p10_pair_key = float(np.percentile(pair_vals, 10)) if len(pair_vals) else 0.0
        median_pair_key = float(np.median(pair_vals)) if len(pair_vals) else 0.0
        max_pair_key = float(pair_vals.max()) if len(pair_vals) else 0.0

        n_outage_hours = sum(
            1 for city in CITIES for h in range(24)
            if raw_by_date[city].get(date_str, (None,))[0] is not None
            and raw_by_date[city][date_str][0][h] >= CLOUD_OUTAGE_THRESHOLD
        )
        outage_bridged_pct[season].append(100.0 * (1 - n_outage_hours / (24 * 8)))

        results[season]["ALG-0"].append(bits0 / 1e9)
        results[season]["ALG-1-blind"].append(bits1b / 1e9)
        results[season]["ALG-1-aware"].append(bits1a / 1e9)
        results[season]["ALG-2"].append(bits2 / 1e9)
        handover_counts[season]["ALG-0"].append(hov0)
        handover_counts[season]["ALG-1-blind"].append(hov1b)
        handover_counts[season]["ALG-1-aware"].append(hov1a)
        pair_min_key[season].append(min_pair_key / 1e9)
        pair_p10_key[season].append(p10_pair_key / 1e9)
        pair_median_key[season].append(median_pair_key / 1e9)
        pair_max_key[season].append(max_pair_key / 1e9)

        # Incremental checkpoint: append this run's raw numbers to
        # temp/v2_percentiles/raw_runs.csv immediately, so partial results
        # are inspectable at any time even if the run is interrupted or
        # still in progress.
        append_run_row([
            season, run + 1, date_str,
            round(bits0 / 1e9, 6), round(bits1b / 1e9, 6),
            round(bits1a / 1e9, 6), round(bits2 / 1e9, 6),
            hov0, hov1b, hov1a,
            round(min_pair_key / 1e9, 6), round(outage_bridged_pct[season][-1], 2),
            round(p10_pair_key / 1e9, 6), round(median_pair_key / 1e9, 6),
            round(max_pair_key / 1e9, 6),
        ])

        per_run_s = (time.time() - trun0) / (run + 1)
        eta_s = per_run_s * (N_RUNS_PER_SEASON - run - 1)
        write_progress(season, run, N_RUNS_PER_SEASON, extra={
            "per_run_s": round(per_run_s, 3), "eta_this_season_s": round(eta_s, 1),
            "alg0_gbit_so_far_mean": round(float(np.mean(results[season]["ALG-0"])), 3),
            "alg1aware_gbit_so_far_mean": round(float(np.mean(results[season]["ALG-1-aware"])), 3),
        })

        # Print every run so progress is visible continuously (not just
        # every 50th), important when this script runs in the background.
        log(f"  run {run+1}/{N_RUNS_PER_SEASON}  date={date_str}  "
            f"ALG0={bits0/1e9:7.2f}  ALG1b={bits1b/1e9:7.2f}  ALG1a={bits1a/1e9:7.2f}  "
            f"ALG2={bits2/1e9:7.2f} Gbit  ETA={eta_s:5.0f}s")

    log(f"  {season} done in {time.time()-trun0:.1f}s")

# ----------------------------------------------------------------
# 6. Table: mean +/- std per ALG per season
# ----------------------------------------------------------------
table_rows = []
log("\n=== Summary: network key (Gbit/day), mean +/- std ===")
for season in SEASONS:
    for alg in ["ALG-0", "ALG-1-blind", "ALG-1-aware", "ALG-2"]:
        vals = np.array(results[season][alg])
        row = {"season": season, "algorithm": alg,
               "mean_gbit_per_day": round(float(vals.mean()), 4),
               "std_gbit_per_day": round(float(vals.std()), 4)}
        table_rows.append(row)
        log(f"  {season:5s} {alg:14s} {vals.mean():8.4f} +/- {vals.std():.4f} Gbit/day")

save_intermediate_csv(table_rows, "algorithm_comparison_summary_v2_percentiles",
                       "v2: Mean +/- std network key (Gbit/day) per algorithm per season, "
                       "200 Monte Carlo draws of real historical days")

# v2: pair fairness table (min/p10/median/max across the 28 pairs, per season)
fairness_rows = []
for season in SEASONS:
    fairness_rows.append({
        "season": season,
        "min_pair_mean_gbit": round(float(np.mean(pair_min_key[season])), 4),
        "p10_pair_mean_gbit": round(float(np.mean(pair_p10_key[season])), 4),
        "median_pair_mean_gbit": round(float(np.mean(pair_median_key[season])), 4),
        "max_pair_mean_gbit": round(float(np.mean(pair_max_key[season])), 4),
        "pct_draws_min_pair_zero": round(100.0 * float(np.mean(np.array(pair_min_key[season]) == 0)), 1),
    })
    log(f"  {season} fairness: min={np.mean(pair_min_key[season]):.3f}  "
        f"p10={np.mean(pair_p10_key[season]):.3f}  median={np.mean(pair_median_key[season]):.3f}  "
        f"max={np.mean(pair_max_key[season]):.3f} Gbit  "
        f"(min==0 in {100.0*np.mean(np.array(pair_min_key[season])==0):.1f}% of draws)")
save_intermediate_csv(fairness_rows, "algorithm_comparison_fairness_v2_percentiles",
                       "v2: pair-key fairness distribution (min/p10/median/max across 28 pairs)")

# ----------------------------------------------------------------
# 7. CDF figure
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True)
for ax, season in zip(axes, SEASONS):
    for alg, color in zip(["ALG-0", "ALG-1-blind", "ALG-1-aware", "ALG-2"],
                          ['#888888', '#2196F3', '#1f3d7a', '#FF5722']):
        vals = np.sort(results[season][alg])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=alg, color=color, linewidth=1.2)
    ax.set_xlabel('Network key (Gbit/day)')
    ax.set_title(season.capitalize(), fontsize=8.5, fontweight='bold')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('CDF')
axes[0].legend(fontsize=6.5, loc='lower right')
plt.suptitle('Algorithm Comparison: CDF of Network Key Delivery (200 real-day MC draws)',
             fontsize=9, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(FIG_DIR, 'fig08b_algorithm_comparison_cdf_v2.png')
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
log(f"\nSaved: {out1}")

# ----------------------------------------------------------------
# 8. Verify + provenance
# ----------------------------------------------------------------
verify = {}
for season in SEASONS:
    m0 = np.mean(results[season]["ALG-0"])
    m1b = np.mean(results[season]["ALG-1-blind"])
    m1a = np.mean(results[season]["ALG-1-aware"])
    m2 = np.mean(results[season]["ALG-2"])
    matching_gain_pct = 100.0 * (m1b - m0) / m0 if m0 > 0 else float('nan')
    weather_gain_pct = 100.0 * (m1a - m1b) / m1b if m1b > 0 else float('nan')
    verify[f"{season}_alg0_mean_gbit_day"] = f"{m0:.4f}"
    verify[f"{season}_alg1blind_mean_gbit_day"] = f"{m1b:.4f}"
    verify[f"{season}_alg1aware_mean_gbit_day"] = f"{m1a:.4f}"
    verify[f"{season}_alg2_pairkey_mean_gbit_day"] = f"{m2:.4f}"
    verify[f"{season}_matching_gain_pct"] = f"{matching_gain_pct:+.2f}"
    verify[f"{season}_weather_info_gain_pct"] = f"{weather_gain_pct:+.2f}"
    verify[f"{season}_alg2_min_pair_key_mean_gbit_day"] = f"{np.mean(pair_min_key[season]):.4f}"
    verify[f"{season}_alg2_p10_pair_key_mean_gbit_day"] = f"{np.mean(pair_p10_key[season]):.4f}"
    verify[f"{season}_alg2_median_pair_key_mean_gbit_day"] = f"{np.mean(pair_median_key[season]):.4f}"
    verify[f"{season}_pct_draws_min_pair_zero"] = f"{100.0*np.mean(np.array(pair_min_key[season])==0):.1f}"
    verify[f"{season}_mean_handovers_alg1aware"] = f"{np.mean(handover_counts[season]['ALG-1-aware']):.1f}"
    verify[f"{season}_mean_outage_bridged_pct"] = f"{np.mean(outage_bridged_pct[season]):.1f}"
    log(f"\n{season}: matching_gain={matching_gain_pct:+.2f}%  weather_info_gain={weather_gain_pct:+.2f}%")

save_verify_numbers(verify, "algorithm_comparison_v2_percentiles")

save_provenance(
    script_name="11b_algorithm_comparison_percentiles",
    params={"n_runs_per_season": N_RUNS_PER_SEASON, "min_elev_deg": MIN_ELEV_DEG,
            "duration_hours": DURATION_HOURS, "seasons": SEASONS, "n_pairs": len(PAIRS),
            "n_passes": len(all_passes), "n_events": len(events)},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1],
    data_sources={"Shell-1 TLE": TLE_PATH,
                  "raw hourly weather (10-year)": "data/raw/era5_hourly_*.csv.gz"},
    formulas={
        "Fair comparison": "SAME 8 stations, SAME 1019 satellites, SAME 24h window for "
                           "ALL algorithms (07-4: this is what the old +678% comparison violated)",
        "Matching gain": "(ALG-1-blind - ALG-0) / ALG-0 -- value of joint matching alone, "
                         "no weather information",
        "Weather-info gain": "(ALG-1-aware - ALG-1-blind) / ALG-1-blind -- value of adding "
                             "weather-awareness on top of matching",
        "ALG-2 pair key": "sum_q K_q, K_q=min(side-A,side-B) trusted-relay bottleneck -- a "
                          "DIFFERENT metric from ALG-0/1's raw per-station generation, not "
                          "directly comparable without noting the min() reduction",
    },
)
log(f"\nTotal runtime: {time.time() - t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
