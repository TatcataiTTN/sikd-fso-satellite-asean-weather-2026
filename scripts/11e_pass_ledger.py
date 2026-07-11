"""
11e_pass_ledger.py — Pass-Level Full Ledger (Task 22, Phase G)
================================================================================
Runs AFTER Task 21 (needs temp/v4_full_enum/raw_runs.csv). Picks 6
representative real days automatically — best / median / worst per season
by the ALG-1-aware network total — plus any extra --date YYYY-MM-DD, and
writes ONE ROW PER SATELLITE PASS with the complete link picture:

    which satellite, over which station, rise/peak/set times (UTC + local
    hour), duration, max elevation, zenith angle (RECEIVE-side angle at
    the station), off-nadir angle (TRANSMIT-side angle at the satellite,
    modules/link_geometry.off_nadir_deg — added with tests for this task),
    slant range, that station's ACTUAL cloud/rain at that local hour on
    that real date, derived visibility, the full channel chain (hg_dB,
    hl_dB, sigma_R2, Psift, QBER, SKR), the key the pass would deliver,
    which city pair ALG-2 credited it to, and whether the ALG-1-aware
    matching actually selected it.

This is the bottom layer of the Phase G transparency stack (07-5 /
02.Deep_Understanding PHẦN 7): every aggregate number in v4 can now be
traced run -> pair -> individual pass.

Per-pass channel values here are computed EXACTLY per pass (no rounding
cache): unlike 11d's weight cache, which rounds elevation to 1 degree for
speed across 620 days, a ledger meant for inspection must not contain
cache-rounded numbers.

Run:
  cd 05_Code_v2 && python scripts/11e_pass_ledger.py
  cd 05_Code_v2 && python scripts/11e_pass_ledger.py --date 2021-07-06
"""
import os
import sys
import csv
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from modules.orbital_mechanics import parse_tle_block, make_skyfield_satellite, GROUND_STATIONS
from modules.pass_analysis import extract_passes, UTC_OFFSET_HOURS
from modules.link_geometry import slant_range_km, off_nadir_deg
from modules.channel_model import compute_channel
from modules.sikd_performance import compute_sikd_performance
from modules.schedulers import (
    schedule_baseline, match_weather_aware, schedule_pairs_greedy,
)
from utils import save_provenance, save_verify_numbers

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


CITIES = list(GROUND_STATIONS.keys())
PAIRS = [(CITIES[i], CITIES[j]) for i in range(8) for j in range(i + 1, 8)]
H_KM = 550.0
MIN_ELEV_DEG = 40.0  # Mask B, same orbital template as 11d
T_START = datetime(2026, 3, 12, 0, 0, 0, tzinfo=timezone.utc)
DURATION_HOURS = 24.0
CLOUD_OUTAGE_THRESHOLD = 85.0

TEMP_DIR = (os.path.join(os.environ['SIKD_VARIANT_DIR'], 'v4_full_enum')
            if os.environ.get('SIKD_VARIANT_DIR') else
            os.path.join(os.path.dirname(__file__), '..', 'temp', 'v4_full_enum'))
RAW_RUNS_CSV = os.path.join(TEMP_DIR, 'raw_runs.csv')
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
TLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'starlink_shell1_real_tle.txt')
_VARIANT = os.environ.get('SIKD_VARIANT_DIR')
if os.environ.get('SIKD_T_START'):
    T_START = datetime.fromisoformat(os.environ['SIKD_T_START'])
if os.environ.get('SIKD_TLE_PATH'):
    TLE_PATH = os.environ['SIKD_TLE_PATH']


_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--date", action="append", default=[],
                     help="extra YYYY-MM-DD real date(s) to ledger, beyond the "
                          "6 auto-picked representative days")
_args = _parser.parse_args()

# ----------------------------------------------------------------
# 0. Sanity check against paper Table II before anything else
# ----------------------------------------------------------------
_ch45 = compute_channel(H_KM, 45.0, V_km=10.0, R_mm_h=0.0)
_perf45 = compute_sikd_performance(_ch45['hg'], _ch45['hl'], _ch45['sigma_X2'])
assert abs(_perf45['SKR_kbps'] - 13280) / 13280 < 0.10, \
    f"Sanity check failed: SKR@zenith45 clear = {_perf45['SKR_kbps']:.0f} kbps, expected ~13,280"
log(f"Sanity check OK: SKR@zenith45 clear sky = {_perf45['SKR_kbps']:.0f} kbps (Table II: 13,280)")

# ----------------------------------------------------------------
# 1. Pick representative days from the v4 full-enumeration results
# ----------------------------------------------------------------
runs = pd.read_csv(RAW_RUNS_CSV)
selected = []  # (label, date_str)
for season in runs['season'].unique():
    sub = runs[runs['season'] == season].sort_values('alg1aware_gbit').reset_index()
    selected.append((f"{season}_worst", sub.iloc[0]['date_str']))
    selected.append((f"{season}_median", sub.iloc[len(sub) // 2]['date_str']))
    selected.append((f"{season}_best", sub.iloc[-1]['date_str']))
for extra in _args.date:
    selected.append(("user", extra))

log("Representative days: " + ", ".join(f"{lbl}={d}" for lbl, d in selected))

# ----------------------------------------------------------------
# 2. Fixed orbital template (identical for every date — same TLE epoch)
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
log(f"Built {len(all_passes)} real passes @ {MIN_ELEV_DEG:.0f} deg mask (fixed template)")

events = []
for p in all_passes:
    events.append((p['t_rise'], +1, p['station'], p['sat_id'], p['max_elev_deg']))
    events.append((p['t_set'], -1, p['station'], p['sat_id'], p['max_elev_deg']))
events.sort(key=lambda e: e[0])

# Index passes by (station, sat_id) for matched-flag resolution below.
passes_by_key = defaultdict(list)
for i, p in enumerate(all_passes):
    passes_by_key[(p['station'], p['sat_id'])].append(i)

# ----------------------------------------------------------------
# 3. Weather loading (only the selected dates are kept in memory)
# ----------------------------------------------------------------
needed_dates = {d for _, d in selected}


def load_weather_for_dates(city, dates):
    df = pd.read_csv(os.path.join(RAW_DIR, f"era5_hourly_{city}.csv.gz"),
                     compression='gzip', parse_dates=['time_local'])
    df['date'] = df['time_local'].dt.date.astype(str)
    df = df[df['date'].isin(dates)]
    df['hour'] = df['time_local'].dt.hour
    out = {}
    for date_str, grp in df.groupby('date'):
        cloud = np.zeros(24)
        precip = np.zeros(24)
        for _, row in grp.iterrows():
            cloud[row['hour']] = row['cloud_cover_pct']
            precip[row['hour']] = row['precip_mm']
        out[date_str] = (cloud, precip)
    return out


log("Loading hourly weather for selected dates...")
weather = {city: load_weather_for_dates(city, needed_dates) for city in CITIES}
log("Weather loaded.")


def local_hour_of(dt, station):
    return (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + UTC_OFFSET_HOURS[station]) % 24.0


def sweep_matched_pass_ids(date_str):
    """Re-run the ALG-1-aware event sweep for this date (same logic as 11d)
    and return the set of pass indices that the matching actually served
    for at least one segment."""
    def weight_of(t, station, sat_id, elev):
        cloud_arr, precip_arr = weather[station].get(date_str, (None, None))
        if cloud_arr is None:
            return 0.0
        hb = int(local_hour_of(t, station)) % 24
        if cloud_arr[hb] >= CLOUD_OUTAGE_THRESHOLD:
            return 0.0
        v_km = max(15.0 - 10.0 * (cloud_arr[hb] / 100.0), 0.5)
        ch = compute_channel(H_KM, 90.0 - round(elev), V_km=v_km, R_mm_h=float(precip_arr[hb]))
        return compute_sikd_performance(ch['hg'], ch['hl'], ch['sigma_X2'])['SKR_kbps']

    open_by_station = defaultdict(dict)
    prev = None
    matched = set()
    for t, typ, station, sat_id, elev in events:
        if typ == 1:
            open_by_station[station][sat_id] = elev
        else:
            open_by_station[station].pop(sat_id, None)
        visibility = {s: [{"sat_id": sid, "elev_deg": e} for sid, e in sats.items()]
                     for s, sats in open_by_station.items()}
        weights = {(s, sid): weight_of(t, s, sid, e)
                  for s, sats in open_by_station.items() for sid, e in sats.items()}
        cur = match_weather_aware(visibility, weights, prev=prev)
        for station_a, sat_a in cur.items():
            if sat_a is None:
                continue
            for idx in passes_by_key.get((station_a, sat_a), []):
                if all_passes[idx]['t_rise'] <= t < all_passes[idx]['t_set']:
                    matched.add(idx)
                    break
        prev = cur
    return matched


# ----------------------------------------------------------------
# 4. Build one ledger per selected date
# ----------------------------------------------------------------
LEDGER_COLS = [
    "sat_id", "station", "t_rise_utc", "t_peak_utc", "t_set_utc",
    "local_hour_peak", "duration_s", "max_elev_deg",
    "zenith_deg", "off_nadir_deg", "slant_range_km",
    "cloud_pct", "rain_mm_h", "V_km",
    "hg_dB", "hl_dB", "sigma_R2", "Psift", "QBER_pct", "SKR_kbps",
    "delivered_gbit", "alg2_assigned_pair", "alg1aware_matched",
]

verify = {"sanity_skr_zenith45_clear_kbps": f"{_perf45['SKR_kbps']:.0f}"}
output_files = []

for label, date_str in selected:
    tday = time.time()
    log(f"\n=== {label}: {date_str} ===")

    matched_ids = sweep_matched_pass_ids(date_str)

    # ALG-2 allocation for this date (same construction as 11d)
    pass_rows = []
    per_pass = []  # full detail, aligned with all_passes order
    for idx, p in enumerate(all_passes):
        station = p['station']
        cloud_arr, precip_arr = weather[station].get(date_str, (None, None))
        hb = int(p['local_hour_peak']) % 24
        if cloud_arr is None:
            cloud_pct, rain_mm_h = 100.0, 0.0
        else:
            cloud_pct, rain_mm_h = float(cloud_arr[hb]), float(precip_arr[hb])

        elev = p['max_elev_deg']
        zenith = 90.0 - elev
        duration_s = (p['t_set'] - p['t_rise']).total_seconds()

        if cloud_pct >= CLOUD_OUTAGE_THRESHOLD:
            v_km = max(15.0 - 10.0 * (cloud_pct / 100.0), 0.5)
            hg_db = hl_db = sigma_r2 = psift = qber = skr_kbps = 0.0
            outage = True
        else:
            v_km = max(15.0 - 10.0 * (cloud_pct / 100.0), 0.5)
            ch = compute_channel(H_KM, zenith, V_km=v_km, R_mm_h=rain_mm_h)
            perf = compute_sikd_performance(ch['hg'], ch['hl'], ch['sigma_X2'])
            hg_db, hl_db = ch['hg_dB'], ch['hl_dB']
            sigma_r2 = ch['sigma_R2']
            psift, qber = perf['Psift'], perf['QBER']
            skr_kbps = perf['SKR_kbps']
            outage = False

        delivered_gbit = skr_kbps * 1e3 * duration_s / 1e9
        per_pass.append({
            "idx": idx, "outage": outage,
            "cloud_pct": cloud_pct, "rain_mm_h": rain_mm_h, "V_km": v_km,
            "hg_dB": hg_db, "hl_dB": hl_db, "sigma_R2": sigma_r2,
            "Psift": psift, "QBER_pct": qber * 100.0, "SKR_kbps": skr_kbps,
            "delivered_gbit": delivered_gbit,
        })
        pass_rows.append({
            "pass_id": str(idx), "station": station,
            "key_bits": skr_kbps * 1e3 * duration_s, "t": p['t_rise'],
        })

    alloc2 = schedule_pairs_greedy(pass_rows, PAIRS, demands=None)
    pair_of_pass = {int(a["pass_id"]): f"{a['pair'][0]}|{a['pair'][1]}" for a in alloc2}

    out_csv = os.path.join(TEMP_DIR, f"pass_ledger_{date_str}.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(LEDGER_COLS)
        for d, p in zip(per_pass, all_passes):
            idx = d["idx"]
            elev = p['max_elev_deg']
            w.writerow([
                p['sat_id'], p['station'],
                p['t_rise'].isoformat(), p['t_peak'].isoformat(), p['t_set'].isoformat(),
                round(p['local_hour_peak'], 3),
                round((p['t_set'] - p['t_rise']).total_seconds(), 1),
                round(elev, 2),
                round(90.0 - elev, 2),
                round(off_nadir_deg(H_KM, elev), 2),
                round(slant_range_km(H_KM, elev), 1),
                round(d["cloud_pct"], 1), round(d["rain_mm_h"], 3), round(d["V_km"], 2),
                round(d["hg_dB"], 2), round(d["hl_dB"], 2), round(d["sigma_R2"], 5),
                round(d["Psift"], 5), round(d["QBER_pct"], 3), round(d["SKR_kbps"], 1),
                round(d["delivered_gbit"], 4),
                pair_of_pass.get(idx, ""),
                idx in matched_ids,
            ])
    output_files.append(out_csv)

    n = len(per_pass)
    n_matched = sum(1 for d in per_pass if d["idx"] in matched_ids)
    n_outage = sum(1 for d in per_pass if d["outage"])
    log(f"  {n} passes -> {out_csv}")
    log(f"  matched by ALG-1-aware: {n_matched} ({100.0*n_matched/n:.1f}%)  |  "
        f"outage (SKR=0): {n_outage} ({100.0*n_outage/n:.1f}%)")

    top10 = sorted(per_pass, key=lambda d: -d["delivered_gbit"])[:10]
    log("  Top-10 passes by delivered key:")
    log(f"    {'rank':>4} {'sat_id':16s} {'station':13s} {'local_h':>7} {'elev':>5} "
        f"{'SKR_kbps':>9} {'Gbit':>7}")
    top10_csv = os.path.join(TEMP_DIR, f"top10_passes_{date_str}.csv")
    with open(top10_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["rank", "sat_id", "station", "t_rise_utc", "t_set_utc",
                    "local_hour_peak", "max_elev_deg", "SKR_kbps", "delivered_gbit"])
        for r, d in enumerate(top10, 1):
            p = all_passes[d["idx"]]
            log(f"    {r:>4} {p['sat_id']:16s} {p['station']:13s} "
                f"{p['local_hour_peak']:7.2f} {p['max_elev_deg']:5.1f} "
                f"{d['SKR_kbps']:9.1f} {d['delivered_gbit']:7.3f}")
            w.writerow([r, p['sat_id'], p['station'],
                        p['t_rise'].isoformat(), p['t_set'].isoformat(),
                        round(p['local_hour_peak'], 3), round(p['max_elev_deg'], 2),
                        round(d['SKR_kbps'], 1), round(d['delivered_gbit'], 4)])
    output_files.append(top10_csv)

    verify[f"{label}_{date_str}_pct_matched"] = f"{100.0*n_matched/n:.1f}"
    verify[f"{label}_{date_str}_pct_outage"] = f"{100.0*n_outage/n:.1f}"
    verify[f"{label}_{date_str}_top1_gbit"] = f"{top10[0]['delivered_gbit']:.3f}" if top10 else "0"
    log(f"  done in {time.time()-tday:.1f}s")

# ----------------------------------------------------------------
# 5. Provenance
# ----------------------------------------------------------------
save_verify_numbers(verify, "pass_ledger")
save_provenance(
    script_name="11e_pass_ledger",
    params={"representative_days": dict(selected), "min_elev_deg": MIN_ELEV_DEG,
            "duration_hours": DURATION_HOURS, "n_passes_template": len(all_passes),
            "channel_values": "computed EXACTLY per pass (no 1-degree rounding cache, "
                              "unlike 11d's speed-oriented weight cache)"},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=output_files,
    data_sources={"v4 raw runs (day selection)": RAW_RUNS_CSV,
                  "Shell-1 TLE": TLE_PATH,
                  "raw hourly weather": "data/raw/era5_hourly_*.csv.gz"},
    formulas={
        "off_nadir_deg": "arcsin(R_E*cos(elev)/(R_E+h)) — transmit-side angle at the "
                          "satellite from nadir (modules/link_geometry.py, tested)",
        "zenith_deg": "90 - max_elev_deg — receive-side angle at the station",
        "delivered_gbit": "SKR_kbps * 1e3 * duration_s / 1e9 (key if the pass is used "
                          "for its full duration)",
    },
)
log(f"\nTotal runtime: {time.time() - t0:.1f}s")
