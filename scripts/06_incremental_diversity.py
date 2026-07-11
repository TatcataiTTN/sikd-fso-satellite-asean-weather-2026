"""Incremental site-diversity curve: how does weather-adaptive multi-site
SKR improvement grow as ground stations are added one at a time, in order
of distance from the primary station (Hanoi)? This mirrors the structure
of Nguyen/Le/Pham/Dang (2023) Fig. 7, which shows availability rising
81.92% (Hanoi alone) -> 96.86% (+Da Nang) -> 99.44% (+HCMC) as sites are
added -- but extended here to the full 8-city ASEAN set, with real
Shell-1 satellites (1019 sats, CelesTrak 25/06/2026) and real ERA5-backed
weather (Open-Meteo, 2015-2020).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from datetime import datetime, timezone
from collections import Counter
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from modules.orbital_mechanics import (
    parse_tle_block, make_skyfield_satellite, GROUND_STATIONS, get_visible_satellites,
    make_time_array,
)
from modules.routing import compute_skr_all_visible
from modules.weather_model import get_city_params

T_REF = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
MIN_ELEV = 30.0
PRIMARY = 'hanoi'


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


hn = GROUND_STATIONS[PRIMARY]
others = [(haversine_km(hn['lat'], hn['lon'], gs['lat'], gs['lon']), c)
          for c, gs in GROUND_STATIONS.items() if c != PRIMARY]
others.sort()
ROLLOUT_ORDER = [PRIMARY] + [c for _, c in others]
print('Rollout order (distance from Hanoi):')
for d, c in [(0, PRIMARY)] + others:
    print(f'  {c:14s} {d:6.0f} km')

TLE_PATH = os.path.join(os.path.dirname(__file__), 'starlink_shell1_real_tle.txt')
tle_dicts = parse_tle_block(open(TLE_PATH).read())
satellites = [make_skyfield_satellite(d) for d in tle_dicts]
print(f'\nLoaded {len(satellites)} real Shell-1 satellites (CelesTrak 25/06/2026)\n')


def run_scenario(month: int, stations: list[str], duration_hours=3.0, step_minutes=1.0):
    t_array = make_time_array(T_REF, duration_hours, step_minutes)
    wx = {s: get_city_params(s, month) for s in stations}
    skr = np.zeros(len(t_array))
    usage = []
    for i, t_i in enumerate(t_array):
        best_skr, best_station = 0.0, None
        for s in stations:
            gs = GROUND_STATIONS[s]
            visible = get_visible_satellites(satellites, gs['lat'], gs['lon'], t_i, gs['alt_m'], MIN_ELEV)
            if not visible:
                continue
            candidates = compute_skr_all_visible(visible, wx[s]['R_mm_h'], wx[s]['V_km'], wx[s]['P_cloud'])
            if candidates and candidates[0]['SKR_effective'] > best_skr:
                best_skr = candidates[0]['SKR_effective']
                best_station = s
        skr[i] = best_skr
        usage.append(best_station)
    return skr, usage


def run_baseline(month: int, duration_hours=3.0, step_minutes=1.0):
    t_array = make_time_array(T_REF, duration_hours, step_minutes)
    wx = get_city_params(PRIMARY, month)
    gs = GROUND_STATIONS[PRIMARY]
    skr = np.zeros(len(t_array))
    for i, t_i in enumerate(t_array):
        visible = get_visible_satellites(satellites, gs['lat'], gs['lon'], t_i, gs['alt_m'], MIN_ELEV)
        if not visible:
            continue
        candidates = compute_skr_all_visible(visible, wx['R_mm_h'], wx['V_km'], wx['P_cloud'])
        baseline_choice = min(candidates, key=lambda c: c['zenith_deg'])
        skr[i] = baseline_choice['SKR_effective']
    return skr


results = {}
for month, label in [(1, 'dry'), (7, 'wet')]:
    baseline_skr = run_baseline(month)
    mean_bl = baseline_skr.mean()
    curve = []
    print(f"=== {label.upper()} (month={month}) === Hanoi-only mean SKR_norm={mean_bl*1e6:.1f}e-6  [diversity ratios vs this]")
    for n in range(1, len(ROLLOUT_ORDER) + 1):
        stations = ROLLOUT_ORDER[:n]
        skr, usage = run_scenario(month, stations)
        mean_ms = skr.mean()
        ratio = mean_ms / mean_bl if mean_bl > 0 else float('nan')
        top = Counter(usage).most_common(1)[0][0] if usage else None
        curve.append({'n_sites': n, 'sites': list(stations), 'skr_ratio': ratio,
                       'mean_skr': mean_ms, 'top_station': top})
        print(f"  n={n}  sites={'+'.join(stations):55s}  ratio={ratio:.2f}x  primary={top}")
    results[label] = curve

out_json = os.path.join(os.path.dirname(__file__), 'incremental_site_diversity.json')
json.dump({
    'rollout_order': ROLLOUT_ORDER,
    'rollout_distance_km': {c: d for d, c in [(0, PRIMARY)] + others},
    'results': results,
}, open(out_json, 'w'), indent=2)
print(f"\nSaved: {out_json}")

# --- Plot: improvement % vs number of sites, dry vs wet ---
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for label, color, marker in [('dry', '#2196F3', 'o'), ('wet', '#FF5722', 's')]:
    curve = results[label]
    x = [c['n_sites'] for c in curve]
    y = [c['skr_ratio'] for c in curve]
    ax.plot(x, y, marker=marker, color=color, label=f'{label.capitalize()} season (Jan)' if label == 'dry' else f'{label.capitalize()} season (Jul)')
    for c in curve:
        if c['n_sites'] > 1:
            ax.annotate(c['sites'][-1], (c['n_sites'], c['improvement_pct']),
                        fontsize=6.5, textcoords="offset points", xytext=(3, 3))

ax.set_xlabel('Number of ASEAN ground stations (cumulative, by distance from Hanoi)')
ax.set_ylabel('Network SKR / Single-station SKR (Hanoi-only) [×]')
ax.set_title('Network SKR vs Number of Ground Stations (Site Diversity Analysis)')
ax.set_xticks(range(1, len(ROLLOUT_ORDER) + 1))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'diagrams', 'fig13b_incremental_site_diversity.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Saved: {out_path}")
