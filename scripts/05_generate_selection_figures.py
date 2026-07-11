"""Multi-site weather-adaptive satellite selection across ALL 8 ASEAN ground
stations -- gives the "weather-aware" claim real content. At a single ground
station, weather is identical across all simultaneously-visible satellites,
so a weather-aware selector is mathematically indistinguishable from a pure
elevation-priority one (verified: +0.0% improvement once the baseline also
re-evaluates every step -- see git history of this file / project notes).

Real weather differs only ACROSS ground stations. This script compares:
  - Weather-adaptive multi-site: at each step, evaluate all 8 ASEAN ground
    stations (Hanoi, Da Nang, HCMC, Bangkok, Singapore, Manila, Jakarta,
    Kuala Lumpur), each with its own real weather, and route to whichever
    (station, satellite) pair gives the highest SKR_effective. The
    algorithm still selects a SATELLITE at the final step -- it is the
    candidate set across which that satellite is chosen that now spans
    8 ground stations instead of one.
  - Baseline: single fixed primary station (Hanoi), elevation-priority
    satellite selection only, no site diversity.
Both use the real Shell-1 TLE snapshot (1019 satellites, CelesTrak 25/06/2026),
confirmed visible from all 8 cities (see check_visibility_8cities.py).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from datetime import datetime, timezone
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from skyfield.api import load
from modules.orbital_mechanics import (
    parse_tle_block, make_skyfield_satellite, GROUND_STATIONS, get_visible_satellites,
    make_time_array,
)
from modules.routing import compute_skr_all_visible
from modules.weather_model import get_city_params

_TS = load.timescale()
T_REF = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
MIN_ELEV = 30.0
STATIONS = ['hanoi', 'danang', 'hcmc', 'bangkok', 'singapore', 'manila', 'jakarta', 'kuala_lumpur']
PRIMARY = 'hanoi'

TLE_PATH = os.path.join(os.path.dirname(__file__), 'starlink_shell1_real_tle.txt')
tle_dicts = parse_tle_block(open(TLE_PATH).read())
satellites = [make_skyfield_satellite(d) for d in tle_dicts]
print(f'Loaded {len(satellites)} REAL Shell-1 satellites (CelesTrak 25/06/2026)')
print(f'Ground stations ({len(STATIONS)}): {STATIONS}')
print(f'Baseline (single-site, no diversity): {PRIMARY}\n')


def run_scenario(month: int, duration_hours: float = 3.0, step_minutes: float = 1.0):
    t_array = make_time_array(T_REF, duration_hours, step_minutes)
    n_steps = len(t_array)

    skr_multisite = np.zeros(n_steps)
    skr_baseline = np.zeros(n_steps)
    station_used = []

    wx = {s: get_city_params(s, month) for s in STATIONS}

    for i in range(n_steps):
        t_i = t_array[i]

        # --- Weather-adaptive multi-site: best (station, satellite) pair
        # across all 8 ASEAN ground stations ---
        best_skr = 0.0
        best_station = None
        for s in STATIONS:
            gs = GROUND_STATIONS[s]
            visible = get_visible_satellites(satellites, gs['lat'], gs['lon'], t_i, gs['alt_m'], MIN_ELEV)
            if not visible:
                continue
            candidates = compute_skr_all_visible(
                visible, wx[s]['R_mm_h'], wx[s]['V_km'], wx[s]['P_cloud'],
            )
            if candidates and candidates[0]['SKR_effective'] > best_skr:
                best_skr = candidates[0]['SKR_effective']
                best_station = s
        skr_multisite[i] = best_skr
        station_used.append(best_station)

        # --- Baseline: single fixed station (Hanoi), elevation-priority ---
        gs_p = GROUND_STATIONS[PRIMARY]
        visible_p = get_visible_satellites(satellites, gs_p['lat'], gs_p['lon'], t_i, gs_p['alt_m'], MIN_ELEV)
        if visible_p:
            candidates_p = compute_skr_all_visible(
                visible_p, wx[PRIMARY]['R_mm_h'], wx[PRIMARY]['V_km'], wx[PRIMARY]['P_cloud'],
            )
            baseline_choice = min(candidates_p, key=lambda c: c['zenith_deg'])
            skr_baseline[i] = baseline_choice['SKR_effective']

    time_hours = np.arange(n_steps) * step_minutes / 60.0
    return {
        'skr_multisite': skr_multisite, 'skr_baseline': skr_baseline,
        'time_hours': time_hours, 'station_used': station_used,
    }


results = {}
for month, label in [(1, 'Dry (Jan)'), (7, 'Wet (Jul)')]:
    for s in STATIONS:
        wx = get_city_params(s, month)
        print(f"  {label} {s:14s}: P_cloud={wx['P_cloud']:.2f} R_mm_h={wx['R_mm_h']:.3f} V_km={wx['V_km']}")
    res = run_scenario(month)
    results[label] = res
    mean_ms = res['skr_multisite'].mean()
    mean_bl = res['skr_baseline'].mean()
    ratio = mean_ms / mean_bl if mean_bl > 0 else float('nan')
    usage = Counter(res['station_used'])
    print(f"{label}: multisite_mean={mean_ms*1e6:.1f} baseline_mean={mean_bl*1e6:.1f} "
          f"(x1e-6 norm) ratio={ratio:.2f}x  station usage={dict(usage)}\n")

# --- Plot: 2 panels (dry/wet), SKR multisite (8-city) vs single-station baseline ---
RB = 1e9
fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.5), sharex=True)
for ax, (label, res), title in zip(
    axes, results.items(),
    ['Dry Season (January)', 'Wet Season (July)'],
):
    skr_ms_kbps = res['skr_multisite'] * RB / 1000.0
    skr_bl_kbps = res['skr_baseline'] * RB / 1000.0
    t = res['time_hours']
    ax.fill_between(t, skr_bl_kbps, alpha=0.15, color='#d62728')
    ax.fill_between(t, skr_ms_kbps, alpha=0.12, color='#1f77b4')
    ax.plot(t, skr_bl_kbps, color='#d62728', linewidth=1.0, linestyle='--',
            label='Single station (Hanoi, elevation-priority)')
    ax.plot(t, skr_ms_kbps, color='#1f77b4', linewidth=1.2, linestyle='-',
            label='Multi-site network (8 ASEAN stations)')
    mean_ms = res['skr_multisite'].mean()
    mean_bl = res['skr_baseline'].mean()
    ratio = mean_ms / mean_bl if mean_bl > 0 else float('nan')
    top_station = Counter(res['station_used']).most_common(1)[0][0]
    ax.text(0.02, 0.92,
            f"Network / single-station: {ratio:.2f}×  |  primary site: {top_station}",
            transform=ax.transAxes, fontsize=8, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.set_ylabel('SKR_effective (kbps)', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7.5)
    y_max = max(skr_ms_kbps.max(), skr_bl_kbps.max())
    if y_max > 0:
        ax.set_ylim(0, y_max / 0.9)

axes[1].set_xlabel('Time (hours from T_REF)', fontsize=9)
plt.suptitle('SKR Time Series: 8-Station ASEAN Network vs Single-Station Operation\n'
             '(1019 real Shell-1 satellites, min_elev=30.0°)',
             fontsize=9, fontweight='bold')
plt.tight_layout(pad=0.5)
out_path = os.path.join(os.path.dirname(__file__), '..', 'diagrams', 'fig13_multisite_routing.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Saved: {out_path}")
