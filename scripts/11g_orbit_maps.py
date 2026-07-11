"""
11g_orbit_maps.py — ASEAN Orbit & Coverage Maps (Task 24 + 24.2, Phase G)
================================================================================
REVISION (06/07/2026) of the Task 24 figure set, after user feedback on the
first version. Seven figures now (four originals, kept where they already
worked, plus three new ones):

  fig09a   Ground tracks over ASEAN, 1-hour window, 8 stations + Hoang Sa/
           Truong Sa marked, top-10 downlink passes highlighted. REVISED:
           context tracks made fainter (density was overwhelming the map),
           top-10 tracks padded +/-3 min so a heading is visible, direction
           arrows added, and the caption now states WHY the top-10 cluster
           where they do that day (dynamic, not hardcoded).
  fig09b   Elevation vs time over Hanoi, 2000 s, every satellite crossing
           the 30 deg mask (unchanged from the first version — not flagged).
  fig09c   Best pass of the day: elevation contour at peak + coverage
           circle, and communication-duration contour. REVISED: islands +
           station name label added.
  fig09d   Top-10 DOWNLINK passes of the day (kept as-is per user request —
           "vẫn ổn, vẫn rất hay" — just relabeled explicitly as downlinks,
           padded/arrowed tracks, islands, and Duration/Window columns
           added to the table).
  fig09d1  NEW — which of the 8 ASEAN ground STATIONS gets the most
           satellite downlink opportunities, and which 3-5 satellites
           serve each one (7-day geometric average, elev >= 30 deg,
           weather-independent). Reported per STATION, not per country:
           an earlier draft summed Vietnam's 3 stations into one "country"
           total and made Vietnam look 3x busier than everyone else, which
           was entirely an artifact of the 3:1:1:1:1:1 station count (the
           user's own site list, not a matched sample) — the same
           "N-stations-vs-1" fairness bug as the retracted +678% claim
           (07-4). Per station the 8 are actually close (1440-1612/day).
  fig09d2  NEW — which single ASEAN station is the busiest store-and-
           forward relay hub, and which 3-5 satellites do that relaying
           (from Task 23's sf_relay_detail_56pairs.csv). Same per-STATION
           fix as fig09d1 — country-level aggregation had hidden the real
           finding (Manila is the busiest single relay hub) behind an
           artifact of Vietnam's station count.
  fig09e   NEW — Top-5 city-PAIR key exchanges. Fixes a conceptual error in
           the first version: fig09d shows single-station DOWNLINKS, not
           "connections between two cities". A pair actually needs TWO
           downlinks (one per station) credited to it by the trusted-node
           greedy allocator (Task 21). fig09e draws both downlinks per pair
           and labels whether the pair is DUAL (simultaneous) or SF
           (store-and-forward, with directed latency from Task 23).

Hoang Sa (Paracel) / Truong Sa (Spratly) are now drawn faint on every
cartopy map (fig09a/c/d/d1-is-not-a-map/e), per explicit user request and
sourced from Vietnamese government/encyclopedia pages (NOT latitude.to,
which mislabels Hoang Sa as belonging to China — see PARACEL_BOX etc.
below and PROMPT-T24.2 in 07-6.Project_Prompts.md for the exact sources).

Uses variant A (temp/epoch_A_20260625, T_START = 2026-06-25, one day AFTER
the TLE epoch) by convention for this revision — set SIKD_VARIANT_DIR /
SIKD_T_START / SIKD_TLE_PATH env vars accordingly before running (the
script's hardcoded defaults are unchanged for backward compatibility with
scripts that don't set these). Output is written to FIG_DIR (variant-aware)
and ALSO copied to latex_paper_3/figures/ (the canonical location the
paper actually includes) whenever a variant dir is used.

Elevation seen from an arbitrary grid point (fig09c) is computed
geometrically (tan(elev) = (cos psi - R/(R+h)) / sin psi, psi = great-
circle angle to the subsatellite point) — pure numpy, vectorized over the
grid, consistent with modules/link_geometry.py.

Run (recommended, variant A):
  cd 05_Code_v2 && SIKD_VARIANT_DIR=$PWD/temp/epoch_A_20260625 \\
      SIKD_T_START=2026-06-25T00:00:00+00:00 python scripts/11g_orbit_maps.py
  ... --date 2016-01-05          # force a specific Task-22 representative day
  ... --balanced                 # pick the day whose top-10 spreads most
                                  # evenly across North/Mid-latitude/Equatorial
                                  # (among the 6 days Task 22 already ledgered)
"""
import os
import sys
import glob
import math
import time
import shutil
import argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skyfield.api import wgs84

from modules.orbital_mechanics import (
    parse_tle_block, make_skyfield_satellite, GROUND_STATIONS,
    make_time_array, compute_elevation_timeseries,
)
from modules.link_geometry import ground_coverage_radius_km, R_EARTH_KM
from modules.pass_analysis import UTC_OFFSET_HOURS
from utils import save_provenance, save_verify_numbers

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

# 3-way LATITUDE grouping, used ONLY for the top-10 "balance" entropy metric
# (fig09a's caption + the --balanced day picker). Not the same grouping as
# COUNTRY_OF below.
REGION_OF = {
    'hanoi': 'North', 'danang': 'North',
    'bangkok': 'Mid-latitude', 'hcmc': 'Mid-latitude', 'manila': 'Mid-latitude',
    'kuala_lumpur': 'Equatorial', 'singapore': 'Equatorial', 'jakarta': 'Equatorial',
}
REGION_ORDER = ['North', 'Mid-latitude', 'Equatorial']
REGION_STATIONS = {r: [s for s, rr in REGION_OF.items() if rr == r] for r in REGION_ORDER}

# Country lookup, used ONLY to color Vietnam's stations consistently in
# fig09d1/fig09d2 (NOT to aggregate — see those sections' docstrings for
# why summing Vietnam's 3 stations into one "country" total was dropped
# after user feedback 06/07/2026: it manufactured a fake ranking result).
COUNTRY_OF = {
    'hanoi': 'Vietnam', 'danang': 'Vietnam', 'hcmc': 'Vietnam',
    'bangkok': 'Thailand', 'kuala_lumpur': 'Malaysia', 'singapore': 'Singapore',
    'manila': 'Philippines', 'jakarta': 'Indonesia',
}

EXTENT = [95.0, 125.0, -10.0, 25.0]  # [lon_min, lon_max, lat_min, lat_max]
T_START = datetime(2026, 3, 12, 0, 0, 0, tzinfo=timezone.utc)

# Hoang Sa (Paracel) / Truong Sa (Spratly) -- Vietnam's sovereignty, drawn
# faint on every map (user feedback 06/07/2026). Sources are Vietnamese
# government/encyclopedia pages, chosen deliberately because latitude.to
# (used in an earlier draft) mislabels Hoang Sa as belonging to China:
#   - region extents: So TT&TT tinh Bac Giang (stttt.bacgiang.gov.vn),
#     "Nhung net chinh ve vi tri dia ly ... quan dao Hoang Sa/Truong Sa"
#   - Hoang Sa namesake island = Phu Lam / Woody Island (largest island of
#     the archipelago): 16 deg 50.2' N, 112 deg 20' E -- same source
#   - Truong Sa namesake island = Truong Sa Lon / Spratly Island (seat of
#     Truong Sa town): 8 deg 38' 30" N, 111 deg 55' 55" E -- Vietnamese
#     Wikipedia, coordinates of the sovereignty marker on the island
PARACEL_BOX = (15.75, 17.25, 111.0, 113.0)       # lat_min, lat_max, lon_min, lon_max
PARACEL_ISLAND = (16.837, 112.333)               # Phu Lam / Woody Island
SPRATLY_BOX = (6.83, 12.0, 111.05, 117.03)
SPRATLY_ISLAND = (8.642, 111.932)                # Truong Sa Lon / Spratly Island

TEMP_DIR = (os.path.join(os.environ['SIKD_VARIANT_DIR'], 'v4_full_enum')
            if os.environ.get('SIKD_VARIANT_DIR') else
            os.path.join(os.path.dirname(__file__), '..', 'temp', 'v4_full_enum'))
from utils import INTER as _INTER
PASS_CSV = os.path.join(str(_INTER), 'pass_table_8cities_7days_elev30.csv')
FEAS_SF_CSV = os.path.join(str(_INTER), 'sf_relay_detail_56pairs.csv')
# ISL multi-hop (Task 24.3) is the CURRENT relay latency model -- the v1
# file above is legacy/single-hop (superseded, kept only for its static
# class/dual_pct columns, which are unaffected by the relay-model choice).
FEAS_SF_ISL_CSV = os.path.join(str(_INTER), 'sf_relay_detail_56pairs_isl.csv')
TLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'starlink_shell1_real_tle.txt')
_VARIANT = os.environ.get('SIKD_VARIANT_DIR')
if os.environ.get('SIKD_T_START'):
    T_START = datetime.fromisoformat(os.environ['SIKD_T_START'])
if os.environ.get('SIKD_TLE_PATH'):
    TLE_PATH = os.environ['SIKD_TLE_PATH']
CANONICAL_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'latex_paper_3', 'figures')
FIG_DIR = (os.path.join(_VARIANT, 'figures') if _VARIANT else CANONICAL_FIG_DIR)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CANONICAL_FIG_DIR, exist_ok=True)

ACCENT = '#c62828'      # single accent: highlighted / Vietnam
CONTEXT = '#9e9e9e'     # recessive gray for context tracks + islands
SIDE_A_COLOR = '#1565C0'   # fig09e side-A downlink (blue)
SIDE_B_COLOR = '#2e7d32'   # fig09e side-B downlink (green)
SEQ_ELEV = 'Blues'      # sequential single hue: elevation magnitude
SEQ_DUR = 'Greens'      # sequential single hue: duration magnitude
SEQ_KEY = 'Oranges'     # sequential single hue: delivered key magnitude

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--date", default=None,
                     help="which Task 22 ledger day to use (default: the day whose "
                          "top-1 pass delivered the most key)")
_parser.add_argument("--balanced", action="store_true",
                     help="pick the Task 22 representative day whose top-10 key is "
                          "most evenly spread across North/Mid-latitude/Equatorial "
                          "(overridden by --date if both are given)")
_args = _parser.parse_args()

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def subpoint_track(sat, start_dt, duration_h, step_min):
    t_arr = make_time_array(start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt,
                            duration_h, step_min)
    sp = wgs84.subpoint(sat.at(t_arr))
    return np.asarray(sp.longitude.degrees), np.asarray(sp.latitude.degrees), t_arr


def new_map(figsize):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f2f2ee', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#eaf3f8', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=':', zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 6}
    return fig, ax


def draw_stations(ax, star_station=None):
    for c in CITIES:
        g = GROUND_STATIONS[c]
        marker = '*' if c == star_station else '^'
        size = 150 if c == star_station else 48
        ax.scatter(g['lon'], g['lat'], marker=marker, s=size, color=ACCENT,
                   edgecolor='black', linewidth=0.4, zorder=6,
                   transform=ccrs.PlateCarree())
        ax.annotate(CITY_LABELS[c], (g['lon'], g['lat']),
                    xytext=(4, 4), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', alpha=0.88,
                              lw=0.4, ec='#bdbdbd'),
                    zorder=7)


def draw_islands(ax):
    """Hoang Sa (Paracel) & Truong Sa (Spratly) archipelagos: a faint
    region box (Vietnamese-government-sourced extent) + a marker at the
    namesake island, labeled in Vietnamese + English. Context only —
    deliberately less prominent than the 8 ground stations."""
    for (lat0, lat1, lon0, lon1), (ilat, ilon), label in [
            (PARACEL_BOX, PARACEL_ISLAND, 'Hoàng Sa\n(Paracel Is.)'),
            (SPRATLY_BOX, SPRATLY_ISLAND, 'Trường Sa\n(Spratly Is.)')]:
        face_rgba = mcolors.to_rgba(CONTEXT, 0.10)
        edge_rgba = mcolors.to_rgba(CONTEXT, 0.35)
        rect = Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                         facecolor=face_rgba, edgecolor=edge_rgba,
                         linewidth=0.5, linestyle=':', zorder=1.5,
                         transform=ccrs.PlateCarree())
        ax.add_patch(rect)
        ax.scatter(ilon, ilat, marker='D', s=18, color='#757575', alpha=0.7,
                   edgecolor='#424242', linewidth=0.3, zorder=2.5,
                   transform=ccrs.PlateCarree())
        ax.annotate(label, (ilon, ilat), xytext=(3, -3), textcoords='offset points',
                    fontsize=5.5, color='#616161', zorder=2.5, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.45, lw=0))


def plot_track_segments(ax, lons, lats, **kwargs):
    """Plot a ground track split at the antimeridian jumps."""
    lons = np.asarray(lons)
    breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
    for seg_lon, seg_lat in zip(np.split(lons, breaks), np.split(np.asarray(lats), breaks)):
        ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(), **kwargs)


def add_direction_arrow(ax, lons, lats, color, frac=0.6, scale=22, zorder=6):
    """Small quiver arrow along a track to show heading (user feedback:
    passes limited to [t_rise, t_set] were too short to read direction from)."""
    n = len(lons)
    if n < 4:
        return
    j = max(1, min(n - 2, int(frac * n)))
    dx, dy = lons[j + 1] - lons[j - 1], lats[j + 1] - lats[j - 1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return
    ax.quiver(lons[j], lats[j], dx / norm, dy / norm, color=color, scale=scale,
             width=0.006, headwidth=4.5, headlength=5.5, zorder=zorder,
             transform=ccrs.PlateCarree())


def plot_track_padded(ax, sat, t_rise, t_set, pad_min=3.0, color=ACCENT,
                      core_lw=2.2, pad_lw=None, core_alpha=0.95, pad_alpha=0.30,
                      zorder=4, arrow=True):
    """Ground-track segment with the [t_rise, t_set] pass core drawn solid/
    dark and a +/- pad_min minute lead-in/lead-out drawn faint, so a
    heading is visible even for very short passes. Returns the core
    (lon, lat) arrays for label/rank-badge placement. t_rise/t_set must be
    plain (non-pandas) tz-aware datetimes."""
    pad_lw = pad_lw if pad_lw is not None else core_lw * 0.45
    t0_pad = t_rise - timedelta(minutes=pad_min)
    total_min = (t_set - t_rise).total_seconds() / 60.0 + 2 * pad_min
    step_min = 0.1
    lons, lats, _ = subpoint_track(sat, t0_pad, total_min / 60.0, step_min)
    n = len(lons)
    times = [t0_pad + timedelta(minutes=i * step_min) for i in range(n)]
    core_mask = np.array([(t >= t_rise) and (t <= t_set) for t in times])
    idx = np.where(core_mask)[0]
    if len(idx) == 0:
        idx = np.array([0, n - 1])
    i0, i1 = int(idx[0]), int(idx[-1])
    if i0 > 0:
        plot_track_segments(ax, lons[:i0 + 1], lats[:i0 + 1], color=color,
                            linewidth=pad_lw, alpha=pad_alpha, zorder=zorder)
    plot_track_segments(ax, lons[i0:i1 + 1], lats[i0:i1 + 1], color=color,
                        linewidth=core_lw, alpha=core_alpha, zorder=zorder + 0.5)
    if i1 < n - 1:
        plot_track_segments(ax, lons[i1:], lats[i1:], color=color,
                            linewidth=pad_lw, alpha=pad_alpha, zorder=zorder)
    if arrow:
        add_direction_arrow(ax, lons[i0:i1 + 1], lats[i0:i1 + 1], color, zorder=zorder + 2)
    return lons[i0:i1 + 1], lats[i0:i1 + 1]


def local_window_str(t_rise, t_set, station):
    off = UTC_OFFSET_HOURS[station]
    lr = t_rise + timedelta(hours=off)
    ls = t_set + timedelta(hours=off)
    return f"{lr.hour:02d}:{lr.minute:02d}-{ls.hour:02d}:{ls.minute:02d}"


def region_shares(df):
    totals = {r: 0.0 for r in REGION_ORDER}
    for _, row in df.iterrows():
        totals[REGION_OF[row['station']]] += row['delivered_gbit']
    s = sum(totals.values())
    if s <= 0:
        return {r: 0.0 for r in REGION_ORDER}
    return {r: v / s for r, v in totals.items()}


def region_entropy(shares):
    vals = [p for p in shares.values() if p > 0]
    if not vals:
        return 0.0
    h = -sum(p * math.log(p) for p in vals)
    return h / math.log(len(REGION_ORDER))


def pick_balanced_date(files):
    scored = []
    for f in files:
        df = pd.read_csv(f)
        score = region_entropy(region_shares(df))
        total = df['delivered_gbit'].sum()
        scored.append((score, total, f))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return scored[0][2]


def compute_dual_window_seconds(full_ledger, station_a, station_b):
    """Max simultaneous-visibility window (seconds) for ANY satellite seen
    by both stations on THIS specific day's fixed 24h orbital template
    (Mask B, elev >= 40 deg) — a concrete, date-specific number, distinct
    from Task 6's 24h-average dual_pct (which is a static geometric
    property, not tied to one calendar day)."""
    sub_a = full_ledger[full_ledger['station'] == station_a]
    sub_b = full_ledger[full_ledger['station'] == station_b]
    common_sats = set(sub_a['sat_id']) & set(sub_b['sat_id'])
    best = 0.0
    for sid in common_sats:
        ra = sub_a[sub_a['sat_id'] == sid]
        rb = sub_b[sub_b['sat_id'] == sid]
        for _, pa in ra.iterrows():
            for _, pb in rb.iterrows():
                lo = max(pa['t_rise_dt'], pb['t_rise_dt'])
                hi = min(pa['t_set_dt'], pb['t_set_dt'])
                overlap = (hi - lo).total_seconds()
                if overlap > best:
                    best = overlap
    return best


output_files = []

# ----------------------------------------------------------------
# 0. Choose the day + load its top-10 passes (Task 22 output)
# ----------------------------------------------------------------
top10_files = sorted(glob.glob(os.path.join(TEMP_DIR, 'top10_passes_*.csv')))
assert top10_files, "No top10_passes_*.csv found — run 11e_pass_ledger.py (Task 22) first"

if _args.date:
    chosen = os.path.join(TEMP_DIR, f'top10_passes_{_args.date}.csv')
    assert os.path.exists(chosen), f"{chosen} not found"
    if _args.balanced:
        print("  --date given; ignoring --balanced")
elif _args.balanced:
    chosen = pick_balanced_date(top10_files)
    print(f"  --balanced: picked {os.path.basename(chosen)} "
          f"(most even top-10 spread across {'/'.join(REGION_ORDER)})")
else:
    def top1_gbit(path):
        return pd.read_csv(path).iloc[0]['delivered_gbit']
    chosen = max(top10_files, key=top1_gbit)

DATE = os.path.basename(chosen).replace('top10_passes_', '').replace('.csv', '')
top10 = pd.read_csv(chosen)
top10['t_rise_dt'] = pd.to_datetime(top10['t_rise_utc'], format='ISO8601')
top10['t_set_dt'] = pd.to_datetime(top10['t_set_utc'], format='ISO8601')
print(f"Chosen day: {DATE} (top-1 pass delivers {top10.iloc[0]['delivered_gbit']:.3f} Gbit)")

# ----------------------------------------------------------------
# 1. Satellites + Task-5 pass table (elev >= 30, 7-day, weather-independent)
# ----------------------------------------------------------------
tle_dicts = parse_tle_block(open(TLE_PATH).read())
sat_by_name = {d['name']: make_skyfield_satellite(d) for d in tle_dicts}
print(f"Loaded {len(sat_by_name)} real Shell-1 satellites")

passes5 = pd.read_csv(PASS_CSV, comment='#')
passes5['t_rise_dt'] = pd.to_datetime(passes5['t_rise'], format='ISO8601')

# ----------------------------------------------------------------
# 2. fig09a — ground tracks, 1h window, top-10 downlinks highlighted
# ----------------------------------------------------------------
print("\nfig09a: ground tracks over ASEAN (1h window)...")
window_end = T_START + timedelta(hours=1)
in_window = passes5[(passes5['t_rise_dt'] >= T_START) & (passes5['t_rise_dt'] < window_end)]
sats_1h = sorted(in_window['sat_id'].unique())
print(f"  {len(sats_1h)} unique satellites have a pass (elev>=30) over >=1 station in the hour")

fig, ax = new_map((7.16, 6.4))
for name in sats_1h:
    lons, lats, _ = subpoint_track(sat_by_name[name], T_START, 1.0, 0.5)
    plot_track_segments(ax, lons, lats, color=CONTEXT, linewidth=0.22, alpha=0.12, zorder=2)

for _, row in top10.iterrows():
    sat = sat_by_name[row['sat_id']]
    plot_track_padded(ax, sat, row['t_rise_dt'].to_pydatetime(), row['t_set_dt'].to_pydatetime(),
                      pad_min=3.0, color=ACCENT, core_lw=2.2, zorder=4)

draw_stations(ax)
draw_islands(ax)
ax.plot([], [], color=CONTEXT, linewidth=1.0, label=f'All visible satellites in 1 h ({len(sats_1h)}, faint)')
ax.plot([], [], color=ACCENT, linewidth=2.0, label='Top-10 downlink passes of the day')
ax.legend(loc='upper right', fontsize=6.5, framealpha=0.9)

shares_a = region_shares(top10)
dom_region = max(shares_a, key=shares_a.get)
if shares_a[dom_region] > 0.6:
    cluster_note = (f"{shares_a[dom_region]*100:.0f}% of top-10 key from {dom_region} "
                    f"({'/'.join(CITY_LABELS[s] for s in REGION_STATIONS[dom_region])}) "
                    f"this particular day's weather")
else:
    cluster_note = "top-10 key spread across regions: " + ", ".join(
        f"{r} {shares_a[r]*100:.0f}%" for r in REGION_ORDER)
ax.set_title(f'Starlink Shell-1 Ground Tracks over ASEAN — 1-hour window\n'
             f'top-10 downlink passes of {DATE} highlighted ({len(sat_by_name)} satellites total)\n'
             f'{cluster_note}',
             fontsize=7.5, fontweight='bold')
out_a = os.path.join(FIG_DIR, 'fig09a_asean_ground_tracks.png')
plt.savefig(out_a, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_a)
print(f"  Saved: {out_a}")

# ----------------------------------------------------------------
# 3. fig09b — elevation vs time over Hanoi, ALL sats crossing the mask
# ----------------------------------------------------------------
print("\nfig09b: elevation vs time over Hanoi (2000 s)...")
win_s = 2000.0
hn = GROUND_STATIONS['hanoi']
hanoi_win = passes5[(passes5['station'] == 'hanoi') &
                    (passes5['t_rise_dt'] >= T_START) &
                    (passes5['t_rise_dt'] < T_START + timedelta(seconds=win_s))]
sats_b = sorted(hanoi_win['sat_id'].unique())
print(f"  {len(sats_b)} satellites cross the 30 deg mask over Hanoi in the window")

t_arr = make_time_array(T_START, win_s / 3600.0, 10.0 / 60.0)
tsec = np.arange(len(t_arr)) * 10.0

best_name, best_peak = None, -1
curves = {}
for name in sats_b:
    elev = compute_elevation_timeseries(sat_by_name[name], hn['lat'], hn['lon'], t_arr, hn['alt_m'])
    curves[name] = elev
    if elev.max() > best_peak:
        best_peak, best_name = elev.max(), name

fig, ax = plt.subplots(figsize=(7.16, 3.4))
for name, elev in curves.items():
    if name == best_name:
        continue
    e = np.where(elev > 0, elev, np.nan)
    ax.plot(tsec, e, color=CONTEXT, linewidth=0.8, alpha=0.6)
e = np.where(curves[best_name] > 0, curves[best_name], np.nan)
ax.plot(tsec, e, color=ACCENT, linewidth=1.8,
        label=f'{best_name} (peak {best_peak:.1f}°)')
imax = int(np.nanargmax(e))
ax.annotate(best_name, (tsec[imax], e[imax]), xytext=(8, 4),
            textcoords='offset points', fontsize=7, color=ACCENT)
ax.axhline(30, color='black', linestyle='--', linewidth=0.9, label='Mask 30° (geometric)')
ax.axhline(40, color='black', linestyle='-', linewidth=0.9, label='Mask 40° (BB84-secure, Mask B)')
ax.set_xlabel('Elapsed time (s)')
ax.set_ylabel('Elevation angle (deg)')
ax.set_xlim(0, win_s)
ax.set_ylim(0, 90)
ax.grid(alpha=0.25, linewidth=0.4)
ax.legend(loc='upper right', fontsize=6.5)
ax.set_title(f'Elevation vs Time over Hanoi — every satellite crossing the 30° mask '
             f'in 2000 s ({len(sats_b)} satellites)', fontsize=8.5, fontweight='bold')
out_b = os.path.join(FIG_DIR, 'fig09b_elevation_vs_time.png')
plt.savefig(out_b, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_b)
print(f"  Saved: {out_b}")

# ----------------------------------------------------------------
# 4. fig09c — best pass: elevation contour at peak + duration contour
# ----------------------------------------------------------------
print("\nfig09c: coverage/duration contours for the best pass...")
bp = top10.iloc[0]
sat = sat_by_name[bp['sat_id']]
print(f"  Best pass: {bp['sat_id']} over {bp['station']}, "
      f"{bp['delivered_gbit']:.3f} Gbit, max elev {bp['max_elev_deg']:.1f} deg")

dur_h = (bp['t_set_dt'] - bp['t_rise_dt']).total_seconds() / 3600.0
gs_bp = GROUND_STATIONS[bp['station']]
t_fine = make_time_array(bp['t_rise_dt'].to_pydatetime(), dur_h, 5.0 / 60.0)
elev_fine = compute_elevation_timeseries(sat, gs_bp['lat'], gs_bp['lon'], t_fine, gs_bp['alt_m'])
i_peak = int(np.argmax(elev_fine))
sp_peak = wgs84.subpoint(sat.at(t_fine[i_peak]))
sub_lat, sub_lon = sp_peak.latitude.degrees, sp_peak.longitude.degrees
h_km = sp_peak.elevation.km
print(f"  Peak: subpoint ({sub_lat:.2f}N, {sub_lon:.2f}E), altitude {h_km:.0f} km")


def elev_from_grid(glat, glon, s_lat, s_lon, h):
    """Elevation angle (deg) of a satellite (subpoint s, altitude h) seen
    from grid points — spherical geometry, vectorized (matches
    link_geometry's Earth model)."""
    p1, p2 = np.radians(glat), np.radians(s_lat)
    dl = np.radians(glon - s_lon)
    cos_psi = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    cos_psi = np.clip(cos_psi, -1, 1)
    psi = np.arccos(cos_psi)
    ratio = R_EARTH_KM / (R_EARTH_KM + h)
    elev = np.degrees(np.arctan2(cos_psi - ratio, np.sin(psi)))
    return elev


glon = np.linspace(EXTENT[0], EXTENT[1], 140)
glat = np.linspace(EXTENT[2], EXTENT[3], 140)
GLON, GLAT = np.meshgrid(glon, glat)
ELEV = elev_from_grid(GLAT, GLON, sub_lat, sub_lon, h_km)

psi40 = ground_coverage_radius_km(h_km, 40.0) / R_EARTH_KM
az = np.linspace(0, 2 * np.pi, 181)
lat0, lon0 = np.radians(sub_lat), np.radians(sub_lon)
circ_lat = np.arcsin(np.sin(lat0) * np.cos(psi40) + np.cos(lat0) * np.sin(psi40) * np.cos(az))
circ_lon = lon0 + np.arctan2(np.sin(az) * np.sin(psi40) * np.cos(lat0),
                             np.cos(psi40) - np.sin(lat0) * np.sin(circ_lat))

pad = timedelta(minutes=10)
t_dur = make_time_array((bp['t_rise_dt'] - pad).to_pydatetime(),
                        dur_h + 20.0 / 60.0, 10.0 / 60.0)
sp_tr = wgs84.subpoint(sat.at(t_dur))
tr_lat = np.asarray(sp_tr.latitude.degrees)
tr_lon = np.asarray(sp_tr.longitude.degrees)
tr_h = np.asarray(sp_tr.elevation.km)

flat_lat, flat_lon = GLAT.ravel()[:, None], GLON.ravel()[:, None]
E_t = elev_from_grid(flat_lat, flat_lon, tr_lat[None, :], tr_lon[None, :], tr_h[None, :])
DUR = (E_t > 40.0).sum(axis=1).reshape(GLAT.shape) * 10.0  # seconds

fig = plt.figure(figsize=(7.16, 3.6))
for k, (field, cmap, label, ttl) in enumerate([
        (ELEV, SEQ_ELEV, 'Elevation angle (deg)',
         f'(a) Elevation seen across ASEAN at peak instant\n40° coverage circle (radius '
         f'{ground_coverage_radius_km(h_km, 40.0):.0f} km)'),
        (DUR, SEQ_DUR, 'Time above 40° (s)',
         '(b) Communication-time duration above 40°\nfor this single pass')]):
    ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=':', zorder=3)
    masked = np.ma.masked_less(field, 0 if k == 0 else 1e-9)
    cf = ax.contourf(GLON, GLAT, masked, levels=14, cmap=cmap, zorder=1)
    fig.colorbar(cf, ax=ax, shrink=0.75, label=label)
    if k == 0:
        ax.plot(np.degrees(circ_lon), np.degrees(circ_lat), color=ACCENT,
                linewidth=1.3, zorder=4, transform=ccrs.PlateCarree())
    draw_islands(ax)
    ax.scatter(sub_lon, sub_lat, marker='x', s=45, color='black', zorder=5,
               transform=ccrs.PlateCarree())
    gsl = GROUND_STATIONS[bp['station']]
    ax.scatter(gsl['lon'], gsl['lat'], marker='*', s=130, color=ACCENT,
               edgecolor='black', linewidth=0.4, zorder=5, transform=ccrs.PlateCarree())
    ax.annotate(CITY_LABELS[bp['station']], (gsl['lon'], gsl['lat']), xytext=(4, 4),
                textcoords='offset points', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', alpha=0.88, lw=0.4, ec='#bdbdbd'),
                zorder=6)
    ax.set_title(ttl, fontsize=7.5, fontweight='bold')
plt.suptitle(f"Best downlink pass of {DATE}: {bp['sat_id']} over {CITY_LABELS[bp['station']]} "
             f"({bp['delivered_gbit']:.2f} Gbit delivered)", fontsize=8.5, fontweight='bold')
plt.tight_layout()
out_c = os.path.join(FIG_DIR, 'fig09c_coverage_duration.png')
plt.savefig(out_c, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_c)
print(f"  Saved: {out_c}")

# ----------------------------------------------------------------
# 5. fig09d — top-10 DOWNLINK passes: tracks colored by delivered key + table
# ----------------------------------------------------------------
print("\nfig09d: top-10 downlink passes map + table...")
fig = plt.figure(figsize=(7.16, 8.4))
ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='#f2f2ee', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#eaf3f8', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=':', zorder=1)

vmin, vmax = top10['delivered_gbit'].min(), top10['delivered_gbit'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap(SEQ_KEY)

for rank_i, row in top10.iterrows():
    sat = sat_by_name[row['sat_id']]
    color = cmap(0.35 + 0.65 * norm(row['delivered_gbit']))
    core_lons, core_lats = plot_track_padded(
        ax, sat, row['t_rise_dt'].to_pydatetime(), row['t_set_dt'].to_pydatetime(),
        pad_min=3.0, color=color, core_lw=2.2, zorder=3)
    mid = len(core_lons) // 2
    ax.annotate(str(row['rank']), (core_lons[mid], core_lats[mid]),
                fontsize=6.5, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='circle,pad=0.12', fc='white', alpha=0.85, lw=0.4))

draw_stations(ax)
draw_islands(ax)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax, shrink=0.7, label='Delivered key (Gbit)')
ax.set_title(f'Top-10 DOWNLINK Passes of {DATE} (satellite-to-ground, single-station link)\n'
             f'by delivered key — see Fig. 09e for paired city-to-city key exchanges',
             fontsize=8.5, fontweight='bold')

ax_t = fig.add_subplot(2, 1, 2)
ax_t.axis('off')
col_labels = ['Rank', 'Satellite', 'Station', 'Local peak', 'Max elev (°)',
              'SKR (kbps)', 'Delivered (Gbit)', 'Duration (s)', 'Window (local)']
cell_rows = []
for _, r in top10.iterrows():
    hh = int(r['local_hour_peak'])
    mm = int(round((r['local_hour_peak'] - hh) * 60))
    dur_s = (r['t_set_dt'] - r['t_rise_dt']).total_seconds()
    window_str = local_window_str(r['t_rise_dt'], r['t_set_dt'], r['station'])
    cell_rows.append([int(r['rank']), r['sat_id'], CITY_LABELS[r['station']],
                      f"{hh:02d}:{mm:02d}", f"{r['max_elev_deg']:.1f}",
                      f"{r['SKR_kbps']:.0f}", f"{r['delivered_gbit']:.3f}",
                      f"{dur_s:.0f}", window_str])
table = ax_t.table(cellText=cell_rows, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(6.5)
table.scale(1.0, 1.3)
for (r_i, c_i), cell in table.get_celld().items():
    cell.set_linewidth(0.3)
    if r_i == 0:
        cell.set_text_props(fontweight='bold')
plt.tight_layout()
out_d = os.path.join(FIG_DIR, 'fig09d_top10_passes.png')
plt.savefig(out_d, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_d)
print(f"  Saved: {out_d}")

# ----------------------------------------------------------------
# 6. fig09d1 — top downlink satellites serving each ASEAN STATION
# ----------------------------------------------------------------
# NOTE (user feedback 06/07/2026): an earlier version aggregated to 6
# "countries", but the 8-station list is 3:1:1:1:1:1 (Vietnam has 3
# stations, everyone else 1) by construction (the user's own site
# selection, not a designed matched sample). Summing across country
# manufactured a fake "Vietnam wins" result: per-STATION the 8 stations
# are actually close (1440-1612 passes/day, ~12% spread) — the 3x-more-
# stations effect was the entire story, not real geometry. Fixed by
# reporting at STATION granularity (8 panels), never merged — exactly
# the same "N-stations-vs-1" fairness bug as the retracted +678% claim
# (07-4), caught before it reached the paper this time.
print("\nfig09d1: top satellites serving each ASEAN station (7-day geometric average)...")
station_top_sats = {}
station_total_avg = {}
for st in CITIES:
    sub = passes5[passes5['station'] == st]
    counts = sub['sat_id'].value_counts()
    station_top_sats[st] = counts.head(5) / 7.0
    station_total_avg[st] = counts.sum() / 7.0

fig = plt.figure(figsize=(9.5, 8.6))
grid = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])
ax_rank = fig.add_subplot(grid[0, :])
order = sorted(CITIES, key=lambda c: -station_total_avg[c])
colors_bar = [ACCENT if COUNTRY_OF[c] == 'Vietnam' else '#607d8b' for c in order]
ax_rank.barh([CITY_LABELS[c] for c in order][::-1],
            [station_total_avg[c] for c in order][::-1], color=colors_bar[::-1])
ax_rank.set_xlabel('Avg. downlink passes / day (elev ≥ 30°, 7-day template)', fontsize=7)
ax_rank.set_title('Which ASEAN ground station gets the most satellite downlink opportunities?\n'
                  '(8 individual stations — no country aggregation)',
                  fontsize=8, fontweight='bold')
ax_rank.tick_params(labelsize=7)

for k, st in enumerate(CITIES):
    ax = fig.add_subplot(grid[1 + k // 4, k % 4])
    top5 = station_top_sats[st]
    bar_color = ACCENT if COUNTRY_OF[st] == 'Vietnam' else '#1565C0'
    ax.barh(range(len(top5)), top5.values[::-1], color=bar_color)
    ax.set_yticks(range(len(top5)))
    ax.set_yticklabels(top5.index[::-1], fontsize=5.2)
    ax.set_xlabel('Avg passes/day', fontsize=6.5)
    ax.set_title(CITY_LABELS[st], fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=6)

plt.suptitle('Top 3-5 Satellites Serving Each ASEAN Station, by Downlink Frequency\n'
             '(shared Starlink Shell-1 constellation — no station operates dedicated satellites; '
             'red = Vietnam)',
             fontsize=8.5, y=1.0)
plt.tight_layout()
out_d1 = os.path.join(FIG_DIR, 'fig09d1_top_satellites_per_station.png')
plt.savefig(out_d1, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_d1)
print(f"  Saved: {out_d1}")

# ----------------------------------------------------------------
# 7. fig09d2 — busiest store-and-forward relay hub per ASEAN STATION
# ----------------------------------------------------------------
# Same station-level fix as fig09d1. This one matters even more: at
# country level Vietnam appeared to be the busiest relay hub (175,588
# relay-opp/7-day, summed over 3 stations) — but per STATION, Manila is
# the real busiest single hub (117,782), a genuine geometric finding
# (Manila's isolated position needs more store-and-forward relaying).
# The country-level view was actively hiding this, not just unfair.
print("\nfig09d2: relay hotspot per ASEAN station (store-and-forward, Task 23)...")
sf_df = pd.read_csv(FEAS_SF_CSV, comment='#')
sf_only = sf_df[sf_df['class'] == 'SF'].copy()


def parse_top5(s):
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.split(';'):
        tok = tok.strip()
        if '(' in tok:
            name, cnt = tok.rsplit('(', 1)
            out.append((name, float(cnt.rstrip(')'))))
    return out


station_relay_total = {st: 0.0 for st in CITIES}
station_relay_sat_counts = {st: {} for st in CITIES}
for _, r in sf_only.iterrows():
    for city_col in ['city_i', 'city_j']:
        st = r[city_col]
        station_relay_total[st] += r['n_relay_opportunities']
        for sat_name, cnt in parse_top5(r['top5_relay_sats']):
            station_relay_sat_counts[st][sat_name] = station_relay_sat_counts[st].get(sat_name, 0.0) + cnt

station_top_relay_sats = {
    st: sorted(station_relay_sat_counts[st].items(), key=lambda x: -x[1])[:5]
    for st in CITIES
}

fig = plt.figure(figsize=(9.5, 8.6))
grid = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])
ax_rank = fig.add_subplot(grid[0, :])
order = sorted(CITIES, key=lambda c: -station_relay_total[c])
colors_bar = [ACCENT if COUNTRY_OF[c] == 'Vietnam' else '#607d8b' for c in order]
ax_rank.barh([CITY_LABELS[c] for c in order][::-1],
            [station_relay_total[c] / 7.0 for c in order][::-1], color=colors_bar[::-1])
ax_rank.set_xlabel('Total SF relay opportunities / day (both directions, 7-day template)', fontsize=7)
ax_rank.set_title('Which ASEAN ground station acts as the busiest store-and-forward relay hub?\n'
                  '(8 individual stations — no country aggregation)',
                  fontsize=8, fontweight='bold')
ax_rank.tick_params(labelsize=7)

for k, st in enumerate(CITIES):
    ax = fig.add_subplot(grid[1 + k // 4, k % 4])
    items = station_top_relay_sats[st]
    if items:
        names = [n for n, _ in items][::-1]
        vals = [v / 7.0 for _, v in items][::-1]
        bar_color = ACCENT if COUNTRY_OF[st] == 'Vietnam' else '#2e7d32'
        ax.barh(range(len(items)), vals, color=bar_color)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(names, fontsize=5.2)
    else:
        ax.text(0.5, 0.5, 'No SF pairs', ha='center', va='center', fontsize=7,
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel('Relay opp./day', fontsize=6.5)
    ax.set_title(CITY_LABELS[st], fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=6)

plt.suptitle('Top 3-5 Relay Satellites by ASEAN Station (Store-and-Forward, Task 23 detail)\n'
             '(red = Vietnam)',
             fontsize=8.5, y=1.0)
plt.tight_layout()
out_d2 = os.path.join(FIG_DIR, 'fig09d2_relay_hotspot_per_station.png')
plt.savefig(out_d2, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_d2)
print(f"  Saved: {out_d2}")

# ----------------------------------------------------------------
# 8. fig09e — top-5 city-PAIR key exchanges (2 downlinks each)
# ----------------------------------------------------------------
print("\nfig09e: top-5 city-pair key exchanges (2 downlinks each)...")
PAIR_LEDGER_CSV = os.path.join(TEMP_DIR, 'pair_ledger.csv')
pair_ledger_all = pd.read_csv(PAIR_LEDGER_CSV)
pl = pair_ledger_all[pair_ledger_all['date_str'] == DATE].sort_values(
    'K_q_throughput_gbit', ascending=False).head(5).reset_index(drop=True)
assert len(pl) > 0, f"No pair-ledger rows found for {DATE} in {PAIR_LEDGER_CSV}"

full_ledger = pd.read_csv(os.path.join(TEMP_DIR, f'pass_ledger_{DATE}.csv'))
full_ledger['t_rise_dt'] = pd.to_datetime(full_ledger['t_rise_utc'], format='ISO8601')
full_ledger['t_set_dt'] = pd.to_datetime(full_ledger['t_set_utc'], format='ISO8601')

sf_lookup = {(r['city_i'], r['city_j']): r for _, r in sf_df.iterrows()}
sf_isl_df = pd.read_csv(FEAS_SF_ISL_CSV, comment='#')
sf_isl_lookup = {(r['city_i'], r['city_j']): r for _, r in sf_isl_df.iterrows()}


def representative_pass(pair_key, station, top_sat):
    sub = full_ledger[(full_ledger['station'] == station) &
                      (full_ledger['alg2_assigned_pair'] == pair_key)]
    if sub.empty:
        return None
    exact = sub[sub['sat_id'] == top_sat]
    target = exact if not exact.empty else sub
    return target.loc[target['delivered_gbit'].idxmax()]


def short_sat(sat_id):
    """STARLINK-3397 -> #3397 -- keeps the table columns narrow; the table
    caption spells out the abbreviation once."""
    return '#' + sat_id.replace('STARLINK-', '')


fig = plt.figure(figsize=(11.0, 13.0))
grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.5])
table_rows = []
for rank_i, prow in pl.iterrows():
    pa, pb = prow['pair_a'], prow['pair_b']
    pair_key = f"{pa}|{pb}"
    row_a = representative_pass(pair_key, pa, prow['top_sat_A'])
    row_b = representative_pass(pair_key, pb, prow['top_sat_B'])

    ax = fig.add_subplot(grid[rank_i // 3, rank_i % 3], projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=':', zorder=1)
    draw_islands(ax)
    for city_key in (pa, pb):
        gst = GROUND_STATIONS[city_key]
        ax.scatter(gst['lon'], gst['lat'], marker='^', s=40, color=ACCENT,
                   edgecolor='black', linewidth=0.3, zorder=6, transform=ccrs.PlateCarree())
        ax.annotate(CITY_LABELS[city_key], (gst['lon'], gst['lat']), xytext=(3, 3),
                    textcoords='offset points', fontsize=6, zorder=7,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', alpha=0.85, lw=0.3))

    dur_a = dur_b = window_a = window_b = 'N/A'
    sat_a_str = sat_b_str = 'N/A'
    if row_a is not None:
        t_r, t_s = row_a['t_rise_dt'], row_a['t_set_dt']
        plot_track_padded(ax, sat_by_name[row_a['sat_id']], t_r.to_pydatetime(), t_s.to_pydatetime(),
                          pad_min=3.0, color=SIDE_A_COLOR, core_lw=2.0, zorder=3)
        dur_a = f"{(t_s - t_r).total_seconds():.0f}"
        window_a = local_window_str(t_r, t_s, pa)
        sat_a_str = short_sat(row_a['sat_id'])
    if row_b is not None:
        t_r, t_s = row_b['t_rise_dt'], row_b['t_set_dt']
        plot_track_padded(ax, sat_by_name[row_b['sat_id']], t_r.to_pydatetime(), t_s.to_pydatetime(),
                          pad_min=3.0, color=SIDE_B_COLOR, core_lw=2.0, zorder=3)
        dur_b = f"{(t_s - t_r).total_seconds():.0f}"
        window_b = local_window_str(t_r, t_s, pb)
        sat_b_str = short_sat(row_b['sat_id'])

    sf_fwd = sf_lookup.get((pa, pb))
    cls = sf_fwd['class'] if sf_fwd is not None else '?'
    if cls == 'DUAL':
        sim_s = compute_dual_window_seconds(full_ledger, pa, pb)
        metric_str = (f"DUAL {sf_fwd['dual_pct']:.1f}% of 24h\n"
                      + (f"{sim_s:.0f}s simultaneous today" if sim_s > 0
                        else "no overlap today"))
    else:
        # ISL multi-hop latency (Task 24.3) -- NOT the legacy single-hop
        # v1 numbers (which ran 700+ min for some pairs due to a wrong
        # modeling assumption, since retracted; see PROVENANCE.md sec 6b).
        isl_fwd = sf_isl_lookup.get((pa, pb))
        isl_rev = sf_isl_lookup.get((pb, pa))
        lat_fwd = isl_fwd['latency_time_optimal_min'] if isl_fwd is not None else float('nan')
        lat_rev = isl_rev['latency_time_optimal_min'] if isl_rev is not None else float('nan')
        metric_str = (f"{CITY_LABELS[pa]}->{CITY_LABELS[pb]}: {lat_fwd:.1f}' (ISL)\n"
                      f"{CITY_LABELS[pb]}->{CITY_LABELS[pa]}: {lat_rev:.1f}' (ISL)")

    ax.set_title(f"#{rank_i+1}: {CITY_LABELS[pa]} <-> {CITY_LABELS[pb]}  [{cls}]",
                fontsize=7, fontweight='bold')

    table_rows.append([
        rank_i + 1, f"{CITY_LABELS[pa]}\n<->{CITY_LABELS[pb]}", cls,
        f"{prow['K_q_throughput_gbit']:.3f}",
        f"{CITY_LABELS[pa]} {sat_a_str}\n{dur_a}s ({window_a})",
        f"{CITY_LABELS[pb]} {sat_b_str}\n{dur_b}s ({window_b})",
        metric_str,
    ])

ax_t = fig.add_subplot(grid[2, :])
ax_t.axis('off')
col_labels = ['Rank', 'City pair', 'Class', 'K_q\n(Gbit)', 'Side A downlink\n(station, sat, dur, window)',
              'Side B downlink\n(station, sat, dur, window)', 'SF latency /\nDUAL window']
col_widths = [0.05, 0.12, 0.07, 0.08, 0.24, 0.24, 0.20]
table = ax_t.table(cellText=table_rows, colLabels=col_labels, loc='center',
                   cellLoc='center', colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.0, 3.4)
for (r_i, c_i), cell in table.get_celld().items():
    cell.set_linewidth(0.3)
    if r_i == 0:
        cell.set_text_props(fontweight='bold')
fig.text(0.5, 0.005, 'Satellite IDs abbreviated: #NNNN = STARLINK-NNNN',
         ha='center', fontsize=6.5, style='italic')

plt.suptitle(f'Top-5 City-Pair Key Exchanges of {DATE} — each pair needs TWO downlinks\n'
             '(trusted-node greedy allocation, Task 21; blue = side A, green = side B)',
             fontsize=8.5, y=1.0)
plt.tight_layout(rect=[0, 0.02, 1, 1])
out_e = os.path.join(FIG_DIR, 'fig09e_top5_pair_exchanges.png')
plt.savefig(out_e, dpi=300, bbox_inches='tight')
plt.close()
output_files.append(out_e)
print(f"  Saved: {out_e}")

# ----------------------------------------------------------------
# 9. Copy to canonical latex_paper_3/figures/ if a variant dir was used
# ----------------------------------------------------------------
if _VARIANT and os.path.abspath(FIG_DIR) != os.path.abspath(CANONICAL_FIG_DIR):
    for f in output_files:
        shutil.copy2(f, os.path.join(CANONICAL_FIG_DIR, os.path.basename(f)))
    print(f"\nAlso copied {len(output_files)} figures to canonical {CANONICAL_FIG_DIR}")

# ----------------------------------------------------------------
# 10. Verify + provenance
# ----------------------------------------------------------------
verify = {
    "chosen_date": DATE,
    "date_selection_mode": ("explicit --date" if _args.date else
                            "--balanced" if _args.balanced else "auto (max top-1 Gbit)"),
    "n_sats_1h_window": str(len(sats_1h)),
    "n_sats_hanoi_2000s": str(len(sats_b)),
    "best_pass": f"{bp['sat_id']} over {bp['station']} ({bp['delivered_gbit']:.3f} Gbit)",
    "best_pass_peak_subpoint": f"({sub_lat:.2f}N, {sub_lon:.2f}E), h={h_km:.0f} km",
    "coverage_radius_40deg_km": f"{ground_coverage_radius_km(h_km, 40.0):.0f}",
    "fig09a_top10_dominant_region": f"{dom_region} ({shares_a[dom_region]*100:.0f}%)",
    "fig09d1_busiest_station_downlink": max(station_total_avg, key=station_total_avg.get),
    "fig09d1_station_spread_pct": f"{(max(station_total_avg.values())/min(station_total_avg.values())-1)*100:.0f}",
    "fig09d2_busiest_station_relay": max(station_relay_total, key=station_relay_total.get),
    "n_pairs_in_fig09e": str(len(pl)),
    "paracel_region_source": "So TT&TT Bac Giang (stttt.bacgiang.gov.vn)",
    "paracel_island_marker": "Phu Lam / Woody Island, 16d50.2'N 112d20'E (same source)",
    "spratly_region_source": "So TT&TT Bac Giang (stttt.bacgiang.gov.vn)",
    "spratly_island_marker": "Truong Sa Lon / Spratly Island, 8d38'30\"N 111d55'55\"E "
                            "(Vietnamese Wikipedia, sovereignty marker)",
}
save_verify_numbers(verify, "orbit_maps")
save_provenance(
    script_name="11g_orbit_maps",
    params={"date": DATE, "extent": EXTENT, "grid": "140x140",
            "window_fig09a_h": 1.0, "window_fig09b_s": 2000,
            "region_grouping_fig09a": REGION_OF,
            "country_color_lookup_fig09d1_d2": COUNTRY_OF,
            "island_boxes": {"paracel": PARACEL_BOX, "spratly": SPRATLY_BOX},
            "island_markers": {"paracel": PARACEL_ISLAND, "spratly": SPRATLY_ISLAND},
            "revision": "Task 24.2 (06/07/2026): fig09a density fix, track padding "
                       "+direction arrows, --balanced day picker, fig09d relabeled "
                       "as downlink-only, fig09d1/fig09d2 NEW (per-STATION, not "
                       "per-country -- an interim version aggregated Vietnam's 3 "
                       "stations into one 'country' total and manufactured a fake "
                       "Vietnam-wins ranking, same bug class as the retracted "
                       "+678% claim; fixed to report all 8 stations individually "
                       "after user feedback), fig09e NEW (paired city-to-city key "
                       "exchanges, 2 downlinks per pair), Hoang Sa/Truong Sa added "
                       "to every map",
            "notebook_reuse": "layout patterns from archived BBM92 cells 35/42/46 only; "
                              "broken cell 44 (all-zero QBER) NOT used — all numbers here "
                              "come from this project's own modules"},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=output_files,
    data_sources={"top-10 passes (Task 22)": chosen,
                  "pass table (Task 5, elev>=30, 7-day)": PASS_CSV,
                  "SF relay detail (Task 23)": FEAS_SF_CSV,
                  "pair ledger (Task 21)": PAIR_LEDGER_CSV,
                  "full pass ledger for chosen day (Task 22)":
                      os.path.join(TEMP_DIR, f'pass_ledger_{DATE}.csv'),
                  "Shell-1 TLE": TLE_PATH},
    formulas={
        "Grid elevation": "tan(elev) = (cos psi - R/(R+h)) / sin psi, psi = great-circle "
                          "angle grid-point -> subsatellite point (vectorized numpy, "
                          "consistent with modules/link_geometry.py)",
        "Coverage circle": "spherical small circle of central angle "
                           "ground_coverage_radius_km(h,40)/R_E around the peak subpoint",
        "Region entropy (--balanced)": "-sum(p*ln(p))/ln(3) over the 3 REGION_ORDER shares "
                                       "of top-10 delivered_gbit; maximized at an even split",
        "DUAL simultaneous window": "max, over satellites visible to both stations that day, "
                                    "of the [t_rise,t_set] interval overlap (seconds)",
    },
)
print(f"\nTotal runtime: {time.time() - t0:.1f}s")
