"""
=============================================================================
Constellation Density Visualization — fig13a-d & fig15a-d
=============================================================================

Tạo hình minh họa cho phân tích mật độ chòm sao vệ tinh:
- fig15: Bản đồ ASEAN với ground tracks cho 4 kích thước (176, 500, 1584, 4000)
- fig13: SKR timeseries routing cho 4 kích thước

Phương pháp:
- Tạo synthetic Walker-delta constellation (TLE) cho mỗi kích thước
- Propagate quỹ đạo bằng Skyfield (SGP4)
- Vẽ ground tracks, vị trí hiện tại, trạm mặt đất

Tác giả: Trương Tuấn Nghĩa (USTH)
Ngày: 2026-05-25
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import datetime, timezone, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from skyfield.api import EarthSatellite, wgs84, load
from orbital_mechanics import (
    _TS, GROUND_STATIONS, MIN_ELEVATION,
    compute_elevation, make_skyfield_satellite,
)
from channel_model import compute_channel
from sikd_performance import compute_sikd_performance

# Try cartopy, fallback to plain matplotlib
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not installed, using plain matplotlib for maps")

# =============================================================================
# IEEE STYLE
# =============================================================================
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# =============================================================================
# CONSTANTS
# =============================================================================
RE_KM = 6371.0
MU_EARTH = 398600.4418  # km³/s² — standard gravitational parameter

# Constellation configurations: (n_planes, sats_per_plane)
CONSTELLATIONS = {
    176:  (8, 22),
    500:  (10, 50),
    1584: (72, 22),
    4000: (40, 100),
}

# System parameters
INC_DEG = 53.0      # Starlink inclination
ALT_KM = 550.0     # orbital altitude
EPOCH_DT = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)

# ASEAN map extent
LON_MIN, LON_MAX = 95.0, 130.0
LAT_MIN, LAT_MAX = -5.0, 25.0

# Weather: Hanoi dry season (January)
V_KM_DRY = 10.0
R_MM_H_DRY = 0.0
P_CLOUD_DRY = 0.20

# SIKD system params
PT = 1.0
MK = 0.05
MD = 0.5
ISO_DB = 15.0
ZETA_SCALE = 2.0
RB = 1e9


# =============================================================================
# PART 1: Synthetic TLE Generation (Walker-Delta)
# =============================================================================

def generate_walker_constellation(n_planes, sats_per_plane, inc_deg, alt_km, epoch_dt):
    """
    Generate synthetic TLE strings for a Walker-delta constellation.

    Walker-delta: RAAN evenly spaced, mean anomaly evenly spaced within each plane,
    with inter-plane phasing = 360° / (n_planes × sats_per_plane).

    Returns list of dicts with 'name', 'line1', 'line2'.
    """
    n_total = n_planes * sats_per_plane

    # Orbital period and mean motion
    a_km = RE_KM + alt_km
    period_s = 2 * np.pi * np.sqrt(a_km**3 / MU_EARTH)
    mean_motion = 86400.0 / period_s  # rev/day

    # Epoch in TLE format: YYDDD.DDDDDDDD
    year_2digit = epoch_dt.year % 100
    day_of_year = (epoch_dt - datetime(epoch_dt.year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 1.0
    epoch_str = f"{year_2digit:02d}{day_of_year:012.8f}"

    satellites = []
    sat_num = 70000  # arbitrary NORAD catalog start

    for plane_idx in range(n_planes):
        raan = 360.0 * plane_idx / n_planes
        for sat_idx in range(sats_per_plane):
            # Walker-delta phasing
            mean_anomaly = (360.0 * sat_idx / sats_per_plane
                           + 360.0 * plane_idx / n_total)
            mean_anomaly = mean_anomaly % 360.0

            sat_num += 1
            name = f"WALKER-{n_total:04d}-{plane_idx:02d}{sat_idx:03d}"

            # Build TLE lines (simplified, checksum=0 for synthetic)
            line1 = (f"1 {sat_num:05d}U 26001A   {epoch_str}  .00000000  "
                     f"00000-0  00000-0 0  9990")
            line2 = (f"2 {sat_num:05d} {inc_deg:8.4f} {raan:8.4f} 0001000 "
                     f"  0.0000 {mean_anomaly:8.4f} {mean_motion:11.8f}00000")

            satellites.append({"name": name, "line1": line1, "line2": line2})

    return satellites


def create_skyfield_satellites(tle_list):
    """Convert TLE dicts to Skyfield EarthSatellite objects."""
    sats = []
    for tle in tle_list:
        try:
            sat = EarthSatellite(tle["line1"], tle["line2"], tle["name"], _TS)
            sats.append(sat)
        except Exception:
            pass  # skip malformed TLEs
    return sats


# =============================================================================
# PART 2: Ground Track Computation
# =============================================================================

def compute_ground_tracks(satellites, t0_skyfield, duration_min=90, dt_s=60):
    """
    Compute ground tracks (subpoint lat/lon) for all satellites.

    Parameters
    ----------
    satellites : list of EarthSatellite
    t0_skyfield : Skyfield Time object (start time)
    duration_min : propagation duration (minutes)
    dt_s : time step (seconds)

    Returns
    -------
    list of (lats, lons) arrays — one per satellite
    """
    n_steps = int(duration_min * 60 / dt_s) + 1
    # Build time array
    jd_base = t0_skyfield.tt
    dt_days = np.arange(n_steps) * dt_s / 86400.0
    times = _TS.tt_jd(jd_base + dt_days)

    tracks = []
    for sat in satellites:
        try:
            geocentric = sat.at(times)
            subpoint = wgs84.subpoint(geocentric)
            lats = subpoint.latitude.degrees
            lons = subpoint.longitude.degrees
            tracks.append((np.array(lats), np.array(lons)))
        except Exception:
            tracks.append((np.array([]), np.array([])))

    return tracks


def compute_current_positions(satellites, t_skyfield):
    """Get current lat/lon for all satellites at time t."""
    positions = []
    for sat in satellites:
        try:
            geocentric = sat.at(t_skyfield)
            subpoint = wgs84.subpoint(geocentric)
            lat = float(subpoint.latitude.degrees)
            lon = float(subpoint.longitude.degrees)
            positions.append((lat, lon))
        except Exception:
            positions.append((None, None))
    return positions


# =============================================================================
# PART 3: fig15 — Coverage Maps with Ground Tracks
# =============================================================================

def add_hoang_sa_truong_sa(ax, transform=None):
    """Add Hoàng Sa and Trường Sa archipelago markers."""
    kwargs = {'transform': transform} if transform else {}

    # Hoàng Sa (Paracel Islands)
    hs_rect = Rectangle((111.0, 15.7), 1.9, 1.4,
                         linewidth=1.2, edgecolor='red', facecolor='none',
                         linestyle='--', **kwargs)
    ax.add_patch(hs_rect)
    ax.text(111.95, 17.2, 'Hoàng Sa', fontsize=6, color='red',
            ha='center', va='bottom', fontweight='bold', **kwargs)

    # Trường Sa (Spratly Islands)
    ts_rect = Rectangle((111.0, 6.5), 6.5, 5.5,
                         linewidth=1.2, edgecolor='red', facecolor='none',
                         linestyle='--', **kwargs)
    ax.add_patch(ts_rect)
    ax.text(114.25, 12.2, 'Trường Sa', fontsize=6, color='red',
            ha='center', va='bottom', fontweight='bold', **kwargs)


def plot_coverage_maps(constellations_data, output_path):
    """
    Plot 2×2 coverage maps with ground tracks for 4 constellation sizes.

    Parameters
    ----------
    constellations_data : dict {n_total: (satellites, tracks, positions)}
    output_path : path to save PNG
    """
    sizes = [176, 500, 1584, 4000]
    labels = ['(a) N = 176 (8×22)', '(b) N = 500 (10×50)',
              '(c) N = 1584 (72×22)', '(d) N = 4000 (40×100)']

    if HAS_CARTOPY:
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0))

    axes_flat = axes.flatten()

    for idx, (n_total, label) in enumerate(zip(sizes, labels)):
        ax = axes_flat[idx]
        sats, tracks, positions = constellations_data[n_total]

        if HAS_CARTOPY:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', color='gray')
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                         xlocs=range(95, 135, 5), ylocs=range(-5, 30, 5),
                         x_inline=False, y_inline=False)
            transform = ccrs.PlateCarree()
        else:
            ax.set_xlim(LON_MIN, LON_MAX)
            ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            transform = None

        # Ground tracks (subsample for large constellations)
        max_tracks_to_plot = min(len(tracks), 200)
        track_indices = np.linspace(0, len(tracks)-1, max_tracks_to_plot, dtype=int)

        for ti in track_indices:
            lats, lons = tracks[ti]
            if len(lats) == 0:
                continue
            # Filter to ASEAN region for cleaner plot
            mask = ((lons >= LON_MIN - 10) & (lons <= LON_MAX + 10) &
                    (lats >= LAT_MIN - 10) & (lats <= LAT_MAX + 10))
            if not np.any(mask):
                continue
            # Split at longitude wraps
            plot_kwargs = {'color': '#4a90d9', 'alpha': 0.15, 'linewidth': 0.4}
            if transform:
                plot_kwargs['transform'] = transform
            # Simple plot (no wrap handling needed for ASEAN region)
            ax.plot(lons[mask], lats[mask], **plot_kwargs)

        # Current positions (scatter)
        pos_lats = [p[0] for p in positions if p[0] is not None]
        pos_lons = [p[1] for p in positions if p[1] is not None]
        # Filter to visible region
        vis_mask = [(LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX)
                    for lat, lon in zip(pos_lats, pos_lons)]
        vis_lats = [lat for lat, m in zip(pos_lats, vis_mask) if m]
        vis_lons = [lon for lon, m in zip(pos_lons, vis_mask) if m]

        scatter_kwargs = {'s': 8, 'color': '#e74c3c', 'zorder': 5, 'alpha': 0.7}
        if transform:
            scatter_kwargs['transform'] = transform
        if vis_lats:
            ax.scatter(vis_lons, vis_lats, **scatter_kwargs)

        # Ground stations
        for city, info in GROUND_STATIONS.items():
            gs_kwargs = {'s': 40, 'marker': '^', 'color': '#2ecc71',
                         'edgecolors': 'black', 'linewidths': 0.5, 'zorder': 10}
            txt_kwargs = {'fontsize': 5.5, 'ha': 'left', 'va': 'bottom'}
            if transform:
                gs_kwargs['transform'] = transform
                txt_kwargs['transform'] = transform
            if LON_MIN <= info['lon'] <= LON_MAX and LAT_MIN <= info['lat'] <= LAT_MAX:
                ax.scatter(info['lon'], info['lat'], **gs_kwargs)
                ax.text(info['lon'] + 0.5, info['lat'] + 0.3,
                        city.replace('_', ' ').title(), **txt_kwargs)

        # Hoàng Sa / Trường Sa
        add_hoang_sa_truong_sa(ax, transform=transform)

        # Title and stats
        n_in_view = len(vis_lats)
        ax.set_title(f"{label}\n{n_in_view} sats in ASEAN view", fontsize=8)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# PART 4: fig13 — SKR Timeseries Routing
# =============================================================================

def compute_skr_for_zenith(zenith_deg, V_km=V_KM_DRY, R_mm_h=R_MM_H_DRY):
    """Compute SKR_norm for a given zenith angle using full pipeline."""
    if zenith_deg >= 80.0:
        return 0.0
    ch = compute_channel(
        H_S_km=ALT_KM,
        zeta_deg=max(zenith_deg, 0.1),
        a_R=0.05,
        lambda_nm=1550.0,
        theta_C_urad=10.0,
        V_km=V_km,
        R_mm_h=R_mm_h,
    )
    perf = compute_sikd_performance(
        hg=ch['hg'], hl=ch['hl'], sigma_X2=ch['sigma_X2'],
        PT=PT, mK=MK, mD=MD, Iso_dB=ISO_DB, zeta_scale=ZETA_SCALE,
    )
    return perf['SKR_norm']


def compute_skr_timeseries_simple(satellites, lat_deg, lon_deg, t0_skyfield,
                                   duration_min=120, dt_s=30):
    """
    Simplified SKR timeseries: at each step, find best visible satellite,
    compute SKR using greedy routing.

    Returns
    -------
    dict with:
        time_min : array of time in minutes
        skr_kbps : array of SKR (kbps), 0 when no satellite visible
        n_visible : array of number of visible satellites
        coverage_frac : fraction of time with at least 1 satellite
    """
    n_steps = int(duration_min * 60 / dt_s) + 1
    jd_base = t0_skyfield.tt
    dt_days = np.arange(n_steps) * dt_s / 86400.0

    time_min = np.arange(n_steps) * dt_s / 60.0
    skr_kbps = np.zeros(n_steps)
    n_visible = np.zeros(n_steps, dtype=int)

    ground = wgs84.latlon(lat_deg, lon_deg)

    for step_i in range(n_steps):
        t = _TS.tt_jd(jd_base + dt_days[step_i])

        # Find visible satellites and their elevations
        best_elev = -90.0
        best_zenith = 90.0

        for sat in satellites:
            try:
                diff = sat - ground
                topo = diff.at(t)
                alt_deg, _, _ = topo.altaz()
                elev = float(alt_deg.degrees)
                if elev >= MIN_ELEVATION:
                    n_visible[step_i] += 1
                    if elev > best_elev:
                        best_elev = elev
                        best_zenith = 90.0 - elev
            except Exception:
                continue

        # Compute SKR for best satellite
        if best_elev >= MIN_ELEVATION:
            skr_norm = compute_skr_for_zenith(best_zenith)
            # Apply 3-state weather: SKR_eff = p_clear × SKR
            p_clear = 1.0 - P_CLOUD_DRY - 0.15
            skr_kbps[step_i] = skr_norm * RB * p_clear / 1e3

    coverage_frac = np.sum(n_visible > 0) / n_steps

    return {
        'time_min': time_min,
        'skr_kbps': skr_kbps,
        'n_visible': n_visible,
        'coverage_frac': coverage_frac,
    }


def plot_skr_timeseries(timeseries_data, output_path):
    """
    Plot 4×1 stacked SKR timeseries for 4 constellation sizes.

    Parameters
    ----------
    timeseries_data : dict {n_total: timeseries_result}
    output_path : path to save PNG
    """
    sizes = [176, 500, 1584, 4000]
    labels = ['(a) N = 176 (8×22)', '(b) N = 500 (10×50)',
              '(c) N = 1584 (72×22)', '(d) N = 4000 (40×100)']
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']

    fig, axes = plt.subplots(4, 1, figsize=(7.16, 8.0), sharex=True)

    # Find global max for consistent y-axis
    all_max = max(np.max(ts['skr_kbps']) for ts in timeseries_data.values())
    y_max = all_max * 1.15

    for idx, (n_total, label, color) in enumerate(zip(sizes, labels, colors)):
        ax = axes[idx]
        ts = timeseries_data[n_total]

        time_min = ts['time_min']
        skr = ts['skr_kbps']
        coverage = ts['coverage_frac']
        avg_skr = np.mean(skr[skr > 0]) if np.any(skr > 0) else 0
        min_skr = np.min(skr[skr > 0]) if np.any(skr > 0) else 0
        max_skr = np.max(skr)

        # Fill area under curve
        ax.fill_between(time_min, 0, skr, alpha=0.3, color=color)
        ax.plot(time_min, skr, color=color, linewidth=0.8)

        # Average line
        ax.axhline(avg_skr, color=color, linestyle='--', linewidth=0.6, alpha=0.7)

        # Annotations
        ax.text(0.02, 0.92, f"{label}", transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

        # Improvement vs baseline (176)
        baseline_avg = np.mean(timeseries_data[176]['skr_kbps'][timeseries_data[176]['skr_kbps'] > 0])
        improvement = (avg_skr / baseline_avg - 1) * 100 if baseline_avg > 0 else 0

        stats_text = (f"Avg: {avg_skr:.0f} kbps | "
                      f"Min: {min_skr:.0f} | Max: {max_skr:.0f}")
        if n_total > 176:
            stats_text += f" | vs 176: +{improvement:.0f}%"

        ax.text(0.98, 0.92, stats_text,
                transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
                color='gray')

        ax.set_ylabel('SKR (kbps)')
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (minutes)')
    axes[0].set_title('SKR Timeseries — Greedy Routing, Hanoi Dry Season (Jan)\n'
                      'More satellites → better zenith angles → higher SKR',
                      fontsize=9, fontweight='bold')

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CONSTELLATION DENSITY VISUALIZATION")
    print("=" * 70)
    print()

    output_dir = os.path.join(os.path.dirname(__file__), 'diagrams')
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Generate constellations ---
    print("[1/4] Generating synthetic Walker-delta constellations...")
    t0 = _TS.from_datetime(EPOCH_DT)

    all_constellations = {}  # {n_total: list of EarthSatellite}

    for n_total, (n_planes, sats_per_plane) in CONSTELLATIONS.items():
        print(f"  N={n_total:>5} ({n_planes}×{sats_per_plane})...", end=" ")
        tle_list = generate_walker_constellation(
            n_planes, sats_per_plane, INC_DEG, ALT_KM, EPOCH_DT
        )
        sats = create_skyfield_satellites(tle_list)
        all_constellations[n_total] = sats
        print(f"{len(sats)} satellites created")

    # --- Step 2: Compute ground tracks ---
    print()
    print("[2/4] Computing ground tracks (90 min propagation)...")

    constellations_map_data = {}  # {n_total: (sats, tracks, positions)}

    for n_total, sats in all_constellations.items():
        print(f"  N={n_total:>5}: ", end="")
        # For large constellations, subsample for track computation
        max_sats_for_tracks = min(len(sats), 300)
        track_indices = np.linspace(0, len(sats)-1, max_sats_for_tracks, dtype=int)
        sats_subset = [sats[i] for i in track_indices]

        tracks = compute_ground_tracks(sats_subset, t0, duration_min=90, dt_s=60)
        positions = compute_current_positions(sats, t0)
        constellations_map_data[n_total] = (sats, tracks, positions)
        print(f"{len(tracks)} tracks, {len(positions)} positions")

    # --- Step 3: Plot fig15 ---
    print()
    print("[3/4] Plotting fig15 — Coverage maps with ground tracks...")
    fig15_path = os.path.join(output_dir, 'fig15_coverage_density_comparison.png')
    plot_coverage_maps(constellations_map_data, fig15_path)

    # --- Step 4: Compute SKR timeseries and plot fig13 ---
    print()
    print("[4/4] Computing SKR timeseries (this may take a few minutes)...")

    # Hanoi coordinates
    hanoi_lat = GROUND_STATIONS['hanoi']['lat']
    hanoi_lon = GROUND_STATIONS['hanoi']['lon']

    timeseries_data = {}

    for n_total, sats in all_constellations.items():
        print(f"  N={n_total:>5}: computing SKR timeseries...", end=" ", flush=True)
        # Use ALL satellites for correct coverage statistics
        # Adjust time step for computational feasibility
        if n_total <= 500:
            dt = 30  # 30s step for small constellations
        elif n_total <= 1584:
            dt = 60  # 60s step
        else:
            dt = 120  # 120s step for 4000

        ts = compute_skr_timeseries_simple(
            sats, hanoi_lat, hanoi_lon, t0,
            duration_min=120, dt_s=dt,
        )
        timeseries_data[n_total] = ts
        print(f"coverage={ts['coverage_frac']*100:.0f}%")

    fig13_path = os.path.join(output_dir, 'fig13_skr_timeseries_density_comparison.png')
    plot_skr_timeseries(timeseries_data, fig13_path)

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'N_sats':>8} | {'Coverage':>10} | {'Avg SKR (kbps)':>15} | {'Sats in ASEAN':>14}")
    print("-" * 55)
    for n_total in [176, 500, 1584, 4000]:
        ts = timeseries_data[n_total]
        avg_skr = np.mean(ts['skr_kbps'][ts['skr_kbps'] > 0]) if np.any(ts['skr_kbps'] > 0) else 0
        _, _, positions = constellations_map_data[n_total]
        n_asean = sum(1 for lat, lon in positions
                      if lat is not None and LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX)
        print(f"{n_total:>8} | {ts['coverage_frac']*100:>8.0f}% | {avg_skr:>13.0f} | {n_asean:>14}")

    print()
    print(f"Output files:")
    print(f"  {fig15_path}")
    print(f"  {fig13_path}")


if __name__ == "__main__":
    main()
