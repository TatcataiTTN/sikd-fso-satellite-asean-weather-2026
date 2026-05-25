"""
=============================================================================
fig13b/c/d + fig15b/c/d — Constellation Density Routing & Coverage
=============================================================================

Tạo figures y hệt style gốc nhưng cho 500, 1584, 4000 vệ tinh:
- fig13b/c/d: SKR timeseries (greedy vs baseline) — same style as fig13
- fig15b/c/d: Coverage map (elevation heatmap) — same style as fig15

Tác giả: Trương Tuấn Nghĩa (USTH)
Ngày: 2026-05-25
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy.spatial import ConvexHull
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from skyfield.api import EarthSatellite, wgs84, load
from orbital_mechanics import (
    _TS, GROUND_STATIONS, make_skyfield_satellite, parse_tle_block,
    compute_coverage_grid,
)
from routing import compute_skr_timeseries, compute_routing_improvement
from weather_model import get_city_params

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================================================================
# IEEE STYLE (same as notebook 04)
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
# CONSTANTS (same as notebook 04)
# =============================================================================
T_REF = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
MIN_ELEV = 10.0
DURATION_HOURS = 3.0
STEP_MINUTES = 1.0
GS = GROUND_STATIONS['hanoi']

# Constellation configs
CONFIGS = {
    500:  (10, 50,  '500 sats (10 planes × 50 sats/plane)'),
    1584: (72, 22,  '1584 sats (72 planes × 22 sats/plane)'),
    4000: (40, 100, '4000 sats (40 planes × 100 sats/plane)'),
}

# TLE params (same as notebook 04 cell-3)
INC = 53.0          # degrees — Starlink inclination
N_REV = 15.05       # rev/day — corresponds to ~550 km altitude
EPOCH = '26071.50000000'  # 2026 day 71.5 = March 12

# Hoàng Sa / Trường Sa coordinates (from notebook 04)
_HOANG_SA_COORDS = [
    (112.3333, 16.8333), (112.7333, 16.6667), (111.2000, 15.7833),
    (111.7667, 16.4667), (111.6000, 16.5333), (111.6667, 16.2500),
    (112.5333, 16.0333), (111.5167, 17.0833), (112.2667, 16.9667),
]
_TRUONG_SA_COORDS = [
    (111.9167, 8.6333), (114.3333, 11.4333), (114.3667, 10.1833),
    (114.3333, 9.8833), (114.4833, 10.3833), (112.9167, 7.8833),
    (113.3167, 8.1500), (115.8167, 9.7167), (114.2167, 11.0500),
    (111.5000, 8.8500), (113.8333, 10.0000),
]


# =============================================================================
# HELPER: Draw Hoàng Sa / Trường Sa (same as notebook 04)
# =============================================================================

def draw_vietnam_islands(ax, use_cartopy=True):
    """Draw Hoang Sa and Truong Sa using actual island positions + convex hull."""
    kwargs = {}
    if use_cartopy:
        kwargs = {'transform': ccrs.PlateCarree()}

    for coords, label, offset in [
        (_HOANG_SA_COORDS, 'Quần đảo\nHoàng Sa', (0.3, 0.3)),
        (_TRUONG_SA_COORDS, 'Quần đảo\nTrường Sa', (0.3, 0.3)),
    ]:
        pts = np.array(coords)
        lons, lats = pts[:, 0], pts[:, 1]
        ax.scatter(lons, lats, s=8, color='#c00000', marker='^', zorder=6,
                   linewidths=0.3, edgecolors='darkred', **kwargs)
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color='#c00000',
                linewidth=1.0, linestyle='--', zorder=5, **kwargs)
        cx, cy = lons.mean(), lats.mean()
        ax.text(cx + offset[0], cy + offset[1], label,
                fontsize=5.5, color='#c00000', ha='center', va='center',
                fontweight='bold', zorder=7, **kwargs)


# =============================================================================
# CONSTELLATION GENERATION (same method as notebook 04 cell-3)
# =============================================================================

def generate_constellation(n_planes, sats_per_plane):
    """
    Generate synthetic Walker constellation.

    Giải thích cấu hình:
    - n_planes: số mặt phẳng quỹ đạo (orbital planes)
      Mỗi plane là 1 vòng tròn quỹ đạo nghiêng 53° so với xích đạo
      Các plane cách đều nhau: RAAN = 360°/n_planes
    - sats_per_plane: số vệ tinh trong mỗi plane
      Cách đều nhau: mean anomaly spacing = 360°/sats_per_plane

    Ví dụ: 8×22 = 8 planes, mỗi plane 22 sats, tổng 176
    """
    lines = []
    sat_num = 50000

    for p_idx in range(n_planes):
        raan = p_idx * (360.0 / n_planes)
        for k in range(sats_per_plane):
            M0 = k * (360.0 / sats_per_plane)
            name = f'SL-P{p_idx:02d}-S{k:03d}'
            l1 = (f'1 {sat_num:05d}U 19029X   {EPOCH}  .00001000  '
                  f'00000-0  10000-3 0  999{k%10}')
            l2 = (f'2 {sat_num:05d} {INC:8.4f} {raan:8.4f} 0001000 '
                  f' 90.0000 {M0:8.4f} {N_REV:.8f} 0000{k%10}')
            lines.append(name)
            lines.append(l1)
            lines.append(l2)
            sat_num += 1

    tle_dicts = parse_tle_block('\n'.join(lines))
    satellites = [make_skyfield_satellite(d) for d in tle_dicts]
    return satellites


# =============================================================================
# FIG15 — COVERAGE MAP (exact same style as fig15_coverage_map_asean.png)
# =============================================================================

def plot_fig15_variant(satellites, n_total, label, output_path):
    """Plot coverage map — exact same style as original fig15."""
    t_snap = _TS.from_datetime(T_REF)

    print(f"  Computing coverage grid ({n_total} sats)...", end=" ", flush=True)
    grid = compute_coverage_grid(
        satellites=satellites,
        t=t_snap,
        lat_range=(0, 25),
        lon_range=(95, 130),
        resolution_deg=1.0,
        min_elevation=MIN_ELEV,
    )
    print(f"coverage={grid['coverage_frac']:.1%}")

    fig, ax = plt.subplots(figsize=(7.16, 5.5),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', color='gray')
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)

    im = ax.contourf(
        grid['lons'], grid['lats'],
        np.where(grid['has_coverage'], grid['max_elevation'], np.nan),
        levels=np.linspace(10, 90, 17), cmap='YlOrRd', extend='neither',
        transform=ccrs.PlateCarree()
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Max Elevation (degrees)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Ground stations
    for city, gs_info in GROUND_STATIONS.items():
        ax.plot(gs_info['lon'], gs_info['lat'], 'b^', markersize=6,
                transform=ccrs.PlateCarree())
        ax.annotate(city, (gs_info['lon'], gs_info['lat']),
                    textcoords='offset points', xytext=(4, 2), fontsize=6,
                    transform=ccrs.PlateCarree())

    # Hoàng Sa / Trường Sa
    draw_vietnam_islands(ax, use_cartopy=True)

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'Starlink Coverage — ASEAN Region ({label})', fontsize=9)
    ax.set_extent([95, 130, 0, 25], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FIG13 — SKR TIMESERIES (exact same style as fig13_skr_timeseries_routing.png)
# =============================================================================

def plot_fig13_variant(ts_dry, ts_wet, n_total, label, output_path):
    """Plot SKR timeseries — exact same style as original fig13."""
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.5), sharex=True)

    for ax, ts, title, c_greedy, c_base in [
        (axes[0], ts_dry, 'Dry Season (January, Hanoi)', '#1f77b4', '#d62728'),
        (axes[1], ts_wet, 'Wet Season (July, Hanoi)',    '#1f77b4', '#d62728'),
    ]:
        skr_g_kbps = ts['skr_greedy']   * 1e9 / 1e3
        skr_b_kbps = ts['skr_baseline'] * 1e9 / 1e3
        t = ts['time_hours']

        # Baseline first (behind), then greedy on top
        ax.fill_between(t, skr_b_kbps, alpha=0.15, color=c_base)
        ax.fill_between(t, skr_g_kbps, alpha=0.12, color=c_greedy)
        ax.plot(t, skr_b_kbps, color=c_base,   linewidth=1.0, linestyle='--',
                label='Baseline (no routing)')
        ax.plot(t, skr_g_kbps, color=c_greedy, linewidth=1.2, linestyle='-',
                label='Greedy routing')

        # Annotate improvement
        imp = compute_routing_improvement(ts['skr_greedy'], ts['skr_baseline'])
        ax.text(0.02, 0.92,
                f"Improvement: +{imp['improvement_pct']:.0f}%  |  "
                f"Avail greedy: {imp['availability_greedy']:.0%}  "
                f"vs baseline: {imp['availability_baseline']:.0%}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_ylabel('SKR_effective (kbps)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        y_max = max(skr_g_kbps.max(), skr_b_kbps.max())
        if y_max > 0:
            ax.set_ylim(0, y_max / 0.9)

    axes[1].set_xlabel('Time (hours from T_REF)', fontsize=9)
    plt.suptitle(f'SKR Time Series: Greedy vs Baseline Routing\n'
                 f'({label}, min_elev={MIN_ELEV}°)',
                 fontsize=9, fontweight='bold')
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FIG13 b/c/d + FIG15 b/c/d — CONSTELLATION DENSITY")
    print("=" * 70)
    print()
    print("Giải thích cấu hình Walker-delta:")
    print("  n_planes × sats_per_plane = tổng số vệ tinh")
    print("  - 8×22 = 176 (Starlink shell 1 baseline)")
    print("  - 10×50 = 500 (mở rộng vừa)")
    print("  - 72×22 = 1584 (Starlink Gen1 full)")
    print("  - 40×100 = 4000 (mega-constellation)")
    print()
    print("  Mỗi 'plane' = 1 mặt phẳng quỹ đạo nghiêng 53° so với xích đạo")
    print("  Các plane cách đều: RAAN spacing = 360°/n_planes")
    print("  Trong mỗi plane, vệ tinh cách đều: 360°/sats_per_plane")
    print()

    output_dir = os.path.join(os.path.dirname(__file__), 'diagrams')
    os.makedirs(output_dir, exist_ok=True)

    # Weather params
    wx_dry = get_city_params('hanoi', 1)
    wx_wet = get_city_params('hanoi', 7)
    print(f"Weather — Dry (Jan): R={wx_dry['R_mm_h']:.4f} mm/h, "
          f"V={wx_dry['V_km']} km, P_cloud={wx_dry['P_cloud']}")
    print(f"Weather — Wet (Jul): R={wx_wet['R_mm_h']:.4f} mm/h, "
          f"V={wx_wet['V_km']} km, P_cloud={wx_wet['P_cloud']}")
    print()

    for n_total, (n_planes, sats_per_plane, label) in CONFIGS.items():
        suffix = {500: 'b', 1584: 'c', 4000: 'd'}[n_total]
        print(f"{'='*70}")
        print(f"  ({suffix}) {label}")
        print(f"{'='*70}")

        # Generate constellation
        print(f"  Generating {n_planes} planes × {sats_per_plane} sats...", end=" ")
        satellites = generate_constellation(n_planes, sats_per_plane)
        print(f"{len(satellites)} satellites OK")

        # --- fig15 ---
        fig15_path = os.path.join(output_dir,
                                  f'fig15{suffix}_coverage_map_{n_total}sats.png')
        plot_fig15_variant(satellites, n_total, label, fig15_path)

        # --- fig13 ---
        print(f"  Computing SKR timeseries (dry)...", end=" ", flush=True)
        ts_dry = compute_skr_timeseries(
            satellites=satellites,
            lat_deg=GS['lat'], lon_deg=GS['lon'],
            t_start=T_REF,
            R_mm_h=wx_dry['R_mm_h'], V_km=wx_dry['V_km'],
            P_cloud=wx_dry['P_cloud'],
            duration_hours=DURATION_HOURS, step_minutes=STEP_MINUTES,
            alt_m=GS['alt_m'], min_elevation=MIN_ELEV,
        )
        imp_dry = compute_routing_improvement(ts_dry['skr_greedy'], ts_dry['skr_baseline'])
        print(f"done (+{imp_dry['improvement_pct']:.0f}%)")

        print(f"  Computing SKR timeseries (wet)...", end=" ", flush=True)
        ts_wet = compute_skr_timeseries(
            satellites=satellites,
            lat_deg=GS['lat'], lon_deg=GS['lon'],
            t_start=T_REF,
            R_mm_h=wx_wet['R_mm_h'], V_km=wx_wet['V_km'],
            P_cloud=wx_wet['P_cloud'],
            duration_hours=DURATION_HOURS, step_minutes=STEP_MINUTES,
            alt_m=GS['alt_m'], min_elevation=MIN_ELEV,
        )
        imp_wet = compute_routing_improvement(ts_wet['skr_greedy'], ts_wet['skr_baseline'])
        print(f"done (+{imp_wet['improvement_pct']:.0f}%)")

        fig13_path = os.path.join(output_dir,
                                  f'fig13{suffix}_skr_timeseries_routing_{n_total}sats.png')
        plot_fig13_variant(ts_dry, ts_wet, n_total, label, fig13_path)

        # Print detailed stats
        print()
        print(f"  --- RESULTS for {label} ---")
        print(f"  {'':>12} | {'Dry (Jan)':>12} | {'Wet (Jul)':>12}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}")
        print(f"  {'Improvement':>12} | +{imp_dry['improvement_pct']:>9.0f}% | "
              f"+{imp_wet['improvement_pct']:>9.0f}%")
        print(f"  {'Avail grdy':>12} | {imp_dry['availability_greedy']:>11.0%} | "
              f"{imp_wet['availability_greedy']:>11.0%}")
        print(f"  {'Avail base':>12} | {imp_dry['availability_baseline']:>11.0%} | "
              f"{imp_wet['availability_baseline']:>11.0%}")
        print(f"  {'Key bits G':>12} | {imp_dry['key_bits_greedy']:>10.3e} | "
              f"{imp_wet['key_bits_greedy']:>10.3e}")
        print(f"  {'Key bits B':>12} | {imp_dry['key_bits_baseline']:>10.3e} | "
              f"{imp_wet['key_bits_baseline']:>10.3e}")
        print(f"  {'Gain':>12} | {imp_dry['key_bits_gain']:>10.3e} | "
              f"{imp_wet['key_bits_gain']:>10.3e}")
        print()

    # --- Also update the combined fig15 density comparison ---
    print("=" * 70)
    print("Updating fig15_coverage_density_comparison.png (2×2 grid)...")
    print("=" * 70)
    plot_combined_fig15(output_dir)

    print()
    print("DONE. All figures saved to diagrams/")


def plot_combined_fig15(output_dir):
    """Plot 2×2 combined coverage map for all 4 sizes (176 + 500 + 1584 + 4000)."""
    all_configs = {
        176:  (8, 22,   '(a) 176 sats\n8 planes × 22 sats'),
        500:  (10, 50,  '(b) 500 sats\n10 planes × 50 sats'),
        1584: (72, 22,  '(c) 1584 sats\n72 planes × 22 sats'),
        4000: (40, 100, '(d) 4000 sats\n40 planes × 100 sats'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.5),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes_flat = axes.flatten()

    t_snap = _TS.from_datetime(T_REF)

    for idx, (n_total, (n_planes, spp, title)) in enumerate(all_configs.items()):
        ax = axes_flat[idx]
        print(f"  Panel {idx+1}/4: {n_total} sats...", end=" ", flush=True)

        sats = generate_constellation(n_planes, spp)
        grid = compute_coverage_grid(
            satellites=sats, t=t_snap,
            lat_range=(0, 25), lon_range=(95, 130),
            resolution_deg=1.0, min_elevation=MIN_ELEV,
        )
        print(f"coverage={grid['coverage_frac']:.0%}")

        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='gray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=':', color='gray')
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)

        im = ax.contourf(
            grid['lons'], grid['lats'],
            np.where(grid['has_coverage'], grid['max_elevation'], np.nan),
            levels=np.linspace(10, 90, 17), cmap='YlOrRd', extend='neither',
            transform=ccrs.PlateCarree()
        )

        # Ground stations (smaller for 2×2)
        for city, gs_info in GROUND_STATIONS.items():
            ax.plot(gs_info['lon'], gs_info['lat'], 'b^', markersize=4,
                    transform=ccrs.PlateCarree())

        draw_vietnam_islands(ax, use_cartopy=True)

        ax.set_extent([95, 130, 0, 25], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=7, fontweight='bold')
        gl = ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.4)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Max Elevation (°)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 0.91, 1], pad=0.3)
    out_path = os.path.join(output_dir, 'fig15_coverage_density_comparison.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
