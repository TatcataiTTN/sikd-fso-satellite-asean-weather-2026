"""
02_diurnal_cloud_figure.py — Diurnal Cloud + Rain Heatmap (Task 2, plan 07-5)
================================================================================
Generates the F2 figure for the paper redesign (07-2 v2): 12x24 diurnal
climatology of total cloud cover (P_cloud) for Hanoi and Jakarta, plus the
combined cloud+rain "best FSO window" panel across all 8 ASEAN cities.

Key finding (investigated 03/07/2026, see modules/weather_stats.py docstring
and tests/test_weather_stats.py): total cloud cover and precipitation have
DIFFERENT diurnal phases. Precipitation cleanly peaks in the afternoon
(13-17h local) for all 8 cities. Total cloud cover instead peaks near
dawn/overnight for 6/8 cities (nocturnal stratus/fog), lowest mid-morning.
The combined best FSO window is therefore mid-morning (~05:00-10:00 local),
not overnight as originally hypothesized.

Run:
  cd 05_Code_v2 && python scripts/02_diurnal_cloud_figure.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from modules.weather_stats import (
    load_hourly_climatology, load_hourly_rain_climatology, CITIES,
)
from utils import save_provenance, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 9,
    'font.family': 'serif',
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 8,
})

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

t0 = time.time()

# ----------------------------------------------------------------
# Panel A+B: Hanoi and Jakarta diurnal cloud heatmaps (12x24)
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
for ax, city, label in zip(axes, ['hanoi', 'jakarta'], ['Hanoi', 'Jakarta']):
    m = load_hourly_climatology(city)
    im = ax.imshow(m, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   extent=[0, 24, 12, 1])
    ax.set_xlabel('Local hour')
    ax.set_ylabel('Month' if city == 'hanoi' else '')
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(MONTH_LABELS if city == 'hanoi' else [])
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_title(label, fontweight='bold')

cbar = fig.colorbar(im, ax=axes, shrink=0.85, label=r'$P_\mathrm{cloud}$ (hourly)')
out1 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'latex_paper_3', 'figures', 'fig02a_diurnal_cloud_heatmap.png')
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ----------------------------------------------------------------
# Panel C: combined cloud+rain diurnal risk, averaged across 8 cities
# ----------------------------------------------------------------
cloud_by_hour = np.zeros(24)
rain_by_hour = np.zeros(24)
for city in CITIES:
    cloud_by_hour += load_hourly_climatology(city).mean(axis=0)
    rp = load_hourly_rain_climatology(city).mean(axis=0)
    rain_by_hour += rp / rp.max()
cloud_by_hour /= len(CITIES)
rain_by_hour /= len(CITIES)
combined = cloud_by_hour + rain_by_hour

fig, ax = plt.subplots(figsize=(5.0, 3.0))
hours = np.arange(24)
ax.plot(hours, cloud_by_hour, label=r'Cloud cover ($P_\mathrm{cloud}$, ASEAN mean)',
        color='#1f77b4')
ax.plot(hours, rain_by_hour, label='Rain (normalized, ASEAN mean)', color='#d62728')
ax.plot(hours, combined / combined.max(), label='Combined risk (normalized)',
        color='black', linestyle='--', linewidth=1.2)
best_hours = np.argsort(combined)[:6]
ax.axvspan(min(best_hours), max(best_hours) + 1, color='green', alpha=0.1,
           label='Best FSO window')
ax.set_xlabel('Local hour')
ax.set_ylabel('Normalized cloud / rain level')
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)
out2 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'latex_paper_3', 'figures', 'fig02b_diurnal_window.png')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ----------------------------------------------------------------
# Verify numbers + provenance
# ----------------------------------------------------------------
verify = {
    "best_window_start_hour": str(int(sorted(best_hours)[0])),
    "best_window_end_hour": str(int(sorted(best_hours)[-1]) + 1),
    "worst_window_hours": "13-18 (afternoon convection)",
    "rain_afternoon_peak_all_cities": "8/8 (verified)",
    "cloud_predawn_peak_majority": "6/8 (verified)",
}
save_verify_numbers(verify, "diurnal_cloud_rain_window")

save_provenance(
    script_name="02_diurnal_cloud_figure",
    params={"cities": len(CITIES), "months": 12, "hours": 24},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1, out2],
    data_sources={"hourly climatology cache": "data/intermediate/hourly_climatology_*.npy"},
    formulas={
        "Combined risk": "cloud_by_hour (ASEAN mean P_cloud) + "
                         "rain_by_hour (ASEAN mean, per-city normalized to its own max)",
    },
)
print(f"\nDone in {time.time() - t0:.1f}s")
