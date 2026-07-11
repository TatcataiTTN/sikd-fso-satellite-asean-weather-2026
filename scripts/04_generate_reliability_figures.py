"""
08_Reliability_Figures.py — Reliability Metrics for GLOBECOM WS-03
===================================================================
Generates fig20–fig24 for the paper:
"Towards Reliable Satellite NTN Communications with SIKD
 under Tropical Weather Dynamics"

Figures:
  fig20 — Link Availability Heatmap (8 cities × 12 months)
  fig21 — Outage Probability Heatmap
  fig22 — Daily Key Delivery Heatmap (Gbits/day)
  fig23 — Seasonal Comparison Bar Chart (wet vs dry SKR_eff)
  fig24 — Information Channel Reliability Heatmap (BER_CC, log10)

Every run also writes a JSON provenance sidecar (run_metadata.json)
next to the PNGs recording: generation timestamp, script/version,
spatial scope (cities), system parameters, and data-source notes —
so the source/time-period of every figure is traceable later
instead of being reconstructed from memory.

Usage:
  cd 05_Code && python 08_Reliability_Figures.py
"""

import sys, os, json, platform, time as time_mod
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from modules.weather_model import (ASEAN_CLIMATE_DATA, get_city_params,
                                   compute_reliability_metrics)
from modules.channel_model import compute_channel, compute_hl
from modules.sikd_performance import compute_sikd_performance
from modules.routing import compute_skr_for_satellite
from utils import save_provenance, save_verify_numbers

_t0 = time_mod.time()

# ================================================================
# IEEE STYLE
# ================================================================
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 9,
    'font.family': 'serif',
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.grid': False,
})

# ================================================================
# CONSTANTS
# ================================================================
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'latex_paper_3', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
# fig21-24 are not currently cited in main.tex (only fig20 -> fig01 is, per
# Task 15 08/07/2026) -- route them to an archive folder instead of the
# paper's figures/ directory, so this script never re-pollutes it.
ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp', 'unused_reliability_figs')
os.makedirs(ARCHIVE_DIR, exist_ok=True)

CITIES = ['hanoi', 'hcmc', 'danang', 'bangkok',
          'singapore', 'manila', 'jakarta', 'kuala_lumpur']
CITY_LABELS = ['Hanoi', 'HCMC', 'Da Nang', 'Bangkok',
               'Singapore', 'Manila', 'Jakarta', 'KL']
MONTHS_SHORT = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
# System parameters (same as routing.py defaults)
PT = 1.0       # W
MK = 0.05
MD = 0.5
ISO_DB = 15.0
ZETA_SCALE = 2.0
RB = 1e9
H_S_KM = 550.0
ZENITH_NOM = 45.0  # nominal zenith for monthly comparison


# ================================================================
# HELPER: Compute SKR_eff for all cities × months
# ================================================================
def compute_skr_matrix():
    """Return (8×12) arrays for SKR_eff (kbps), availability, outage, K_day, BER_CC."""
    n_cities = len(CITIES)
    skr_eff = np.zeros((n_cities, 12))
    avail = np.zeros((n_cities, 12))
    outage = np.zeros((n_cities, 12))
    k_day_gbits = np.zeros((n_cities, 12))
    ber_cc = np.zeros((n_cities, 12))

    for i, city in enumerate(CITIES):
        for m in range(12):
            month = m + 1
            params = get_city_params(city, month)
            result = compute_skr_for_satellite(
                zenith_deg=ZENITH_NOM,
                R_mm_h=params['R_mm_h'],
                V_km=params['V_km'],
                P_cloud=params['P_cloud'],
                PT=PT, mK=MK, mD=MD, Iso_dB=ISO_DB,
                zeta_scale=ZETA_SCALE, Rb=RB, H_S_km=H_S_KM
            )
            skr_eff_kbps = result['SKR_effective'] * RB / 1000.0
            skr_eff[i, m] = skr_eff_kbps
            ber_cc[i, m] = result['BER_CC']

            metrics = compute_reliability_metrics(city, month,
                                                  skr_eff_kbps=skr_eff_kbps)
            avail[i, m] = metrics['A']
            outage[i, m] = metrics['P_out']
            k_day_gbits[i, m] = metrics['K_day'] / 1e9 if metrics['K_day'] else 0.0

    return skr_eff, avail, outage, k_day_gbits, ber_cc


# ================================================================
# FIGURE 20: Availability Heatmap
# ================================================================
def plot_fig20(avail):
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    im = ax.imshow(avail, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=1.0)

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS_SHORT)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITY_LABELS)

    # Annotate cells
    for i in range(len(CITIES)):
        for j in range(12):
            val = avail[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Link Availability $A = 1 - P_{out}$')
    ax.set_title('Link Availability for ASEAN Cities (LEO 550 km)')
    ax.set_xlabel('Month')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig01_availability_heatmap.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# FIGURE 21: Outage Probability Heatmap
# ================================================================
def plot_fig21(outage):
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    im = ax.imshow(outage, aspect='auto', cmap='YlOrRd', vmin=0.0, vmax=0.7)

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS_SHORT)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITY_LABELS)

    for i in range(len(CITIES)):
        for j in range(12):
            val = outage[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Outage Probability $P_{out} = P_{cloud}$')
    ax.set_title('Cloud Outage Probability for ASEAN Cities')
    ax.set_xlabel('Month')

    plt.tight_layout()
    path = os.path.join(ARCHIVE_DIR, 'fig21_outage_heatmap.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# FIGURE 22: Daily Key Delivery Heatmap
# ================================================================
def plot_fig22(k_day_gbits):
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    im = ax.imshow(k_day_gbits, aspect='auto', cmap='viridis',
                   vmin=0, vmax=np.nanmax(k_day_gbits))

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS_SHORT)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITY_LABELS)

    for i in range(len(CITIES)):
        for j in range(12):
            val = k_day_gbits[i, j]
            color = 'white' if val < np.nanmax(k_day_gbits) * 0.5 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Daily Key Delivery $K_{day}$ (Gbits/day)')
    ax.set_title('Daily Secure Key Delivery at $\\zeta = 45°$')
    ax.set_xlabel('Month')

    plt.tight_layout()
    path = os.path.join(ARCHIVE_DIR, 'fig22_daily_key_heatmap.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# FIGURE 23: Seasonal Comparison
# ================================================================
def plot_fig23(skr_eff):
    """Bar chart: drier-half vs wetter-half SKR_eff per city.

    Dry/wet are defined per city as the 6 months with the lowest vs.
    highest monthly rainfall (relative split), rather than a fixed
    mm threshold. An absolute threshold leaves near-equatorial cities
    (e.g. Singapore, rain > 100 mm every month) with an empty "dry"
    group and a missing bar; the relative split always yields 6+6.
    """
    wet_skr = []
    dry_skr = []

    for i, city in enumerate(CITIES):
        rain_mm = np.array([ASEAN_CLIMATE_DATA[city][m][0] for m in range(12)])
        order = np.argsort(rain_mm)
        dry_months, wet_months = order[:6], order[6:]
        dry_skr.append(np.mean(skr_eff[i, dry_months]))
        wet_skr.append(np.mean(skr_eff[i, wet_months]))

    x = np.arange(len(CITIES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.16, 4.0), dpi=150)
    ax.bar(x - width/2, dry_skr, width, label='Drier 6 months',
           color='#2196F3', alpha=0.85)
    ax.bar(x + width/2, wet_skr, width, label='Wetter 6 months',
           color='#FF5722', alpha=0.85)

    ax.set_xlabel('City')
    ax.set_ylabel('Mean $SKR_{eff}$ (kbps)')
    ax.set_title('Seasonal SKR Comparison: Drier vs Wetter Half-Year')
    ax.set_xticks(x)
    ax.set_xticklabels(CITY_LABELS, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Reserve headroom above the tallest bar so ratio labels never
    # collide with the title, regardless of how tall a city's bar is.
    ymax = max(max(dry_skr), max(wet_skr))
    ax.set_ylim(0, ymax * 1.30)

    for i in range(len(CITIES)):
        ratio = dry_skr[i] / wet_skr[i] if wet_skr[i] > 0 else float('nan')
        ypos = max(dry_skr[i], wet_skr[i]) + ymax * 0.04
        ax.text(x[i], ypos, f'{ratio:.1f}×', ha='center', fontsize=7, color='#333')

    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(ARCHIVE_DIR, 'fig23_seasonal_comparison.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# FIGURE 24: Information Channel Reliability (BER_CC)
# ================================================================
BER_CC_THRESHOLD = 1e-6  # classical-channel "reliable" cutoff (see test_sikd_performance.py)


def plot_fig24(ber_cc):
    """Heatmap of log10(BER_CC) — reliability of the classical data channel,
    computed under the same per-city, per-month weather states as fig20-23
    (clear-state performance, see compute_skr_for_satellite). Cells show the
    actual BER_CC value (consistent with the fig20-22 style of annotating
    real numbers rather than a qualitative pass/fail label)."""
    log_ber = np.log10(np.clip(ber_cc, 1e-20, 1.0))

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    im = ax.imshow(log_ber, aspect='auto', cmap='YlGnBu_r',
                   vmin=-20, vmax=-2)

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS_SHORT)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITY_LABELS)

    for i in range(len(CITIES)):
        for j in range(12):
            val = ber_cc[i, j]
            color = 'white' if val > 1e-4 else 'black'
            ax.text(j, i, f'{val:.0e}', ha='center', va='center',
                    fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'$\log_{10}(\mathrm{BER}_{CC})$')
    ax.set_title(f'Information Channel Reliability '
                 f'(threshold: BER$_{{CC}}$ $<$ {BER_CC_THRESHOLD:.0e})')
    ax.set_xlabel('Month')

    plt.tight_layout()
    path = os.path.join(ARCHIVE_DIR, 'fig24_ber_cc_reliability.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# PROVENANCE / RUN METADATA
# ================================================================
def save_run_metadata(skr_eff, avail, outage, k_day_gbits, ber_cc, figures):
    """Write a JSON sidecar recording when/how these figures were generated,
    so the source and time-period behind every number is traceable later
    instead of being reconstructed from memory or guessed in the paper."""
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
        "script": os.path.basename(__file__),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "spatial_scope": {
            "cities": CITIES,
            "n_cities": len(CITIES),
            "region": "ASEAN (8 cities, see weather_model.ASEAN_CLIMATE_DATA)",
        },
        "temporal_scope": {
            "months_modeled": MONTHS_SHORT,
            "note": ("Climatological (long-term average) monthly profile, NOT tied to a "
                     "specific calendar year. Rain-attenuation regression (alpha=0.509, "
                     "rho=0.63) is from ground measurements in Abidjan, Cote d'Ivoire, "
                     "Jan 2018-Dec 2022 (Kone et al. 2024, doi:10.12691/ijp-12-6-1). "
                     "Per-city P_cloud/V/R values are climatological estimates under a "
                     "tropical-climate analogy, not direct ASEAN measurements.")
        },
        "system_parameters": {
            "PT_W": PT, "mK": MK, "mD": MD, "Iso_dB": ISO_DB,
            "zeta_scale": ZETA_SCALE, "Rb_bps": RB, "H_S_km": H_S_KM,
            "zenith_nominal_deg": ZENITH_NOM,
            "ber_cc_threshold": BER_CC_THRESHOLD,
        },
        "result_ranges": {
            "skr_eff_kbps": [float(np.min(skr_eff)), float(np.max(skr_eff))],
            "availability": [float(np.min(avail)), float(np.max(avail))],
            "outage_probability": [float(np.min(outage)), float(np.max(outage))],
            "k_day_gbits": [float(np.min(k_day_gbits)), float(np.max(k_day_gbits))],
            "ber_cc": [float(np.min(ber_cc)), float(np.max(ber_cc))],
        },
        "figures_generated": figures,
    }
    path = os.path.join(ARCHIVE_DIR, 'fig20-24_run_metadata.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {path}")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("08_Reliability_Figures — GLOBECOM WS-03")
    print("=" * 60)

    print("\nComputing SKR/BER_CC matrix for 8 cities × 12 months...")
    skr_eff, avail, outage, k_day_gbits, ber_cc = compute_skr_matrix()

    print(f"\nAvailability range: {avail.min():.2f} – {avail.max():.2f}")
    print(f"Outage range: {outage.min():.2f} – {outage.max():.2f}")
    print(f"Daily key range: {k_day_gbits.min():.0f} – {k_day_gbits.max():.0f} Gbits/day")
    print(f"BER_CC range: {ber_cc.min():.2e} – {ber_cc.max():.2e}")

    print("\n--- Fig 20: Availability Heatmap ---")
    plot_fig20(avail)

    print("--- Fig 21: Outage Probability Heatmap ---")
    plot_fig21(outage)

    print("--- Fig 22: Daily Key Delivery ---")
    plot_fig22(k_day_gbits)

    print("--- Fig 23: Seasonal Comparison ---")
    plot_fig23(skr_eff)

    print("--- Fig 24: Information Channel Reliability (BER_CC) ---")
    plot_fig24(ber_cc)

    figures = ['fig01_availability_heatmap.png']
    archived_figures = ['fig21_outage_heatmap.png', 'fig22_daily_key_heatmap.png',
               'fig23_seasonal_comparison.png', 'fig24_ber_cc_reliability.png']
    save_run_metadata(skr_eff, avail, outage, k_day_gbits, ber_cc, figures)

    # Only fig20 (renamed fig01_availability_heatmap.png at the paper's
    # figures/ folder, Task 15) is actually used in the paper; fig21-24
    # are archived (05_Code_v2/temp/unused_reliability_figs/) -- kept here
    # for completeness/history, not currently cited in main.tex.
    verify = {
        "availability_min": f"{avail.min():.3f}",
        "availability_max": f"{avail.max():.3f}",
        "outage_min": f"{outage.min():.3f}",
        "outage_max": f"{outage.max():.3f}",
        "k_day_gbits_min": f"{k_day_gbits.min():.1f}",
        "k_day_gbits_max": f"{k_day_gbits.max():.1f}",
    }
    save_verify_numbers(verify, "reliability_figures")
    save_provenance(
        script_name="04_generate_reliability_figures",
        params={"cities": CITIES, "zenith_nominal_deg": ZENITH_NOM,
                "PT": PT, "mK": MK, "mD": MD, "Iso_dB": ISO_DB,
                "H_S_km": H_S_KM,
                "note": "F1 (fig01_availability_heatmap.png) is the only "
                        "output of this script currently used in the paper "
                        "(Task 15, 08/07/2026); fig21-24 archived"},
        key_numbers=verify,
        runtime_secs=time_mod.time() - _t0,
        output_files=[os.path.join(OUT_DIR, f) for f in figures] +
                     [os.path.join(ARCHIVE_DIR, f) for f in archived_figures],
        data_sources={"weather climatology": "modules/weather_model.ASEAN_CLIMATE_DATA "
                                             "(real ERA5/Open-Meteo monthly climatology)"},
    )

    print("\nDone! All figures + provenance metadata saved to:", OUT_DIR)
