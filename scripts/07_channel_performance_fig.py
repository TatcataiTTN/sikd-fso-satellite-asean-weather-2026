"""
07_channel_performance_fig.py — SIKD Channel Performance vs Elevation Angle
============================================================================
Generates fig07_channel_performance.png: two-panel figure showing:
  - Panel 1 (top): SKR (kbps) vs elevation angle
  - Panel 2 (bottom): QBER (%) vs elevation angle, with BB84 11% threshold

Clear-sky conditions: V = 10 km, R = 0, theta_C = 10 urad, H_S = 550 km.

Run:
  cd 05_Code_v2
  python scripts/07_channel_performance_fig.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from channel_model import compute_channel
from sikd_performance import compute_sikd_performance
from utils import save_provenance, save_verify_numbers

# ----------------------------------------------------------------
# IEEE figure style
# ----------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi':       300,
    'font.size':        9,
    'font.family':      'serif',
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'lines.linewidth':  1.5,
})

# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------
THETA_C_URAD = 10.0
H_S_KM       = 550.0
V_KM_CLEAR   = 10.0
R_MM_H       = 0.0
BB84_THRESH  = 11.0   # % — Shor-Preskill security threshold

ZENITH_MIN = 0
ZENITH_MAX = 60      # elevation cutoff = 30°
N_POINTS   = 61      # 0° to 60° in 1° steps

zeniths    = np.linspace(ZENITH_MIN, ZENITH_MAX, N_POINTS)
elevations = 90.0 - zeniths

# ----------------------------------------------------------------
# Compute performance across zenith range
# ----------------------------------------------------------------
t0 = time.time()

skr_vals  = []
qber_vals = []
hg_dB     = []
hl_dB     = []
sigmaR2   = []

for z in zeniths:
    ch   = compute_channel(H_S_KM, float(z), V_km=V_KM_CLEAR,
                           R_mm_h=R_MM_H, theta_C_urad=THETA_C_URAD)
    perf = compute_sikd_performance(ch['hg'], ch['hl'], ch['sigma_X2'])
    skr_vals.append(perf['SKR_kbps'])
    qber_vals.append(perf['QBER'] * 100.0)
    hg_dB.append(ch['hg_dB'])
    hl_dB.append(ch['hl_dB'])
    sigmaR2.append(ch['sigma_R2'])

skr_vals  = np.array(skr_vals)
qber_vals = np.array(qber_vals)

# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.8), sharex=True)
fig.subplots_adjust(hspace=0.10)

# --- Panel 1: SKR ---
ax1.plot(elevations, skr_vals / 1e3, color='#1f77b4', linewidth=1.5)
ax1.set_ylabel(r'SKR (Mbps)')
ax1.set_ylim(8, 16)
ax1.set_yticks([8, 10, 12, 14, 16])
ax1.grid(True, alpha=0.3, linewidth=0.5)
ax1.set_title(r'Clear sky: $V = 10$ km, $R = 0$, $\theta_C = 10\ \mu$rad', fontsize=8)

# Mark reference points
for elev_mark, label_y in [(90, 0.92), (45, 0.92)]:
    z = 90 - elev_mark
    idx = int(np.round(z))
    ax1.plot(elev_mark, skr_vals[idx] / 1e3, 'o', color='#1f77b4',
             markersize=4, zorder=5)

# --- Panel 2: QBER ---
ax2.plot(elevations, qber_vals, color='#d62728', linewidth=1.5)

# BB84 threshold
ax2.axhline(BB84_THRESH, color='black', linestyle='--', linewidth=1.0,
            label=r'BB84 threshold (11%)')

# Shade insecure region
qber_clipped = np.maximum(qber_vals, BB84_THRESH)
ax2.fill_between(elevations, BB84_THRESH, qber_clipped,
                 alpha=0.15, color='#d62728', label='Insecure region')

ax2.set_ylabel(r'QBER (%)')
ax2.set_xlabel(r'Elevation angle (deg)')
ax2.set_ylim(9.5, 20)
ax2.set_yticks([10, 12, 14, 16, 18, 20])
ax2.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
ax2.grid(True, alpha=0.3, linewidth=0.5)

# Vertical guideline: elevation = 40° (safe cutoff with fixed thresholds)
for ax in (ax1, ax2):
    ax.axvline(40.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.set_xlim(30, 90)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

# Annotation for the 40° line
ax2.annotate(r'$40^\circ$', xy=(40, 9.7), fontsize=7, color='gray',
             ha='center')

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)
OUT = os.path.abspath(os.path.join(
    SCRIPT_DIR, '..', '..', 'latex_paper_3', 'figures',
    'fig06_channel_performance.png'))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close()
print(f"[fig06] Saved to: {OUT}")

runtime = time.time() - t0

# ----------------------------------------------------------------
# Console summary (verify against paper Table)
# ----------------------------------------------------------------
print("\nVerification table (matches paper Table II):")
print(f"{'Zenith':>7} {'Elev':>5} {'hg_dB':>8} {'hl_dB':>8} "
      f"{'QBER%':>8} {'SKR_kbps':>10}")
print("-" * 57)
verify = {}
for z_check in [0, 30, 45, 50, 55, 60]:
    idx = int(z_check)
    print(f"{z_check:>6}°  {90-z_check:>4}°  {hg_dB[idx]:>8.1f}  "
          f"{hl_dB[idx]:>8.2f}  {qber_vals[idx]:>7.2f}%  "
          f"{skr_vals[idx]:>10.1f}")
    verify[f"zenith{z_check}_hg_dB"] = f"{hg_dB[idx]:.1f}"
    verify[f"zenith{z_check}_hl_dB"] = f"{hl_dB[idx]:.2f}"
    verify[f"zenith{z_check}_QBER_pct"] = f"{qber_vals[idx]:.2f}"
    verify[f"zenith{z_check}_SKR_kbps"] = f"{skr_vals[idx]:.1f}"
save_verify_numbers(verify, "channel_performance")

save_provenance(
    script_name="07_channel_performance_fig",
    params={"H_S_km": H_S_KM, "V_km_clear": V_KM_CLEAR, "R_mm_h": R_MM_H,
            "theta_C_urad": THETA_C_URAD, "bb84_threshold_pct": BB84_THRESH,
            "zenith_range_deg": [ZENITH_MIN, ZENITH_MAX], "n_points": N_POINTS},
    key_numbers=verify,
    runtime_secs=runtime,
    output_files=[OUT],
    formulas={"channel_chain": "compute_channel -> compute_sikd_performance, "
                               "clear sky (modules/channel_model.py, modules/sikd_performance.py)"},
)
print(f"\nRuntime: {runtime:.1f}s")
