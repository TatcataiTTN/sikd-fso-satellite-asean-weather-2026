"""
Generate Paper 2 inspired figures for LaTeX report.
Figures A–D saved to latex_report_demo/figures/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05_Code'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from modules.channel_model import compute_channel, compute_hl
from modules.sikd_performance import compute_sikd_performance

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
    'font.family': 'DejaVu Sans',
})

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '05_Code', 'diagrams')
os.makedirs(OUT_DIR, exist_ok=True)

H_S_KM = 500.0
ZETA_DEG = 60.0   # Paper 2 uses ζS = 60° for Figs 3–5
PT_NOM_W = 10 ** (30 / 10) * 1e-3  # 30 dBm = 1 W

ch60 = compute_channel(H_S_KM, ZETA_DEG)
hl_clear = compute_hl(zeta_deg=ZETA_DEG, V_km=15.0, R_mm_h=0.0)

print(f"Channel at ζ={ZETA_DEG}°: hg={ch60['hg']:.3e}, hl={hl_clear:.4f}, σX²={ch60['sigma_X2']:.4f}")

# ============================================================
# Figure A: BER_CC vs Transmit Power PT (dBm) — different mD
# Inspired by Paper 2 Fig 2
# ============================================================
print("\n--- Figure A: BER_CC vs PT ---")

PT_dBm = np.linspace(0, 40, 300)
PT_W   = 10 ** (PT_dBm / 10) * 1e-3

mD_vals   = [0.3, 0.5, 0.7, 0.9]
colors_A  = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
styles_A  = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(3.5, 3.0))

for mD, col, ls in zip(mD_vals, colors_A, styles_A):
    ber_vals = []
    for PT in PT_W:
        p = compute_sikd_performance(
            hg=ch60['hg'], hl=hl_clear, sigma_X2=ch60['sigma_X2'],
            PT=PT, mK=0.05, mD=mD, Iso_dB=15.0, zeta_scale=2.0, Rb=1e9
        )
        ber_vals.append(p['BER_CC'])
    ber_arr = np.array(ber_vals)
    mask = ber_arr > 0
    ax.semilogy(PT_dBm[mask], ber_arr[mask], ls, color=col, linewidth=1.8,
                label=f'$m_D = {mD}$')

ax.axhline(1e-9, color='k', linestyle='--', linewidth=1.2, alpha=0.7,
           label='BER target $10^{-9}$')

ax.set_xlabel('Transmit Power $P_T$ (dBm)', fontsize=11)
ax.set_ylabel('BER (Classical Channel)', fontsize=11)
ax.set_title('BER$_{CC}$ vs Transmit Power\n'
             r'($\zeta_S=60°$, $m_K=0.05$, $I_{so}=15$ dB, $H_S=500$ km)',
             fontsize=10)
ax.legend(fontsize=9)

# Annotation: physical interpretation
ax.text(0.98, 0.55, 'Higher $m_D$ → stronger\ndata signal → lower BER',
        transform=ax.transAxes, fontsize=7, ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
ax.set_xlim(0, 40)
ax.set_ylim(1e-15, 1)

# Secondary x-axis in Watts
ax2 = ax.twiny()
ax2.set_xlim(0, 40)
ticks_dBm = [0, 10, 20, 30, 40]
ax2.set_xticks(ticks_dBm)
ax2.set_xticklabels([f'{10**(v/10)*1e-3:.3g} W' for v in ticks_dBm], fontsize=8)
ax2.set_xlabel('Transmit Power (W)', fontsize=9)

plt.tight_layout(pad=0.5)
path_A = os.path.join(OUT_DIR, 'figA_ber_cc_vs_pt.png')
plt.savefig(path_A, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {path_A}")

# ============================================================
# Figure B: Psift and QBER vs mK — different Iso values
# Inspired by Paper 2 Fig 3
# ============================================================
print("\n--- Figure B: Psift/QBER vs mK ---")

mK_range = np.linspace(0.001, 0.30, 300)
Iso_vals  = [10.0, 15.0, 20.0, 25.0]
colors_B  = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
styles_B  = [':', '-.', '--', '-']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

for Iso, col, ls in zip(Iso_vals, colors_B, styles_B):
    psift_vals, qber_vals = [], []
    for mK in mK_range:
        p = compute_sikd_performance(
            hg=ch60['hg'], hl=hl_clear, sigma_X2=ch60['sigma_X2'],
            PT=PT_NOM_W, mK=mK, mD=0.5, Iso_dB=Iso, zeta_scale=2.0, Rb=1e9
        )
        psift_vals.append(p['Psift'])
        qber_vals.append(p['QBER'])
    ax1.semilogy(mK_range, psift_vals, ls, color=col, linewidth=1.8,
                 label=f'$I_{{so}}={Iso:.0f}$ dB')
    ax2.plot(mK_range, np.array(qber_vals) * 100, ls, color=col, linewidth=1.8,
             label=f'$I_{{so}}={Iso:.0f}$ dB')

ax1.set_xlabel('QKD Modulation Index $m_K$', fontsize=11)
ax1.set_ylabel('Sifted Key Probability $P_{sift}$', fontsize=11)
ax1.set_title('$P_{sift}$ vs $m_K$\n'
              r'($P_T=30$ dBm, $m_D=0.5$, $\zeta_S=60°$)', fontsize=10)
ax1.legend(fontsize=9)
ax1.set_xlim(0, 0.30)
ax1.text(0.95, 0.05, 'Higher $m_K$ → more\nQKD photons sifted',
         transform=ax1.transAxes, fontsize=6.5, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax2.axhline(11.0, color='k', linestyle='--', linewidth=1.2, alpha=0.7,
            label='QBER limit 11%')
ax2.set_xlabel('QKD Modulation Index $m_K$', fontsize=11)
ax2.set_ylabel('QBER (%)', fontsize=11)
ax2.set_title('QBER vs $m_K$\n'
              r'($P_T=30$ dBm, $m_D=0.5$, $\zeta_S=60°$)', fontsize=10)
ax2.legend(fontsize=9)
ax2.set_xlim(0, 0.30)
ax2.set_ylim(0, 55)
ax2.text(0.95, 0.85, 'QBER > 11% → no\nsecure key possible',
         transform=ax2.transAxes, fontsize=6.5, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8))

plt.tight_layout(pad=0.5)
path_B = os.path.join(OUT_DIR, 'figB_psift_qber_vs_mk.png')
plt.savefig(path_B, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {path_B}")

# ============================================================
# Figure C: Minimum mK required vs Filter Isolation Iso
# Inspired by Paper 2 Fig 4
# ============================================================
print("\n--- Figure C: min mK vs Iso ---")

Iso_range = np.linspace(5, 35, 200)
mD_vals_C = [0.3, 0.5, 0.7, 0.9]
colors_C  = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
styles_C  = ['-', '--', '-.', ':']

QBER_THRESHOLD = 0.11  # 11%

fig, ax = plt.subplots(figsize=(3.5, 3.0))

for mD, col, ls in zip(mD_vals_C, colors_C, styles_C):
    min_mK_vals = []
    for Iso in Iso_range:
        # Binary search for minimum mK such that QBER < threshold
        lo, hi = 0.001, 0.30
        found = None
        for _ in range(40):
            mid = (lo + hi) / 2
            p = compute_sikd_performance(
                hg=ch60['hg'], hl=hl_clear, sigma_X2=ch60['sigma_X2'],
                PT=PT_NOM_W, mK=mid, mD=mD, Iso_dB=Iso, zeta_scale=2.0, Rb=1e9
            )
            if p['QBER'] < QBER_THRESHOLD:
                found = mid
                hi = mid
            else:
                lo = mid
        min_mK_vals.append(found if found is not None else np.nan)
    ax.plot(Iso_range, min_mK_vals, ls, color=col, linewidth=1.8,
            label=f'$m_D = {mD}$')

ax.set_xlabel('BPF Isolation $I_{so}$ (dB)', fontsize=11)
ax.set_ylabel('Minimum $m_K$ for QBER $< 11\\%$', fontsize=11)
ax.set_title('Minimum QKD Modulation Index vs Filter Isolation\n'
             r'($P_T=30$ dBm, $\zeta_S=60°$, $H_S=500$ km)', fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(5, 35)
ax.set_ylim(0, 0.30)
ax.text(0.95, 0.95, 'Better filter → lower\n$m_K$ needed → less\nimpact on data channel',
        transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(pad=0.5)
path_C = os.path.join(OUT_DIR, 'figC_min_mk_vs_iso.png')
plt.savefig(path_C, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {path_C}")

# ============================================================
# Figure D: SKR heatmap as function of mD × mK
# Inspired by Paper 2 Fig 5
# ============================================================
print("\n--- Figure D: SKR heatmap mD x mK ---")

mK_grid = np.linspace(0.01, 0.25, 60)
mD_grid = np.linspace(0.1, 0.9, 60)
SKR_map = np.zeros((len(mD_grid), len(mK_grid)))

for i, mD in enumerate(mD_grid):
    for j, mK in enumerate(mK_grid):
        p = compute_sikd_performance(
            hg=ch60['hg'], hl=hl_clear, sigma_X2=ch60['sigma_X2'],
            PT=PT_NOM_W, mK=mK, mD=mD, Iso_dB=10.0, zeta_scale=2.0, Rb=1e9
        )
        SKR_map[i, j] = p['SKR_kbps']

SKR_log = np.log10(np.where(SKR_map > 0, SKR_map, np.nan))

fig, ax = plt.subplots(figsize=(7.16, 4.5))
im = ax.contourf(mK_grid, mD_grid, SKR_log, levels=20, cmap='viridis')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('$\\log_{10}$(SKR) [kbps]', fontsize=10)

# Contour line at SKR = 1 kbps (log10 = 0)
cs = ax.contour(mK_grid, mD_grid, SKR_log, levels=[0], colors='white',
                linewidths=1.5, linestyles='--')
ax.clabel(cs, fmt='SKR=1 kbps', fontsize=8, colors='white')

# Mark nominal operating point
ax.plot(0.05, 0.5, 'r*', markersize=12, label='Nominal ($m_K=0.05$, $m_D=0.5$)')
ax.legend(fontsize=9, loc='upper right')

ax.set_xlabel('QKD Modulation Index $m_K$', fontsize=11)
ax.set_ylabel('Data Modulation Index $m_D$', fontsize=11)
ax.set_title('SKR Heatmap: $m_D \\times m_K$\n'
             r'($P_T=30$ dBm, $I_{so}=10$ dB, $\zeta_S=60°$, $H_S=500$ km)',
             fontsize=10)
ax.text(0.02, 0.02, 'Trade-off: increasing $m_K$ boosts SKR\nbut degrades classical BER',
        transform=ax.transAxes, fontsize=6.5, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout(pad=0.5)
path_D = os.path.join(OUT_DIR, 'figD_skr_heatmap_md_mk.png')
plt.savefig(path_D, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {path_D}")

print("\nAll figures generated successfully.")

