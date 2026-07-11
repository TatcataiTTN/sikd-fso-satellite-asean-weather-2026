"""
10_intercity_correlation.py — Inter-City Cloud Correlation & Joint Availability
==================================================================================
Task 3, plan 07-5. Uses modules/weather_stats.py (Task 2) to:
  1. Compute 8x8 Pearson correlation matrices of daily cloud cover for the
     dry season (Nov-Mar) and wet season (Jun-Sep), and plot both as heatmaps.
  2. Compute joint clear-day availability for cumulative station combinations
     in the same order as Dang et al. (2023) Fig. 7 (Hanoi -> +Da Nang ->
     +Ho Chi Minh City), compared directly against their reported annual
     81.92% / 96.86% / 99.44% progression.
  3. Save intermediate CSVs + verify numbers (utils.py) + provenance.

This is the quantitative backbone behind the "Jakarta breaks monsoon
correlation" claim used elsewhere in the paper: r(Hanoi, Da Nang | Jul) vs
r(Hanoi, Jakarta | Jul) are computed directly here rather than asserted.

Run:
  cd 05_Code_v2 && python scripts/10_intercity_correlation.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from modules.weather_stats import CITIES, correlation_matrix, joint_clear_probability
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 9,
    'font.family': 'serif',
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

t0 = time.time()

CITY_LABELS = {
    'hanoi': 'Hanoi', 'danang': 'Da Nang', 'hcmc': 'HCMC', 'bangkok': 'Bangkok',
    'singapore': 'Singapore', 'manila': 'Manila', 'jakarta': 'Jakarta',
    'kuala_lumpur': 'Kuala Lumpur',
}
LABELS = [CITY_LABELS[c] for c in CITIES]

DRY_MONTHS = [11, 12, 1, 2, 3]
WET_MONTHS = [6, 7, 8, 9]

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                        'latex_paper_3', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------
# 1. Correlation matrices: dry vs wet
# ----------------------------------------------------------------
M_dry = correlation_matrix(DRY_MONTHS)
M_wet = correlation_matrix(WET_MONTHS)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, M, title in zip(axes, [M_dry, M_wet],
                        ['Dry season (Nov-Mar)', 'Wet season (Jun-Sep)']):
    im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(8))
    ax.set_xticklabels(LABELS, rotation=90)
    ax.set_yticks(range(8))
    ax.set_yticklabels(LABELS)
    ax.set_title(title, fontweight='bold')
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=5.5, color='white' if abs(M[i, j]) > 0.5 else 'black')
# The right panel's y-tick labels sit immediately left of its axis; with
# the default spacing they land on top of the left panel's last column
# of cell-value text. A wider wspace keeps the two panels from colliding.
fig.subplots_adjust(wspace=0.55)
fig.colorbar(im, ax=axes, shrink=0.8, label='Pearson $r$ (daily cloud cover)')

out1 = os.path.join(FIG_DIR, 'fig03a_correlation_matrix.png')
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

i_hanoi = CITIES.index('hanoi')
i_danang = CITIES.index('danang')
i_jakarta = CITIES.index('jakarta')
M_jul = correlation_matrix([7])
r_hanoi_danang_jul = M_jul[i_hanoi, i_danang]
r_hanoi_jakarta_jul = M_jul[i_hanoi, i_jakarta]

print("\nJuly correlation check (monsoon vs cross-equatorial):")
print(f"  r(Hanoi, Da Nang | Jul)  = {r_hanoi_danang_jul:+.3f}  (same monsoon regime)")
print(f"  r(Hanoi, Jakarta | Jul)  = {r_hanoi_jakarta_jul:+.3f}  (cross-equatorial)")

# ----------------------------------------------------------------
# 2. Joint availability progression vs Dang 2023 Fig. 7
# ----------------------------------------------------------------
DANG_ORDER = ['hanoi', 'danang', 'hcmc']
DANG_REPORTED = [81.92, 96.86, 99.44]  # %, Nguyen/Le/Pham/Dang (2023) Fig. 7

rows = []
for k in range(1, len(DANG_ORDER) + 1):
    combo = DANG_ORDER[:k]
    ours = joint_clear_probability(combo, months=list(range(1, 13))) * 100.0
    rows.append({
        "n_sites": k,
        "sites": "+".join(combo),
        "ours_pct": round(ours, 2),
        "dang2023_pct": DANG_REPORTED[k - 1],
        "delta_pct": round(ours - DANG_REPORTED[k - 1], 2),
    })

print("\nJoint availability (annual, all 12 months) vs Dang et al. (2023) Fig. 7:")
for r in rows:
    print(f"  n={r['n_sites']}  {r['sites']:22s} ours={r['ours_pct']:6.2f}%  "
          f"Dang2023={r['dang2023_pct']:6.2f}%  delta={r['delta_pct']:+6.2f}pp")

save_intermediate_csv(
    rows, "joint_availability_vs_dang2023",
    "Joint clear-day availability (annual), cumulative Hanoi->+Danang->+HCMC, "
    "vs Dang et al. (2023) Fig. 7 reported values",
)

# Bar chart: ours vs Dang 2023, grouped by n_sites
fig, ax = plt.subplots(figsize=(4.2, 3.0))
x = np.arange(len(rows))
w = 0.35
ax.bar(x - w / 2, [r["ours_pct"] for r in rows], width=w,
       label='This work (binary 85% cloud threshold)', color='#1f77b4')
ax.bar(x + w / 2, [r["dang2023_pct"] for r in rows], width=w,
       label='Dang et al. (2023) (Gamma-CLWC, 30 dB budget)', color='#ff7f0e')
ax.set_xticks(x)
ax.set_xticklabels([r["sites"] for r in rows], fontsize=7)
ax.set_ylabel('Joint clear-day availability (%)')
ax.set_ylim(0, 105)
ax.legend(fontsize=6.5, loc='lower right')
ax.grid(alpha=0.3, axis='y')
for i, r in enumerate(rows):
    ax.text(i - w / 2, r["ours_pct"] + 1, f'{r["ours_pct"]:.1f}', ha='center', fontsize=6)
    ax.text(i + w / 2, r["dang2023_pct"] + 1, f'{r["dang2023_pct"]:.1f}', ha='center', fontsize=6)

out2 = os.path.join(FIG_DIR, 'fig03b_joint_availability_dang2023.png')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ----------------------------------------------------------------
# 3. Verify + provenance
# ----------------------------------------------------------------
verify = {
    "r_hanoi_danang_jul": f"{r_hanoi_danang_jul:.3f}",
    "r_hanoi_jakarta_jul": f"{r_hanoi_jakarta_jul:.3f}",
    "joint_1site_hanoi_pct": f"{rows[0]['ours_pct']:.2f}",
    "joint_2site_hanoi_danang_pct": f"{rows[1]['ours_pct']:.2f}",
    "joint_3site_hanoi_danang_hcmc_pct": f"{rows[2]['ours_pct']:.2f}",
    "dang2023_1site_pct": "81.92",
    "dang2023_2site_pct": "96.86",
    "dang2023_3site_pct": "99.44",
}
save_verify_numbers(verify, "intercity_correlation_joint_availability")

save_provenance(
    script_name="10_intercity_correlation",
    params={
        "dry_months": DRY_MONTHS,
        "wet_months": WET_MONTHS,
        "n_cities": len(CITIES),
        "dang_comparison_order": DANG_ORDER,
    },
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1, out2],
    data_sources={
        "daily cloud cover cache": "data/intermediate/daily_cloud_*.npz",
    },
    formulas={
        "Correlation": "Pearson r on daily mean cloud_cover_pct, restricted to given "
                        "calendar months, over the 2015-2024 (10-year) joint record "
                        "(modules.weather_stats.correlation_matrix)",
        "Joint clear probability": "P(>=1 city has daily-mean cloud_cover < 85%), "
                                    "computed directly from the joint daily record "
                                    "(no independence assumption); annual figure pools "
                                    "all 12 calendar months for comparability with "
                                    "Dang et al. (2023) Fig. 7, which is not "
                                    "month-specific in the source paper.",
    },
)
print(f"\nDone in {time.time() - t0:.1f}s")
