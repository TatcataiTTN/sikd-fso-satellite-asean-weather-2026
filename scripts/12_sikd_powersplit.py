"""
12_sikd_powersplit.py — Pareto Frontier: Key vs Data Modulation Split (Task 8)
================================================================================
Plots the (SKR, data throughput) Pareto frontier over the (m_K, m_D) power
split, at three representative channel states:
  1. Clear sky, zenith 0 deg  (elev 90 deg, best case)
  2. Clear sky, elev 40 deg   (zenith 50 deg, BB84-secure boundary, Mask B)
  3. Rain, elev 40 deg        (zenith 50 deg, degraded channel)

Marks the current fixed operating point (m_K=0.05, m_D=0.5) and the knee
point (closest to the top-right corner of the normalized objective space)
on each curve.

Run:
  cd 05_Code_v2 && python scripts/10_sikd_powersplit.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from modules.channel_model import compute_channel
from modules.sikd_powersplit import evaluate_split, pareto_frontier
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 9, 'font.family': 'serif',
    'axes.titlesize': 8.5, 'axes.labelsize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
})

t0 = time.time()

SCENARIOS = [
    ("Clear, zenith 0°", dict(H_S_km=550.0, zeta_deg=0.0, V_km=10.0, R_mm_h=0.0)),
    ("Clear, elev 40°", dict(H_S_km=550.0, zeta_deg=50.0, V_km=10.0, R_mm_h=0.0)),
    ("Rain, elev 40°", dict(H_S_km=550.0, zeta_deg=50.0, V_km=4.0, R_mm_h=10.0)),
]

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), sharey=False)
all_rows = []
verify = {}

for ax, (label, ch_params) in zip(axes, SCENARIOS):
    ch = compute_channel(**ch_params)
    channel_state = {"hg": ch["hg"], "hl": ch["hl"], "sigma_X2": ch["sigma_X2"]}

    front = pareto_frontier(channel_state, n_grid=15)
    skrs = [p["SKR_kbps"] for p in front]
    thrs = [p["data_throughput_gbps"] for p in front]
    ax.plot(skrs, thrs, color='#1f77b4', marker='o', markersize=2.5, linewidth=1.0)

    # Current fixed operating point
    op = evaluate_split(0.05, 0.5, channel_state["hg"], channel_state["hl"], channel_state["sigma_X2"])
    ax.scatter([op["SKR_kbps"]], [op["data_throughput_gbps"]], color='red', s=35,
               zorder=5, label='Current (0.05, 0.5)')

    # Knee point: closest to normalized (1,1) corner
    if front:
        skr_arr = np.array(skrs)
        thr_arr = np.array(thrs)
        skr_n = (skr_arr - skr_arr.min()) / (skr_arr.max() - skr_arr.min() + 1e-30)
        thr_n = (thr_arr - thr_arr.min()) / (thr_arr.max() - thr_arr.min() + 1e-30)
        dist = np.sqrt((1 - skr_n) ** 2 + (1 - thr_n) ** 2)
        knee_idx = int(np.argmin(dist))
        knee = front[knee_idx]
        ax.scatter([knee["SKR_kbps"]], [knee["data_throughput_gbps"]], color='green',
                   marker='*', s=60, zorder=5, label='Knee point')
        verify[f"knee_mK_{label.replace(' ', '_').replace(',', '')}"] = f"{knee['m_K']:.3f}"
        verify[f"knee_mD_{label.replace(' ', '_').replace(',', '')}"] = f"{knee['m_D']:.3f}"

    ax.set_xlabel('SKR (kbps)')
    if ax is axes[0]:
        ax.set_ylabel('Data throughput (Gbps)')
    ax.set_title(label, fontsize=8)
    ax.legend(fontsize=5.5, loc='lower left')
    ax.grid(alpha=0.3)

    for p in front:
        all_rows.append({"scenario": label, **p})

plt.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'latex_paper_3', 'figures', 'fig12_powersplit_pareto.png')
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

save_intermediate_csv(all_rows, "powersplit_pareto_frontiers",
                       "Pareto frontier points (SKR_kbps, data_throughput_gbps, m_K, m_D) "
                       "for 3 channel scenarios")
save_verify_numbers(verify, "sikd_powersplit_pareto")

save_provenance(
    script_name="12_sikd_powersplit",
    params={"scenarios": [s[0] for s in SCENARIOS], "n_grid": 15},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1],
    formulas={
        "Pareto frontier": "grid search m_K in [0.01,0.30], m_D in [0.10,0.90], "
                           "m_K+m_D<=1; non-dominated filter; modules/sikd_powersplit.py",
        "Data throughput": "R_b * (1 - BER), uncoded goodput approximation (chosen "
                           "over 1-H2(BER) or 1-2*BER per Task 8 spec)",
    },
)
print(f"\nDone in {time.time() - t0:.2f}s")
