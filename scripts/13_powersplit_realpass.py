"""
13_powersplit_realpass.py — Task 9: Adaptive vs Fixed Power-Split on Real Passes
================================================================================
Evaluates modules/sikd_powersplit.adaptive_split (Task 8) against the fixed
operating point (m_K=0.05, m_D=0.5) on the REAL 7-day pass table (Task 5,
Hanoi + Jakarta), across 3 key-demand levels and 2 seasons (Jan/dry, Jul/wet).

Design choices (documented, not hidden assumptions):
  - Pass geometry reused from the existing Task 5 pass table (TLE-propagated
    window starting 2026-03-12). This is treated as season-invariant: Shell-1
    is NOT sun-synchronous, so pass frequency/duration/elevation statistics
    do not depend on calendar month -- only the WEATHER differs between
    January and July, which is what this script actually varies.
  - Per-pass weather: P_cloud and rain rate looked up from Task 2's HOURLY
    climatology (modules/weather_stats.load_hourly_climatology /
    load_hourly_rain_climatology) at (month, local_hour_peak of that pass),
    not the flat monthly average -- matches the Task 9 spec.
  - Each pass's expected performance mixes CLEAR and RAIN channel states
    using the same 3-state weighting used everywhere else in this project:
        p_clear = 1 - P_cloud - 0.15,  p_rain = 0.15 (fixed rain fraction)
    (cloud-blocked passes contribute 0 to both FIXED and ADAPTIVE equally,
    so they drop out of the comparison and are not simulated explicitly.)
  - ADAPTIVE split is chosen from the CLEAR-sky channel state of each pass
    (representing real-time channel sensing at pass start); the resulting
    (m_K, m_D) is then evaluated under BOTH clear and rain states with the
    SAME weighting as FIXED, so the two policies are compared on an
    identical expectation basis, differing only in which split is used.
    If adaptive_split finds no feasible split (demand too high even at
    max m_K), a max-key fallback (m_K=0.30, m_D=0.70) is used for that pass.

Run:
  cd 05_Code_v2 && python scripts/13_powersplit_realpass.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from modules.channel_model import compute_channel
from modules.sikd_powersplit import evaluate_split, adaptive_split
from modules.weather_stats import load_hourly_climatology, load_hourly_rain_climatology
from utils import save_provenance, save_intermediate_csv, save_verify_numbers

plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 9, 'font.family': 'serif',
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
})

t0 = time.time()

CITIES = ["hanoi", "jakarta"]
SEASONS = {"dry (Jan)": 1, "wet (Jul)": 7}
K_REQ_LEVELS = [
    ("low (1 Gbit/pass)", 1e9),
    ("medium (3 Gbit/pass)", 3e9),
    ("high (5 Gbit/pass)", 5e9),
]
H_KM = 550.0
RAIN_FRACTION = 0.15  # project convention (CLAUDE.md 3-state model)
FIXED_MK, FIXED_MD = 0.05, 0.5
FALLBACK_MK, FALLBACK_MD = 0.30, 0.70  # used if adaptive_split finds nothing feasible
N_DAYS_WINDOW = 7.0  # Task 5 pass table spans 7 days

# ----------------------------------------------------------------
# Load real pass table (Task 5), filter to Hanoi + Jakarta
# ----------------------------------------------------------------
PASS_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'intermediate',
                        'pass_table_8cities_7days_elev30.csv')
passes_by_city = {c: [] for c in CITIES}
with open(PASS_CSV, newline='', encoding='utf-8') as f:
    next(f)  # skip description comment line
    reader = csv.DictReader(f)
    for row in reader:
        if row["station"] in passes_by_city:
            passes_by_city[row["station"]].append({
                "max_elev_deg": float(row["max_elev_deg"]),
                "duration_s": float(row["duration_s"]),
                "local_hour_peak": float(row["local_hour_peak"]),
            })
for c in CITIES:
    print(f"Loaded {len(passes_by_city[c])} real passes for {c} (elev >= 30 deg, 7-day window)")


def channel_state_at(elev_deg, V_km, R_mm_h):
    zenith = 90.0 - elev_deg
    ch = compute_channel(H_KM, zenith, V_km=V_km, R_mm_h=R_mm_h)
    return {"hg": ch["hg"], "hl": ch["hl"], "sigma_X2": ch["sigma_X2"]}


# ----------------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------------
rows = []
summary = {}  # (season, city, k_label) -> dict of totals (Gbit/day)

for season_label, month in SEASONS.items():
    for city in CITIES:
        cloud_matrix = load_hourly_climatology(city)       # (12,24), P_cloud
        rain_matrix = load_hourly_rain_climatology(city)   # (12,24), mm/h

        for k_label, K_req in K_REQ_LEVELS:
            fixed_key_gbit = 0.0
            fixed_data_gbit = 0.0
            adaptive_key_gbit = 0.0
            adaptive_data_gbit = 0.0
            n_fallback = 0

            for p in passes_by_city[city]:
                hour_bin = int(p["local_hour_peak"]) % 24
                p_cloud = float(cloud_matrix[month - 1, hour_bin])
                r_mm_h = float(rain_matrix[month - 1, hour_bin])
                v_km = max(15.0 - 10.0 * p_cloud, 0.5)
                p_clear = max(1.0 - p_cloud - RAIN_FRACTION, 0.0)
                p_rain = RAIN_FRACTION

                ch_clear = channel_state_at(p["max_elev_deg"], v_km, 0.0)
                ch_rain = channel_state_at(p["max_elev_deg"], v_km, r_mm_h)
                T = p["duration_s"]

                out_c = evaluate_split(FIXED_MK, FIXED_MD, **ch_clear)
                out_r = evaluate_split(FIXED_MK, FIXED_MD, **ch_rain)
                fixed_skr_kbps = p_clear * out_c["SKR_kbps"] + p_rain * out_r["SKR_kbps"]
                fixed_thr_gbps = p_clear * out_c["data_throughput_gbps"] + p_rain * out_r["data_throughput_gbps"]
                fixed_key_gbit += fixed_skr_kbps * T / 1e6
                fixed_data_gbit += fixed_thr_gbps * T

                split = adaptive_split(K_req, T, ch_clear)
                if split is None:
                    n_fallback += 1
                    m_K, m_D = FALLBACK_MK, FALLBACK_MD
                else:
                    m_K, m_D = split
                out_c2 = evaluate_split(m_K, m_D, **ch_clear)
                out_r2 = evaluate_split(m_K, m_D, **ch_rain)
                adap_skr_kbps = p_clear * out_c2["SKR_kbps"] + p_rain * out_r2["SKR_kbps"]
                adap_thr_gbps = p_clear * out_c2["data_throughput_gbps"] + p_rain * out_r2["data_throughput_gbps"]
                adaptive_key_gbit += adap_skr_kbps * T / 1e6
                adaptive_data_gbit += adap_thr_gbps * T

            n_passes = len(passes_by_city[city])
            fixed_key_per_day = fixed_key_gbit / N_DAYS_WINDOW
            fixed_data_per_day = fixed_data_gbit / N_DAYS_WINDOW
            adap_key_per_day = adaptive_key_gbit / N_DAYS_WINDOW
            adap_data_per_day = adaptive_data_gbit / N_DAYS_WINDOW

            data_gain_pct = (100.0 * (adap_data_per_day - fixed_data_per_day) / fixed_data_per_day
                            if fixed_data_per_day > 0 else float('nan'))
            key_gain_pct = (100.0 * (adap_key_per_day - fixed_key_per_day) / fixed_key_per_day
                           if fixed_key_per_day > 0 else float('nan'))

            summary[(season_label, city, k_label)] = {
                "fixed_key_gbit_per_day": fixed_key_per_day,
                "adaptive_key_gbit_per_day": adap_key_per_day,
                "fixed_data_gbit_per_day": fixed_data_per_day,
                "adaptive_data_gbit_per_day": adap_data_per_day,
                "data_gain_pct": data_gain_pct,
                "key_gain_pct": key_gain_pct,
                "n_fallback": n_fallback,
            }
            rows.append({
                "season": season_label, "city": city, "k_req_level": k_label,
                "fixed_key_gbit_per_day": round(fixed_key_per_day, 3),
                "adaptive_key_gbit_per_day": round(adap_key_per_day, 3),
                "fixed_data_gbit_per_day": round(fixed_data_per_day, 3),
                "adaptive_data_gbit_per_day": round(adap_data_per_day, 3),
                "data_gain_pct": round(data_gain_pct, 2),
                "key_gain_pct": round(key_gain_pct, 2),
                "n_passes": n_passes, "n_fallback": n_fallback,
            })
            print(f"  {season_label:12s} {city:8s} {k_label:24s} "
                  f"data_gain={data_gain_pct:+7.2f}%  key_gain={key_gain_pct:+7.2f}%  "
                  f"(fallback: {n_fallback}/{n_passes})")

save_intermediate_csv(rows, "powersplit_realpass_comparison",
                       "Fixed (0.05,0.5) vs adaptive_split on real 7-day passes, "
                       "Hanoi+Jakarta, Jan/Jul, 3 K_req levels")

# ----------------------------------------------------------------
# Figure: data gain % vs K_req level, 2 seasons, 2 cities (2 panels)
# ----------------------------------------------------------------
k_labels = [k[0] for k in K_REQ_LEVELS]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True)
for ax, city in zip(axes, CITIES):
    x = np.arange(len(k_labels))
    w = 0.35
    for i, (season_label, color) in enumerate(zip(SEASONS, ['#2196F3', '#FF5722'])):
        gains = [summary[(season_label, city, k)]["data_gain_pct"] for k in k_labels]
        ax.bar(x + (i - 0.5) * w, gains, width=w, label=season_label, color=color)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split(' (')[0] for k in k_labels], fontsize=7)
    ax.set_title(city.capitalize(), fontsize=8.5, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
axes[0].set_ylabel('Data throughput gain, adaptive vs fixed (%)')
axes[0].legend(fontsize=7)
plt.suptitle('Adaptive Power-Split Data Gain vs Key-Demand Level', fontsize=9, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'latex_paper_3', 'figures', 'fig13_powersplit_adaptive_gain.png')
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ----------------------------------------------------------------
# Verify + provenance
# ----------------------------------------------------------------
verify = {}
for (season_label, city, k_label), d in summary.items():
    tag = f"{city}_{season_label.split(' ')[0]}_{k_label.split(' ')[0]}"
    verify[f"{tag}_data_gain_pct"] = f"{d['data_gain_pct']:.2f}"
    verify[f"{tag}_key_gain_pct"] = f"{d['key_gain_pct']:.2f}"
# Headline number required by Task 9 acceptance: low demand, dry season
headline_gains = [summary[("dry (Jan)", c, K_REQ_LEVELS[0][0])]["data_gain_pct"] for c in CITIES]
verify["headline_data_gain_pct_low_demand_dry"] = f"{np.mean(headline_gains):.2f}"
save_verify_numbers(verify, "powersplit_realpass_comparison")

save_provenance(
    script_name="13_powersplit_realpass",
    params={"cities": CITIES, "seasons": SEASONS, "k_req_levels_bits": [k[1] for k in K_REQ_LEVELS],
            "rain_fraction": RAIN_FRACTION, "fixed_split": [FIXED_MK, FIXED_MD],
            "fallback_split": [FALLBACK_MK, FALLBACK_MD], "n_days_window": N_DAYS_WINDOW},
    key_numbers=verify,
    runtime_secs=time.time() - t0,
    output_files=[out1],
    data_sources={"pass table (Task 5)": PASS_CSV,
                  "hourly climatology (Task 2)": "modules/weather_stats.py"},
    formulas={
        "Per-pass expectation": "p_clear*eval(clear) + p_rain*eval(rain), same weighting as "
                                "SKR_eff elsewhere in this project (CLAUDE.md 3-state model)",
        "Adaptive split target": "chosen from adaptive_split(K_req, T_pass, clear_state); "
                                 "fallback (0.30,0.70) if infeasible even at max m_K",
    },
)
print(f"\nDone in {time.time() - t0:.2f}s")
