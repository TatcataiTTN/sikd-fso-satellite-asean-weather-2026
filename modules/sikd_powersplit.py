"""
sikd_powersplit.py — SIKD Key/Data Power-Split Optimization (Task 8-9, plan 07-5)
================================================================================
Explores the trade-off between the QKD channel modulation depth m_K and the
classical data channel modulation depth m_D on a shared SCM optical carrier.
Reuses modules/sikd_performance.compute_sikd_performance directly for the
key channel (it already accepts mK/mD as parameters — no refactor of that
module was needed, so the existing 207+ tests built on it are untouched).

Two noise-coupling directions exist on this shared carrier (vu2022dtdd):
  - Data channel leaking into the key receiver: sigma_CT^2 ~ m_D^2
    (already implemented as sikd_performance.compute_noise's sigma2_CT term).
  - Key channel leaking into the data receiver: sigma_CT^2 ~ m_K^2 (small,
    since m_K << m_D in practice) — NOT present in sikd_performance.py, so
    this module adds it via crosstalk_variance(), reusing the identical
    functional form with the leaking channel's own modulation index.

Data channel throughput definition (Task 8 spec: "chọn 1, ghi rõ"): this
module uses the simple UNCODED goodput approximation
    data_throughput = R_b * (1 - BER)
i.e. the fraction of bits assumed correctly received with no forward error
correction, rather than a channel-capacity (1 - H2(BER)) or a "1-2*BER"
heuristic (the latter has no clean derivation for this channel model).
"""
import numpy as np

from modules.sikd_performance import (
    compute_sikd_performance, compute_BER_CC,
    RE_DEFAULT, T_DEFAULT, RL_DEFAULT, RB_DEFAULT, ISO_DEFAULT,
    Q_ELECTRON, K_BOLTZMANN,
)

PT_DEFAULT = 1.0  # W, matches sikd_performance.compute_sikd_performance default


def crosstalk_variance(m_leak, hg, hl, PT=PT_DEFAULT, Re=RE_DEFAULT, Iso_dB=ISO_DEFAULT):
    """RF crosstalk variance leaking from one SCM channel into the other's
    receiver, after the shared optical bandpass filter (vu2022dtdd). Same
    functional form regardless of which channel leaks:
      - m_leak = m_D  -> data-into-key crosstalk (sikd_performance's sigma2_CT)
      - m_leak = m_K  -> key-into-data crosstalk (used below for the data
        channel's own noise budget)
    Scales as m_leak^2 by construction.
    """
    return (Re * PT * m_leak * hg * hl) ** 2 * 10.0 ** (-Iso_dB / 10.0) / 2.0


def _data_channel_noise_std(m_K, m_D, hg, hl, PT=PT_DEFAULT, Re=RE_DEFAULT,
                            T=T_DEFAULT, RL=RL_DEFAULT, Rb=RB_DEFAULT,
                            Iso_dB=ISO_DEFAULT):
    """Total noise std dev at the DATA channel receiver: its own shot and
    thermal noise (driven by m_D, its own signal) plus crosstalk leaking in
    from the key channel (driven by m_K, small since m_K << m_D)."""
    delta_f = Rb / 2.0
    p_signal_d = 0.5 * PT * m_D * hg * hl

    sigma2_shot_d = 2.0 * Q_ELECTRON * Re * p_signal_d * delta_f
    sigma2_thermal_d = (4.0 * K_BOLTZMANN * T / RL) * delta_f
    sigma2_ct_reverse = crosstalk_variance(m_K, hg, hl, PT, Re, Iso_dB)

    return np.sqrt(sigma2_shot_d + sigma2_thermal_d + sigma2_ct_reverse)


def evaluate_split(m_K, m_D, hg, hl, sigma_X2, PT=PT_DEFAULT, Re=RE_DEFAULT,
                   T=T_DEFAULT, RL=RL_DEFAULT, Rb=RB_DEFAULT, Iso_dB=ISO_DEFAULT):
    """Evaluate one (m_K, m_D) operating point on the given clear-air
    channel state, returning {SKR_kbps, data_ber, data_throughput_gbps}.

    Raises ValueError if m_K + m_D > 1 (modulation budget violation).
    """
    if m_K + m_D > 1.0:
        raise ValueError(f"m_K + m_D = {m_K + m_D:.3f} exceeds the modulation "
                         f"budget of 1.0 (m_K={m_K}, m_D={m_D})")

    perf = compute_sikd_performance(hg, hl, sigma_X2, PT=PT, mK=m_K, mD=m_D,
                                    Iso_dB=Iso_dB, Re=Re, T=T, RL=RL, Rb=Rb)

    sigma_n_d = _data_channel_noise_std(m_K, m_D, hg, hl, PT, Re, T, RL, Rb, Iso_dB)
    ber = compute_BER_CC(hg, hl, sigma_X2, PT, m_D, sigma_n_d, Re, Gd=1.0)
    throughput_gbps = Rb * (1.0 - ber) / 1e9

    return {
        "SKR_kbps": perf["SKR_kbps"],
        "data_ber": float(ber),
        "data_throughput_gbps": float(throughput_gbps),
    }


def pareto_frontier(channel_state, n_grid=15):
    """Non-dominated (SKR_kbps, data_throughput_gbps) frontier over a grid
    of (m_K, m_D) splits, sorted by SKR ascending with strictly decreasing
    throughput (ties broken by dropping the dominated/duplicate point).

    channel_state: {"hg": ..., "hl": ..., "sigma_X2": ...}
    """
    hg, hl, sigma_x2 = channel_state["hg"], channel_state["hl"], channel_state["sigma_X2"]
    mk_vals = np.linspace(0.01, 0.30, n_grid)
    md_vals = np.linspace(0.10, 0.90, n_grid)

    points = []
    for mk in mk_vals:
        for md in md_vals:
            if mk + md > 1.0:
                continue
            out = evaluate_split(mk, md, hg, hl, sigma_x2)
            points.append((out["SKR_kbps"], out["data_throughput_gbps"], float(mk), float(md)))

    non_dominated = []
    for i, (skr_i, thr_i, mk_i, md_i) in enumerate(points):
        dominated = any(
            j != i and skr_j >= skr_i and thr_j >= thr_i and (skr_j > skr_i or thr_j > thr_i)
            for j, (skr_j, thr_j, _, _) in enumerate(points)
        )
        if not dominated:
            non_dominated.append((skr_i, thr_i, mk_i, md_i))

    non_dominated.sort(key=lambda p: p[0])

    frontier = []
    last_thr = None
    for skr, thr, mk, md in non_dominated:
        if last_thr is not None and thr >= last_thr:
            continue
        frontier.append({"SKR_kbps": skr, "data_throughput_gbps": thr, "m_K": mk, "m_D": md})
        last_thr = thr

    return frontier


def adaptive_split(K_req_bits, T_pass_s, channel_state, m_k_grid=None):
    """Smallest m_K (with m_D given the rest of the modulation budget,
    1 - m_K) such that SKR_kbps * 1000 * T_pass_s >= K_req_bits.

    Returns (m_K, m_D) or None if infeasible even at the largest m_K tried.
    """
    hg, hl, sigma_x2 = channel_state["hg"], channel_state["hl"], channel_state["sigma_X2"]
    if m_k_grid is None:
        m_k_grid = np.linspace(0.01, 0.30, 30)

    for m_k in m_k_grid:
        m_d = 1.0 - m_k
        if m_d < 0.0:
            continue
        out = evaluate_split(float(m_k), float(m_d), hg, hl, sigma_x2)
        if out["SKR_kbps"] * 1e3 * T_pass_s >= K_req_bits:
            return (float(m_k), float(m_d))

    return None
