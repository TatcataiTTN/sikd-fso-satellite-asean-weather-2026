"""
sikd_performance.py — SIKD QKD Performance Analysis
=====================================================

Tính hiệu năng của hệ thống SIKD (Simultaneous Information and Key Distribution)
dùng CV-QKD với Dual-Threshold / Direct Detection (DT/DD).

Pipeline tính toán
------------------
    Inputs: channel params (hg, hl, σX²) + system params (PT, mK, mD, Iso, ...)
        ↓
    compute_noise()       → σ²_N (shot + thermal + background + crosstalk)
        ↓
    compute_thresholds()  → d0, d1 (dual-threshold decision boundaries)
        ↓
    compute_Psift_QBER()  → Psift, QBER  (via Gauss-Hermite quadrature)
        ↓
    compute_SKR()         → SKR [bit/s/Hz]  (normalized secret key rate)
    compute_BER_CC()      → BER_CC          (classical channel bit error rate)

Ký hiệu chính
-------------
    PT   — average optical transmit power (W)
    mK   — modulation index, QKD channel (f1), mK ≪ 1
    mD   — modulation index, classical data channel (f2), mD ~ 0.5–0.9
    Iso  — BPF filter isolation between f1 and f2 (dB)
    Re   — photodetector responsivity (A/W)
    T    — receiver temperature (K)
    RL   — load resistance (Ω)
    Rb   — raw bit rate (bps)
    Pbg  — background optical power (W)
    delta (δ) — SIM modulation depth (Paper 1 notation), same role as mK
    zeta (ζ)  — DT scale coefficient (threshold width parameter)

References
----------
    [P1] Vu et al., IEEE Access 2022, Eq. 7–31
    [P2] Vu, PTIT 2025, Eq. 1–31
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
Q_ELECTRON = 1.602e-19   # C
K_BOLTZMANN = 1.381e-23  # J/K

# Gauss-Hermite quadrature nodes and weights (computed once at import)
_GH_N = 20
_GH_XI, _GH_WI = hermgauss(_GH_N)

# ---------------------------------------------------------------------------
# Default system parameters (Paper 2 Table I)
# ---------------------------------------------------------------------------
RE_DEFAULT   = 0.9      # A/W  — InGaAs PIN responsivity at 1550 nm
T_DEFAULT    = 280.0    # K    — receiver temperature
RL_DEFAULT   = 1e3      # Ω    — load resistance
RB_DEFAULT   = 1e9      # bps  — raw bit rate (Rk = 1 Gbps)
PBG_DEFAULT  = 0.0      # W    — background power (0 = night / filtered)
ISO_DEFAULT  = 15.0     # dB   — BPF filter isolation (Paper 2 Table I nominal)
GK_DEFAULT   = 1.0      # —    — key receiver branch gain
GD_DEFAULT   = 1.0      # —    — data receiver branch gain


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _q_function(x: float | np.ndarray) -> float | np.ndarray:
    """Q(x) = 0.5 · erfc(x / √2)."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def binary_entropy(p: float) -> float:
    """
    Binary entropy function H(p) = -p·log2(p) - (1-p)·log2(1-p).

    Returns 0 for p=0 or p=1 (boundary cases).

    Parameters
    ----------
    p : probability ∈ [0, 1]

    Returns
    -------
    H : entropy in bits ∈ [0, 1]
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


# ---------------------------------------------------------------------------
# 1. NOISE MODEL
# ---------------------------------------------------------------------------

def compute_noise(
    hg: float,
    hl: float,
    PT: float,
    mK: float,
    Re: float = RE_DEFAULT,
    T: float = T_DEFAULT,
    RL: float = RL_DEFAULT,
    Rb: float = RB_DEFAULT,
    Pbg: float = PBG_DEFAULT,
    mD: float = 0.5,
    Iso_dB: float = ISO_DEFAULT,
) -> dict:
    """
    Tính các thành phần nhiễu tại kênh khóa QKD (f1).

    Tổng nhiễu:
        σ²_N = σ²_shot + σ²_thermal + σ²_bg + σ²_CT

    Công thức từng thành phần [P2, Eq. 5-6]:

        σ²_shot   = 2q · Re · (1/2)·PT·mK·hg·hl · Δf     [P2 Eq.23: coefficient 1/2]
        σ²_thermal = (4·kB·T / RL) · Δf
        σ²_bg     = 2q · Re · Pbg · Δf
        σ²_CT     = (Re·PT·mD·hg·hl)² · 10^(-Iso/10) / 2  [P2 Eq.6, sau khi qua optical channel]

    với Δf = Rb/2 (efficient bandwidth).

    Parameters
    ----------
    hg     : geometric loss (linear)
    hl     : atmospheric attenuation (linear)
    PT     : average transmit power (W) — Paper 2 nominal: 1 W (30 dBm)
    mK     : QKD channel modulation index
    Re     : photodetector responsivity (A/W)
    T      : receiver temperature (K)
    RL     : load resistance (Ω)
    Rb     : raw bit rate (bps)
    Pbg    : background optical power (W)
    mD     : data channel modulation index (for crosstalk)
    Iso_dB : BPF filter isolation (dB)

    Returns
    -------
    dict
        sigma2_shot    : shot noise variance (A²)
        sigma2_thermal : thermal noise variance (A²)
        sigma2_bg      : background noise variance (A²)
        sigma2_CT      : RF crosstalk variance (optical-domain W², Paper 2 Eq.6)
        sigma2_total   : total noise variance
        sigma_N        : total noise std dev
    """
    delta_f = Rb / 2.0

    # Signal power at receiver (mean photocurrent, coefficient 1/2 per Paper 2 Eq.23)
    P_signal = 0.5 * PT * mK * hg * hl

    sigma2_shot    = 2.0 * Q_ELECTRON * Re * P_signal * delta_f
    sigma2_thermal = (4.0 * K_BOLTZMANN * T / RL) * delta_f
    sigma2_bg      = 2.0 * Q_ELECTRON * Re * Pbg * delta_f
    # Crosstalk: RF leakage at f1 after optical channel, P2 Eq.6
    # iCT = Re·PT·mD·hg·hl·10^(-Iso/20)·cos(...) → σCT² = (Re·PT·mD·hg·hl)²·10^(-Iso/10)/2
    sigma2_CT      = (Re * PT * mD * hg * hl)**2 * 10.0**(-Iso_dB / 10.0) / 2.0

    sigma2_total = sigma2_shot + sigma2_thermal + sigma2_bg + sigma2_CT

    return {
        "sigma2_shot":    sigma2_shot,
        "sigma2_thermal": sigma2_thermal,
        "sigma2_bg":      sigma2_bg,
        "sigma2_CT":      sigma2_CT,
        "sigma2_total":   sigma2_total,
        "sigma_N":        np.sqrt(sigma2_total),
    }


# ---------------------------------------------------------------------------
# 2. DUAL-THRESHOLD DETECTION
# ---------------------------------------------------------------------------

def compute_thresholds(
    hg: float,
    hl: float,
    PT: float,
    mK: float,
    sigma_N: float,
    zeta_scale: float = 2.0,
    Re: float = RE_DEFAULT,
    Gk: float = GK_DEFAULT,
) -> dict:
    """
    Tính ngưỡng quyết định d0, d1 cho DT/DD.

    Giá trị trung bình tín hiệu nhận (không có turbulence) [P2, Eq. 23]:
        E[i_0] = -(1/2) · Re · Gk · PT · mK · hg · hl   (bit '0')
        E[i_1] = +(1/2) · Re · Gk · PT · mK · hg · hl   (bit '1')

    Ngưỡng [P1, Eq. 24-25 / P2, Eq. 21-23]:
        d0 = E[i_0] - ζ · σN   (lower threshold)
        d1 = E[i_1] + ζ · σN   (upper threshold)

    Guard zone [d0, d1]: tín hiệu rơi vào đây → discard (vùng X).

    Parameters
    ----------
    hg         : geometric loss (linear)
    hl         : atmospheric attenuation (linear)
    PT         : average transmit power (W)
    mK         : QKD channel modulation index
    sigma_N    : total noise std dev (A)
    zeta_scale : DT scale coefficient ζ (dimensionless). Lớn hơn → vùng X rộng hơn.
    Re         : photodetector responsivity (A/W)
    Gk         : key receiver branch gain (Paper 2 Eq. 23)

    Returns
    -------
    dict
        i_mean_pos : E[i_1] — mean current for bit '1' (A)
        i_mean_neg : E[i_0] — mean current for bit '0' (A)
        d0         : lower threshold (A)
        d1         : upper threshold (A)
        guard_width: d1 - d0 (A)
    """
    i_mean = 0.5 * Re * Gk * PT * mK * hg * hl   # Paper 2 Eq. 23

    i_mean_pos = +i_mean   # bit '1'
    i_mean_neg = -i_mean   # bit '0'

    d0 = i_mean_neg - zeta_scale * sigma_N
    d1 = i_mean_pos + zeta_scale * sigma_N

    return {
        "i_mean_pos": i_mean_pos,
        "i_mean_neg": i_mean_neg,
        "d0":         d0,
        "d1":         d1,
        "guard_width": d1 - d0,
    }


# ---------------------------------------------------------------------------
# 3. SIFT PROBABILITY AND QBER
# ---------------------------------------------------------------------------

def _P_detect(
    i_mean: float,
    threshold: float,
    sigma_N: float,
    sigma_X2: float,
    direction: str,
) -> float:
    """
    Tính xác suất phát hiện P(detect | bit x) qua Gauss-Hermite quadrature.

    Tích phân qua log-normal turbulence fading:
        P = (1/√π) · Σ_k w_k · Q(arg_k)

    với thay biến: ha_k = exp(2·√(2σX²)·x_k + 2μX), μX = -σX²

    Parameters
    ----------
    i_mean    : mean photocurrent without fading (A)
    threshold : d0 (for bit '0') or d1 (for bit '1')
    sigma_N   : noise std dev (A)
    sigma_X2  : log-amplitude variance
    direction : 'below' → Q((i - d0)/σN), 'above' → Q((d1 - i)/σN)

    Returns
    -------
    P_detect : detection probability ∈ [0, 0.5]
    """
    mu_X = -sigma_X2
    result = 0.0

    for xi, wi in zip(_GH_XI, _GH_WI):
        ha_i = np.exp(2.0 * np.sqrt(2.0 * sigma_X2) * xi + 2.0 * mu_X)
        i_received = i_mean * ha_i

        if direction == "below":
            arg = (i_received - threshold) / sigma_N
        else:  # 'above'
            arg = (threshold - i_received) / sigma_N

        result += wi * _q_function(arg)

    return result / np.sqrt(np.pi)


def compute_Psift_QBER(
    hg: float,
    hl: float,
    sigma_X2: float,
    sigma_N: float,
    d0: float,
    d1: float,
    PT: float,
    mK: float,
    Re: float = RE_DEFAULT,
    Gk: float = GK_DEFAULT,
) -> dict:
    """
    Tính Sift Probability và QBER bằng Gauss-Hermite quadrature.

    Định nghĩa [P1, Eq. 20-31]:
        P(U|C)(0|0) = P(i ≤ d0 | bit=0)  — correct detection of '0'
        P(U|C)(1|1) = P(i ≥ d1 | bit=1)  — correct detection of '1'
        P(U|C)(1|0) = P(i ≥ d1 | bit=0)  — error: '0' detected as '1'
        P(U|C)(0|1) = P(i ≤ d0 | bit=1)  — error: '1' detected as '0'

        Psift  = (1/2) · [P(0|0) + P(1|0) + P(0|1) + P(1|1)]
        Perror = (1/2) · [P(1|0) + P(0|1)]
        QBER   = Perror / Psift

    Parameters
    ----------
    hg       : geometric loss (linear)
    hl       : atmospheric attenuation (linear)
    sigma_X2 : log-amplitude variance (from compute_rytov_variance)
    sigma_N  : total noise std dev (A)
    d0       : lower threshold (A)
    d1       : upper threshold (A)
    PT       : average transmit power (W)
    mK       : QKD channel modulation index
    Re       : photodetector responsivity (A/W)
    Gk       : key receiver branch gain (Paper 2 Eq. 23)

    Returns
    -------
    dict
        Psift  : sift probability ∈ (0, 0.5]
        QBER   : quantum bit error rate ∈ [0, 0.5]
        Perror : error probability
        P00    : P(detect 0 | sent 0)
        P11    : P(detect 1 | sent 1)
        P10    : P(detect 1 | sent 0)  — error
        P01    : P(detect 0 | sent 1)  — error
    """
    i_mean = 0.5 * Re * Gk * PT * mK * hg * hl   # Paper 2 Eq. 23

    # Correct detections
    P00 = _P_detect(-i_mean, d0, sigma_N, sigma_X2, "below")  # bit 0 → detect 0
    P11 = _P_detect(+i_mean, d1, sigma_N, sigma_X2, "above")  # bit 1 → detect 1

    # Errors (cross-detections)
    P10 = _P_detect(-i_mean, d1, sigma_N, sigma_X2, "above")  # bit 0 → detect 1
    P01 = _P_detect(+i_mean, d0, sigma_N, sigma_X2, "below")  # bit 1 → detect 0

    Psift  = 0.5 * (P00 + P11 + P10 + P01)
    Perror = 0.5 * (P10 + P01)
    QBER   = Perror / Psift if Psift > 0 else 0.5

    return {
        "Psift":  Psift,
        "QBER":   QBER,
        "Perror": Perror,
        "P00":    P00,
        "P11":    P11,
        "P10":    P10,
        "P01":    P01,
    }


# ---------------------------------------------------------------------------
# 4. SECRET KEY RATE
# ---------------------------------------------------------------------------

def compute_SKR(
    Psift: float,
    QBER: float,
    chi_E: float = 0.0,
    reconciliation_eff: float = 1.0,
) -> float:
    """
    Tính normalized Secret Key Rate (bit/s/Hz).

    Công thức [P2, Eq. 25]:
        SKR_norm = Psift · [β · I(A;B) - χ(A:E)]

    với:
        I(A;B) ≈ 1 - H(QBER)   (mutual information, binary entropy approx)
        χ(A:E)                  (Holevo information of Eve, default 0)
        β                       (reconciliation efficiency, default 1)

    SKR = 0 nếu QBER vượt ngưỡng bảo mật (I(A;B) ≤ χ(A:E)).

    Parameters
    ----------
    Psift            : sift probability
    QBER             : quantum bit error rate
    chi_E            : Holevo information of Eve (0 = no eavesdropping assumed)
    reconciliation_eff: error reconciliation efficiency β ∈ (0, 1]

    Returns
    -------
    SKR_norm : normalized SKR (bit/s/Hz), ≥ 0
    """
    I_AB = 1.0 - binary_entropy(QBER)
    raw  = Psift * (reconciliation_eff * I_AB - chi_E)
    return max(0.0, raw)


def compute_SKR_bps(
    Psift: float,
    QBER: float,
    Rb: float = RB_DEFAULT,
    chi_E: float = 0.0,
    reconciliation_eff: float = 1.0,
) -> float:
    """
    Tính Secret Key Rate tuyệt đối (bps).

    SKR_bps = SKR_norm × Rb

    Parameters
    ----------
    Psift            : sift probability
    QBER             : quantum bit error rate
    Rb               : raw bit rate (bps)
    chi_E            : Holevo information of Eve
    reconciliation_eff: reconciliation efficiency

    Returns
    -------
    SKR_bps : secret key rate (bps)
    """
    return compute_SKR(Psift, QBER, chi_E, reconciliation_eff) * Rb


# ---------------------------------------------------------------------------
# 5. BER CLASSICAL DATA CHANNEL
# ---------------------------------------------------------------------------

def compute_BER_CC(
    hg: float,
    hl: float,
    sigma_X2: float,
    PT: float,
    mD: float,
    sigma_N_D: float,
    Re: float = RE_DEFAULT,
    Gd: float = 1.0,
) -> float:
    """
    Tính BER trung bình của kênh dữ liệu cổ điển (f2) qua turbulence fading.

    Công thức [P2, Eq. 27-31]:
        SNR_D(ha) = (Gd · Re · PT · mD · hg · hl · ha)² / (2 · σ²_N,D)
        Pe(ha)    = Q(√(2 · SNR_D(ha)))
        BER_avg   = (1/√π) · Σ_k w_k · Q(SNR_amplitude_k)

    với Gauss-Hermite quadrature qua log-normal ha.

    Parameters
    ----------
    hg        : geometric loss (linear)
    hl        : atmospheric attenuation (linear)
    sigma_X2  : log-amplitude variance
    PT        : average transmit power (W)
    mD        : data channel modulation index
    sigma_N_D : noise std dev at data channel receiver (A)
    Re        : photodetector responsivity (A/W)
    Gd        : data channel amplifier gain

    Returns
    -------
    BER_avg : average BER of classical data channel ∈ [0, 0.5]
    """
    mu_X = -sigma_X2
    # Deterministic signal amplitude (without fading)
    A0 = Gd * Re * PT * mD * hg * hl

    result = 0.0
    for xi, wi in zip(_GH_XI, _GH_WI):
        ha_i = np.exp(2.0 * np.sqrt(2.0 * sigma_X2) * xi + 2.0 * mu_X)
        snr_amplitude = A0 * ha_i / sigma_N_D
        result += wi * _q_function(snr_amplitude)

    return result / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# 6. FULL SIKD PERFORMANCE — convenience wrapper
# ---------------------------------------------------------------------------

def compute_sikd_performance(
    hg: float,
    hl: float,
    sigma_X2: float,
    PT: float = 1.0,
    mK: float = 0.05,
    mD: float = 0.5,
    Iso_dB: float = ISO_DEFAULT,
    zeta_scale: float = 2.0,
    Re: float = RE_DEFAULT,
    T: float = T_DEFAULT,
    RL: float = RL_DEFAULT,
    Rb: float = RB_DEFAULT,
    Pbg: float = PBG_DEFAULT,
    chi_E: float = 0.0,
    Gk: float = GK_DEFAULT,
    Gd: float = GD_DEFAULT,
) -> dict:
    """
    Tính toàn bộ hiệu năng SIKD cho một điều kiện kênh.

    Đây là wrapper chính — gọi tuần tự:
        compute_noise → compute_thresholds → compute_Psift_QBER
        → compute_SKR → compute_BER_CC

    Ví dụ sử dụng
    -------------
    >>> from modules.channel_model import compute_channel
    >>> ch = compute_channel(H_S_km=550, zeta_deg=45, V_km=10)
    >>> perf = compute_sikd_performance(ch['hg'], ch['hl'], ch['sigma_X2'])
    >>> print(f"SKR = {perf['SKR_kbps']:.2f} kbps, QBER = {perf['QBER']:.2e}")

    Parameters
    ----------
    hg, hl, sigma_X2 : channel components from compute_channel()
    PT       : transmit power (W)
    mK       : QKD modulation index
    mD       : data channel modulation index
    Iso_dB   : BPF filter isolation (dB)
    zeta_scale: DT threshold scale coefficient
    Re, T, RL, Rb, Pbg : receiver parameters
    chi_E    : Eve's Holevo information

    Returns
    -------
    dict
        SKR_norm  : normalized SKR (bit/s/Hz)
        SKR_kbps  : SKR in kbps (= SKR_norm × Rb / 1000)
        QBER      : quantum bit error rate
        Psift     : sift probability
        BER_CC    : classical channel BER
        sigma_N   : total noise std dev (A)
        d0, d1    : decision thresholds (A)
        I_AB      : mutual information I(A;B)
        noise     : full noise breakdown dict
    """
    noise = compute_noise(hg, hl, PT, mK, Re, T, RL, Rb, Pbg, mD, Iso_dB)
    sigma_N = noise["sigma_N"]

    thr = compute_thresholds(hg, hl, PT, mK, sigma_N, zeta_scale, Re, Gk)

    qkd = compute_Psift_QBER(
        hg, hl, sigma_X2, sigma_N, thr["d0"], thr["d1"], PT, mK, Re, Gk
    )

    skr_norm = compute_SKR(qkd["Psift"], qkd["QBER"], chi_E)
    skr_kbps = skr_norm * Rb / 1e3

    ber_cc = compute_BER_CC(hg, hl, sigma_X2, PT, mD, sigma_N, Re, Gd)

    return {
        "SKR_norm": skr_norm,
        "SKR_kbps": skr_kbps,
        "QBER":     qkd["QBER"],
        "Psift":    qkd["Psift"],
        "BER_CC":   ber_cc,
        "sigma_N":  sigma_N,
        "d0":       thr["d0"],
        "d1":       thr["d1"],
        "I_AB":     1.0 - binary_entropy(qkd["QBER"]),
        "noise":    noise,
    }
