"""
channel_model.py — FSO Satellite-to-Ground Channel Model
=========================================================

Mô hình kênh truyền FSO (Free-Space Optical) cho liên kết vệ tinh LEO → mặt đất,
dùng trong hệ thống SIKD (Simultaneous Information and Key Distribution).

Mô hình ba thành phần
---------------------
Hệ số kênh tổng hợp:

    h(t) = hg × hl × ha(t)

    hg  — geometric spreading loss (tất định)
            Mất mát do chùm laser Gaussian lan rộng theo khoảng cách.
            Phụ thuộc: bước sóng λ, góc phân kỳ θC, khoảng cách L, aperture aR.

    hl  — atmospheric attenuation (tất định, phụ thuộc thời tiết)
            Suy hao Beer-Lambert qua cột khí quyển (H_atm = 20 km).
            Nguồn: tầm nhìn V (Kruse/Kim model) + lượng mưa R (tropical model).

    ha(t) — turbulence fading (ngẫu nhiên, stochastic)
            Biến động cường độ do nhiễu loạn chiết suất khí quyển.
            Phân phối: Log-normal (yếu, σR² < 1) hoặc Gamma-Gamma (trung bình–mạnh).

Cấu trúc module
---------------
    Nhóm 1 — Geometric loss  : compute_beam_waist, compute_beam_radius, compute_hg
    Nhóm 2 — Attenuation     : compute_sigma_visibility, compute_sigma_rain,
                                compute_sigma_fog, compute_hl
    Nhóm 3 — Turbulence      : compute_Cn2_HV, compute_rytov_variance,
                                sample_ha_lognormal, pdf_ha_lognormal,
                                compute_gamma_gamma_params, pdf_ha_gamma_gamma
    Nhóm 4 — Wrapper         : compute_channel

Tham số mặc định (Paper 1 & 2)
-------------------------------
    λ = 1550 nm   (telecom C-band)
    θC = 10 μrad  (beam divergence)
    aR = 5 cm     (receiver aperture radius)
    HS = 550 km   (Starlink LEO altitude)
    W  = 21 m/s   (high-altitude wind, H-V model)
    Cn²(0) = 1.7×10⁻¹⁴ m⁻²/³  (moderate ground turbulence)

Lưu ý đơn vị
-------------
    - Độ cao (h, H_U): mét [m]
    - Khoảng cách khí quyển (L_atm): km [km]
    - Hệ số suy hao σ: km⁻¹
    - Cn²: m⁻²/³
    - Tất cả hệ số kênh (hg, hl, ha): không thứ nguyên ∈ (0, 1]

References
----------
    [P1] Vu et al., "Satellite-Based CV-QKD with BBM92 and DT/DD,"
         IEEE Access, 2022. DOI: 10.1109/ACCESS.2022.3217220
    [P2] Vu, "SIKD-SCM Satellite QKD System," PTIT, 2025.
    [P5] Koné et al., "FSO Channel Modeling for Tropical Climate (Abidjan),"
         Int. J. Photonics, 2024.
    [AP] Andrews & Phillips, "Laser Beam Propagation through Random Media,"
         SPIE Press, 2005.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, gamma as gamma_func, kv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDA_NM_DEFAULT = 1550       # nm — telecom C-band
LAMBDA_M_DEFAULT  = 1550e-9    # m
H_ATM_KM          = 20.0       # km — effective atmospheric height (Beer-Lambert)
H_ATM_M           = 20_000.0   # m
W_WIND_DEFAULT     = 21.0      # m/s — high-altitude wind speed (H-V model)
CN2_0_DEFAULT      = 1.7e-14   # m^(-2/3) — moderate ground turbulence


# ---------------------------------------------------------------------------
# 1. GEOMETRIC SPREADING LOSS
# ---------------------------------------------------------------------------

def compute_beam_waist(lambda_m: float, theta_C_rad: float) -> float:
    """
    Beam waist w0 = λ / (2·θC).

    Parameters
    ----------
    lambda_m   : wavelength (m)
    theta_C_rad: beam divergence half-angle (rad)

    Returns
    -------
    w0 : beam waist (m)
    """
    return lambda_m / (2.0 * theta_C_rad)


def compute_beam_radius(w0: float, L: float, lambda_m: float) -> float:
    """
    Gaussian beam radius at propagation distance L.

    wL = w0 * sqrt(1 + (L·λ / π·w0²)²)

    [Paper 2, Eq. 9]

    Parameters
    ----------
    w0      : beam waist (m)
    L       : propagation distance (m)
    lambda_m: wavelength (m)

    Returns
    -------
    wL : beam radius at distance L (m)
    """
    return w0 * np.sqrt(1.0 + (L * lambda_m / (np.pi * w0**2))**2)


def compute_hg(
    H_S_km: float,
    zeta_deg: float,
    a_R: float = 0.05,
    lambda_nm: float = LAMBDA_NM_DEFAULT,
    theta_C_urad: float = 10.0,
) -> tuple[float, float, float]:
    """
    Geometric spreading loss at zero pointing error.

    hg(0; L) = A0 = [erf(νR)]²

    where νR = √2 · aR / wL

    [Paper 1, Eq. 13 / Paper 2, Eq. 11]

    Parameters
    ----------
    H_S_km      : satellite altitude (km)
    zeta_deg    : zenith angle (degrees)
    a_R         : receiver aperture radius (m), default 5 cm (Paper 1)
    lambda_nm   : wavelength (nm)
    theta_C_urad: beam divergence half-angle (μrad)

    Returns
    -------
    hg      : geometric loss coefficient [0, 1]
    wL      : beam radius at receiver (m)
    nu_R    : normalized aperture radius
    """
    lambda_m = lambda_nm * 1e-9
    theta_C  = theta_C_urad * 1e-6
    zeta     = np.radians(zeta_deg)

    w0 = compute_beam_waist(lambda_m, theta_C)
    L  = (H_S_km * 1e3) / np.cos(zeta)          # slant path (m)
    wL = compute_beam_radius(w0, L, lambda_m)

    nu_R = np.sqrt(2.0) * a_R / wL
    A0   = erf(nu_R)**2
    return A0, wL, nu_R


# ---------------------------------------------------------------------------
# 2. ATMOSPHERIC ATTENUATION (Beer-Lambert + weather)
# ---------------------------------------------------------------------------

def _kruse_kim_exponent(V_km: float) -> float:
    """
    Wavelength exponent q(V) from Kruse/Kim model.

    [Paper 1, Eq. 15-16]
    """
    if V_km > 50.0:
        return 1.6
    elif V_km > 6.0:
        return 1.3
    elif V_km > 1.0:
        return 0.16 * V_km + 0.34
    elif V_km > 0.5:
        return V_km - 0.5
    else:
        return 0.0


def compute_sigma_visibility(V_km: float, lambda_nm: float = LAMBDA_NM_DEFAULT) -> float:
    """
    Atmospheric attenuation coefficient from visibility (Kruse/Kim model).

    σ(λ) = (3.912 / V) × (λ/550)^(-q(V))    [km⁻¹]

    [Paper 1, Eq. 15]

    Parameters
    ----------
    V_km      : visibility (km)
    lambda_nm : wavelength (nm)

    Returns
    -------
    sigma : attenuation coefficient (km⁻¹)
    """
    q = _kruse_kim_exponent(V_km)
    return (3.912 / V_km) * (lambda_nm / 550.0)**(-q)


def compute_sigma_rain(R_mm_h: float, alpha: float = 0.509, rho: float = 0.63) -> float:
    """
    Rain attenuation coefficient for tropical climate.

    βrain = α · R^ρ    [dB/km]
    σrain = βrain / 4.343    [km⁻¹]

    Parameters α=0.509, ρ=0.63 from Paper 5 (Koné 2024, Abidjan — tropical).

    [Paper 5, Eq. 1]

    Parameters
    ----------
    R_mm_h : rainfall rate (mm/h)
    alpha  : empirical coefficient (tropical default: 0.509)
    rho    : empirical exponent  (tropical default: 0.63)

    Returns
    -------
    sigma_rain : attenuation coefficient (km⁻¹)
    """
    if R_mm_h <= 0.0:
        return 0.0
    beta_dB_km = alpha * R_mm_h**rho
    return beta_dB_km / 4.343


def compute_sigma_fog(V_km: float, lambda_nm: float = LAMBDA_NM_DEFAULT) -> float:
    """
    Fog attenuation coefficient using Kim's model.

    βfog = (3.91 / V) × (λ/550)^(-p)    [dB/km]

    [Paper 5, Eq. 2-3]

    Parameters
    ----------
    V_km      : visibility (km)
    lambda_nm : wavelength (nm)

    Returns
    -------
    sigma_fog : attenuation coefficient (km⁻¹)
    """
    if V_km <= 0.0:
        return np.inf
    p = _kruse_kim_exponent(V_km)   # Kim uses same p(V) as Kruse
    beta_dB_km = (3.91 / V_km) * (lambda_nm / 550.0)**(-p)
    return beta_dB_km / 4.343


def compute_hl(
    zeta_deg: float,
    V_km: float = 10.0,
    R_mm_h: float = 0.0,
    lambda_nm: float = LAMBDA_NM_DEFAULT,
    H_U_km: float = 0.0,
    use_rain_model: bool = True,
) -> float:
    """
    Atmospheric attenuation factor via Beer-Lambert law.

        hl = exp(-σ_total · L_atm)

    với L_atm = (H_atm - H_U) / cos(ζ)  là slant path qua khí quyển hiệu dụng
    (H_atm = 20 km — độ cao mà phần lớn suy hao xảy ra bên dưới).

    Mô hình suy hao tổng hợp (additive):
        σ_total = σ_clear(V) + σ_rain(R)

    Trong đó:
        σ_clear — Kruse/Kim model từ tầm nhìn V (bao gồm cả sương mù)
        σ_rain  — tropical rain model từ lượng mưa R (Paper 5)

    Hai thành phần được cộng lại vì chúng là các cơ chế độc lập.
    Khi R=0, chỉ có σ_clear. Khi V lớn (trời trong), σ_clear nhỏ.

    Parameters
    ----------
    zeta_deg      : zenith angle (degrees). Phạm vi hợp lệ: [0°, 85°].
    V_km          : visibility (km). Dùng cho σ_clear (Kruse/Kim).
    R_mm_h        : rainfall rate (mm/h). 0 = không mưa.
    lambda_nm     : wavelength (nm). Default: 1550 nm.
    H_U_km        : ground station altitude (km). Default: 0 (sea level).
    use_rain_model: nếu False, bỏ qua σ_rain (dùng khi chỉ muốn clear-air).

    Returns
    -------
    hl : atmospheric transmission coefficient ∈ (0, 1]
    """
    zeta    = np.radians(zeta_deg)
    L_atm   = (H_ATM_KM - H_U_km) / np.cos(zeta)   # km

    sigma_clear = compute_sigma_visibility(V_km, lambda_nm)
    sigma_rain  = compute_sigma_rain(R_mm_h) if use_rain_model else 0.0

    sigma_total = sigma_clear + sigma_rain
    return np.exp(-sigma_total * L_atm)


# ---------------------------------------------------------------------------
# 3. TURBULENCE FADING
# ---------------------------------------------------------------------------

def compute_Cn2_HV(h: float, W: float = W_WIND_DEFAULT, Cn2_0: float = CN2_0_DEFAULT) -> float:
    """
    Hufnagel-Valley (H-V 5/7) refractive index structure parameter profile.

    Ba thành phần đại diện cho ba lớp turbulence:

        Cn²(h) = 0.00594·(W/27)²·(10⁻⁵h)^10·exp(-h/1000)   ← jet stream (~10 km)
               + 2.7×10⁻¹⁶·exp(-h/1500)                     ← tầng giữa (~1.5 km)
               + Cn²(0)·exp(-h/100)                           ← tầng thấp (~100 m)

    Lưu ý: số mũ 10 trong `(10⁻⁵h)^10` là đặc trưng của H-V 5/7 model.
    Lỗi phổ biến là dùng `(10⁻⁵h)^1` — sẽ cho Cn² sai 10 bậc độ lớn.

    [P1, Eq. 19 / P2, Eq. 16]

    Parameters
    ----------
    h     : altitude above ground (m). Phải ≥ 0.
    W     : high-altitude wind speed (m/s). Điển hình: 21 m/s.
    Cn2_0 : ground-level structure parameter (m⁻²/³).
            Yếu: 10⁻¹⁷, Trung bình: 1.7×10⁻¹⁴, Mạnh: 10⁻¹³.

    Returns
    -------
    Cn2 : refractive index structure parameter tại độ cao h (m⁻²/³)
    """
    term1 = 0.00594 * (W / 27.0)**2 * (1e-5 * h)**10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = Cn2_0 * np.exp(-h / 100.0)
    return term1 + term2 + term3


def compute_rytov_variance(
    zeta_deg: float,
    H_S_km: float = 550.0,
    H_U_m: float = 0.0,
    lambda_nm: float = LAMBDA_NM_DEFAULT,
    W: float = W_WIND_DEFAULT,
    Cn2_0: float = CN2_0_DEFAULT,
) -> float:
    """
    Rytov variance — measure of turbulence strength.

    σR² ≈ 0.56 · k^(7/6) · sec^(11/6)(ζ) · ∫ Cn²(h)·(h-HU)^(5/6) dh

    σX² = σR² / 4   (log-amplitude variance)

    [Paper 1, Eq. 18 / Paper 2, Eq. 15]

    Parameters
    ----------
    zeta_deg  : zenith angle (degrees)
    H_S_km    : satellite altitude (km)
    H_U_m     : ground station altitude (m)
    lambda_nm : wavelength (nm)
    W         : high-altitude wind speed (m/s)
    Cn2_0     : ground-level Cn² (m^(-2/3))

    Returns
    -------
    sigma_R2 : Rytov variance (dimensionless)
    sigma_X2 : log-amplitude variance = sigma_R2 / 4
    """
    lambda_m = lambda_nm * 1e-9
    k        = 2.0 * np.pi / lambda_m
    zeta     = np.radians(zeta_deg)
    H_S_m    = H_S_km * 1e3

    def integrand(h):
        return compute_Cn2_HV(h, W, Cn2_0) * (h - H_U_m)**(5.0 / 6.0)

    # Start integration slightly above H_U to avoid (h-H_U)^(5/6) singularity at h=H_U
    h_start = H_U_m + 1.0
    integral, _ = quad(integrand, h_start, H_S_m, limit=200)

    sigma_R2 = 0.56 * k**(7.0 / 6.0) * (1.0 / np.cos(zeta))**(11.0 / 6.0) * integral
    sigma_X2 = 0.25 * sigma_R2
    return sigma_R2, sigma_X2


def sample_ha_lognormal(sigma_X2: float, n_samples: int = 1, rng=None) -> np.ndarray:
    """
    Draw turbulence fading samples from log-normal distribution.

    ha = exp(2X),  X ~ N(μX, σX²),  μX = -σX²  (ensures E[ha] = 1)

    [Paper 1, Eq. 17]

    Parameters
    ----------
    sigma_X2  : log-amplitude variance
    n_samples : number of samples
    rng       : numpy Generator (for reproducibility)

    Returns
    -------
    ha : array of fading coefficients, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()
    mu_X = -sigma_X2
    X    = rng.normal(loc=mu_X, scale=np.sqrt(sigma_X2), size=n_samples)
    return np.exp(2.0 * X)


def pdf_ha_lognormal(ha: np.ndarray, sigma_X2: float) -> np.ndarray:
    """
    PDF of log-normal turbulence fading.

    f(ha) = 1 / (√(8π) · ha · σX) · exp(-(ln(ha) - 2μX)² / (8σX²))

    [Paper 1, Eq. 17]

    Parameters
    ----------
    ha       : fading values (must be > 0)
    sigma_X2 : log-amplitude variance

    Returns
    -------
    pdf values at ha
    """
    ha      = np.asarray(ha, dtype=float)
    mu_X    = -sigma_X2
    sigma_X = np.sqrt(sigma_X2)
    coeff   = 1.0 / (np.sqrt(8.0 * np.pi) * ha * sigma_X)
    exponent = -((np.log(ha) - 2.0 * mu_X)**2) / (8.0 * sigma_X2)
    return coeff * np.exp(exponent)


def compute_gamma_gamma_params(sigma_R2: float) -> tuple[float, float]:
    """
    Gamma-Gamma distribution parameters from Rytov variance.

    For plane wave (satellite downlink approximation):
        α = (exp(0.49·σR² / (1 + 1.11·σR^(12/5))^(7/6)) - 1)^(-1)
        β = (exp(0.51·σR² / (1 + 0.69·σR^(12/5))^(5/6)) - 1)^(-1)

    Valid for moderate-to-strong turbulence (σR² ≥ 0.5).

    Parameters
    ----------
    sigma_R2 : Rytov variance

    Returns
    -------
    alpha, beta : Gamma-Gamma shape parameters
    """
    sr = sigma_R2
    alpha = 1.0 / (np.exp(0.49 * sr / (1.0 + 1.11 * sr**(6.0/5.0))**(7.0/6.0)) - 1.0)
    beta  = 1.0 / (np.exp(0.51 * sr / (1.0 + 0.69 * sr**(6.0/5.0))**(5.0/6.0)) - 1.0)
    return alpha, beta


def pdf_ha_gamma_gamma(ha: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    PDF of Gamma-Gamma turbulence fading.

    f(ha) = 2(αβ)^((α+β)/2) / (Γ(α)·Γ(β)) · ha^((α+β)/2 - 1) · K_{α-β}(2√(αβ·ha))

    where K_ν is the modified Bessel function of the second kind.

    Parameters
    ----------
    ha    : fading values (must be > 0)
    alpha : large-scale scintillation parameter
    beta  : small-scale scintillation parameter

    Returns
    -------
    pdf values at ha
    """
    ha   = np.asarray(ha, dtype=float)
    nu   = alpha - beta
    coeff = (2.0 * (alpha * beta)**((alpha + beta) / 2.0)
             / (gamma_func(alpha) * gamma_func(beta)))
    power = ha**((alpha + beta) / 2.0 - 1.0)
    bessel = kv(nu, 2.0 * np.sqrt(alpha * beta * ha))
    return coeff * power * bessel


# ---------------------------------------------------------------------------
# 4. FULL CHANNEL — convenience wrapper
# ---------------------------------------------------------------------------

def compute_channel(
    H_S_km: float,
    zeta_deg: float,
    a_R: float = 0.05,
    lambda_nm: float = LAMBDA_NM_DEFAULT,
    theta_C_urad: float = 10.0,
    V_km: float = 10.0,
    R_mm_h: float = 0.0,
    W: float = W_WIND_DEFAULT,
    Cn2_0: float = CN2_0_DEFAULT,
    H_U_km: float = 0.0,
) -> dict:
    """
    Tính toàn bộ các thành phần kênh FSO cho một liên kết satellite-to-ground.

    Đây là hàm wrapper chính — gọi compute_hg, compute_hl, compute_rytov_variance
    và trả về tất cả kết quả trong một dict để dùng trong notebook/pipeline.

    Ví dụ sử dụng
    -------------
    >>> ch = compute_channel(H_S_km=550, zeta_deg=45, V_km=10, R_mm_h=0)
    >>> print(f"hg = {ch['hg_dB']:.1f} dB, hl = {ch['hl_dB']:.1f} dB")
    >>> print(f"σR² = {ch['sigma_R2']:.4f}  →  {'weak' if ch['sigma_R2'] < 1 else 'strong'}")

    Parameters
    ----------
    H_S_km       : satellite altitude (km). Starlink: 550 km.
    zeta_deg     : zenith angle (degrees). Elevation = 90° - zeta.
    a_R          : receiver aperture radius (m). Paper 1: 5 cm.
    lambda_nm    : wavelength (nm). Default: 1550 nm.
    theta_C_urad : beam divergence half-angle (μrad). Paper 1: 10 μrad.
    V_km         : visibility (km). Clear: 10–50 km, Fog: < 1 km.
    R_mm_h       : rainfall rate (mm/h). Tropical heavy: 10–50 mm/h.
    W            : high-altitude wind speed (m/s). H-V model: 21 m/s.
    Cn2_0        : ground-level Cn² (m⁻²/³). Moderate: 1.7×10⁻¹⁴.
    H_U_km       : ground station altitude (km). Default: 0 (sea level).

    Returns
    -------
    dict
        hg         : geometric loss coefficient (linear)
        hl         : atmospheric transmission (linear)
        sigma_R2   : Rytov variance (dùng để chọn turbulence model)
        sigma_X2   : log-amplitude variance = sigma_R2 / 4
        wL         : beam radius tại receiver (m)
        nu_R       : normalized aperture radius νR = √2·aR/wL
        L_slant_km : slant path length (km), tính từ satellite đến mặt đất
        hg_dB      : hg in dB (âm, thường -40 đến -70 dB)
        hl_dB      : hl in dB (âm, thường -1 đến -10 dB)
    """
    hg, wL, nu_R = compute_hg(H_S_km, zeta_deg, a_R, lambda_nm, theta_C_urad)
    hl           = compute_hl(zeta_deg, V_km, R_mm_h, lambda_nm, H_U_km)
    sigma_R2, sigma_X2 = compute_rytov_variance(
        zeta_deg, H_S_km, H_U_km * 1e3, lambda_nm, W, Cn2_0
    )

    zeta      = np.radians(zeta_deg)
    L_slant   = (H_S_km - H_U_km) / np.cos(zeta)

    return {
        "hg":         hg,
        "hl":         hl,
        "sigma_R2":   sigma_R2,
        "sigma_X2":   sigma_X2,
        "wL":         wL,
        "nu_R":       nu_R,
        "L_slant_km": L_slant,
        "hg_dB":      10.0 * np.log10(hg) if hg > 0 else -np.inf,
        "hl_dB":      10.0 * np.log10(hl) if hl > 0 else -np.inf,
    }
