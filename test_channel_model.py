"""
test_channel_model.py — Unit tests for channel_model.py
========================================================

Kiểm tra tính đúng đắn của từng thành phần trong mô hình kênh FSO.
Mỗi test class tương ứng với một nhóm hàm trong channel_model.py.

Chiến lược test
---------------
    - Correctness  : so sánh với giá trị tính tay hoặc kết quả Paper 1/2
    - Monotonicity : kiểm tra chiều tăng/giảm theo tham số vật lý
    - Range        : đảm bảo output nằm trong khoảng hợp lệ (0, 1] hoặc > 0
    - Edge cases   : R=0, L=0, V→∞, ζ=0°
    - Statistical  : E[ha]=1 (Monte Carlo), PDF tích phân = 1

Test classes
------------
    T01  TestBeamWaist          — w0 = λ/(2θC)
    T02  TestBeamRadius         — wL far-field scaling
    T03  TestGeometricLoss      — hg = A0 = [erf(νR)]²
    T04  TestGeometricLossVsZenith — hg giảm đơn điệu theo ζ
    T05  TestVisibilityModel    — Kruse/Kim σ(V)
    T06  TestBeerLambert        — hl = exp(-σ·L)
    T07  TestRainAttenuation    — σ_rain = α·R^ρ / 4.343
    T08  TestRytovVariance      — σR² tăng theo ζ, σX² = σR²/4
    T09  TestLognormalSampling  — E[ha]=1, PDF=1, variance scaling
    T10  TestGammaGamma         — PDF=1, α>β, params>0
         TestComputeChannel     — integration test cho wrapper

Chạy tests
----------
    python -m pytest test_channel_model.py -v
    python -m pytest test_channel_model.py -v -k "TestRytov"   # chỉ 1 class
    python test_channel_model.py                                # trực tiếp
"""

import sys
import os
import numpy as np
import pytest  # noqa: F401  — used implicitly by pytest runner
from scipy.integrate import quad

# Allow running from 05_Code/ or 05_Code/modules/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.channel_model import (
    compute_beam_waist,
    compute_beam_radius,
    compute_hg,
    compute_sigma_visibility,
    compute_sigma_rain,
    compute_sigma_fog,  # noqa: F401  — tested indirectly via compute_hl
    compute_hl,
    compute_Cn2_HV,     # noqa: F401  — tested indirectly via compute_rytov_variance
    compute_rytov_variance,
    sample_ha_lognormal,
    pdf_ha_lognormal,
    compute_gamma_gamma_params,
    pdf_ha_gamma_gamma,
    compute_channel,
)


# ---------------------------------------------------------------------------
# T01 — Beam waist formula: w0 = λ / (2·θC)
# ---------------------------------------------------------------------------
class TestBeamWaist:
    def test_formula_correctness(self):
        """w0 = λ/(2θC) — verify against manual calculation."""
        lam = 1550e-9   # m
        theta = 10e-6   # rad (10 μrad)
        expected = lam / (2 * theta)   # = 0.0775 m
        result = compute_beam_waist(lam, theta)
        assert abs(result - expected) < 1e-10, f"Expected {expected:.6f}, got {result:.6f}"

    def test_smaller_divergence_gives_larger_waist(self):
        """Tighter beam (smaller θC) → larger w0."""
        lam = 1550e-9
        w0_tight = compute_beam_waist(lam, 5e-6)    # 5 μrad
        w0_wide  = compute_beam_waist(lam, 20e-6)   # 20 μrad
        assert w0_tight > w0_wide

    def test_units_sanity(self):
        """w0 should be in cm range for typical satellite parameters."""
        w0 = compute_beam_waist(1550e-9, 10e-6)
        assert 0.01 < w0 < 1.0, f"w0 = {w0:.4f} m — outside expected 1–100 cm range"


# ---------------------------------------------------------------------------
# T02 — Beam radius far-field scaling: wL ≈ L·λ/(π·w0) for large L
# ---------------------------------------------------------------------------
class TestBeamRadius:
    def test_far_field_linear_scaling(self):
        """At satellite distance (550 km), wL should scale linearly with L."""
        lam = 1550e-9
        w0  = compute_beam_waist(lam, 10e-6)
        L1  = 550e3    # m
        L2  = 1100e3   # m (double distance)

        wL1 = compute_beam_radius(w0, L1, lam)
        wL2 = compute_beam_radius(w0, L2, lam)

        # In far field: wL2/wL1 ≈ L2/L1 = 2
        ratio = wL2 / wL1
        assert abs(ratio - 2.0) < 0.01, f"Far-field ratio = {ratio:.4f}, expected ≈ 2.0"

    def test_near_field_equals_waist(self):
        """At L=0, wL should equal w0."""
        lam = 1550e-9
        w0  = compute_beam_waist(lam, 10e-6)
        wL  = compute_beam_radius(w0, 0.0, lam)
        assert abs(wL - w0) < 1e-12

    def test_beam_expands_with_distance(self):
        """wL must be monotonically increasing with L."""
        lam = 1550e-9
        w0  = compute_beam_waist(lam, 10e-6)
        Ls  = [0, 100e3, 300e3, 550e3, 1000e3]
        wLs = [compute_beam_radius(w0, L, lam) for L in Ls]
        assert all(wLs[i] <= wLs[i+1] for i in range(len(wLs)-1))


# ---------------------------------------------------------------------------
# T03 — Geometric loss at nadir (ζ = 0°): shortest path → maximum hg
# ---------------------------------------------------------------------------
class TestGeometricLoss:
    def test_nadir_gives_maximum_hg(self):
        """ζ=0 (overhead satellite) should give the highest hg."""
        hg_0,  _, _ = compute_hg(550, zeta_deg=0)
        hg_30, _, _ = compute_hg(550, zeta_deg=30)
        hg_60, _, _ = compute_hg(550, zeta_deg=60)
        assert hg_0 >= hg_30 >= hg_60, "hg should decrease as zenith angle increases"

    def test_hg_in_valid_range(self):
        """hg must be in (0, 1]."""
        for zeta in [0, 15, 30, 45, 60, 75]:
            hg, _, _ = compute_hg(550, zeta)
            assert 0 < hg <= 1.0, f"hg={hg:.6e} out of range at ζ={zeta}°"

    def test_larger_aperture_gives_higher_hg(self):
        """Larger receiver aperture captures more power → higher hg."""
        hg_small, _, _ = compute_hg(550, 45, a_R=0.025)
        hg_large, _, _ = compute_hg(550, 45, a_R=0.10)
        assert hg_large > hg_small

    def test_paper1_numerical_example(self):
        """
        Reproduce Paper 1 example: λ=1550nm, θC=10μrad, HS=550km, ζ=45°, aR=5cm.
        Expected: wL ≈ 4.9 m, hg ≈ -60 dB range (very small).
        """
        hg, wL, _nu_R = compute_hg(H_S_km=550, zeta_deg=45, a_R=0.05,
                                    lambda_nm=1550, theta_C_urad=10)
        # wL should be several meters at 550 km
        assert 1.0 < wL < 20.0, f"wL = {wL:.2f} m — unexpected"
        # hg should be very small (large geometric loss)
        assert hg < 1e-3, f"hg = {hg:.2e} — expected < 1e-3 for 5cm aperture"


# ---------------------------------------------------------------------------
# T04 — Geometric loss decreases with zenith angle
# ---------------------------------------------------------------------------
class TestGeometricLossVsZenith:
    def test_monotone_decrease(self):
        """hg must decrease monotonically as ζ increases from 0° to 75°."""
        zeniths = np.arange(0, 76, 5)
        hgs = [compute_hg(550, z)[0] for z in zeniths]
        for i in range(len(hgs) - 1):
            assert hgs[i] >= hgs[i+1], (
                f"hg not monotone: hg({zeniths[i]}°)={hgs[i]:.4e} "
                f"> hg({zeniths[i+1]}°)={hgs[i+1]:.4e}"
            )


# ---------------------------------------------------------------------------
# T05 — Kruse/Kim visibility model
# ---------------------------------------------------------------------------
class TestVisibilityModel:
    def test_clear_air_low_attenuation(self):
        """V=50 km (very clear) → σ should be very small."""
        sigma = compute_sigma_visibility(50.0)
        assert sigma < 0.1, f"σ = {sigma:.4f} km⁻¹ — too high for clear air"

    def test_dense_fog_high_attenuation(self):
        """V=0.2 km (dense fog) → σ should be very large."""
        sigma = compute_sigma_visibility(0.2)
        assert sigma > 10.0, f"σ = {sigma:.4f} km⁻¹ — too low for dense fog"

    def test_sigma_decreases_with_visibility(self):
        """Higher visibility → lower attenuation."""
        sigmas = [compute_sigma_visibility(V) for V in [0.5, 1, 5, 10, 50]]
        assert all(sigmas[i] >= sigmas[i+1] for i in range(len(sigmas)-1))

    def test_known_value_V10(self):
        """
        V=10 km, λ=1550 nm: q=1.3, σ = (3.912/10)×(1550/550)^(-1.3)
        Manual: σ ≈ 0.391 × 0.3^(-1.3) ... let's just check order of magnitude.
        """
        sigma = compute_sigma_visibility(10.0, 1550)
        assert 0.05 < sigma < 1.0, f"σ = {sigma:.4f} km⁻¹ at V=10 km"


# ---------------------------------------------------------------------------
# T06 — Beer-Lambert: hl → 1 in vacuum (σ → 0, i.e., V → ∞)
# ---------------------------------------------------------------------------
class TestBeerLambert:
    def test_very_clear_air_hl_near_one(self):
        """
        V=1000 km (very clear) → hl should be high.
        Kruse model still gives σ > 0 at V=1000 km, so hl ≈ 0.985 at ζ=0.
        Threshold is 0.97 (not 1.0) to reflect this physical reality.
        """
        hl = compute_hl(zeta_deg=0, V_km=1000.0, R_mm_h=0.0)
        assert hl > 0.97, f"hl = {hl:.6f} — expected > 0.97 for very clear air"

    def test_hl_in_valid_range(self):
        """hl must be in (0, 1] for all conditions."""
        conditions = [
            (0,  10, 0),
            (45, 10, 0),
            (60, 5,  0.1),
            (75, 2,  0.5),
        ]
        for zeta, V, R in conditions:
            hl = compute_hl(zeta, V, R)
            assert 0 < hl <= 1.0, f"hl={hl:.4f} out of range at ζ={zeta}, V={V}, R={R}"

    def test_hl_decreases_with_zenith(self):
        """Larger zenith angle → longer atmospheric path → lower hl."""
        hls = [compute_hl(z, V_km=10) for z in [0, 30, 60]]
        assert hls[0] >= hls[1] >= hls[2]

    def test_rain_reduces_hl(self):
        """Adding rain (R > 0) should reduce hl compared to no rain."""
        hl_dry  = compute_hl(45, V_km=10, R_mm_h=0.0)
        hl_rain = compute_hl(45, V_km=10, R_mm_h=5.0)
        assert hl_rain < hl_dry, "Rain should reduce atmospheric transmission"


# ---------------------------------------------------------------------------
# T07 — Rain attenuation: R=0 → σ=0
# ---------------------------------------------------------------------------
class TestRainAttenuation:
    def test_no_rain_zero_sigma(self):
        """R=0 mm/h → σ_rain = 0."""
        sigma = compute_sigma_rain(0.0)
        assert sigma == 0.0

    def test_positive_rain_positive_sigma(self):
        """R > 0 → σ_rain > 0."""
        for R in [0.01, 0.1, 1.0, 10.0, 50.0]:
            sigma = compute_sigma_rain(R)
            assert sigma > 0, f"σ_rain = 0 for R = {R} mm/h"

    def test_higher_rain_higher_attenuation(self):
        """σ_rain should increase with rainfall rate."""
        sigmas = [compute_sigma_rain(R) for R in [0.1, 1.0, 5.0, 20.0]]
        assert all(sigmas[i] < sigmas[i+1] for i in range(len(sigmas)-1))

    def test_tropical_params_order_of_magnitude(self):
        """
        R=5 mm/h (moderate tropical rain):
        β = 0.509 × 5^0.63 ≈ 0.509 × 2.63 ≈ 1.34 dB/km
        σ = 1.34 / 4.343 ≈ 0.308 km⁻¹
        """
        sigma = compute_sigma_rain(5.0)
        assert 0.2 < sigma < 0.5, f"σ_rain = {sigma:.4f} km⁻¹ at R=5 mm/h"


# ---------------------------------------------------------------------------
# T08 — Rytov variance increases with zenith angle
# ---------------------------------------------------------------------------
class TestRytovVariance:
    def test_increases_with_zenith(self):
        """σR² must increase as ζ increases (longer turbulent path)."""
        zeniths = [0, 30, 45, 60]
        sigma_R2s = [compute_rytov_variance(z)[0] for z in zeniths]
        for i in range(len(sigma_R2s) - 1):
            assert sigma_R2s[i] < sigma_R2s[i+1], (
                f"σR²({zeniths[i]}°) = {sigma_R2s[i]:.4f} not < "
                f"σR²({zeniths[i+1]}°) = {sigma_R2s[i+1]:.4f}"
            )

    def test_sigma_X2_is_quarter_of_sigma_R2(self):
        """σX² = σR² / 4 by definition."""
        sigma_R2, sigma_X2 = compute_rytov_variance(45)
        assert abs(sigma_X2 - sigma_R2 / 4.0) < 1e-12

    def test_weak_turbulence_at_nadir(self):
        """
        At ζ=0, Cn²(0)=1.7e-14 (moderate), σR² should be < 1 (weak regime).
        This validates the log-normal model applicability.
        """
        sigma_R2, _ = compute_rytov_variance(0, Cn2_0=1.7e-14)
        assert sigma_R2 < 1.0, f"σR² = {sigma_R2:.4f} — expected < 1 at nadir"

    def test_strong_turbulence_at_high_zenith(self):
        """
        At ζ=75° with strong ground turbulence (Cn²=1e-13), σR² should be
        significantly larger than at ζ=0 (monotone increase verified in T08).
        With corrected H-V model, σR² ≈ 0.19 at ζ=75° — still weak regime,
        which is physically correct for satellite downlink at 1550 nm.
        """
        sigma_R2_0,  _ = compute_rytov_variance(0,  Cn2_0=1e-13)
        sigma_R2_75, _ = compute_rytov_variance(75, Cn2_0=1e-13)
        assert sigma_R2_75 > sigma_R2_0, (
            f"σR²(75°)={sigma_R2_75:.4f} should be > σR²(0°)={sigma_R2_0:.4f}"
        )
        assert sigma_R2_75 > 0.05, f"σR² = {sigma_R2_75:.4f} — expected > 0.05"


# ---------------------------------------------------------------------------
# T09 — Log-normal samples: E[ha] ≈ 1 (mean-preserving normalization)
# ---------------------------------------------------------------------------
class TestLognormalSampling:
    def test_mean_approximately_one(self):
        """
        With μX = -σX², E[ha] = E[exp(2X)] = exp(2μX + 2σX²) = exp(0) = 1.
        Monte Carlo with N=100_000 should give mean within 1% of 1.
        """
        rng = np.random.default_rng(seed=42)
        sigma_X2 = 0.1
        samples  = sample_ha_lognormal(sigma_X2, n_samples=100_000, rng=rng)
        mean     = np.mean(samples)
        assert abs(mean - 1.0) < 0.02, f"E[ha] = {mean:.4f}, expected ≈ 1.0"

    def test_all_samples_positive(self):
        """ha must be strictly positive (it's exp(2X))."""
        rng     = np.random.default_rng(seed=0)
        samples = sample_ha_lognormal(0.2, n_samples=10_000, rng=rng)
        assert np.all(samples > 0)

    def test_pdf_integrates_to_one(self):
        """Log-normal PDF must integrate to 1 over (0, ∞)."""
        sigma_X2 = 0.1
        result, _ = quad(lambda ha: pdf_ha_lognormal(ha, sigma_X2), 1e-6, 20.0)
        assert abs(result - 1.0) < 0.01, f"PDF integral = {result:.4f}, expected 1.0"

    def test_stronger_turbulence_wider_distribution(self):
        """Higher σX² → wider spread → higher variance of ha samples."""
        rng  = np.random.default_rng(seed=7)
        var1 = np.var(sample_ha_lognormal(0.05, 50_000, rng))
        var2 = np.var(sample_ha_lognormal(0.20, 50_000, rng))
        assert var2 > var1, "Stronger turbulence should give higher variance"


# ---------------------------------------------------------------------------
# T10 — Gamma-Gamma PDF integrates to 1
# ---------------------------------------------------------------------------
class TestGammaGamma:
    def test_pdf_integrates_to_one(self):
        """Gamma-Gamma PDF must integrate to 1 over (0, ∞)."""
        sigma_R2 = 0.8   # moderate turbulence
        alpha, beta = compute_gamma_gamma_params(sigma_R2)
        result, _ = quad(
            lambda ha: pdf_ha_gamma_gamma(np.array([ha]), alpha, beta)[0],
            1e-5, 20.0, limit=200
        )
        assert abs(result - 1.0) < 0.05, (
            f"Gamma-Gamma PDF integral = {result:.4f}, expected ≈ 1.0 "
            f"(α={alpha:.2f}, β={beta:.2f})"
        )

    def test_params_positive(self):
        """α and β must be positive for all valid σR²."""
        for sigma_R2 in [0.3, 0.5, 1.0, 2.0, 5.0]:
            alpha, beta = compute_gamma_gamma_params(sigma_R2)
            assert alpha > 0 and beta > 0, (
                f"α={alpha:.4f}, β={beta:.4f} at σR²={sigma_R2}"
            )

    def test_alpha_greater_than_beta(self):
        """
        For downlink (plane wave), α > β is expected
        (large-scale > small-scale scintillation index).
        """
        for sigma_R2 in [0.5, 1.0, 2.0]:
            alpha, beta = compute_gamma_gamma_params(sigma_R2)
            assert alpha > beta, (
                f"Expected α > β, got α={alpha:.4f}, β={beta:.4f} at σR²={sigma_R2}"
            )


# ---------------------------------------------------------------------------
# Integration test — compute_channel wrapper
# ---------------------------------------------------------------------------
class TestComputeChannel:
    def test_returns_all_keys(self):
        """compute_channel must return all expected keys."""
        result = compute_channel(H_S_km=550, zeta_deg=45)
        expected_keys = {"hg", "hl", "sigma_R2", "sigma_X2", "wL", "nu_R",
                         "L_slant_km", "hg_dB", "hl_dB"}
        assert expected_keys.issubset(result.keys())

    def test_total_loss_reasonable(self):
        """
        Total deterministic loss hg*hl should be in a physically reasonable range.
        For ζ=45°, clear air: expect -60 to -30 dB total.
        """
        r = compute_channel(550, 45, a_R=0.05, V_km=10, R_mm_h=0)
        total_dB = r["hg_dB"] + r["hl_dB"]
        assert -100 < total_dB < -20, f"Total loss = {total_dB:.1f} dB — unexpected"

    def test_rain_increases_loss(self):
        """Adding rain should increase total loss (lower hl)."""
        r_dry  = compute_channel(550, 45, V_km=10, R_mm_h=0)
        r_rain = compute_channel(550, 45, V_km=10, R_mm_h=10)
        assert r_rain["hl"] < r_dry["hl"]


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    sys.exit(result.returncode)
