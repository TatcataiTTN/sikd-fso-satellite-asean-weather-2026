"""
test_sikd_performance.py — Unit tests for sikd_performance.py
==============================================================

Kiểm tra tính đúng đắn của từng thành phần trong pipeline SIKD.

Chiến lược test
---------------
    - Correctness  : so sánh với giá trị tính tay từ Paper 1/2
    - Monotonicity : kiểm tra chiều tăng/giảm theo tham số vật lý
    - Range        : đảm bảo output nằm trong khoảng hợp lệ
    - Edge cases   : Iso→∞, mD=0, QBER=0, QBER=0.5
    - Physics      : trade-off mD↑ → BER_CC↓ nhưng crosstalk↑ → SKR↓

Test classes (30 tests tổng)
----------------------------
    T01  TestBinaryEntropy       (3 tests) — H(0)=0, H(0.5)=1, H(0.11)
    T02  TestNoiseModel          (6 tests) — thermal dominates, crosstalk scaling,
                                             Iso effect, mD effect, Pbg effect, range
    T03  TestThresholds          (4 tests) — symmetry, guard zone width, zeta effect,
                                             range
    T04  TestPsiftQBER           (6 tests) — Psift range, QBER range, QBER < 0.5,
                                             better channel → lower QBER,
                                             higher zeta → lower QBER, Psift > 0
    T05  TestSKR                 (5 tests) — SKR ≥ 0, QBER=0 → max SKR,
                                             QBER=0.5 → SKR=0, reconciliation effect,
                                             Psift scaling
    T06  TestBERCC               (3 tests) — BER ∈ [0, 0.5], higher mD → lower BER,
                                             stronger turbulence → higher BER
    T07  TestSIKDWrapper         (3 tests) — all keys present, higher Iso → higher SKR,
                                             mD trade-off

Chạy tests
----------
    python -m pytest test_sikd_performance.py -v
    python -m pytest test_sikd_performance.py -v -k "TestNoise"
"""

import sys
import os
import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.sikd_performance import (
    binary_entropy,
    compute_noise,
    compute_thresholds,
    compute_Psift_QBER,
    compute_SKR,
    compute_SKR_bps,
    compute_BER_CC,
    compute_sikd_performance,
    Q_ELECTRON,
    K_BOLTZMANN,
)

# ---------------------------------------------------------------------------
# Shared fixture — representative channel state (ζ=45°, clear air)
# Computed from channel_model.compute_channel(550, 45, V_km=10)
# hg ≈ 2.6e-4, hl ≈ 0.075, sigma_X2 ≈ 0.0067
# ---------------------------------------------------------------------------
HG_REF    = 2.6e-4
HL_REF    = 0.075
SX2_REF   = 0.0067
PT_REF    = 1e-3      # 1 mW
MK_REF    = 0.05
MD_REF    = 0.5
ISO_REF   = 40.0      # dB


# ---------------------------------------------------------------------------
# T01 — Binary entropy
# ---------------------------------------------------------------------------
class TestBinaryEntropy:
    def test_zero_probability(self):
        """H(0) = 0 — no uncertainty."""
        assert binary_entropy(0.0) == 0.0

    def test_one_probability(self):
        """H(1) = 0 — no uncertainty."""
        assert binary_entropy(1.0) == 0.0

    def test_half_probability_is_one_bit(self):
        """H(0.5) = 1 — maximum entropy."""
        assert abs(binary_entropy(0.5) - 1.0) < 1e-10

    def test_bb84_qber_threshold(self):
        """
        BB84 security threshold: QBER = 11% → H(0.11) ≈ 0.5.
        At this point I(A;B) = 1 - H(0.11) ≈ 0.5 — still positive.
        """
        h = binary_entropy(0.11)
        assert 0.4 < h < 0.6, f"H(0.11) = {h:.4f}, expected ≈ 0.5"

    def test_small_qber_near_zero_entropy(self):
        """QBER = 0.001 → H ≈ 0.014 (very small)."""
        h = binary_entropy(0.001)
        assert 0.0 < h < 0.05

    def test_symmetry(self):
        """H(p) = H(1-p) by definition."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert abs(binary_entropy(p) - binary_entropy(1.0 - p)) < 1e-12


# ---------------------------------------------------------------------------
# T02 — Noise model
# ---------------------------------------------------------------------------
class TestNoiseModel:
    def test_all_components_positive(self):
        """All noise variances must be ≥ 0."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        assert n["sigma2_shot"]    >= 0
        assert n["sigma2_thermal"] >= 0
        assert n["sigma2_bg"]      >= 0
        assert n["sigma2_CT"]      >= 0
        assert n["sigma2_total"]   >  0

    def test_total_equals_sum_of_components(self):
        """σ²_total = σ²_shot + σ²_thermal + σ²_bg + σ²_CT."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        expected = (n["sigma2_shot"] + n["sigma2_thermal"]
                    + n["sigma2_bg"] + n["sigma2_CT"])
        assert abs(n["sigma2_total"] - expected) < 1e-30

    def test_sigma_N_is_sqrt_of_total(self):
        """σ_N = √(σ²_total)."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        assert abs(n["sigma_N"] - np.sqrt(n["sigma2_total"])) < 1e-20

    def test_higher_Iso_lower_crosstalk(self):
        """Higher filter isolation → lower crosstalk variance."""
        n40 = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF, Iso_dB=40)
        n60 = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF, Iso_dB=60)
        assert n60["sigma2_CT"] < n40["sigma2_CT"]

    def test_crosstalk_scales_with_mD_squared(self):
        """
        σ²_CT = (PT² · mD² · 10^(-Iso/10)) / 2
        → doubling mD quadruples σ²_CT.
        """
        n1 = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF, mD=0.3, Iso_dB=40)
        n2 = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF, mD=0.6, Iso_dB=40)
        ratio = n2["sigma2_CT"] / n1["sigma2_CT"]
        assert abs(ratio - 4.0) < 0.01, f"CT ratio = {ratio:.4f}, expected 4.0"

    def test_thermal_noise_formula(self):
        """
        σ²_thermal = 4·kB·T·Δf / RL
        Manual: T=280K, RL=1kΩ, Rb=1Gbps → Δf=500MHz
        σ²_thermal = 4 × 1.381e-23 × 280 × 500e6 / 1000 ≈ 7.73e-12 A²
        """
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF,
                          T=280, RL=1e3, Rb=1e9)
        expected = 4 * K_BOLTZMANN * 280 * (1e9 / 2) / 1e3
        assert abs(n["sigma2_thermal"] - expected) / expected < 1e-6

    def test_no_background_zero_bg_noise(self):
        """Pbg=0 → σ²_bg = 0."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF, Pbg=0.0)
        assert n["sigma2_bg"] == 0.0

    def test_infinite_isolation_zero_crosstalk(self):
        """Iso=200 dB → σ²_CT = PT²·mD²·10^(-20)/2 ≈ 1.25e-27 ≈ negligible."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF, Iso_dB=200)
        assert n["sigma2_CT"] < 1e-25


# ---------------------------------------------------------------------------
# T03 — Dual-threshold detection
# ---------------------------------------------------------------------------
class TestThresholds:
    def test_d1_greater_than_d0(self):
        """d1 > d0 always (guard zone must be non-empty)."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        t = compute_thresholds(HG_REF, HL_REF, PT_REF, MK_REF, n["sigma_N"])
        assert t["d1"] > t["d0"], f"d1={t['d1']:.4e} ≤ d0={t['d0']:.4e}"

    def test_symmetry_around_zero(self):
        """
        d0 = -d1 by symmetry (i_mean_pos = -i_mean_neg, same ζ·σN offset).
        """
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        t = compute_thresholds(HG_REF, HL_REF, PT_REF, MK_REF, n["sigma_N"])
        assert abs(t["d0"] + t["d1"]) < 1e-20, (
            f"d0 + d1 = {t['d0'] + t['d1']:.4e}, expected ≈ 0"
        )

    def test_larger_zeta_wider_guard_zone(self):
        """Larger ζ_scale → wider guard zone [d0, d1]."""
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        t1 = compute_thresholds(HG_REF, HL_REF, PT_REF, MK_REF, n["sigma_N"],
                                 zeta_scale=1.0)
        t2 = compute_thresholds(HG_REF, HL_REF, PT_REF, MK_REF, n["sigma_N"],
                                 zeta_scale=3.0)
        assert t2["guard_width"] > t1["guard_width"]

    def test_guard_width_formula(self):
        """
        guard_width = d1 - d0 = 2·(i_mean + ζ·σN).
        """
        n = compute_noise(HG_REF, HL_REF, PT_REF, MK_REF)
        zeta = 2.0
        t = compute_thresholds(HG_REF, HL_REF, PT_REF, MK_REF, n["sigma_N"],
                                zeta_scale=zeta)
        i_mean = 0.5 * 0.9 * PT_REF * MK_REF * HG_REF * HL_REF  # Paper 2 Eq. 23: coefficient 1/2
        expected_width = 2.0 * (i_mean + zeta * n["sigma_N"])
        assert abs(t["guard_width"] - expected_width) / expected_width < 1e-6


# ---------------------------------------------------------------------------
# T04 — Psift and QBER
# ---------------------------------------------------------------------------
class TestPsiftQBER:
    def _get_qkd(self, hg=HG_REF, hl=HL_REF, sx2=SX2_REF,
                 zeta_scale=2.0, Iso_dB=ISO_REF):
        n = compute_noise(hg, hl, PT_REF, MK_REF, Iso_dB=Iso_dB)
        t = compute_thresholds(hg, hl, PT_REF, MK_REF, n["sigma_N"], zeta_scale)
        return compute_Psift_QBER(hg, hl, sx2, n["sigma_N"],
                                  t["d0"], t["d1"], PT_REF, MK_REF)

    def test_Psift_in_valid_range(self):
        """Psift ∈ (0, 0.5] — cannot exceed 0.5 (each bit has 50% prior)."""
        r = self._get_qkd()
        assert 0 < r["Psift"] <= 0.5, f"Psift = {r['Psift']:.4e}"

    def test_QBER_in_valid_range(self):
        """QBER ∈ [0, 0.5]."""
        r = self._get_qkd()
        assert 0 <= r["QBER"] <= 0.5, f"QBER = {r['QBER']:.4e}"

    def test_QBER_less_than_half(self):
        """QBER < 0.5 for any reasonable channel (not pure noise)."""
        r = self._get_qkd()
        assert r["QBER"] < 0.5

    def test_better_channel_lower_QBER(self):
        """
        Better channel (higher hl) → stronger signal → lower QBER.
        """
        r_good = self._get_qkd(hl=0.15)   # better attenuation
        r_bad  = self._get_qkd(hl=0.03)   # worse attenuation
        assert r_good["QBER"] <= r_bad["QBER"], (
            f"QBER(good)={r_good['QBER']:.4e} > QBER(bad)={r_bad['QBER']:.4e}"
        )

    def test_higher_zeta_lower_QBER(self):
        """
        Larger ζ_scale → wider guard zone → fewer errors → lower QBER.
        (Trade-off: Psift also decreases.)
        """
        r1 = self._get_qkd(zeta_scale=1.0)
        r2 = self._get_qkd(zeta_scale=3.0)
        assert r2["QBER"] <= r1["QBER"], (
            f"QBER(ζ=3)={r2['QBER']:.4e} > QBER(ζ=1)={r1['QBER']:.4e}"
        )

    def test_higher_zeta_lower_Psift(self):
        """
        Larger ζ_scale → wider guard zone → more discards → lower Psift.
        """
        r1 = self._get_qkd(zeta_scale=1.0)
        r2 = self._get_qkd(zeta_scale=3.0)
        assert r2["Psift"] <= r1["Psift"]

    def test_QBER_equals_Perror_over_Psift(self):
        """QBER = Perror / Psift by definition."""
        r = self._get_qkd()
        if r["Psift"] > 0:
            expected = r["Perror"] / r["Psift"]
            assert abs(r["QBER"] - expected) < 1e-10

    def test_higher_Iso_lower_QBER(self):
        """Higher filter isolation → less crosstalk → lower QBER."""
        r_low  = self._get_qkd(Iso_dB=20)
        r_high = self._get_qkd(Iso_dB=60)
        assert r_high["QBER"] <= r_low["QBER"]


# ---------------------------------------------------------------------------
# T05 — Secret Key Rate
# ---------------------------------------------------------------------------
class TestSKR:
    def test_SKR_nonnegative(self):
        """SKR ≥ 0 always."""
        for qber in [0.0, 0.001, 0.1, 0.5, 0.9]:
            skr = compute_SKR(1e-3, qber)
            assert skr >= 0.0, f"SKR < 0 at QBER={qber}"

    def test_zero_QBER_maximum_IAB(self):
        """QBER=0 → H(QBER)=0 → I(A;B)=1 → SKR = Psift."""
        Psift = 1e-3
        skr = compute_SKR(Psift, QBER=0.0, chi_E=0.0)
        assert abs(skr - Psift) < 1e-12, f"SKR={skr:.6e}, expected {Psift:.6e}"

    def test_high_QBER_zero_SKR(self):
        """QBER = 0.5 → H(0.5)=1 → I(A;B)=0 → SKR=0."""
        skr = compute_SKR(1e-3, QBER=0.5)
        assert skr == 0.0

    def test_SKR_scales_with_Psift(self):
        """Doubling Psift should double SKR (linear relationship)."""
        skr1 = compute_SKR(1e-3, 0.001)
        skr2 = compute_SKR(2e-3, 0.001)
        assert abs(skr2 / skr1 - 2.0) < 1e-6

    def test_reconciliation_reduces_SKR(self):
        """β < 1 (imperfect reconciliation) → lower SKR."""
        skr_perfect = compute_SKR(1e-3, 0.001, reconciliation_eff=1.0)
        skr_imperfect = compute_SKR(1e-3, 0.001, reconciliation_eff=0.9)
        assert skr_imperfect < skr_perfect

    def test_SKR_bps_equals_norm_times_Rb(self):
        """SKR_bps = SKR_norm × Rb."""
        Psift, QBER, Rb = 1e-3, 0.001, 1e9
        skr_norm = compute_SKR(Psift, QBER)
        skr_bps  = compute_SKR_bps(Psift, QBER, Rb)
        assert abs(skr_bps - skr_norm * Rb) < 1e-6

    def test_chi_E_reduces_SKR(self):
        """Positive χ(A:E) (Eve has information) → lower SKR."""
        skr_no_eve = compute_SKR(1e-3, 0.001, chi_E=0.0)
        skr_eve    = compute_SKR(1e-3, 0.001, chi_E=0.1)
        assert skr_eve < skr_no_eve


# ---------------------------------------------------------------------------
# T06 — BER classical data channel
# ---------------------------------------------------------------------------
class TestBERCC:
    def test_BER_in_valid_range(self):
        """BER ∈ [0, 0.5]."""
        ber = compute_BER_CC(HG_REF, HL_REF, SX2_REF, PT_REF, MD_REF,
                             sigma_N_D=1e-9)
        assert 0 <= ber <= 0.5, f"BER = {ber:.4e}"

    def test_higher_mD_lower_BER(self):
        """
        Higher mD → stronger data signal → better SNR → lower BER.
        (Assuming noise is dominated by thermal, not crosstalk.)
        """
        ber_low  = compute_BER_CC(HG_REF, HL_REF, SX2_REF, PT_REF,
                                   mD=0.3, sigma_N_D=1e-9)
        ber_high = compute_BER_CC(HG_REF, HL_REF, SX2_REF, PT_REF,
                                   mD=0.8, sigma_N_D=1e-9)
        assert ber_high < ber_low

    def test_stronger_turbulence_higher_BER(self):
        """Higher σX² → more fading → higher average BER."""
        ber_weak   = compute_BER_CC(HG_REF, HL_REF, 0.01, PT_REF, MD_REF,
                                    sigma_N_D=1e-9)
        ber_strong = compute_BER_CC(HG_REF, HL_REF, 0.10, PT_REF, MD_REF,
                                    sigma_N_D=1e-9)
        assert ber_strong > ber_weak

    def test_very_high_SNR_near_zero_BER(self):
        """Very low noise → BER ≈ 0."""
        ber = compute_BER_CC(HG_REF, HL_REF, SX2_REF, PT_REF, MD_REF,
                             sigma_N_D=1e-15)
        assert ber < 1e-6, f"BER = {ber:.4e} — expected near 0 for high SNR"

    def test_very_low_SNR_near_half_BER(self):
        """Very high noise → BER → 0.5 (random guessing)."""
        ber = compute_BER_CC(HG_REF, HL_REF, SX2_REF, PT_REF, MD_REF,
                             sigma_N_D=1e3)
        assert ber > 0.4, f"BER = {ber:.4f} — expected ≈ 0.5 for very low SNR"


# ---------------------------------------------------------------------------
# T07 — Full SIKD wrapper
# ---------------------------------------------------------------------------
class TestSIKDWrapper:
    def test_returns_all_keys(self):
        """compute_sikd_performance must return all expected keys."""
        result = compute_sikd_performance(HG_REF, HL_REF, SX2_REF)
        expected = {"SKR_norm", "SKR_kbps", "QBER", "Psift",
                    "BER_CC", "sigma_N", "d0", "d1", "I_AB", "noise"}
        assert expected.issubset(result.keys())

    def test_higher_Iso_higher_SKR(self):
        """
        Higher filter isolation → less crosstalk → lower QBER → higher SKR.
        """
        r_low  = compute_sikd_performance(HG_REF, HL_REF, SX2_REF, Iso_dB=20)
        r_high = compute_sikd_performance(HG_REF, HL_REF, SX2_REF, Iso_dB=60)
        assert r_high["SKR_norm"] >= r_low["SKR_norm"]

    def test_mD_tradeoff(self):
        """
        mD↑ → BER_CC↓ (good for data) but crosstalk↑ → QBER↑ → SKR↓.
        At Iso=40dB, this trade-off should be visible.
        """
        r_low  = compute_sikd_performance(HG_REF, HL_REF, SX2_REF,
                                          mD=0.2, Iso_dB=40)
        r_high = compute_sikd_performance(HG_REF, HL_REF, SX2_REF,
                                          mD=0.8, Iso_dB=40)
        # Higher mD → lower BER_CC
        assert r_high["BER_CC"] <= r_low["BER_CC"]
        # Higher mD → higher crosstalk → higher QBER (at low Iso)
        assert r_high["QBER"] >= r_low["QBER"]

    def test_SKR_kbps_consistent_with_norm(self):
        """SKR_kbps = SKR_norm × Rb / 1000."""
        r = compute_sikd_performance(HG_REF, HL_REF, SX2_REF, Iso_dB=40)
        expected_kbps = r["SKR_norm"] * 1e9 / 1e3
        assert abs(r["SKR_kbps"] - expected_kbps) < 1e-6

    def test_I_AB_consistent_with_QBER(self):
        """I(A;B) = 1 - H(QBER)."""
        r = compute_sikd_performance(HG_REF, HL_REF, SX2_REF)
        from modules.sikd_performance import binary_entropy
        expected = 1.0 - binary_entropy(r["QBER"])
        assert abs(r["I_AB"] - expected) < 1e-10


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
