"""Tests for modules/sikd_powersplit.py (Tasks 8-9, plan 07-5) — WRITTEN FIRST (TDD).

Contract:
    crosstalk_variance(m_D, hg, hl, ...) -> float
        RF leakage of the classical channel into the QKD receiver;
        must scale with m_D^2 (vu2022dtdd eq. for sigma_CT^2).
    evaluate_split(m_K, m_D, hg, hl, sigma_X2) -> dict
        {SKR_kbps, data_ber, data_throughput_gbps}
    pareto_frontier(channel_state, n_grid=...) -> list[dict]
        Non-dominated (SKR, data throughput) points, sorted by SKR asc.
    adaptive_split(K_req_bits, T_pass_s, channel_state) -> (m_K, m_D) | None
        Smallest m_K meeting the key demand, rest of the modulation
        budget to data; None if infeasible even at max m_K.

Channel states for testing use the verified clear-sky numbers:
    zenith 45°: hg ~ -35.9 dB, hl ~ -12.5 dB, sigma_X2 small (weak turb).
"""
import math
import pytest

ps = pytest.importorskip("modules.sikd_powersplit",
                         reason="modules/sikd_powersplit.py not implemented yet (Task 8)")

# Representative clear-sky channel at zenith 45° (verified Table II numbers).
HG = 10 ** (-35.9 / 10)
HL = 10 ** (-12.5 / 10)
SX2 = 0.01


class TestCrosstalkScaling:
    def test_quadratic_in_mD(self):
        c1 = ps.crosstalk_variance(0.3, HG, HL)
        c2 = ps.crosstalk_variance(0.6, HG, HL)
        assert c2 / c1 == pytest.approx(4.0, rel=1e-6)

    def test_zero_data_channel_zero_crosstalk(self):
        assert ps.crosstalk_variance(0.0, HG, HL) == pytest.approx(0.0, abs=1e-30)


class TestEvaluateSplit:
    def test_returns_required_keys(self):
        out = ps.evaluate_split(0.05, 0.5, HG, HL, SX2)
        assert {"SKR_kbps", "data_ber", "data_throughput_gbps"} <= set(out)

    def test_default_split_matches_existing_pipeline(self):
        # m_K=0.05, m_D=0.5 is the current system operating point; SKR must
        # be within 10% of the verified 13,280 kbps at zenith 45° clear sky.
        out = ps.evaluate_split(0.05, 0.5, HG, HL, SX2)
        assert out["SKR_kbps"] == pytest.approx(13280, rel=0.10)

    def test_more_key_power_no_worse_skr_at_low_crosstalk(self):
        lo = ps.evaluate_split(0.02, 0.5, HG, HL, SX2)
        hi = ps.evaluate_split(0.10, 0.5, HG, HL, SX2)
        assert hi["SKR_kbps"] >= lo["SKR_kbps"]

    def test_more_data_power_degrades_qkd(self):
        # Raising m_D raises sigma_CT^2 quadratically -> SKR must not improve.
        quiet = ps.evaluate_split(0.05, 0.2, HG, HL, SX2)
        loud = ps.evaluate_split(0.05, 0.8, HG, HL, SX2)
        assert loud["SKR_kbps"] <= quiet["SKR_kbps"]

    def test_ber_is_probability(self):
        out = ps.evaluate_split(0.05, 0.5, HG, HL, SX2)
        assert 0.0 <= out["data_ber"] <= 0.5

    def test_rejects_budget_violation(self):
        with pytest.raises(ValueError):
            ps.evaluate_split(0.5, 0.6, HG, HL, SX2)  # m_K + m_D > 1


class TestParetoFrontier:
    def test_frontier_nonempty_and_sorted(self):
        front = ps.pareto_frontier({"hg": HG, "hl": HL, "sigma_X2": SX2})
        assert len(front) >= 3
        skrs = [p["SKR_kbps"] for p in front]
        assert skrs == sorted(skrs)

    def test_frontier_is_nondominated(self):
        front = ps.pareto_frontier({"hg": HG, "hl": HL, "sigma_X2": SX2})
        # Sorted by SKR ascending -> throughput must be strictly decreasing,
        # otherwise a point would dominate its neighbor.
        thr = [p["data_throughput_gbps"] for p in front]
        assert all(a > b for a, b in zip(thr, thr[1:]))


class TestAdaptiveSplit:
    CH = {"hg": HG, "hl": HL, "sigma_X2": SX2}

    def test_meets_key_demand(self):
        K_req = 100e6  # 100 Mbit over one pass
        T = 180.0
        split = ps.adaptive_split(K_req, T, self.CH)
        assert split is not None
        m_K, m_D = split
        out = ps.evaluate_split(m_K, m_D, HG, HL, SX2)
        assert out["SKR_kbps"] * 1e3 * T >= K_req

    def test_low_demand_gives_more_data_than_high_demand(self):
        T = 180.0
        lo = ps.adaptive_split(1e6, T, self.CH)
        hi = ps.adaptive_split(1e9, T, self.CH)
        if lo is not None and hi is not None:
            assert lo[1] >= hi[1]  # more modulation budget left for data

    def test_infeasible_demand_returns_none(self):
        # No split can deliver 1 Tbit of key in a 3-minute pass.
        assert ps.adaptive_split(1e12, 180.0, self.CH) is None
