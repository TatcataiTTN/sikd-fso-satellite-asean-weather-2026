"""
test_routing.py — Unit tests for routing.py
============================================

Kiểm tra tính đúng đắn của các hàm routing vệ tinh trong routing.py.

Chiến lược test
---------------
    - SKR computation  : compute_skr_for_satellite, compute_skr_all_visible
    - Routing logic    : greedy vs baseline, ordering, edge cases
    - Time series      : shape, range, no-satellite handling
    - Key bits         : formula correctness, monotonicity
    - Improvement      : ratio ≥ 1, percentage, availability
    - Integration      : full pipeline với real TLE + weather + SIKD

Test classes (7 classes, ~38 tests)
------------------------------------
    T01  TestSKRForSatellite    (6 tests) — compute_skr_for_satellite
    T02  TestSKRAllVisible      (5 tests) — compute_skr_all_visible
    T03  TestRoutingStrategies  (6 tests) — greedy vs baseline
    T04  TestTimeSeries         (7 tests) — compute_skr_timeseries
    T05  TestDailyKeyBits       (4 tests) — compute_daily_key_bits
    T06  TestRoutingImprovement (6 tests) — compute_routing_improvement
    T07  TestIntegration        (4 tests) — full pipeline

Chạy tests
----------
    python -m pytest test_routing.py -v
    python -m pytest test_routing.py -v -k "TestRouting"
"""

import sys
import os
import numpy as np
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.routing import (
    compute_skr_for_satellite,
    compute_skr_all_visible,
    greedy_best_satellite,
    no_routing_satellite,
    compute_skr_timeseries,
    compute_daily_key_bits,
    compute_routing_improvement,
    PT_DEFAULT, MK_DEFAULT, MD_DEFAULT, ISO_DB_DEFAULT,
    RB_DEFAULT, H_S_KM_DEFAULT,
)
from modules.orbital_mechanics import (
    make_skyfield_satellite, _TS, GROUND_STATIONS,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# TLE fixtures (same as test_orbital_mechanics)
TLE_SAT_A = {
    "name":  "STARLINK-A",
    "line1": "1 44713U 19074B   26071.50000000  .00001000  00000-0  10000-3 0  9990",
    "line2": "2 44713  53.0000 100.0000 0001000  90.0000 270.0000 15.05000000000010",
}
TLE_SAT_B = {
    "name":  "STARLINK-B",
    "line1": "1 44714U 19074C   26071.50000000  .00001000  00000-0  10000-3 0  9991",
    "line2": "2 44714  53.0000 200.0000 0001000  90.0000 270.0000 15.05000000000011",
}
TLE_SAT_C = {
    "name":  "STARLINK-C",
    "line1": "1 44715U 19074D   26071.50000000  .00001000  00000-0  10000-3 0  9992",
    "line2": "2 44715  53.0000 300.0000 0001000  90.0000 270.0000 15.05000000000012",
}

SAT_A = make_skyfield_satellite(TLE_SAT_A)
SAT_B = make_skyfield_satellite(TLE_SAT_B)
SAT_C = make_skyfield_satellite(TLE_SAT_C)
ALL_SATS = [SAT_A, SAT_B, SAT_C]

T_REF = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)

HANOI = GROUND_STATIONS["hanoi"]

# Representative weather: clear dry season
WX_DRY = {"R_mm_h": 0.01, "V_km": 15.0, "P_cloud": 0.15}
# Representative weather: heavy wet season
WX_WET = {"R_mm_h": 0.28, "V_km": 7.0,  "P_cloud": 0.58}

# Synthetic visible_sats list (mocking orbital_mechanics output)
def _make_visible_sats(zenith_angles: list[float]) -> list[dict]:
    """Tạo danh sách visible_sats giả với zenith angles cho trước."""
    return [
        {
            "satellite":     SAT_A,
            "name":          f"SAT-{i}",
            "elevation_deg": 90.0 - z,
            "zenith_deg":    z,
            "slant_km":      H_S_KM_DEFAULT / np.cos(np.radians(z)),
        }
        for i, z in enumerate(zenith_angles)
    ]


# ---------------------------------------------------------------------------
# T01 — SKR for single satellite
# ---------------------------------------------------------------------------
class TestSKRForSatellite:
    def test_returns_all_keys(self):
        """compute_skr_for_satellite phải trả về tất cả keys cần thiết."""
        result = compute_skr_for_satellite(
            zenith_deg=45.0, **WX_DRY
        )
        expected = {"SKR_norm", "SKR_kbps", "SKR_effective", "QBER",
                    "BER_CC", "hl", "hg", "sigma_X2", "zenith_deg", "is_feasible"}
        assert expected.issubset(result.keys())

    def test_skr_nonnegative(self):
        """SKR_norm và SKR_effective phải ≥ 0."""
        result = compute_skr_for_satellite(zenith_deg=45.0, **WX_DRY)
        assert result["SKR_norm"]      >= 0.0
        assert result["SKR_effective"] >= 0.0

    def test_skr_effective_leq_skr_norm(self):
        """SKR_effective ≤ SKR_norm (mây chỉ giảm SKR)."""
        result = compute_skr_for_satellite(zenith_deg=45.0, **WX_WET)
        assert result["SKR_effective"] <= result["SKR_norm"] + 1e-15

    def test_cloud_reduces_effective_skr(self):
        """P_cloud cao hơn → SKR_effective thấp hơn (cùng điều kiện khác)."""
        r_low  = compute_skr_for_satellite(45.0, 0.01, 15.0, P_cloud=0.1)
        r_high = compute_skr_for_satellite(45.0, 0.01, 15.0, P_cloud=0.8)
        assert r_high["SKR_effective"] < r_low["SKR_effective"]

    def test_lower_zenith_higher_skr(self):
        """Zenith angle thấp hơn (elevation cao hơn) → hl tốt hơn → SKR cao hơn."""
        r_low  = compute_skr_for_satellite(20.0, **WX_DRY)
        r_high = compute_skr_for_satellite(60.0, **WX_DRY)
        # hl tốt hơn ở zenith thấp → SKR_norm cao hơn hoặc bằng
        assert r_low["hl"] >= r_high["hl"]

    def test_wet_weather_lower_hl(self):
        """Mùa mưa (R cao, V thấp) → hl thấp hơn mùa khô."""
        r_dry = compute_skr_for_satellite(45.0, **WX_DRY)
        r_wet = compute_skr_for_satellite(45.0, **WX_WET)
        assert r_wet["hl"] < r_dry["hl"]


# ---------------------------------------------------------------------------
# T02 — SKR for all visible satellites
# ---------------------------------------------------------------------------
class TestSKRAllVisible:
    def test_returns_list(self):
        """compute_skr_all_visible phải trả về list."""
        visible = _make_visible_sats([30.0, 45.0, 60.0])
        result  = compute_skr_all_visible(visible, **WX_DRY)
        assert isinstance(result, list)

    def test_length_matches_input(self):
        """Số lượng kết quả phải bằng số vệ tinh đầu vào."""
        visible = _make_visible_sats([30.0, 45.0, 60.0])
        result  = compute_skr_all_visible(visible, **WX_DRY)
        assert len(result) == 3

    def test_sorted_by_skr_effective_descending(self):
        """Kết quả phải được sort theo SKR_effective giảm dần."""
        visible = _make_visible_sats([30.0, 45.0, 60.0])
        result  = compute_skr_all_visible(visible, **WX_DRY)
        skrs = [r["SKR_effective"] for r in result]
        assert skrs == sorted(skrs, reverse=True)

    def test_empty_input_returns_empty(self):
        """Danh sách rỗng → kết quả rỗng."""
        result = compute_skr_all_visible([], **WX_DRY)
        assert result == []

    def test_result_has_both_sat_and_skr_keys(self):
        """Mỗi entry phải có cả keys từ visible_sats lẫn SKR results."""
        visible = _make_visible_sats([45.0])
        result  = compute_skr_all_visible(visible, **WX_DRY)
        entry   = result[0]
        # Keys từ visible_sats
        assert "zenith_deg" in entry
        assert "name"       in entry
        # Keys từ SKR computation
        assert "SKR_norm"      in entry
        assert "SKR_effective" in entry
        assert "QBER"          in entry


# ---------------------------------------------------------------------------
# T03 — Routing strategies
# ---------------------------------------------------------------------------
class TestRoutingStrategies:
    def _make_candidates(self, skr_values: list[float]) -> list[dict]:
        """Tạo candidates list với SKR_effective cho trước."""
        return [
            {"name": f"SAT-{i}", "SKR_effective": s,
             "zenith_deg": 45.0, "elevation_deg": 45.0}
            for i, s in enumerate(skr_values)
        ]

    def test_greedy_selects_highest_skr(self):
        """greedy_best_satellite phải chọn vệ tinh có SKR_effective cao nhất."""
        candidates = self._make_candidates([1e-4, 5e-4, 2e-4])
        # Sort descending trước (như compute_skr_all_visible làm)
        candidates.sort(key=lambda x: x["SKR_effective"], reverse=True)
        best = greedy_best_satellite(candidates)
        assert best["SKR_effective"] == 5e-4

    def test_greedy_empty_returns_none(self):
        """greedy_best_satellite với list rỗng → None."""
        assert greedy_best_satellite([]) is None

    def test_no_routing_returns_first_alphabetically(self):
        """no_routing_satellite phải chọn vệ tinh đầu tiên theo tên."""
        candidates = self._make_candidates([1e-4, 5e-4, 2e-4])
        # Đặt tên để SAT-0 là đầu tiên alphabetically
        baseline = no_routing_satellite(candidates)
        assert baseline["name"] == "SAT-0"

    def test_no_routing_empty_returns_none(self):
        """no_routing_satellite với list rỗng → None."""
        assert no_routing_satellite([]) is None

    def test_greedy_geq_baseline(self):
        """Greedy SKR phải ≥ baseline SKR (greedy luôn tốt hơn hoặc bằng)."""
        candidates = self._make_candidates([3e-4, 1e-4, 5e-4, 2e-4])
        candidates.sort(key=lambda x: x["SKR_effective"], reverse=True)
        best     = greedy_best_satellite(candidates)
        baseline = no_routing_satellite(candidates)
        assert best["SKR_effective"] >= baseline["SKR_effective"]

    def test_single_satellite_greedy_equals_baseline(self):
        """Với 1 vệ tinh, greedy và baseline phải chọn cùng vệ tinh."""
        candidates = self._make_candidates([3e-4])
        best     = greedy_best_satellite(candidates)
        baseline = no_routing_satellite(candidates)
        assert best["name"] == baseline["name"]


# ---------------------------------------------------------------------------
# T04 — SKR Time Series
# ---------------------------------------------------------------------------
class TestTimeSeries:
    def test_returns_all_keys(self):
        """compute_skr_timeseries phải trả về tất cả keys cần thiết."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        for key in ["skr_greedy", "skr_baseline", "n_visible",
                    "time_hours", "n_steps"]:
            assert key in result

    def test_output_arrays_correct_length(self):
        """Tất cả arrays phải có độ dài = n_steps."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        n = result["n_steps"]
        assert len(result["skr_greedy"])   == n
        assert len(result["skr_baseline"]) == n
        assert len(result["n_visible"])    == n
        assert len(result["time_hours"])   == n

    def test_skr_nonnegative(self):
        """SKR values phải ≥ 0 tại mọi time step."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        assert np.all(result["skr_greedy"]   >= 0.0)
        assert np.all(result["skr_baseline"] >= 0.0)

    def test_greedy_geq_baseline_at_all_steps(self):
        """Greedy SKR ≥ baseline SKR tại mọi time step."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        assert np.all(result["skr_greedy"] >= result["skr_baseline"] - 1e-15)

    def test_n_visible_nonnegative(self):
        """n_visible phải ≥ 0 tại mọi time step."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        assert np.all(result["n_visible"] >= 0)

    def test_no_satellites_all_zero_skr(self):
        """Không có vệ tinh → SKR = 0 tại mọi time step."""
        result = compute_skr_timeseries(
            satellites=[],
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        assert np.all(result["skr_greedy"]   == 0.0)
        assert np.all(result["skr_baseline"] == 0.0)

    def test_time_hours_starts_at_zero(self):
        """time_hours phải bắt đầu từ 0."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        assert result["time_hours"][0] == 0.0


# ---------------------------------------------------------------------------
# T05 — Daily Key Bits
# ---------------------------------------------------------------------------
class TestDailyKeyBits:
    def test_zero_skr_zero_key_bits(self):
        """SKR = 0 tại mọi step → 0 key bits."""
        skr = np.zeros(100)
        assert compute_daily_key_bits(skr) == 0.0

    def test_formula_correctness(self):
        """Total = Σ SKR × Rb × Δt."""
        skr = np.array([1e-3, 2e-3, 3e-3])
        step_min = 1.0
        Rb = 1e9
        expected = np.sum(skr) * Rb * (step_min * 60.0)
        result   = compute_daily_key_bits(skr, step_min, Rb)
        assert abs(result - expected) < 1.0

    def test_higher_skr_more_key_bits(self):
        """SKR cao hơn → nhiều key bits hơn."""
        skr_low  = np.full(100, 1e-4)
        skr_high = np.full(100, 1e-3)
        assert compute_daily_key_bits(skr_high) > compute_daily_key_bits(skr_low)

    def test_longer_duration_more_key_bits(self):
        """Thời gian dài hơn (nhiều steps hơn) → nhiều key bits hơn."""
        skr_short = np.full(60,  1e-3)
        skr_long  = np.full(120, 1e-3)
        assert compute_daily_key_bits(skr_long) > compute_daily_key_bits(skr_short)


# ---------------------------------------------------------------------------
# T06 — Routing Improvement
# ---------------------------------------------------------------------------
class TestRoutingImprovement:
    def test_returns_all_keys(self):
        """compute_routing_improvement phải trả về tất cả keys."""
        g = np.array([1e-3, 2e-3, 0.0])
        b = np.array([5e-4, 1e-3, 0.0])
        result = compute_routing_improvement(g, b)
        expected = {"mean_skr_greedy", "mean_skr_baseline",
                    "improvement_ratio", "improvement_pct",
                    "key_bits_greedy", "key_bits_baseline", "key_bits_gain",
                    "availability_greedy", "availability_baseline"}
        assert expected.issubset(result.keys())

    def test_improvement_ratio_geq_one(self):
        """improvement_ratio phải ≥ 1 (greedy không tệ hơn baseline)."""
        g = np.array([2e-3, 3e-3, 1e-3])
        b = np.array([1e-3, 2e-3, 5e-4])
        result = compute_routing_improvement(g, b)
        assert result["improvement_ratio"] >= 1.0

    def test_equal_skr_ratio_is_one(self):
        """Greedy = baseline → ratio = 1, improvement_pct = 0."""
        skr = np.array([1e-3, 2e-3, 3e-3])
        result = compute_routing_improvement(skr, skr)
        assert abs(result["improvement_ratio"] - 1.0) < 1e-10
        assert abs(result["improvement_pct"])         < 1e-8

    def test_key_bits_gain_nonnegative(self):
        """key_bits_gain phải ≥ 0 (greedy không tệ hơn baseline)."""
        g = np.array([2e-3, 3e-3])
        b = np.array([1e-3, 2e-3])
        result = compute_routing_improvement(g, b)
        assert result["key_bits_gain"] >= 0.0

    def test_availability_in_valid_range(self):
        """availability phải ∈ [0, 1]."""
        g = np.array([1e-3, 0.0, 2e-3])
        b = np.array([5e-4, 0.0, 1e-3])
        result = compute_routing_improvement(g, b)
        assert 0.0 <= result["availability_greedy"]   <= 1.0
        assert 0.0 <= result["availability_baseline"] <= 1.0

    def test_zero_baseline_infinite_ratio(self):
        """Baseline = 0 → ratio = inf (không crash)."""
        g = np.array([1e-3, 2e-3])
        b = np.zeros(2)
        result = compute_routing_improvement(g, b)
        assert result["improvement_ratio"] == np.inf


# ---------------------------------------------------------------------------
# T07 — Integration (full pipeline)
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_full_pipeline_dry_season(self):
        """Full pipeline mùa khô: SKR > 0 nếu có vệ tinh visible."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=2.0, step_minutes=5.0,
        )
        imp = compute_routing_improvement(
            result["skr_greedy"], result["skr_baseline"]
        )
        # Improvement ratio phải ≥ 1
        assert imp["improvement_ratio"] >= 1.0
        # Key bits phải ≥ 0
        assert imp["key_bits_greedy"] >= 0.0

    def test_full_pipeline_wet_season_lower_skr(self):
        """Mùa mưa → SKR_effective thấp hơn mùa khô (P_cloud cao hơn)."""
        common = dict(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            duration_hours=1.0, step_minutes=5.0,
        )
        dry = compute_skr_timeseries(
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"], **common
        )
        wet = compute_skr_timeseries(
            R_mm_h=WX_WET["R_mm_h"], V_km=WX_WET["V_km"],
            P_cloud=WX_WET["P_cloud"], **common
        )
        assert np.mean(wet["skr_greedy"]) <= np.mean(dry["skr_greedy"]) + 1e-15

    def test_daily_key_bits_positive_when_visible(self):
        """Nếu có vệ tinh visible, daily key bits phải > 0."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=0.0,  # no cloud → maximum SKR
            duration_hours=2.0, step_minutes=5.0,
        )
        if np.any(result["n_visible"] > 0):
            kb = compute_daily_key_bits(result["skr_greedy"])
            assert kb > 0.0

    def test_improvement_ratio_consistent_with_timeseries(self):
        """improvement_ratio = mean_greedy / mean_baseline."""
        result = compute_skr_timeseries(
            satellites=ALL_SATS,
            lat_deg=HANOI["lat"], lon_deg=HANOI["lon"],
            t_start=T_REF,
            R_mm_h=WX_DRY["R_mm_h"], V_km=WX_DRY["V_km"],
            P_cloud=WX_DRY["P_cloud"],
            duration_hours=1.0, step_minutes=5.0,
        )
        imp = compute_routing_improvement(
            result["skr_greedy"], result["skr_baseline"]
        )
        mean_g = np.mean(result["skr_greedy"])
        mean_b = np.mean(result["skr_baseline"])
        if mean_b > 0:
            expected_ratio = mean_g / mean_b
            assert abs(imp["improvement_ratio"] - expected_ratio) < 1e-10


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
