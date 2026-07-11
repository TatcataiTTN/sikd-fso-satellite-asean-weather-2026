"""Tests for modules/link_geometry.py (Task 6-7, plan 07-5) — WRITTEN FIRST (TDD).

Contract:
    ground_coverage_radius_km(h_km, min_elev_deg) -> float
    slant_range_km(h_km, elev_deg) -> float
    dual_downlink_max_separation_km(h_km, min_elev_deg) -> float
    haversine_km(lat1, lon1, lat2, lon2) -> float
    classify_pair(dist_km, h_km, min_elev_deg) -> str  ('DUAL' | 'SF')

Reference numbers verified analytically 03/07/2026:
    R_E = 6371 km, h = 550 km
    elev 30°: ground radius 793 km, slant range 993 km, max sep 1587 km
    elev 40°: ground radius 573 km, slant range 812 km
NOTE: the old paper text said "950 km" for the coverage radius — that was
the slant range, not the ground-arc radius. These tests lock in the fix.
"""
import math
import pytest

lg = pytest.importorskip("modules.link_geometry",
                         reason="modules/link_geometry.py not implemented yet (Task 6)")

H = 550.0


class TestGroundCoverageRadius:
    def test_radius_at_30deg(self):
        assert lg.ground_coverage_radius_km(H, 30.0) == pytest.approx(793, abs=5)

    def test_radius_at_40deg(self):
        assert lg.ground_coverage_radius_km(H, 40.0) == pytest.approx(573, abs=5)

    def test_radius_shrinks_with_elevation(self):
        radii = [lg.ground_coverage_radius_km(H, e) for e in (10, 20, 30, 40, 50)]
        assert all(a > b for a, b in zip(radii, radii[1:]))

    def test_radius_not_confused_with_slant_range(self):
        # The 950-km bug: ground radius must be clearly below slant range at 30°.
        assert lg.ground_coverage_radius_km(H, 30.0) < 850


class TestSlantRange:
    def test_slant_at_zenith_equals_altitude(self):
        assert lg.slant_range_km(H, 90.0) == pytest.approx(H, abs=1)

    def test_slant_at_30deg(self):
        assert lg.slant_range_km(H, 30.0) == pytest.approx(993, abs=5)

    def test_slant_at_40deg(self):
        assert lg.slant_range_km(H, 40.0) == pytest.approx(812, abs=5)


class TestDualDownlink:
    def test_max_separation_is_twice_radius(self):
        r = lg.ground_coverage_radius_km(H, 30.0)
        assert lg.dual_downlink_max_separation_km(H, 30.0) == pytest.approx(2 * r, rel=1e-6)

    def test_hanoi_danang_is_dual(self):
        # 606 km apart — comfortably within one footprint at 30°.
        assert lg.classify_pair(606.0, H, 30.0) == "DUAL"

    def test_hanoi_jakarta_is_store_and_forward(self):
        # 3031 km apart — no single satellite can see both at >=30°.
        assert lg.classify_pair(3031.0, H, 30.0) == "SF"

    def test_boundary_pair_flips_class_with_elevation_mask(self):
        # Bangkok–Singapore 1427 km: DUAL at 30° mask, SF at 40° mask.
        assert lg.classify_pair(1427.0, H, 30.0) == "DUAL"
        assert lg.classify_pair(1427.0, H, 40.0) == "SF"


class TestOffNadir:
    # Added for Task 22 (pass ledger): the TRANSMIT-side pointing angle at
    # the satellite, measured from nadir. Sine rule in the Earth-center
    # triangle gives sin(eta) = R_E * cos(elev) / (R_E + h).
    def test_zero_at_zenith(self):
        # Satellite directly overhead: line of sight IS the nadir direction.
        assert lg.off_nadir_deg(H, 90.0) == pytest.approx(0.0, abs=1e-9)

    def test_decreases_with_elevation(self):
        angles = [lg.off_nadir_deg(H, e) for e in (30, 40, 50, 70, 90)]
        assert all(a > b for a, b in zip(angles, angles[1:]))

    def test_reference_value_at_30deg(self):
        # sin(eta) = 6371*cos(30deg)/6921 = 0.7972 -> eta ~ 52.9 deg
        assert lg.off_nadir_deg(H, 30.0) == pytest.approx(52.9, abs=0.5)


class TestHaversine:
    def test_hanoi_jakarta(self):
        d = lg.haversine_km(21.0285, 105.8542, -6.2088, 106.8456)
        assert d == pytest.approx(3031, abs=10)

    def test_symmetric(self):
        a = lg.haversine_km(21.0285, 105.8542, 13.7563, 100.5018)
        b = lg.haversine_km(13.7563, 100.5018, 21.0285, 105.8542)
        assert a == pytest.approx(b, rel=1e-9)

    def test_zero_distance(self):
        assert lg.haversine_km(10.0, 105.0, 10.0, 105.0) == pytest.approx(0.0, abs=1e-6)
