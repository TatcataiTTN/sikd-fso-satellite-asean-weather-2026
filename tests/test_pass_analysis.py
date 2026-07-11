"""Tests for modules/pass_analysis.py (Task 5, plan 07-5) — WRITTEN FIRST (TDD).

Contract:
    extract_passes(satellites, station_key, t_start_utc, duration_hours,
                   min_elev_deg) -> list[dict]
        Each pass dict: {sat_id, t_rise, t_set, t_peak, max_elev_deg,
                         duration_s, local_hour_peak}
    pass_frequency_per_day(passes, duration_hours) -> float
    passes_dataframe(...) -> rows for CSV export

Structural invariants tested with the real Shell-1 TLE snapshot
(1019 satellites, CelesTrak 25/06/2026). Marked slow where propagation
over hours is required.
"""
import os
import sys
from datetime import datetime, timezone

import pytest

pa = pytest.importorskip("modules.pass_analysis",
                         reason="modules/pass_analysis.py not implemented yet (Task 5)")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TLE_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                        "starlink_shell1_real_tle.txt")
T0 = datetime(2026, 3, 12, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def shell1():
    from modules.orbital_mechanics import parse_tle_block, make_skyfield_satellite
    dicts = parse_tle_block(open(TLE_PATH).read())
    return [make_skyfield_satellite(d) for d in dicts]


@pytest.fixture(scope="module")
def hanoi_passes(shell1):
    # 3-hour window keeps runtime manageable; Shell-1 gives ~dozens of passes.
    return pa.extract_passes(shell1, "hanoi", T0, duration_hours=3.0,
                             min_elev_deg=30.0)


@pytest.mark.slow
class TestPassStructure:
    def test_some_passes_found(self, hanoi_passes):
        # 14-15 satellites are visible at any instant, so a 3h window
        # must contain many distinct passes.
        assert len(hanoi_passes) >= 10

    def test_required_keys(self, hanoi_passes):
        required = {"sat_id", "t_rise", "t_set", "t_peak",
                    "max_elev_deg", "duration_s", "local_hour_peak"}
        for p in hanoi_passes:
            assert required <= set(p.keys())

    def test_rise_before_set(self, hanoi_passes):
        for p in hanoi_passes:
            assert p["t_rise"] < p["t_set"]
            assert p["t_rise"] <= p["t_peak"] <= p["t_set"]

    def test_max_elev_respects_mask(self, hanoi_passes):
        for p in hanoi_passes:
            assert p["max_elev_deg"] >= 30.0

    def test_duration_physical(self, hanoi_passes):
        # Above a 30° mask at 550 km, a single pass lasts seconds to a few
        # minutes — never longer than ~12 minutes.
        for p in hanoi_passes:
            assert 0 < p["duration_s"] <= 12 * 60

    def test_no_overlapping_passes_same_satellite(self, hanoi_passes):
        by_sat = {}
        for p in hanoi_passes:
            by_sat.setdefault(p["sat_id"], []).append(p)
        for sat_id, plist in by_sat.items():
            plist.sort(key=lambda p: p["t_rise"])
            for a, b in zip(plist, plist[1:]):
                assert a["t_set"] <= b["t_rise"], sat_id

    def test_local_hour_in_range(self, hanoi_passes):
        for p in hanoi_passes:
            assert 0.0 <= p["local_hour_peak"] < 24.0


@pytest.mark.slow
class TestPassFrequency:
    def test_frequency_positive(self, hanoi_passes):
        f = pa.pass_frequency_per_day(hanoi_passes, duration_hours=3.0)
        assert f > 0

    def test_stricter_mask_fewer_passes(self, shell1):
        loose = pa.extract_passes(shell1, "hanoi", T0, 1.0, min_elev_deg=30.0)
        strict = pa.extract_passes(shell1, "hanoi", T0, 1.0, min_elev_deg=40.0)
        assert len(strict) <= len(loose)
