"""Tests for scripts/09_citypair_feasibility.py logic (Task 6, plan 07-5) —
WRITTEN FIRST (TDD). Encodes the 04/07/2026 design fix: store-and-forward
(SF) latency between two cities is DIRECTED and must NOT be assumed
symmetric, unlike DUAL-downlink classification (a static, distance-only,
symmetric condition).

Physical reasoning: a satellite's ground track has a fixed heading at any
given pass (ascending: south-to-north, descending: north-to-south). For two
cities at different latitudes, the order in which a single satellite visits
them (city i first, then city j, or vice versa) depends on that heading, so
the waiting time "pass over i -> next pass over j on the SAME satellite" is
generally different from "pass over j -> next pass over i". Because this
project has no inter-satellite link (07-5 decision), the relay must be the
same physical satellite carrying the key across both passes, so this
directionality is not an artifact -- it is the actual constraint.

Contract (to be implemented in modules/link_geometry.py /
scripts/09_citypair_feasibility.py):

    build_pairwise_matrices(cities, dist_km_fn, pass_table, h_km, min_elev_deg)
        -> (CLASS, DUAL_PCT, SF_LATENCY_MIN)   each an (N, N) np.ndarray

    sf_latency_minutes(pass_table, city_i, city_j) -> float | None
        Median minutes from a pass over city_i (pickup) to the SAME
        satellite's next pass over city_j (drop-off). None if no such
        relay opportunity exists in the given pass_table window.

`pass_table` here is a list of pass records in the same shape produced by
modules/pass_analysis.extract_passes (Task 5): dicts with at least
{sat_id, station, t_rise, t_set} (station added per-city when merging
tables from multiple cities into one chronological list).
"""
import numpy as np
import pytest
from datetime import datetime, timedelta

cpf = pytest.importorskip(
    "modules.citypair_feasibility",
    reason="modules/citypair_feasibility.py not implemented yet (Task 6)",
)


def _t(minutes_from_epoch):
    return datetime(2026, 3, 12, 0, 0, 0) + timedelta(minutes=minutes_from_epoch)


@pytest.fixture
def asymmetric_pass_table():
    """Two satellites with OPPOSITE headings relative to cities A and B:

    Sat DESC (descending, north-to-south): visits A (north) at t=0-2min,
    then B (south) at t=10-12min -> A->B relay latency ~ 10 min.

    Sat ASC (ascending, south-to-north): visits B (south) at t=100-102min,
    then A (north) at t=118-120min -> B->A relay latency ~ 18 min.

    These two directions are deliberately different (10 vs 18 minutes) so
    a correct implementation must NOT report the same number for both.
    """
    return [
        {"sat_id": "DESC-1", "station": "A", "t_rise": _t(0), "t_set": _t(2)},
        {"sat_id": "DESC-1", "station": "B", "t_rise": _t(10), "t_set": _t(12)},
        {"sat_id": "ASC-1", "station": "B", "t_rise": _t(100), "t_set": _t(102)},
        {"sat_id": "ASC-1", "station": "A", "t_rise": _t(118), "t_set": _t(120)},
    ]


class TestSFLatencyDirectionality:
    def test_a_to_b_and_b_to_a_are_computed_independently(self, asymmetric_pass_table):
        lat_ab = cpf.sf_latency_minutes(asymmetric_pass_table, "A", "B")
        lat_ba = cpf.sf_latency_minutes(asymmetric_pass_table, "B", "A")
        assert lat_ab is not None and lat_ba is not None
        assert lat_ab != pytest.approx(lat_ba, rel=0.01), (
            "SF latency must not be silently mirrored between directions"
        )

    def test_a_to_b_matches_expected_relay(self, asymmetric_pass_table):
        lat_ab = cpf.sf_latency_minutes(asymmetric_pass_table, "A", "B")
        # DESC-1 rises over A at t=0, then over B at t=10 -> ~10 min relay.
        assert lat_ab == pytest.approx(10.0, abs=1.0)

    def test_b_to_a_matches_expected_relay(self, asymmetric_pass_table):
        lat_ba = cpf.sf_latency_minutes(asymmetric_pass_table, "B", "A")
        # ASC-1 rises over B at t=100, then over A at t=118 -> ~18 min relay.
        assert lat_ba == pytest.approx(18.0, abs=1.0)

    def test_no_relay_opportunity_returns_none(self):
        # Only one city ever visited -- no satellite ever links to the other.
        table = [{"sat_id": "X", "station": "A", "t_rise": _t(0), "t_set": _t(2)}]
        assert cpf.sf_latency_minutes(table, "A", "B") is None

    def test_relay_requires_same_satellite(self):
        # Sat X visits A, Sat Y visits B -- no single satellite links them,
        # so this must NOT be treated as a valid relay (no ISL in scope).
        table = [
            {"sat_id": "X", "station": "A", "t_rise": _t(0), "t_set": _t(2)},
            {"sat_id": "Y", "station": "B", "t_rise": _t(5), "t_set": _t(7)},
        ]
        assert cpf.sf_latency_minutes(table, "A", "B") is None


class TestPairwiseMatrices:
    CITIES = ["hanoi", "danang", "jakarta"]

    def _dist_km(self, a, b):
        # Symmetric synthetic distances: hanoi-danang close (DUAL-eligible
        # at a generous mask), hanoi-jakarta / danang-jakarta far (SF).
        d = {
            frozenset(["hanoi", "danang"]): 600.0,
            frozenset(["hanoi", "jakarta"]): 3031.0,
            frozenset(["danang", "jakarta"]): 2480.0,
        }
        return d[frozenset([a, b])]

    def test_class_matrix_symmetric(self):
        CLASS, _, _ = cpf.build_pairwise_matrices(
            self.CITIES, self._dist_km, pass_table=[], h_km=550.0, min_elev_deg=30.0
        )
        CLASS = np.asarray(CLASS)
        assert CLASS.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert CLASS[i, j] == CLASS[j, i]

    def test_dual_pct_matrix_symmetric(self):
        _, DUAL_PCT, _ = cpf.build_pairwise_matrices(
            self.CITIES, self._dist_km, pass_table=[], h_km=550.0, min_elev_deg=30.0
        )
        DUAL_PCT = np.asarray(DUAL_PCT)
        assert np.allclose(DUAL_PCT, DUAL_PCT.T)

    def test_sf_latency_matrix_not_forced_symmetric(self, asymmetric_pass_table):
        # Relabel the synthetic pass table onto two of the three cities to
        # confirm the matrix builder preserves directional asymmetry rather
        # than mirroring SF_LATENCY_MIN like it correctly does for CLASS
        # and DUAL_PCT above.
        table = [dict(p, station={"A": "hanoi", "B": "jakarta"}[p["station"]])
                 for p in asymmetric_pass_table]
        _, _, SF_LATENCY_MIN = cpf.build_pairwise_matrices(
            self.CITIES, self._dist_km, pass_table=table, h_km=550.0, min_elev_deg=30.0
        )
        SF_LATENCY_MIN = np.asarray(SF_LATENCY_MIN)
        i, j = self.CITIES.index("hanoi"), self.CITIES.index("jakarta")
        assert SF_LATENCY_MIN[i, j] != pytest.approx(SF_LATENCY_MIN[j, i], rel=0.01)

    def test_matrix_shape(self):
        CLASS, DUAL_PCT, SF_LATENCY_MIN = cpf.build_pairwise_matrices(
            self.CITIES, self._dist_km, pass_table=[], h_km=550.0, min_elev_deg=30.0
        )
        for M in (CLASS, DUAL_PCT, SF_LATENCY_MIN):
            assert np.asarray(M).shape == (3, 3)
