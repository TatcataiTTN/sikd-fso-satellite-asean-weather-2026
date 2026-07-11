"""Tests for modules/isl_topology.py (Task 24.3, plan 07-5) — WRITTEN FIRST
(TDD). Encodes the 07/07/2026 architecture fix: the single-hop
store-and-forward model (Task 6/23, modules/citypair_feasibility.py)
assumed a key can only be relayed by the SAME satellite physically
revisiting both cities, which forces waiting almost a full orbital period
(measured: up to 711 minutes). Real LEO constellations with laser
inter-satellite links (ISL) can relay a key across MANY neighboring
satellites almost instantly instead.

This module builds that ISL mesh graph from real TLE orbital elements:
RAAN (right ascension of ascending node, identifies the orbital PLANE —
all satellites in one Walker-constellation plane share ~the same RAAN) and
mean anomaly (identifies POSITION within the plane). Adjacency follows the
standard "+Grid" model used in constellation-networking literature: each
satellite links to its fore/aft neighbor in the same plane (2 links) and
to the nearest-phase satellite in each of the two RAAN-adjacent planes
(2 links) — degree <= 4.

TLE column offsets (0-indexed Python slice of line 2), verified against a
real Starlink TLE (STARLINK-3075: raan=110.3449, mean_anomaly=280.2438):
    inclination  = line2[8:16]
    raan_deg     = line2[17:25]
    mean_anomaly_deg = line2[43:51]
"""
import numpy as np
import pytest

isl_topology = pytest.importorskip(
    "modules.isl_topology",
    reason="modules/isl_topology.py not implemented yet (Task 24.3)",
)


def _make_tle_dict(name, raan_deg, mean_anomaly_deg, inc_deg=53.1602):
    """Build a minimal-but-column-accurate synthetic TLE dict for one
    satellite, with a fixed line1 (unused by orbital-element extraction)
    and a line2 whose fixed-width fields match the real TLE format."""
    line1 = "1 49409U 21104B   26175.30423187  .00198324  00000+0  11523-1 0  9992"
    line2 = (
        f"2 49409 {inc_deg:8.4f} {raan_deg:8.4f} 0001062 "
        f"{79.8674:8.4f} {mean_anomaly_deg:8.4f} 15.10725229254739"
    )
    return {"name": name, "line1": line1, "line2": line2}


@pytest.fixture
def three_plane_fixture():
    """3 orbital planes x 4 satellites, RAAN well-separated (~120 deg
    apart, all satellites within a plane sharing the SAME raan, as in a
    real Walker constellation), mean_anomaly spaced 90 deg apart within
    each plane."""
    planes_raan = {"P0": 10.0, "P1": 130.0, "P2": 250.0}
    tle_dicts = []
    expected_plane_of = {}
    for pname, raan in planes_raan.items():
        for k, ma in enumerate([0.0, 90.0, 180.0, 270.0]):
            sat_id = f"{pname}-{k}"
            tle_dicts.append(_make_tle_dict(sat_id, raan, ma))
            expected_plane_of[sat_id] = pname
    return tle_dicts, expected_plane_of


class TestExtractOrbitalElements:
    def test_columns_match_real_tle_layout(self, three_plane_fixture):
        tle_dicts, _ = three_plane_fixture
        rows = isl_topology.extract_orbital_elements(tle_dicts)
        by_id = {r["sat_id"]: r for r in rows}
        assert by_id["P1-2"]["raan_deg"] == pytest.approx(130.0, abs=0.01)
        assert by_id["P1-2"]["mean_anomaly_deg"] == pytest.approx(180.0, abs=0.01)

    def test_returns_one_row_per_satellite(self, three_plane_fixture):
        tle_dicts, _ = three_plane_fixture
        rows = isl_topology.extract_orbital_elements(tle_dicts)
        assert len(rows) == 12


class TestClusterPlanes:
    def test_assigns_same_plane_id_within_a_plane(self, three_plane_fixture):
        tle_dicts, expected_plane_of = three_plane_fixture
        rows = isl_topology.extract_orbital_elements(tle_dicts)
        raan = np.array([r["raan_deg"] for r in rows])
        plane_id = isl_topology.cluster_planes(raan, bin_width_deg=5.0)
        # All satellites sharing the real (label-agnostic) plane must get
        # the same numeric plane_id; different real planes must differ.
        groups = {}
        for r, pid in zip(rows, plane_id):
            groups.setdefault(expected_plane_of[r["sat_id"]], set()).add(pid)
        for real_plane, assigned_ids in groups.items():
            assert len(assigned_ids) == 1, (
                f"satellites in real plane {real_plane} got split into "
                f"different plane_id: {assigned_ids}"
            )
        distinct_assigned = {list(v)[0] for v in groups.values()}
        assert len(distinct_assigned) == 3, "the 3 real planes must not be merged"

    def test_handles_raan_wraparound_near_0_360(self):
        # Two satellites at 359 and 1 degree RAAN are the SAME plane
        # (2 degree gap through the 0/360 seam), not two different ones.
        raan = np.array([359.0, 1.0, 180.0])
        plane_id = isl_topology.cluster_planes(raan, bin_width_deg=5.0)
        assert plane_id[0] == plane_id[1]
        assert plane_id[2] != plane_id[0]


class TestBuildISLGraph:
    def test_degree_at_most_four(self, three_plane_fixture):
        tle_dicts, _ = three_plane_fixture
        graph = isl_topology.build_isl_graph(tle_dicts, bin_width_deg=5.0)
        for sat_id, neighbors in graph.items():
            assert len(neighbors) <= 4, f"{sat_id} has degree {len(neighbors)} > 4"

    def test_fore_aft_links_within_plane_including_wraparound(self, three_plane_fixture):
        tle_dicts, _ = three_plane_fixture
        graph = isl_topology.build_isl_graph(tle_dicts, bin_width_deg=5.0)
        # Within plane P0 (mean_anomaly 0,90,180,270 -> ordered P0-0,P0-1,
        # P0-2,P0-3), fore/aft neighbors form a ring: 0-1,1-2,2-3,3-0.
        assert "P0-1" in graph["P0-0"]
        assert "P0-3" in graph["P0-0"], "ring must wrap around (P0-0 <-> P0-3)"
        assert "P0-0" in graph["P0-1"]
        assert "P0-2" in graph["P0-1"]

    def test_inter_plane_links_to_both_neighbor_planes(self, three_plane_fixture):
        tle_dicts, _ = three_plane_fixture
        graph = isl_topology.build_isl_graph(tle_dicts, bin_width_deg=5.0)
        # P0-0 (raan=10) must link to its nearest-phase satellite in BOTH
        # RAAN-adjacent planes P2 (raan=250, "before" wrapping) and P1
        # (raan=130, "after") -- i.e. at least one P1-* and one P2-* peer.
        neighbors = graph["P0-0"]
        assert any(n.startswith("P1-") for n in neighbors)
        assert any(n.startswith("P2-") for n in neighbors)

    def test_graph_is_undirected(self, three_plane_fixture):
        tle_dicts, _ = three_plane_fixture
        graph = isl_topology.build_isl_graph(tle_dicts, bin_width_deg=5.0)
        for sat_id, neighbors in graph.items():
            for n in neighbors:
                assert sat_id in graph[n], f"edge {sat_id}-{n} not mirrored back"
