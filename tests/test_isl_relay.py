"""Tests for modules/isl_relay.py (Task 24.3, plan 07-5) — WRITTEN FIRST
(TDD). Two sub-algorithms for relaying a key across the ISL mesh
(modules/isl_topology.build_isl_graph) from a pickup satellite/time to a
city_j drop-off, replacing the single-hop wait-for-the-same-satellite
model of Task 6/23:

  time_optimal_relay     minimize wall-clock delivery latency.
  capacity_optimal_relay among candidates within a latency budget, pick
                         the delivery pass with the BEST elevation angle
                         (-> highest SKR -> most delivered key), trading
                         a little extra time for a lot more capacity.

Both walk the same BFS hop-distance from the pickup satellite, assuming
the ISL graph topology is stable over the (short, minutes-scale) relay
duration -- reasonable since n_hops * hop_latency_s << orbital period
(~95 min for a ~550 km shell).
"""
from datetime import datetime, timedelta

import pytest

isl_relay = pytest.importorskip(
    "modules.isl_relay",
    reason="modules/isl_relay.py not implemented yet (Task 24.3)",
)


def _t(minutes_from_epoch):
    return datetime(2026, 3, 12, 0, 0, 0) + timedelta(minutes=minutes_from_epoch)


@pytest.fixture
def chain_graph():
    """A <-> B <-> C <-> D chain, plus E linked only to A.
    Hop distance from A: A=0, B=1, E=1, C=2, D=3.
    """
    return {
        "A": {"B", "E"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C"},
        "E": {"A"},
    }


class TestBFSHopDistance:
    def test_hop_distances_from_a(self, chain_graph):
        hops = isl_relay.bfs_hop_distance(chain_graph, "A")
        assert hops["A"] == 0
        assert hops["B"] == 1
        assert hops["E"] == 1
        assert hops["C"] == 2
        assert hops["D"] == 3

    def test_disconnected_satellite_not_in_result_or_infinite(self):
        graph = {"A": {"B"}, "B": {"A"}, "Z": set()}
        hops = isl_relay.bfs_hop_distance(graph, "A")
        assert "Z" not in hops or hops["Z"] == float("inf")


@pytest.fixture
def relay_scenario(chain_graph):
    """Pickup at satellite A, t_pick = t(0). Candidate delivery passes over
    city_j:
      - satellite B (1 hop, arrives at t_pick + 1*hop_latency): pass rises
        at t(3), LOW elevation (20 deg) -- fast but poor channel.
      - satellite C (2 hops, arrives at t_pick + 2*hop_latency): pass
        rises at t(4), HIGH elevation (80 deg) -- a little slower, much
        better channel.
      - satellite D (3 hops): pass rises at t(1), i.e. BEFORE the key can
        possibly have arrived there (t_pick + 3*hop_latency > t(1)) --
        must be excluded as an invalid candidate.
    hop_latency_s = 30.0 (0.5 min) as in the project default, so 1 hop =
    0.5 min, 2 hops = 1.0 min -- both trivially before t(3)/t(4), leaving
    the pass rise times as the binding constraint (mirrors the real
    regime where n_hops * hop_latency_s is tiny next to pass timescales).
    """
    t_pick = _t(0)
    pass_table_j = [
        {"sat_id": "B", "t_rise": _t(3), "t_set": _t(3.5), "max_elev_deg": 20.0},
        {"sat_id": "C", "t_rise": _t(4), "t_set": _t(4.6), "max_elev_deg": 80.0},
        {"sat_id": "D", "t_rise": _t(1), "t_set": _t(1.5), "max_elev_deg": 60.0},
    ]
    return t_pick, pass_table_j


class TestTimeOptimalRelay:
    def test_picks_earliest_valid_candidate(self, chain_graph, relay_scenario):
        t_pick, pass_table_j = relay_scenario
        result = isl_relay.time_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j, hop_latency_s=30.0,
        )
        assert result is not None
        assert result["sat_id"] == "B"
        assert result["n_hops"] == 1
        assert result["latency_min"] == pytest.approx(3.0, abs=0.1)

    def test_excludes_candidate_arriving_before_key_could_reach_it(self, chain_graph, relay_scenario):
        t_pick, pass_table_j = relay_scenario
        result = isl_relay.time_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j, hop_latency_s=30.0,
        )
        assert result["sat_id"] != "D"

    def test_returns_none_when_no_candidate_reachable(self, chain_graph):
        t_pick = _t(0)
        pass_table_j = [{"sat_id": "ZZZ", "t_rise": _t(5), "t_set": _t(5.5), "max_elev_deg": 50.0}]
        result = isl_relay.time_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j, hop_latency_s=30.0,
        )
        assert result is None


class TestCapacityOptimalRelay:
    def test_prefers_higher_elevation_over_pure_speed(self, chain_graph, relay_scenario):
        t_pick, pass_table_j = relay_scenario
        result = isl_relay.capacity_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j,
            hop_latency_s=30.0, latency_budget_factor=3.0,
        )
        assert result is not None
        assert result["sat_id"] == "C", (
            "capacity-optimal must trade B's speed for C's much better "
            "elevation/capacity when C is within the latency budget"
        )
        assert result["elev_deg"] == pytest.approx(80.0)
        assert result["delivered_gbit"] > 0

    def test_differs_from_time_optimal_on_this_fixture(self, chain_graph, relay_scenario):
        t_pick, pass_table_j = relay_scenario
        time_opt = isl_relay.time_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j, hop_latency_s=30.0,
        )
        cap_opt = isl_relay.capacity_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j,
            hop_latency_s=30.0, latency_budget_factor=3.0,
        )
        assert time_opt["sat_id"] != cap_opt["sat_id"], (
            "fixture must be discriminating: if both algorithms always "
            "agree, the fixture (or capacity_optimal_relay) isn't testing "
            "the time-vs-capacity trade-off"
        )

    def test_returns_none_when_no_candidate_reachable(self, chain_graph):
        t_pick = _t(0)
        pass_table_j = [{"sat_id": "ZZZ", "t_rise": _t(5), "t_set": _t(5.5), "max_elev_deg": 50.0}]
        result = isl_relay.capacity_optimal_relay(
            "A", t_pick, chain_graph, pass_table_j, hop_latency_s=30.0,
        )
        assert result is None
