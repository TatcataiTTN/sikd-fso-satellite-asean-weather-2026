"""
isl_relay.py — Two ISL Multi-Hop Relay Sub-Algorithms (Task 24.3)
================================================================================
Replaces the single-hop store-and-forward model of Task 6/23 (a key could
only be relayed by the SAME satellite physically revisiting both ground
stations, forcing a wait of up to ~711 minutes in the worst measured case).
With a real inter-satellite link (ISL) mesh (modules/isl_topology.py), a
key picked up at one satellite can hop across many neighboring satellites
almost instantly and be delivered by WHICHEVER satellite in the mesh next
gets a good pass over the destination city.

Two sub-algorithms, as requested (07/07/2026 feedback):

  time_optimal_relay      minimize wall-clock delivery latency: BFS the
                          (static) ISL graph once from the pickup
                          satellite, then take the EARLIEST feasible
                          drop-off pass among all reachable satellites.
  capacity_optimal_relay  among candidates within a latency budget of the
                          time-optimal answer, pick the drop-off pass with
                          the BEST elevation angle (-> lowest atmospheric
                          path loss/turbulence -> highest SKR -> most
                          delivered key), trading a little extra time for
                          much more capacity.

Modeling assumption (documented, not measured): the ISL graph topology is
treated as STATIC over the relay duration. This is reasonable because
n_hops * ISL_HOP_LATENCY_S is at most a few minutes even for many hops,
while the shell's orbital period is ~95 minutes -- the mesh does not
meaningfully reconfigure on that timescale.

ISL_HOP_LATENCY_S = 30.0 (0.5 min) default: a deliberately CONSERVATIVE
(pessimistic) per-hop cost, representing on-board trusted-node processing/
re-keying overhead -- NOT laser propagation delay (propagation over a
1000-5000 km ISL hop is only a few milliseconds, negligible next to
orbital timescales, so it is ignored here). The conservative choice means
that even in the pessimistic case, ISL relay must still beat the
single-hop model's ~711-minute worst case by orders of magnitude for the
result to be credible -- see 07-6 PROMPT-T24.3 mục 1 for the full
reasoning and the standing question to confirm with the supervisor.
"""
from collections import deque
from datetime import timedelta

from modules.channel_model import compute_channel
from modules.sikd_performance import compute_sikd_performance

H_KM_DEFAULT = 550.0
ISL_HOP_LATENCY_S_DEFAULT = 30.0


def bfs_hop_distance(isl_graph: dict, source_sat: str) -> dict:
    """Minimum hop count from `source_sat` to every satellite reachable in
    `isl_graph` (undirected adjacency dict, modules.isl_topology.build_isl_graph).
    Unreachable satellites are simply absent from the returned dict."""
    hops = {source_sat: 0}
    queue = deque([source_sat])
    while queue:
        cur = queue.popleft()
        for neighbor in isl_graph.get(cur, ()):
            if neighbor not in hops:
                hops[neighbor] = hops[cur] + 1
                queue.append(neighbor)
    return hops


def _feasible_candidates(pickup_sat, t_pick, isl_graph, pass_table_j, hop_latency_s,
                         precomputed_hops=None):
    """Every pass in `pass_table_j` (passes of various satellites over the
    destination city j) that is reachable via the ISL mesh from
    `pickup_sat` AND whose t_rise is not before the key could physically
    have arrived there (t_pick + n_hops * hop_latency_s).

    `precomputed_hops`: optional pre-computed bfs_hop_distance(isl_graph,
    pickup_sat) result. Hop distance depends ONLY on pickup_sat (not on
    t_pick or the destination), so callers evaluating many pickup events
    from the same small set of satellites (e.g. scripts/12b) should
    compute it once per unique satellite and pass it in here to avoid
    re-running BFS on every call."""
    hops = precomputed_hops if precomputed_hops is not None else bfs_hop_distance(isl_graph, pickup_sat)
    out = []
    for p in pass_table_j:
        sat_id = p["sat_id"]
        if sat_id not in hops:
            continue
        n_hops = hops[sat_id]
        t_arrive = t_pick + timedelta(seconds=n_hops * hop_latency_s)
        if p["t_rise"] < t_arrive:
            continue
        latency_min = (p["t_rise"] - t_pick).total_seconds() / 60.0
        out.append({
            "sat_id": sat_id,
            "n_hops": n_hops,
            "latency_min": latency_min,
            "t_rise": p["t_rise"],
            "t_set": p["t_set"],
            "max_elev_deg": p["max_elev_deg"],
        })
    return out


def time_optimal_relay(pickup_sat: str, t_pick, isl_graph: dict, pass_table_j: list,
                       hop_latency_s: float = ISL_HOP_LATENCY_S_DEFAULT,
                       precomputed_hops: dict = None) -> dict | None:
    """Minimum-latency delivery: earliest feasible drop-off pass over city
    j reachable via the ISL mesh from `pickup_sat`. Returns None if no
    satellite in the mesh ever passes over city j (disconnected graph or
    empty `pass_table_j`).

    Returns
    -------
    dict {sat_id, n_hops, latency_min, t_rise, t_set, max_elev_deg} | None
    """
    candidates = _feasible_candidates(pickup_sat, t_pick, isl_graph, pass_table_j,
                                      hop_latency_s, precomputed_hops=precomputed_hops)
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["latency_min"])


def capacity_optimal_relay(pickup_sat: str, t_pick, isl_graph: dict, pass_table_j: list,
                           hop_latency_s: float = ISL_HOP_LATENCY_S_DEFAULT,
                           latency_budget_factor: float = 3.0,
                           H_KM: float = H_KM_DEFAULT, V_km: float = 10.0,
                           R_mm_h: float = 0.0,
                           precomputed_hops: dict = None) -> dict | None:
    """Best-capacity delivery within a latency budget: among feasible
    candidates whose latency is <= `latency_budget_factor` times the
    time-optimal latency, pick the one whose peak elevation gives the
    HIGHEST delivered key (Gbit) -- trading a little extra time for a lot
    more capacity when a much-better-angled pass is only slightly slower.

    Channel evaluated at clear sky (V_km=10, R_mm_h=0) by default; pass in
    real per-hour weather if reusing a specific day's ledger (Task 21/22).

    Returns
    -------
    dict {sat_id, n_hops, latency_min, elev_deg, SKR_kbps, delivered_gbit} | None
    """
    candidates = _feasible_candidates(pickup_sat, t_pick, isl_graph, pass_table_j,
                                      hop_latency_s, precomputed_hops=precomputed_hops)
    if not candidates:
        return None

    time_opt_latency = min(c["latency_min"] for c in candidates)
    budget = time_opt_latency * latency_budget_factor
    within_budget = [c for c in candidates if c["latency_min"] <= budget]

    scored = []
    for c in within_budget:
        zenith_deg = 90.0 - c["max_elev_deg"]
        ch = compute_channel(H_KM, zenith_deg, V_km=V_km, R_mm_h=R_mm_h)
        perf = compute_sikd_performance(ch["hg"], ch["hl"], ch["sigma_X2"])
        duration_s = (c["t_set"] - c["t_rise"]).total_seconds()
        delivered_gbit = perf["SKR_kbps"] * 1e3 * duration_s / 1e9
        scored.append({
            **c,
            "elev_deg": c["max_elev_deg"],
            "SKR_kbps": perf["SKR_kbps"],
            "delivered_gbit": delivered_gbit,
        })

    return max(scored, key=lambda c: c["delivered_gbit"])
