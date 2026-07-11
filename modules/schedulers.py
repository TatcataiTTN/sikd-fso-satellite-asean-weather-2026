"""
schedulers.py — Station-Satellite Matching & Pass Allocation (Tasks 10-14, plan 07-5)
========================================================================================
Implements the formulation of latex_paper_3/main.tex Section IV "Problem
Formulation" (Task 10). This module docstring mirrors that section's
notation exactly so the paper and the code stay in sync; see the LaTeX for
the full derivation and the modeling-scope discussion (acquisition time
folded into the handover penalty, background noise assumed zero / night
operation, no inter-satellite links).

STATUS: schedule_baseline (ALG-0, Task 11), match_weather_aware (ALG-1,
Task 12), and schedule_pairs_greedy / schedule_pairs_ilp / pair_key_totals
(ALG-2, Task 13) are all implemented below. This is LINK scheduling (pass
allocation to city pairs on a shared trusted-relay satellite), NOT network
routing -- no multi-hop path selection is performed, per Thầy Minh's
"selection/scheduling, not routing" framing (07-4).

Sets and decision variables
----------------------------
    G           8 ASEAN ground stations.
    S           Starlink Shell-1 satellites.
    V_g(t)      satellites visible from station g at time t (elevation
                mask determines this; 793 km ground radius at 30 deg,
                573 km at 40 deg -- modules/link_geometry.py).
    x_{g,s}(t)  in {0,1}: station g is receiving from satellite s at time t.

Per-step constraints (matching, not per-station argmax)
---------------------------------------------------------
    sum_s x_{g,s}(t) <= 1        for all g   (one optical head per station)
    sum_g x_{g,s}(t) <= 1        for all s   (one optical head per satellite)
    x_{g,s}(t) = 0               if s not in V_g(t)

Per-step objective (ALG-1, schedule_baseline / match_weather_aware below)
---------------------------------------------------------------------------
    w_{g,s}(t) = SKR_eff(g, s, t | month, local hour)   (weight; already
        encodes the diurnal cloud/rain cycle from modules/weather_stats.py,
        so no separate day/night term is needed in the objective itself)

    max_x  sum_{g,s} w_{g,s}(t) x_{g,s}(t)
           - c_h * sum_g 1[x_{g,s}(t)=1, x_{g,s}(t-1)=0, s' in V_g(t)]

    c_h folds in satellite acquisition/re-tracking time (~10-30s) so a
    separate timing sub-model is not required. Setting w_{g,s}(t) =
    -zeta_s(t) (elevation priority) with c_h = 0 recovers the sticky
    elevation-priority baseline (ALG-0).

Pass-level allocation (ALG-2, schedule_pairs_greedy / schedule_pairs_ilp)
---------------------------------------------------------------------------
    A pass is p = (g, s, [t_rise, t_set]) produced by the matching above.
    Station pairs farther than 1,587 km (30 deg mask) cannot share a single
    pass -- store-and-forward relay is required, with DIRECTIONAL latency
    (modules/citypair_feasibility.py measured a ~100x difference between
    the two directions for one ASEAN pair, driven by ascending/descending
    orbit heading). This is captured by accumulating key contributions
    across passes over time, not by a simultaneity constraint:

    y_{p,q} in {0,1}, q = (A,B) in Q      pass p credited to pair q
    sum_q y_{p,q} <= 1  for all p          each pass credited to <=1 pair
    y_{p,q} = 0 if station(p) not in q     only pairs containing that station

    Under the trusted-relay model (satellite holds both stations' keys,
    broadcasts their bitwise XOR), pair q's usable key is limited by
    whichever side received less:

    K_q = min( sum_{p: station(p)=A} k_p * y_{p,q},
               sum_{p: station(p)=B} k_p * y_{p,q} )

    where k_p = w_{station(p),sat(p)}(t_p) * (t_set - t_rise) is the key
    delivered during pass p.

    max_y  sum_q U(K_q)          (concave utility, throughput-oriented)
    or
    max_y  min_q K_q             (max-min fairness)

Fair comparison (see 07-4 for why the old +678% comparison was invalid)
---------------------------------------------------------------------------
    ALG-0, ALG-1, ALG-2 must be evaluated on the SAME stations, the SAME
    satellite constellation, and the SAME time window. Improvement claims
    must state the baseline in the same sentence (07-3 section XI).
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds

_BIG_COST = 1e12  # represents "not visible" -- effectively infeasible


def schedule_baseline(visibility, prev=None):
    """ALG-0: sticky, weather-blind, per-station elevation-priority baseline.

    For each station g independently: if the satellite g was serving at the
    previous step (`prev[g]`) is still in `visibility[g]`, KEEP it, even if
    another visible satellite now has higher elevation. Only hand over to
    the highest-elevation visible satellite once the current one sets
    (leaves visibility). If nothing is visible, the station is unassigned
    (None).

    Why sticky, not per-step re-evaluation (history, see CLAUDE.md
    "History of this number" and 07-4): an earlier version of this project
    re-evaluated the highest-elevation satellite at EVERY time step, even
    for the baseline. At high satellite density this let the baseline
    itself "hop" between satellites opportunistically, which inflated the
    apparent improvement of any weather-aware scheme compared against it --
    the baseline was already silently benefiting from frequent handovers,
    not just from being weather-blind. Making the baseline sticky (matching
    the thesis's pass-level operation, where a ground station stays locked
    to one satellite for the duration of a pass) isolates the value of
    weather-awareness from the value of handover frequency, which is
    exactly the comparison ALG-1 (match_weather_aware) is meant to isolate.

    This baseline is deliberately per-station and independent: it does NOT
    prevent two stations from locking onto the SAME satellite (no
    coordination). That is the actual weakness ALG-1's bipartite matching
    fixes (Section~IV of the paper); it is not a bug here, it is the
    correct definition of "no coordination" for a fair comparison.

    Parameters
    ----------
    visibility : dict[str, list[dict]]
        station -> list of {"sat_id": str, "elev_deg": float} currently
        visible from that station.
    prev : dict[str, str] | None
        station -> sat_id the station was serving at the previous step.

    Returns
    -------
    dict[str, str | None]
        station -> sat_id assigned (None if nothing visible).
    """
    prev = prev or {}
    assignment = {}

    for station, sats in visibility.items():
        prev_sat = prev.get(station)
        if prev_sat is not None and any(s["sat_id"] == prev_sat for s in sats):
            assignment[station] = prev_sat
            continue

        if sats:
            best = max(sats, key=lambda s: s["elev_deg"])
            assignment[station] = best["sat_id"]
        else:
            assignment[station] = None

    return assignment


def match_weather_aware(visibility, weights, prev=None, handover_cost=0.0):
    """ALG-1: max-weight bipartite matching between stations and visible
    satellites, weighted by weather-aware SKR_eff (Eq. obj_step in the
    paper's Problem Formulation).

    Unlike schedule_baseline (ALG-0, per-station and independent), this
    enforces the two hard constraints jointly: one satellite serves at
    most one station, and one station uses at most one satellite, even
    when two stations both want the same satellite (Section~IV; ~14 of 28
    ASEAN city pairs share a footprint at the 30 deg mask, so this
    conflict is not a corner case). Solved via scipy's Hungarian algorithm
    (linear_sum_assignment) on the negated weight matrix, with infeasible
    (station, satellite) pairs -- satellite not visible from that station
    -- padded to a very large cost so they are never chosen unless truly
    forced by having more stations than satellites.

    handover_cost rewards keeping the previous assignment: it is added
    directly to the weight of (station, prev[station]) before solving,
    folding satellite acquisition/re-tracking cost into the matching
    objective rather than a separate timing sub-model (07-5 Task 10
    "Modeling Scope").

    Parameters
    ----------
    visibility : dict[str, list[dict]]
        station -> list of {"sat_id": str, "elev_deg": float}.
    weights : dict[(str, str), float]
        (station, sat_id) -> SKR_eff weight.
    prev : dict[str, str] | None
        station -> sat_id served at the previous step.
    handover_cost : float
        Bonus added to the previous satellite's weight for that station.

    Returns
    -------
    dict[str, str | None]
        station -> sat_id assigned (None if no feasible pairing).
    """
    prev = prev or {}
    stations = list(visibility.keys())
    sat_ids = sorted({s["sat_id"] for sats in visibility.values() for s in sats})

    if not stations or not sat_ids:
        return {station: None for station in stations}

    visible_lookup = {
        station: {s["sat_id"] for s in sats} for station, sats in visibility.items()
    }

    cost = np.full((len(stations), len(sat_ids)), _BIG_COST)
    for i, station in enumerate(stations):
        for j, sat_id in enumerate(sat_ids):
            if sat_id not in visible_lookup[station]:
                continue
            w = weights.get((station, sat_id), 0.0)
            if prev.get(station) == sat_id:
                w += handover_cost
            cost[i, j] = -w

    row_idx, col_idx = linear_sum_assignment(cost)

    assignment = {station: None for station in stations}
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < _BIG_COST:
            assignment[stations[r]] = sat_ids[c]

    return assignment


def run_matching_over_time(visibility_sequence, weights_sequence, handover_cost=0.0):
    """Run match_weather_aware sequentially over a time series, feeding
    each step's assignment as `prev` to the next step, and counting
    handovers. Prep for Task 14's fair ALG-0/1/2 comparison harness, which
    needs a running simulation rather than isolated single-step calls.

    Parameters
    ----------
    visibility_sequence : list[dict[str, list[dict]]]
        One visibility snapshot per time step.
    weights_sequence : list[dict[(str, str), float]]
        One weights dict per time step (same length as
        visibility_sequence).
    handover_cost : float
        Passed through to match_weather_aware at every step.

    Returns
    -------
    dict
        assignments : list[dict[str, str | None]], one per time step.
        n_handovers : int, total station-satellite changes across all
            steps and stations (a station going from unassigned to
            assigned does not count as a handover; only assigned-to-a-
            different-satellite transitions do).
    """
    assignments = []
    prev = None
    n_handovers = 0

    for vis_t, w_t in zip(visibility_sequence, weights_sequence):
        cur = match_weather_aware(vis_t, w_t, prev=prev, handover_cost=handover_cost)
        if prev is not None:
            for station, sat in cur.items():
                prev_sat = prev.get(station)
                if sat is not None and prev_sat is not None and prev_sat != sat:
                    n_handovers += 1
        assignments.append(cur)
        prev = cur

    return {"assignments": assignments, "n_handovers": n_handovers}


def pair_key_totals(alloc, passes):
    """Compute each pair's usable key K_q = min(side-A total, side-B
    total) under the trusted-relay model (Eq. pair_key in the paper's
    Problem Formulation): the satellite holds both stations' keys and
    broadcasts their bitwise XOR, so a pair can only use as much key as
    its WORSE-served side received.

    Parameters
    ----------
    alloc : list[dict]
        [{"pass_id": str, "pair": (str, str)}, ...] as returned by
        schedule_pairs_greedy / schedule_pairs_ilp.
    passes : list[dict]
        [{"pass_id": str, "station": str, "key_bits": float, ...}, ...].

    Returns
    -------
    dict[(str, str), float]
        pair -> K_q (bits).
    """
    key_bits_by_pass = {p["pass_id"]: p["key_bits"] for p in passes}
    station_by_pass = {p["pass_id"]: p["station"] for p in passes}

    side_totals = {}
    pairs_seen = set()
    for a in alloc:
        pair = a["pair"]
        pairs_seen.add(pair)
        pid = a["pass_id"]
        station = station_by_pass[pid]
        bits = key_bits_by_pass[pid]
        side_totals[(pair, station)] = side_totals.get((pair, station), 0.0) + bits

    totals = {}
    for pair in pairs_seen:
        A, B = pair
        a_bits = side_totals.get((pair, A), 0.0)
        b_bits = side_totals.get((pair, B), 0.0)
        totals[pair] = min(a_bits, b_bits)
    return totals


def schedule_pairs_greedy(passes, pairs, demands=None):
    """ALG-2 (greedy): allocate each pass to the eligible city pair it
    helps most, one pass at a time (Eq. obj_pairs, throughput-oriented
    variant: max sum_q K_q).

    A pure "marginal gain in current K_q" rule is degenerate at a cold
    start: min(x, 0) = 0 for any x, so every candidate looks equally
    useless until BOTH sides of a pair have received at least one pass.
    Instead, passes are processed largest-key-bits first, and each pass is
    assigned to the eligible pair (station in pair) with the largest
    DEFICIT -- the other side's current total minus this station's
    current total -- which is exactly the pair that benefits most from
    more of this station's key, and naturally balances both sides of each
    pair toward a higher min(). Pairs whose demand is already met are
    skipped in favor of pairs still below their `demands` target, when
    both kinds of eligible pair are available for a given pass.

    Parameters
    ----------
    passes : list[dict]
        [{"pass_id": str, "station": str, "key_bits": float, ...}, ...].
    pairs : list[(str, str)]
        Candidate city pairs.
    demands : dict[(str, str), float] | None
        Optional per-pair key demand (bits), used only to break ties
        toward pairs not yet meeting their target.

    Returns
    -------
    list[dict]
        [{"pass_id": str, "pair": (str, str)}, ...].
    """
    demands = demands or {}
    station_pairs = {}
    for pair in pairs:
        for station in pair:
            station_pairs.setdefault(station, []).append(pair)

    side_totals = {(pair, station): 0.0 for pair in pairs for station in pair}

    def current_k(pair):
        A, B = pair
        return min(side_totals[(pair, A)], side_totals[(pair, B)])

    alloc = []
    for p in sorted(passes, key=lambda p: -p["key_bits"]):
        station = p["station"]
        candidates = station_pairs.get(station, [])
        if not candidates:
            continue

        unmet = [pair for pair in candidates
                if current_k(pair) < demands.get(pair, float("inf"))]
        eligible = unmet if unmet else candidates

        def deficit(pair):
            A, B = pair
            other = B if station == A else A
            return side_totals[(pair, other)] - side_totals[(pair, station)]

        best_pair = max(eligible, key=deficit)
        side_totals[(best_pair, station)] += p["key_bits"]
        alloc.append({"pass_id": p["pass_id"], "pair": best_pair})

    return alloc


def schedule_pairs_ilp(passes, pairs, demands=None):
    """ALG-2 (optimal): MILP formulation of pass-to-pair allocation,
    maximizing total pair key sum_q K_q with K_q = min(side-A, side-B)
    linearized as K_q <= side-A and K_q <= side-B (Eq. obj_pairs,
    throughput-oriented variant). Solved with scipy.optimize.milp (HiGHS),
    avoiding a pulp/CBC dependency not otherwise used in this project.

    `demands` is accepted for API symmetry with schedule_pairs_greedy but
    is not a hard constraint here -- Eq. obj_pairs maximizes total pair
    key unconditionally; per-pair demand-driven power-split decisions are
    handled separately in modules/sikd_powersplit.py (Task 8-9).

    Parameters
    ----------
    passes : list[dict]
        [{"pass_id": str, "station": str, "key_bits": float, ...}, ...].
    pairs : list[(str, str)]
        Candidate city pairs.
    demands : dict[(str, str), float] | None
        Unused (kept for API symmetry with schedule_pairs_greedy).

    Returns
    -------
    list[dict]
        [{"pass_id": str, "pair": (str, str)}, ...] -- the optimal
        allocation, or [] if no pass is eligible for any pair.
    """
    eligible = []
    for p in passes:
        station = p["station"]
        for pair in pairs:
            if station in pair:
                eligible.append((p["pass_id"], pair, station, p["key_bits"]))

    if not eligible:
        return []

    y_index = {(pass_id, pair): i for i, (pass_id, pair, _, _) in enumerate(eligible)}
    n_y = len(eligible)
    k_index = {pair: n_y + i for i, pair in enumerate(pairs)}
    n_vars = n_y + len(pairs)

    c = np.zeros(n_vars)
    for pair in pairs:
        c[k_index[pair]] = -1.0  # maximize sum K_q -> minimize -sum K_q

    integrality = np.zeros(n_vars)
    integrality[:n_y] = 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[n_y:] = np.inf

    constraints = []

    passes_seen = {}
    for pass_id, pair, station, key_bits in eligible:
        passes_seen.setdefault(pass_id, []).append(y_index[(pass_id, pair)])
    for pass_id, cols in passes_seen.items():
        row = np.zeros(n_vars)
        row[cols] = 1.0
        constraints.append(LinearConstraint(row, -np.inf, 1.0))

    for pair in pairs:
        A, B = pair
        for side_station in (A, B):
            row = np.zeros(n_vars)
            row[k_index[pair]] = 1.0
            for pass_id, pr, station, key_bits in eligible:
                if pr == pair and station == side_station:
                    row[y_index[(pass_id, pair)]] -= key_bits
            constraints.append(LinearConstraint(row, -np.inf, 0.0))

    res = milp(c, constraints=constraints, integrality=integrality,
              bounds=Bounds(lb, ub))

    alloc = []
    if res.success:
        x = res.x
        for pass_id, pair, station, key_bits in eligible:
            if x[y_index[(pass_id, pair)]] > 0.5:
                alloc.append({"pass_id": pass_id, "pair": pair})
    return alloc


def schedule_pairs_greedy_maxmin(passes, pairs, demands=None):
    """ALG-2 (fairness-oriented greedy): max-min variant of
    schedule_pairs_greedy (Eq. obj_pairs's fairness-oriented alternative:
    max min_q K_q, instead of max sum_q K_q).

    schedule_pairs_greedy always assigns each pass to whichever eligible
    pair gives the largest marginal gain in K_q -- a throughput-maximizing
    rule that can abandon already-poor pairs in favor of pairs with bigger
    absolute gains. This variant instead always tries to help the
    CURRENTLY WORST-OFF pair first (standard water-filling heuristic),
    raising the floor at the cost of a lower total sum in general.

    Added after the Task 14 Monte Carlo comparison found min-pair key was
    exactly 0 in 83-88% of draws under throughput-max ALG-2: a single hub
    station (in 7 of 28 pairs) having a fully-clouded real day zeroes
    every pair through it. That is a genuine physical limit -- no
    algorithm can deliver key to a station with zero real channel
    capacity that day -- so this variant cannot always raise the floor
    above zero, but it never does WORSE on the floor than the
    throughput-max variant when there IS spare capacity to redistribute.

    Parameters
    ----------
    passes : list[dict]
        [{"pass_id": str, "station": str, "key_bits": float, ...}, ...].
    pairs : list[(str, str)]
        Candidate city pairs.
    demands : dict[(str, str), float] | None
        Unused (kept for API symmetry with schedule_pairs_greedy).

    Returns
    -------
    list[dict]
        [{"pass_id": str, "pair": (str, str)}, ...].
    """
    side_totals = {(pair, station): 0.0 for pair in pairs for station in pair}
    passes_by_station = defaultdict(list)
    for p in passes:
        passes_by_station[p["station"]].append(p)

    def current_k(pair):
        A, B = pair
        return min(side_totals[(pair, A)], side_totals[(pair, B)])

    alloc = []
    active_pairs = set(pairs)

    while active_pairs:
        worst_pair = min(active_pairs, key=current_k)
        A, B = worst_pair

        # A pair is "capped" when one side's pool is permanently empty (no
        # more passes will ever arrive for it) AND the other side already
        # meets or exceeds that fixed value -- further assignment there is
        # wasted (min() is stuck at the fixed side), so free those passes
        # for pairs that can still improve. Cold-start note: raw marginal
        # gain in min(A,B) is degenerate at 0 whenever one side is still
        # empty (min(x,0)=0 for any x), so capped-detection -- not gain --
        # is what must decide when to stop, not "gain <= 0" (that was the
        # bug caught during testing: it discarded every pair on the very
        # first iteration, before either side had received anything).
        capped = False
        for station in (A, B):
            other = B if station == A else A
            if not passes_by_station.get(other) and \
               side_totals[(worst_pair, station)] >= side_totals[(worst_pair, other)]:
                capped = True
                break
        if capped:
            active_pairs.discard(worst_pair)
            continue

        # Among the sides with passes remaining, help whichever is
        # currently furthest BEHIND (deficit) -- this is what actually
        # drives progress toward a higher min(), including from a cold
        # start where both sides are still at 0.
        best_deficit = None
        best_pass = None
        best_station = None
        for station in (A, B):
            pool = passes_by_station.get(station)
            if not pool:
                continue
            other = B if station == A else A
            deficit = side_totals[(worst_pair, other)] - side_totals[(worst_pair, station)]
            if best_deficit is None or deficit > best_deficit:
                best_deficit = deficit
                best_station = station
                best_pass = max(pool, key=lambda p: p["key_bits"])

        if best_pass is None:
            active_pairs.discard(worst_pair)
            continue

        side_totals[(worst_pair, best_station)] += best_pass["key_bits"]
        alloc.append({"pass_id": best_pass["pass_id"], "pair": worst_pair})
        passes_by_station[best_station].remove(best_pass)

    return alloc
