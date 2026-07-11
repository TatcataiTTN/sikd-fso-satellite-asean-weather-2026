"""
citypair_feasibility.py — Directed Store-and-Forward Relay Latency (Task 6)
================================================================================
Computes store-and-forward (SF) relay latency between city pairs using a
REAL, chronologically ordered pass table (modules/pass_analysis.extract_passes,
Task 5). Because this project has no inter-satellite link (07-5 decision),
relaying a key from city i to city j requires the SAME physical satellite to
visit i first (pickup) and then j (drop-off).

This is deliberately DIRECTIONAL: a satellite's ground track has a fixed
heading on any given pass (ascending: south-to-north, descending:
north-to-south). For two cities at different latitudes, whether a satellite
visits i-then-j or j-then-i depends on that heading, so the relay latency
i->j is generally NOT equal to j->i (see tests/test_citypair_feasibility.py,
which encodes this with a synthetic two-satellite fixture giving 10 min one
way and 18 min the other).

Contrast with modules/link_geometry.py: the DUAL/SF classification and the
dual-downlink coverage percentage ARE symmetric (static geometry, distance
only) — only the store-and-forward LATENCY is directional. build_pairwise_
matrices() below keeps these as three separate (N, N) matrices rather than
packing them into one triangular matrix, since SF latency needs two
independent numbers per pair (i->j and j->i), not one.
"""
from collections import defaultdict

import numpy as np

from modules.link_geometry import classify_pair


def sf_latency_minutes(pass_table, city_i, city_j):
    """Median minutes from a pass over city_i (pickup) to the SAME
    satellite's next pass over city_j (drop-off), using the pass rise time
    as the reference instant for both ends. Returns None if no satellite in
    `pass_table` ever visits city_i followed by a later visit to city_j.

    `pass_table`: list of dicts with at least {sat_id, station, t_rise,
    t_set} (as produced by modules/pass_analysis.extract_passes, with a
    `station` key merged in per city when pooling multiple stations'
    tables together).
    """
    by_sat = defaultdict(lambda: {"i": [], "j": []})
    for p in pass_table:
        if p["station"] == city_i:
            by_sat[p["sat_id"]]["i"].append(p["t_rise"])
        elif p["station"] == city_j:
            by_sat[p["sat_id"]]["j"].append(p["t_rise"])

    latencies = []
    for sat_id, d in by_sat.items():
        i_times = sorted(d["i"])
        j_times = sorted(d["j"])
        if not i_times or not j_times:
            continue
        for t_i in i_times:
            candidates = [t_j for t_j in j_times if t_j > t_i]
            if not candidates:
                continue
            t_j_next = min(candidates)
            latencies.append((t_j_next - t_i).total_seconds() / 60.0)

    if not latencies:
        return None
    return float(np.median(latencies))


def build_pairwise_matrices(cities, dist_km_fn, pass_table, h_km, min_elev_deg,
                            dual_pct_fn=None):
    """Build three (N, N) matrices for the given ordered list of `cities`:

        CLASS[i,j]          in {'DUAL', 'SF', 'SELF'} — symmetric
        DUAL_PCT[i,j]       % time simultaneous dual-downlink coverage — symmetric
        SF_LATENCY_MIN[i,j] directed store-and-forward relay latency (minutes)
                            i->j, NOT mirrored from j->i

    `dist_km_fn(city_a, city_b) -> float`: symmetric great-circle distance.
    `dual_pct_fn(city_a, city_b) -> float`, optional: real simulated
    simultaneous-visibility percentage for a DUAL-classified pair (e.g. from
    a full time-resolved orbital_mechanics visibility run in script
    09_citypair_feasibility.py). If not given, DUAL pairs get a static
    placeholder (1.0) and SF pairs 0.0 — sufficient for symmetry/shape
    checks but NOT a real coverage percentage.
    """
    n = len(cities)
    CLASS = np.empty((n, n), dtype=object)
    DUAL_PCT = np.zeros((n, n))
    SF_LATENCY_MIN = np.full((n, n), np.nan)

    for i in range(n):
        CLASS[i, i] = "SELF"

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_km_fn(cities[i], cities[j])
            cls = classify_pair(d, h_km, min_elev_deg)
            CLASS[i, j] = cls
            CLASS[j, i] = cls

            if cls == "DUAL":
                pct = dual_pct_fn(cities[i], cities[j]) if dual_pct_fn else 1.0
                DUAL_PCT[i, j] = pct
                DUAL_PCT[j, i] = pct
            else:
                lat_ij = sf_latency_minutes(pass_table, cities[i], cities[j])
                lat_ji = sf_latency_minutes(pass_table, cities[j], cities[i])
                if lat_ij is not None:
                    SF_LATENCY_MIN[i, j] = lat_ij
                if lat_ji is not None:
                    SF_LATENCY_MIN[j, i] = lat_ji

    return CLASS, DUAL_PCT, SF_LATENCY_MIN
