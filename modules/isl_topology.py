"""
isl_topology.py — Inter-Satellite Link (ISL) Mesh Graph from Real TLE (Task 24.3)
================================================================================
Replaces the single-hop store-and-forward assumption of Task 6/23
(modules/citypair_feasibility.py), which required the SAME satellite to
physically revisit both ground stations to relay a key -- forcing a wait
of nearly a full orbital period in the worst case (measured: up to 711
minutes, jakarta<->manila). Real LEO constellations with laser
inter-satellite links (ISL) can instead relay a key across many
neighboring satellites almost instantly (see modules/isl_relay.py for the
routing algorithms that use the graph built here).

Orbital elements identify each satellite's position in the constellation:
  RAAN (Right Ascension of Ascending Node) identifies the orbital PLANE --
    in an ideal Walker constellation, all satellites of one plane share
    ~the same RAAN; different planes are spaced apart.
  mean anomaly identifies the satellite's PHASE (position along its orbit
    within that plane).

TLE line-2 column offsets (0-indexed Python slice), verified against a
real Starlink TLE (STARLINK-3075: raan_deg=110.3449, mean_anomaly_deg=
280.2438):
    inclination_deg  = line2[8:16]
    raan_deg         = line2[17:25]
    eccentricity     = line2[26:33]   (implied decimal, unused here)
    arg_perigee_deg  = line2[34:42]   (unused here)
    mean_anomaly_deg = line2[43:51]
    mean_motion      = line2[52:63]   (unused here; see 02_filter_shell1_tle.py)

Adjacency model: the standard "+Grid" ISL mesh used in constellation-
networking literature -- each satellite links to its fore/aft neighbor
within its own plane (2 links, ring topology with wraparound) and to the
nearest-phase satellite in each of the two RAAN-adjacent planes (2 links)
-- degree <= 4. Real Starlink disables inter-plane links near the polar
seam (rapid relative motion between planes there); this V1 model does NOT
model that (documented simplification -- see 07-6 PROMPT-T24.3 mục 1).

`bin_width_deg` for cluster_planes() MUST be chosen from an actual RAAN
histogram of the real filtered constellation (scripts/12a_build_isl_graph.py
step 0), not assumed a priori -- the filtered Shell-1 subset does not
necessarily match any publicly documented Starlink plane count.
"""
from collections import defaultdict

import numpy as np


def extract_orbital_elements(tle_dicts: list[dict]) -> list[dict]:
    """RAAN and mean anomaly (degrees) for each satellite, from TLE line 2.

    Parameters
    ----------
    tle_dicts : list of {name, line1, line2} dicts (modules.orbital_mechanics.parse_tle_block)

    Returns
    -------
    list of {sat_id, raan_deg, mean_anomaly_deg} dicts, one per satellite.
    """
    rows = []
    for d in tle_dicts:
        l2 = d["line2"]
        rows.append({
            "sat_id": d["name"],
            "raan_deg": float(l2[17:25]),
            "mean_anomaly_deg": float(l2[43:51]),
        })
    return rows


def cluster_planes(raan_deg: np.ndarray, bin_width_deg: float = 5.0) -> np.ndarray:
    """Group satellites into orbital planes by 1-D clustering of RAAN.

    Sorts RAAN values and starts a new plane whenever the gap to the next
    value exceeds `bin_width_deg`; then merges the first and last plane if
    they are actually the same plane wrapping around the 0/360 degree
    seam. This is a simple, inspectable clustering appropriate for Walker
    constellations where plane RAANs are well separated relative to
    within-plane RAAN jitter -- verify this assumption holds by plotting
    the real RAAN histogram before trusting `bin_width_deg` on real data.

    Returns
    -------
    np.ndarray of int plane_id, same length/order as `raan_deg`.
    """
    raan_deg = np.asarray(raan_deg, dtype=float)
    n = len(raan_deg)
    order = np.argsort(raan_deg)
    sorted_raan = raan_deg[order]

    plane_sorted = np.zeros(n, dtype=int)
    current = 0
    for k in range(1, n):
        if sorted_raan[k] - sorted_raan[k - 1] > bin_width_deg:
            current += 1
        plane_sorted[k] = current

    if current > 0:
        wrap_gap = (sorted_raan[0] + 360.0) - sorted_raan[-1]
        if wrap_gap <= bin_width_deg:
            last_id = plane_sorted[-1]
            plane_sorted[plane_sorted == last_id] = 0

    plane_id = np.zeros(n, dtype=int)
    plane_id[order] = plane_sorted
    return plane_id


def _circular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def build_isl_graph(tle_dicts: list[dict], bin_width_deg: float = 5.0) -> dict:
    """Build the undirected ISL mesh adjacency list ("+Grid" model).

    Returns
    -------
    dict[str, set[str]] : sat_id -> set of neighbor sat_ids (degree <= 4).
    """
    rows = extract_orbital_elements(tle_dicts)
    raan = np.array([r["raan_deg"] for r in rows])
    plane_id = cluster_planes(raan, bin_width_deg=bin_width_deg)

    by_plane = defaultdict(list)
    for r, pid in zip(rows, plane_id):
        by_plane[int(pid)].append(r)

    graph = {r["sat_id"]: set() for r in rows}

    # Fore/aft ring within each plane (ordered by phase = mean anomaly).
    for members in by_plane.values():
        ordered = sorted(members, key=lambda r: r["mean_anomaly_deg"])
        m = len(ordered)
        for k in range(m):
            a = ordered[k]["sat_id"]
            b = ordered[(k + 1) % m]["sat_id"]
            graph[a].add(b)
            graph[b].add(a)

    # Inter-plane links to the two RAAN-adjacent planes.
    plane_ids_sorted = sorted(
        by_plane.keys(),
        key=lambda pid: float(np.mean([r["raan_deg"] for r in by_plane[pid]])),
    )
    n_planes = len(plane_ids_sorted)
    plane_rank = {pid: i for i, pid in enumerate(plane_ids_sorted)}

    for pid, members in by_plane.items():
        if n_planes <= 1:
            continue
        rank = plane_rank[pid]
        neighbor_planes = {
            plane_ids_sorted[(rank - 1) % n_planes],
            plane_ids_sorted[(rank + 1) % n_planes],
        }
        neighbor_planes.discard(pid)
        for sat in members:
            for npid in neighbor_planes:
                nearest = min(
                    by_plane[npid],
                    key=lambda r: _circular_diff(r["mean_anomaly_deg"], sat["mean_anomaly_deg"]),
                )
                graph[sat["sat_id"]].add(nearest["sat_id"])
                graph[nearest["sat_id"]].add(sat["sat_id"])

    return graph
