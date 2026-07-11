"""
link_geometry.py — Static Satellite Footprint Geometry (Task 6-7, plan 07-5)
================================================================================
Pure Earth/orbit geometry: ground coverage radius, slant range, and the
resulting dual-downlink feasibility between two ground stations. All
functions here depend only on altitude, elevation mask, and distance — no
time, no satellite ephemeris — and are therefore symmetric by construction
(haversine(A,B) == haversine(B,A), so classify_pair(A,B) == classify_pair(B,A)).

This corrects a bug in the original paper text ("~950 km ground coverage
radius" at a 30 deg mask): 950 km was actually the SLANT RANGE, not the
ground-track radius. The true ground radius is ~793 km; see
TestGroundCoverageRadius.test_radius_not_confused_with_slant_range.

Contrast with modules/citypair_feasibility.py: store-and-forward relay
LATENCY between two cities is direction-dependent (satellite ground-track
heading differs between ascending/descending passes), so that module
deliberately does NOT reuse this symmetric classify_pair() output as if it
applied to latency — only to the DUAL/SF classification itself.
"""
import math

R_EARTH_KM = 6371.0


def ground_coverage_radius_km(h_km: float, min_elev_deg: float) -> float:
    """Ground-track radius of the coverage circle within which a ground
    station can see a satellite at altitude h_km at or above min_elev_deg.

    psi = arccos( R_E / (R_E + h) * cos(elev) ) - elev   (Earth central angle)
    radius = R_E * psi
    """
    elev_rad = math.radians(min_elev_deg)
    ratio = R_EARTH_KM / (R_EARTH_KM + h_km)
    psi = math.acos(ratio * math.cos(elev_rad)) - elev_rad
    return R_EARTH_KM * psi


def slant_range_km(h_km: float, elev_deg: float) -> float:
    """Line-of-sight (slant) distance from ground station to satellite at
    the given elevation angle."""
    elev_rad = math.radians(elev_deg)
    r = R_EARTH_KM
    return math.sqrt((r + h_km) ** 2 - (r * math.cos(elev_rad)) ** 2) - r * math.sin(elev_rad)


def dual_downlink_max_separation_km(h_km: float, min_elev_deg: float) -> float:
    """Maximum great-circle distance between two ground stations for which
    a single satellite pass can, at some instant, be visible to both at or
    above min_elev_deg (both footprint circles overlap at their edges)."""
    return 2.0 * ground_coverage_radius_km(h_km, min_elev_deg)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    r = R_EARTH_KM
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def off_nadir_deg(h_km: float, elev_deg: float) -> float:
    """Transmit-side pointing angle at the satellite, measured from nadir,
    when the ground station sees it at elevation elev_deg. Sine rule in the
    triangle (Earth center, station, satellite):
        sin(eta) = R_E * cos(elev) / (R_E + h)
    Zero when the satellite is at the station's zenith (line of sight is
    the nadir direction); grows as elevation drops. Complements
    zenith_deg = 90 - elev (the RECEIVE-side angle at the ground station)
    in the Task 22 pass ledger."""
    elev_rad = math.radians(elev_deg)
    sin_eta = R_EARTH_KM * math.cos(elev_rad) / (R_EARTH_KM + h_km)
    return math.degrees(math.asin(sin_eta))


def classify_pair(dist_km: float, h_km: float, min_elev_deg: float) -> str:
    """'DUAL' if a single satellite pass can serve both stations
    simultaneously at the given mask, else 'SF' (store-and-forward
    required). Symmetric by construction (depends only on distance)."""
    max_sep = dual_downlink_max_separation_km(h_km, min_elev_deg)
    return "DUAL" if dist_km <= max_sep else "SF"
