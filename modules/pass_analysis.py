"""
pass_analysis.py — Satellite Pass Extraction & Statistics (Task 5, plan 07-5)
================================================================================
Extracts discrete satellite passes (contiguous windows where elevation is at
or above a mask) for a ground station over the real Starlink Shell-1
constellation, and computes descriptive statistics (frequency, duration,
elevation, local-hour distribution). Feeds:
  - Task 6 (citypair_feasibility): store-and-forward relay latency needs the
    actual chronological pass table, matched by satellite ID and time order
    (see tests/test_citypair_feasibility.py — that latency is DIRECTIONAL,
    not symmetric, because satellite ground-track heading differs between
    ascending and descending passes).
  - Task 13/14 (schedulers): ALG-0/ALG-2 operate at the pass level.
  - Task 18 (paper Results): cross-checked against the diurnal cloud/rain
    "best FSO window" finding from modules/weather_stats.py (Task 2, see
    07-5 section 0.2b) — the best combined cloud+rain window is mid-morning
    (roughly 05:00-10:00 local), not overnight as originally assumed.

Local-hour convention: fixed civil UTC offsets (no ASEAN country in this
set observes DST), matching the timezone="auto" convention used by
Open-Meteo in scripts/01_fetch_weather_era5.py, so pass timing and weather
timing are directly comparable.

Rise/set refinement: elevation is sampled at a fixed step (default 30 s);
the true threshold-crossing time is then linearly interpolated between the
straddling samples, so pass duration is not biased by the sampling grid
(important since some high-elevation-mask passes can be very brief).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.orbital_mechanics import (
    GROUND_STATIONS, compute_elevation_timeseries, make_time_array,
)

# Fixed civil UTC offsets (no DST observed by any of these countries),
# matching Open-Meteo's timezone="auto" convention used in Task 1/2.
UTC_OFFSET_HOURS = {
    "hanoi": 7.0, "danang": 7.0, "hcmc": 7.0, "bangkok": 7.0,
    "singapore": 8.0, "manila": 8.0, "jakarta": 7.0, "kuala_lumpur": 8.0,
}


def _interp_crossing_time(t_a, e_a, t_b, e_b, threshold):
    """Linear interpolation of the time at which elevation crosses
    `threshold`, given samples (t_a, e_a) and (t_b, e_b) straddling it."""
    if e_b == e_a:
        return t_a
    frac = (threshold - e_a) / (e_b - e_a)
    frac = min(max(frac, 0.0), 1.0)
    return t_a + (t_b - t_a) * frac


def _local_hour(dt, utc_offset_h):
    return (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + utc_offset_h) % 24.0


def _build_pass_record(sat, py_times, elev, start_i, end_i, min_elev_deg, utc_offset):
    peak_i = start_i + int(np.argmax(elev[start_i:end_i + 1]))

    if start_i > 0:
        t_rise = _interp_crossing_time(
            py_times[start_i - 1], elev[start_i - 1],
            py_times[start_i], elev[start_i], min_elev_deg,
        )
    else:
        t_rise = py_times[start_i]

    if end_i < len(elev) - 1:
        t_set = _interp_crossing_time(
            py_times[end_i], elev[end_i],
            py_times[end_i + 1], elev[end_i + 1], min_elev_deg,
        )
    else:
        t_set = py_times[end_i]

    t_peak = py_times[peak_i]
    duration_s = max((t_set - t_rise).total_seconds(), 0.0)

    return {
        "sat_id": sat.name,
        "t_rise": t_rise,
        "t_set": t_set,
        "t_peak": t_peak,
        "max_elev_deg": float(elev[peak_i]),
        "duration_s": duration_s,
        "local_hour_peak": _local_hour(t_peak, utc_offset),
    }


def extract_passes(satellites, station_key, t_start_utc, duration_hours,
                    min_elev_deg, step_seconds=30.0):
    """Extract all passes (elevation >= min_elev_deg) for every satellite in
    `satellites`, seen from `station_key`, over
    [t_start_utc, t_start_utc + duration_hours).

    Returns a list of dicts: sat_id, t_rise, t_set, t_peak, max_elev_deg,
    duration_s, local_hour_peak (local civil hour of peak elevation, 0-24).
    """
    gs = GROUND_STATIONS[station_key]
    utc_offset = UTC_OFFSET_HOURS[station_key]
    step_minutes = step_seconds / 60.0
    t_array = make_time_array(t_start_utc, duration_hours, step_minutes)
    py_times = t_array.utc_datetime()

    passes = []
    for sat in satellites:
        elev = compute_elevation_timeseries(sat, gs['lat'], gs['lon'], t_array, gs['alt_m'])
        mask = elev >= min_elev_deg

        in_pass = False
        start_i = 0
        n = len(mask)
        for i in range(n):
            if mask[i] and not in_pass:
                in_pass = True
                start_i = i
            elif not mask[i] and in_pass:
                in_pass = False
                passes.append(_build_pass_record(
                    sat, py_times, elev, start_i, i - 1, min_elev_deg, utc_offset))
        if in_pass:
            passes.append(_build_pass_record(
                sat, py_times, elev, start_i, n - 1, min_elev_deg, utc_offset))

    return passes


def pass_frequency_per_day(passes, duration_hours):
    """Mean number of passes per day, extrapolated from the observed count
    over `duration_hours`."""
    if duration_hours <= 0:
        return 0.0
    return len(passes) * (24.0 / duration_hours)


def passes_dataframe(passes):
    """Convert a list of pass dicts (as returned by extract_passes) into a
    list of flat rows suitable for utils.save_intermediate_csv."""
    rows = []
    for p in passes:
        rows.append({
            "sat_id": p["sat_id"],
            "t_rise": p["t_rise"].isoformat(),
            "t_set": p["t_set"].isoformat(),
            "t_peak": p["t_peak"].isoformat(),
            "max_elev_deg": round(p["max_elev_deg"], 2),
            "duration_s": round(p["duration_s"], 1),
            "local_hour_peak": round(p["local_hour_peak"], 3),
        })
    return rows
