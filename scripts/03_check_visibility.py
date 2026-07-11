"""Verify that the real Shell-1 TLE subset (1019 satellites, CelesTrak
25/06/2026) actually provides visibility coverage across ALL 8 ASEAN cities,
not just the 3 Vietnamese ones used so far. Shell-1 orbital planes are at
~53 deg inclination, so their ground tracks sweep all latitudes from -53 to
+53 deg as the planes precess -- this should cover the whole ASEAN region
(roughly -6 to +21 deg latitude), but we check it explicitly with Skyfield
rather than assuming it."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from datetime import datetime, timedelta, timezone
import numpy as np

from modules.orbital_mechanics import (
    parse_tle_block, make_skyfield_satellite, GROUND_STATIONS, get_visible_satellites,
    make_time_array,
)

T_REF = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
MIN_ELEV = 10.0

TLE_PATH = os.path.join(os.path.dirname(__file__), 'starlink_shell1_real_tle.txt')
tle_dicts = parse_tle_block(open(TLE_PATH).read())
satellites = [make_skyfield_satellite(d) for d in tle_dicts]
print(f'Loaded {len(satellites)} real Shell-1 satellites (CelesTrak 25/06/2026)\n')

CITIES = ['hanoi', 'danang', 'hcmc', 'bangkok', 'singapore', 'manila', 'jakarta', 'kuala_lumpur']

t_array = make_time_array(T_REF, duration_hours=24.0, step_minutes=5.0)

print(f"{'City':14s} {'lat':>8s} {'lon':>9s} {'mean_vis':>9s} {'max_vis':>8s} {'min_vis':>8s} {'%time_0vis':>11s}")
for city in CITIES:
    gs = GROUND_STATIONS[city]
    n_vis = []
    for t_i in t_array:
        visible = get_visible_satellites(satellites, gs['lat'], gs['lon'], t_i, gs['alt_m'], MIN_ELEV)
        n_vis.append(len(visible))
    n_vis = np.array(n_vis)
    pct_zero = 100.0 * np.mean(n_vis == 0)
    print(f"{city:14s} {gs['lat']:8.4f} {gs['lon']:9.4f} {n_vis.mean():9.1f} {n_vis.max():8d} "
          f"{n_vis.min():8d} {pct_zero:10.1f}%")
