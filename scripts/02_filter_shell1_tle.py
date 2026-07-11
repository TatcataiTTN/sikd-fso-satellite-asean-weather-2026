"""Filter the real Starlink TLE snapshot (starlink_real_tle_20260625.txt) down
to the Shell-1-like subset (~53 deg inclination, ~550 km altitude) that this
project's routing/coverage figures model, instead of using all ~10,630
satellites across every Starlink shell/inclination.

Run (paths are relative to this script, work from any cwd):
  cd 05_Code_v2 && python scripts/02_filter_shell1_tle.py
"""
import os
import sys
import json
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import save_provenance, save_verify_numbers

t0 = time.time()

MU_EARTH = 398600.4418       # km^3/s^2
R_EARTH = 6378.137           # km, WGS84 equatorial

INC_MIN, INC_MAX = 52.0, 54.0     # degrees
ALT_MIN, ALT_MAX = 530.0, 580.0   # km

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IN_PATH = os.path.join(_DATA_DIR, "starlink_real_tle_20260625.txt")
OUT_PATH = os.path.join(_DATA_DIR, "starlink_shell1_real_tle.txt")
META_PATH = os.path.join(_DATA_DIR, "starlink_shell1_filter_metadata.json")


def mean_motion_to_altitude_km(n_rev_per_day: float) -> float:
    n_rad_per_sec = n_rev_per_day * 2 * math.pi / 86400.0
    a_km = (MU_EARTH / (n_rad_per_sec ** 2)) ** (1.0 / 3.0)
    return a_km - R_EARTH


def main():
    lines = [l.rstrip("\n") for l in open(IN_PATH) if l.strip()]
    n_total = len(lines) // 3

    kept = []
    incs, alts = [], []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        inc = float(l2[8:16])
        mean_motion = float(l2[52:63])
        alt = mean_motion_to_altitude_km(mean_motion)
        if INC_MIN <= inc <= INC_MAX and ALT_MIN <= alt <= ALT_MAX:
            kept.append((name, l1, l2))
            incs.append(inc)
            alts.append(alt)

    with open(OUT_PATH, "w") as f:
        for name, l1, l2 in kept:
            f.write(f"{name}\n{l1}\n{l2}\n")

    meta = {
        "input_file": IN_PATH,
        "input_total_satellites": n_total,
        "filter_criteria": {
            "inclination_deg": [INC_MIN, INC_MAX],
            "altitude_km": [ALT_MIN, ALT_MAX],
            "altitude_derivation": "Kepler's third law from TLE mean motion (rev/day)",
        },
        "n_matched": len(kept),
        "inclination_range_matched": [min(incs), max(incs)] if incs else None,
        "altitude_range_matched": [round(min(alts), 1), round(max(alts), 1)] if alts else None,
        "output_file": OUT_PATH,
    }
    json.dump(meta, open(META_PATH, "w"), indent=2)

    print(f"Total satellites in snapshot: {n_total}")
    print(f"Matched Shell-1-like (inc {INC_MIN}-{INC_MAX} deg, alt {ALT_MIN}-{ALT_MAX} km): {len(kept)}")
    if incs:
        print(f"  inclination range matched: {min(incs):.3f} - {max(incs):.3f} deg")
        print(f"  altitude range matched: {min(alts):.1f} - {max(alts):.1f} km")
    print(f"Saved: {OUT_PATH}, {META_PATH}")

    verify = {
        "n_total_snapshot": str(n_total),
        "n_matched_shell1": str(len(kept)),
        "inclination_range_matched": f"{min(incs):.3f}-{max(incs):.3f}" if incs else "N/A",
        "altitude_range_matched_km": f"{min(alts):.1f}-{max(alts):.1f}" if alts else "N/A",
    }
    save_verify_numbers(verify, "tle_filter")
    save_provenance(
        script_name="02_filter_shell1_tle",
        params={"inclination_deg": [INC_MIN, INC_MAX], "altitude_km": [ALT_MIN, ALT_MAX],
                "altitude_derivation": "Kepler's third law from TLE mean motion"},
        key_numbers=verify,
        runtime_secs=time.time() - t0,
        output_files=[OUT_PATH, META_PATH],
        data_sources={"raw TLE snapshot": IN_PATH + " (CelesTrak, fetch date in filename)"},
    )


if __name__ == "__main__":
    main()
