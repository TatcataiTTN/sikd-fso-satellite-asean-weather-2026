"""
run_all.py — Master Orchestration (Task 16, plan 07-5)
================================================================================
Runs the current pipeline in dependency order. Every step declares its own
INPUT files (checked to exist before running -- if missing, the step is
skipped with a clear reason instead of failing with a confusing traceback)
and OUTPUT files (checked to exist and to have a FRESH mtime after running,
so a step that silently no-ops or writes to the wrong path is caught here
instead of discovered later). Finally cross-checks every number in
data/verify/*.txt against latex_paper_3/main.tex.

This does NOT prove main.tex is correct (a verify number could itself be
wrong), only that the two are NOT silently out of sync, and that every
step's data lineage (which raw file fed which script) is explicit rather
than assumed.

Task-number -> script mapping (abstract 07-5 task numbers do not equal
script filenames; this is the authoritative mapping as of 08/07/2026):
  Task 1  -> 01_fetch_weather_era5.py        (skipped by --skip-fetch)
  Task 2  -> 02_diurnal_cloud_figure.py
  Task 3  -> 10_intercity_correlation.py
  Task 5  -> 08_pass_analysis.py
  Task 6  -> 09_citypair_feasibility.py
  Task 8  -> 12_sikd_powersplit.py
  Task 9  -> 13_powersplit_realpass.py
  Task 14/21 -> 11d_algorithm_comparison_fullenum.py   (SLOW, ~12-45 min;
                skipped by --skip-slow)
  Task 22 -> 11e_pass_ledger.py               (needs 11d's raw_runs.csv)
  Task 23 -> 11f_sf_relay_detail.py           (legacy v1 SF latency, kept
                                                for history -- see Task 24.3)
  Task 24/24.2 -> 11g_orbit_maps.py           (needs 11e + 09 outputs)
  Task 24.3 -> 12a_build_isl_graph.py, 12b_isl_relay_recompute.py
  F1  -> 04_generate_reliability_figures.py   (fixed import bug, Task 15)
  F6  -> 07_channel_performance_fig.py
  TLE filter -> 02_filter_shell1_tle.py        (needs a raw, unfiltered TLE
                snapshot on disk -- currently ABSENT; skipped automatically
                with a warning, since the already-filtered
                starlink_shell1_real_tle.txt is present and used by
                everything downstream)

Run:
  cd 05_Code_v2 && python scripts/run_all.py                # everything
  cd 05_Code_v2 && python scripts/run_all.py --skip-fetch    # no ERA5/TLE API calls
  cd 05_Code_v2 && python scripts/run_all.py --skip-slow     # also skip 11d/11e/11g
"""
import os
import re
import sys
import glob
import time
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DATA = os.path.join(ROOT, 'data')
RAW = os.path.join(DATA, 'raw')
INTER = os.path.join(DATA, 'intermediate')
MAIN_TEX = os.path.join(ROOT, '..', 'latex_paper_3', 'main.tex')
FIG_DIR = os.path.join(ROOT, '..', 'latex_paper_3', 'figures')
VERIFY_DIR = os.path.join(DATA, 'verify')

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--skip-fetch", action="store_true",
                     help="skip 01_fetch_weather_era5.py and 02_filter_shell1_tle.py "
                          "(no external API calls; reuse existing data/*.csv.gz and TLE)")
_parser.add_argument("--skip-slow", action="store_true",
                     help="also skip 11d_algorithm_comparison_fullenum.py, "
                          "11e_pass_ledger.py, 11g_orbit_maps.py "
                          "(620-day full enumeration + downstream, 12-45 min)")
args = _parser.parse_args()

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def era5_files(cities=("hanoi", "danang", "hcmc", "bangkok", "singapore",
                       "manila", "jakarta", "kuala_lumpur")):
    return [os.path.join(RAW, f"era5_hourly_{c}.csv.gz") for c in cities]


TLE_RAW = os.path.join(DATA, "starlink_real_tle_20260625.txt")
TLE_FILTERED = os.path.join(DATA, "starlink_shell1_real_tle.txt")
PASS_TABLE = os.path.join(INTER, "pass_table_8cities_7days_elev30.csv")
FEAS_MATRIX = os.path.join(INTER, "citypair_feasibility_matrix.csv")
RAW_RUNS = os.path.join(os.path.join(ROOT, 'temp', 'v4_full_enum'), "raw_runs.csv")

# ----------------------------------------------------------------
# Epoch safety (PROVENANCE.md sec. 6b): the default T_START (2026-03-12,
# hardcoded in several scripts) predates the live TLE epoch (2026-06-25)
# by 104 days, which is fine for AGGREGATE statistics (matching_gain,
# weather_info_gain -- verified robust across epochs) but WRONG for any
# result that depends on a SPECIFIC satellite's position/identity at a
# specific time (orbit ground tracks, top-N passes) -- 0/10 overlap was
# observed between the legacy and corrected epoch for "top satellite"
# rankings. 11g_orbit_maps.py explicitly documents that it must run under
# variant A; 11e_pass_ledger.py feeds it and must match. 11d's own output
# (algorithm comparison) is the epoch-robust one and is intentionally left
# on the default/canonical epoch, matching what main.tex already cites.
VARIANT_A_DIR = os.path.join(ROOT, 'temp', 'epoch_A_20260625')
VARIANT_A_ENV = {
    "SIKD_VARIANT_DIR": VARIANT_A_DIR,
    "SIKD_T_START": "2026-06-25T00:00:00+00:00",
}
VARIANT_A_RAW_RUNS = os.path.join(VARIANT_A_DIR, "v4_full_enum", "raw_runs.csv")

# ----------------------------------------------------------------
# Ordered step list: each step declares its OWN input/output files, so a
# missing input skips cleanly (instead of a confusing traceback) and a
# missing/stale output is caught right after the step runs.
# ----------------------------------------------------------------
STEPS = [
    dict(label="Task 1: fetch ERA5 weather", script="01_fetch_weather_era5.py",
        skip=args.skip_fetch, inputs=[], outputs=era5_files()),
    dict(label="TLE filter (Shell-1 subset)", script="02_filter_shell1_tle.py",
        skip=args.skip_fetch, inputs=[TLE_RAW], outputs=[TLE_FILTERED]),
    dict(label="Task 2: diurnal cloud figure", script="02_diurnal_cloud_figure.py",
        skip=False, inputs=era5_files(("hanoi", "jakarta")),
        outputs=[os.path.join(FIG_DIR, "fig02a_diurnal_cloud_heatmap.png"),
                os.path.join(FIG_DIR, "fig02b_diurnal_window.png")]),
    dict(label="Task 3: inter-city correlation", script="10_intercity_correlation.py",
        skip=False, inputs=era5_files(),
        outputs=[os.path.join(FIG_DIR, "fig03a_correlation_matrix.png"),
                os.path.join(FIG_DIR, "fig03b_joint_availability_dang2023.png")]),
    dict(label="F1: reliability/availability figure", script="04_generate_reliability_figures.py",
        skip=False, inputs=[], outputs=[os.path.join(FIG_DIR, "fig01_availability_heatmap.png")]),
    dict(label="F6: channel performance figure", script="07_channel_performance_fig.py",
        skip=False, inputs=[], outputs=[os.path.join(FIG_DIR, "fig06_channel_performance.png")]),
    dict(label="Task 5: pass analysis", script="08_pass_analysis.py",
        skip=False, inputs=[TLE_FILTERED],
        outputs=[PASS_TABLE, os.path.join(FIG_DIR, "fig04a_pass_timeline.png"),
                os.path.join(FIG_DIR, "fig04b_pass_hour_histogram.png")]),
    dict(label="Task 6: city-pair feasibility", script="09_citypair_feasibility.py",
        skip=False, inputs=[TLE_FILTERED, PASS_TABLE],
        outputs=[FEAS_MATRIX, os.path.join(FIG_DIR, "fig05_citypair_feasibility_map.png")]),
    dict(label="Task 8: SIKD power-split Pareto", script="12_sikd_powersplit.py",
        skip=False, inputs=[], outputs=[os.path.join(FIG_DIR, "fig12_powersplit_pareto.png")]),
    dict(label="Task 9: adaptive power-split (real passes)", script="13_powersplit_realpass.py",
        skip=False, inputs=[TLE_FILTERED, PASS_TABLE],
        outputs=[os.path.join(FIG_DIR, "fig13_powersplit_adaptive_gain.png")]),
    dict(label="Task 14/21: algorithm comparison (full enumeration, SLOW)",
        script="11d_algorithm_comparison_fullenum.py", skip=args.skip_slow,
        inputs=[TLE_FILTERED] + era5_files(), outputs=[RAW_RUNS]),
    dict(label="Task 22: pass-level ledger (variant A -- epoch-sensitive, see above)",
        script="11e_pass_ledger.py",
        skip=args.skip_slow, inputs=[VARIANT_A_RAW_RUNS, TLE_FILTERED],
        outputs=[], env=VARIANT_A_ENV),
    dict(label="Task 23: SF relay detail (legacy v1, kept for history)",
        script="11f_sf_relay_detail.py", skip=False,
        inputs=[PASS_TABLE, FEAS_MATRIX],
        outputs=[os.path.join(INTER, "sf_relay_detail_56pairs.csv")]),
    dict(label="Task 24.3a: ISL graph diagnostic", script="12a_build_isl_graph.py",
        skip=False, inputs=[TLE_FILTERED],
        outputs=[os.path.join(INTER, "isl_graph_shell1.csv"),
                os.path.join(FIG_DIR, "fig11_isl_graph_diagnostic.png")]),
    dict(label="Task 24.3b: ISL relay recompute", script="12b_isl_relay_recompute.py",
        skip=False, inputs=[TLE_FILTERED, PASS_TABLE, FEAS_MATRIX],
        outputs=[os.path.join(INTER, "sf_relay_detail_56pairs_isl.csv"),
                os.path.join(FIG_DIR, "fig10_isl_relay_latency_heatmap.png")]),
    dict(label="Task 24/24.2: orbit maps (variant A -- epoch-sensitive, see above)",
        script="11g_orbit_maps.py",
        skip=args.skip_slow, inputs=[TLE_FILTERED, VARIANT_A_RAW_RUNS],
        outputs=[os.path.join(FIG_DIR, "fig09a_asean_ground_tracks.png"),
                os.path.join(FIG_DIR, "fig09e_top5_pair_exchanges.png")],
        env=VARIANT_A_ENV),
]


def check_files(paths, when):
    missing = [p for p in paths if not os.path.exists(p)]
    return missing


results = []
for step in STEPS:
    label, script = step["label"], step["script"]
    script_path = os.path.join(SCRIPT_DIR, script)

    if step["skip"]:
        log(f"SKIPPED  {label} ({script})")
        results.append((label, script, "skipped", None))
        continue

    missing_in = check_files(step["inputs"], "before")
    if missing_in:
        log(f"SKIPPED  {label} ({script}) -- missing input(s): "
            f"{', '.join(os.path.relpath(p, ROOT) for p in missing_in)}")
        results.append((label, script, "missing-input", None))
        continue

    if not os.path.exists(script_path):
        log(f"MISSING  {label} ({script}) -- script not found, skipping")
        results.append((label, script, "missing-script", None))
        continue

    step_env = dict(os.environ)
    extra_env = step.get("env")
    if extra_env:
        step_env.update(extra_env)
        log(f"  (using env override: {extra_env})")

    t_before_run = time.time()
    log(f"RUNNING  {label} ({script})...")
    t_step = time.time()
    proc = subprocess.run([sys.executable, script_path], cwd=ROOT,
                          capture_output=True, text=True, env=step_env)
    dt = time.time() - t_step
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        log(f"  FAILED after {dt:.1f}s (exit {proc.returncode}): {tail}")
        results.append((label, script, "failed", dt))
        continue

    missing_out = check_files(step["outputs"], "after")
    stale_out = [p for p in step["outputs"]
                if p not in missing_out and os.path.getmtime(p) < t_before_run]
    if missing_out:
        log(f"  OK in {dt:.1f}s but MISSING declared output(s): "
            f"{', '.join(os.path.relpath(p, ROOT) for p in missing_out)}")
        results.append((label, script, "output-missing", dt))
    elif stale_out:
        log(f"  OK in {dt:.1f}s but output(s) NOT refreshed (stale mtime): "
            f"{', '.join(os.path.relpath(p, ROOT) for p in stale_out)}")
        results.append((label, script, "output-stale", dt))
    else:
        log(f"  OK in {dt:.1f}s, {len(step['outputs'])} output(s) verified fresh")
        results.append((label, script, "ok", dt))

# ----------------------------------------------------------------
# Cross-check: every number in data/verify/*.txt vs main.tex
# ----------------------------------------------------------------
# Legacy/superseded verify files are excluded: they belong to earlier
# methodologies (Monte Carlo sampling, single-hop SF relay) that main.tex
# no longer cites, so comparing their numbers against the paper is pure
# noise (Task 19, 10/07/2026 -- this used to report ~212/247 "mismatches"
# that were almost entirely this noise, masking real problems).
LEGACY_VERIFY_FILES = {
    "algorithm_comparison.txt",              # script 11, superseded by v4 (full enumeration)
    "algorithm_comparison_v2_percentiles.txt",  # script 11b, superseded by v4
    "algorithm_comparison_v3_maxmin.txt",       # script 11c, superseded by v4
    "sf_relay_detail.txt",                   # script 11f v1 single-hop, superseded by isl_relay_recompute.txt
}

log("\nCross-checking verify numbers against main.tex...")
tex = open(MAIN_TEX, encoding='utf-8').read() if os.path.exists(MAIN_TEX) else ""

mismatches = []
checked = 0
skipped_legacy = 0
for vfile in sorted(glob.glob(os.path.join(VERIFY_DIR, "*.txt"))):
    if os.path.basename(vfile) in LEGACY_VERIFY_FILES:
        skipped_legacy += 1
        continue
    for line in open(vfile, encoding='utf-8'):
        m = re.match(r"\s*([\w.]+)\s+([\d.\-+xX%'\"]+)\s*$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        num_match = re.search(r"-?\d+\.\d+|-?\d+", value)
        if not num_match:
            continue
        num_str = num_match.group(0)
        checked += 1
        if num_str not in tex:
            mismatches.append((os.path.basename(vfile), key, num_str))

log(f"Checked {checked} verify numbers across "
    f"{len(glob.glob(os.path.join(VERIFY_DIR, '*.txt'))) - skipped_legacy} files "
    f"({skipped_legacy} legacy/superseded files excluded: {sorted(LEGACY_VERIFY_FILES)}).")
if mismatches:
    log(f"{len(mismatches)} verify numbers NOT found verbatim in main.tex "
        f"(review manually -- some are expected: numbers for figures/scripts "
        f"not yet cited in the paper, or formatted differently, e.g. rounded):")
    for fname, key, num in mismatches[:40]:
        log(f"    {fname:45s} {key:40s} {num}")
    if len(mismatches) > 40:
        log(f"    ... and {len(mismatches) - 40} more")
else:
    log("All verify numbers found verbatim in main.tex. 0 mismatches.")

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
log("\n=== Summary ===")
for label, script, status, dt in results:
    dt_str = f"{dt:.1f}s" if dt is not None else "-"
    log(f"  [{status:16s}] {dt_str:>8s}  {label}")
n_bad = sum(1 for _, _, s, _ in results if s in ("failed", "output-missing", "output-stale"))
log(f"\nTotal runtime: {time.time() - t0:.1f}s. {n_bad} step(s) failed or produced no/stale output.")
sys.exit(1 if n_bad else 0)
