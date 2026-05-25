#!/usr/bin/env python3
"""
run_all.py — Chạy toàn bộ hệ thống SIKD/FSO: tests + regenerate figures.

Usage:
    python run_all.py              # Chạy tất cả
    python run_all.py --test-only  # Chỉ chạy tests
    python run_all.py --figs-only  # Chỉ regenerate figures
    python run_all.py --no-nb04    # Bỏ qua notebook 04 (satellite, chậm)
"""
import sys
import os
import time
import subprocess
import argparse

# Paths
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(CODE_DIR, '..', 'latex_report_demo')
DIAGRAMS_DIR = os.path.join(CODE_DIR, 'diagrams')

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_cmd(cmd, cwd=None, label='', timeout=600):
    """Run a command and return (success, stdout, stderr, elapsed)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=cwd or CODE_DIR, timeout=timeout,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0
        return success, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        return False, '', f'TIMEOUT after {timeout}s', time.time() - t0


def section(title):
    print(f'\n{BOLD}{CYAN}{"="*60}{RESET}')
    print(f'{BOLD}{CYAN}  {title}{RESET}')
    print(f'{BOLD}{CYAN}{"="*60}{RESET}\n')


def status(ok, label, elapsed):
    icon = f'{GREEN}PASS{RESET}' if ok else f'{RED}FAIL{RESET}'
    print(f'  [{icon}] {label} ({elapsed:.1f}s)')
    return ok


def main():
    parser = argparse.ArgumentParser(description='Run all SIKD/FSO tests and figures')
    parser.add_argument('--test-only', action='store_true', help='Only run tests')
    parser.add_argument('--figs-only', action='store_true', help='Only regenerate figures')
    parser.add_argument('--no-nb04', action='store_true', help='Skip notebook 04 (slow)')
    args = parser.parse_args()

    t_total = time.time()
    all_ok = True
    results = []

    # ── PHASE 1: TESTS ──────────────────────────────────────────────────────
    if not args.figs_only:
        section('PHASE 1: Running All Tests')

        # Module tests
        test_files = [
            'test_channel_model.py',
            'test_sikd_performance.py',
            'test_weather_model.py',
            'test_orbital_mechanics.py',
            'test_routing.py',
        ]
        ok, out, err, t = run_cmd(
            [sys.executable, '-m', 'pytest'] + test_files + ['-q', '--tb=short'],
            cwd=CODE_DIR, label='Module tests',
        )
        all_ok &= status(ok, f'Module tests (207)', t)
        if not ok:
            print(f'    {RED}{err[-500:]}{RESET}')
        results.append(('Module tests', ok, t))

        # Figure tests
        fig_tests = [
            'test_figures.py',
            'test_figure_data.py',
            'test_figure_generation.py',
        ]
        ok, out, err, t = run_cmd(
            [sys.executable, '-m', 'pytest'] + fig_tests + ['-q', '--tb=short'],
            cwd=REPORT_DIR, label='Figure tests', timeout=300,
        )
        all_ok &= status(ok, f'Figure tests (44)', t)
        if not ok:
            print(f'    {RED}{err[-500:]}{RESET}')
        results.append(('Figure tests', ok, t))

    if args.test_only:
        print(f'\n{BOLD}Total: {time.time()-t_total:.1f}s{RESET}')
        sys.exit(0 if all_ok else 1)

    # ── PHASE 2: REGENERATE FIGURES ──────────────────────────────────────────
    if not args.test_only:
        section('PHASE 2: Regenerating All Figures')

        # generate_figures.py (figA-figD)
        ok, out, err, t = run_cmd(
            [sys.executable, 'generate_figures.py'],
            cwd=REPORT_DIR, label='generate_figures.py',
        )
        all_ok &= status(ok, 'generate_figures.py (figA-figD)', t)
        if not ok:
            print(f'    {RED}{err[-300:]}{RESET}')
        results.append(('generate_figures.py', ok, t))

        # Notebooks
        notebooks = [
            ('02_SIKD_Paper2_Core.ipynb', 'fig01-fig06'),
            ('03_Weather_Vietnam.ipynb', 'fig07-fig11'),
            ('04_Satellite_ASEAN_Routing.ipynb', 'fig12-fig15'),
            ('05_Vietnam_Coverage_Analysis.ipynb', 'fig16-fig19'),
        ]

        for nb_file, figs in notebooks:
            if args.no_nb04 and '04_' in nb_file:
                print(f'  [{YELLOW}SKIP{RESET}] {nb_file} (--no-nb04)')
                continue

            ok, out, err, t = run_cmd(
                ['jupyter', 'nbconvert', '--to', 'notebook',
                 '--execute', '--inplace', nb_file],
                cwd=CODE_DIR, label=nb_file, timeout=600,
            )
            all_ok &= status(ok, f'{nb_file} ({figs})', t)
            if not ok:
                print(f'    {RED}{err[-300:]}{RESET}')
            results.append((nb_file, ok, t))

        # Mermaid diagrams
        mmd_files = [
            'diagram_01_project_architecture.mmd',
            'diagram_02_channel_model.mmd',
            'diagram_03_sikd_pipeline.mmd',
            'diagram_04_weather_model.mmd',
            'diagram_05_satellite_routing.mmd',
        ]
        mmd_ok = True
        t0 = time.time()
        for mmd in mmd_files:
            mmd_path = os.path.join(DIAGRAMS_DIR, mmd)
            png_path = mmd_path.replace('.mmd', '.png')
            ok_m, _, err_m, _ = run_cmd(
                ['mmdc', '-i', mmd_path, '-o', png_path, '-w', '1600', '-b', 'white'],
                timeout=30,
            )
            mmd_ok &= ok_m
        t_mmd = time.time() - t0
        all_ok &= status(mmd_ok, f'Mermaid diagrams (5)', t_mmd)
        results.append(('Mermaid diagrams', mmd_ok, t_mmd))

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    section('SUMMARY')
    total_time = time.time() - t_total
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)

    for label, ok, t in results:
        icon = f'{GREEN}OK{RESET}' if ok else f'{RED}FAIL{RESET}'
        print(f'  {icon}  {label} ({t:.1f}s)')

    print(f'\n  {BOLD}Total: {n_pass} passed, {n_fail} failed, {total_time:.1f}s{RESET}')

    # Count output files
    png_count = len([f for f in os.listdir(DIAGRAMS_DIR) if f.endswith('.png')])
    print(f'  Output: {png_count} PNG files in diagrams/')

    if all_ok:
        print(f'\n  {GREEN}{BOLD}ALL SYSTEMS GO{RESET}')
    else:
        print(f'\n  {RED}{BOLD}SOME STEPS FAILED — check output above{RESET}')

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
