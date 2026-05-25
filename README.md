# SIKD/FSO Satellite QKD — Reproducible Code Package

**Project:** Simultaneous Information and Key Distribution over LEO Satellite FSO Channels
**Date:** 2026-05-25
**Authors:** Trương Tuấn Nghĩa (USTH), TS. Vũ Quang Minh (QTALab, PTIT)

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy matplotlib pandas jupyter skyfield Pillow cartopy

# 2. Run all tests (207 total — should take ~10s)
python -m pytest test_channel_model.py test_sikd_performance.py test_weather_model.py test_orbital_mechanics.py test_routing.py -q

# 3. Run comprehensive validation (25 checks)
python debug_comprehensive.py

# 4. Regenerate all figures
python run_all.py --figs-only --no-nb04

# 5. Run NB04 separately (slow, ~5 min — requires internet for TLE)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 04_Satellite_ASEAN_Routing.ipynb

# 6. Constellation density analysis (fig13b-d, fig15b-d)
python 07_Fig13_Fig15_Density.py
python 06_Constellation_Density_Figures.py
python constellation_density_analysis.py
```

## Expected Output

After steps 4-6, `diagrams/` should contain 31 PNG files:
- `fig01` – `fig06`: Core SIKD analysis
- `fig07` – `fig11`: Weather model results
- `fig12` – `fig15`: Satellite routing (after step 5)
- `fig13b/c/d`: SKR timeseries for 500, 1584, 4000 satellites
- `fig15b/c/d`: Coverage maps for 500, 1584, 4000 satellites
- `fig13_skr_timeseries_density_comparison`: 4-panel SKR comparison
- `fig15_coverage_density_comparison`: 2×2 coverage grid
- `fig16` – `fig19`: Vietnam coverage analysis
- `figA` – `figD`: Supplementary figures

## Module Structure

```
modules/
├── channel_model.py        — Geometric loss, Beer-Lambert, H-V turbulence
├── sikd_performance.py     — DT/DD detection, Psift, QBER, SKR, BER
├── weather_model.py        — 8 ASEAN cities × 12 months climate data
├── orbital_mechanics.py    — Skyfield SGP4, TLE fetch, coverage grid
└── routing.py              — Greedy routing, fixed-satellite baseline, SKR timeseries
```

## Constellation Density Scripts

```
06_Constellation_Density_Figures.py  — Combined density comparison figures
07_Fig13_Fig15_Density.py            — Individual fig13b/c/d + fig15b/c/d
constellation_density_analysis.py    — Numerical proof of diminishing returns
```

## Routing Algorithm

### Greedy (weather-adaptive)
At each time step, select the satellite with highest SKR_effective.
Performs continuous handover to maintain optimal link quality.

### Baseline (fixed-satellite, no routing)
Track one satellite until it sets below min_elevation (10°).
When it sets, acquire the **best available** satellite at that moment, then stick with it.
Simulates a system without continuous routing optimization — commits to one satellite per pass.

### Improvement sources
1. **Continuous re-evaluation**: Greedy switches to better satellite every time step
2. **Avoids degradation during pass**: Baseline sticks with one satellite as its elevation changes
3. **Optimal handover timing**: Greedy never waits for a satellite to set before switching

### Results (Hanoi, 3h simulation)

| N_sats | Config | Dry improvement | Wet improvement | Greedy avg (kbps) |
|--------|--------|----------------|----------------|-------------------|
| 176 | 8×22 | +48% | +55% | 7,450 |
| 500 | 10×50 | +88% | +172% | 7,813 |
| 1584 | 72×22 | +70% | +133% | 10,667 |
| 4000 | 40×100 | +58% | +131% | 10,668 |

**Diminishing returns**: Greedy SKR saturates at ~1584 sats (10,667 → 10,668 kbps for 4000).
Improvement % also decreases (88% → 70% → 58%) as baseline improves with more satellites.
Bottleneck shifts from geometry to weather (P_cloud = 55% in wet season).

## Validation for Third-Party Review

To independently validate results:

```bash
# Step 1: Run unit tests
python -m pytest -v --tb=short 2>&1 | tee test_results.txt

# Step 2: Run physics validation
python debug_comprehensive.py 2>&1 | tee validation_results.txt

# Step 3: Run constellation density analysis
python constellation_density_analysis.py 2>&1 | tee density_results.txt

# Step 4: Check for determinism (run twice, compare)
python run_all.py --figs-only --no-nb04
md5sum diagrams/fig*.png > checksums_run1.txt
python run_all.py --figs-only --no-nb04
md5sum diagrams/fig*.png > checksums_run2.txt
diff checksums_run1.txt checksums_run2.txt  # Should be empty (deterministic)
```

**Note:** NB04 (satellite routing) uses live TLE data from Celestrak, so results may vary slightly with different TLE epochs. All other notebooks and scripts are fully deterministic.

## Key Parameters (Paper 2 defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| H_S | 500 km | Satellite altitude |
| λ | 1550 nm | Wavelength |
| θ_C | 10 μrad | Beam divergence |
| a_R | 5 cm | Receiver aperture |
| P_T | 1 W (30 dBm) | Transmit power |
| m_K | 0.05 | QKD modulation index |
| m_D | 0.5 | Data modulation index |
| I_so | 15 dB | Filter isolation |
| R_b | 1 Gbps | Bit rate |

## Known Limitations

1. Turbulence model assumes weak-to-moderate regime (lognormal); strong turbulence uses Gamma-Gamma approximation
2. Weather data is climatological monthly averages — no real-time variability
3. Cloud outage is binary (ON/OFF), not gradual
4. Routing baseline uses fixed-satellite strategy (track until set, then acquire best available satellite and commit)
5. No adaptive optics or error correction modeling
6. Constant weather across sky — no spatial weather diversity (limits routing improvement to geometry only)

## Python Version & Dependencies

Tested with Python 3.12 (Anaconda). Required packages:
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7
- pandas >= 2.0
- skyfield >= 1.45
- Pillow >= 9.0
- jupyter >= 1.0
- pytest >= 7.0
