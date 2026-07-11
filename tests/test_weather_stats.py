"""Tests for modules/weather_stats.py (Tasks 1-3, plan 07-5) — WRITTEN FIRST (TDD).

Contract:
    load_hourly_climatology(city) -> np.ndarray shape (12, 24)
        P_cloud per (month, local hour), built from ERA5 hourly 2015-2024.
    diurnal_amplitude(city, month) -> float
        max-min of the 24-hour cloud profile for that month.
    daily_cloud_correlation(city_a, city_b, months) -> float in [-1, 1]
        Pearson correlation of daily mean cloud cover over the shared record.
    joint_clear_probability(cities, month) -> float in [0, 1]
        P(at least one city has a clear day), comparable to Dang 2023 Fig.7.
    correlation_matrix(months) -> np.ndarray shape (8, 8)

Physics expectations encoded below:
    - Hanoi wet season (Jun-Aug) is cloudier than dry season (Nov-Jan).
    - Nearby same-monsoon cities correlate more strongly than
      cross-equatorial pairs in July (Hanoi-Danang vs Hanoi-Jakarta).
    - Precipitation (true convective signal) peaks in the afternoon
      (13-17h local), minimum near dawn (4-7h) — holds for ALL 8 cities
      (verified 03/07/2026 against the raw hourly record).
    - Total cloud cover (ERA5 `cloud_cover`, an areal fraction across all
      layers) has a DIFFERENT diurnal phase from precipitation: it peaks
      near dawn/overnight and is lowest mid-morning (9-12h) for a MAJORITY
      of cities (6/8), consistent with nocturnal radiative-cooling
      stratus/fog that dissipates through the morning before afternoon
      convection rebuilds cloud. This was investigated directly (not
      assumed) after the original "cloud peaks in the afternoon" hypothesis
      failed 2/8 — see modules/weather_stats.py docstring for the full
      cross-check against precipitation.
"""
import numpy as np
import pytest

ws = pytest.importorskip("modules.weather_stats",
                         reason="modules/weather_stats.py not implemented yet (Task 2)")

CITIES = ["hanoi", "danang", "hcmc", "bangkok",
          "singapore", "manila", "jakarta", "kuala_lumpur"]


class TestHourlyClimatology:
    def test_shape_is_12x24(self):
        m = ws.load_hourly_climatology("hanoi")
        assert np.asarray(m).shape == (12, 24)

    def test_values_are_probabilities(self):
        for city in CITIES:
            m = np.asarray(ws.load_hourly_climatology(city))
            assert np.all(m >= 0.0) and np.all(m <= 1.0), city

    def test_hanoi_wet_season_cloudier_than_dry(self):
        m = np.asarray(ws.load_hourly_climatology("hanoi"))
        wet = m[[5, 6, 7], :].mean()   # Jun, Jul, Aug (0-indexed months)
        dry = m[[10, 11, 0], :].mean()  # Nov, Dec, Jan
        assert wet > dry

    def test_predawn_cloud_peak_majority_of_cities(self):
        # Total cloud cover (areal fraction, all layers) peaks near
        # dawn/overnight and is lowest mid-morning for a MAJORITY of
        # cities -- the opposite phase from precipitation (see
        # test_rain_afternoon_peak_all_cities below and the module
        # docstring for the physical explanation and cross-check).
        n_ok = 0
        for city in CITIES:
            m = np.asarray(ws.load_hourly_climatology(city))
            annual_mean_by_hour = m.mean(axis=0)
            night = np.concatenate([annual_mean_by_hour[0:5],
                                     annual_mean_by_hour[20:24]]).mean()
            midmorning = annual_mean_by_hour[9:13].mean()
            if night >= midmorning:
                n_ok += 1
        assert n_ok >= 6

    def test_rain_afternoon_peak_all_cities(self):
        # Precipitation is the true convective-timing signal: afternoon
        # (13-17h local) rain exceeds dawn (4-7h) rain for every one of
        # the 8 cities -- this is what "convective peak" should test,
        # not total cloud cover (which has the opposite diurnal phase).
        for city in CITIES:
            r = np.asarray(ws.load_hourly_rain_climatology(city))
            annual_mean_by_hour = r.mean(axis=0)
            afternoon = annual_mean_by_hour[13:18].mean()
            dawn = annual_mean_by_hour[4:8].mean()
            assert afternoon >= dawn, city

    def test_diurnal_amplitude_positive(self):
        assert ws.diurnal_amplitude("hanoi", 7) > 0.0


class TestDailyCorrelation:
    def test_range_and_symmetry(self):
        r_ab = ws.daily_cloud_correlation("hanoi", "danang", months=[7])
        r_ba = ws.daily_cloud_correlation("danang", "hanoi", months=[7])
        assert -1.0 <= r_ab <= 1.0
        assert r_ab == pytest.approx(r_ba, abs=1e-9)

    def test_self_correlation_is_one(self):
        assert ws.daily_cloud_correlation("hanoi", "hanoi", months=[7]) == pytest.approx(1.0, abs=1e-9)

    def test_regional_pair_beats_cross_equatorial_in_july(self):
        # Hanoi-Danang share the northern monsoon; Hanoi-Jakarta do not.
        same = ws.daily_cloud_correlation("hanoi", "danang", months=[7])
        cross = ws.daily_cloud_correlation("hanoi", "jakarta", months=[7])
        assert same > cross

    def test_matrix_shape_and_diagonal(self):
        M = np.asarray(ws.correlation_matrix(months=[7]))
        assert M.shape == (8, 8)
        assert np.allclose(np.diag(M), 1.0, atol=1e-9)
        assert np.allclose(M, M.T, atol=1e-9)


class TestJointClearProbability:
    def test_single_city_matches_availability(self):
        # With one city, joint clear probability is that city's own
        # clear-day fraction — must lie in (0, 1).
        p = ws.joint_clear_probability(["hanoi"], month=7)
        assert 0.0 < p < 1.0

    def test_monotone_in_number_of_sites(self):
        # Adding a site can only help: P(at least one clear) is monotone.
        p1 = ws.joint_clear_probability(["hanoi"], month=7)
        p2 = ws.joint_clear_probability(["hanoi", "danang"], month=7)
        p3 = ws.joint_clear_probability(["hanoi", "danang", "jakarta"], month=7)
        assert p1 <= p2 <= p3

    def test_cross_equatorial_site_helps_more_in_july(self):
        # Jakarta (dry in July) should lift joint availability more than
        # another correlated mainland site.
        base = ws.joint_clear_probability(["hanoi", "danang"], month=7)
        with_mainland = ws.joint_clear_probability(["hanoi", "danang", "bangkok"], month=7)
        with_jakarta = ws.joint_clear_probability(["hanoi", "danang", "jakarta"], month=7)
        assert (with_jakarta - base) >= (with_mainland - base)


class TestTenYearRecord:
    def test_record_covers_ten_years(self):
        meta = ws.record_metadata()
        assert meta["start"].startswith("2015")
        assert meta["end"].startswith("2024")

    def test_all_cities_present(self):
        meta = ws.record_metadata()
        assert set(CITIES) <= set(meta["cities"])
