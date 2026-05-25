"""
test_weather_model.py — Unit tests for weather_model.py
========================================================

Kiểm tra tính đúng đắn của dữ liệu khí hậu ASEAN và các hàm
xử lý thời tiết trong weather_model.py.

Chiến lược test
---------------
    - Data integrity  : đủ 8 thành phố × 12 tháng, giá trị hợp lệ
    - Physics         : R, V, P_cloud trong khoảng vật lý hợp lệ
    - Seasonal pattern: mùa mưa R cao hơn mùa khô, V thấp hơn
    - Conversion      : mm/month → mm/h đúng công thức
    - Cloud model     : P_cloud ảnh hưởng đúng đến SKR_effective
    - Integration     : compute_seasonal_hl gọi channel_model đúng
    - Availability    : logic tính toán đúng

Test classes (8 classes, ~35 tests)
------------------------------------
    T01  TestDataIntegrity      (5 tests) — cấu trúc dữ liệu
    T02  TestDataRanges         (5 tests) — giá trị vật lý hợp lệ
    T03  TestSeasonalPatterns   (5 tests) — mùa mưa vs mùa khô
    T04  TestRainConversion     (4 tests) — mm/month → mm/h
    T05  TestCityAccess         (4 tests) — get_city_params API
    T06  TestTurbulenceModel    (3 tests) — lognormal vs gamma_gamma
    T07  TestSeasonalHL         (5 tests) — tích hợp channel_model
    T08  TestAvailability       (6 tests) — SKR_effective, availability

Chạy tests
----------
    python -m pytest test_weather_model.py -v
    python -m pytest test_weather_model.py -v -k "TestSeasonal"
"""

import sys
import os
import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.weather_model import (
    ASEAN_CLIMATE_DATA,
    CITY_COORDS,
    MONTHS,
    RAIN_FRACTION_DEFAULT,
    rain_mm_month_to_mm_h,
    get_city_params,
    get_all_months,
    list_cities,
    get_season,
    get_wet_months,
    get_dry_months,
    get_turbulence_model,
    compute_seasonal_hl,
    compute_link_availability,
    compute_effective_skr,
    compute_annual_stats,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
ALL_CITIES = [
    "hanoi", "hcmc", "danang",
    "bangkok", "singapore", "manila", "jakarta", "kuala_lumpur",
]

# Known wet months for spot-checks
HANOI_WET_MONTHS   = [5, 6, 7, 8, 9, 10]   # R > 100 mm
HCMC_WET_MONTHS    = [5, 6, 7, 8, 9, 10, 11]
JAKARTA_WET_MONTHS = [1, 2, 3, 4, 11, 12]  # inverted season


# ---------------------------------------------------------------------------
# T01 — Data Integrity
# ---------------------------------------------------------------------------
class TestDataIntegrity:
    def test_all_cities_present(self):
        """Tất cả 8 thành phố phải có trong ASEAN_CLIMATE_DATA."""
        for city in ALL_CITIES:
            assert city in ASEAN_CLIMATE_DATA, f"Missing city: {city}"

    def test_each_city_has_12_months(self):
        """Mỗi thành phố phải có đúng 12 tháng dữ liệu."""
        for city in ALL_CITIES:
            assert len(ASEAN_CLIMATE_DATA[city]) == 12, (
                f"{city}: expected 12 months, got {len(ASEAN_CLIMATE_DATA[city])}"
            )

    def test_each_month_has_3_values(self):
        """Mỗi tháng phải có đúng 3 giá trị: (R_mm_month, V_km, P_cloud)."""
        for city in ALL_CITIES:
            for i, entry in enumerate(ASEAN_CLIMATE_DATA[city]):
                assert len(entry) == 3, (
                    f"{city} month {i+1}: expected 3 values, got {len(entry)}"
                )

    def test_city_coords_complete(self):
        """Tất cả 8 thành phố phải có tọa độ trong CITY_COORDS."""
        for city in ALL_CITIES:
            assert city in CITY_COORDS, f"Missing coords for {city}"
            lat, lon, alt = CITY_COORDS[city]
            assert isinstance(lat, float) and isinstance(lon, float)

    def test_months_list_length(self):
        """MONTHS phải có đúng 12 phần tử."""
        assert len(MONTHS) == 12


# ---------------------------------------------------------------------------
# T02 — Data Ranges (physical validity)
# ---------------------------------------------------------------------------
class TestDataRanges:
    def test_rainfall_nonnegative(self):
        """R_mm_month ≥ 0 cho tất cả thành phố và tháng."""
        for city in ALL_CITIES:
            for i, (R, V, P) in enumerate(ASEAN_CLIMATE_DATA[city]):
                assert R >= 0, f"{city} month {i+1}: R={R} < 0"

    def test_visibility_positive(self):
        """V_km > 0 cho tất cả thành phố và tháng."""
        for city in ALL_CITIES:
            for i, (R, V, P) in enumerate(ASEAN_CLIMATE_DATA[city]):
                assert V > 0, f"{city} month {i+1}: V={V} ≤ 0"

    def test_visibility_reasonable_range(self):
        """V_km ∈ [1, 50] km — khoảng hợp lệ cho khí hậu nhiệt đới."""
        for city in ALL_CITIES:
            for i, (R, V, P) in enumerate(ASEAN_CLIMATE_DATA[city]):
                assert 1.0 <= V <= 50.0, (
                    f"{city} month {i+1}: V={V} km out of [1, 50]"
                )

    def test_cloud_probability_valid(self):
        """P_cloud ∈ [0, 1] cho tất cả thành phố và tháng."""
        for city in ALL_CITIES:
            for i, (R, V, P) in enumerate(ASEAN_CLIMATE_DATA[city]):
                assert 0.0 <= P <= 1.0, (
                    f"{city} month {i+1}: P_cloud={P} out of [0, 1]"
                )

    def test_rainfall_reasonable_max(self):
        """R_mm_month ≤ 500 mm — giới hạn vật lý hợp lý cho nhiệt đới."""
        for city in ALL_CITIES:
            for i, (R, V, P) in enumerate(ASEAN_CLIMATE_DATA[city]):
                assert R <= 500, (
                    f"{city} month {i+1}: R={R} mm/month > 500 (unrealistic)"
                )


# ---------------------------------------------------------------------------
# T03 — Seasonal Patterns
# ---------------------------------------------------------------------------
class TestSeasonalPatterns:
    def test_hanoi_wet_season_higher_rainfall(self):
        """Hà Nội: tháng 6 (đỉnh mưa) phải có R cao hơn tháng 1 (khô)."""
        R_jun = ASEAN_CLIMATE_DATA["hanoi"][5][0]   # tháng 6
        R_jan = ASEAN_CLIMATE_DATA["hanoi"][0][0]   # tháng 1
        assert R_jun > R_jan, f"Hanoi Jun R={R_jun} ≤ Jan R={R_jan}"

    def test_hcmc_wet_season_lower_visibility(self):
        """TP.HCM: tháng 7 (mùa mưa) phải có V thấp hơn tháng 1 (mùa khô)."""
        V_jul = ASEAN_CLIMATE_DATA["hcmc"][6][1]   # tháng 7
        V_jan = ASEAN_CLIMATE_DATA["hcmc"][0][1]   # tháng 1
        assert V_jul < V_jan, f"HCMC Jul V={V_jul} ≥ Jan V={V_jan}"

    def test_jakarta_inverted_season(self):
        """Jakarta: mùa mưa tháng 1 (R cao) và mùa khô tháng 7 (R thấp)."""
        R_jan = ASEAN_CLIMATE_DATA["jakarta"][0][0]
        R_jul = ASEAN_CLIMATE_DATA["jakarta"][6][0]
        assert R_jan > R_jul, (
            f"Jakarta: Jan R={R_jan} should be > Jul R={R_jul} (inverted season)"
        )

    def test_wet_season_higher_cloud_probability(self):
        """Mùa mưa phải có P_cloud cao hơn mùa khô cho Hà Nội."""
        P_jun = ASEAN_CLIMATE_DATA["hanoi"][5][2]   # tháng 6
        P_jan = ASEAN_CLIMATE_DATA["hanoi"][0][2]   # tháng 1
        assert P_jun > P_jan, f"Hanoi Jun P_cloud={P_jun} ≤ Jan P_cloud={P_jan}"

    def test_singapore_rain_all_year(self):
        """Singapore: mưa quanh năm — không có tháng nào R = 0."""
        for i, (R, V, P) in enumerate(ASEAN_CLIMATE_DATA["singapore"]):
            assert R > 0, f"Singapore month {i+1}: R=0 (unexpected for Singapore)"

    def test_danang_peak_rain_oct(self):
        """Đà Nẵng: tháng 10 phải là tháng mưa nhiều nhất."""
        rainfall = [ASEAN_CLIMATE_DATA["danang"][i][0] for i in range(12)]
        peak_month = rainfall.index(max(rainfall)) + 1
        assert peak_month == 10, (
            f"Danang peak rain month = {peak_month}, expected 10 (Oct)"
        )


# ---------------------------------------------------------------------------
# T04 — Rain Conversion
# ---------------------------------------------------------------------------
class TestRainConversion:
    def test_zero_rain_gives_zero_rate(self):
        """R_mm_month = 0 → R_mm_h = 0."""
        assert rain_mm_month_to_mm_h(0.0) == 0.0

    def test_negative_rain_gives_zero(self):
        """R_mm_month < 0 → R_mm_h = 0 (guard)."""
        assert rain_mm_month_to_mm_h(-10.0) == 0.0

    def test_conversion_formula(self):
        """R_mm_h = R_mm_month / (30 × 24 × rain_fraction)."""
        R_month = 200.0
        frac = 0.15
        expected = R_month / (30.0 * 24.0 * frac)
        result = rain_mm_month_to_mm_h(R_month, frac)
        assert abs(result - expected) < 1e-10, (
            f"Got {result:.6f}, expected {expected:.6f}"
        )

    def test_higher_rain_fraction_lower_rate(self):
        """
        Cùng R_mm_month, rain_fraction lớn hơn → R_mm_h nhỏ hơn.
        (Mưa trải đều hơn → intensity thấp hơn.)
        """
        r1 = rain_mm_month_to_mm_h(200.0, rain_fraction=0.10)
        r2 = rain_mm_month_to_mm_h(200.0, rain_fraction=0.30)
        assert r1 > r2


# ---------------------------------------------------------------------------
# T05 — City Access API
# ---------------------------------------------------------------------------
class TestCityAccess:
    def test_get_city_params_returns_all_keys(self):
        """get_city_params phải trả về tất cả keys cần thiết."""
        result = get_city_params("hanoi", 6)
        expected_keys = {"R_mm_month", "R_mm_h", "V_km", "P_cloud",
                         "city", "month", "month_name"}
        assert expected_keys.issubset(result.keys())

    def test_get_city_params_month_name(self):
        """month_name phải khớp với MONTHS list."""
        for m in range(1, 13):
            result = get_city_params("hanoi", m)
            assert result["month_name"] == MONTHS[m - 1]

    def test_invalid_city_raises(self):
        """City không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_city_params("tokyo", 1)

    def test_invalid_month_raises(self):
        """Month ngoài [1, 12] phải raise ValueError."""
        with pytest.raises(ValueError, match="Month must be"):
            get_city_params("hanoi", 13)
        with pytest.raises(ValueError, match="Month must be"):
            get_city_params("hanoi", 0)

    def test_get_all_months_length(self):
        """get_all_months phải trả về đúng 12 phần tử."""
        result = get_all_months("hcmc")
        assert len(result) == 12

    def test_list_cities_complete(self):
        """list_cities phải trả về đủ 8 thành phố."""
        cities = list_cities()
        assert len(cities) == 8
        for city in ALL_CITIES:
            assert city in cities

    def test_case_insensitive_city_name(self):
        """City name không phân biệt hoa/thường."""
        r1 = get_city_params("hanoi", 1)
        r2 = get_city_params("Hanoi", 1)
        assert r1["R_mm_month"] == r2["R_mm_month"]


# ---------------------------------------------------------------------------
# T06 — Turbulence Model Selection
# ---------------------------------------------------------------------------
class TestTurbulenceModel:
    def test_weak_turbulence_lognormal(self):
        """σR² < 0.3 → lognormal."""
        assert get_turbulence_model(0.01) == "lognormal"
        assert get_turbulence_model(0.29) == "lognormal"

    def test_moderate_turbulence_gamma_gamma(self):
        """σR² ≥ 0.3 → gamma_gamma."""
        assert get_turbulence_model(0.30) == "gamma_gamma"
        assert get_turbulence_model(1.0)  == "gamma_gamma"
        assert get_turbulence_model(5.0)  == "gamma_gamma"

    def test_boundary_value(self):
        """σR² = 0.3 → gamma_gamma (boundary belongs to gamma_gamma)."""
        assert get_turbulence_model(0.3) == "gamma_gamma"


# ---------------------------------------------------------------------------
# T07 — Seasonal HL (integration with channel_model)
# ---------------------------------------------------------------------------
class TestSeasonalHL:
    def test_returns_all_keys(self):
        """compute_seasonal_hl phải trả về tất cả keys cần thiết."""
        result = compute_seasonal_hl("hanoi", 6, zeta_deg=45)
        expected = {"hl", "hl_dB", "R_mm_h", "V_km", "P_cloud",
                    "season", "city", "month", "month_name"}
        assert expected.issubset(result.keys())

    def test_hl_in_valid_range(self):
        """hl ∈ (0, 1] cho tất cả thành phố và tháng."""
        for city in ALL_CITIES:
            for month in [1, 6, 12]:
                result = compute_seasonal_hl(city, month, zeta_deg=45)
                assert 0 < result["hl"] <= 1.0, (
                    f"{city} month {month}: hl={result['hl']:.4f} out of (0, 1]"
                )

    def test_dry_season_higher_hl(self):
        """Mùa khô (ít mưa, tầm nhìn tốt) → hl cao hơn mùa mưa."""
        dry = compute_seasonal_hl("hanoi", 1, zeta_deg=45)   # tháng 1 — khô
        wet = compute_seasonal_hl("hanoi", 6, zeta_deg=45)   # tháng 6 — mưa
        assert dry["hl"] > wet["hl"], (
            f"Dry hl={dry['hl']:.4f} ≤ Wet hl={wet['hl']:.4f}"
        )

    def test_season_label_correct(self):
        """season label phải khớp với lượng mưa."""
        dry = compute_seasonal_hl("hanoi", 1, zeta_deg=45)
        wet = compute_seasonal_hl("hanoi", 6, zeta_deg=45)
        assert dry["season"] == "dry"
        assert wet["season"] == "wet"

    def test_higher_zenith_lower_hl(self):
        """Zenith angle lớn hơn → đường dài hơn → hl thấp hơn."""
        hl_30 = compute_seasonal_hl("hanoi", 6, zeta_deg=30)["hl"]
        hl_60 = compute_seasonal_hl("hanoi", 6, zeta_deg=60)["hl"]
        assert hl_30 > hl_60, (
            f"hl(30°)={hl_30:.4f} ≤ hl(60°)={hl_60:.4f}"
        )


# ---------------------------------------------------------------------------
# T08 — Availability and Effective SKR
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_zero_cloud_full_availability(self):
        """P_cloud = 0 → availability = 1 (nếu SKR > 0)."""
        avail = compute_link_availability(skr_norm=1e-3, P_cloud=0.0)
        assert avail == 1.0

    def test_full_cloud_zero_availability(self):
        """P_cloud = 1 → availability = 0 (link luôn bị chặn)."""
        avail = compute_link_availability(skr_norm=1e-3, P_cloud=1.0)
        assert avail == 0.0

    def test_zero_skr_zero_availability(self):
        """SKR = 0 → availability = 0 (không có key)."""
        avail = compute_link_availability(skr_norm=0.0, P_cloud=0.3)
        assert avail == 0.0

    def test_effective_skr_formula(self):
        """SKR_effective = SKR_clear × (1 - P_cloud)."""
        skr = 1e-3
        P = 0.4
        expected = skr * (1.0 - P)
        result = compute_effective_skr(skr, P)
        assert abs(result - expected) < 1e-15

    def test_effective_skr_zero_cloud(self):
        """P_cloud = 0 → SKR_effective = SKR_clear."""
        skr = 5e-4
        assert compute_effective_skr(skr, 0.0) == skr

    def test_annual_stats_structure(self):
        """compute_annual_stats phải trả về tất cả keys cần thiết."""
        skr_by_month = [1e-3] * 12
        result = compute_annual_stats("hanoi", skr_by_month)
        expected = {"skr_annual_mean", "skr_effective_mean", "availability_mean",
                    "skr_wet_mean", "skr_dry_mean", "wet_months", "dry_months"}
        assert expected.issubset(result.keys())

    def test_annual_stats_wet_dry_split(self):
        """wet_months + dry_months phải bao phủ đủ 12 tháng."""
        skr_by_month = [1e-3] * 12
        result = compute_annual_stats("hanoi", skr_by_month)
        all_months = sorted(result["wet_months"] + result["dry_months"])
        assert all_months == list(range(1, 13)), (
            f"wet+dry months = {all_months}, expected 1–12"
        )

    def test_annual_stats_effective_less_than_clear(self):
        """SKR_effective_mean ≤ SKR_annual_mean (mây luôn giảm SKR)."""
        skr_by_month = [1e-3] * 12
        result = compute_annual_stats("hanoi", skr_by_month)
        assert result["skr_effective_mean"] <= result["skr_annual_mean"]

    def test_annual_stats_wrong_length_raises(self):
        """skr_by_month với độ dài sai phải raise ValueError."""
        with pytest.raises(ValueError, match="12 elements"):
            compute_annual_stats("hanoi", [1e-3] * 11)

    def test_wet_season_lower_effective_skr(self):
        """
        Mùa mưa có P_cloud cao hơn → SKR_effective thấp hơn mùa khô,
        ngay cả khi SKR_clear bằng nhau.
        """
        # Lấy P_cloud tháng 6 (mưa) và tháng 1 (khô) của Hà Nội
        p_wet = get_city_params("hanoi", 6)["P_cloud"]
        p_dry = get_city_params("hanoi", 1)["P_cloud"]
        skr_clear = 1e-3
        eff_wet = compute_effective_skr(skr_clear, p_wet)
        eff_dry = compute_effective_skr(skr_clear, p_dry)
        assert eff_wet < eff_dry, (
            f"Wet SKR_eff={eff_wet:.6e} ≥ Dry SKR_eff={eff_dry:.6e}"
        )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    sys.exit(result.returncode)
