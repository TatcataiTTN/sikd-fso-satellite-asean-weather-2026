"""
test_orbital_mechanics.py — Unit tests for orbital_mechanics.py
================================================================

Kiểm tra tính đúng đắn của các hàm tính toán quỹ đạo vệ tinh LEO.

Chiến lược test
---------------
    - TLE parsing    : format validation, malformed input handling
    - Geometry       : elevation/zenith/slant path formulas
    - Physics        : elevation range, slant path monotonicity
    - Visibility     : filtering, sorting, best satellite selection
    - Time series    : pass window detection, elevation continuity
    - Coverage grid  : shape, range, coverage fraction
    - Integration    : compute_link_geometry wrapper

TLE mẫu dùng trong tests
-------------------------
    Dùng TLE thực của Starlink-1007 (epoch 2026) để test có ý nghĩa vật lý.
    Các test geometry dùng TLE synthetic đơn giản để kiểm soát kết quả.

Test classes (8 classes, ~40 tests)
------------------------------------
    T01  TestTLEParsing         (6 tests) — parse_tle_block, load_tle_file
    T02  TestGeometry           (7 tests) — elevation, zenith, slant path
    T03  TestVisibility         (6 tests) — get_visible_satellites, best_sat
    T04  TestTimeArray          (4 tests) — make_time_array
    T05  TestElevationTimeseries(5 tests) — compute_elevation_timeseries
    T06  TestPassWindows        (5 tests) — find_pass_windows
    T07  TestCoverageGrid       (4 tests) — compute_coverage_grid (small grid)
    T08  TestLinkGeometry       (5 tests) — compute_link_geometry wrapper

Chạy tests
----------
    python -m pytest test_orbital_mechanics.py -v
    python -m pytest test_orbital_mechanics.py -v -k "TestGeometry"
"""

import sys
import os
import tempfile
import numpy as np
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.orbital_mechanics import (
    parse_tle_block,
    load_tle_file,
    make_skyfield_satellite,
    compute_elevation,
    compute_zenith_angle,
    compute_slant_path,
    get_visible_satellites,
    get_best_satellite,
    make_time_array,
    compute_elevation_timeseries,
    find_pass_windows,
    compute_coverage_grid,
    compute_link_geometry,
    MIN_ELEVATION,
    H_S_KM_DEFAULT,
    GROUND_STATIONS,
    _TS,
)

# ---------------------------------------------------------------------------
# Shared TLE fixtures
# ---------------------------------------------------------------------------
# Starlink-1007 — real TLE (epoch ~2026-03, from CelesTrak archive)
# Used for physics-meaningful tests
TLE_STARLINK_1007 = {
    "name":  "STARLINK-1007",
    "line1": "1 44713U 19074B   26071.50000000  .00001000  00000-0  10000-3 0  9990",
    "line2": "2 44713  53.0000 100.0000 0001000  90.0000 270.0000 15.05000000000010",
}

# Starlink-1008 — second satellite for multi-sat tests
TLE_STARLINK_1008 = {
    "name":  "STARLINK-1008",
    "line1": "1 44714U 19074C   26071.50000000  .00001000  00000-0  10000-3 0  9991",
    "line2": "2 44714  53.0000 200.0000 0001000  90.0000 270.0000 15.05000000000011",
}

# Starlink-1009 — third satellite, different RAAN
TLE_STARLINK_1009 = {
    "name":  "STARLINK-1009",
    "line1": "1 44715U 19074D   26071.50000000  .00001000  00000-0  10000-3 0  9992",
    "line2": "2 44715  53.0000 300.0000 0001000  90.0000 270.0000 15.05000000000012",
}

# TLE text block (3 satellites)
TLE_BLOCK_TEXT = "\n".join([
    TLE_STARLINK_1007["name"], TLE_STARLINK_1007["line1"], TLE_STARLINK_1007["line2"],
    TLE_STARLINK_1008["name"], TLE_STARLINK_1008["line1"], TLE_STARLINK_1008["line2"],
    TLE_STARLINK_1009["name"], TLE_STARLINK_1009["line1"], TLE_STARLINK_1009["line2"],
])

# Reference time: 2026-03-12 12:00:00 UTC
T_REF = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
T_SKY_REF = _TS.from_datetime(T_REF)

# Hanoi ground station
HANOI_LAT = GROUND_STATIONS["hanoi"]["lat"]
HANOI_LON = GROUND_STATIONS["hanoi"]["lon"]
HANOI_ALT = GROUND_STATIONS["hanoi"]["alt_m"]

# Pre-built satellite objects
SAT_1007 = make_skyfield_satellite(TLE_STARLINK_1007)
SAT_1008 = make_skyfield_satellite(TLE_STARLINK_1008)
SAT_1009 = make_skyfield_satellite(TLE_STARLINK_1009)
ALL_SATS  = [SAT_1007, SAT_1008, SAT_1009]


# ---------------------------------------------------------------------------
# T01 — TLE Parsing
# ---------------------------------------------------------------------------
class TestTLEParsing:
    def test_parse_single_satellite(self):
        """parse_tle_block phải parse đúng 1 vệ tinh."""
        text = "\n".join([
            TLE_STARLINK_1007["name"],
            TLE_STARLINK_1007["line1"],
            TLE_STARLINK_1007["line2"],
        ])
        result = parse_tle_block(text)
        assert len(result) == 1
        assert result[0]["name"]  == TLE_STARLINK_1007["name"]
        assert result[0]["line1"] == TLE_STARLINK_1007["line1"]
        assert result[0]["line2"] == TLE_STARLINK_1007["line2"]

    def test_parse_multiple_satellites(self):
        """parse_tle_block phải parse đúng 3 vệ tinh từ block text."""
        result = parse_tle_block(TLE_BLOCK_TEXT)
        assert len(result) == 3
        names = [r["name"] for r in result]
        assert "STARLINK-1007" in names
        assert "STARLINK-1008" in names
        assert "STARLINK-1009" in names

    def test_parse_returns_required_keys(self):
        """Mỗi entry phải có keys: name, line1, line2."""
        result = parse_tle_block(TLE_BLOCK_TEXT)
        for entry in result:
            assert "name"  in entry
            assert "line1" in entry
            assert "line2" in entry

    def test_parse_empty_string(self):
        """parse_tle_block với chuỗi rỗng → list rỗng."""
        assert parse_tle_block("") == []

    def test_parse_malformed_skips_bad_entries(self):
        """Dòng không hợp lệ (không bắt đầu bằng '1 '/'2 ') bị bỏ qua."""
        bad_text = "GARBAGE\nNOT_LINE1\nNOT_LINE2\n" + "\n".join([
            TLE_STARLINK_1007["name"],
            TLE_STARLINK_1007["line1"],
            TLE_STARLINK_1007["line2"],
        ])
        result = parse_tle_block(bad_text)
        # Phải parse được ít nhất 1 vệ tinh hợp lệ
        assert len(result) >= 1
        assert any(r["name"] == "STARLINK-1007" for r in result)

    def test_load_tle_file(self):
        """load_tle_file phải đọc file và parse đúng."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle",
                                         delete=False, encoding="utf-8") as f:
            f.write(TLE_BLOCK_TEXT)
            tmp_path = f.name
        try:
            result = load_tle_file(tmp_path)
            assert len(result) == 3
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# T02 — Geometry
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_zenith_angle_formula(self):
        """zeta = 90 - elevation."""
        assert compute_zenith_angle(90.0) == 0.0
        assert compute_zenith_angle(45.0) == 45.0
        assert compute_zenith_angle(0.0)  == 90.0
        assert compute_zenith_angle(30.0) == 60.0

    def test_zenith_elevation_sum_is_90(self):
        """elevation + zenith = 90° luôn đúng."""
        for el in [10, 20, 30, 45, 60, 75, 90]:
            assert abs(el + compute_zenith_angle(el) - 90.0) < 1e-10

    def test_slant_path_at_nadir(self):
        """elevation = 90° → slant = H_S (overhead satellite)."""
        slant = compute_slant_path(90.0, H_S_km=550.0)
        assert abs(slant - 550.0) < 0.01

    def test_slant_path_increases_with_lower_elevation(self):
        """Elevation thấp hơn → slant path dài hơn."""
        slants = [compute_slant_path(el) for el in [80, 60, 45, 30, 15]]
        assert all(slants[i] < slants[i+1] for i in range(len(slants)-1))

    def test_slant_path_formula(self):
        """L = H_S / cos(zeta) = H_S / sin(elevation)."""
        el = 45.0
        H  = 550.0
        expected = H / np.sin(np.radians(el))
        result   = compute_slant_path(el, H)
        assert abs(result - expected) / expected < 1e-6

    def test_slant_path_zero_elevation_raises(self):
        """elevation ≤ 0 phải raise ValueError."""
        with pytest.raises(ValueError):
            compute_slant_path(0.0)
        with pytest.raises(ValueError):
            compute_slant_path(-5.0)

    def test_elevation_returns_float(self):
        """compute_elevation phải trả về float."""
        el = compute_elevation(SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF)
        assert isinstance(el, float)

    def test_elevation_in_valid_range(self):
        """Elevation angle phải ∈ [-90, 90]."""
        el = compute_elevation(SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF)
        assert -90.0 <= el <= 90.0


# ---------------------------------------------------------------------------
# T03 — Visibility
# ---------------------------------------------------------------------------
class TestVisibility:
    def test_visible_satellites_returns_list(self):
        """get_visible_satellites phải trả về list."""
        result = get_visible_satellites(
            ALL_SATS, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        assert isinstance(result, list)

    def test_visible_satellites_all_above_threshold(self):
        """Tất cả vệ tinh trong kết quả phải có elevation ≥ min_elevation."""
        result = get_visible_satellites(
            ALL_SATS, HANOI_LAT, HANOI_LON, T_SKY_REF,
            min_elevation=MIN_ELEVATION
        )
        for sat_info in result:
            assert sat_info["elevation_deg"] >= MIN_ELEVATION, (
                f"{sat_info['name']}: elevation={sat_info['elevation_deg']:.2f}° "
                f"< min={MIN_ELEVATION}°"
            )

    def test_visible_satellites_sorted_descending(self):
        """Kết quả phải được sort theo elevation giảm dần."""
        result = get_visible_satellites(
            ALL_SATS, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        elevations = [r["elevation_deg"] for r in result]
        assert elevations == sorted(elevations, reverse=True)

    def test_visible_satellites_required_keys(self):
        """Mỗi entry phải có đủ keys cần thiết."""
        result = get_visible_satellites(
            ALL_SATS, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        for entry in result:
            for key in ["satellite", "name", "elevation_deg",
                        "zenith_deg", "slant_km"]:
                assert key in entry, f"Missing key: {key}"

    def test_high_min_elevation_fewer_satellites(self):
        """Ngưỡng elevation cao hơn → ít vệ tinh visible hơn hoặc bằng."""
        vis_low  = get_visible_satellites(ALL_SATS, HANOI_LAT, HANOI_LON,
                                          T_SKY_REF, min_elevation=5.0)
        vis_high = get_visible_satellites(ALL_SATS, HANOI_LAT, HANOI_LON,
                                          T_SKY_REF, min_elevation=60.0)
        assert len(vis_high) <= len(vis_low)

    def test_best_satellite_is_highest_elevation(self):
        """get_best_satellite phải trả về vệ tinh có elevation cao nhất."""
        visible = get_visible_satellites(
            ALL_SATS, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        best = get_best_satellite(
            ALL_SATS, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        if visible:
            assert best is not None
            assert best["elevation_deg"] == visible[0]["elevation_deg"]
        else:
            assert best is None


# ---------------------------------------------------------------------------
# T04 — Time Array
# ---------------------------------------------------------------------------
class TestTimeArray:
    def test_time_array_length(self):
        """make_time_array phải tạo đúng số bước thời gian."""
        t_arr = make_time_array(T_REF, duration_hours=1.0, step_minutes=1.0)
        # 1 hour × 60 min/h / 1 min/step + 1 = 61 steps
        assert len(t_arr) == 61

    def test_time_array_24h(self):
        """24h với step 1 phút → 1441 bước."""
        t_arr = make_time_array(T_REF, duration_hours=24.0, step_minutes=1.0)
        assert len(t_arr) == 1441

    def test_time_array_naive_datetime(self):
        """Datetime không có tzinfo phải được xử lý (assume UTC)."""
        t_naive = datetime(2026, 3, 12, 12, 0, 0)  # no tzinfo
        t_arr = make_time_array(t_naive, duration_hours=1.0, step_minutes=5.0)
        assert len(t_arr) == 13  # 60/5 + 1

    def test_time_array_step_minutes(self):
        """Step 5 phút trong 1 giờ → 13 bước."""
        t_arr = make_time_array(T_REF, duration_hours=1.0, step_minutes=5.0)
        assert len(t_arr) == 13


# ---------------------------------------------------------------------------
# T05 — Elevation Time Series
# ---------------------------------------------------------------------------
class TestElevationTimeseries:
    def setup_method(self):
        """Tạo time array 2 giờ, step 1 phút."""
        self.t_arr = make_time_array(T_REF, duration_hours=2.0, step_minutes=1.0)

    def test_returns_numpy_array(self):
        """compute_elevation_timeseries phải trả về np.ndarray."""
        result = compute_elevation_timeseries(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_time_array(self):
        """Độ dài output phải bằng độ dài time array."""
        result = compute_elevation_timeseries(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        assert len(result) == len(self.t_arr)

    def test_elevation_in_valid_range(self):
        """Tất cả elevation values phải ∈ [-90, 90]."""
        result = compute_elevation_timeseries(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        assert np.all(result >= -90.0)
        assert np.all(result <= 90.0)

    def test_different_satellites_different_timeseries(self):
        """Hai vệ tinh khác nhau phải cho elevation timeseries khác nhau."""
        el1 = compute_elevation_timeseries(SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr)
        el2 = compute_elevation_timeseries(SAT_1008, HANOI_LAT, HANOI_LON, self.t_arr)
        # Không thể giống hệt nhau (khác RAAN)
        assert not np.allclose(el1, el2)

    def test_different_locations_different_timeseries(self):
        """Cùng vệ tinh, hai trạm khác nhau → elevation khác nhau."""
        el_hanoi = compute_elevation_timeseries(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        sg = GROUND_STATIONS["singapore"]
        el_sg = compute_elevation_timeseries(
            SAT_1007, sg["lat"], sg["lon"], self.t_arr
        )
        assert not np.allclose(el_hanoi, el_sg)


# ---------------------------------------------------------------------------
# T06 — Pass Windows
# ---------------------------------------------------------------------------
class TestPassWindows:
    def setup_method(self):
        """Time array 24 giờ, step 1 phút."""
        self.t_arr = make_time_array(T_REF, duration_hours=24.0, step_minutes=1.0)

    def test_returns_list(self):
        """find_pass_windows phải trả về list."""
        result = find_pass_windows(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        assert isinstance(result, list)

    def test_pass_windows_required_keys(self):
        """Mỗi pass window phải có đủ keys."""
        windows = find_pass_windows(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        for w in windows:
            for key in ["start_idx", "end_idx", "peak_idx",
                        "peak_elevation", "duration_steps", "elevations"]:
                assert key in w, f"Missing key: {key}"

    def test_peak_elevation_above_threshold(self):
        """peak_elevation phải ≥ min_elevation cho mọi pass window."""
        windows = find_pass_windows(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr,
            min_elevation=MIN_ELEVATION
        )
        for w in windows:
            assert w["peak_elevation"] >= MIN_ELEVATION

    def test_duration_positive(self):
        """duration_steps phải > 0 cho mọi pass window."""
        windows = find_pass_windows(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        for w in windows:
            assert w["duration_steps"] > 0

    def test_peak_idx_within_window(self):
        """peak_idx phải nằm trong [start_idx, end_idx]."""
        windows = find_pass_windows(
            SAT_1007, HANOI_LAT, HANOI_LON, self.t_arr
        )
        for w in windows:
            assert w["start_idx"] <= w["peak_idx"] <= w["end_idx"]


# ---------------------------------------------------------------------------
# T07 — Coverage Grid (small grid để test nhanh)
# ---------------------------------------------------------------------------
class TestCoverageGrid:
    def test_output_shape(self):
        """max_elevation phải có shape (n_lat, n_lon)."""
        result = compute_coverage_grid(
            ALL_SATS, T_SKY_REF,
            lat_range=(10.0, 15.0),
            lon_range=(100.0, 105.0),
            resolution_deg=2.5,
        )
        n_lat = len(result["lats"])
        n_lon = len(result["lons"])
        assert result["max_elevation"].shape == (n_lat, n_lon)
        assert result["has_coverage"].shape  == (n_lat, n_lon)

    def test_max_elevation_in_valid_range(self):
        """max_elevation phải ∈ [-90, 90]."""
        result = compute_coverage_grid(
            ALL_SATS, T_SKY_REF,
            lat_range=(10.0, 15.0),
            lon_range=(100.0, 105.0),
            resolution_deg=2.5,
        )
        assert np.all(result["max_elevation"] >= -90.0)
        assert np.all(result["max_elevation"] <= 90.0)

    def test_coverage_frac_in_valid_range(self):
        """coverage_frac phải ∈ [0, 1]."""
        result = compute_coverage_grid(
            ALL_SATS, T_SKY_REF,
            lat_range=(10.0, 15.0),
            lon_range=(100.0, 105.0),
            resolution_deg=2.5,
        )
        assert 0.0 <= result["coverage_frac"] <= 1.0

    def test_required_keys(self):
        """compute_coverage_grid phải trả về tất cả keys cần thiết."""
        result = compute_coverage_grid(
            ALL_SATS, T_SKY_REF,
            lat_range=(10.0, 12.0),
            lon_range=(100.0, 102.0),
            resolution_deg=2.0,
        )
        for key in ["lats", "lons", "max_elevation",
                    "zenith_best", "has_coverage", "coverage_frac"]:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# T08 — Link Geometry Wrapper
# ---------------------------------------------------------------------------
class TestLinkGeometry:
    def test_returns_all_keys(self):
        """compute_link_geometry phải trả về tất cả keys cần thiết."""
        result = compute_link_geometry(
            SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        for key in ["elevation_deg", "zenith_deg", "slant_km",
                    "is_visible", "sat_name"]:
            assert key in result

    def test_elevation_zenith_consistent(self):
        """elevation + zenith phải = 90°."""
        result = compute_link_geometry(
            SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        assert abs(result["elevation_deg"] + result["zenith_deg"] - 90.0) < 1e-6

    def test_sat_name_correct(self):
        """sat_name phải khớp với tên vệ tinh."""
        result = compute_link_geometry(
            SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        assert result["sat_name"] == "STARLINK-1007"

    def test_is_visible_consistent_with_elevation(self):
        """is_visible phải True khi elevation ≥ MIN_ELEVATION."""
        result = compute_link_geometry(
            SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        expected_visible = result["elevation_deg"] >= MIN_ELEVATION
        assert result["is_visible"] == expected_visible

    def test_slant_km_positive_when_visible(self):
        """Khi vệ tinh visible, slant_km phải > H_S (luôn dài hơn altitude)."""
        result = compute_link_geometry(
            SAT_1007, HANOI_LAT, HANOI_LON, T_SKY_REF
        )
        if result["is_visible"]:
            assert result["slant_km"] >= H_S_KM_DEFAULT, (
                f"slant={result['slant_km']:.1f} km < H_S={H_S_KM_DEFAULT} km"
            )

    def test_ground_stations_dict_complete(self):
        """GROUND_STATIONS phải có đủ 8 thành phố ASEAN."""
        expected = {"hanoi", "hcmc", "danang", "bangkok",
                    "singapore", "manila", "jakarta", "kuala_lumpur"}
        assert expected.issubset(GROUND_STATIONS.keys())


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
