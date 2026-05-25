# Debug1.md — Tổng hợp công việc ngày 2026-05-25

## Ngày: 2026-05-25

---

## 1. Các file đã tạo mới

| File | Mô tả |
|------|--------|
| `06_Constellation_Density_Figures.py` | Script tạo fig15 (2×2 grid) + fig13 (4-panel SKR comparison) |
| `07_Fig13_Fig15_Density.py` | Script tạo fig13b/c/d + fig15b/c/d riêng lẻ (y hệt style gốc) |
| `constellation_density_analysis.py` | Chứng minh bằng code: diminishing returns sau ~1500 sats |
| `Debug2.md` | Phân tích chi tiết routing algorithm (baseline vs greedy) |

## 2. Figures đã tạo

| Figure | File | Mô tả |
|--------|------|--------|
| fig13b | `fig13b_skr_timeseries_routing_500sats.png` | SKR timeseries 500 sats (dry+wet) |
| fig13c | `fig13c_skr_timeseries_routing_1584sats.png` | SKR timeseries 1584 sats |
| fig13d | `fig13d_skr_timeseries_routing_4000sats.png` | SKR timeseries 4000 sats |
| fig15b | `fig15b_coverage_map_500sats.png` | Coverage map ASEAN 500 sats |
| fig15c | `fig15c_coverage_map_1584sats.png` | Coverage map ASEAN 1584 sats |
| fig15d | `fig15d_coverage_map_4000sats.png` | Coverage map ASEAN 4000 sats |
| fig15 grid | `fig15_coverage_density_comparison.png` | 2×2 grid so sánh 4 kích thước |
| fig13 grid | `fig13_skr_timeseries_density_comparison.png` | 4-panel SKR comparison |

## 3. Thay đổi code quan trọng

### 3.1 `modules/routing.py` — Baseline algorithm

**Trước**: `no_routing_satellite()` chọn vệ tinh đầu tiên theo alphabet
- Vấn đề: với N lớn (1584+), luôn chọn cùng 1 vệ tinh → SKR cực thấp → improvement hàng triệu %

**Sau**: Fixed-satellite baseline trong `compute_skr_timeseries()`:
1. Track vệ tinh hiện tại cho đến khi lặn (elevation < 10°)
2. Khi lặn, chọn vệ tinh **mới xuất hiện** (just risen, elevation thấp nhất)
3. Nếu không có vệ tinh mới, chọn vệ tinh ở **median elevation**
4. Mô phỏng: hệ thống đơn giản chỉ kết nối vệ tinh tiếp theo bay qua

### 3.2 Kết quả routing improvement (mới)

| N_sats | Dry (Jan) | Wet (Jul) | Avg Greedy (kbps) | Avg Baseline (kbps) |
|--------|-----------|-----------|-------------------|---------------------|
| 176 | +108% | +209% | 7,450 | 3,589 |
| 500 | +110% | +225% | 7,813 | 3,714 |
| 1584 | +174% | +378% | 10,667 | 3,886 |
| 4000 | +270% | +906% | 10,668 | 2,880 |

### 3.3 Insight: Diminishing Returns

**Greedy SKR (absolute)** bão hòa:
- 176 → 500: 7,450 → 7,813 kbps (+5%)
- 500 → 1584: 7,813 → 10,667 kbps (+37%)
- 1584 → 4000: 10,667 → 10,668 kbps (+0.01%) ← **BÃO HÒA**

**Lý do**: Với 1584+ sats, luôn có vệ tinh gần thiên đỉnh (ζ < 5°).
Thêm vệ tinh không cải thiện ζ_min → SKR không tăng.
Bottleneck chuyển sang P_cloud (55% thời gian link OFF).

## 4. Giải thích cấu hình Walker-Delta

```
n_planes × sats_per_plane = tổng số vệ tinh

8×22 = 176:  Starlink shell 1 baseline
              8 mặt phẳng quỹ đạo, mỗi plane 22 vệ tinh
              RAAN spacing = 45°, MA spacing = 16.4°

10×50 = 500: Mở rộng vừa
              10 planes, 50 sats/plane
              RAAN spacing = 36°, MA spacing = 7.2°

72×22 = 1584: Starlink Gen1 full
              72 planes, 22 sats/plane
              RAAN spacing = 5°, MA spacing = 16.4°
              → Continuous global coverage

40×100 = 4000: Mega-constellation
              40 planes, 100 sats/plane
              RAAN spacing = 9°, MA spacing = 3.6°
              → Overkill, diminishing returns
```

## 5. Vấn đề còn tồn tại

### 5.1 Routing improvement tăng theo N (không giảm)

Improvement % tăng (108% → 174% → 270%) vì:
- Greedy tốt hơn (luôn có vệ tinh ở thiên đỉnh)
- Baseline tệ hơn (track 1 vệ tinh qua cả pass, bao gồm lúc elevation thấp)
- Với N lớn, greedy "nhảy" giữa nhiều vệ tinh → luôn ở peak
- Baseline vẫn phải chịu elevation thấp ở đầu/cuối pass

**Đây không phải bug** — đây là hành vi đúng. "Diminishing returns" là về **absolute SKR** (greedy), không phải improvement %.

### 5.2 Constant weather limitation

Trong mô hình hiện tại, weather (R_mm_h, V_km) là constant cho toàn bộ sky.
→ Vệ tinh có elevation cao nhất LUÔN có SKR cao nhất
→ Greedy ≡ Best elevation
→ Routing improvement chỉ đến từ handover optimization, không từ weather-adaptive

Để có weather-adaptive routing thực sự, cần:
- Spatial weather diversity (mây ở hướng A, quang ở hướng B)
- Hoặc temporal weather changes (mưa bắt đầu/kết thúc trong simulation)

## 6. Tests

Tất cả 38 routing tests PASS sau thay đổi.
Tất cả 207 module tests vẫn PASS (không ảnh hưởng).

## 7. Commands để reproduce

```bash
cd "/Users/tuannghiat/Downloads/Bài Quantum Communication PTIT /05_Code"

# Chạy fig13b/c/d + fig15b/c/d
python 07_Fig13_Fig15_Density.py

# Chạy fig13/fig15 density comparison (2×2 grid)
python 06_Constellation_Density_Figures.py

# Chạy constellation density analysis (chứng minh diminishing returns)
python constellation_density_analysis.py

# Run tests
python -m pytest test_routing.py -q
```
