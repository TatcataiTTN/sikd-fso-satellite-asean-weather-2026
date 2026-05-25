# Debug2.md — Phân tích Routing Algorithm: Baseline vs Greedy

## Ngày: 2026-05-25

---

## 1. Vấn đề gốc

Khi chạy fig13b/c/d cho 500, 1584, 4000 vệ tinh:
- Baseline cũ (alphabetical): improvement lên hàng triệu % → vô nghĩa
- Baseline mới (best elevation): improvement = 0% → greedy ≡ baseline
- Baseline median: improvement vài % → hợp lý hơn nhưng chưa đúng bản chất

**Câu hỏi cốt lõi**: Greedy routing thực sự tốt hơn "không routing" ở điểm nào?

---

## 2. Phân tích Pipeline hiện tại

### 2.1 Flow tại mỗi time step

```
get_visible_satellites(satellites, lat, lon, t)
  → list of {name, elevation_deg, zenith_deg, slant_km}
  → sorted by elevation DESCENDING (best first)

compute_skr_all_visible(visible_sats, R_mm_h, V_km, P_cloud)
  → for each sat: compute_skr_for_satellite(zenith_deg, R_mm_h, V_km, P_cloud)
    → compute_channel(H_S_km, zenith_deg) → hg, sigma_X2
    → compute_hl(zenith_deg, V_km, R_mm_h=0) → hl_clear
    → compute_hl(zenith_deg, V_km, R_mm_h) → hl_rain
    → compute_sikd_performance(hg, hl_clear, sigma_X2) → SKR_clear
    → compute_sikd_performance(hg, hl_rain, sigma_X2) → SKR_rain
    → SKR_effective = p_clear × SKR_clear + p_rain × SKR_rain
  → sorted by SKR_effective DESCENDING

greedy_best_satellite(candidates) → candidates[0] (highest SKR_effective)
no_routing_satellite(candidates) → ???
```

### 2.2 Tại sao best_elevation ≡ best_SKR?

Trong mô hình hiện tại, SKR_effective phụ thuộc:
- `hg` = geometric loss = f(zenith_deg) — GIẢM khi zenith tăng
- `hl` = atmospheric loss = exp(-σ/cos(zenith)) — GIẢM khi zenith tăng
- `sigma_X2` = turbulence = f(zenith_deg) — TĂNG khi zenith tăng

**Tất cả đều monotonic theo zenith angle!**

→ Vệ tinh có zenith thấp nhất (elevation cao nhất) LUÔN có SKR cao nhất
→ Greedy (best SKR) ≡ Best elevation
→ Không có trade-off nào giữa các vệ tinh

### 2.3 Khi nào routing TẠO SỰ KHÁC BIỆT?

Routing chỉ có ý nghĩa khi có **trade-off** giữa các vệ tinh:

1. **Weather thay đổi theo hướng** (spatial weather diversity):
   - Vệ tinh A: elevation 80° nhưng hướng Bắc → mây dày
   - Vệ tinh B: elevation 50° nhưng hướng Nam → trời quang
   - → B tốt hơn A dù elevation thấp hơn

2. **Multi-hop routing** (ISL — inter-satellite link):
   - Chọn đường đi qua nhiều vệ tinh để tránh vùng mây

3. **Handover optimization**:
   - Vệ tinh A đang ở 70° nhưng sắp lặn (elevation giảm)
   - Vệ tinh B đang ở 40° nhưng đang lên (elevation tăng)
   - → Chọn B để tránh handover sớm

---

## 3. Mô hình Baseline ĐÚNG cho paper

### 3.1 Baseline: "Fixed satellite" (không handover)

**Ý tưởng**: Trạm mặt đất kết nối với 1 vệ tinh duy nhất cho đến khi nó lặn dưới horizon, rồi mới chuyển sang vệ tinh tiếp theo (theo thứ tự thời gian xuất hiện).

**Hành vi thực tế**:
- Vệ tinh LEO bay qua trong ~10 phút (1 pass)
- Trong pass đó: elevation tăng → đạt max → giảm
- SKR theo hình chuông: thấp → cao → thấp
- Giữa các pass: GAP (không có vệ tinh → SKR = 0)

**Đây là baseline hợp lý nhất** vì:
- Mô phỏng hệ thống KHÔNG có routing intelligence
- Trạm mặt đất chỉ track 1 vệ tinh tại 1 thời điểm
- Khi vệ tinh lặn, chờ vệ tinh tiếp theo xuất hiện
- Không chọn lựa — chỉ dùng vệ tinh "tiếp theo" (next available)

### 3.2 Greedy: "Best satellite at each step"

**Ý tưởng**: Tại mỗi time step, chọn vệ tinh có SKR cao nhất trong số tất cả visible.

**Hành vi**:
- Luôn dùng vệ tinh gần thiên đỉnh nhất
- Handover liên tục (chuyển vệ tinh khi có cái tốt hơn)
- Không có gap (nếu có ≥1 visible)
- SKR luôn ở mức cao

### 3.3 Sự khác biệt Greedy vs Fixed

| Aspect | Fixed (baseline) | Greedy |
|--------|-----------------|--------|
| Chọn vệ tinh | Next available | Best available |
| Handover | Chỉ khi vệ tinh lặn | Liên tục |
| SKR profile | Hình chuông (1 pass) | Phẳng ở mức cao |
| Gap time | Có (giữa các pass) | Không (nếu N đủ lớn) |
| Improvement source | Giảm gap + tăng avg SKR | — |

### 3.4 Với N vệ tinh khác nhau

- **N = 176**: Fixed có nhiều gap → Greedy cải thiện đáng kể (giảm gap)
- **N = 500**: Fixed ít gap hơn → Greedy cải thiện vừa (tăng avg SKR)
- **N = 1584**: Fixed gần continuous → Greedy cải thiện nhỏ (chỉ tăng avg)
- **N = 4000**: Fixed continuous → Greedy cải thiện rất nhỏ (diminishing returns)

---

## 4. Algorithm mới cần implement

### 4.1 `fixed_satellite_baseline()`

```python
def fixed_satellite_baseline(satellites, lat, lon, t_array, min_elev):
    """
    Baseline: track 1 vệ tinh cho đến khi nó lặn, rồi chuyển sang
    vệ tinh tiếp theo xuất hiện (first-come-first-served).

    Algorithm:
    1. current_sat = None
    2. For each time step t:
       a. If current_sat is visible (elev >= min_elev):
          → continue tracking current_sat
          → SKR = compute_skr(current_sat.zenith)
       b. Else (current_sat lặn hoặc chưa có):
          → Tìm vệ tinh visible tiếp theo (first in list)
          → current_sat = first_visible
          → If none visible: SKR = 0 (GAP)
    """
```

### 4.2 `greedy_routing()`

```python
def greedy_routing(satellites, lat, lon, t_array, min_elev):
    """
    Greedy: tại mỗi step, chọn vệ tinh có SKR cao nhất.

    Algorithm:
    1. For each time step t:
       a. Find all visible satellites
       b. Compute SKR for each
       c. Select satellite with max SKR_effective
       d. If none visible: SKR = 0
    """
```

### 4.3 Kỳ vọng kết quả

Với mô hình constant weather (R_mm_h, V_km cố định):
- Greedy = best elevation (vì SKR monotonic theo zenith)
- Fixed = track 1 vệ tinh qua cả pass (elevation thay đổi)

**Improvement sources**:
1. **Gap reduction**: Fixed có gap giữa passes, Greedy không (nếu N đủ)
2. **Better average**: Fixed dùng vệ tinh ở elevation thấp (đầu/cuối pass), Greedy luôn dùng vệ tinh ở elevation cao nhất

---

## 5. Dự đoán kết quả sau fix

| N_sats | Fixed gaps | Greedy gaps | Improvement (est.) |
|--------|-----------|-------------|-------------------|
| 176 | Nhiều (~40-60% time) | Ít (~10-20%) | +80-150% |
| 500 | Ít (~10-20%) | Gần 0 | +30-60% |
| 1584 | Gần 0 | 0 | +10-20% |
| 4000 | 0 | 0 | +2-5% |

Đây là pattern "diminishing returns" mà paper cần chứng minh.

---

## 6. Implementation Plan

1. Implement `fixed_satellite_baseline()` trong `routing.py`
2. Sửa `compute_skr_timeseries()` để dùng baseline mới
3. Giữ nguyên `greedy_best_satellite()` (đã đúng)
4. Fix test `test_no_routing_returns_first_alphabetically`
5. Re-run fig13 gốc (176 sats) → verify improvement ~86-176%
6. Re-run fig13b/c/d → verify diminishing returns pattern
7. Update fig13_skr_timeseries_routing.png

---

## 7. Tại sao vệ tinh "cách xa" Hà Nội trên bản đồ?

**Hoàn toàn bình thường.** Giải thích:

- Vệ tinh LEO ở 550 km, inclination 53°
- Tại bất kỳ thời điểm t, vệ tinh phân bố trên toàn dải 53°S–53°N
- Footprint (vùng nhìn thấy) từ 1 điểm: bán kính ~2400 km
- Trên bản đồ ASEAN (95-130°E, 0-25°N):
  - Chỉ vệ tinh trong bán kính 2400 km từ Hà Nội mới visible
  - Nhưng ta vẽ TẤT CẢ vệ tinh trên bản đồ (kể cả không visible)
  - → Nhiều chấm ở xa là bình thường

- fig15 (elevation heatmap) cho thấy: vùng nào có vệ tinh ở góc cao
  - Màu đỏ đậm = có vệ tinh gần thiên đỉnh (elevation > 70°)
  - Màu vàng nhạt = chỉ có vệ tinh ở gần horizon (elevation 10-30°)
  - Trắng = không có vệ tinh visible

---

## 8. Giải thích 8×22, 10×50, 72×22, 40×100

```
Walker-Delta Constellation:
═══════════════════════════

n_planes = số mặt phẳng quỹ đạo (orbital planes)
sats_per_plane = số vệ tinh trong mỗi plane

Hình dung:
- Trái Đất ở giữa
- Mỗi plane = 1 vòng tròn lớn nghiêng 53° so với xích đạo
- Các plane xoay đều quanh trục Trái Đất (RAAN spacing = 360°/n_planes)
- Trong mỗi plane, vệ tinh cách đều (mean anomaly spacing = 360°/sats_per_plane)

Ví dụ 8×22 = 176:
  Plane 0: RAAN=0°,   22 sats cách 16.4°
  Plane 1: RAAN=45°,  22 sats cách 16.4°
  Plane 2: RAAN=90°,  22 sats cách 16.4°
  ...
  Plane 7: RAAN=315°, 22 sats cách 16.4°

Tại sao 72×22 = 1584 (Starlink Gen1)?
  - 72 planes → RAAN spacing = 5° → phủ kín mọi hướng
  - 22 sats/plane → cách 16.4° → phủ kín dọc quỹ đạo
  - → Continuous global coverage ở mọi vĩ độ < 53°
```
