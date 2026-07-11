# Cách hiểu Store-and-Forward Relay — giải thích chi tiết cho `fig10_isl_relay_latency_heatmap.png`

> **Cập nhật 08/07/2026 — model hiện dùng là ISL multi-hop (Task 24.3).**
> Model cũ (single-hop, chỉ 1 vệ tinh trung gian phải tự quay lại đúng vị
> trí) đã bị loại bỏ hoàn toàn khỏi paper vì sai kiến trúc (sinh số liệu
> vô lý: bất đối xứng jakarta↔manila 102×, tới 711 phút một chiều). Không
> còn giữ song song để so sánh — chỉ dùng ISL model. **Đọc mục 10 trước**
> (model hiện tại). Mục 1-9 mô tả model CŨ (single-hop) — chỉ giữ làm tài
> liệu lịch sử, không dùng cho paper nữa. Code/CSV của model cũ vẫn còn
> trên đĩa (`modules/citypair_feasibility.py`, `sf_relay_detail_56pairs.csv`)
> nhưng không còn dùng cho paper.

Folder này giải thích **từng bước tính toán** đứng sau
`latex_paper_3/figures/fig10_isl_relay_latency_heatmap.png`, sinh ra bởi
`05_Code_v2/scripts/12a_build_isl_graph.py` + `12b_isl_relay_recompute.py`
(Task 24.3, xem `07-5.Plan_Optimization_Redesign.md`).

Xem sơ đồ logic đầy đủ ở [`relay_logic.mmd`](./relay_logic.mmd) (render bằng
`mmdc -i relay_logic.mmd -o relay_logic.png -w 1800 -b white`).

---

## 1. Bài toán: tại sao cần "relay"?

Hệ thống này là **trusted-node key exchange**: 2 thành phố (A, B) muốn chia sẻ
khóa QKD với nhau, nhưng vệ tinh chỉ nói chuyện được với MẶT ĐẤT (downlink),
không có liên kết vệ tinh-vệ tinh (inter-satellite link, ISL) — đây là quyết
định phạm vi cố ý của dự án (xem `07-5` mục scope).

Có 2 khả năng:

1. **DUAL (đồng thời):** nếu A và B đủ gần nhau, tồn tại những khoảnh khắc mà
   MỘT vệ tinh nhìn thấy CẢ HAI trạm cùng lúc (ở góc ngẩng ≥ ngưỡng mask) →
   vệ tinh phát khóa xuống cả 2 trạm trong cùng 1 lượt bay, không cần "chuyển
   tiếp" gì cả.
2. **SF (Store-and-Forward — chuyển tiếp lưu trữ):** nếu A và B quá xa nhau để
   một vệ tinh nhìn thấy cả hai cùng lúc, thì phải dùng "hộp thư trung gian":
   **CÙNG MỘT vệ tinh** bay qua A trước (nhận khóa từ A — "pickup"), lưu khóa
   trên vệ tinh, rồi bay tiếp và ghé qua B sau (giao khóa cho B —
   "drop-off"). Độ trễ giữa 2 lần đó gọi là **relay latency**.

`fig10` là bản đồ nhiệt 8×8 thể hiện: với mỗi cặp thành phố có hướng
(pickup → drop-off), độ trễ SF trung vị là bao nhiêu phút, hoặc nếu là cặp
DUAL thì tô xanh lá và ghi "DUAL".

---

## 2. Input: bảng pass 7 ngày (Task 5)

File nguồn: `data/intermediate/pass_table_8cities_7days_elev30.csv` (sinh bởi
`scripts/08_pass_analysis.py`). Mỗi dòng là **một lượt bay (pass)** của một vệ
tinh Starlink Shell-1 thật (TLE CelesTrak) qua MỘT trong 8 trạm mặt đất ASEAN,
với góc ngẩng cực đại ≥ 30° (mask hình học tối thiểu), trong cửa sổ 7 ngày cố
định bắt đầu tại `T_START`. Cột quan trọng: `sat_id`, `station`, `t_rise`
(giờ UTC vệ tinh mọc lên qua ngưỡng 30°), `t_set`, `max_elev_deg`.

**Lưu ý:** đây là hình học TĨNH (không phụ thuộc thời tiết hay ngày thật cụ
thể) — cùng một vệ tinh, cùng một quỹ đạo, lặp lại y hệt mỗi 7 ngày mô phỏng.
Đây là lý do các con số relay latency (7.0'/711.4' ...) là **thuộc tính cấu
trúc** của chòm vệ tinh + vị trí 8 trạm, không đổi theo ngày thời tiết cụ thể
— đã verify lại qua 3 epoch TLE khác nhau (xem `PROVENANCE.md` mục 6b).

---

## 3. Bước 1 — Phân loại DUAL vs SF (Task 6, tĩnh, đối xứng)

Hàm: `modules/link_geometry.classify_pair(dist_km, h_km, min_elev_deg)`

```text
ground_coverage_radius_km(h, elev) = R_E · ψ
  với ψ = arccos( R_E/(R_E+h) · cos(elev) ) − elev      (góc tâm Trái Đất)

dual_downlink_max_separation_km(h, elev) = 2 × ground_coverage_radius_km(h, elev)

classify_pair(dist_km, h, elev):
    DUAL   nếu dist_km ≤ dual_downlink_max_separation_km(h, elev)
    SF     nếu ngược lại
```

Trực giác: mỗi trạm có một "vòng tròn phủ sóng" bán kính `ground_coverage_
radius_km` quanh nó (nơi vệ tinh còn ở trên ngưỡng góc ngẩng). Hai trạm DUAL
được khi 2 vòng tròn đó **chồng lấn nhau** — tức khoảng cách 2 trạm ≤ tổng 2
bán kính = 2×bán kính (vì 2 trạm dùng chung 1 bán kính ở cùng độ cao vệ
tinh).

Bước này **đối xứng** (chỉ phụ thuộc khoảng cách tĩnh giữa 2 trạm, không phụ
thuộc hướng bay) — khác với latency SF ở bước sau.

Với cặp DUAL: có thêm `dual_pct` = % thời gian trong 24h có ≥1 vệ tinh nhìn
thấy đồng thời cả 2 trạm (mô phỏng time-resolved thật, `scripts/
09_citypair_feasibility.py`, KHÔNG dùng lại trong `11f_sf_relay_detail.py` mà
chỉ đọc lại từ CSV Task 6).

---

## 4. Bước 2 — Với mỗi cặp SF: tìm relay opportunities (bất đối xứng!)

Đây là phần lõi, hàm `relay_opportunities(city_i, city_j)` trong
`11f_sf_relay_detail.py` (định nghĩa giống hệt, đã cross-check, với
`modules/citypair_feasibility.sf_latency_minutes`).

**Định nghĩa 1 "relay opportunity":** một lượt vệ tinh S bay qua thành phố
`i` tại thời điểm mọc `t_pick` (pickup), sau đó CÙNG vệ tinh S đó bay qua
thành phố `j` tại thời điểm mọc `t_drop` GẦN NHẤT SAU `t_pick` (dropoff).
`latency = t_drop − t_pick` (phút).

Thuật toán (vector hóa bằng numpy, không vòng lặp lồng nhau chậm):

1. Gom pass theo `(station, sat_id)` → mảng `t_rise` đã sắp xếp thời gian
   tăng dần: `times_by[station][sat_id]`.
2. Với cặp `(city_i, city_j)`: lấy **giao** tập vệ tinh mà CẢ HAI trạm đều
   nhìn thấy trong 7 ngày (`sats_i.keys() & sats_j.keys()`) — vệ tinh không
   bay qua cả 2 trạm thì không thể relay giữa chúng.
3. Với mỗi vệ tinh chung đó, với MỖI lần nó ghé qua `city_i` (mỗi `t_pick`
   trong `t_i`), dùng `np.searchsorted(t_j, t_pick, side='right')` để tìm vị
   trí `t_drop` đầu tiên trong mảng `t_j` mà **lớn hơn** `t_pick` — chính là
   lần ghé qua `city_j` sớm nhất SAU lần pickup này.
4. Nếu tìm được (`idx < len(t_j)`), ghi lại `(latency_phút, sat_id, t_pick,
   t_drop)` vào danh sách "relay opportunities" của cặp đó.

**Vì sao bất đối xứng (i→j ≠ j→i)?** Vệ tinh Starlink Shell-1 có góc
nghiêng quỹ đạo cố định (~53°); trên một lượt bay, nó di chuyển theo MỘT
hướng cố định (ascending: nam→bắc, hoặc descending: bắc→nam). Nếu i ở phía
"trước" theo hướng bay đó và j ở phía "sau", thì i→j nhanh (chờ vài phút) —
nhưng j→i phải chờ vệ tinh **hoàn thành gần hết 1 vòng quỹ đạo** (~90-100
phút) rồi quay lại đúng hướng mới ghé j trước rồi tới i. Đây là lý do đo
được ví dụ cực đoan: `jakarta → manila` trung vị chỉ 7.0 phút, nhưng
`manila → jakarta` trung vị tới 711.4 phút (~11.85 giờ), gấp ~102 lần. Số
liệu này **tái lập ổn định qua 3 epoch TLE khác nhau** (xem `PROVENANCE.md`
mục 6b) — là thuộc tính cấu trúc, không phải nhiễu ngẫu nhiên.

Hai chiều `i→j` và `j→i` được tính **HOÀN TOÀN ĐỘC LẬP** (không suy ra chiều
này từ chiều kia) — đây là điểm sửa lỗi thiết kế quan trọng của Task 6 (bản
đầu từng nhầm gộp chung 1 ma trận đối xứng).

---

## 5. Bước 3 — Tổng hợp thống kê cho mỗi cặp có hướng

Từ danh sách relay opportunities của cặp `(i, j)`:

- `n_relay_opportunities` = số lượng cơ hội relay tìm được trong 7 ngày.
- `latency_min_min` / `latency_median_min` / `latency_max_min` = min/trung
  vị/max của độ trễ (phút) trên toàn bộ cơ hội.
- `top5_relay_sats` = 5 vệ tinh xuất hiện làm relay nhiều lần nhất (đếm tần
  suất `sat_id` trong danh sách cơ hội), định dạng
  `"STARLINK-XXXX(count);..."`.
- `fastest_sat` / `fastest_latency_min` = cơ hội relay NHANH NHẤT tìm được
  (không phải trung vị) + giờ địa phương pickup/dropoff tương ứng — cho thấy
  dù trung vị một hướng có thể rất chậm (~700'), vẫn luôn tồn tại một vài
  lượt cực nhanh (0.3–5') nếu chọn đúng vệ tinh — đây chính là "cơ hội" mà
  bộ lập lịch (ALG-2, Task 21) khai thác được.

Kết quả: **56 dòng có hướng** (8 thành phố × 7 thành phố còn lại, mỗi chiều
1 dòng) lưu vào `data/intermediate/sf_relay_detail_56pairs.csv`.

---

## 6. Bước 4 — Kiểm chứng chéo (cross-check) trước khi vẽ

Script tự so sánh kết quả trung vị tính "thủ công" (numpy, tối ưu tốc độ) với
kết quả của hàm module đã có unit test riêng
(`modules/citypair_feasibility.sf_latency_minutes`, vòng lặp Python thuần,
chậm hơn nhưng đã được `tests/test_citypair_feasibility.py` xác minh đúng
với fixture tổng hợp 10'/18' bất đối xứng). So sánh trên 3 cặp mẫu
(`jakarta→manila`, `manila→jakarta`, `hanoi→singapore`), yêu cầu lệch
< 0.05 phút. Nếu khớp → script mới không có lỗi logic trôi (logic drift) so
với hàm đã kiểm chứng.

## 7. Bước 5 — Vẽ `fig10_sf_relay_latency_heatmap.png`

- Ma trận 8×8 `median_matrix` (NaN ở ô DUAL và ô đường chéo `i=j`).
- `imshow` với colormap `YlOrRd` (1 hue liên tục — đúng chuẩn dataviz cho đại
  lượng magnitude, không dùng rainbow).
- Ô DUAL: phủ màu xanh lá nhạt (`#c8e6c9`) + chữ "DUAL" — tách biệt trực
  quan khỏi thang màu latency (vì DUAL không có "độ trễ", ý nghĩa khác hẳn).
- Ô SF: ghi số phút trung vị, màu chữ trắng nếu ô quá đậm (giá trị >
  60% giá trị lớn nhất) để vẫn đọc được.
- Trục X = thành phố NHẬN (drop-off `j`), trục Y = thành phố GỬI (pickup
  `i`) → đọc theo dòng để biết "từ trạm này, gửi đi các trạm khác mất bao
  lâu".

---

## 8. Giới hạn / điều cần biết khi trích số liệu

1. **Chỉ 1 vệ tinh relay (single-hop):** không xét trường hợp chuyển tiếp
   qua ≥ 2 vệ tinh khác nhau (đa chặng) — nằm ngoài phạm vi dự án (không có
   ISL, xem `07-5`).
2. **Mốc thời gian là `t_rise` (lúc mọc), không phải `t_set` hay đỉnh
   pass:** độ trễ đo từ lúc BẮT ĐẦU nhìn thấy vệ tinh ở trạm pickup tới lúc
   BẮT ĐẦU nhìn thấy nó ở trạm dropoff — là xấp xỉ hợp lý (nhất quán, đơn
   giản) nhưng không phải giờ khóa thực sự sẵn sàng (có thể trễ thêm vài
   phút bằng thời lượng pass).
3. **Mask 30° (Task 5/6), KHÁC với mask 40° "Mask B" dùng ở lập lịch thật
   (Task 21+):** đây là điểm KHÔNG NHẤT QUÁN đã biết giữa các phần của dự
   án — số liệu relay latency ở đây dùng ngưỡng hình học tối thiểu (30°),
   trong khi các thuật toán ALG-1/ALG-2 sinh khóa thật dùng ngưỡng bảo mật
   40°. Không trộn lẫn 2 con số này khi viết bài.
4. **Hình học tĩnh, không đổi theo ngày/thời tiết:** vì dùng bảng pass 7
   ngày cố định từ `T_START` — các con số relay là TÍNH CHẤT QUỸ ĐẠO, đã
   verify robust qua 3 epoch TLE (variant A/B/C, xem `PROVENANCE.md`), nhưng
   KHÔNG áp dụng để suy ra hoạt động thực tế của một ngày thời tiết cụ thể
   (đó là việc của Task 21-22, pass ledger).
5. **`fig09e` (Task 24.2) dùng lại đúng số liệu này:** cột `latency_median_
   min` của `sf_relay_detail_56pairs.csv` được nạp trực tiếp vào bảng top-5
   city-pair key exchanges để hiển thị độ trễ 2 chiều cho các cặp SF.

## 9. File liên quan (v1)

| File | Vai trò |
| --- | --- |
| `scripts/08_pass_analysis.py` | Sinh bảng pass 7 ngày (Task 5) — input gốc |
| `scripts/09_citypair_feasibility.py` | Phân loại DUAL/SF + dual_pct (Task 6, VẪN ĐÚNG — chỉ latency SF single-hop sai) |
| `modules/link_geometry.py` | `classify_pair`, `ground_coverage_radius_km`, `dual_downlink_max_separation_km` |
| `modules/citypair_feasibility.py` | `sf_latency_minutes` (hàm đã unit-test, dùng để cross-check) |
| `scripts/11f_sf_relay_detail.py` | Script chính sinh `fig10` + 56-row CSV (Task 23, SUPERSEDED) |
| `tests/test_citypair_feasibility.py` | Unit test xác nhận tính bất đối xứng đúng (fixture 10'/18') |
| `scripts/11g_orbit_maps.py` | `fig09e` dùng lại latency 2 chiều từ đây (v1, có thể cần đổi sang v2) |

## 10. Mô hình MỚI v2 — ISL Multi-Hop (Task 24.3, tóm tắt)

Thay vì chờ 1 vệ tinh duy nhất quay lại đúng vị trí, khóa được relay qua
**inter-satellite link (ISL)** — nhiều vệ tinh liền kề nối laser với nhau
thành lưới mesh, mỗi hop chỉ mất một khoảng cố định nhỏ (mặc định 0.5
phút/hop, con số thận trọng cho xử lý trusted-node, KHÔNG phải độ trễ
truyền tín hiệu thật vốn chỉ ~vài ms).

**4 bước chính:**
1. **Dựng lưới ISL từ TLE thật** (`modules/isl_topology.py`): xác định mặt
   phẳng quỹ đạo của mỗi vệ tinh qua RAAN (nhóm theo ngưỡng chọn TỪ histogram
   thật, không giả định trước — trên 1.019 vệ tinh Shell-1: 62 mặt phẳng,
   bin_width=2.0°), rồi nối mỗi vệ tinh với hàng xóm trước/sau CÙNG mặt
   phẳng (2 link) + vệ tinh gần pha nhất ở 2 mặt phẳng liền kề (2 link nữa)
   → tối đa 4 hàng xóm/vệ tinh (mô hình "+Grid" chuẩn học thuật).
2. **BFS 1 lần** từ vệ tinh pickup → biết số hop tới MỌI vệ tinh khác trong
   lưới (`bfs_hop_distance`).
3. **`time_optimal_relay`**: trong các vệ tinh có pass qua thành phố đích
   VÀ khóa đã kịp đến (t_rise ≥ t_pick + n_hop×0.5'), chọn pass ĐẾN SỚM
   NHẤT.
4. **`capacity_optimal_relay`**: trong các candidate nằm trong ngân sách
   thời gian (≤ 3× latency của time-optimal), chọn pass có góc ngẩng TỐT
   NHẤT (→ SKR/dung lượng cao nhất qua `channel_model`/`sikd_performance`)
   — đánh đổi vài phút để lấy dung lượng lớn hơn nhiều.

**Kết quả thật (28 cặp SF, sample 200 pickup/thành phố gốc):** latency
time-optimal hội tụ về **2.9-4.1 phút MỌI HƯỚNG** — không còn phụ thuộc
khoảng cách địa lý (khác hẳn model cũ, từng cho 7'-725' rất phân tán).
jakarta↔manila giờ 3.49'/3.78' — gần đối xứng. **75.7% số lần pickup**
(trung vị qua 28 cặp), time-optimal và capacity-optimal chọn HAI VỆ TINH
KHÁC NHAU — xác nhận trade-off thời gian/dung lượng là thật và phổ biến,
không phải hiện tượng hiếm.

| File | Vai trò |
| --- | --- |
| `modules/isl_topology.py` | Dựng lưới ISL từ TLE (RAAN/mean anomaly, "+Grid") |
| `modules/isl_relay.py` | `bfs_hop_distance`, `time_optimal_relay`, `capacity_optimal_relay` |
| `scripts/12a_build_isl_graph.py` | Sinh lưới + `fig11_isl_graph_diagnostic.png` |
| `scripts/12b_isl_relay_recompute.py` | Tính 28 cặp SF bằng cả 2 sub-algorithm |
| `tests/test_isl_topology.py`, `test_isl_relay.py` | Viết TRƯỚC (TDD), 16 test |
| `sf_relay_detail_56pairs_isl.csv` | Output chính, 28 dòng có hướng |
| `fig10_isl_relay_latency_heatmap.png` | Heatmap latency time-optimal — hình chính |

Chi tiết đầy đủ: `07-5.Plan_Optimization_Redesign.md` Task 24.3.
