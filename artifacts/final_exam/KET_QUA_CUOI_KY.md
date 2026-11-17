# Cuối kỳ: cải thiện LSTM-chuẩn và giải thích dự báo bằng XAI

Mọi thí nghiệm dùng cách chia **A train (2015–2017), B chọn mô hình (2018), C kiểm tra (2019)**, mỗi mô hình 5 seed.
Không phương án nào được chọn theo điểm trên C; c, việc dùng c chung hay theo giờ (fit nửa đầu B, thử nửa sau B) và mọi lựa chọn khác đều được quyết định trên B.
Lưu ý: C đã được xem ở giữa kỳ (lỗi giờ dự báo đầu tiên thấy trên C), và nhiều phiên bản được chấm trên C. Mọi phiên bản đã huấn luyện đều được báo cáo.
Ranh giới A/B/C tính theo tỷ lệ số giờ: C bắt đầu 31/12/2018 19:00 UTC.

## 1. Bài toán chọn từ giữa kỳ
LSTM-chuẩn có RMSE 3455 ± 232 MW. Mô hình không tốt hơn DLinear một cách có ý nghĩa (p = 0.33) và kết quả dao động mạnh giữa các seed.

## 2. Chẩn đoán trên tập B (2018): cơ sở để chọn cải tiến, không dùng C
Kết quả chẩn đoán trên B (`diagnosis_B/`, chạy bằng `python reproduction/final/diagnose_on_validation.py`):

| Phép đo trên B, ensemble LSTM-chuẩn | Kết quả |
|---|---|
| MAE vào ngày lễ toàn quốc / Giáng sinh / ngày khác | **10,571 / 7,527 / 2,152 MW** |
| RMSE tăng khi xáo phụ tải Đức / phụ tải Áo / lịch giờ | **+4 / +226 / +6,128 MW** |

Hai hướng cải tiến (chuẩn hóa phụ tải Áo, thêm cờ ngày lễ) được chọn dựa trên kết quả này.
Lỗi phụ tải Áo chưa chuẩn hóa đã được ghi nhận từ giữa kỳ; XAI xác nhận và đo được tác hại của nó.
Kết quả trên C ở mục dưới cho cùng một bức tranh.

## 2b. XAI trên tập C (mô hình giữa kỳ, không huấn luyện lại)
| Kỹ thuật | Phát hiện |
|---|---|
| Permutation importance | Xáo **phụ tải Đức** trong quá khứ chỉ làm RMSE tăng **21 MW**, còn xáo lịch giờ làm tăng 6019 MW. Mô hình **bỏ qua quá khứ**, chỉ dự báo theo lịch. |
| Integrated Gradients | Phụ tải Đức chỉ chiếm 0.5% đóng góp vào μ. Trong phần quá khứ, chỉ **phụ tải Áo** (biến chưa chuẩn hóa, giá trị hàng nghìn) được mô hình chú ý. |
| Phân tích ngày cụ thể | Ngày lễ: MAE **12,593 MW**, so với 1,996 MW ở ngày thường. Đêm Giáng sinh sai 13,384 MW. |
| Deletion test | Bỏ 200 ô quan trọng nhất theo IG làm RMSE tăng lên 3789 MW, bỏ 200 ô ngẫu nhiên chỉ tăng lên 3370 MW: lời giải thích trung thực với mô hình. |

Kết luận: (a) phụ tải Áo **chưa chuẩn hóa** lấn át đầu vào của LSTM; (b) mô hình **không biết ngày lễ**.

## 3. Cải thiện (chỉ thay đổi dữ liệu đầu vào, giữ nguyên kiến trúc và siêu tham số)
| Biến thể | Thay đổi |
|---|---|
| `at` | Thêm phụ tải Áo vào nhóm được chuẩn hóa min-max (fit trên A) |
| `at_hol` (sau ensemble và hiệu chỉnh gọi là **LSTM cải tiến**) | Như `at`, thêm `is_holiday` (ngày lễ toàn quốc Đức) và `is_xmas` (24/12–01/01) cho decoder. Cả hai là thông tin lịch, biết trước. |
| `hol` (đối chứng) | Chỉ thêm 2 cờ ngày lễ, phụ tải Áo vẫn chưa chuẩn hóa: ô còn thiếu của thiết kế 2×2 |
| `noat` (đối chứng) | Bỏ hẳn phụ tải Áo khỏi encoder (4 biến, 33,452 tham số): kiểm tra cơ chế của C1 |
| Ensemble 5 seed | μ = trung bình các μ; σ tính từ hỗn hợp Gaussian của 5 seed |
| Hiệu chỉnh trên B | c = phân vị 95% (split conformal) của \|y−μ\|/(1.96σ) trên tập B |

## 4. Kết quả trên tập C (2019)
| Mô hình | RMSE (MW) | MAE (MW) | PICP | Độ rộng khoảng (MW) | MIS (MW) | MAE ngày lễ (MW) |
|---|---|---|---|---|---|---|
| DLinear, ensemble | 3514 | 2354 | – | – | – | – |
| DLinear + cùng 48 cột lịch và ngày lễ, ensemble | 2542 | 1840 | – | – | – | 4,192 |
| LSTM-chuẩn, từng seed | 3455 ± 232 | 2302 | 97.3% | 14,132 | 17,853 | – |
| LSTM-chuẩn, ensemble + c | 3375 | 2233 | 95.5% | 10,971 | 16,898 | 12,593 |
| `at`, từng seed | 2175 ± 74 | 1383 | 98.4% | 9,811 | 11,835 | – |
| `at`, ensemble + c | 2126 | 1339 | 95.8% | 6,885 | 10,464 | 7,183 |
| `at_hol`, từng seed | 1612 ± 64 | 1165 | 98.6% | 8,685 | 9,658 | – |
| **`at_hol`, ensemble + c** | **1571** | **1127** | **95.2%** | **5,930** | **8,173** | **1,302** |
| `hol`, từng seed / ensemble + c | 2399 ± 78 / 2341 | 1752 | 95.9% | 9,147 | 11,728 | 1,623 |
| `noat`, từng seed / ensemble + c | 2242 ± 118 / 2184 | 1371 | 95.6% | 6,992 | 10,918 | 7,724 |
| DLinear + lịch, trọng số riêng từng giờ, ensemble | 2469 | 1762 | – | – | – | 3,905 |

So với LSTM-chuẩn của giữa kỳ (ensemble), bản cuối **giảm RMSE 53%**, **độ rộng khoảng hẹp đi 46%**, **sai số ngày lễ giảm khoảng 10 lần**, và PICP vẫn sát 95%.
Hiệu chỉnh c riêng cho từng giờ cho kết quả gần như bằng c chung khi kiểm tra trong B (nửa đầu → nửa sau: PICP 95.36 so với 95.34%), nên giữ c chung cho đơn giản.
Lưu ý: B vừa dùng để dừng sớm vừa để hiệu chỉnh và các cửa sổ chồng lấn, nên độ phủ 95% chỉ là kết quả thực nghiệm, không phải bảo đảm của conformal.

### Thiết kế 2×2 (RMSE ensemble trên C)
| | Không cờ ngày lễ | Có cờ ngày lễ | Tác dụng của C2 |
|---|---|---|---|
| Phụ tải Áo chưa chuẩn hóa | 3375 (LSTM-chuẩn) | 2341 (`hol`) | −31% |
| Phụ tải Áo đã chuẩn hóa | 2126 (`at`) | 1571 (`at_hol`) | −26% |
| Tác dụng của C1 | −37% | −33% | |

C2 sửa ngày lễ (MAE lễ 12,593 → 1,623 dù encoder còn hỏng) nhưng không sửa giờ 1 (h=1 vẫn 2,873 MW); C1 sửa giờ 1 (1,624 MW) nhưng ngày lễ còn 7,183 MW. `noat` (2184) gần bằng `at` (2126): tác hại của phụ tải Áo đến từ thang đo; `at` vs `noat` không có ý nghĩa bền (DM p = 0.059, 14/24 mốc giờ).

### Chỉ số bổ sung (`python reproduction/final/review_checks.py`)
| | LSTM-chuẩn | `at` | `hol` | `noat` | `at_hol` | DLinear | DLinear + lịch | + lịch theo giờ |
|---|---|---|---|---|---|---|---|---|
| MAPE | 4.13% | 2.46% | 3.19% | 2.54% | 2.05% | 4.35% | 3.51% | 3.36% |
| CRPS sau hiệu chỉnh (MW) | 1599 | 976 | 1253 | 1001 | 818 | – | – | – |
| Độ phủ khoảng 50 / 80 / 90% | 53 / 85 / 92 | 56 / 85 / 92 | 53 / 83 / 92 | 56 / 84 / 92 | 55 / 83 / 92 | – | – | – |
| MAE ngày ngay sau ngày lễ (MW) | 3502 | 2410 | 2796 | 2487 | 1925 | 3292 | 3828 | 3606 |
| MAE ngày cầu 31/05, 04/10 (MW) | 5413 | 1561 | 5284 | 1435 | 1589 | 2843 | 3251 | 3142 |

RMSE ở giờ h=1, trung bình từng seed / ensemble: DLinear 992 / 849, LSTM-chuẩn 4164 / 4052, `at` 1687 / 1624, `at_hol` 1445 / 1377.

### Kiểm định
| So sánh | Kiểm định | p |
|---|---|---|
| `at_hol` với DLinear, từng seed (n = 5) | Welch t | 1.0e-7 |
| `at_hol` ensemble với DLinear ensemble | DM, 359 ngày không chồng lấn | 2.0e-17 |
| `at_hol` ensemble với LSTM-chuẩn ensemble | DM | 3.7e-8 |
| `at_hol` ensemble với `at` ensemble (đóng góp của đặc trưng ngày lễ) | DM | 0.004 |
| `at` ensemble với LSTM-chuẩn ensemble (đóng góp của việc chuẩn hóa AT) | DM | 1.8e-8 |
| LSTM-chuẩn ensemble với DLinear ensemble (**không bền, đã rút lại**: chỉ 12/24 mốc giờ, t từ −4.3 đến +3.7) | DM | 0.003 |
| `at_hol` ensemble với DLinear + lịch ensemble | DM | 3.8e-26 |
| `at_hol` ensemble với DLinear + lịch theo giờ ensemble | DM | 1.3e-19 |
| `hol` với LSTM-chuẩn (C2 khi chưa có C1) | DM | 2.8e-4 |
| `at_hol` với `hol` (C1 khi đã có C2) | DM | 1.6e-13 |
| `noat` với LSTM-chuẩn | DM | 1.1e-8 |
| `at` với `noat` | DM | 0.059 |

Lặp lại DM với cả 24 mốc giờ phát hành trong ngày: mọi cải tiến (C1, C2 theo cả hai hàng của 2×2) và mọi so sánh `at_hol` với các mốc đều có ý nghĩa ở 24/24 mốc. C2 tốt hơn ở 223/359 ngày (kiểm định dấu p = 5e-6, Wilcoxon p = 1e-9, bootstrap khối 7 ngày không chứa 0), dù 10 ngày tốt nhất chiếm 93% mức giảm.

### Kiểm tra bổ sung (`python reproduction/final/extra_checks.py`)
Kiểm định DM với phương sai Newey–West (7 ngày trễ) cho cùng kết luận: C1 p = 0.001, C2 p = 0.008, `at_hol` với LSTM-chuẩn p = 1.6e-4, với DLinear p = 6e-15, với DLinear + lịch p = 5e-17; LSTM-chuẩn ensemble với DLinear ensemble p = 0.028.

Độ phủ của khoảng 95% đã hiệu chỉnh, theo nhóm giờ trên C (`results/coverage_breakdown.csv`):

| Nhóm giờ | Tỷ trọng | LSTM-chuẩn | `at` | `at_hol` |
|---|---|---|---|---|
| Cả năm | 100% | 95.5% | 95.8% | 95.2% |
| Mùa đông / xuân / hè / thu | ~25% mỗi mùa | 94.9 / 94.4 / 96.7 / 95.9% | 95.0 / 95.4 / 96.5 / 96.2% | 92.7 / 96.2 / 96.1 / 95.7% |
| Ngày làm việc / cuối tuần (không lễ) | 68% / 28% | 97.9 / 96.8% | 97.4 / 95.9% | 96.0 / 94.4% |
| Ngày lễ toàn quốc | 2.2% | 17.2% | 50.0% | 92.3% |
| Giáng sinh – năm mới (không lễ) | 1.6% | 74.9% | 89.7% | 80.2% |

Hệ số c chỉ bảo đảm độ phủ tính chung. Cờ ngày lễ đưa độ phủ ngày lễ từ 17% lên 92%, nhưng mùa đông (92.7%) và giai đoạn Giáng sinh (80.2%) còn thấp hơn 95%.

## 5. XAI sau khi cải thiện: mô hình đã "nhìn" đúng chỗ
| | LSTM-chuẩn | `at` | `at_hol` |
|---|---|---|---|
| RMSE tăng khi xáo **phụ tải Đức** | +21 MW | **+4867 MW** | +4124 MW |
| Tỉ lệ IG của phụ tải Đức trên μ | 0.5% | 17.0% | 14.2% |
| RMSE tăng khi xáo cột ngày lễ | – | – | +1312 MW |
| Ngày sai nhất | 24/12 (16,144) | 01/05, Quốc tế Lao động (12,565) | 20/06, lễ Mình Thánh Chúa (9,443) |

- Thống kê cổng LSTM của encoder (`xai_extra.py`, trung bình 5 seed): ở LSTM-chuẩn và `hol` (phụ tải Áo thô) 99% giá trị cổng bị bão hòa, |tiền kích hoạt| ≈ 285 và phụ tải Áo chiếm 99.98% tín hiệu vào cổng; ở `noat`, `at`, `at_hol` chỉ 7–9% bão hòa, |tiền kích hoạt| ≈ 1.2. Tỉ lệ IG của phụ tải Đức trên 5 seed: 0.0–0.6% (LSTM-chuẩn), 0.0–0.3% (`hol`), 13–24% (`noat`), 12–20% (`at`), 12–15% (`at_hol`).
- Deletion test lặp 20 lần chọn ngẫu nhiên, cho cả 5 seed: với `at_hol`, che 10 ô theo IG cho 4.0–4.4 nghìn MW, che ngẫu nhiên 1.5–1.7 nghìn MW. Xếp hạng chung cả ô lịch: 170/200 ô đầu là ô lịch của decoder.
- Theo IG theo độ trễ, mô hình mới dùng **vài giờ gần nhất** và **cùng giờ hôm trước** (đỉnh quanh lag 20–24 h), đúng như quy luật của phụ tải. Dữ liệu cũ hơn khoảng 48 giờ gần như không được dùng.
- Ngày sai nhất của `at_hol` là **20/06/2019 (lễ Mình Thánh Chúa, Fronleichnam)**. Đây là ngày lễ của **một số bang** nên không có trong danh sách ngày lễ toàn quốc. Hướng tiếp theo: dùng cờ ngày lễ có trọng số theo dân số từng bang.

## 6. Cách chạy lại (không cần huấn luyện)
```bash
python reproduction/final/evaluate_improvements.py         # bảng mục 4 và kiểm định
python reproduction/final/xai_analysis.py                  # XAI cho LSTM-chuẩn  -> xai/
python reproduction/final/xai_analysis.py at_hol           # XAI cho bản tốt nhất -> xai/at_hol/
python reproduction/final/run_dlinear_calendar.py          # DLinear thêm cột lịch -> dlinear_calendar/
python reproduction/final/extra_checks.py                  # độ phủ theo nhóm giờ, DM Newey-West
python reproduction/final/review_checks.py                 # DM theo 24 mốc giờ, MAPE, CRPS, ngày sau lễ, c chọn trong B
python reproduction/final/run_dlinear_calendar.py --per-horizon   # DLinear + lịch, trọng số theo giờ
python reproduction/final/xai_extra.py baseline at hol noat at_hol  # cổng LSTM, IG và deletion cho cả 5 seed
python demo/run_demo.py                                    # demo có chọn mô hình và giải thích IG
```
Huấn luyện lại các biến thể (khoảng 3–4 giờ trên CPU): `bash reproduction/final/run_all_variants.sh <python>`; hai biến thể đối chứng: `bash reproduction/final/run_control_variants.sh <python>`.

| Thư mục | Nội dung |
|---|---|
| `runs/{at,at_hol,hol,noat}_seed*/` | Mô hình, log theo từng epoch, dự báo trên B và C |
| `runs/baseline_seed*/` | Dự báo trên B và C của LSTM-chuẩn (giữa kỳ) |
| `results/` | `summary.csv`, `per_horizon.csv`, `tests.json`, `coverage_breakdown.csv`, `dm_robust.json`, `review_checks.json`, `review_metrics.csv` |
| `xai/`, `xai/at/`, `xai/at_hol/` | Hình và bảng XAI cho từng mô hình; `xai/extra_<biến thể>.json`: cổng LSTM, IG và deletion cho cả 5 seed |
| `dlinear_calendar/`, `dlinear_calendar_perh/` | DLinear + lịch, trọng số chung và theo giờ |
