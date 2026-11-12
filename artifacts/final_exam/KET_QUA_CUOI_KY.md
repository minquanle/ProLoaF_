# Cuối kỳ: cải thiện LSTM-chuẩn và giải thích dự báo bằng XAI

Mọi thí nghiệm dùng cách chia **A train (2015–2017), B chọn mô hình (2018), C kiểm tra (2019)**, mỗi mô hình 5 seed.
Tập C chỉ được dùng **một lần** để chấm điểm; c và mọi lựa chọn khác đều được quyết định trên B.

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
| `at_hol` | Như `at`, thêm `is_holiday` (ngày lễ toàn quốc Đức) và `is_xmas` (24/12–01/01) cho decoder. Cả hai là thông tin lịch, biết trước. |
| Ensemble 5 seed | μ = trung bình các μ; σ tính từ hỗn hợp Gaussian của 5 seed |
| Hiệu chỉnh trên B | c = phân vị 95% (split conformal) của \|y−μ\|/(1.96σ) trên tập B |

## 4. Kết quả trên tập C (2019)
| Mô hình | RMSE (MW) | MAE (MW) | PICP | Độ rộng khoảng (MW) | MIS (MW) | MAE ngày lễ (MW) |
|---|---|---|---|---|---|---|
| DLinear, ensemble | 3514 | 2354 | – | – | – | – |
| LSTM-chuẩn, từng seed | 3455 ± 232 | 2302 | 97.3% | 14,132 | 17,853 | – |
| LSTM-chuẩn, ensemble + c | 3375 | 2233 | 95.5% | 10,971 | 16,898 | 12,593 |
| `at`, từng seed | 2175 ± 74 | 1383 | 98.4% | 9,811 | 11,835 | – |
| `at`, ensemble + c | 2126 | 1339 | 95.8% | 6,885 | 10,464 | 7,183 |
| `at_hol`, từng seed | 1612 ± 64 | 1165 | 98.6% | 8,685 | 9,658 | – |
| **`at_hol`, ensemble + c** | **1571** | **1127** | **95.2%** | **5,930** | **8,173** | **1,302** |

So với LSTM-chuẩn của giữa kỳ (ensemble), bản cuối **giảm RMSE 53%**, **độ rộng khoảng hẹp đi 46%**, **sai số ngày lễ giảm khoảng 10 lần**, và PICP vẫn sát 95%.
Hiệu chỉnh c riêng cho từng giờ cho kết quả gần như bằng c chung, nên giữ c chung cho đơn giản.

### Kiểm định
| So sánh | Kiểm định | p |
|---|---|---|
| `at_hol` với DLinear, từng seed (n = 5) | Welch t | 1.0e-7 |
| `at_hol` ensemble với DLinear ensemble | DM, 359 ngày không chồng lấn | 2.0e-17 |
| `at_hol` ensemble với LSTM-chuẩn ensemble | DM | 3.7e-8 |
| `at_hol` ensemble với `at` ensemble (đóng góp của đặc trưng ngày lễ) | DM | 0.004 |
| `at` ensemble với LSTM-chuẩn ensemble (đóng góp của việc chuẩn hóa AT) | DM | 1.8e-8 |
| LSTM-chuẩn ensemble với DLinear ensemble | DM | 0.003 |

## 5. XAI sau khi cải thiện: mô hình đã "nhìn" đúng chỗ
| | LSTM-chuẩn | `at` | `at_hol` |
|---|---|---|---|
| RMSE tăng khi xáo **phụ tải Đức** | +21 MW | **+4867 MW** | +4124 MW |
| Tỉ lệ IG của phụ tải Đức trên μ | 0.5% | 17.0% | 14.2% |
| RMSE tăng khi xáo cột ngày lễ | – | – | +1312 MW |
| Ngày sai nhất | 24/12 (16,144) | 01/05, Quốc tế Lao động (12,565) | 20/06, lễ Mình Thánh Chúa (9,443) |

- Theo IG theo độ trễ, mô hình mới dùng **vài giờ gần nhất** và **cùng giờ hôm trước** (đỉnh quanh lag 20–24 h), đúng như quy luật của phụ tải. Dữ liệu cũ hơn khoảng 48 giờ gần như không được dùng.
- Ngày sai nhất của `at_hol` là **20/06/2019 (lễ Mình Thánh Chúa, Fronleichnam)**. Đây là ngày lễ của **một số bang** nên không có trong danh sách ngày lễ toàn quốc. Hướng tiếp theo: dùng cờ ngày lễ có trọng số theo dân số từng bang.

## 6. Cách chạy lại (không cần huấn luyện)
```bash
python reproduction/final/evaluate_improvements.py         # bảng mục 4 và kiểm định
python reproduction/final/xai_analysis.py                  # XAI cho LSTM-chuẩn  -> xai/
python reproduction/final/xai_analysis.py at_hol           # XAI cho bản tốt nhất -> xai/at_hol/
```
Huấn luyện lại các biến thể (khoảng 3–4 giờ trên CPU): `bash reproduction/final/run_all_variants.sh <python>`.

| Thư mục | Nội dung |
|---|---|
| `runs/{at,at_hol}_seed*/` | Mô hình, log theo từng epoch, dự báo trên B và C |
| `runs/baseline_seed*/` | Dự báo trên B và C của LSTM-chuẩn (giữa kỳ) |
| `results/` | `summary.csv`, `per_horizon.csv`, `tests.json` |
| `xai/`, `xai/at/`, `xai/at_hol/` | Hình và bảng XAI cho từng mô hình |
