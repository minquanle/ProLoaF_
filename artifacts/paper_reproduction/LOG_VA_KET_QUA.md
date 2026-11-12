# Log và file kết quả (không cần huấn luyện lại)

Mọi mô hình đã được huấn luyện sẵn. Log, checkpoint và dự báo trên tập kiểm tra đều đã được lưu.
Để kiểm tra lại toàn bộ số liệu trong báo cáo mà **không huấn luyện**, chạy lệnh sau (mất khoảng vài giây):

```bash
python reproduction/check_saved_results.py
```

Script đọc các file `.npz` đã lưu, tính lại RMSE và MAE (đơn vị MW) cho từng seed, rồi so với các file tóm tắt đã lưu.
Nó cũng tính lại p-value Welch của phép so sánh LSTM với DLinear.
Kết quả được ghi vào `saved_results_check.csv`.

## Log huấn luyện

| Thí nghiệm | Log |
|---|---|
| LSTM-gốc (cách chia của tác giả) | `training.log`, `tensorboard/` |
| LSTM-chuẩn seed 1 | `strict_602020_20261013.stdout.log` |
| LSTM-chuẩn seed 2–5 | `strict_602020_seed{2..5}.log` |
| LSTM-chuẩn, loss theo từng epoch | `strict_602020_*/epoch_history.csv` |
| DLinear seed 1–3 | `dlinear_opsd_20261015/epoch_history.csv` |
| DLinear seed 4–5 | `dlinear_seed45.stdout.log`, `dlinear_opsd_seed45/epoch_history.csv` |

## Mô hình đã huấn luyện

| Mô hình | File |
|---|---|
| LSTM-gốc | `opsd_recurrent.pkl` |
| LSTM-chuẩn seed 1–5 | `strict_602020_*/opsd_recurrent.pkl` |
| DLinear seed 1–5 | `dlinear_opsd_*/dlinear_seed*.pt` |
| Prophet | `prophet_paper_protocol/prophet_model.json` |

## Kết quả

| Nội dung | Thư mục hoặc file |
|---|---|
| LSTM-gốc trên 2018–2019 | `evaluation/` |
| LSTM-chuẩn: từng seed và tổng hợp 5 seed | `strict_602020_*/test_summary.csv`, `strict_602020_5seeds_summary/` |
| Hiệu chỉnh khoảng (c ≈ 0.842) | `interval_calibration/` |
| Naive và Seasonal naive | `baseline_comparison/` |
| Prophet | `prophet_paper_protocol/` |
| DLinear | `dlinear_opsd_20261015/`, `dlinear_opsd_seed45/` |
| So sánh LSTM với DLinear (p ≈ 0.33) | `lstm_vs_dlinear_5seeds/tests.json` |
| Phần dư so với Hình 3 của bài báo | `residual_comparison/` |

## Số liệu chính (khớp với báo cáo)

| Mô hình | RMSE (MW) | PICP |
|---|---|---|
| LSTM-gốc | 3262.6 | 97.10% |
| LSTM-gốc sau hiệu chỉnh | 3262.6 | 95.06% |
| LSTM-chuẩn, trung bình 5 seed | 3455.5 ± 231.9 | |
| DLinear, trung bình 5 seed | 3570.4 ± 13.4 | |
| Prophet | 3946.5 | 95.40% |
| Seasonal naive 24h | 6932.1 | 94.92% |
| Naive | 11120.7 | 95.82% |

Welch t = −1.105, p = 0.331: chênh lệch giữa LSTM-chuẩn và DLinear không có ý nghĩa thống kê.
