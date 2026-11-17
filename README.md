# Tái lập ProLoaF: dự báo phụ tải điện Đức 24 giờ tới

Dự án giữa kỳ và cuối kỳ môn **Phân tích dữ liệu dự báo** (INT3234E 1, Trường Đại học Công nghệ – ĐHQGHN).

Dự án chạy lại ví dụ trong bài báo *ProLoaF: Probabilistic load forecasting for power systems* (SoftwareX, 2023). Mô hình xem 147 giờ gần nhất rồi dự báo phụ tải điện nước Đức cho 24 giờ tiếp theo. Mỗi giờ có hai kết quả: một giá trị dự báo và một khoảng 95%.

## Những gì đã làm

1. **Chạy lại mô hình của tác giả.** Mô hình là LSTM encoder–decoder, dùng đúng mã nguồn và dữ liệu OPSD của tác giả: 43.824 giờ, từ 2015 đến 2019.
2. **Phát hiện và sửa cách chia dữ liệu.** Mã gốc dùng luôn năm 2019, tức dữ liệu chấm điểm, để chọn mô hình. Dự án chia lại theo thời gian: 60% để học, 20% (năm 2018) để chọn mô hình, 20% (năm 2019) chỉ để chấm điểm. Mô hình được huấn luyện lại 5 lần với 5 seed.
3. **So sánh với các phương pháp khác** trên cùng năm 2019:
   - Simple Naive và Seasonal Naive;
   - Prophet;
   - DLinear, lấy từ một bài báo khác (AAAI 2023), cũng chạy 5 seed. Kiểm định t Welch được dùng để xem chênh lệch với LSTM có ý nghĩa thống kê không.
4. **Hiệu chỉnh khoảng dự báo.** Khoảng gốc rộng quá mức cần. Nhân σ với một hệ số c = 0,84, tìm trên năm 2018, thì khoảng hẹp đi 15,8% và tỷ lệ giá trị thật nằm trong khoảng về 95,06%.
5. **Làm một trang web demo** để xem lại dự báo ở bất kỳ giờ nào của năm 2019.

## Kết quả chính (năm 2019)

| Phương pháp | RMSE (MW) | MAE (MW) |
|---|---:|---:|
| Simple Naive | 11.121 | 8.778 |
| Seasonal Naive | 6.932 | 4.541 |
| Prophet | 3.946 | 2.898 |
| **LSTM, chia dữ liệu đúng (TB 5 seed)** | **3.456 ± 232** | **2.302 ± 202** |
| DLinear (TB 5 seed) | 3.570 ± 13 | 2.427 ± 28 |

Cách đọc bảng:

- LSTM tốt hơn rõ rệt so với Naive và Prophet.
- LSTM chỉ hơn DLinear khoảng 3%. Kiểm định t Welch trên 5 seed cho p ≈ 0,33, nên chênh lệch này chưa có ý nghĩa thống kê.
- DLinear dự báo tốt hơn trong 7 giờ đầu và ổn định hơn nhiều giữa các lần chạy.
- Phần hiệu chỉnh khoảng và trang demo vẫn dùng mô hình chạy theo mã gốc.

## Phần cuối kỳ: cải thiện mô hình và giải thích bằng XAI

Đề cuối kỳ yêu cầu cải thiện lời giải của giữa kỳ và dùng một kỹ thuật XAI để giải thích kết quả dự báo.

1. **Chẩn đoán trên năm 2018 (tập chọn mô hình), không dùng năm 2019.** Permutation importance cho thấy LSTM gần như bỏ qua phụ tải Đức trong quá khứ: xáo cột này chỉ làm RMSE tăng 4 MW. Sai số vào ngày lễ cao khoảng 5 lần ngày thường.
2. **Hai cải tiến về dữ liệu, giữ nguyên kiến trúc.** Chuẩn hóa phụ tải Áo (mã gốc bỏ sót) và thêm hai cột ngày lễ cho decoder. Thêm ensemble 5 seed và hiệu chỉnh khoảng trên năm 2018.
3. **XAI**: permutation importance, Integrated Gradients cho cả μ và σ, occlusion, deletion test để kiểm tra độ tin cậy, và phân tích các ngày cụ thể.

| Năm 2019 | RMSE (MW) | PICP | Độ rộng khoảng (MW) | MAE ngày lễ (MW) |
|---|---:|---:|---:|---:|
| LSTM-chuẩn (giữa kỳ), ensemble + hiệu chỉnh | 3.375 | 95,5% | 10.971 | 12.593 |
| **LSTM cải tiến, ensemble + hiệu chỉnh** | **1.571** | **95,2%** | **5.930** | **1.302** |
| DLinear, ensemble | 3.514 | – | – | – |
| DLinear + cùng cột lịch, ngày lễ, ensemble | 2.542 | – | – | 4.192 |
| DLinear + cùng cột lịch, trọng số riêng từng giờ, ensemble | 2.469 | – | – | 3.905 |

Hai biến thể đối chứng hoàn thành thiết kế 2×2 và kiểm tra cơ chế: `hol` (chỉ cờ ngày lễ, RMSE 2.341) và `noat` (bỏ hẳn phụ tải Áo, RMSE 2.184). Chuẩn hóa phụ tải Áo giảm RMSE 33–37% và cờ ngày lễ giảm 26–31%, dù đứng một mình hay đi cùng nhau; bỏ hẳn phụ tải Áo cho kết quả gần bằng chuẩn hóa nó, và các cổng LSTM của encoder cũ bão hòa gần như hoàn toàn.

Sau khi sửa, XAI cho thấy mô hình đã dùng phụ tải Đức của những giờ gần nhất và cùng giờ hôm trước. Mọi cải tiến có ý nghĩa thống kê (Diebold–Mariano, cả với phương sai Newey–West) ở cả 24 mốc giờ phát hành dự báo. Riêng so sánh LSTM-chuẩn với DLinear không bền (chỉ 12/24 mốc), nên hai mô hình này được coi là ngang nhau. Demo (`python .\demo\run_demo.py`) cho chọn LSTM-gốc của giữa kỳ hoặc LSTM cải tiến và hiển thị giải thích Integrated Gradients cho từng dự báo; thêm `?origin=8425` vào địa chỉ để mở thẳng một thời điểm. Chi tiết ở `artifacts/final_exam/KET_QUA_CUOI_KY.md`.

```powershell
python .\reproduction\final\diagnose_on_validation.py   # chẩn đoán trên 2018
python .\reproduction\final\evaluate_improvements.py    # bảng kết quả và kiểm định
python .\reproduction\final\xai_analysis.py at_hol      # XAI cho mô hình cuối kỳ
python .\reproduction\final\run_dlinear_calendar.py     # DLinear thêm cột lịch (khoảng 2 phút)
python .\reproduction\final\extra_checks.py            # độ phủ theo mùa, loại ngày; DM với Newey–West
python .\reproduction\final\review_checks.py           # DM theo 24 mốc giờ, MAPE, CRPS, ngày sau lễ
python .\reproduction\final\xai_extra.py baseline at hol noat at_hol   # cổng LSTM, IG, deletion cho 5 seed
bash reproduction/final/run_all_variants.sh <python>    # huấn luyện lại (khoảng 3–4 giờ)
bash reproduction/final/run_control_variants.sh <python> # hai biến thể đối chứng hol, noat (khoảng 1,5 giờ)
```

## Cấu trúc thư mục

```text
paper_source/                  mã nguồn gốc của tác giả (submodule)
reproduction/                  các script chạy thí nghiệm giữa kỳ
reproduction/final/            script cuối kỳ: cải tiến, đánh giá, XAI
artifacts/paper_reproduction/  mô hình, dự báo và kết quả giữa kỳ
artifacts/final_exam/          mô hình, dự báo, kết quả và hình XAI cuối kỳ
demo/                          trang web demo
```

## Cách chạy

```powershell
git clone --recurse-submodules https://github.com/minquanle/ProLoaF_.git
cd ProLoaF_
conda create -n proloaf-paper-repro python=3.9 -y
conda activate proloaf-paper-repro
pip install -r requirements.txt
pip install --no-deps -e .\paper_source

python .\demo\run_demo.py   # mở http://127.0.0.1:8501
```

Muốn huấn luyện lại LSTM theo cách chia đúng (mỗi seed khoảng 30 phút trên CPU):

```powershell
python .\reproduction\run_lstm_strict_602020.py --output-dir .\artifacts\paper_reproduction\lan_chay_moi --seed 1
```

Tổng hợp 5 seed và tính p-value:

```powershell
python .\reproduction\aggregate_lstm_seeds.py --seeds 1,2,3,4,5
python .\reproduction\compare_lstm_dlinear_seeds.py
```

Các script khác trong `reproduction/` chạy Naive, Prophet, DLinear (`--seeds` để chọn seed) và bước hiệu chỉnh khoảng.

## Nguồn

- Gürses-Tran, Oppermann, Monti (2023). *ProLoaF: Probabilistic load forecasting for power systems.* SoftwareX 23, 101487.
- Zeng và cộng sự (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI.
- Dữ liệu: Open Power System Data.
