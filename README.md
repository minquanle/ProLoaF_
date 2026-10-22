# Tái lập ProLoaF: dự báo phụ tải điện Đức 24 giờ tới

Dự án giữa kỳ môn **Phân tích dữ liệu dự báo** (INT3234E 1, Trường Đại học Công nghệ – ĐHQGHN).

Dự án chạy lại ví dụ trong bài báo *ProLoaF: Probabilistic load forecasting for power systems* (SoftwareX, 2023). Mô hình xem 147 giờ gần nhất rồi dự báo phụ tải điện nước Đức cho 24 giờ tiếp theo. Mỗi giờ có hai kết quả: một giá trị dự báo và một khoảng 95%.

## Những gì đã làm

1. **Chạy lại mô hình của tác giả.** Mô hình là LSTM encoder–decoder, dùng đúng mã nguồn và dữ liệu OPSD của tác giả: 43.824 giờ, từ 2015 đến 2019.
2. **Phát hiện và sửa cách chia dữ liệu.** Mã gốc dùng luôn năm 2019, tức dữ liệu chấm điểm, để chọn mô hình. Dự án chia lại theo thời gian: 60% để học, 20% (năm 2018) để chọn mô hình, 20% (năm 2019) chỉ để chấm điểm. Mô hình được huấn luyện lại 3 lần với 3 seed.
3. **So sánh với các phương pháp khác** trên cùng năm 2019:
   - Simple Naive và Seasonal Naive;
   - Prophet;
   - DLinear, lấy từ một bài báo khác (AAAI 2023).
4. **Hiệu chỉnh khoảng dự báo.** Khoảng gốc rộng quá mức cần. Nhân σ với một hệ số c = 0,84, tìm trên năm 2018, thì khoảng hẹp đi 15,8% và tỷ lệ giá trị thật nằm trong khoảng về 95,06%.
5. **Làm một trang web demo** để xem lại dự báo ở bất kỳ giờ nào của năm 2019.

## Kết quả chính (năm 2019)

| Phương pháp | RMSE (MW) | MAE (MW) |
|---|---:|---:|
| Simple Naive | 11.121 | 8.778 |
| Seasonal Naive | 6.932 | 4.541 |
| Prophet | 3.946 | 2.898 |
| **LSTM, chia dữ liệu đúng (TB 3 seed)** | **3.355 ± 264** | **2.206 ± 218** |
| DLinear (TB 3 seed) | 3.570 ± 19 | 2.433 ± 36 |

Cách đọc bảng:

- LSTM tốt hơn rõ rệt so với Naive và Prophet.
- LSTM chỉ hơn DLinear khoảng 6%. Khoảng cách này nhỏ hơn mức dao động giữa các lần chạy LSTM.
- DLinear dự báo tốt hơn trong 6 giờ đầu.
- Phần hiệu chỉnh khoảng và trang demo vẫn dùng mô hình chạy theo mã gốc.

## Cấu trúc thư mục

```text
paper_source/                  mã nguồn gốc của tác giả (submodule)
reproduction/                  các script chạy thí nghiệm
artifacts/paper_reproduction/  mô hình đã huấn luyện, dự báo và kết quả
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

Các script khác trong `reproduction/` chạy Naive, Prophet, DLinear và bước hiệu chỉnh khoảng.

## Nguồn

- Gürses-Tran, Oppermann, Monti (2023). *ProLoaF: Probabilistic load forecasting for power systems.* SoftwareX 23, 101487.
- Zeng và cộng sự (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI.
- Dữ liệu: Open Power System Data.
