# ProLoaF Midterm Project

Mini-project về dự báo xác suất phụ tải điện ngày kế tiếp cho Đức, được xây dựng
dựa trên bài báo/repository ProLoaF và bộ dữ liệu Open Power System Data (OPSD).

## Phạm vi bài giữa kỳ

- Dataset: OPSD Time Series, phiên bản `2020-10-06`.
- Biến mục tiêu: `DE_load_actual_entsoe_transparency` (MW).
- Tần suất dữ liệu: 1 giờ.
- Cửa sổ đầu vào: 168 giờ gần nhất.
- Khoảng dự báo: 24 giờ tiếp theo.
- Baseline bắt buộc: Seasonal Naive (chu kỳ 24 giờ và/hoặc 168 giờ).
- Mô hình chính dự kiến: GRU hoặc LSTM.
- Đánh giá điểm: MAE, RMSE và sMAPE.
- Đánh giá xác suất: coverage và độ rộng khoảng dự báo 95%.

Đây là một bài tái hiện có phạm vi (scoped reproduction), không phải chạy lại toàn
bộ repository ProLoaF hoặc tái tạo mọi thí nghiệm trong paper.

## Cấu trúc dự án

```text
ProLoaF_/
├── data/
│   ├── raw/                 # dữ liệu gốc, không chỉnh sửa
│   └── processed/           # dữ liệu sau làm sạch/feature engineering
├── demo/app.py              # ứng dụng demo ở giai đoạn cuối
├── models/                  # checkpoint mô hình
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_gru_model.ipynb
│   └── 04_evaluation.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/                     # mã nguồn có thể tái sử dụng
├── requirements.txt         # ràng buộc dependency tối thiểu
└── requirements-lock.txt    # phiên bản đã cài trong môi trường hiện tại
```

## Thiết lập môi trường trên Windows PowerShell

Yêu cầu Python 3.11. Nếu PowerShell chưa nhận lệnh `conda activate`, đóng rồi mở
lại terminal sau khi chạy `conda init powershell`.

```powershell
conda create -n proloaf-midterm python=3.11 -y
conda activate proloaf-midterm
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name proloaf-midterm --display-name "Python (ProLoaF Midterm)"
```

Kiểm tra môi trường:

```powershell
python -c "import sys, torch, pandas, sklearn; print(sys.executable); print(sys.version); print(torch.__version__); print(pandas.__version__); print(sklearn.__version__)"
jupyter kernelspec list
```

Kết quả mong đợi: Python trỏ tới
`miniconda3\envs\proloaf-midterm\python.exe`, phiên bản Python là 3.11 và kernel
`proloaf-midterm` xuất hiện trong danh sách.

## Chạy notebook

Có thể mở notebook trực tiếp trong editor có hỗ trợ Jupyter, hoặc chạy JupyterLab:

```powershell
conda activate proloaf-midterm
jupyter lab
```

Trong notebook, chọn kernel **Python (ProLoaF Midterm)**. Chạy notebook theo thứ
tự `01` → `02` → `03` → `04` vì đầu ra của các bước sau phụ thuộc bước trước.

## Dữ liệu

Nguồn chính thức:
[OPSD Time Series 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06/).

Dữ liệu tải về được đặt trong `data/raw/` và không commit lên Git. File đã xử lý
đặt trong `data/processed/`. Không chỉnh sửa trực tiếp file dữ liệu gốc.

## Trạng thái

Setup ban đầu đã hoàn tất. Bước tiếp theo là tải đúng file OPSD, kiểm tra schema,
chọn khoảng thời gian nghiên cứu và thực hiện EDA trong notebook `01`.
