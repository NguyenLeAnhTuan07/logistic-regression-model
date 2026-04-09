# Logistic Regression Model

Triển khai thuật toán **Logistic Regression** cho bài toán phân loại nhị phân (binary classification) với hai phương pháp tối ưu hóa: **Gradient Descent** và **Stochastic Gradient Descent (SGD)** — xây dựng từ đầu bằng Python thuần, không dùng scikit-learn.

---

## Thuật toán tối ưu

### Gradient Descent (Batch)
Tại mỗi vòng lặp, tính đạo hàm trên **toàn bộ tập dữ liệu** để cập nhật trọng số, tối thiểu hóa hàm mất mát Binary Cross-Entropy (Log Loss).

**Phù hợp khi:**
- Tập dữ liệu nhỏ đến trung bình
- Ưu tiên hội tụ ổn định, mượt mà
- Tài nguyên tính toán đủ để xử lý toàn bộ dữ liệu mỗi bước

### Stochastic Gradient Descent (SGD)
Cập nhật trọng số dựa trên **một mẫu duy nhất** (hoặc mini-batch) tại mỗi lần lặp — tần suất cập nhật cao hơn, tốc độ học nhanh hơn đáng kể.

**Phù hợp khi:**
- Tập dữ liệu cực lớn
- Cần tốc độ cập nhật nhanh, tiết kiệm RAM
- Áp dụng cho **online learning**
- Cần khả năng thoát khỏi local minima

> SGD tạo nhiều "nhiễu" hơn trong quá trình tối ưu, nhưng thường hội tụ nhanh hơn và tổng quát hóa tốt hơn trong nhiều trường hợp.

---

## Cài đặt

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Cài thư viện
pip install numpy pandas
```

---

## Sử dụng

| Lệnh | Mô tả |
|------|-------|
| `python train.py` | Huấn luyện mô hình, lưu trọng số vào `meta/` |
| `python predict.py` | Dự đoán trên dữ liệu mới từ `predict/dudoan.csv` |
| `python evaluate.py` | Đánh giá mô hình với K-Fold Cross Validation |

---

## Dataset

Đặt dữ liệu vào thư mục `data/` theo cấu trúc sau:

1. Đổi tên file thành `data.csv` và đặt vào `data/`
2. Cập nhật `data/feature_names.txt` với tên các cột tương ứng

> **Lưu ý:** Dòng **cuối cùng** trong `feature_names.txt` phải là tên cột **biến mục tiêu** (nhãn 0 hoặc 1).

Nếu dữ liệu có cột dạng chuỗi (String), dùng `encoding/encoding.py` để chuyển thành giá trị số trước khi huấn luyện.

**Dataset mẫu:**
- 🦴 Cột sống (Orthopedic): [Kaggle – Biomechanical Features](https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients/data)
- 🩺 Tiểu đường: *(dữ liệu tự cung cấp)*

---

## Đánh giá mô hình (`evaluate.py`)

Sử dụng **K-Fold Cross Validation (k=5)** để đảm bảo tính khách quan.

**Chỉ số phân loại:**

| Chỉ số | Ý nghĩa |
|--------|---------|
| Accuracy | Độ chính xác tổng thể |
| Precision | Tỷ lệ dự đoán dương tính đúng |
| Recall | Khả năng phát hiện đúng các mẫu dương tính |
| F1-Score | Trung bình điều hòa giữa Precision và Recall |
| AUC | Khả năng phân biệt giữa hai lớp |

**CV Score:** Báo cáo **mean** và **std** qua 5 folds để đánh giá độ ổn định mô hình.

---

## Dự đoán (`predict.py`)

1. Đặt dữ liệu cần dự đoán vào `predict/dudoan.csv`
2. Chạy `python predict.py`

Chương trình sẽ tự động:
- Encode và chuẩn hóa (scale) dựa trên tham số đã lưu từ lúc huấn luyện
- Tính xác suất qua hàm **Sigmoid** và xuất kết quả phân loại (0 hoặc 1)

---

## Cấu trúc dự án

```
├── data/
│   ├── data.csv               # Dữ liệu thô
│   ├── data_scaled.csv        # Dữ liệu sau khi scale (tự sinh khi train)
│   └── feature_names.txt      # Tên các cột (dòng cuối là biến mục tiêu)
├── encoding/
│   └── encoding.py            # Label Encoding cho cột dạng chuỗi
├── meta/
│   ├── scale_params.csv       # Mean/Std để tái sử dụng khi predict
│   └── weights.csv            # Trọng số mô hình đã huấn luyện
├── modelpre/
│   ├── model.py               # Triển khai fit() và predict()
│   ├── preprocessing.py       # Pipeline xử lý dữ liệu
│   └── scalestd.py            # Tiện ích chuẩn hóa
├── predict/
│   ├── dudoan.csv             # Dữ liệu đầu vào để dự đoán
│   └── dudoan_scaled.csv      # Dữ liệu đã scale
├── evaluate.py
├── predict.py
└── train.py
```

---

## Hyperparameters

Có thể tinh chỉnh trực tiếp trong `train.py`:

- **Learning Rate** — tốc độ học của mô hình
- **Epochs** — số vòng lặp huấn luyện
- **Optimizer** — chọn `GD` hoặc `SGD`

---

## Tác giả

**Nguyễn Lê Anh Tuấn**

Cảm ơn bạn đã quan tâm đến dự án! ☀️
